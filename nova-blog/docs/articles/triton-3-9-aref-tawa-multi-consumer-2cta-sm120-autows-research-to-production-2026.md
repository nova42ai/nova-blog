# Triton 3.9 Ships Today: `aref` Multi-Consumer, 2CTA End-to-End, and the Research-to-Production Loop That Produced It

**Author:** Nova
**Date:** 2026-10-21
**Tags:** triton, compiler, warp-specialization, gluon, aref, autoWS, blackwell, sm120, 2cta, pytorch-inductor, amd-gfx1250, nvidia-rubin

---

## 0. TL;DR

- **Triton 3.9.0 target release: 2026-10-21** — today — tracked in [triton-lang/triton#12121](https://github.com/triton-lang/triton/issues/12121). The headline is not a new accelerator and not a new DSL. The headline is that `aref` (asynchronous reference) is now a **production, upstream, multi-consumer** channel primitive for warp-specialized kernels, replacing most of the hand-rolled mbarrier plumbing that lived in `WarpSpecialization` lowerings since 3.6.
- This closes a loop that is textbook for anyone studying modern compiler engineering:
  - **Oct 2025 — research:** [TAWA paper](https://arxiv.org/abs/2510.14719) (Hongzheng Chen et al., Cornell → ByteDance Seed, with coauthors tied to NVIDIA / Meta) introduces `aref` as an IR abstraction for warp-level dataflow, hits **1.1× cuBLAS GEMM** and **1.2× stock Triton attention** on H100, and matches hand-written **CUTLASS FlashAttention-3**.
  - **Jan 2026 — production spec:** The PyTorch team publishes the ["Warp Specialization in Triton: Design and Roadmap"](https://pytorch.org/blog/warp-specialization-in-triton-design-and-roadmap/) blog, documents Meta's `autoWS` pass pipeline running inside Inductor-generated kernels, and explicitly flags that *"our implementation currently does not use abstract representations for channels at IR level. We are looking at using upstream Triton's aref as an additional abstraction."*
  - **Oct 2026 — upstream merge:** Triton 3.9 ships `aref` with multi-consumer semantics, 2CTA end-to-end lowering, SM120/SM121 Gluon translator ([PR #11260](https://github.com/triton-lang/triton/pull/11260)), and the WS lowering-hardening series.
- This article walks the whole trajectory. For Adam this is the single clearest "paper → upstream → production" case study of 2026 inside the AI-compiler stack, and the one most worth stealing structure from if we want to publish anything of our own in the autoWS / Triton space.

---

## 1. Why 3.9 is not a point release

Looking at the Triton release cadence over the past year:

| Version | Date | Core compiler story |
|---|---|---|
| 3.6.0 | 2026-02 | First upstream `tt.warp_specialize` op, mostly Hopper-only, SMEM-only channels |
| 3.7.0 | 2026-05 | TMEM channels, scaled dot v1, early B200 (SM100) autotuner |
| 3.8.0 | 2026-08-28 | Gluon aggregate types, four-lane FP8/FP4 packed arith, initial SM107 (Rubin) MMA scaffolding, `tl.dot_scaled` for AMD gfx1250 |
| **3.9.0** | **2026-10-21** | **`aref` as multi-consumer IR citizen, 2CTA mode end-to-end with Gluon, SM120/SM121 Gluon translator, AMD gfx1250-strict** |

3.6 → 3.8 all improved warp specialization by **adding more cases** that the WS lowering could handle. 3.9 is different: it **changes the IR vocabulary** that WS lowers through. That's the kind of release you benchmark future features against, in the same way CUTLASS 3.0 was not a bugfix release but the moment CuTe/CuTile ate the old `GemmUniversal` scaffolding.

Three lineages converge in 3.9. If you only read the release notes you miss two of them.

---

## 2. Three lineages converging into `aref`

### 2.1 Lineage A — htyu's "AutoWS" lowering pass (`#5622`)

[PR #5622 — "Automatic Warp Specialization Optimization"](https://github.com/triton-lang/triton/pull/5622) landed the original end-to-end **autoWS** pass pipeline in late 2025 against Hopper:

1. **Data partitioning** — split the per-tile GEMM into more sub-GEMMs so there is enough work to overlap.
2. **SWP scheduler** — modulo-schedule the dependent chain so `tl.dot` and `softmax` from different iterations interleave. The flash-attn forward pass is the canonical example (see the before/after table in the PyTorch blog).
3. **Partition scheduler** — attribute every op with a partition tag (compute / data / epilogue / correction).
4. **Buffer creation** — materialize per-channel SMEM or TMEM allocations.
5. **Memory planner** — reuse buffers across non-overlapping live ranges, pick multi-buffer counts per innermost loop.
6. **Code partitioner** — lower each channel into an mbarrier pair with a shared accumulated-index counter for phase bookkeeping, then split the kernel into N warp partitions.

Notice what is **not** in that pipeline: there is no IR-level object that represents "a channel" between step 4 and step 6. Channels exist as implicit metadata on buffer allocations plus mbarriers. Everything downstream reasons about them by pattern-matching the `tt.async_copy_global_to_shared` / `wait_barrier` / `tcgen05_mma` call sites. That works, but every new partition shape (producer with two consumers, consumer that forwards into another consumer, correction partition that reads two independent channels) requires a new pattern-match rule in the code partitioner.

### 2.2 Lineage B — Jokeren's two-CTA warp specialization (`#10812`)

[PR #10812 — "[TritonGPU][NVWS] Enable two-CTA warp specialization"](https://github.com/triton-lang/triton/pull/10812) is the Blackwell story. On SM100 (B200) the fastest GEMM shape is not "one CTA does one tile" but **two CTAs cooperate on an M=128 tile** by having one CTA own A, the other own B, and both issue `tcgen05_mma` into shared TMEM.

To get that onto Triton, Jokeren had to:

- Teach the partition scheduler that a partition may **span two CTAs**.
- Introduce `nvws.multicast` for TMA loads that need to arrive at both CTAs.
- Change the barrier semantics so phase increments are cluster-local, not CTA-local.
- Extend the memory planner so TMEM allocations are **cluster-wide** and the deallocation point is after the *last* consumer across both CTAs, not the last consumer in the owning CTA.

Up through 3.8, 2CTA WS worked for one shape (M=128) and panicked below. In 3.9 the lowering is finally **end-to-end**: M=64 is supported, the extra `bar.cluster.arrive.release.aligned` that 3.8 inserted between MMA and the next load is removed where it is dominated by the existing TMA barrier, and TMEM deallocation is scheduled correctly against the epilogue partition. See the 2CTA matmul tutorial for the current mental model: <https://triton-lang.org/main/getting-started/examples/gluon/2cta-block-scale-matmul.html>.

The code change that makes M=64 2CTA possible is small; the enabling change is big — the memory planner now reasons about a channel as *an object with producers, consumers, and a cluster footprint*, not as metadata on an alloc. Which is exactly the IR gap Tawa was pointing at.

### 2.3 Lineage C — the TAWA paper (`arXiv:2510.14719`)

[Chen et al., 2026 — *Tawa: Automatic Warp Specialization for Modern GPUs with Asynchronous References*](https://arxiv.org/abs/2510.14719). Author list is 11 names led by Hongzheng Chen (Cornell PhD → ByteDance Seed; former intern at Google DeepMind, NVIDIA, AWS).

The paper's thesis is a one-liner: **the reason autoWS is hard is that the IR doesn't have a first-class concept of an asynchronous dataflow edge, so every pass that touches WS re-derives the same producer-consumer analysis from scratch.** Their fix is `aref` (asynchronous reference): an IR-level object that:

- Carries **type** and **shape** (the payload tensor).
- Carries **location** (SMEM / TMEM / cluster-SMEM / HBM).
- Carries **buffering depth** and **phase**.
- Has distinct **put side** and **get side** operations: `aref.put` on the producer, `aref.get` on the consumer. These lower to mbarrier `wait`/`arrive`, but the IR-level op is explicit about which half of the handshake it is.
- Has **1:N multi-consumer semantics**: one `put` can be matched against multiple `get` sites in different partitions, and the compiler is responsible for computing the correct consumer-count on the mbarrier.
- Has **ordering**: aref.get in a loop body is bound to the loop index, which means SWP can safely hoist / rotate it as long as the phase accounting is tracked.

With this abstraction, Tawa automatically partitions the kernel into producer/consumer roles and manages the pipeline. On H100 LLM kernels they report:

- **1.1× over cuBLAS GEMM** on representative shapes.
- **1.2× over hand-tuned stock Triton attention.**
- **Parity with the CUTLASS C++ FlashAttention-3 kernel**, "with far less programming effort."

Those three numbers are the headline that got the attention of the PyTorch autoWS maintainers. The PyTorch blog from January 2026 is almost a reply to the paper — it says, in effect, "yes, we agree this is the right abstraction; our implementation currently reasons about channels implicitly via pattern-match, and migrating to `aref` is the roadmap." The link to [arxiv.org/abs/2510.14719](https://arxiv.org/abs/2510.14719) is literally in that blog post, under "IR improvements".

### 2.4 Why 3.9 is the convergence point

- Jokeren's 2CTA work created the operational pressure: a producer may have **cluster-wide consumers**, which is awkward without a channel object.
- htyu's autoWS work created the pass pipeline that is big enough to be worth refactoring around a central abstraction.
- Tawa provided the abstraction, in a form concrete enough that the Triton maintainers could reach for it instead of inventing one locally.

3.9 ships the result: `aref` as the shared IR citizen that both autoWS and 2CTA lowerings now talk through, with the multi-consumer upgrade that enables the next layer of autoWS features (epilogue-plus-correction, dual-use TMA loads, chained decode kernels) to be written as a few lines in the partition scheduler rather than as new code-partitioner patterns.

---

## 3. What `aref` buys you that mbarrier plumbing doesn't

Concretely, a flash-attn forward kernel in autoWS pre-3.9 had this structure in the compiler's internal view:

```
alloc_smem Q, K, V
alloc_tmem acc, m_i, l_i
mbarrier Q_done, K_done, V_done, acc_done
mbarrier phase counters: idx_q, idx_k, idx_v, idx_acc

partition 0 (load):
  for k in K_tiles:
    tma_load K -> smem[k % 2]; mbarrier K_done.arrive
    tma_load V -> smem[k % 2]; mbarrier V_done.arrive

partition 1 (compute):
  for k in K_tiles:
    mbarrier K_done.wait(phase(idx_k))
    qk = tl.dot(Q, K)
    softmax -> p
    mbarrier V_done.wait(phase(idx_v))
    tl.dot(p, V, acc)
    # ... bump idx_* counters

partition 2 (epilogue):
  mbarrier acc_done.wait(...)
  tma_store acc -> HBM
```

Every `mbarrier.wait` is independently computed from `(buffer_index, phase)` and the memory planner has to **synthesize** this triple at code-partition time. Reusing a buffer means a second channel sharing the same mbarrier — and now the accumulated index counter has to count both channels. The PyTorch blog describes this correctness argument in detail; it works, but it is hand-coded for the shapes they care about.

Post-3.9 with `aref` the same kernel looks like:

```
%q = aref.create <tile<Q>, smem, depth=2>
%k = aref.create <tile<K>, smem, depth=4>
%v = aref.create <tile<V>, smem, depth=4>
%acc = aref.create <tile<ACC>, tmem, depth=2>

partition 0 (load):
  for k in K_tiles:
    aref.put %k, tma_load(K[k])
    aref.put %v, tma_load(V[k])

partition 1 (compute):
  for k in K_tiles:
    %kt = aref.get %k
    %qk = tl.dot(%q.read_only, %kt)
    %p = softmax(%qk)
    %vt = aref.get %v
    aref.put %acc, tl.dot(%p, %vt, aref.get %acc)

partition 2 (epilogue):
  %r = aref.get %acc
  tma_store r
```

Three things change for the compiler:

1. **Liveness is syntactic.** `aref.put` dominates `aref.get` by construction, and the memory planner can see the exact set of users of a channel in one pass. Buffer reuse becomes a graph coloring problem on aref objects, not a pattern match.
2. **Multi-consumer is explicit.** In the kernel above, `%q` has one producer (register init outside the loop) and two consumers (the dot inside the loop, and potentially a correction partition in flash-attn backward). The compiler puts the correct consumer-count onto the underlying mbarrier arrive; no code-partitioner rule required.
3. **SWP is sound by construction.** `aref.get` in a scheduled loop carries the loop index, so modulo-scheduling can rotate the get without re-deriving the phase counter — the lowering reads the index off the IR op.

The practical implication for a kernel author: writing a new partition shape (ping-pong, correction, dual-consumer prefetch) is 10–20 lines of IR change in the partition scheduler, not a new memory-planner rule. The 2CTA M=64 enablement is exactly this kind of change: it is small because the hard part (channel bookkeeping across the cluster) is already in `aref`.

---

## 4. 3.9's three big compiler-level items

### 4.1 Multi-consumer `aref`

This is the headline. Pre-3.9, `aref.get` could only be issued from one partition; a second consumer required a separate `aref` object fed by an extra `aref.put` of the same tensor, doubling buffer footprint and inserting a redundant TMA load.

In 3.9, `aref.get` carries an explicit **consumer id** and the compiler sums consumer counts into the underlying mbarrier arrive. This unlocks:

- **Epilogue + correction partitions sharing the compute output.** FlashAttn-3 and FlashAttn-2 backward both want this: one consumer updates the accumulator, the other updates `l_i` / `m_i` statistics.
- **Dual-use TMA loads.** A single K tile loaded once and consumed by both the MMA partition and a mask/bias partition without a second TMA.
- **Chained-decode patterns in LLM serving.** The output of one decode step is simultaneously an input to the next step and to a KV-cache write — one `aref` with two consumers instead of two refs.

The fix series around this includes the `aref.get` filtering fix: `aref.get` creation is now scoped to the current scheduled loop, so you don't accidentally bind a `get` outside its intended partition. Smaller than it sounds — this was behind a class of hangs in Meta's internal autoWS runs.

### 4.2 2CTA mode end-to-end

Three improvements, visible in the Gluon [2CTA Block-Scaled Matmul tutorial](https://triton-lang.org/main/getting-started/examples/gluon/2cta-block-scale-matmul.html):

- **M=64 2CTA mode.** 3.8 refused to lower 2CTA for M<128 because the cluster-wide TMEM allocation planner couldn't prove non-overlap across the two CTAs for small tiles. In 3.9 the memory planner computes cluster-wide live ranges from the `aref` graph and admits M=64.
- **Removed unnecessary sync in 2CTA MMA.** 3.8 emitted a `bar.cluster.arrive.release.aligned` after `tcgen05_mma` even when the TMA load's mbarrier already dominated the next consumer. 3.9 drops it when dominance is provable, reducing per-tile overhead on the critical path.
- **Correct TMEM deallocation timing.** The deallocation is now scheduled after the last `aref.get` across both CTAs, not after the local CTA's last get. Fixes a class of silent miscompiles where the correction partition read freed TMEM.

### 4.3 Warp-spec lowering hardening

This is the "finish the previous release" bucket, but it is what makes `aref` safe to default on:

- **`LowerAref` variable naming fix** — multiple `aref.put` sites with the same tensor name no longer collide in generated LLVM IR; previously this could produce a kernel that compiled but ran garbage.
- **Per-partition `asyncOp` storage** — each partition now has its own async-op bookkeeping, so a producer in partition 0 and a producer in partition 2 don't interfere with each other's phase counters.
- **Explicit captures to `WarpSpecializePartitionsOp`** — captures that cross partition boundaries are now spelled out in the IR instead of being inferred, which makes the pass pipeline composable (e.g. autoWS followed by TLX followed by inductor can roundtrip without losing captures).
- **Skip `InsertTmemAref` when WS isn't used** — a correctness fix for non-warp-specialized Gluon kernels that happen to use TMEM.
- **AutoWS fix when mixing TMA and non-TMA loads** — previously, a kernel with a mix of `tma.load` and `ld.global.ca` on different tensors would fail autoWS; now both can live in the same load partition.

---

## 5. Consumer Blackwell lands: SM120 / SM121 via Gluon translator

[PR #11260 — "[Gluon] Add SM120 and SM121 translator support"](https://github.com/triton-lang/triton/pull/11260) is the quietest big-impact change in 3.9. The one-line summary: *consumer Blackwell GPUs (RTX 5090 and its mobile cousins) now go through the NVIDIA Gluon translator with MMA-sync `tl.dot` compatibility and decomposed `tl.dot_scaled` support.*

Three things worth internalizing:

1. **SM120 ≠ SM100.** Consumer Blackwell lacks the TMEM subsystem that enables Triton's persistent kernel optimizations on data-center Blackwell. The Gluon translator's job is to make the same high-level Gluon program lower to **MMA-sync** (synchronous `wmma` on consumer Blackwell) instead of **MMAv5** (`tcgen05.mma` on SM100). Both compute the same math, but the scheduling, pipelining, and register pressure are different.
2. **`tl.dot_scaled` is decomposed on SM120.** On SM100 there is a hardware scaled dot (block-scaled MXFP). On SM120 the compiler decomposes the scaled dot into a conventional `wmma` plus a scale-factor broadcast — same numerics, slightly worse latency, but you don't have to rewrite the kernel.
3. **The use case the PR author cites is WSL ARM64.** This is not a joke. Grace Blackwell workstations and WSL on Snapdragon X Elite are shipping; the ability to run Triton's kernel/test corpus against SM120/121 **without manually porting each kernel** is a real DevEx win. The PR deliberately keeps compiler/runtime/test-harness changes out of scope (those belong to the broader proposal in [#11243](https://github.com/triton-lang/triton/pull/11243)); #11260 is pure translator.

For Adam-the-compiler-reader: this PR is the smallest viable "add a new NVIDIA target to Triton" example you can clone. It is ~250 LoC. If we ever want a weekend spike that teaches us the Gluon translator layer, this is the file to start with: `third_party/nvidia/backend/translator/nvidia_target.py` (approximate path; check main at head).

---

## 6. The AMD half: gfx1250-strict

Triton 3.9 inherits the AMD gfx950 (MI350) and gfx1250 (RDNA4 / future AMD consumer) maturation from 3.8, and adds one cherry-pick:

[PR #11945 / #12134 — "gfx1250-strict support"](https://github.com/triton-lang/triton/pull/11945). "Strict" variants of gfx1250 refuse unsupported WMMA and conversion instructions at runtime, which previously caused Triton-generated kernels to crash with "illegal instruction" when autoWS inserted an unsupported WMMA shape. The fix teaches the AMD backend to query the strict feature flag and to drop illegal WMMA shapes down to supported ones at codegen.

The reason this matters cross-vendor: the aref abstraction is **not NVIDIA-specific**. AMD wave specialization (the AMD equivalent of NVIDIA warp specialization) is on the autoWS roadmap in the PyTorch blog, and the gfx1250 cherry-pick plus the earlier gfx950 scaled-dot work are the groundwork. Expect Triton 3.10 or 3.11 to have at least one shape of autoWS running against gfx950 through the same aref IR path. The cross-vendor payoff of landing aref now is exactly this: one abstraction, two backends, one memory planner.

---

## 7. Reading the cherry-pick list as a stability signal

The 3.9 release tracker lists three post-branch-cut cherry-picks. Each of them is small and each of them is a load-bearing fix:

1. **#11945 / #12134 — AMD gfx1250-strict.** Already covered. Compiler safety fix.
2. **#12148 — CUDA driver initialization for PyTorch 2.15 / vLLM.** PyTorch 2.15 shipped a new CUDA driver initialization sequence that races with Triton's own first-kernel-launch handshake. The fix defers Triton's driver init until the first launch. This is the kind of integration bug that is hard to catch pre-release and easy to catch in the first week of vLLM users upgrading.
3. **#12372 — Revert "predicate operands" for NVIDIA async copies.** This one is interesting as a compiler-engineering lesson. Predicate operands on `cp.async` landed earlier in the cycle to let the compiler elide bounds-check branches around async loads. In testing it hit a PyTorch regression (`pytorch/pytorch#201139`); rather than block the release on a fix, the team reverted and will ship the forward-fix in 3.9.1. **Reverting a feature at RC time is a maturity signal, not a failure signal** — it means the project treats "ship on date with known fixes" as higher-value than "ship with experimental features and silent regressions."

If I were summarizing 3.9's stability posture: ship aref, ship 2CTA E2E, ship SM120 translator, defer anything that triggers downstream regressions.

---

## 8. Where Meta's autoWS still differs from upstream 3.9

Even with aref landing in 3.9, the Meta `autoWS` fork (facebookexperimental/triton) and the OSS Triton main branch are not yet aligned. From the PyTorch blog's roadmap, these are the gaps:

| Area | 3.9 upstream | Meta autoWS |
|---|---|---|
| Channel IR | `aref` with multi-consumer | Still pattern-matched channels in Code Partitioner |
| SWP scheduler | Loop-attribute heuristics | Loop-attribute heuristics + Helion autotuning |
| Memory planner | Live-range + aref graph coloring | Dependency-chain analysis, aggressive reuse |
| Partition scheduler | Simple compute/data/epilogue | + mixed partitions (CUDA + data), ping-pong |
| Profile-guided scheduling | Not present | Short-term roadmap |
| Debuggability | `nvws.print_ws_schedule` dumps | TTGIR → TLX kernel converter, visualizations |

The direction of travel is clear: autoWS adopts `aref` next, then donates its memory planner and ping-pong scheduler upstream, probably in 3.10 or 3.11. The thing to watch is whether Meta commits to "autoWS == upstream autoWS modulo profiles" or whether the two forks continue to drift. The Jan 2026 blog promised convergence; 3.9's `aref` landing is the first material step.

---

## 9. Numbers worth memorizing

For anyone writing or interviewing in the Triton / autoWS space:

- **Tawa on H100:** 1.1× cuBLAS GEMM, 1.2× stock Triton attention, parity with CUTLASS FlashAttention-3.
- **Meta autoWS on B200 (per PyTorch blog):** flash-attn forward within 10–20% of cuDNN, 1.5–2× stock Triton, close to Gluon and cuDNN after Helion autotuning + ptxas advanced flags.
- **Triton 3.9 target: 2026-10-21.** Python support window: `>=3.10, <=3.14`. Glibc 2.27+ on x86-64 and AArch64.
- **Supported NVIDIA SMs:** 80 (A100), 86, 89 (Ada), 90 (H100), 100 (B200), 103 (B300), 107 (Rubin, scaffolding), 120 / 121 (consumer Blackwell, via Gluon translator).
- **Supported AMD arches:** gfx942 (MI300), gfx950 (MI350), gfx1250 / gfx1250-strict (RDNA4 / consumer).
- **MLIR version in-tree:** 322 commits landed in October 2026 ([MLIR News 78](https://discourse.llvm.org/t/mlir-news-78th-edition-12th-october-2026/92490)); the one that matters for Triton is strict property assembly becoming the default, which Triton's TTGIR already migrated to in 3.8.

---

## 10. For the compiler-career reader

Three concrete lessons that are worth carrying out of this release into our own work:

1. **IR primitives are the moat.** Tawa's win is not an optimization pass. It is an IR object (`aref`). Once the abstraction exists, every downstream pass gets simpler and every new hardware target gets cheaper. If we want to publish anything in the autoWS / scheduling space, aim for an IR citizen, not a transformation.
2. **"Paper → upstream → production" takes ~12 months in Triton.** Tawa posted Oct 2025, upstream aref lands Oct 2026, Meta autoWS adopts it in 3.10/3.11. That is a tight cycle by compiler standards (LLVM-side RFCs regularly take 2–3 years). The compressing factor is that Triton has an aligned downstream (Meta's Inductor-generated kernels + Helion) that provides workload pressure.
3. **The highest-value PRs to read before a Triton interview are small ones.** [#11260](https://github.com/triton-lang/triton/pull/11260) (SM120 translator) is 250 lines and teaches you the Gluon translator layer. [#10812](https://github.com/triton-lang/triton/pull/10812) teaches you 2CTA WS. [#5622](https://github.com/triton-lang/triton/pull/5622) teaches you the autoWS pass pipeline. The Tawa paper gives you the IR vocabulary. In that order they are a four-week self-study plan.

For my own purposes (Compiler-Path.md): the aref lineage goes into the "canonical case studies" column, next to CuTile / CUTLASS 3.0 and MLIR TOSA Block-Scaled. The capstone project — spconv on Rubin — doesn't need autoWS yet, but the aref mental model is the right one for the sparse-conv indexing kernels I'll have to write.

---

## 11. Takeaway

Triton 3.9 is not a release where you look at the perf numbers of a specific kernel and say "nice, +15% on this shape." It is a release where you look at the IR and say "the compiler can now say what it was always doing." Multi-consumer `aref` is small on the diff but large in what it unlocks: 2CTA M=64 lowering was a 50-line patch *because* aref already carries the cluster-wide channel semantics; the SM120 Gluon translator is 250 lines *because* the Gluon dialect doesn't have to invent new channel objects; Meta's autoWS can adopt it as a drop-in abstraction *because* the IR vocabulary is now shared.

If you only track one Triton release per year, this is probably the one. If you're on a compiler-career trajectory and you're going to read one AI-compiler paper this month, read [arXiv:2510.14719](https://arxiv.org/abs/2510.14719) — and then read [PR #11260](https://github.com/triton-lang/triton/pull/11260) to see the paper's abstraction crossing the production threshold.

---

## Sources

- [Triton 3.9.0 Release Tracker — triton-lang/triton#12121](https://github.com/triton-lang/triton/issues/12121)
- [Triton RELEASE.md on main](https://github.com/triton-lang/triton/blob/main/RELEASE.md)
- [PR #11260 — Gluon SM120/SM121 translator support](https://github.com/triton-lang/triton/pull/11260)
- [PR #10812 — TritonGPU NVWS two-CTA warp specialization (Jokeren)](https://github.com/triton-lang/triton/pull/10812)
- [PR #5622 — Automatic Warp Specialization Optimization (htyu)](https://github.com/triton-lang/triton/pull/5622)
- [Chen et al., *Tawa: Automatic Warp Specialization for Modern GPUs with Asynchronous References*, arXiv:2510.14719](https://arxiv.org/abs/2510.14719)
- [PyTorch blog — Warp Specialization in Triton: Design and Roadmap (2026-01-08)](https://pytorch.org/blog/warp-specialization-in-triton-design-and-roadmap/)
- [Gluon Warp Specialization tutorial](https://triton-lang.org/main/getting-started/tutorials/gluon/warp-specialization.html)
- [Gluon 2CTA Block-Scaled Matmul tutorial](https://triton-lang.org/main/getting-started/examples/gluon/2cta-block-scale-matmul.html)
- [MLIR News, 78th Edition (12th October 2026)](https://discourse.llvm.org/t/mlir-news-78th-edition-12th-october-2026/92490)
- [Hongzheng Chen — personal page](https://chhzh123.github.io/)
