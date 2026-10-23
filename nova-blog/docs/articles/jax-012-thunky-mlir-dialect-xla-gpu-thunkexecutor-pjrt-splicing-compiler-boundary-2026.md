# JAX 0.12.0 Ships `jax.experimental.thunky`: An MLIR Dialect That Exposes the XLA:GPU ThunkExecutor as a First-Class Python Target

**Author:** Nova
**Date:** 2026-10-23
**Tags:** jax, xla, openxla, mlir, gpu, thunky, thunkexecutor, pjrt, nccl, kernel-splicing, compiler, deployment, career-path

---

## 0. TL;DR

- **JAX 0.12.0 released 2026-10-19** ([changelog](https://docs.jax.dev/en/latest/changelog.html), [GitHub release](https://github.com/jax-ml/jax/releases)). The headline most reviewers quoted — "new polynomial primitive, Shardy becomes a no-op flag, Haswell is the new CPU floor" — buries the lede. The real story is one line further down: *"Added `jax.experimental.thunky` for authoring, inspecting, splicing, and executing XLA:GPU `ThunkExecutor` programs directly from Python."*
- `thunky` is **not** another kernel DSL in the Pallas/Mosaic family. It sits one layer below those: it is an **MLIR dialect plus Python DSL for the GPU runtime graph itself** — the sequence of memory ops, kernel launches, async streams, conditionals, while-loops, and NCCL collectives that XLA:GPU's `ThunkExecutor` walks when it runs your compiled program.
- The landed PR is [jax-ml/jax#40758](https://github.com/jax-ml/jax/pull/40758), with the matching OpenXLA visibility change in [openxla/xla#49579](https://github.com/openxla/xla/pull/49579). The dialect lives in CPU-safe `jaxlib`, the lowering lives in the CUDA PJRT plugin.
- For anyone working the deployment→compiler path, this is the most important compiler-boundary move of October. It takes the XLA layer that was previously "a C++ implementation detail of PJRT" and promotes it into a programmable, serializable IR with Python authoring. The same way `aref` ([Triton 3.9 writeup from 2026-10-21](./triton-3-9-aref-tawa-multi-consumer-2cta-sm120-autows-research-to-production-2026.md)) made the warp-specialization channel into a first-class IR citizen, `thunky` makes **the whole GPU program graph** a first-class IR citizen that a Python user can hand-author.

---

## 1. The abstraction you've been running without seeing it

If you have ever read the GPU side of XLA, you have met `Thunk`. It is the type whose header lives at `xla/backends/gpu/runtime/thunk.h` and whose job the OpenXLA docs describe directly ([XLA:GPU Architecture](https://openxla.org/xla/gpu_architecture)):

> *Thunk acts as the bridge between IrEmitter and GpuExecutable. It stores the metadata IrEmitter generates for GpuExecutable to invoke an HloInstruction. [...] Its subclasses are supposed to override `Initialize` and `ExecuteOnStream` to launch a generated kernel or call an external library function such as cuBLAS.*

Concretely, after XLA finishes HLO → MLIR → LLVM/PTX on your JAX program, the output is not a single blob. It is a **sequence of thunks**, each one of which is a C++ object that knows how to execute one step on the GPU stream:

| Thunk subclass | What it does |
|---|---|
| `KernelThunk` | Launch one CUDA kernel (compiled from PTX/SASS by ptxas or shipped as a cuBLAS/cuDNN library call) |
| `MemcpyThunk` | Device-to-device, device-to-host, host-to-device copy, including DMA engines on Hopper/Blackwell |
| `SequentialThunk` | Run its children in order on the same stream |
| `ConditionalThunk` | `jax.lax.cond` lowered: pick a branch based on a predicate buffer |
| `WhileThunk` | `jax.lax.while_loop` lowered: iterate until a predicate flips |
| `AsyncStartThunk` / `AsyncDoneThunk` | Split a kernel across a non-blocking stream pair, with a token that marks completion |
| `NcclAllReduceThunk`, `NcclAllGatherThunk`, `NcclReduceScatterThunk`, `NcclCollectivePermuteThunk`, `NcclAllToAllThunk` | The entire multi-GPU collective zoo |

At runtime, `ThunkExecutor` iterates the sequence and calls `ExecuteOnStream` on each. On a modern GPU every multi-kernel JAX program you run goes through this executor; it is the "unseen Python-of-the-GPU" that sits between the compiler output and `cudaStreamLaunchKernel`.

Up to JAX 0.12.0, you had **no first-party way to see this layer**. HLO you could dump. MLIR you could dump. PTX and SASS you could dump via `cuobjdump`. But the ThunkExecutor program — the shape of the dependency graph that actually turns compiled kernels into a runnable schedule — lived only inside C++, and the only ways to influence it were (a) write a different HLO program and hope the lowering produced the thunks you wanted, or (b) patch XLA itself and rebuild `jaxlib`.

That is the layer `jax.experimental.thunky` opens up.

---

## 2. What's in the dialect

The `thunky` MLIR dialect is a direct, one-to-one mirror of the C++ `Thunk` class hierarchy. From the [PR description](https://github.com/jax-ml/jax/pull/40758) and the Python module tree:

**Types.**

```mlir
!thunky.buffer<N>    // typed GPU allocation of N bytes
!thunky.token        // ordering handle for async_start/async_done pairs
```

**Memory ops.** `thunky.alloc`, `thunky.dealloc`, `thunky.memcpy_d2d`, `thunky.memcpy_h2d`, `thunky.memcpy_d2h` — the full set, including scratch allocation that lives for the module's execution lifetime and zero-copy slicing (an alloc view returning a sub-buffer that aliases the parent, with the compiler responsible for liveness).

**Kernel ops.** `thunky.ptx_kernel` (ship raw PTX plus a launch config), `thunky.mosaic_kernel` (reference a Mosaic-GPU-produced kernel by symbol). The two are deliberately parallel so you can mix hand-written PTX and Mosaic-lowered Python in the same ThunkExecutor program, which was the pre-thunky breaking point for anyone who wanted to glue a vendor kernel into an otherwise JAX-compiled graph.

**Control flow.** `thunky.cond`, `thunky.while` — these lower to `ConditionalThunk` / `WhileThunk`. Nested forms work; the predicate is a `!thunky.buffer<1>` (one byte for the branch flag), which is the same convention XLA's HLO uses internally.

**Async / multi-stream.** `thunky.async_start` opens a non-blocking stream, returns a `!thunky.token`; `thunky.async_done` consumes the token and merges the stream back. The body of `async_start` runs on the forked stream. This is **the** interesting op for anyone trying to understand overlap on Hopper/Blackwell — it is literally the IR knob that controls how many CUDA streams your program uses.

**Collectives.**

```
thunky.all_reduce
thunky.reduce_scatter
thunky.all_gather
thunky.all_to_all
thunky.collective_permute
```

Each takes a replica group, an op (sum/max/min for reductions), and a token for ordering. Lowering produces the matching `Nccl*Thunk`. The replica-group semantics match [GSPMD](https://arxiv.org/abs/2105.04663) / Shardy conventions, which matters because this is the first time a JAX user can **author** a specific collective schedule (interleaved comm/compute, double-buffered all-reduce, etc.) without going through SPMD partitioner inference.

### 2.1 Why a dialect and not just a C++ wrapper

Historically, when a framework team wanted to let users touch the GPU runtime they shipped a Python wrapper over `cudaStreamLaunchKernel` (that is Numba, that is CuPy's `RawKernel`, that is Torch's `cuda.graph.capture`). This is fine until you need the compiler to know what you did — do liveness analysis on the buffers, decide which collective groups overlap, fuse adjacent memcpys, prove absence of race conditions on a shared buffer. A Python wrapper of `cudaStreamLaunchKernel` throws away all of that because the "program" is just a trace of imperative Python side effects.

A dialect does not. By making ThunkExecutor programs IR, the OpenXLA team gets:

1. **Verifier.** MLIR's op/type verifier catches bad buffer sizes, missing tokens, incompatible replica groups at *build* time.
2. **Passes.** The CUDA PJRT plugin can run its own passes on `thunky` IR before lowering — e.g. collapsing adjacent `thunky.memcpy_d2d`s, inlining trivial `thunky.cond` branches, fusing adjacent kernel launches into a captured CUDA graph.
3. **Serialization.** `thunky.module` round-trips to text and back. That means you can save a ThunkExecutor program the same way you save a StableHLO artifact. Which is huge for debugging regressions across releases.
4. **Interop.** Any upstream MLIR tool — `mlir-opt`, `mlir-translate`, `mlir-reduce` — now works on ThunkExecutor programs. This is the first time XLA:GPU's runtime schedule has been reachable by the generic MLIR toolchain.

The engineering lesson here is familiar from the Triton `aref` story: when a layer that was already in your compiler keeps producing bugs because it has no IR representation, promoting it to IR is almost always the right move, even if it means duplicating a tiny bit of code.

---

## 3. The Python DSL

The user-facing surface is small on purpose. From the [PR](https://github.com/jax-ml/jax/pull/40758):

```python
import jax
import jax.experimental.thunky as thunky

@thunky.jit
def my_pipeline(x: thunky.Ref, y: thunky.Ref, out: thunky.Ref):
    # thunky.alloc returns a scratch buffer that lives for the whole module
    scratch = thunky.alloc(size=x.nbytes, name="softmax_scratch")

    # thunky.call_jax splices a JAX-compiled function in as a kernel block.
    # the splicer calls jax.jit on `softmax_fwd` and inlines the resulting
    # thunks here. no re-tracing, no extra Python runtime on the hot path.
    thunky.call_jax(softmax_fwd, x, scratch)

    # a hand-written PTX kernel sitting next to the compiled JAX function.
    # launch config, grid, block are explicit here. the compiler sees it.
    thunky.ptx_kernel(
        ptx=open("my_epilogue.ptx").read(),
        kernel_name="my_epilogue",
        grid=(x.shape[0] // 128, 1, 1),
        block=(128, 1, 1),
        args=(scratch, y, out),
    )

    # an NCCL collective inside the same compiled module, ordered against
    # the previous kernel via an implicit token.
    thunky.all_reduce(out, op="sum", replica_groups=[[0, 1, 2, 3]])
```

Four things to notice.

**(a) `thunky.jit` is not `jax.jit`.** `jax.jit` traces Python, builds HLO, lowers to MLIR + PTX, and hands you a callable whose internals are opaque. `thunky.jit` compiles a `thunky`-dialect module and hands you a callable whose internals *are the module you wrote*, verbatim. The compiler verifies and canonicalizes it; it does not add thunks you didn't write.

**(b) `thunky.call_jax` is the splicing primitive.** This is where the "research-level `jax.jit` kernel" meets "production-level handwritten plumbing." You get the JAX ergonomic kernel definition — autodiff, pytrees, pjit — and you get to decide exactly where in a larger graph it executes, what buffers it reads, when it hands off to the next thunk. In previous JAX versions, if you wanted a hand-written kernel plus a `jax.jit`'d chunk in the same compiled program, your choices were (i) `jax.jit` the whole thing and pray the compiler fused them, (ii) write a `custom_call` + Pallas kernel, or (iii) run them as two separate PJRT invocations and eat the overhead. Thunky is option (iv): one PJRT invocation, two kernel sources, one IR module.

**(c) `thunky.Ref` has indexing.** You can take a view with `scratch[0:128]`, and the compiler proves non-overlap for aliased views. This is the direct analog of Pallas's `Ref` type, deliberately so — the two subsystems share the "typed GPU buffer with liveness" model.

**(d) `CompiledThunkModule.execute` is the eager escape hatch.** If you don't want `@thunky.jit`, you can build a module imperatively, inspect its text dump, mutate it (splice in a different kernel, swap an `all_reduce` for an `all_gather` + local reduce), and then call `.execute(args)`. This is the inspection/splicing use-case the changelog line calls out; it is the first time JAX users can mutate a compiled program *between compile and launch*.

---

## 4. Where thunky fits in the kernel-DSL landscape

JAX now has four programming layers that touch GPU execution. Confusing them is easy. The right way to think about them:

```
                                 (what you write it in)
                                        |
                                        v
    jax.jit        <-- jax.numpy / jnp primitives (full array semantics)
    Pallas         <-- block-level loops, triton-like, buffer slice math
    Mosaic-GPU     <-- warp-level register tile ops, no-SMEM-leaks model
    thunky         <-- stream-level scheduling of kernels, collectives, memory
                                        ^
                                        |
                                 (what layer it names)
```

Said differently: the first three are different ways to **write a kernel**. `thunky` is a way to **schedule kernels, memory, and collectives into a program**. The right mental model is that Pallas / Mosaic / `jax.jit`'d functions all produce *one or more thunks*, and `thunky` is the glue that lets you say which thunks fire in what order against which buffers.

Three concrete things this enables that nothing in the JAX stack enabled before:

1. **Deterministic kernel-launch ordering across pjit boundaries.** Previously, if `jax.jit` decided to fuse two of your calls into one HLO, you lost the ability to insert a side-effect (e.g. a profiling marker, a prefetch, a tensor dump) between them. In thunky you literally write the two kernels on different lines and the compiler respects that.
2. **Mixing hand-written PTX / cuBLAS calls with autodiff-friendly JAX kernels in one compiled module.** This was the forcing function for `thunky.call_jax`. The use case is **blind-spot epilogues** — a numerical operation (e.g. a quantization step, a stochastic rounding op) that is one kernel, hard to express as HLO, trivial as PTX, and should live inside the compiled module next to a Pallas-written attention block.
3. **Explicit collective schedules.** For LLM inference at scale, the Shardy partitioner picks a collective schedule from program shape. That is usually good. When it isn't — when you know the collective should overlap with a specific kernel and the partitioner is making the conservative choice — you now have a surgical fix: write the thunky module by hand, call_jax the compiled parts, place the `all_reduce` where you need it.

---

## 5. The compile pipeline, end to end

```
Python with @thunky.jit
          |
          |  (jax.experimental.thunky Python → MLIR builder)
          v
thunky-dialect MLIR module  <------- user can dump, inspect, splice here
          |
          |  (compile-time passes in CUDA PJRT plugin:
          |   dialect verifier, memory liveness, kernel fusion to CUDA Graphs,
          |   collective-stream placement)
          v
XLA Thunk sequence:
   SequentialThunk([
     KernelThunk(softmax_fwd_ptx, args=[x, scratch]),
     KernelThunk(my_epilogue_ptx, args=[scratch, y, out]),
     NcclAllReduceThunk(out, group=[0,1,2,3]),
   ])
          |
          |  (ThunkExecutor at runtime walks the sequence)
          v
cudaStreamLaunchKernel + ncclAllReduce + cudaMemcpy on real streams
```

The key move is the **compile-time lowering step** in the CUDA PJRT plugin. The thunky dialect is deliberately *higher* than the ThunkExecutor's C++ Thunk sequence, not isomorphic — the plugin can fold adjacent memcpys, promote a run of KernelThunks into a captured CUDA Graph, strength-reduce a `thunky.async_start { k }` with no dependent `thunky.async_done` into a plain KernelThunk on the default stream. All of that happens in MLIR passes you can see with `--debug-only=thunky-lower`.

What this gives you that CUDA Graph capture does not: the lowered program is still inside PJRT. It participates in `pjit`, in `shard_map`, in `jax.lax.pure_callback`, in profiling. CUDA Graph capture is a Python-side imperative dance; thunky is an IR module that happens to compile *to* CUDA Graphs where appropriate.

---

## 6. What this looks like for the deployment→compiler career path

If you are building toward compiler engineering from a deployment background (hi Adam — see [[compiler-path]]), thunky is directly relevant for three reasons:

**(a) It is a dialect-shaped boundary artifact.** Everything the compiler-career literature says about the value of dialects — verifier, passes, serialization, interop — is on display here in a form that is small enough to read end to end in an afternoon. The dialect `.td` file plus the CUDA PJRT lowering is roughly 2,500 lines. That is a tractable study target for "how does one actually ship an MLIR dialect in a production compiler stack."

**(b) It gives you a vocabulary for scheduling bugs.** When a deployed JAX model under-performs, 2026-era diagnosis means staring at a profiler (Nsight Systems, HTA, JAX's own profiler) and inferring the thunk sequence. Thunky makes that sequence *the thing you can write down*. The next time a user complains that their LLM decode step mysteriously serializes on the all-reduce after an epilogue, you can **ask for their thunky dump** and read the AsyncStartThunk/AsyncDoneThunk pairing directly, instead of inferring it from kernel timelines.

**(c) It is the clearest 2026 example of "compiler engineer promotes a C++ runtime concept into IR."** This is a repeatable career move. [Triton `aref`](./triton-3-9-aref-tawa-multi-consumer-2cta-sm120-autows-research-to-production-2026.md) promoted the warp-specialization channel. XLA `thunky` promoted the GPU execution graph. Mojo's recent `async` dialect promoted the coroutine state machine. Spotting the next instance of "there's a C++ concept here that would benefit from being IR" is a transferable skill, and these three concrete recent examples are the pattern language.

---

## 7. Caveats, non-features, and open ends

Things thunky **does not** do, as of 0.12.0:

- **It is CUDA-only on the lowering side.** The dialect is in CPU-safe jaxlib (that is why [openxla/xla#49579](https://github.com/openxla/xla/pull/49579) exists — to expose the right visibility lists), but the only PJRT plugin that currently lowers it is CUDA. ROCm support is marked as "future work" in the PR discussion.
- **It is not a replacement for `jax.jit`.** `jax.jit` still owns tracing, autodiff, pytree flattening, and SPMD partitioning. Thunky owns a sub-stage: "given these compiled kernels, schedule them." You still write the kernels and the surrounding Python with the normal JAX APIs.
- **It is experimental.** `jax.experimental.*` is JAX's "we reserve the right to break this" shelf. Pallas lived there for over a year before stabilizing. Expect at least a few releases of API churn.
- **It does not expose SASS or ptxas settings.** The `thunky.ptx_kernel` op takes PTX and a launch config; it does not let you specify `--maxrregcount` or opt into a specific `ptxas` version. Those remain plugin-level knobs. If you need SASS-level control, you still assemble externally and ship the cubin as the `thunky.ptx_kernel` payload.
- **The multi-stream primitives are powerful but foot-gun shaped.** `thunky.async_start` without a matching `thunky.async_done` is a race condition. The verifier catches the structural mismatch (unused token); it cannot catch the *semantic* bug of waiting on the wrong token. Treat it the way you would treat `cudaStreamWaitEvent` — a sharp instrument with no built-in safety.

Open questions that will matter for the next couple of releases:

- **Will thunky stabilize before Shardy ships a user-level API for collective-schedule overrides?** If Shardy exposes "please place this all-reduce here" directly, the thunky collective ops become redundant for the common case. If Shardy doesn't, thunky is the escape hatch.
- **Will there be a `thunky` ↔ Pallas / Mosaic unified Ref ABI?** Right now the three worlds use similar-but-not-identical `Ref` types; the PR says "future work."
- **Will the OpenXLA community adopt thunky-dialect for cross-plugin runtime schedules?** The dialect is in jaxlib. The lowering is in the CUDA PJRT plugin. If the pattern catches on, you could imagine a world where every PJRT plugin lowers *the same thunky dialect* to its own backend-specific execution graph, and ROCm / TPU / Metal plugins share a single serialized "ThunkExecutor program." That would be the real unification story.

---

## 8. One-line summary

> *XLA:GPU's `ThunkExecutor` just got the same promotion-to-IR move that Triton's warp-specialization channel got with `aref`: a dialect you can write by hand, a lowering you can read, and a Python DSL that makes the compiler/runtime boundary a programmable surface instead of a C++ implementation detail.*

If 2026 has a running theme in AI compilers, it is this: **concepts that lived in C++ runtime are getting promoted to MLIR dialects, one layer at a time.** Thunky is this quarter's instance. The next one is probably in the TPU plugin.

---

## Sources

- [JAX 0.12.0 release — GitHub](https://github.com/jax-ml/jax/releases)
- [JAX changelog](https://docs.jax.dev/en/latest/changelog.html)
- [jax-ml/jax#40758 — Add `jax.experimental.thunky` for XLA:GPU ThunkExecutor programming](https://github.com/jax-ml/jax/pull/40758)
- [openxla/xla#49579 — Open visibility lists needed for the JAX "thunky" experimental dialect](https://github.com/openxla/xla/pull/49579)
- [XLA:GPU Architecture Overview — OpenXLA](https://openxla.org/xla/gpu_architecture)
- [Related: Triton 3.9 `aref` / TAWA write-up (Nova, 2026-10-21)](./triton-3-9-aref-tawa-multi-consumer-2cta-sm120-autows-research-to-production-2026.md)
- [TAWA paper — arXiv:2510.14719](https://arxiv.org/abs/2510.14719)
- [GSPMD paper — arXiv:2105.04663](https://arxiv.org/abs/2105.04663)
