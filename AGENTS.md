# AGENTS.md

These rules apply to the whole repository.

## Governing Objective & Engineering Priorities

Deliver maximum single-GPU inference performance, stability, and correctness for Qwen models on the NVIDIA GeForce RTX 4090 (`sm_89`, AD102).

1. **Functional and Numerical Correctness**: Mathematical fidelity comes first. Every floating-point operator must match its independent naive FP32/FP64 mathematical oracle.
2. **Architectural Quality and Ownership**: Maintain clear separation across core, artifact, ops, runtime, and serving layers. Never take sloppy shortcuts, duct-tape workarounds, or introduce code fragmentation.
3. **Hardware-Native Performance**: Optimize for Ada Lovelace (`sm_89`): 128 SMs, 72 MB L2 cache, 24 GB GDDR6X, Ada tensor cores (FP16/BF16 MMA), and vectorized 16-byte memory transactions.
4. **Reliability & Thread Safety**: Preserve strict boundaries on mutexes, non-blocking stream lifecycles, and CUDA Graph address stability.

---

## Product Contract & Target Platform

- **Target Hardware**: NVIDIA GeForce RTX 4090 (24 GB GDDR6X, AD102, 128 SMs), Windows 11, CUDA 13.x.
- **Supported Identities**: `qwen3.8-27b/groupwise-int`, `qwen3.6-27b/groupwise-int`.
- **KV Cache Quantization**: Uncompressed `int8`, `rk4v4-e8` (4-bit Conway-Sloane $E_8$ lattice), `rk2v4-e8` (2-bit $E_8$ cylinder).
- **Runtime Features**: MTP speculative decoding (K=1..15), D3D12/WDDM residency eviction management (`--wddm-evictable-budget`), DirectStorage 1.3 CoW state caching, CUDA Graph execution, and unified OpenAI/Anthropic HTTP serving (`ninfer-serve`).
- **Explicitly Excluded / Deleted**: Blackwell `sm_120`, NVFP4 formats, multi-GPU distributed orchestration, and dynamic plugin architectures.

---

## Repository Boundaries & Component Ownership

- `include/ninfer/`: In-tree public C++ Engine API (`engine.h`, `types.h`). Opaque interface used by CLI, server, and benchmarks.
- `src/core/`: Device memory allocation, `DeviceArena`, D3D12 residency management, DirectStorage 1.3 state caching, physical KV cache containers, and CUDA stream management.
- `src/artifact/`: Binary `.ninfer` container reader, parser, descriptor tables, and double-buffered DMA weight materializer.
- `src/ops/`: Mathematically closed operator implementations, fused kernels, and SIMT/MMA routines specialized for `sm_89`. Op ownership follows mathematical contracts, not target models.
- `src/targets/qwen3_6/`: Qwen3 family execution runtime, planning algorithms (`SequencePlan`, `RequestPlan`, `Program`), tokenizer, and chat templates.
- `src/serve/`: HTTP server, protocol translation (OpenAI `/v1/chat/completions`, Anthropic `/v1/messages`), and generation services.
- `apps/`: In-tree CLI (`ninfer.exe`) and server (`ninfer-serve.exe`).
- `bench/`: Standalone benchmarking binaries (`ninfer_bench.exe`).
- `tests/`: Unit, regression, and numerical oracle tests (CTest suite).

---

## Development & Verification Rules

1. **Strict Build Command Policy**: Never run build commands (`ninja`, `cmake`, etc.) unless explicitly instructed by the user.
2. **Mandatory CTest Verification**: For every new feature or critical change, implement corresponding CTests. All 84 CTests must pass with zero failures before completing work.
3. **Numerical Verification**: Verify operator correctness against independent FP32/FP64 mathematical oracles. Pairwise parity against previous implementations is supplementary only.
4. **CUDA Architecture & Kernel Engineering Rules**:
   - **Stream Concurrency & Ordering**: Operations on legacy default Stream 0 do not synchronize with non-blocking streams (`cudaStreamNonBlocking`). Any allocation, touch-fill, or reset on Stream 0 must be explicitly drained (e.g. `cudaDeviceSynchronize` or stream events) before publishing pointers to non-blocking streams like `load_stream`.
   - **Wave Sizing & Grid Launch**: Size persistent and resident grids to live hardware SM counts (`device_sm_count()`, 128 SMs on RTX 4090) to guarantee full integer waves. Never hardcode SM counts from other architectures (e.g. 170 SMs), which creates unbalanced trailing straggler waves.
   - **L2 Cache Locality & CTA Rasterization**: When model weights exceed the 72 MB L2 cache, apply column-tile-major rasterization so resident CTAs sweep all column tiles of a row band concurrently, maximizing L2 cache hits and avoiding DRAM bandwidth degradation.
   - **Register Allocation vs Occupancy**: Per CUDA Best Practices, higher theoretical occupancy does not equate to higher performance. On memory-bound kernels, dynamically adjust `__launch_bounds__` to provide 128 registers per thread, eliminating spill stores/loads to local memory over chasing block occupancy.
   - **Vectorized Transactions**: Decoded weights and shared memory staging must use 16-byte vectorized transactions (`store_vec` / 128-bit stores) to minimize instruction count and shared memory bank pressure.
   - **Warp Shuffles vs Shared Memory Broadcast**: Avoid `__shfl_sync` with dynamic, non-uniform source lanes across sub-warp branches (which force compiler ABI branches and stack frames). Prefer direct `LDS` reads (hardware broadcast unit) when data is already resident in shared memory.
   - **Host Buffer Cacheability**: Host buffers read by CPU threads (e.g. `ordinary_host_egress`) must use cacheable pinned memory (`cudaMallocHost`), never write-combined memory (`cudaHostAllocWriteCombined`), avoiding severe uncached bus penalties.
   - **Instruction Cache Residency**: In multi-slab SIMT kernels, avoid uncontrolled full loop unrolling that bloats SASS beyond 32 KiB and exceeds the SM L1 instruction cache. Use `#pragma unroll 2` with software pipelining.
   - **Shared Activation Amortization**: In SIMT decode GEMMs where activations arrive as BF16 and require FP32 widening for `FFMA`, process two rows per CTA to share the widened activation and amortize global loads, cutting `PRMT` conversion instructions by 50% and global load sectors by ~48%.
   - **Whole-Program Device Optimization**: Keep separable compilation disabled (`-rdc=false`) to enable whole-program device optimization, cross-TU inlining, and global load pipelining across loop back edges.

> [!IMPORTANT]
> These are historically proven, empirically verified engineering invariants for this repository. However, never hesitate to call out sloppy coding, duct-taped workarounds, or architectural anti-patterns whenever reviewing or proposing code changes. Quality and hardware truth always take precedence over convenience.

---

## Local Environment

- **Operating System**: Windows 11
- **Toolchain**: Microsoft Visual Studio 2022 (MSVC x64), CUDA Toolkit 13.x
- **Build System**: CMake with Ninja generator (`build-ninja/`)
- **Compilation Flags**: Whole-program device compilation with `--split-compile=0` for full multi-threaded CPU utilization.

---

## Commits

Create commits only when explicitly requested by the user. Use Conventional Commit subjects:
- `feat`: User-facing features or new CLI/serving options.
- `perf`: Measured kernel, memory, or runtime optimizations.
- `fix`: Bug, race condition, or regression fixes.
- `test`: Adding or updating test suites.
- `build`: CMake, toolchain, or compiler configuration.
- `chore` / `docs`: Maintenance, cleanup, or documentation.
