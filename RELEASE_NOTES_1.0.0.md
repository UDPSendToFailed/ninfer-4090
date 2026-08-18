# NInfer-4090 v1.0.0 Release Notes

NInfer-4090 v1.0.0 adds Direct3D 12 memory residency management to evict background Windows allocations, kernel optimizations across GDN linear attention, FlashAttention prefill, and quantized GEMMs, configurable vision scratchpads, and 2D asynchronous memory batching.

Context scaling is verified up to 567,000 tokens (text-only) and 380,000 tokens (with Vision + MTP4), with a verified 100% recall (5/5 needles) on a 359,169-token needle-in-a-haystack test on a single 24 GB RTX 4090.

---

## What's Changed

### 1. Direct3D 12 WDDM Memory Residency & Eviction
* **D3D12 Residency Locking (`src/core/arena.cu`):** Primary device memory arenas allocate via a Direct3D 12 shared heap configured with `D3D12_RESIDENCY_PRIORITY_MAXIMUM` and `D3D12_RESIDENCY_FLAG_DENY_OVERBUDGET`, imported directly into CUDA via external memory interoperability.
* **WDDM Background Eviction:** Eagerly commits physical pages at startup via `cudaMemset`, forcing Windows WDDM (`VidMm`) to page cold desktop compositor and background application buffers into system RAM instead of throttling CUDA allocations over PCIe.
* **DXGI LUID Adapter Matching:** Resolves the exact DXGI adapter matching the active CUDA device LUID, ensuring correct allocation on multi-GPU and hybrid iGPU systems.
* **WDDM Evictable Capacity Planning (`src/targets/registry.cpp`):** Updates memory planning to budget against physical VRAM minus a 512 MiB DWM scanout floor, replacing the conservative `cudaMemGetInfo` unevicted snapshot and unlocking up to ~23.5 GiB of usable VRAM.

### 2. Prefill Attention Causal Tile Partitioning
* **Branchless Interior Tiles (`src/ops/kernel/gqa_attention_prefill_*.cuh`):** Partitions KV block traversal in BF16 and INT8 GQA prefill kernels into Phase 1 (fully causal interior tiles with branchless online softmax) and Phase 2 (boundary diagonal tiles with exact per-element causal masking), eliminating warp divergence on interior tiles.
* **Specialized `cp.async` Staging:** Splits KV staging into `gqa_prefill_stage_kv_full` (unconditional 128-bit `cp.async.cg` copies) and `gqa_prefill_stage_kv_partial`.
* **Vectorized Indexing:** Replaced runtime division/modulo operations with bit shifts (`head_page_index << 14`, `element >> 8`, `element & 255`) and hoisted epilogue output pointers to write packed 32-bit `__nv_bfloat162` pairs.
* **Performance:** Reduces prefill attention kernel latency from ~76 µs to 56–60 µs.

### 3. Configurable Vision Scratchpad (`--vision-max-tokens`)
* **Dynamic Sizing:** Added `--vision-max-tokens` (alias `--vision-limit`) to CLI and server options, defaulting to 8192 tokens (previously hardcoded to 32,768).
* **VRAM Recovery:** Saves ~1.3 GiB to 1.5 GiB of static VRAM workspace for standard image workloads while allowing scaling up to 36k+ tokens for video when requested.
* **Admission Budget Enforcement:** Synchronized the frontend preprocessor budget with the configured limit so requests exceeding the scratchpad fail cleanly with HTTP 413 `media_budget_exceeded` before invoking the encoder.

### 4. Quantized GEMM Dequantization & Bank Conflict Elimination
* **Hardware PTX `bfe.s32` (`src/ops/linear/`):** Replaced multi-instruction bit-shifting, masking, and sign-extension ALU sequences in Q4, Q5, and Q6 MMA decode atoms with single-cycle `bfe.s32` signed bitfield extractions.
* **Zero-Bank-Conflict Staging:** Restructured Q4 and Q5 shared memory staging into aligned 32-bit loads (`uint32_t`) on lanes 0–7 and distributed codes via warp shuffles, eliminating 4-way shared memory bank conflicts (128 serialized replay cycles per row).
* **Scale Caching:** Converted FP16 scales once on lane 0 and distributed across the warp via `__shfl_sync` register moves.
* **Address Arithmetic Hoisting:** Hoisted base row-group calculations in `stage_quant` across codes, high bits, and scale planes in Q4, Q5, Q6, and W8 row-split MMA kernels.

### 5. GDN Recurrent State & Causal Conv1D Vectorization
* **XOR Butterfly All-Reduce (`src/ops/linear_attention/`):** Replaced serialized per-row reductions and broadcast trees in `apply_gdn_transition` with a 5-step butterfly all-reduce via `__shfl_xor_sync` across all 4 rows simultaneously, chained with single-cycle `fmaf`.
* **Packed Value Loads:** Introduced 64-bit packed value loads (`RawValuePack` / `Bf16x4Pack`) on lane 0 to distribute values at load time, eliminating per-token gather shuffles.
* **Constant Stride Bumps:** Hoisted coordinate math in `recurrent_fold_body` and replaced iterative multi-index math with compile-time pointer stride increments.
* **Vectorized Causal Conv1D (`src/ops/launcher/causal_conv1d.cu`):** Added 32-bit `__nv_bfloat162` packed loads/stores and dual FP32 accumulation across all 5,120 channels with alignment guards and scalar fallbacks.
* **Chunked GDN Optimizations:** Vectorized shared memory zero-initialization (`float4`) and fragment storage (`float2`), and pruned decaying product calculations on strictly upper-triangular tiles.

### 6. 2D Asynchronous Memory Batching
* **Batched State Pool & Cyclic KV Copies (`src/core/`):** Replaced iterative per-layer 1D `cudaMemcpyAsync` and `cudaMemsetAsync` loops in `LinearAttentionStatePool` and `CyclicKVCache` with uniform-stride 2D operations (`cudaMemcpy2DAsync`, `cudaMemset2DAsync`).
* **Overhead Reduction:** Reduced turn-checkpoint state copy dispatch overhead from 96 discrete driver calls to 2 batched operations (~5 µs per call).
* **Shift-Based Indexing:** Replaced 64-bit division/modulo math in `kv_cache_append_prefix.cuh` with power-of-2 bit shifts (`>> 3`, `<< 7`, `<< 6`) and bitwise masks (`& 7`).

### 7. Host Inlining & Build System
* **MSVC Host Optimization (`CMakeLists.txt`):** Explicitly added `/O2 /Ob2 /Oi /Ot` for C, CXX, and NVCC host compilation under Ninja, resolving host-dispatch stalls between kernel launches.
* **CUDA Graph Allowance Tightening (`src/targets/qwen3_6/impl/runtime/layouts_impl.h`):** Replaced the flat 1,024 MiB graph reservation with calibrated allocations of 64 MiB (No-Spec) and 256–320 MiB (MTP), recovering ~768 MiB of VRAM padding.
* **Linker & Testing:** Added `d3d12.lib` and `dxgi.lib` to `ninfer_core` on Windows, registered `ninfer_test_e8_codec` with CTest, and resolved NVFP4 A4 architecture guards on non-Blackwell hardware.
* **OpenAI Schema Compatibility (`src/serve/openai_schema.cpp`):** Supported content-part arrays on tool-role messages in Chat Completions parsing.

---

## Measured Performance on NVIDIA GeForce RTX 4090

Evaluated on official Qwen3.8-27B (16.96 GiB groupwise `.ninfer` artifact, CUDA 13.3, 24 GB RTX 4090):

### Standard Product Benchmark Matrix (`ninfer_bench`)

| Test Case | Configuration | Throughput | Notes / Acceptance |
|---|---|---:|---|
| **Prefill (`pp2048`)** | `pp2048`, Chunk 1024, INT8 KV | **`1,863.8 ± 1.8 tok/s`** | Saturated compute, sub-2 tok/s variance |
| **Prefill (`pp4096`)** | `pp4096`, Chunk 1024, INT8 KV | **`1,849.3 ± 2.1 tok/s`** | Deep chunked prefill |
| **Prefill (`pp512`)** | `pp512`, Chunk 1024, INT8 KV | **`1,736.8 ± 36.0 tok/s`** | Low-latency shallow prefill |
| **Decode: Code / Math (MTP3)** | `tg128`–`tg1024`, `--greedy`, MTP3 | **`103.5 – 148.2 tok/s`** | 55–91% draft acceptance on code |
| **Decode: Code & Schemas (MTP4)** | `tg128`–`tg1024`, `--greedy`, MTP4 | **`96.8 – 129.9 tok/s`** | 46–88% draft acceptance on schemas |
| **Decode: Bench Corpus (MTP3)** | `tg128`, MTP3 + Draft Head | **`83.7 ± 2.9 tok/s`** | 36.0% acceptance on mixed corpus |
| **Decode: Baseline (MTP0)** | `tg128`, no speculation, CUDA Graph | **`51.4 ± 0.5 tok/s`** | Single-token base autoregressive decode |
| **360k Needle-in-a-Haystack** | 359,169 prompt tokens, `rk2v4-e8` | **`100% (5/5 Needles)`** | 666.7 tok/s avg prefill, exact recall |

---

## Verified Context Ceilings Matrix (RTX 4090, 24 GB)

The physical memory boundaries below were binary-searched on an RTX 4090 (24 GB) under D3D12 WDDM residency management (rounded to the nearest thousand):

| Profile / Mode | Speculation | KV Mode | Physical Max Context | Cosine Sim vs FP32 | Recommended Safe Context |
|---|---|---|---:|---|---:|
| **Text-Only** | No-Spec (MTP0) | **`rk2v4-e8`** (2-bit $E_8$ Cylinder) | **`567,000 tok`** | 96.2% | **`500,000 tok`** |
| **Text-Only** | No-Spec (MTP0) | **`rk4v4-e8`** (4-bit $E_8$ Lattice) | **`433,000 tok`** | 98.7% | **`400,000 tok`** |
| **Text-Only** | No-Spec (MTP0) | **`rk4v4`** (Hadamard 4-bit) | **`433,000 tok`** | 97.8% | **`400,000 tok`** |
| **Text-Only** | No-Spec (MTP0) | **`rk8v4`** (Hadamard 8-bit) | **`294,000 tok`** | 99.4% | **`270,000 tok`** |
| **Text-Only** | No-Spec (MTP0) | **`int8`** (Uncompressed INT8) | **`223,000 tok`** | 99.8% | **`200,000 tok`** |
| **Text-Only** | MTP4 Speculation | **`rk2v4-e8`** | **`462,000 tok`** | 96.2% | **`430,000 tok`** |
| **Text-Only** | MTP4 Speculation | **`rk4v4-e8`** | **`352,000 tok`** | 98.7% | **`320,000 tok`** |
| **Text-Only** | MTP4 Speculation | **`rk4v4`** | **`352,000 tok`** | 97.8% | **`320,000 tok`** |
| **Text-Only** | MTP4 Speculation | **`rk8v4`** | **`239,000 tok`** | 99.4% | **`210,000 tok`** |
| **Text-Only** | MTP4 Speculation | **`int8`** | **`181,000 tok`** | 99.8% | **`160,000 tok`** |
| **Vision (8k Default)** | MTP4 Speculation | **`rk2v4-e8`** | **`415,000 tok`** | 96.2% | **`380,000 tok`** |
| **Vision (8k Default)** | MTP4 Speculation | **`rk4v4-e8`** | **`317,000 tok`** | 98.7% | **`280,000 tok`** |
| **Vision (8k Default)** | MTP4 Speculation | **`int8`** | **`163,000 tok`** | 99.8% | **`140,000 tok`** |
| **Vision (4k Small)** | MTP4 Speculation | **`rk2v4-e8`** | **`434,000 tok`** | 96.2% | **`400,000 tok`** |
| **Vision (4k Small)** | MTP4 Speculation | **`rk4v4-e8`** | **`332,000 tok`** | 98.7% | **`300,000 tok`** |

---

## Package Contents

* `ninfer.exe`: Interactive CLI generation binary.
* `ninfer-serve.exe`: OpenAI / Anthropic compatible HTTP API server.
* `ninfer_bench.exe`: Product throughput benchmark harness.
* Required runtime media & networking DLLs.
