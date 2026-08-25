# NInfer-4090 v1.1.0 Release Notes

NInfer-4090 v1.1.0 expands linear Multi-Token Prediction (MTP) draft capacity up to K=15, implements DirectStorage 1.3 NVMe-to-VRAM DMA prompt caching with Copy-on-Write (CoW) page journaling, eliminates active DRAM register spills in Small-T GEMM and sampling kernels, accelerates transcendental activations via hardware SFU intrinsics, aligns GQA decode grids to 128-SM wave boundaries, and embeds a WebUI dashboard into the server binary.

---

## What's Changed

### 1. DirectStorage 1.3 CoW Prompt State Cache (`--disk-cache`)
* **DirectStorage 1.3 Engine (`src/core/direct_storage_engine.*`):** Implements DirectStorage 1.3 DMA streaming on Windows. Uses shared D3D12 hardware fences imported into CUDA via `cudaImportExternalSemaphore` to synchronize NVMe-to-VRAM page transfers directly onto CUDA streams without CPU intervention.
* **Copy-on-Write Page Journaling (`src/core/disk_state_cache.*`):** Stores deduplicated 64-token Text KV pages in a shared 4 KiB sector-aligned file (`pool_data.ninfer_pages`) and persistent binary index (`pool_index.ninfer_idx`). Persists recurrent states (GDN linear state, MTP KV, tail hidden) in metadata manifests (`.ninfer_manifest`). Saves only newly materialized delta pages on turn completion.
* **Compaction Guards & Bulk LRU Eviction:** Implements multi-condition compaction guards ($\ge 256\text{ dead pages}$, $\ge 2.0\text{ GiB dead space}$, $\ge 25\%\text{ fragmentation}$) and 75% low-watermark bulk LRU pruning to prevent SSD write wear.
* **Page-Major Staging Kernels (`src/ops/kernel/kv_paged_staging.cuh`):** Added `gather_paged_kv_kernel` and `scatter_paged_kv_kernel` to organize VRAM staging in Page-Major order across heterogeneous cache planes.
* **Driver Synchronization:** Synchronizes D3D12 CPU fences in `release_staging()` before COM teardown to prevent driver crashes during rapid restore cycles.

### 2. Speculative Decoding Extension ($K \le 15$, $W \le 16$)
* **MTP Capacity Expansion:** Extends the linear MTP draft window ceiling from $K=5$ to $K=15$ ($W \le 16$) across configuration, round state layouts, CLI options, and verification wrappers (`src/ops/wrapper/mtp_round.cpp`).
* **Dynamic Shared Memory Allocation (`src/ops/launcher/gqa_attention_decode.cu`):** Configures `DynamicArena = true` for $TokenTile \ge 7$ ($W_c = 6$ for 27B group-6, $W_c = 8$ for 35B group-8) via `cudaFuncSetAttribute`, keeping static shared memory allocation under the 48 KB linker limit on sm_89.
* **Small-T Verification Routing (`src/ops/wrapper/gqa_attention.cpp`):** Sets `kSmallTChunkTokens = 8` and routes verification widths $W \le 16$ on 27B (`q_heads = 24`) to `ChunkedSmallT`, synchronizing `gqa_attention_cached` with `gqa_attention_resolve_route` to guarantee exact workspace allocation parity.

### 3. Hardware SFU & Arithmetic Vectorization
* **Base-2 SFU Math Intrinsics (`src/ops/common/math.cuh`):** Rewrote `silu`, `sigmoid`, and GDN `softplus` using base-2 SFU instructions (`ex2.approx.f32`, `lg2.approx.f32`, `__frcp_rn`) via the $kNegLog2e$ identity, reducing transcendental evaluation latency from ~40–50 cycles to 5 hardware instructions.
* **$E_8$ Root Codec Assembly Acceleration (`src/ops/kernel/e8_root_codec.cuh`):** Replaced manual shift-mask-or bit-packing sequences with single-cycle inline PTX `bfi.b32` instructions and hardware `redux.sync.add.s32` sub-warp reductions.
* **Packed BF162 Vectorization:** Vectorized residual and bias additions across all layers with native `__hadd2` and `__hadd` in `add_bias.cuh`, `residual_add.cuh`, and quantized GEMM epilogues (`q5_rowsplit_gemm_mma.cuh`, `w8_rowsplit_gemm_mma.cuh`).

### 4. Register Spill Elimination & Launch Bounds Tuning
* **Launch Bounds Tuning (`src/ops/linear/q4/q4_small_t_mma.cuh`):** Relaxes launch bounds to `__launch_bounds__(256, 2)`, raising the per-thread register ceiling to 128 and eliminating all DRAM local stack spills across 114 template instances.
* **Sampling Stack Elimination (`src/ops/kernel/sampling_device.cuh`):** Replaces thread-local stack arrays with direct indexing into shared memory merge buffers, eliminating the 160-byte stack frame in `sample_row_kernel` and `speculative_accept_greedy_drafts_kernel`.
* **128-SM Wave Alignment (`src/ops/kernel/gqa_attention_geometry.cuh`):** Sets `DecodeSplits` to $64 \times \text{Scale}$ ($256\text{ CTAs} \implies 2.0\text{ full waves}$ across 128 SMs on AD102), eliminating the 34.4% idle tail wave of $S = 85$ ($340\text{ CTAs} \implies 2.65\text{ waves}$). Aligns deep-context key steps to 512 tokens.
* **Narrow-Tile Q5 Scheduling (`src/ops/linear/q5/q5_rowsplit_gemm_mma.cu`):** Sized intermediate MMA tiles to $R64 \times C32$ for $T \in [17, 64]$ to reduce padding waste on Ada Tensor Cores.

### 5. YaRN RoPE Long-Context Scaling
* **YaRN 1M Frequency Table (`src/targets/qwen3_6/impl/runtime/yarn.h`):** Implements YaRN 3-band inverse frequency interpolation and attention temperature scaling anchored at 1,048,576 tokens.
* **Dynamic Frequency Updates (`src/ops/launcher/rope.cu`):** Added `ops::set_text_rope_frequencies` to update device constant memory frequency tables dynamically at runtime initialization and reset upon teardown.
* **Dynamic Attention Scaling:** Threads dynamic `attn_scale` through `ExecutionCore` and `TextContext` to GQA kernels across prefill, decode, and speculative verification.

### 6. Serving, WebUI & Protocol Compatibility (`ninfer-serve`)
* **Embedded WebUI (`tools/ui/embed.cpp`, `cmake/ninfer_ui.cmake`):** Embeds WebUI static assets directly into the `ninfer-serve` binary, served on `GET /` with client-side SPA route fallbacks.
* **Runtime Endpoints:** Added `/props`, `/slots`, and Prometheus `/metrics` endpoints.
* **Live Token Timings:** Implemented per-token timings serialization in streaming chat completion chunks (`timings_per_token` / `return_progress`).
* **TCP Socket Optimization:** Enabled `TCP_NODELAY` on server sockets and added W3C SSE keep-alive comments (`: ping\n\n`) every 2 seconds during prompt prefill to prevent client proxy timeouts.
* **Reasoning Control & Template Preservation:** Added `--reasoning-effort <low|medium|xhigh>` CLI flag and updated `chat_template.cpp` to extract `<think>` blocks cleanly without duplicate tags.
* **Vision Budget Synchronization:** Bounds `image_max_pixels` and `video_max_pixels` to `vision_max_tokens` for automatic downscaling of high-resolution images.

---

## Measured Performance on NVIDIA GeForce RTX 4090

Evaluated on official Qwen3.8-27B (16.67 GiB groupwise `.ninfer` artifact, CUDA 13.3, 24 GB RTX 4090):

### Standard Product Benchmark Matrix (`ninfer_bench`)

| Test Case | Configuration | Throughput | Notes / Acceptance |
|---|---|---:|---|
| **Prefill (`pp2048`, `int8`)** | `pp2048`, Chunk 1024, `int8` | **`2,093.50 ± 0.68 tok/s`** | Saturated compute |
| **Prefill (`pp4096`, `int8`)** | `pp4096`, Chunk 1024, `int8` | **`2,079.51 ± 0.37 tok/s`** | Deep chunked prefill |
| **Prefill (`pp512`, `int8`)** | `pp512`, Chunk 1024, `int8` | **`1,738.07 ± 82.94 tok/s`** | Low-latency shallow prefill |
| **Prefill (`pp2048`, `rk4v4-e8`)** | `pp2048`, Chunk 1024, `rk4v4-e8` | **`2,085.58 ± 1.20 tok/s`** | Saturated compute |
| **Prefill (`pp4096`, `rk4v4-e8`)** | `pp4096`, Chunk 1024, `rk4v4-e8` | **`2,072.52 ± 0.76 tok/s`** | Deep chunked prefill |
| **Decode: MTP7 @ 2k (`pp2048+tg128`, `int8`)** | `pp2048+tg128`, `--greedy`, MTP7 | **`218.33 ± 0.91 tok/s`** | 88.0% draft acceptance ($7.11\text{ tok/round}$) |
| **Decode: MTP7 @ 2k (`pp2048+tg128`, `rk4v4-e8`)** | `pp2048+tg128`, `--greedy`, MTP7 | **`216.85 ± 1.11 tok/s`** | 88.0% draft acceptance ($7.11\text{ tok/round}$) |
| **Decode: MTP7 @ 32k (`pp32768+tg128`, `rk4v4-e8`)** | `pp32768+tg128`, `--greedy`, MTP7 | **`229.86 ± 0.09 tok/s`** | 100% draft acceptance ($8.00\text{ tok/round}$) |
| **Decode: MTP4 @ 0k (`tg128`, `rk4v4-e8`)** | `tg128`, `--greedy`, MTP4 | **`89.20 ± 3.45 tok/s`** | 30.7% draft acceptance (cold seed) |
| **Decode: Baseline (MTP0, `int8`)** | `tg128`, no speculation, CUDA Graph | **`52.77 ± 0.03 tok/s`** | Single-token base decode |
| **DirectStorage Cold Restore** | 77,615 tokens ($1.51\text{ GiB}$) | **`150 ms (10.1 GB/s)`** | Slashing TTFT from 52.59s to 1.86s |
