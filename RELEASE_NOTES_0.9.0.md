# NInfer-4090 v0.9.0 Release Notes

NInfer-4090 v0.9.0 brings **$E_8$ lattice and cylinder KV cache quantization**, expanding resident context on a single 24 GB RTX 4090 up to **350,000+ tokens**, alongside direct L1 block table lookups with a **1M token context ceiling**, and warp-cooperative CUDA optimizations.

---

## What's Changed

### 1. $E_8$ Lattice & Cylinder Factorized KV Quantization
* **`rk4v4-e8` (4-bit $E_8$ Lattice Keys + 4-bit Values, 136 B/tok):**
  * Evaluates nearest lattice points on the 8D Gosset $E_8$ lattice: $\Lambda = D_8 \cup (D_8 + \frac{1}{2}\mathbf{1})$ using register shuffles and Conway-Sloane algebraic projections.
  * Yields **`98.67%`** cosine similarity vs FP32 and achieves **`100%`** needle-in-a-haystack retrieval across 1M tokens.
* **`rk2v4-e8` (2-bit $E_8$ Cylinder Keys + 4-bit Values, 100 B/tok):**
  * Factorizes 8D key subvectors onto the unit sphere $S^7$ and radius $\mathbb{R}^+$.
  * Encodes the primary direction via an 8-bit index over the 240 minimal $E_8$ root vectors, paired with a 4-bit log-scale subspace radius and a 4-bit residual hyperoctahedral axis.
  * Drops the total KV cache footprint to **100 bytes/token**, allowing **350,000+ tokens** to fit inside 24 GB VRAM on a single GPU.

### 2. Rotated 4-Bit KV Cache (`rk4v4`)
* **`rk4v4` (4-bit Keys + 4-bit Values, 132 B/tok):**
  * Applies randomized Hadamard rotations to smooth out activation outliers before uniform 4-bit scalar quantization.
  * Fits ~200k resident context in 24 GB VRAM with full speculative MTP acceleration.

### 3. Direct L1 Block Table Lookups & 1M Context Envelope
* Replaced the static shared memory block table array in GQA decode kernels with direct L1-cached global lookups.
* Lifted `kNativeContext` and attention visible key ceilings from **262,144 (256k)** to **1,048,576 (1,024k / 1M tokens)**.

### 4. Warp-Cooperative Quantization & CUDA Prefill Acceleration
* **Warp-Cooperative 8-Lane Reductions:** Replaced single-thread serial loops with 3-step butterfly shuffles (`__shfl_xor_sync`) to find the optimal $(i, j)$ root pair in ~6 clock cycles with 100% active lane occupancy ($2.7\times$ faster encoding).
* **L1 Cache Table Lookups:** Moved the 240-root codebook table from `__constant__` to `__device__ const` accessed via `__ldg()`, removing up to 32-way constant memory serialization during attention loops.
* **Aligned 32-bit Vectorized Loads:** Replaced scalar byte reads with `uint32_t` loads in `issue_kv_tile` across prefill and decode kernels.

### 5. Unified Symmetric Naming
* Standardized naming across CLI flags, server options, runtime layouts, and benchmarks:
  * `--kv-dtype int8` (260 B/tok)
  * `--kv-dtype rk8v4` (196 B/tok)
  * `--kv-dtype rk4v4` (132 B/tok)
  * `--kv-dtype rk4v4-e8` (136 B/tok)
  * `--kv-dtype rk2v4-e8` (100 B/tok)

---

## Measured Context Capacities on RTX 4090 (24 GB)

| Mode (`--kv-dtype`) | Key Precision | Value Precision | Bytes / Token | Verified Max Context | Cosine Similarity |
|---|---|---|---|---:|---|
| **`rk2v4-e8`** | $E_8$ Cylinder 2.0-bit | 4-bit | 100 B | **`~350,000 tokens`** | 96.2% |
| **`rk4v4-e8`** | $E_8$ Lattice 4.0-bit | 4-bit | 136 B | **`~200,000 tokens`** | 98.7% (MTP enabled) |
| **`rk4v4`** | Hadamard Rotated 4.0-bit | 4-bit | 132 B | **`~200,000 tokens`** | 97.8% (MTP enabled) |
| **`rk8v4`** | Hadamard Rotated 8.0-bit | 4-bit | 196 B | **`~130,000 tokens`** | 99.4% (MTP enabled) |
| **`int8`** | INT8 per-channel | INT8 per-channel | 260 B | **`~100,000 tokens`** | 99.8% (MTP enabled) |

---

## Package Contents

* `ninfer.exe`: Interactive CLI generation binary.
* `ninfer-serve.exe`: OpenAI / Anthropic compatible HTTP API server.
* `ninfer_bench.exe`: Product throughput benchmark harness.
* Required runtime media & networking DLLs.
