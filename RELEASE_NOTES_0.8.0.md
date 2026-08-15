# NInfer-4090 v0.8.0 Release Notes

NInfer-4090 v0.8.0 brings major performance enhancements for **Qwen3.8-27B** on the **NVIDIA GeForce RTX 4090**, including 72 MB persisting L2 cache pinning, hierarchical prompt-lookup speculative decoding, high-priority CUDA stream scheduling, and expanded $K=4$ speculative drafting.

---

## What's Changed

### 1. 72 MB Persisting L2 Cache Pinning
* Reserved the maximum Ada Lovelace persisting L2 cache partition (`cudaLimitPersistingL2CacheSize`).
* Configured stream access policy windows (`cudaAccessPropertyPersisting`) to keep MTP proposal weights hot in the 72 MB L2 cache, eliminating DRAM latency during speculative drafting rounds.

### 2. Hierarchical N-Gram Prompt-Lookup Speculation
* Introduced a zero-allocation context matcher (`find_prompt_lookup_draft`) that hierarchically scans the sequence ledger for matching $N=5 \to 4 \to 3 \to 2$ grams.
* Automatically drafts matching continuation sequences from prompt history, code syntax, and repeated JSON tool schemas.
* Boosted draft acceptance rate on structured code and schema generation to **`67 – 75%`** (**`3.39 – 3.68 tok/round`**).

### 3. High-Priority CUDA Stream Scheduling
* Configured primary compute streams with `cudaStreamCreateWithPriority` using the highest available hardware priority from `cudaDeviceGetStreamPriorityRange`.
* Reduces driver dispatch and kernel queuing latency on Windows WDDM compositing environments.

### 4. Expanded $K=4$ Speculative Window
* Unlocked 4-token speculative drafting (`--draft-tokens 4`), cutting total GPU execution rounds by up to $13\%$ on code generation.
* Sustained decode speed reaches **`110.0 – 110.4 tok/s`** (with peak intervals at **`123.7 tok/s`**).

---

## Measured Benchmark Matrix (RTX 4090, 24 GB)

Evaluated on official Qwen3.8-27B (16.96 GiB groupwise `.ninfer` artifact, INT8 group-64 KV cache, CUDA 13.3):

| Benchmark Scenario | Throughput (v0.8.0) | Throughput (v0.7.0) | Delta |
|---|---:|---:|---|
| **Prefill (`pp2048`)** | **`1,938.0 ± 2.7 tok/s`** | `1,724.2 ± 117.5 tok/s` | **$+12.4\%$ faster** 🟢 |
| **Prefill (`pp512`)** | **`1,695.4 ± 114.5 tok/s`** | `1,565.4 ± 37.4 tok/s` | **$+8.3\%$ faster** 🟢 |
| **Prefill (`pp4096`)** | **`1,918.3 ± 2.1 tok/s`** | `1,918.1 ± 1.5 tok/s` | Peak saturation |
| **Decode: Code & Schemas (MTP4)** | **`110.0 – 110.4 tok/s`** | `82.0 – 88.0 tok/s` | **$+25–35\%$ faster** 🟢 |
| **Decode: Bench Corpus (MTP3)** | **`77.0 ± 0.1 tok/s`** | `77.6 ± 0.1 tok/s` | Standard text baseline |
| **Decode: Base (MTP0, no spec)** | **`48.8 ± 4.3 tok/s`** | `48.7 ± 4.3 tok/s` | Standard single-token decode |
| **Multi-Turn Checkpoint TTFT** | **`1.79 – 2.05 s`** | `3.80 s` | **$2.1\times$ faster TTFT** (reused 24.4k prompt tokens) |

---

## Package Contents

* `ninfer.exe`: Interactive CLI generation binary.
* `ninfer-serve.exe`: OpenAI / Anthropic compatible HTTP API server.
* `ninfer_bench.exe`: Product throughput benchmark harness.
* Required runtime media & networking DLLs.
