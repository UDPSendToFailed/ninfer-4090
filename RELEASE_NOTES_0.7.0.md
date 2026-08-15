# NInfer-4090 v0.7.0 Release Notes

NInfer-4090 v0.7.0 delivers native **sm_89 (Ada Lovelace)** specialization for the **NVIDIA GeForce RTX 4090**, targeting first-class performance on **Qwen3.8-27B**.

---

## Key Highlights & Performance

### 1. Native `sm_89` Specialization & Asynchronous DMA
* **Hardware-Native SASS Generation:** Directly compiled for Ada Lovelace (`-arch=sm_89`) without Ampere or Blackwell compatibility shims.
* **Double-Buffered `cp.async.cg` DMA in W8 Small-T MMA:** Overlaps global VRAM weight and scale transfers with Tensor Core computation using hardware-level circular DMA buffering, completely hiding memory latency during decode.
* **Full $T=48$ Exact Tile Scheduling:** Sized register budgets and tile schedules specifically for Ada's 128 SMs, eliminating multi-pass column slicing.

### 2. Speculative Decoding & ReplaySSM State Transactions
* **Linear MTP3 Speculative Decoding:** Native 3-step proposal head with draft verification, delivering **`126.4 – 148.2 tok/s`** on code/math tasks and **`65 – 98 tok/s`** on deep multi-turn chat.
* **ReplaySSM Recurrent State Integrity:** Zero-copy linear attention state tracking for Qwen3.8 Gated DeltaNet (GDN) layers, enabling instant sub-second prefix cache reuse (**`949 ms` TTFT** across turns).

### 3. Paged Deep-Context KV Compression (`rk8v4`)
* **Extreme Context Scaling:** Supports up to **140k–170k resident tokens** in 24 GB VRAM alongside MTP3 speculative decoding and CUDA Graphs.
* Verified sustained prefill at **`1,179 tok/s`** and decode at **`92.8 tok/s`** at 134,000 tokens depth.

---

## Measured Benchmark Matrix (RTX 4090, 24 GB)

| Benchmark Scenario | Throughput | Draft Acceptance / Notes |
|---|---:|---|
| **Prefill (`pp4096`)** | **`1,918.1 ± 1.5 tok/s`** | Peak saturated prefill |
| **Prefill (`pp2048`)** | **`1,724.2 ± 117.5 tok/s`** | Standard 2k prefill |
| **Prefill (`pp512`)** | **`1,565.4 ± 37.4 tok/s`** | Low-latency shallow prefill |
| **Decode: Code / Math (MTP3)** | **`126.4 – 148.2 tok/s`** | 80–91% draft acceptance |
| **Decode: Bench Corpus (MTP3)** | **`77.6 – 106.5 tok/s`** | 30–49% draft acceptance |
| **Decode: Baseline (MTP0)** | **`48.7 ± 4.3 tok/s`** | Single-token base decode |
| **Turn Checkpoint Restoration** | **`949 ms` TTFT** | Reused 19.1k prefix tokens |

---

## Package Contents

* `ninfer.exe`: Interactive CLI generation binary.
* `ninfer-serve.exe`: OpenAI / Anthropic compatible HTTP API server.
* `ninfer_bench.exe`: Product throughput benchmark harness.
* Required runtime media & networking DLLs.
