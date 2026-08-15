# NInfer-4090

NInfer-4090 is a specialized, high-performance C++20/CUDA inference engine for **Qwen3.8-27B** on a single 24 GB **NVIDIA GeForce RTX 4090** (`sm_89`).

The engine loads the official groupwise `.ninfer` artifact, serves OpenAI- and Anthropic-compatible HTTP APIs, and features native Ada Lovelace MMA tensor core execution, asynchronous double-buffered DMA memory staging, paged KV caching with 4-bit compression, compatible-prefix reuse, CUDA Graphs, and ReplaySSM linear attention state transactions.

---

## Measured Performance on NVIDIA GeForce RTX 4090

Tested on NVIDIA GeForce RTX 4090 (24 GB GDDR6X, 128 SMs, CUDA 13.3) using the official 16.96 GiB Qwen3.8-27B artifact.

### Standard Product Benchmark Matrix (`ninfer_bench`)

Evaluated over the standard `bench/fixtures/bench_corpus.ids` token corpus with INT8 group-64 KV cache and CUDA Graphs enabled:

| Test Case | Configuration | Throughput | Draft Acceptance / Notes |
|---|---|---:|---|
| **Prefill (`pp2048`)** | `pp2048`, Chunk 1024, INT8 KV | **`1,938.0 ± 2.7 tok/s`** | Standard 2k context prefill |
| **Prefill (`pp4096`)** | `pp4096`, Chunk 1024, INT8 KV | **`1,918.3 ± 2.1 tok/s`** | Peak saturated prefill |
| **Prefill (`pp512`)** | `pp512`, Chunk 1024, INT8 KV | **`1,695.4 ± 114.5 tok/s`** | Low-latency shallow prefill |
| **Decode: Code & Schemas (MTP4)** | `tg128`, `--greedy`, MTP4 + Draft Head | **`110.0 – 110.4 tok/s`** | 67–75% draft acceptance |
| **Decode: Code / Math (MTP3)** | `tg128`, `--greedy`, MTP3 + Draft Head | **`126.4 – 148.2 tok/s`** | 80–91% draft acceptance |
| **Decode: Bench Corpus (MTP3)** | `tg128`, MTP3 + Draft Head | **`77.0 – 106.5 tok/s`** | 30–49% draft acceptance |
| **Decode: Baseline (MTP0)** | `tg128`, no speculation, CUDA Graph | **`48.8 ± 4.3 tok/s`** | Single-token base decode |

---

## Architectural Features

* **Native `sm_89` Instruction Generation:** Built directly for Ada Lovelace without Ampere or Blackwell compatibility shims.
* **Ada 72 MB Persisting L2 Cache Pinning:** Pins MTP proposal head weights directly in the hardware L2 cache partition using stream access policy windows, eliminating DRAM access latency on speculative proposals.
* **Hierarchical N-Gram Context Speculation:** Zero-allocation $N=5 \to 4 \to 3 \to 2$ sequence pattern matcher that drafts continuation chains from prompt history and repeated tool calling schemas.
* **Double-Buffered `cp.async.cg` DMA:** Implements a 2-stage asynchronous circular buffer for codes, scales, and activations in `w8_small_t_mma`, overlapping global memory transfers with Tensor Core computation during decode rounds.
* **High-Priority CUDA Stream Queues:** Initializes primary compute streams with hardware-level priority to minimize Windows WDDM driver dispatch latency.
* **Full $T=48$ Exact MMA Tile Scheduling:** Extends single-pass matrix-vector tile execution up to 48 tokens without multi-pass slicing.
* **ReplaySSM Recurrent State Transactions:** Zero-copy linear attention state tracking for Qwen3.8 Gated DeltaNet (GDN) layers, guaranteeing numerical consistency across speculative verification and rollbacks.
* **Paged KV with `rk8v4` Compression:** Grouped 4-bit value quantization with normalized keys, enabling up to 170k tokens of resident context in 24 GB VRAM.

---

## Quick Start & Usage

### 1. HTTP Serving (`ninfer-serve`)

Serves OpenAI Chat Completions, Responses, and Anthropic Messages at `http://127.0.0.1:8080/v1`:

```powershell
.\build-sm89\apps\Release\ninfer-serve.exe "qwen3_8_27b.ninfer" --spec mtp --draft-tokens 3 --lm-head-draft --kv-dtype rk8v4 --max-context 160000
```

### 2. Interactive CLI (`ninfer`)

Direct single-request generation:

```powershell
.\build-sm89\apps\Release\ninfer.exe "qwen3_8_27b.ninfer" --prompt "Write an optimized C++ implementation of fast matrix multiplication." --spec mtp --draft-tokens 3 --lm-head-draft --greedy
```

### 3. Standard Benchmark (`ninfer_bench`)

Executes the standard product prefill and decode throughput benchmark matrix:

```powershell
.\build-sm89\bench\Release\ninfer_bench.exe --weights "qwen3_8_27b.ninfer" --corpus bench/fixtures/bench_corpus.ids --kv-dtype int8 --mtp-draft-tokens 3 --lm-head-draft -p 512,2048,4096 -n 128
```

---

## Build from Source

### Prerequisites
* Windows 11 / Windows Server or Linux
* NVIDIA CUDA Toolkit 12.8+ / 13.x
* Visual Studio 2022 (MSVC 19.4x) or GCC 13+
* CMake 3.28+

### Build Commands
```powershell
cmake -B build-sm89 -G "Visual Studio 17 2022" -A x64 -DCMAKE_CUDA_ARCHITECTURES=89 -DNINFER_BUILD_APPS=ON -DNINFER_BUILD_BENCHMARKS=ON
cmake --build build-sm89 --config Release --target ninfer ninfer-serve ninfer_bench --parallel 32
```

---

## License & Credits

* Apache License 2.0.
* Derived from [Neroued/ninfer](https://github.com/Neroued/ninfer) and [Don-Chad/ninfer-3090](https://github.com/Don-Chad/ninfer-3090).
* Specialized for native **sm_89** single-GPU execution on the **RTX 4090**.
