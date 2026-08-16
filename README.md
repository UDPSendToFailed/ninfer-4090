# NInfer-4090

NInfer-4090 is a specialized, high-performance C++20/CUDA inference engine for **Qwen3.8-27B** on a single 24 GB **NVIDIA GeForce RTX 4090** (`sm_89`).

The engine loads the official groupwise `.ninfer` artifact, serves OpenAI- and Anthropic-compatible HTTP APIs, and features native Ada Lovelace MMA tensor core execution, asynchronous double-buffered DMA memory staging, paged KV caching with 2-bit and 4-bit lattice/cylinder quantization, compatible-prefix reuse, CUDA Graphs, and ReplaySSM linear attention state transactions.

---

## Measured Performance on NVIDIA GeForce RTX 4090

Tested on NVIDIA GeForce RTX 4090 (24 GB GDDR6X, 128 SMs, CUDA 13.3) using the official 16.96 GiB Qwen3.8-27B artifact:

### Standard Product Benchmark Matrix (`ninfer_bench`)

| Test Case | Configuration | Throughput | Notes |
|---|---|---:|---|
| **Prefill (`pp2048`)** | `pp2048`, Chunk 1024, INT8 KV | **`1,938.0 ± 2.7 tok/s`** | Standard 2k context prefill |
| **Prefill (`pp4096`)** | `pp4096`, Chunk 1024, INT8 KV | **`1,918.3 ± 2.1 tok/s`** | Saturated prefill |
| **Prefill (`pp512`)** | `pp512`, Chunk 1024, INT8 KV | **`1,695.4 ± 114.5 tok/s`** | Low-latency shallow prefill |
| **Decode: Code & Schemas (MTP4)** | `tg128`, `--greedy`, MTP4 + Draft Head | **`110.0 – 110.4 tok/s`** | 67–75% draft acceptance |
| **Decode: Code / Math (MTP3)** | `tg128`, `--greedy`, MTP3 + Draft Head | **`126.4 – 148.2 tok/s`** | 80–91% draft acceptance |
| **Decode: Bench Corpus (MTP3)** | `tg128`, MTP3 + Draft Head | **`77.0 – 106.5 tok/s`** | 30–49% draft acceptance |
| **Decode: Baseline (MTP0)** | `tg128`, no speculation, CUDA Graph | **`48.8 ± 4.3 tok/s`** | Single-token base decode |

---

## KV Cache Quantization Modes & VRAM Context Limits

On a single 24 GB RTX 4090, model weights occupy ~15.9–16.7 GiB (depending on MTP draft heads and vision allocations), leaving ~7.0–7.5 GiB for resident KV cache. Choose `--kv-dtype` based on your context length and precision needs:

| Mode (`--kv-dtype`) | Key Format | Value Format | Bytes / Token | Verified 24 GB Max Context | Cosine Sim vs FP32 | Recommended Use Case |
|---|---|---|---|---:|---|---|
| **`rk2v4-e8`** | $E_8$ Cylinder 2-bit (240-root) | 4-bit per-channel | 100 B | **`~350k tokens`** | 96.2% | Ultra-long context, massive document RAG, needle retrieval |
| **`rk4v4-e8`** | $E_8$ Conway-Sloane 4-bit | 4-bit per-channel | 136 B | **`~200k tokens`** | 98.7% | High-fidelity coding, math, agentic workflows (MTP) |
| **`rk4v4`** | Hadamard Rotated 4-bit | 4-bit per-channel | 132 B | **`~200k tokens`** | 97.8% | Standard 4-bit compressed context |
| **`rk8v4`** | Hadamard Rotated 8-bit | 4-bit per-channel | 196 B | **`~130k tokens`** | 99.4% | Balanced daily chat and serving (MTP) |
| **`int8`** | INT8 per-channel | INT8 per-channel | 260 B | **`~100k tokens`** | 99.8% | Maximum numerical precision on shorter contexts |

*The GQA decode engine uses direct L1-cached block table lookups with a native context ceiling of **1,048,576 (1M) tokens**.*

---

## Use Cases & Example Serving Commands (`ninfer-serve`)

All serving commands expose OpenAI (`/v1/chat/completions`, `/v1/responses`) and Anthropic (`/v1/messages`) endpoints at `http://127.0.0.1:8080`.

### 1. Ultra-Long Context / Massive Document RAG (350k Tokens)
Uses 2-bit $E_8$ cylinder-factorized key quantization to fit 350,000 tokens in 24 GB VRAM. `--prefill-chunk 4096` ensures smooth chunked processing over massive prompts:
```powershell
.\build-sm89\apps\Release\ninfer-serve.exe "qwen3_8_27b.ninfer" --kv-dtype rk2v4-e8 --max-context 350000 --prefill-chunk 4096 --preserve-thinking
```

### 2. High-Precision Coding & Complex Reasoning (200k Tokens)
Uses 8D $E_8$ Conway-Sloane lattice quantization for high key fidelity (98.7% cosine similarity) paired with 4-token speculative MTP drafting for 110–140 tok/s decode:
```powershell
.\build-sm89\apps\Release\ninfer-serve.exe "qwen3_8_27b.ninfer" --kv-dtype rk4v4-e8 --spec mtp --draft-tokens 4 --lm-head-draft --max-context 200000 --preserve-thinking
```

### 3. General Daily Chat & Fast Speculative Serving (130k Tokens)
Rotated 8-bit keys and 4-bit values (99.4% cosine similarity) with 130k tokens capacity:
```powershell
.\build-sm89\apps\Release\ninfer-serve.exe "qwen3_8_27b.ninfer" --kv-dtype rk8v4 --spec mtp --draft-tokens 4 --lm-head-draft --max-context 130000
```

### 4. Vision & Multimodal Processing (Images & Video)
Enables media preprocessors and fixed vision GPU buffers for image/video understanding:
```powershell
.\build-sm89\apps\Release\ninfer-serve.exe "qwen3_8_27b.ninfer" --vision --kv-dtype rk4v4-e8 --max-context 150000
```

### 5. Maximum Precision / Shorter Context (< 100k Tokens)
Uncompressed INT8 key-value cache for strict numerical fidelity:
```powershell
.\build-sm89\apps\Release\ninfer-serve.exe "qwen3_8_27b.ninfer" --kv-dtype int8 --spec mtp --draft-tokens 4 --lm-head-draft --max-context 100000
```

---

## Interactive CLI (`ninfer`)

Direct single-prompt evaluation in the terminal:

```powershell
.\build-sm89\apps\Release\ninfer.exe "qwen3_8_27b.ninfer" --prompt "Write an optimized C++ CUDA kernel for warp-level reduction." --spec mtp --draft-tokens 4 --lm-head-draft --greedy
```

---

## Benchmarking (`ninfer_bench`)

Run the prefill and decode throughput benchmark over the standard token corpus:

```powershell
.\build-sm89\bench\Release\ninfer_bench.exe --weights "qwen3_8_27b.ninfer" --corpus bench/fixtures/bench_corpus.ids --kv-dtype rk4v4-e8 --mtp-draft-tokens 4 --lm-head-draft -p 512,2048,4096 -n 128
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
