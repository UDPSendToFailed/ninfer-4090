# NInfer-4090

NInfer-4090 is a specialized, high-performance C++20/CUDA inference engine for **Qwen3.8-27B** on a single 24 GB **NVIDIA GeForce RTX 4090** (`sm_89`).

The engine loads the official groupwise `.ninfer` artifact, serves OpenAI- and Anthropic-compatible HTTP APIs, and features native Ada Lovelace MMA tensor core execution, asynchronous double-buffered DMA memory staging, paged KV caching with 2-bit and 4-bit lattice/cylinder quantization, direct L1 block table lookups up to 1M tokens, D3D12 kernel residency management for Windows memory eviction, compatible-prefix reuse, CUDA Graphs, and ReplaySSM linear attention state transactions.

---

### Standard Product Benchmark Matrix (`ninfer_bench`)

Evaluated on official Qwen3.8-27B (16.96 GiB groupwise `.ninfer` artifact, CUDA 13.3, single 24 GB RTX 4090):

| Test Case | Configuration | Throughput | Notes / Acceptance |
|---|---|---:|---|
| **Prefill (`pp2048`)** | `pp2048`, Chunk 1024, INT8 KV | **`2,093.5 ± 0.7 tok/s`** | Saturated compute |
| **Prefill (`pp4096`)** | `pp4096`, Chunk 1024, INT8 KV | **`2,079.5 ± 0.4 tok/s`** | Deep chunked prefill |
| **Prefill (`pp512`)** | `pp512`, Chunk 1024, INT8 KV | **`1,738.1 ± 82.9 tok/s`** | Low-latency shallow prefill |
| **Decode: Deep Context MTP7 (`pp32768+tg128`)** | `pp32768+tg128`, `--greedy`, MTP7, `rk4v4-e8` | **`229.9 ± 0.1 tok/s`** | 100% draft acceptance (8.00 tok/round) |
| **Decode: Prompt-Cached MTP7 (`pp2048+tg128`)** | `pp2048+tg128`, `--greedy`, MTP7, INT8 KV | **`218.3 ± 0.9 tok/s`** | 88.0% draft acceptance (7.11 tok/round) |
| **Decode: Prompt-Cached MTP7 (`pp2048+tg128`)** | `pp2048+tg128`, `--greedy`, MTP7, `rk4v4-e8` | **`216.9 ± 1.1 tok/s`** | 88.0% draft acceptance (7.11 tok/round) |
| **Decode: Cold Bench Corpus (MTP4)** | `tg128`, `--greedy`, MTP4, `rk4v4-e8` | **`89.2 ± 3.5 tok/s`** | 30.7% acceptance on cold seed |
| **Decode: Baseline (MTP0)** | `tg128`, no speculation, INT8 KV, CUDA Graph | **`52.8 ± 0.0 tok/s`** | Single-token base autoregressive decode |
| **DirectStorage 1.3 Cold DMA Restore** | 77,615 prompt tokens ($1.51\text{ GiB}$) | **`150 ms (10.1 GB/s)`** | Drops cold TTFT from 52.6s to 1.86s |
| **360k Needle-in-a-Haystack** | 359,169 prompt tokens, `rk2v4-e8` | **`100% (5/5 Needles)`** | 666.7 tok/s avg prefill, exact recall |

---

## Verified Context Ceilings Matrix (RTX 4090, 24 GB)

The table below reflects the exact physical memory limits binary-searched on a 24 GB card under Windows WDDM residency management (rounded to the nearest thousand below).

> **Operating Recommendation:** For sustained maximum throughput, set `--max-context` roughly **20,000 to 30,000 tokens below** the physical ceiling shown in the table. This guarantees zero desktop memory contention and keeps all buffers resident in pure on-chip GDDR6X.

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

*Direct L1-cached GQA decode block table lookups support a native context envelope up to **1,048,576 (1M) tokens**.*

---

## Example Serving Commands (`ninfer-serve`)

All serving commands expose OpenAI (`/v1/chat/completions`, `/v1/responses`) and Anthropic (`/v1/messages`) endpoints at `http://127.0.0.1:8080`.

### 1. Ultra-Long Context RAG (500k Tokens, Pure Text)
Uses 2-bit $E_8$ cylinder key quantization to fit 500,000 resident tokens inside 24 GB VRAM:
```powershell
.\build-ninja\apps\ninfer-serve.exe "qwen3_8_27b.ninfer" --kv-dtype rk2v4-e8 --max-context 500000 --preserve-thinking
```

### 2. High-Precision Coding & Math with MTP4 (320k Tokens)
Uses 8D $E_8$ Conway-Sloane lattice quantization for 98.7% key fidelity paired with 4-token speculative drafting for 110–130 tok/s decode:
```powershell
.\build-ninja\apps\ninfer-serve.exe "qwen3_8_27b.ninfer" --kv-dtype rk4v4-e8 --spec mtp --draft-tokens 4 --lm-head-draft --max-context 320000 --preserve-thinking
```

### 3. Multimodal & Vision with 4-Token Speculation (280k Tokens)
Enables image/video processing with high-fidelity 4-bit lattice keys and MTP4 drafting:
```powershell
.\build-ninja\apps\ninfer-serve.exe "qwen3_8_27b.ninfer" --vision --kv-dtype rk4v4-e8 --spec mtp --draft-tokens 4 --lm-head-draft --max-context 280000 --preserve-thinking
```

### 4. Maximum Capacity Multimodal Serving (380k Tokens)
Combines 2-bit $E_8$ keys with image/video inputs and 4-token MTP speculation:
```powershell
.\build-ninja\apps\ninfer-serve.exe "qwen3_8_27b.ninfer" --vision --kv-dtype rk2v4-e8 --spec mtp --draft-tokens 4 --lm-head-draft --max-context 380000 --preserve-thinking
```

### 5. Maximum Precision Uncompressed INT8 (160k Tokens)
Standard INT8 per-channel KV cache with 4-token MTP speculative decoding:
```powershell
.\build-ninja\apps\ninfer-serve.exe "qwen3_8_27b.ninfer" --kv-dtype int8 --spec mtp --draft-tokens 4 --lm-head-draft --max-context 160000 --preserve-thinking
```

---

## Interactive CLI (`ninfer`)

Direct single-prompt evaluation in the terminal:

```powershell
.\build-ninja\apps\ninfer.exe "qwen3_8_27b.ninfer" --prompt "Write an optimized C++ CUDA kernel for warp-level reduction." --spec mtp --draft-tokens 4 --lm-head-draft --greedy
```

---

## Benchmarking (`ninfer_bench`)

Run the prefill and decode throughput benchmark over the standard token corpus:

```powershell
.\build-ninja\bench\ninfer_bench.exe --weights "qwen3_8_27b.ninfer" --corpus bench/fixtures/bench_corpus.ids --kv-dtype rk4v4-e8 --mtp-draft-tokens 4 --lm-head-draft -p 512,2048,4096 -n 128
```

---


## Build from Source

### Prerequisites
* **OS:** Windows 11
* **CUDA Toolkit:** CUDA 13.3
* **Compiler:** Visual Studio 2022
* **Build Tools:** CMake 4.4.2 and Ninja

---

### Build & Test (Ninja)

Open PowerShell and initialize the MSVC x64 developer environment to configure and build:

#### 1. Configure and Build Full Suite
```powershell
cmd /c "call ""C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"" && set NVCC_PREPEND_FLAGS=--split-compile=0 && cmake -B build-ninja -G Ninja -DNINFER_BUILD_BENCHMARKS=ON && ninja -C build-ninja -j 32"
```

#### 2. Run Full Test Suite
```powershell
cmd /c "call ""C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"" && cd build-ninja && ctest --output-on-failure -j 8"
```

#### 3. Build Apps Only
```powershell
cmd /c "call ""C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"" && ninja -C build-ninja apps/ninfer.exe apps/ninfer-serve.exe bench/ninfer_bench.exe -j 32"
```

---

## Disclaimer

This is a fork of NInfer I am developing for fun to push the limits of the speed and context window for Qwen 3.8 27B on the RTX 4090. Things might break or regress with updates, I offer no guarantees, use this at your own risk.

Co-developed with Gemini 3.7 Flash.

## License & Credits

* Apache License 2.0.
* Derived from [Neroued/ninfer](https://github.com/Neroued/ninfer) and [Don-Chad/ninfer-3090](https://github.com/Don-Chad/ninfer-3090).
* Specialized for native **sm_89** single-GPU execution on the **RTX 4090**.
