#pragma once

#include <cuda_runtime.h>

#include <cstddef>

namespace ninfer {

void cuda_check(cudaError_t err, const char* expr, const char* file, int line);

#define CUDA_CHECK(expr) ::ninfer::cuda_check((expr), #expr, __FILE__, __LINE__)

// Streaming-multiprocessor count of the device this process runs on, queried once and cached.
// Launch geometry that deliberately fills exactly one resident wave reads the count from here
// instead of transcribing a per-part literal; the product runs one resident model on one device,
// so a single cached query is the whole device set.
int device_sm_count();

// Compile-time mirror of device_sm_count() for the architecture this build targets. __device__
// launch policies cannot query the runtime, and the host launcher that must reproduce such a
// policy exactly has to agree with it at compile time; those two sites use this constant, every
// other site uses device_sm_count().
#if defined(NINFER_SM89)
inline constexpr int kTargetSmCount = 128; // NVIDIA GeForce RTX 4090
#elif defined(NINFER_SM86)
inline constexpr int kTargetSmCount = 82; // NVIDIA GeForce RTX 3090
#else
#error "NInfer requires NINFER_SM86 or NINFER_SM89"
#endif

struct DeviceContext {
    int device               = 0;
    cudaStream_t stream      = nullptr;
    cudaStream_t load_stream = nullptr;
    cudaDeviceProp props{};

    explicit DeviceContext(int device_id = 0);
    ~DeviceContext();

    DeviceContext(const DeviceContext&)            = delete;
    DeviceContext& operator=(const DeviceContext&) = delete;
    DeviceContext(DeviceContext&& other) noexcept;
    DeviceContext& operator=(DeviceContext&& other) noexcept;

    int sm() const noexcept;
    std::size_t total_vram() const noexcept;
    void synchronize() const;
    void set_persisting_l2_window(const void* ptr, std::size_t num_bytes,
                                  float hit_ratio = 1.0f) const;
    void clear_persisting_l2_window() const;
};

class CudaEventTimer {
public:
    explicit CudaEventTimer(const DeviceContext& ctx);
    ~CudaEventTimer();

    CudaEventTimer(const CudaEventTimer&)            = delete;
    CudaEventTimer& operator=(const CudaEventTimer&) = delete;
    CudaEventTimer(CudaEventTimer&& other) noexcept;
    CudaEventTimer& operator=(CudaEventTimer&& other) noexcept;

    void start();
    void record_stop();
    [[nodiscard]] float elapsed_ms() const;
    float stop_ms();

private:
    cudaStream_t stream_ = nullptr;
    cudaEvent_t start_   = nullptr;
    cudaEvent_t stop_    = nullptr;
};

} // namespace ninfer
