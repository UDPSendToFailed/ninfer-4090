#include "core/device.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>

namespace ninfer {
namespace {

std::string cuda_error_message(const char* prefix, cudaError_t err) {
    return std::string(prefix) + ": " + cudaGetErrorName(err) + ": " + cudaGetErrorString(err);
}

void log_cuda_error(const char* op, cudaError_t err) noexcept {
    if (err != cudaSuccess) {
        std::fprintf(stderr, "CUDA cleanup failed during %s: %s: %s\n", op, cudaGetErrorName(err),
                     cudaGetErrorString(err));
    }
}

void destroy_stream(cudaStream_t& stream) noexcept {
    if (stream != nullptr) {
        log_cuda_error("cudaStreamDestroy", cudaStreamDestroy(stream));
        stream = nullptr;
    }
}

void destroy_event(cudaEvent_t& event) noexcept {
    if (event != nullptr) {
        log_cuda_error("cudaEventDestroy", cudaEventDestroy(event));
        event = nullptr;
    }
}

} // namespace

void cuda_check(cudaError_t err, const char* expr, const char* file, int line) {
    if (err == cudaSuccess) { return; }
    std::fprintf(stderr, "%s:%d: CUDA_CHECK(%s) failed: %s: %s\n", file, line, expr,
                 cudaGetErrorName(err), cudaGetErrorString(err));
    std::abort();
}

DeviceContext::DeviceContext(int device_id) : device(device_id) {
    int count       = 0;
    cudaError_t err = cudaGetDeviceCount(&count);
    if (err != cudaSuccess) {
        throw std::runtime_error(cuda_error_message("cudaGetDeviceCount failed", err));
    }
    if (count <= 0) { throw std::runtime_error("no CUDA devices available"); }
    if (device_id < 0 || device_id >= count) { throw std::runtime_error("invalid CUDA device id"); }

    err = cudaSetDevice(device_id);
    if (err != cudaSuccess) {
        throw std::runtime_error(cuda_error_message("cudaSetDevice failed", err));
    }

    err = cudaGetDeviceProperties(&props, device_id);
    if (err != cudaSuccess) {
        throw std::runtime_error(cuda_error_message("cudaGetDeviceProperties failed", err));
    }

    // Reserve an L2 persistence window on GPUs supporting persisting L2 cache
    // (CUDA C++ Programming Guide Section 6.2.3).
    if (props.persistingL2CacheMaxSize > 0) {
        const std::size_t persisting_size =
            std::min<std::size_t>(static_cast<std::size_t>(props.l2CacheSize * 0.6),
                                  static_cast<std::size_t>(props.persistingL2CacheMaxSize));
        if (persisting_size > 0) {
            cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, persisting_size);
        }
    }

    int lowest_priority  = 0;
    int highest_priority = 0;
    cudaDeviceGetStreamPriorityRange(&lowest_priority, &highest_priority);

    cudaStream_t compute = nullptr;
    cudaStream_t load    = nullptr;
    err = cudaStreamCreateWithPriority(&compute, cudaStreamNonBlocking, highest_priority);
    if (err != cudaSuccess) {
        err = cudaStreamCreateWithFlags(&compute, cudaStreamNonBlocking);
        if (err != cudaSuccess) {
            throw std::runtime_error(
                cuda_error_message("cudaStreamCreateWithPriority(stream) failed", err));
        }
    }

    err = cudaStreamCreateWithFlags(&load, cudaStreamNonBlocking);
    if (err != cudaSuccess) {
        destroy_stream(compute);
        throw std::runtime_error(
            cuda_error_message("cudaStreamCreateWithFlags(load_stream) failed", err));
    }

    stream      = compute;
    load_stream = load;
}

DeviceContext::~DeviceContext() {
    if (stream != nullptr || load_stream != nullptr) {
        log_cuda_error("cudaSetDevice", cudaSetDevice(device));
    }
    destroy_stream(load_stream);
    destroy_stream(stream);
}

DeviceContext::DeviceContext(DeviceContext&& other) noexcept
    : device(other.device), stream(other.stream), load_stream(other.load_stream),
      props(other.props) {
    other.stream      = nullptr;
    other.load_stream = nullptr;
}

DeviceContext& DeviceContext::operator=(DeviceContext&& other) noexcept {
    if (this == &other) { return *this; }

    if (stream != nullptr || load_stream != nullptr) {
        log_cuda_error("cudaSetDevice", cudaSetDevice(device));
    }
    destroy_stream(load_stream);
    destroy_stream(stream);

    device      = other.device;
    props       = other.props;
    stream      = other.stream;
    load_stream = other.load_stream;

    other.stream      = nullptr;
    other.load_stream = nullptr;
    return *this;
}

int DeviceContext::sm() const noexcept { return props.major * 10 + props.minor; }

std::size_t DeviceContext::total_vram() const noexcept { return props.totalGlobalMem; }

void DeviceContext::synchronize() const { CUDA_CHECK(cudaStreamSynchronize(stream)); }

void DeviceContext::set_persisting_l2_window(const void* ptr, std::size_t num_bytes,
                                             float hit_ratio) const {
    if (ptr == nullptr || num_bytes == 0 || stream == nullptr) { return; }
    if (props.persistingL2CacheMaxSize == 0) { return; }

    const std::size_t max_window = (props.accessPolicyMaxWindowSize > 0)
                                       ? static_cast<std::size_t>(props.accessPolicyMaxWindowSize)
                                       : static_cast<std::size_t>(props.persistingL2CacheMaxSize);
    const std::size_t window_bytes = std::min(max_window, num_bytes);
    if (window_bytes == 0) { return; }

    cudaStreamAttrValue attr{};
    attr.accessPolicyWindow.base_ptr  = const_cast<void*>(ptr);
    attr.accessPolicyWindow.num_bytes = window_bytes;
    attr.accessPolicyWindow.hitRatio  = std::clamp(hit_ratio, 0.0f, 1.0f);
    attr.accessPolicyWindow.hitProp   = cudaAccessPropertyPersisting;
    attr.accessPolicyWindow.missProp  = cudaAccessPropertyStreaming;

    const cudaError_t err =
        cudaStreamSetAttribute(stream, cudaStreamAttributeAccessPolicyWindow, &attr);
    if (err != cudaSuccess) {
        // If driver or device does not accept persisting window, log and continue without error
        log_cuda_error("cudaStreamSetAttribute(AccessPolicyWindow)", err);
    }
}

void DeviceContext::clear_persisting_l2_window() const {
    if (stream == nullptr || props.persistingL2CacheMaxSize == 0) { return; }

    cudaStreamAttrValue attr{};
    attr.accessPolicyWindow.base_ptr  = nullptr;
    attr.accessPolicyWindow.num_bytes = 0;
    attr.accessPolicyWindow.hitRatio  = 0.0f;
    attr.accessPolicyWindow.hitProp   = cudaAccessPropertyStreaming;
    attr.accessPolicyWindow.missProp  = cudaAccessPropertyStreaming;

    const cudaError_t err =
        cudaStreamSetAttribute(stream, cudaStreamAttributeAccessPolicyWindow, &attr);
    if (err != cudaSuccess) {
        log_cuda_error("cudaStreamSetAttribute(AccessPolicyWindow clear)", err);
    }
}

CudaEventTimer::CudaEventTimer(const DeviceContext& ctx) : stream_(ctx.stream) {
    cudaError_t err = cudaSetDevice(ctx.device);
    if (err != cudaSuccess) {
        throw std::runtime_error(cuda_error_message("cudaSetDevice(timer) failed", err));
    }

    cudaEvent_t start = nullptr;
    cudaEvent_t stop  = nullptr;
    err               = cudaEventCreate(&start);
    if (err != cudaSuccess) {
        throw std::runtime_error(cuda_error_message("cudaEventCreate(start) failed", err));
    }

    err = cudaEventCreate(&stop);
    if (err != cudaSuccess) {
        destroy_event(start);
        throw std::runtime_error(cuda_error_message("cudaEventCreate(stop) failed", err));
    }

    start_ = start;
    stop_  = stop;
}

CudaEventTimer::~CudaEventTimer() {
    destroy_event(stop_);
    destroy_event(start_);
}

CudaEventTimer::CudaEventTimer(CudaEventTimer&& other) noexcept
    : stream_(other.stream_), start_(other.start_), stop_(other.stop_) {
    other.stream_ = nullptr;
    other.start_  = nullptr;
    other.stop_   = nullptr;
}

CudaEventTimer& CudaEventTimer::operator=(CudaEventTimer&& other) noexcept {
    if (this == &other) { return *this; }

    destroy_event(stop_);
    destroy_event(start_);

    stream_ = other.stream_;
    start_  = other.start_;
    stop_   = other.stop_;

    other.stream_ = nullptr;
    other.start_  = nullptr;
    other.stop_   = nullptr;
    return *this;
}

void CudaEventTimer::start() { CUDA_CHECK(cudaEventRecord(start_, stream_)); }

void CudaEventTimer::record_stop() { CUDA_CHECK(cudaEventRecord(stop_, stream_)); }

float CudaEventTimer::elapsed_ms() const {
    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start_, stop_));
    return ms;
}

float CudaEventTimer::stop_ms() {
    record_stop();
    CUDA_CHECK(cudaEventSynchronize(stop_));
    return elapsed_ms();
}

} // namespace ninfer
