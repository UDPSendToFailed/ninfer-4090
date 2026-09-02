#include "ops/linear/q5/q5_launch.h"

#include "core/device.h"
#include "ops/linear/q5/q5_small_t_mma.cuh"

#include <array>
#include <cstddef>
#include <stdexcept>
#include <utility>

namespace ninfer::ops::detail {
namespace {

using Geom1024x5120  = Q5SmallTGeometry<1024, 5120>;
using Geom6144x5120  = Q5SmallTGeometry<6144, 5120>;
using Geom7168x5120  = Q5SmallTGeometry<7168, 5120>;
using Geom5120x6144  = Q5SmallTGeometry<5120, 6144>;
using Geom5120x17408 = Q5SmallTGeometry<5120, 17408>;

template <class Geometry, int TileTokens, int ActiveTokens>
void launch_exact(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    using Schedule = Q5SmallTSchedule;
    constexpr int kBlocks =
        (Geometry::kOutputRows + Schedule::kRowsPerCta - 1) / Schedule::kRowsPerCta;
    const auto in_ld  = static_cast<std::int32_t>(x.nb[1] / sizeof(__nv_bfloat16));
    const auto out_ld = static_cast<std::int32_t>(out.nb[1] / sizeof(__nv_bfloat16));

    q5_small_t_mma_kernel<Geometry, TileTokens, ActiveTokens>
        <<<kBlocks, Schedule::kThreads, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(x.data),
            static_cast<const std::uint8_t*>(weight.qdata),
            static_cast<const std::uint8_t*>(weight.qhigh),
            static_cast<const std::uint8_t*>(weight.scales),
            static_cast<__nv_bfloat16*>(out.data), in_ld, out_ld);
    CUDA_CHECK(cudaGetLastError());
}

template <class Geometry, std::size_t... Offsets>
constexpr auto make_launchers(std::index_sequence<Offsets...>) {
    return std::array<Q5Launch, sizeof...(Offsets)>{
        &launch_exact<Geometry,
                      ((1 + static_cast<int>(Offsets)) <= 8 ? 8 : 16),
                      1 + static_cast<int>(Offsets)>...};
}

constexpr auto kLaunchers1024x5120 =
    make_launchers<Geom1024x5120>(std::make_index_sequence<16>{});
constexpr auto kLaunchers6144x5120 =
    make_launchers<Geom6144x5120>(std::make_index_sequence<16>{});
constexpr auto kLaunchers7168x5120 =
    make_launchers<Geom7168x5120>(std::make_index_sequence<16>{});
constexpr auto kLaunchers5120x6144 =
    make_launchers<Geom5120x6144>(std::make_index_sequence<16>{});
constexpr auto kLaunchers5120x17408 =
    make_launchers<Geom5120x17408>(std::make_index_sequence<16>{});

template <class Geometry, int TileTokens, int ActiveTokens>
void launch_residual_exact(const Tensor& x, const Weight& weight, Tensor& out,
                           cudaStream_t stream) {
    using Schedule = Q5SmallTSchedule;
    constexpr int kBlocks =
        (Geometry::kOutputRows + Schedule::kRowsPerCta - 1) / Schedule::kRowsPerCta;
    const auto in_ld  = static_cast<std::int32_t>(x.nb[1] / sizeof(__nv_bfloat16));
    const auto out_ld = static_cast<std::int32_t>(out.nb[1] / sizeof(__nv_bfloat16));
    constexpr int kSplits = 2;
    const dim3 grid(static_cast<unsigned>(kBlocks), static_cast<unsigned>(kSplits), 1u);
    const Q5SmallTMmaResidualAtomicEpilogue epilogue{};

    q5_small_t_mma_kernel<Geometry, TileTokens, ActiveTokens, Q5SmallTMmaResidualAtomicEpilogue,
                          Q5SmallTMmaIdentityRows, kSplits>
        <<<grid, Schedule::kThreads, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(x.data),
            static_cast<const std::uint8_t*>(weight.qdata),
            static_cast<const std::uint8_t*>(weight.qhigh),
            static_cast<const std::uint8_t*>(weight.scales),
            static_cast<__nv_bfloat16*>(out.data), in_ld, out_ld, epilogue);
    CUDA_CHECK(cudaGetLastError());
}

template <class Geometry, std::size_t... Offsets>
constexpr auto make_residual_launchers(std::index_sequence<Offsets...>) {
    return std::array<Q5Launch, sizeof...(Offsets)>{
        &launch_residual_exact<Geometry,
                               ((1 + static_cast<int>(Offsets)) <= 8 ? 8 : 16),
                               1 + static_cast<int>(Offsets)>...};
}

constexpr auto kResidualLaunchers5120x6144 =
    make_residual_launchers<Geom5120x6144>(std::make_index_sequence<16>{});
constexpr auto kResidualLaunchers5120x17408 =
    make_residual_launchers<Geom5120x17408>(std::make_index_sequence<16>{});

} // namespace

void launch_q5_small_t_mma(const Tensor& x, const Weight& weight, Tensor& out,
                           cudaStream_t stream) {
    const int t = x.ne[1];
    if (t < 1 || t > 16) {
        throw std::invalid_argument("q5 small-t mma: unsupported token count");
    }

    const std::size_t idx = static_cast<std::size_t>(t - 1);

    if (weight.n == 1024 && weight.k == 5120) {
        kLaunchers1024x5120[idx](x, weight, out, stream);
        return;
    }
    if (weight.n == 6144 && weight.k == 5120) {
        kLaunchers6144x5120[idx](x, weight, out, stream);
        return;
    }
    if (weight.n == 7168 && weight.k == 5120) {
        kLaunchers7168x5120[idx](x, weight, out, stream);
        return;
    }
    if (weight.n == 5120 && weight.k == 6144) {
        kLaunchers5120x6144[idx](x, weight, out, stream);
        return;
    }
    if (weight.n == 5120 && weight.k == 17408) {
        kLaunchers5120x17408[idx](x, weight, out, stream);
        return;
    }

    throw std::invalid_argument("q5 small-t mma: unsupported shape");
}

void launch_q5_linear_add_small_t_mma(const Tensor& x, const Weight& weight, Tensor& residual_out,
                                      cudaStream_t stream) {
    const int t = x.ne[1];
    if (t < 1 || t > 16) {
        throw std::invalid_argument("q5 small-t mma linear_add: unsupported token count");
    }

    const std::size_t idx = static_cast<std::size_t>(t - 1);

    if (weight.n == 5120 && weight.k == 6144) {
        kResidualLaunchers5120x6144[idx](x, weight, residual_out, stream);
        return;
    }
    if (weight.n == 5120 && weight.k == 17408) {
        kResidualLaunchers5120x17408[idx](x, weight, residual_out, stream);
        return;
    }

    throw std::invalid_argument("q5 small-t mma linear_add: unsupported shape");
}

} // namespace ninfer::ops::detail
