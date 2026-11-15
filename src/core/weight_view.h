#pragma once

#include "core/tensor.h"
#include "core/weight.h"

#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <span>
#include <vector>

namespace ninfer {

// Geometry describes one complete encoded parent. Regions address its logical elements;
// padding and scale words do not belong to that element domain.
struct WeightGeometry {
    QType format       = QType::BF16_CTRL;
    QuantLayout layout = QuantLayout::Contiguous;
    std::vector<std::uint64_t> shape;
    std::uint64_t elements            = 0;
    std::uint64_t bytes               = 0;
    std::uint64_t alignment           = 256;
    std::uint64_t padded_columns      = 0;
    std::uint64_t group_size          = 0;
    std::uint64_t code_bytes_per_row  = 0;
    std::uint64_t high_bytes_per_row  = 0;
    std::uint64_t scale_bytes_per_row = 0;
    std::uint64_t code_bytes          = 0;
    std::uint64_t high_offset         = 0;
    std::uint64_t high_bytes          = 0;
    std::uint64_t scale_offset        = 0;
    std::uint64_t scale_bytes         = 0;
};

[[nodiscard]] WeightGeometry weight_geometry(QType format, QuantLayout layout,
                                             std::span<const std::uint64_t> shape);
[[nodiscard]] std::uint64_t weight_element_count(std::span<const std::uint64_t> shape);

struct WeightParent {
    WeightGeometry geometry;
    const std::byte* data = nullptr;
};

struct WeightRegion {
    const WeightParent* parent = nullptr;
    std::uint64_t begin        = 0;
    std::uint64_t end          = 0;
};

struct WeightView {
    std::vector<std::uint64_t> shape;
    std::vector<WeightRegion> parts;
};

// Coalesce adjacent logical regions without moving their bytes. Fails for a gather or
// multiple parents; those remain explicit inputs for a consumer supporting that form.
[[nodiscard]] WeightRegion contiguous_weight_region(const WeightView& view);
[[nodiscard]] bool is_complete_weight(const WeightView& view);
[[nodiscard]] std::uint64_t weight_scale_offset(const WeightGeometry& geometry, std::uint64_t row,
                                                std::uint64_t group);

struct WeightRowPlanes {
    const std::byte* codes        = nullptr;
    const std::byte* high         = nullptr;
    const std::byte* scales       = nullptr;
    std::uint64_t row_begin       = 0;
    std::uint64_t row_count       = 0;
    std::uint64_t code_row_bytes  = 0;
    std::uint64_t high_row_bytes  = 0;
    std::uint64_t scale_row_bytes = 0;
};

[[nodiscard]] WeightRowPlanes weight_row_planes(const WeightRegion& region);
[[nodiscard]] Tensor weight_tensor(const WeightView& view,
                                   std::initializer_list<std::int32_t> internal_shape);
// Existing Weight ABI: complete quantized parents, direct regions, and RowSplit row views.
[[nodiscard]] Weight native_weight(const WeightView& view);

// Concatenate multiple row-wise WeightViews along axis 0 if they form a contiguous region
// on the same parent. Returns an empty optional if they are not contiguous on one parent.
[[nodiscard]] bool try_coalesce_rows(std::span<const WeightView* const> views, WeightView& out);

} // namespace ninfer
