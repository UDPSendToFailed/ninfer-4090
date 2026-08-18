#include "core/linear_attention_state.h"

#include "core/device.h"

#include <limits>
#include <stdexcept>
#include <string>

namespace ninfer {
namespace {

constexpr std::size_t kArenaAlign = 256;

void validate_positive(std::int32_t value, const char* message) {
    if (value <= 0) { throw std::invalid_argument(message); }
}

void validate_layer_slot(const LinearAttentionStatePool& pool, std::uint32_t layer,
                         std::int32_t slot, const char* label) {
    if (layer >= pool.layer_count()) {
        throw std::out_of_range(std::string(label) + " layer out of range");
    }
    if (slot < 0 || slot >= pool.slot_count()) {
        throw std::out_of_range(std::string(label) + " slot out of range");
    }
}

void validate_state_tensor(const Tensor& tensor, DType dtype,
                           std::initializer_list<std::int32_t> shape, const char* label) {
    const Tensor expected(nullptr, dtype, shape);
    if (tensor.data == nullptr || tensor.dtype != dtype || !tensor.is_contiguous() ||
        tensor.bytes() != expected.bytes()) {
        throw std::logic_error(std::string("LinearAttentionStatePool ") + label +
                               " tensor is inconsistent");
    }
    for (int dim = 0; dim < 4; ++dim) {
        if (tensor.ne[dim] != expected.ne[dim]) {
            throw std::logic_error(std::string("LinearAttentionStatePool ") + label +
                                   " shape is inconsistent");
        }
    }
}

std::int64_t layer_stride_bytes(const std::vector<Tensor>& tensors, const char* label) {
    if (tensors.empty()) {
        throw std::logic_error(std::string("LinearAttentionStatePool has no ") + label + " layers");
    }
    if (tensors.size() == 1) { return 0; }

    const auto first  = reinterpret_cast<std::uintptr_t>(tensors[0].data);
    const auto second = reinterpret_cast<std::uintptr_t>(tensors[1].data);
    if (second <= first ||
        second - first > static_cast<std::uintptr_t>(std::numeric_limits<std::int64_t>::max())) {
        throw std::logic_error(std::string("LinearAttentionStatePool ") + label +
                               " layer stride is invalid");
    }
    const auto stride = static_cast<std::int64_t>(second - first);
    if (static_cast<std::uint64_t>(stride) < tensors.front().bytes()) {
        throw std::logic_error(std::string("LinearAttentionStatePool ") + label +
                               " layers overlap");
    }
    for (std::size_t layer = 2; layer < tensors.size(); ++layer) {
        const auto previous = reinterpret_cast<std::uintptr_t>(tensors[layer - 1].data);
        const auto current  = reinterpret_cast<std::uintptr_t>(tensors[layer].data);
        if (current <= previous || current - previous != static_cast<std::uintptr_t>(stride)) {
            throw std::logic_error(std::string("LinearAttentionStatePool ") + label +
                                   " layer stride is not constant");
        }
    }
    return stride;
}

} // namespace

LinearAttentionStatePoolLayout
plan_linear_attention_state_pool(LayoutBuilder& builder, const LinearAttentionStatePoolSpec& spec) {
    if (spec.layers == 0) {
        throw std::invalid_argument("LinearAttentionStatePool layers must be nonzero");
    }
    if (spec.layers > static_cast<std::uint32_t>(std::numeric_limits<std::int32_t>::max())) {
        throw std::overflow_error("LinearAttentionStatePool layer count exceeds int32");
    }
    validate_positive(spec.conv_channels,
                      "LinearAttentionStatePool conv_channels must be positive");
    validate_positive(spec.conv_width, "LinearAttentionStatePool conv_width must be positive");
    validate_positive(spec.value_heads, "LinearAttentionStatePool value_heads must be positive");
    validate_positive(spec.value_head_dim,
                      "LinearAttentionStatePool value_head_dim must be positive");
    validate_positive(spec.key_head_dim, "LinearAttentionStatePool key_head_dim must be positive");
    validate_positive(spec.slot_count, "LinearAttentionStatePool slot_count must be positive");
    if (spec.conv_dtype != DType::BF16 && spec.conv_dtype != DType::FP32) {
        throw std::invalid_argument("LinearAttentionStatePool conv_dtype must be BF16 or FP32");
    }

    const Tensor conv_shape(nullptr, spec.conv_dtype,
                            {spec.conv_channels, spec.conv_width, spec.slot_count});
    const Tensor recurrent_shape(
        nullptr, DType::FP32,
        {spec.key_head_dim, spec.value_head_dim, spec.value_heads, spec.slot_count});

    LinearAttentionStatePoolLayout layout;
    layout.spec = spec;
    layout.conv.reserve(spec.layers);
    layout.recurrent.reserve(spec.layers);
    for (std::uint32_t layer = 0; layer < spec.layers; ++layer) {
        const std::string prefix = "Linear Attention layer " + std::to_string(layer);
        layout.conv.push_back(builder.add(conv_shape.bytes(), kArenaAlign, prefix + " conv"));
        layout.recurrent.push_back(
            builder.add(recurrent_shape.bytes(), kArenaAlign, prefix + " recurrent"));
    }
    return layout;
}

LinearAttentionStatePool::LinearAttentionStatePool(DeviceSpan backing,
                                                   const LinearAttentionStatePoolLayout& layout)
    : spec(layout.spec) {
    if (layout.conv.empty() || layout.recurrent.size() != layout.conv.size() ||
        layout.conv.size() != spec.layers) {
        throw std::invalid_argument(
            "LinearAttentionStatePool layout layer counts are inconsistent");
    }

    const Tensor conv_shape(nullptr, spec.conv_dtype,
                            {spec.conv_channels, spec.conv_width, spec.slot_count});
    const Tensor recurrent_shape(
        nullptr, DType::FP32,
        {spec.key_head_dim, spec.value_head_dim, spec.value_heads, spec.slot_count});
    conv.reserve(layout.conv.size());
    recurrent.reserve(layout.recurrent.size());
    for (std::size_t layer = 0; layer < layout.conv.size(); ++layer) {
        if (layout.conv[layer].bytes != conv_shape.bytes() ||
            layout.recurrent[layer].bytes != recurrent_shape.bytes()) {
            throw std::logic_error(
                "LinearAttentionStatePool layout tensor byte size is inconsistent");
        }
        conv.emplace_back(layout.conv[layer].bind(backing).data, spec.conv_dtype,
                          std::initializer_list<std::int32_t>{spec.conv_channels, spec.conv_width,
                                                              spec.slot_count});
        recurrent.emplace_back(
            layout.recurrent[layer].bind(backing).data, DType::FP32,
            std::initializer_list<std::int32_t>{spec.key_head_dim, spec.value_head_dim,
                                                spec.value_heads, spec.slot_count});
    }
}

std::uint32_t LinearAttentionStatePool::layer_count() const noexcept {
    return static_cast<std::uint32_t>(conv.size());
}

std::int32_t LinearAttentionStatePool::slot_count() const noexcept { return spec.slot_count; }

std::int64_t LinearAttentionStatePool::conv_slot_stride_elements() const noexcept {
    return static_cast<std::int64_t>(spec.conv_channels) *
           static_cast<std::int64_t>(spec.conv_width);
}

std::int64_t LinearAttentionStatePool::recurrent_slot_stride_elements() const noexcept {
    return static_cast<std::int64_t>(spec.key_head_dim) *
           static_cast<std::int64_t>(spec.value_head_dim) *
           static_cast<std::int64_t>(spec.value_heads);
}

LinearAttentionStateAllLayersView LinearAttentionStatePool::all_layers_view() const {
    if (conv.size() != spec.layers || recurrent.size() != spec.layers || conv.empty()) {
        throw std::logic_error("LinearAttentionStatePool layer inventory is inconsistent");
    }
    for (std::size_t layer = 0; layer < conv.size(); ++layer) {
        validate_state_tensor(conv[layer], spec.conv_dtype,
                              {spec.conv_channels, spec.conv_width, spec.slot_count}, "conv");
        validate_state_tensor(
            recurrent[layer], DType::FP32,
            {spec.key_head_dim, spec.value_head_dim, spec.value_heads, spec.slot_count},
            "recurrent");
    }
    return LinearAttentionStateAllLayersView{
        .conv_layer0                  = conv.front(),
        .recurrent_layer0             = recurrent.front(),
        .conv_layer_stride_bytes      = layer_stride_bytes(conv, "conv"),
        .recurrent_layer_stride_bytes = layer_stride_bytes(recurrent, "recurrent"),
        .spec                         = spec,
    };
}

Tensor LinearAttentionStatePool::conv_slot(std::uint32_t layer, std::int32_t slot) const {
    validate_layer_slot(*this, layer, slot, "LinearAttentionStatePool conv_slot");
    return conv.at(layer).slice(2, slot, 1).view({spec.conv_channels, spec.conv_width});
}

Tensor LinearAttentionStatePool::recurrent_slot(std::uint32_t layer, std::int32_t slot) const {
    validate_layer_slot(*this, layer, slot, "LinearAttentionStatePool recurrent_slot");
    return recurrent.at(layer)
        .slice(3, slot, 1)
        .view({spec.key_head_dim, spec.value_head_dim, spec.value_heads});
}

void LinearAttentionStatePool::copy_slot(std::int32_t src, std::int32_t dst, cudaStream_t stream) {
    validate_layer_slot(*this, 0, src, "LinearAttentionStatePool copy_slot source");
    validate_layer_slot(*this, 0, dst, "LinearAttentionStatePool copy_slot destination");
    if (src == dst) { return; }
    const std::uint32_t layers = layer_count();
    if (layers == 0) { return; }

    const std::size_t conv_bytes = conv_slot(0, src).bytes();
    const std::size_t rec_bytes  = recurrent_slot(0, src).bytes();

    if (layers == 1) {
        CUDA_CHECK(cudaMemcpyAsync(conv_slot(0, dst).data, conv_slot(0, src).data, conv_bytes,
                                   cudaMemcpyDeviceToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(recurrent_slot(0, dst).data, recurrent_slot(0, src).data,
                                   rec_bytes, cudaMemcpyDeviceToDevice, stream));
        return;
    }

    bool uniform_conv       = true;
    const auto conv_first   = reinterpret_cast<std::uintptr_t>(conv[0].data);
    const auto conv_second  = reinterpret_cast<std::uintptr_t>(conv[1].data);
    const auto conv_stride  = static_cast<std::size_t>(conv_second - conv_first);
    for (std::size_t l = 2; l < layers; ++l) {
        if (reinterpret_cast<std::uintptr_t>(conv[l].data) -
                reinterpret_cast<std::uintptr_t>(conv[l - 1].data) !=
            conv_stride) {
            uniform_conv = false;
            break;
        }
    }

    if (uniform_conv) {
        CUDA_CHECK(cudaMemcpy2DAsync(conv_slot(0, dst).data, conv_stride, conv_slot(0, src).data,
                                     conv_stride, conv_bytes, layers, cudaMemcpyDeviceToDevice,
                                     stream));
    } else {
        for (std::uint32_t layer = 0; layer < layers; ++layer) {
            const Tensor source      = conv_slot(layer, src);
            const Tensor destination = conv_slot(layer, dst);
            CUDA_CHECK(cudaMemcpyAsync(destination.data, source.data, conv_bytes,
                                       cudaMemcpyDeviceToDevice, stream));
        }
    }

    bool uniform_rec       = true;
    const auto rec_first   = reinterpret_cast<std::uintptr_t>(recurrent[0].data);
    const auto rec_second  = reinterpret_cast<std::uintptr_t>(recurrent[1].data);
    const auto rec_stride  = static_cast<std::size_t>(rec_second - rec_first);
    for (std::size_t l = 2; l < layers; ++l) {
        if (reinterpret_cast<std::uintptr_t>(recurrent[l].data) -
                reinterpret_cast<std::uintptr_t>(recurrent[l - 1].data) !=
            rec_stride) {
            uniform_rec = false;
            break;
        }
    }

    if (uniform_rec) {
        CUDA_CHECK(cudaMemcpy2DAsync(recurrent_slot(0, dst).data, rec_stride,
                                     recurrent_slot(0, src).data, rec_stride, rec_bytes, layers,
                                     cudaMemcpyDeviceToDevice, stream));
    } else {
        for (std::uint32_t layer = 0; layer < layers; ++layer) {
            const Tensor source      = recurrent_slot(layer, src);
            const Tensor destination = recurrent_slot(layer, dst);
            CUDA_CHECK(cudaMemcpyAsync(destination.data, source.data, rec_bytes,
                                       cudaMemcpyDeviceToDevice, stream));
        }
    }
}

void LinearAttentionStatePool::zero_slot(std::int32_t slot, cudaStream_t stream) {
    validate_layer_slot(*this, 0, slot, "LinearAttentionStatePool zero_slot");
    const std::uint32_t layers = layer_count();
    if (layers == 0) { return; }

    const std::size_t conv_bytes = conv_slot(0, slot).bytes();
    const std::size_t rec_bytes  = recurrent_slot(0, slot).bytes();

    if (layers == 1) {
        CUDA_CHECK(cudaMemsetAsync(conv_slot(0, slot).data, 0, conv_bytes, stream));
        CUDA_CHECK(cudaMemsetAsync(recurrent_slot(0, slot).data, 0, rec_bytes, stream));
        return;
    }

    bool uniform_conv      = true;
    const auto conv_first  = reinterpret_cast<std::uintptr_t>(conv[0].data);
    const auto conv_second = reinterpret_cast<std::uintptr_t>(conv[1].data);
    const auto conv_stride = static_cast<std::size_t>(conv_second - conv_first);
    for (std::size_t l = 2; l < layers; ++l) {
        if (reinterpret_cast<std::uintptr_t>(conv[l].data) -
                reinterpret_cast<std::uintptr_t>(conv[l - 1].data) !=
            conv_stride) {
            uniform_conv = false;
            break;
        }
    }

    if (uniform_conv) {
        CUDA_CHECK(cudaMemset2DAsync(conv_slot(0, slot).data, conv_stride, 0, conv_bytes, layers,
                                     stream));
    } else {
        for (std::uint32_t layer = 0; layer < layers; ++layer) {
            CUDA_CHECK(cudaMemsetAsync(conv_slot(layer, slot).data, 0, conv_bytes, stream));
        }
    }

    bool uniform_rec      = true;
    const auto rec_first  = reinterpret_cast<std::uintptr_t>(recurrent[0].data);
    const auto rec_second = reinterpret_cast<std::uintptr_t>(recurrent[1].data);
    const auto rec_stride = static_cast<std::size_t>(rec_second - rec_first);
    for (std::size_t l = 2; l < layers; ++l) {
        if (reinterpret_cast<std::uintptr_t>(recurrent[l].data) -
                reinterpret_cast<std::uintptr_t>(recurrent[l - 1].data) !=
            rec_stride) {
            uniform_rec = false;
            break;
        }
    }

    if (uniform_rec) {
        CUDA_CHECK(cudaMemset2DAsync(recurrent_slot(0, slot).data, rec_stride, 0, rec_bytes,
                                     layers, stream));
    } else {
        for (std::uint32_t layer = 0; layer < layers; ++layer) {
            CUDA_CHECK(cudaMemsetAsync(recurrent_slot(layer, slot).data, 0, rec_bytes, stream));
        }
    }
}

} // namespace ninfer
