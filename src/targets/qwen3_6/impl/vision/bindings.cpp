#include <ninfer/targets/qwen3_6/vision.h>

#include "artifact/materializer.h"
#include "artifact/views.h"
#include "core/weight_view.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <stdexcept>
#include <string>
#include <string_view>

namespace ninfer::targets::qwen3_6 {
namespace {

Weight coalesce_qkv(const artifact::MaterializedArtifact& materialized,
                    const artifact::ParameterReference& q, const artifact::ParameterReference& k,
                    const artifact::ParameterReference& v) {
    const WeightView q_view = artifact::bind_view(q, materialized);
    const WeightView k_view = artifact::bind_view(k, materialized);
    const WeightView v_view = artifact::bind_view(v, materialized);
    WeightView combined;
    const std::array<const WeightView*, 3> parts{&q_view, &k_view, &v_view};
    if (!try_coalesce_rows(parts, combined)) {
        throw std::runtime_error("vision attention q/k/v must be contiguous on one parent");
    }
    return native_weight(combined);
}

Tensor coalesce_qkv_bias(const artifact::MaterializedArtifact& materialized,
                         const artifact::ParameterReference& q,
                         const artifact::ParameterReference& k,
                         const artifact::ParameterReference& v) {
    const WeightView q_view = artifact::bind_view(q, materialized);
    const WeightView k_view = artifact::bind_view(k, materialized);
    const WeightView v_view = artifact::bind_view(v, materialized);
    WeightView combined;
    combined.shape = {3 * VisionBackboneConfig::hidden};
    for (const WeightView* part_view : {&q_view, &k_view, &v_view}) {
        for (const auto& region : part_view->parts) { combined.parts.push_back(region); }
    }
    return weight_tensor(combined, {3 * VisionBackboneConfig::hidden});
}

} // namespace

VisionBackbonePlan bind_vision_backbone(artifact::Binder& binder) {
    const auto bind_dev = [&](std::string_view name, std::initializer_list<std::uint64_t> shape,
                              std::optional<QType> exact_format = {}) {
        return binder.parameter(name, shape, artifact::Residency::Device, exact_format);
    };

    VisionBackbonePlan out;
    out.patch_embedding = bind_dev("vision/patch_embedding",
                                   {VisionBackboneConfig::hidden, VisionBackboneConfig::patch_dim});
    out.patch_embedding_bias = bind_dev("vision/patch_embedding_bias",
                                        {VisionBackboneConfig::hidden}, QType::BF16_CTRL);
    out.position_embedding =
        bind_dev("vision/position_embedding",
                 {VisionBackboneConfig::position_embeddings, VisionBackboneConfig::hidden},
                 QType::BF16_CTRL);

    for (std::size_t layer = 0; layer < out.layers.size(); ++layer) {
        VisionLayerPlan& target  = out.layers[layer];
        const std::string prefix = "vision/layers/" + std::to_string(layer) + "/";
        target.query = bind_dev(prefix + "attention/query",
                                {VisionBackboneConfig::hidden, VisionBackboneConfig::hidden});
        target.key   = bind_dev(prefix + "attention/key",
                                {VisionBackboneConfig::hidden, VisionBackboneConfig::hidden});
        target.value = bind_dev(prefix + "attention/value",
                                {VisionBackboneConfig::hidden, VisionBackboneConfig::hidden});
        target.query_bias = bind_dev(prefix + "attention/query_bias",
                                     {VisionBackboneConfig::hidden}, QType::BF16_CTRL);
        target.key_bias   = bind_dev(prefix + "attention/key_bias", {VisionBackboneConfig::hidden},
                                     QType::BF16_CTRL);
        target.value_bias = bind_dev(prefix + "attention/value_bias",
                                     {VisionBackboneConfig::hidden}, QType::BF16_CTRL);
        target.output     = bind_dev(prefix + "attention/output",
                                     {VisionBackboneConfig::hidden, VisionBackboneConfig::hidden});
        target.output_bias = bind_dev(prefix + "attention/output_bias",
                                      {VisionBackboneConfig::hidden}, QType::BF16_CTRL);
        target.fc1         = bind_dev(prefix + "mlp/fc1",
                                      {VisionBackboneConfig::intermediate, VisionBackboneConfig::hidden});
        target.fc1_bias    = bind_dev(prefix + "mlp/fc1_bias",
                                      {VisionBackboneConfig::intermediate}, QType::BF16_CTRL);
        target.fc2         = bind_dev(prefix + "mlp/fc2",
                                      {VisionBackboneConfig::hidden, VisionBackboneConfig::intermediate});
        target.fc2_bias =
            bind_dev(prefix + "mlp/fc2_bias", {VisionBackboneConfig::hidden}, QType::BF16_CTRL);
        target.norm1_weight =
            bind_dev(prefix + "norm1_weight", {VisionBackboneConfig::hidden}, QType::BF16_CTRL);
        target.norm1_bias =
            bind_dev(prefix + "norm1_bias", {VisionBackboneConfig::hidden}, QType::BF16_CTRL);
        target.norm2_weight =
            bind_dev(prefix + "norm2_weight", {VisionBackboneConfig::hidden}, QType::BF16_CTRL);
        target.norm2_bias =
            bind_dev(prefix + "norm2_bias", {VisionBackboneConfig::hidden}, QType::BF16_CTRL);
    }
    return out;
}

VisionMergerInputPlan bind_vision_merger_input(artifact::Binder& binder) {
    return VisionMergerInputPlan{
        .fc1 = binder.parameter(
            "vision/merger/fc1",
            {VisionBackboneConfig::merger_hidden, VisionBackboneConfig::merger_hidden}),
        .fc1_bias = binder.parameter("vision/merger/fc1_bias",
                                     {VisionBackboneConfig::merger_hidden},
                                     artifact::Residency::Device, QType::BF16_CTRL),
    };
}

VisionMergerNormPlan bind_vision_merger_norm(artifact::Binder& binder) {
    return VisionMergerNormPlan{
        .weight = binder.parameter("vision/merger/norm_weight", {VisionBackboneConfig::hidden},
                                   artifact::Residency::Device, QType::BF16_CTRL),
        .bias   = binder.parameter("vision/merger/norm_bias", {VisionBackboneConfig::hidden},
                                   artifact::Residency::Device, QType::BF16_CTRL),
    };
}

VisionCommonWeights materialize_vision_common(const artifact::MaterializedArtifact& materialized,
                                              const VisionBackbonePlan& backbone,
                                              const VisionMergerInputPlan& merger_input,
                                              const VisionMergerNormPlan& merger_norm) {
    VisionCommonWeights out;
    out.patch_embedding =
        native_weight(artifact::bind_view(backbone.patch_embedding, materialized));
    out.patch_embedding_bias =
        weight_tensor(artifact::bind_view(backbone.patch_embedding_bias, materialized),
                      {VisionBackboneConfig::hidden});
    out.position_embedding =
        weight_tensor(artifact::bind_view(backbone.position_embedding, materialized),
                      {VisionBackboneConfig::hidden, VisionBackboneConfig::position_embeddings});

    for (std::size_t layer = 0; layer < out.layers.size(); ++layer) {
        const VisionLayerPlan& source = backbone.layers[layer];
        VisionLayerWeights& target    = out.layers[layer];
        target.qkv = coalesce_qkv(materialized, source.query, source.key, source.value);
        target.qkv_bias =
            coalesce_qkv_bias(materialized, source.query_bias, source.key_bias, source.value_bias);
        target.output = native_weight(artifact::bind_view(source.output, materialized));
        target.output_bias =
            weight_tensor(artifact::bind_view(source.output_bias, materialized),
                          {VisionBackboneConfig::hidden});
        target.fc1      = native_weight(artifact::bind_view(source.fc1, materialized));
        target.fc1_bias = weight_tensor(artifact::bind_view(source.fc1_bias, materialized),
                                        {VisionBackboneConfig::intermediate});
        target.fc2      = native_weight(artifact::bind_view(source.fc2, materialized));
        target.fc2_bias = weight_tensor(artifact::bind_view(source.fc2_bias, materialized),
                                        {VisionBackboneConfig::hidden});
        target.norm1_weight =
            weight_tensor(artifact::bind_view(source.norm1_weight, materialized),
                          {VisionBackboneConfig::hidden});
        target.norm1_bias = weight_tensor(artifact::bind_view(source.norm1_bias, materialized),
                                          {VisionBackboneConfig::hidden});
        target.norm2_weight =
            weight_tensor(artifact::bind_view(source.norm2_weight, materialized),
                          {VisionBackboneConfig::hidden});
        target.norm2_bias = weight_tensor(artifact::bind_view(source.norm2_bias, materialized),
                                          {VisionBackboneConfig::hidden});
    }

    out.merger_fc1 = native_weight(artifact::bind_view(merger_input.fc1, materialized));
    out.merger_fc1_bias =
        weight_tensor(artifact::bind_view(merger_input.fc1_bias, materialized),
                      {VisionBackboneConfig::merger_hidden});
    out.merger_norm_weight =
        weight_tensor(artifact::bind_view(merger_norm.weight, materialized),
                      {VisionBackboneConfig::hidden});
    out.merger_norm_bias =
        weight_tensor(artifact::bind_view(merger_norm.bias, materialized),
                      {VisionBackboneConfig::hidden});
    return out;
}

} // namespace ninfer::targets::qwen3_6
