#pragma once

#include <ninfer/targets/qwen3_6_27b/package.h>
#include <ninfer/targets/qwen3_6/frontend_resources.h>
#include <ninfer/targets/qwen3_6/model_view.h>
#include <ninfer/targets/qwen3_6/startup_features.h>
#include <ninfer/targets/qwen3_6/vision.h>

#include "artifact/binder.h"
#include "artifact/materializer.h"
#include "core/tensor.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <utility>
#include <variant>

namespace ninfer::targets::qwen3_6_27b::detail {

inline constexpr std::size_t kTextLayers          = 64;
inline constexpr std::size_t kFullAttentionLayers = 16;
inline constexpr std::size_t kGdnLayers           = 48;

struct MlpPlan {
    artifact::ParameterReference gate;
    artifact::ParameterReference up;
    artifact::ParameterReference down;
};

struct FullAttentionPlan {
    artifact::ParameterReference query;
    artifact::ParameterReference key;
    artifact::ParameterReference gate;
    artifact::ParameterReference value;
    artifact::ParameterReference query_norm;
    artifact::ParameterReference key_norm;
    artifact::ParameterReference output;
};

struct GdnPlan {
    artifact::ParameterReference a_log;
    artifact::ParameterReference dt_bias;
    artifact::ParameterReference convolution;
    artifact::ParameterReference a_projection;
    artifact::ParameterReference b_projection;
    artifact::ParameterReference query;
    artifact::ParameterReference key;
    artifact::ParameterReference value;
    artifact::ParameterReference z;
    artifact::ParameterReference norm;
    artifact::ParameterReference output;
};

struct TextLayerPlan {
    artifact::ParameterReference input_norm;
    FullAttentionPlan attention{};
    GdnPlan gdn{};
    bool is_full_attention = false;
    artifact::ParameterReference post_attention_norm;
    MlpPlan mlp;
};

struct MtpPlan {
    artifact::ParameterReference input_projection;
    artifact::ParameterReference embedding_norm;
    artifact::ParameterReference hidden_norm;
    artifact::ParameterReference input_norm;
    artifact::ParameterReference query;
    artifact::ParameterReference key;
    artifact::ParameterReference gate;
    artifact::ParameterReference value;
    artifact::ParameterReference query_norm;
    artifact::ParameterReference key_norm;
    artifact::ParameterReference output;
    artifact::ParameterReference post_attention_norm;
    MlpPlan mlp;
    artifact::ParameterReference final_norm;
};

struct VisionPlan {
    qwen3_6::VisionBackbonePlan backbone;
    qwen3_6::VisionMergerInputPlan merger_input;
    artifact::ParameterReference merger_fc2;
    artifact::ParameterReference merger_fc2_bias;
    qwen3_6::VisionMergerNormPlan merger_norm;
};

struct ProposalPlan {
    artifact::ParameterReference head;
    artifact::ParameterReference token_ids;
};

struct BindingPlan {
    qwen3_6::FrontendResourcePlan frontend;
    qwen3_6::StartupFeatures features;

    artifact::ParameterReference token_embedding;
    std::array<TextLayerPlan, kTextLayers> text_layers;
    artifact::ParameterReference final_norm;
    artifact::ParameterReference output_head;
    std::optional<ProposalPlan> proposal;
    std::optional<MtpPlan> mtp;
    std::optional<VisionPlan> vision;
};

struct ArtifactLoadPlan {
    BindingPlan bindings;
    artifact::MaterializationPlan materialization;
};

ArtifactLoadPlan bind_artifact(artifact::Binder& binder, WeightsProfile weights_profile,
                               qwen3_6::StartupFeatures features);

struct DensePostMixerPayload {
    Weight gate_up;
    Weight down;
};

struct SplitAttentionProjectionPayload {
    Weight query_key;
    Weight gate_value;
};

struct FusedAttentionProjectionPayload {
    Weight query_key_gate_value;
};

using FullAttentionProjectionPayload =
    std::variant<SplitAttentionProjectionPayload, FusedAttentionProjectionPayload>;

struct SplitGdnInputProjectionPayload {
    Weight query_key;
    Weight value_z;
};

struct FusedGdnInputProjectionPayload {
    Weight query_key_value_z;
};

using GdnInputProjectionPayload =
    std::variant<SplitGdnInputProjectionPayload, FusedGdnInputProjectionPayload>;

struct GdnProjectionPayload {
    Tensor a_log;
    Tensor dt_bias;
    Weight a_projection;
    Weight b_projection;
    GdnInputProjectionPayload input_projection;
};

struct MtpAttentionPayload {
    Weight packed;
    Weight query;
    Weight key;
    Weight output_gate;
    Weight value;
};

using RuntimeModelView =
    qwen3_6::ModelView<FullAttentionProjectionPayload, GdnProjectionPayload, DensePostMixerPayload,
                       MtpAttentionPayload, DensePostMixerPayload, qwen3_6::DFlashWeights<6>,
                       kFullAttentionLayers, kGdnLayers>;
using FullAttentionWeights = RuntimeModelView::FullLayer;
using GdnWeights           = RuntimeModelView::GdnLayer;
using MtpWeights           = RuntimeModelView::MtpLayer;

class LoadedModelData {
public:
    LoadedModelData(BindingPlan plan, artifact::MaterializedArtifact materialized);

    LoadedModelData(const LoadedModelData&)            = delete;
    LoadedModelData& operator=(const LoadedModelData&) = delete;
    LoadedModelData(LoadedModelData&&)                 = delete;
    LoadedModelData& operator=(LoadedModelData&&)      = delete;

    artifact::MaterializedArtifact backing;
    qwen3_6::FrontendResources frontend;
    RuntimeModelView runtime;
};

class LoadedModel::Impl {
public:
    Impl(WeightsProfile weights_profile_in, BindingPlan plan,
         artifact::MaterializedArtifact materialized)
        : weights_profile(weights_profile_in), data(std::move(plan), std::move(materialized)) {}

    WeightsProfile weights_profile;
    LoadedModelData data;
};

} // namespace ninfer::targets::qwen3_6_27b::detail
