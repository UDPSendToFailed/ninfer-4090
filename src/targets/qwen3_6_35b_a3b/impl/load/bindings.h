#pragma once

#include <ninfer/targets/qwen3_6_35b_a3b/package.h>
#include <ninfer/targets/qwen3_6/frontend_resources.h>
#include <ninfer/targets/qwen3_6/model_view.h>
#include <ninfer/targets/qwen3_6/startup_features.h>
#include <ninfer/targets/qwen3_6/vision.h>

#include "artifact/binder.h"
#include "artifact/materializer.h"
#include "core/tensor.h"
#include "ninfer/ops/sparse_moe.h"

#include <array>
#include <cstddef>
#include <optional>
#include <utility>

namespace ninfer::targets::qwen3_6_35b_a3b::detail {

inline constexpr std::size_t kTextLayers          = 40;
inline constexpr std::size_t kFullAttentionLayers = 10;
inline constexpr std::size_t kGdnLayers           = 30;
inline constexpr std::size_t kDFlashLayers        = 6;

struct MoePlan {
    artifact::ParameterReference router_shared_gate;
    artifact::ParameterReference routed_gate_up;
    artifact::ParameterReference routed_down;
    artifact::ParameterReference shared_gate_up;
    artifact::ParameterReference shared_down;
};

struct FullAttentionPlan {
    artifact::ParameterReference query_key_gate_value;
    artifact::ParameterReference query_norm;
    artifact::ParameterReference key_norm;
    artifact::ParameterReference output;
};

struct GdnPlan {
    artifact::ParameterReference a_log;
    artifact::ParameterReference dt_bias;
    artifact::ParameterReference convolution;
    artifact::ParameterReference a_b_projection;
    artifact::ParameterReference query_key_value_z;
    artifact::ParameterReference norm;
    artifact::ParameterReference output;
};

struct TextLayerPlan {
    artifact::ParameterReference input_norm;
    FullAttentionPlan attention{};
    GdnPlan gdn{};
    bool is_full_attention = false;
    artifact::ParameterReference post_attention_norm;
    MoePlan moe;
};

struct MtpPlan {
    artifact::ParameterReference input_projection;
    artifact::ParameterReference embedding_norm;
    artifact::ParameterReference hidden_norm;
    artifact::ParameterReference input_norm;
    FullAttentionPlan attention;
    artifact::ParameterReference post_attention_norm;
    MoePlan moe;
    artifact::ParameterReference final_norm;
};

struct DFlashLayerPlan {
    artifact::ParameterReference input_norm;
    artifact::ParameterReference query_key_value;
    artifact::ParameterReference query_norm;
    artifact::ParameterReference key_norm;
    artifact::ParameterReference attention_output;
    artifact::ParameterReference post_attention_norm;
    artifact::ParameterReference gate_up;
    artifact::ParameterReference down;
};

struct DFlashPlan {
    artifact::ParameterReference feature_projection;
    artifact::ParameterReference context_norm;
    std::array<DFlashLayerPlan, kDFlashLayers> layers;
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
    std::optional<DFlashPlan> dflash;
};

struct ArtifactLoadPlan {
    BindingPlan bindings;
    artifact::MaterializationPlan materialization;
};

ArtifactLoadPlan bind_artifact(artifact::Binder& binder, qwen3_6::StartupFeatures features);

struct SparseMoePayload {
    ops::SparseMoeWeights op;
};

struct AttentionProjectionPayload {
    Weight query_key_gate_value;
};

struct GdnProjectionPayload {
    Tensor a_log;
    Tensor dt_bias;
    Weight a_b_projection;
    Weight query_key_value_z;
};

using RuntimeModelView =
    qwen3_6::ModelView<AttentionProjectionPayload, GdnProjectionPayload, SparseMoePayload,
                       AttentionProjectionPayload, SparseMoePayload,
                       qwen3_6::DFlashWeights<kDFlashLayers>, kFullAttentionLayers, kGdnLayers>;
using FullAttentionWeights = RuntimeModelView::FullLayer;
using GdnWeights           = RuntimeModelView::GdnLayer;
using MtpWeights           = RuntimeModelView::MtpLayer;
using DFlashWeights        = RuntimeModelView::DFlash;
using DFlashLayerWeights   = qwen3_6::DFlashLayerWeights;

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
    Impl(WeightsProfile weights_profile, BindingPlan plan, artifact::MaterializedArtifact materialized)
        : weights_profile(weights_profile), data(std::move(plan), std::move(materialized)) {}

    WeightsProfile weights_profile = WeightsProfile::GroupwiseInt;
    LoadedModelData data;
};

} // namespace ninfer::targets::qwen3_6_35b_a3b::detail
