#include "targets/qwen3_6_35b_a3b/impl/load/bindings.h"

#include "artifact/views.h"
#include "core/weight_view.h"

#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace ninfer::targets::qwen3_6_35b_a3b::detail {
namespace {

bool is_full_layer(std::size_t layer) { return layer >= 3 && (layer - 3) % 4 == 0; }

Weight row_view(const Weight& block, std::int32_t row_begin, std::int32_t row_count) {
    if (row_begin < 0 || row_count <= 0 || row_begin + row_count > block.n ||
        block.qtype != QType::W8G32_F16S || block.layout != QuantLayout::RowSplit ||
        block.group != 32) {
        throw std::logic_error("invalid DFlash W8 row view");
    }
    const std::uint64_t code_row = static_cast<std::uint64_t>(block.padded_shape[1]);
    const std::uint64_t scale_row =
        static_cast<std::uint64_t>(block.padded_shape[1] / block.group) * sizeof(std::uint16_t);
    Weight out = block;
    out.qdata  = static_cast<const std::byte*>(block.qdata) +
                static_cast<std::uint64_t>(row_begin) * code_row;
    out.scales = static_cast<const std::byte*>(block.scales) +
                 static_cast<std::uint64_t>(row_begin) * scale_row;
    out.n               = row_count;
    out.shape[0]        = row_count;
    out.padded_shape[0] = row_count;
    return out;
}

MoePlan bind_moe(artifact::Binder& binder, const std::string& prefix) {
    const auto bind_dev = [&](std::string_view name, std::initializer_list<std::uint64_t> shape,
                              std::optional<QType> exact_format = {}) {
        return binder.parameter(name, shape, artifact::Residency::Device, exact_format);
    };
    return MoePlan{
        .router_shared_gate = bind_dev(prefix + "router_shared_gate", {257, 2048}, QType::BF16_CTRL),
        .routed_gate_up     = bind_dev(prefix + "routed_gate_up", {262144, 2048}),
        .routed_down        = bind_dev(prefix + "routed_down", {524288, 512}),
        .shared_gate_up     = bind_dev(prefix + "shared_gate_up", {1024, 2048}),
        .shared_down        = bind_dev(prefix + "shared_down", {2048, 512}),
    };
}

SparseMoePayload load_moe(const MoePlan& plan, const artifact::MaterializedArtifact& materialized) {
    return SparseMoePayload{
        .op = {
            .router_shared_gate =
                native_weight(artifact::bind_view(plan.router_shared_gate, materialized)),
            .routed_gate_up = native_weight(artifact::bind_view(plan.routed_gate_up, materialized)),
            .routed_down    = native_weight(artifact::bind_view(plan.routed_down, materialized)),
            .shared_gate_up = native_weight(artifact::bind_view(plan.shared_gate_up, materialized)),
            .shared_down    = native_weight(artifact::bind_view(plan.shared_down, materialized)),
        }};
}

void validate_draft_ids(artifact::Binder& binder, const artifact::ParameterReference& token_ids) {
    constexpr std::size_t kDraftVocab     = 131072;
    constexpr std::size_t kTokenizerVocab = 248077;
    const auto values                     = binder.values(token_ids.binding, QType::I32_CTRL);
    const auto ids                        = values.integers();
    if (ids.size() != kDraftVocab) {
        throw artifact::ArtifactError("invalid optimized draft-head token ids");
    }
    std::vector<bool> seen(kTokenizerVocab, false);
    for (const std::int32_t id : ids) {
        if (id < 0 || static_cast<std::size_t>(id) >= kTokenizerVocab ||
            seen[static_cast<std::size_t>(id)]) {
            throw artifact::ArtifactError("invalid optimized draft-head token ids");
        }
        seen[static_cast<std::size_t>(id)] = true;
    }
}

} // namespace

ArtifactLoadPlan bind_artifact(artifact::Binder& binder, qwen3_6::StartupFeatures features) {
    const auto bind_dev = [&](std::string_view name, std::initializer_list<std::uint64_t> shape,
                              std::optional<QType> exact_format = {}) {
        return binder.parameter(name, shape, artifact::Residency::Device, exact_format);
    };

    ArtifactLoadPlan load_plan;
    BindingPlan& out    = load_plan.bindings;
    out.frontend        = qwen3_6::bind_frontend_resources(binder);
    out.features        = features;
    out.token_embedding = bind_dev("text/token_embedding", {248320, 2048});

    for (std::size_t layer = 0; layer < kTextLayers; ++layer) {
        TextLayerPlan& target    = out.text_layers[layer];
        const std::string prefix = "text/layers/" + std::to_string(layer) + "/";
        target.input_norm        = bind_dev(prefix + "input_norm", {2048}, QType::BF16_CTRL);
        target.is_full_attention = is_full_layer(layer);
        if (target.is_full_attention) {
            target.attention.query_key_gate_value =
                bind_dev(prefix + "attention/query_key_gate_value", {9216, 2048});
            target.attention.query_norm =
                bind_dev(prefix + "attention/query_norm", {256}, QType::BF16_CTRL);
            target.attention.key_norm =
                bind_dev(prefix + "attention/key_norm", {256}, QType::BF16_CTRL);
            target.attention.output = bind_dev(prefix + "attention/output", {2048, 4096});
        } else {
            target.gdn.a_log   = bind_dev(prefix + "gdn/a_log", {32}, QType::FP32_CTRL);
            target.gdn.dt_bias = bind_dev(prefix + "gdn/dt_bias", {32}, QType::FP32_CTRL);
            target.gdn.convolution =
                bind_dev(prefix + "gdn/convolution", {4, 8192}, QType::BF16_CTRL);
            target.gdn.a_b_projection =
                bind_dev(prefix + "gdn/a_b_projection", {64, 2048}, QType::BF16_CTRL);
            target.gdn.query_key_value_z = bind_dev(prefix + "gdn/query_key_value_z", {12288, 2048});
            target.gdn.norm              = bind_dev(prefix + "gdn/norm", {128}, QType::BF16_CTRL);
            target.gdn.output            = bind_dev(prefix + "gdn/output", {2048, 4096});
        }
        target.post_attention_norm =
            bind_dev(prefix + "post_attention_norm", {2048}, QType::BF16_CTRL);
        target.moe = bind_moe(binder, prefix + "moe/");
    }

    out.final_norm  = bind_dev("text/final_norm", {2048}, QType::BF16_CTRL);
    out.output_head = bind_dev("text/output_head", {248320, 2048});

    if (features.optimized_proposal()) {
        ProposalPlan proposal;
        proposal.head      = bind_dev("proposal/head", {131072, 2048});
        proposal.token_ids = bind_dev("proposal/token_ids", {131072}, QType::I32_CTRL);
        validate_draft_ids(binder, proposal.token_ids);
        out.proposal = std::move(proposal);
    }

    if (features.mtp()) {
        MtpPlan mtp;
        mtp.input_projection = bind_dev("mtp/input_projection", {2048, 4096});
        mtp.embedding_norm   = bind_dev("mtp/embedding_norm", {2048}, QType::BF16_CTRL);
        mtp.hidden_norm      = bind_dev("mtp/hidden_norm", {2048}, QType::BF16_CTRL);
        mtp.input_norm       = bind_dev("mtp/layers/0/input_norm", {2048}, QType::BF16_CTRL);
        mtp.attention.query_key_gate_value =
            bind_dev("mtp/layers/0/attention/query_key_gate_value", {9216, 2048});
        mtp.attention.query_norm =
            bind_dev("mtp/layers/0/attention/query_norm", {256}, QType::BF16_CTRL);
        mtp.attention.key_norm =
            bind_dev("mtp/layers/0/attention/key_norm", {256}, QType::BF16_CTRL);
        mtp.attention.output = bind_dev("mtp/layers/0/attention/output", {2048, 4096});
        mtp.post_attention_norm =
            bind_dev("mtp/layers/0/post_attention_norm", {2048}, QType::BF16_CTRL);
        mtp.moe        = bind_moe(binder, "mtp/layers/0/moe/");
        mtp.final_norm = bind_dev("mtp/final_norm", {2048}, QType::BF16_CTRL);
        out.mtp        = std::move(mtp);
    }

    if (features.vision) {
        out.vision = VisionPlan{
            .backbone        = qwen3_6::bind_vision_backbone(binder),
            .merger_input    = qwen3_6::bind_vision_merger_input(binder),
            .merger_fc2      = bind_dev("vision/merger/fc2", {2048, 4608}),
            .merger_fc2_bias = bind_dev("vision/merger/fc2_bias", {2048}, QType::BF16_CTRL),
            .merger_norm     = qwen3_6::bind_vision_merger_norm(binder),
        };
    }

    if (features.dflash()) {
        if (!binder.contains("dflash/feature_projection")) {
            throw artifact::ArtifactError(
                "DFlash was requested but this compact artifact has no DFlash weights");
        }
        DFlashPlan dflash;
        dflash.feature_projection = bind_dev("dflash/feature_projection", {2048, 16384});
        dflash.context_norm       = bind_dev("dflash/context_norm", {2048}, QType::BF16_CTRL);
        for (std::size_t layer = 0; layer < kDFlashLayers; ++layer) {
            DFlashLayerPlan& target  = dflash.layers[layer];
            const std::string prefix = "dflash/layers/" + std::to_string(layer) + "/";
            target.input_norm        = bind_dev(prefix + "input_norm", {2048}, QType::BF16_CTRL);
            target.query_key_value   = bind_dev(prefix + "attention/query_key_value", {6144, 2048});
            target.query_norm = bind_dev(prefix + "attention/query_norm", {128}, QType::BF16_CTRL);
            target.key_norm   = bind_dev(prefix + "attention/key_norm", {128}, QType::BF16_CTRL);
            target.attention_output = bind_dev(prefix + "attention/output", {2048, 4096});
            target.post_attention_norm =
                bind_dev(prefix + "post_attention_norm", {2048}, QType::BF16_CTRL);
            target.gate_up = bind_dev(prefix + "mlp/gate_up", {12288, 2048});
            target.down    = bind_dev(prefix + "mlp/down", {2048, 6144});
        }
        dflash.final_norm = bind_dev("dflash/final_norm", {2048}, QType::BF16_CTRL);
        out.dflash        = std::move(dflash);
    }

    load_plan.materialization = std::move(binder).finish();
    return load_plan;
}

LoadedModelData::LoadedModelData(BindingPlan plan, artifact::MaterializedArtifact materialized)
    : backing(std::move(materialized)) {
    frontend = qwen3_6::take_frontend_resources(backing, plan.frontend);

    runtime.weights_arena = &backing.device_arena();
    runtime.features      = plan.features;
    auto& token_embedding = runtime.token_embedding;
    auto& full_layers     = runtime.full_layers;
    auto& gdn_layers      = runtime.gdn_layers;
    auto& final_norm      = runtime.final_norm;
    auto& output_head     = runtime.output_head;

    token_embedding = native_weight(artifact::bind_view(plan.token_embedding, backing));

    std::size_t full_index = 0;
    std::size_t gdn_index  = 0;
    for (std::size_t layer = 0; layer < kTextLayers; ++layer) {
        const TextLayerPlan& source = plan.text_layers[layer];
        if (source.is_full_attention) {
            FullAttentionWeights& target = full_layers.at(full_index++);
            target.input_norm =
                weight_tensor(artifact::bind_view(source.input_norm, backing), {2048});
            target.projection.query_key_gate_value =
                native_weight(artifact::bind_view(source.attention.query_key_gate_value, backing));
            target.query_norm =
                weight_tensor(artifact::bind_view(source.attention.query_norm, backing), {256});
            target.key_norm =
                weight_tensor(artifact::bind_view(source.attention.key_norm, backing), {256});
            target.output = native_weight(artifact::bind_view(source.attention.output, backing));
            target.post_attention_norm =
                weight_tensor(artifact::bind_view(source.post_attention_norm, backing), {2048});
            target.post_mixer = load_moe(source.moe, backing);
        } else {
            GdnWeights& target = gdn_layers.at(gdn_index++);
            target.input_norm =
                weight_tensor(artifact::bind_view(source.input_norm, backing), {2048});
            target.projection.a_log =
                weight_tensor(artifact::bind_view(source.gdn.a_log, backing), {32});
            target.projection.dt_bias =
                weight_tensor(artifact::bind_view(source.gdn.dt_bias, backing), {32});
            target.convolution =
                weight_tensor(artifact::bind_view(source.gdn.convolution, backing), {8192, 4});
            target.projection.a_b_projection =
                native_weight(artifact::bind_view(source.gdn.a_b_projection, backing));
            target.projection.query_key_value_z =
                native_weight(artifact::bind_view(source.gdn.query_key_value_z, backing));
            target.norm =
                weight_tensor(artifact::bind_view(source.gdn.norm, backing), {128});
            target.output = native_weight(artifact::bind_view(source.gdn.output, backing));
            target.post_attention_norm =
                weight_tensor(artifact::bind_view(source.post_attention_norm, backing), {2048});
            target.post_mixer = load_moe(source.moe, backing);
        }
    }
    if (full_index != full_layers.size() || gdn_index != gdn_layers.size()) {
        throw std::logic_error("35B Text topology binding is incomplete");
    }

    final_norm  = weight_tensor(artifact::bind_view(plan.final_norm, backing), {2048});
    output_head = native_weight(artifact::bind_view(plan.output_head, backing));
    if (plan.proposal) {
        auto& proposal     = runtime.optimized_proposal.emplace();
        proposal.head      = native_weight(artifact::bind_view(plan.proposal->head, backing));
        proposal.token_ids = weight_tensor(artifact::bind_view(plan.proposal->token_ids, backing), {131072});
    }

    if (plan.mtp) {
        auto& mtp            = runtime.mtp.emplace();
        mtp.input_projection = native_weight(artifact::bind_view(plan.mtp->input_projection, backing));
        mtp.embedding_norm =
            weight_tensor(artifact::bind_view(plan.mtp->embedding_norm, backing), {2048});
        mtp.hidden_norm =
            weight_tensor(artifact::bind_view(plan.mtp->hidden_norm, backing), {2048});
        mtp.input_norm =
            weight_tensor(artifact::bind_view(plan.mtp->input_norm, backing), {2048});
        mtp.attention.query_key_gate_value = native_weight(
            artifact::bind_view(plan.mtp->attention.query_key_gate_value, backing));
        mtp.query_norm =
            weight_tensor(artifact::bind_view(plan.mtp->attention.query_norm, backing), {256});
        mtp.key_norm =
            weight_tensor(artifact::bind_view(plan.mtp->attention.key_norm, backing), {256});
        mtp.output = native_weight(artifact::bind_view(plan.mtp->attention.output, backing));
        mtp.post_attention_norm =
            weight_tensor(artifact::bind_view(plan.mtp->post_attention_norm, backing), {2048});
        mtp.post_mixer = load_moe(plan.mtp->moe, backing);
        mtp.final_norm =
            weight_tensor(artifact::bind_view(plan.mtp->final_norm, backing), {2048});
    }

    if (plan.vision) {
        auto& vision  = runtime.vision.emplace();
        vision.common = qwen3_6::materialize_vision_common(
            backing, plan.vision->backbone, plan.vision->merger_input, plan.vision->merger_norm);
        vision.merger_fc2 = native_weight(artifact::bind_view(plan.vision->merger_fc2, backing));
        vision.merger_fc2_bias =
            weight_tensor(artifact::bind_view(plan.vision->merger_fc2_bias, backing), {2048});
    }

    if (plan.dflash) {
        DFlashWeights& target     = runtime.dflash.emplace();
        target.feature_projection = native_weight(artifact::bind_view(plan.dflash->feature_projection, backing));
        target.context_norm =
            weight_tensor(artifact::bind_view(plan.dflash->context_norm, backing), {2048});
        for (std::size_t layer = 0; layer < kDFlashLayers; ++layer) {
            const DFlashLayerPlan& source = plan.dflash->layers[layer];
            DFlashLayerWeights& weights   = target.layers[layer];
            weights.input_norm =
                weight_tensor(artifact::bind_view(source.input_norm, backing), {2048});
            weights.query_key_value =
                native_weight(artifact::bind_view(source.query_key_value, backing));
            weights.context_key   = row_view(weights.query_key_value, 4096, 1024);
            weights.context_value = row_view(weights.query_key_value, 5120, 1024);
            weights.query_norm =
                weight_tensor(artifact::bind_view(source.query_norm, backing), {128});
            weights.key_norm =
                weight_tensor(artifact::bind_view(source.key_norm, backing), {128});
            weights.attention_output =
                native_weight(artifact::bind_view(source.attention_output, backing));
            weights.post_attention_norm =
                weight_tensor(artifact::bind_view(source.post_attention_norm, backing), {2048});
            weights.gate_up = native_weight(artifact::bind_view(source.gate_up, backing));
            weights.down    = native_weight(artifact::bind_view(source.down, backing));
        }
        target.final_norm =
            weight_tensor(artifact::bind_view(plan.dflash->final_norm, backing), {2048});
    }
}

} // namespace ninfer::targets::qwen3_6_35b_a3b::detail
