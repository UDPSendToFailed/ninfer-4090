#include "targets/qwen3_6_27b/impl/load/bindings.h"

#include "artifact/views.h"
#include "core/weight_view.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace ninfer::targets::qwen3_6_27b::detail {
namespace {

bool is_full_layer(std::size_t layer) { return layer >= 3 && (layer - 3) % 4 == 0; }

Weight coalesce_two(const artifact::MaterializedArtifact& materialized,
                    const artifact::ParameterReference& a, const artifact::ParameterReference& b,
                    const char* label) {
    const WeightView va = artifact::bind_view(a, materialized);
    const WeightView vb = artifact::bind_view(b, materialized);
    WeightView combined;
    const std::array<const WeightView*, 2> parts{&va, &vb};
    if (!try_coalesce_rows(parts, combined)) {
        throw std::runtime_error(std::string(label) + ": rows must be contiguous on one parent");
    }
    return native_weight(combined);
}

DensePostMixerPayload load_mlp(const MlpPlan& plan,
                               const artifact::MaterializedArtifact& materialized) {
    DensePostMixerPayload out;
    out.gate_up = coalesce_two(materialized, plan.gate, plan.up, "mlp/gate_up");
    out.down    = native_weight(artifact::bind_view(plan.down, materialized));
    return out;
}

FullAttentionProjectionPayload
load_attention_projection(const FullAttentionPlan& plan,
                          const artifact::MaterializedArtifact& materialized) {
    const WeightView q = artifact::bind_view(plan.query, materialized);
    const WeightView k = artifact::bind_view(plan.key, materialized);
    const WeightView g = artifact::bind_view(plan.gate, materialized);
    const WeightView v = artifact::bind_view(plan.value, materialized);
    WeightView fused;
    const std::array<const WeightView*, 4> all{&q, &k, &g, &v};
    if (try_coalesce_rows(all, fused)) {
        return FusedAttentionProjectionPayload{.query_key_gate_value = native_weight(fused)};
    }
    WeightView qk;
    WeightView gv;
    const std::array<const WeightView*, 2> qk_parts{&q, &k};
    const std::array<const WeightView*, 2> gv_parts{&g, &v};
    if (!try_coalesce_rows(qk_parts, qk) || !try_coalesce_rows(gv_parts, gv)) {
        throw std::runtime_error("attention q/k/g/v layout is neither fused nor (q,k)+(g,v) split");
    }
    return SplitAttentionProjectionPayload{
        .query_key  = native_weight(qk),
        .gate_value = native_weight(gv),
    };
}

GdnInputProjectionPayload
load_gdn_input_projection(const GdnPlan& plan, const artifact::MaterializedArtifact& materialized) {
    const WeightView q = artifact::bind_view(plan.query, materialized);
    const WeightView k = artifact::bind_view(plan.key, materialized);
    const WeightView v = artifact::bind_view(plan.value, materialized);
    const WeightView z = artifact::bind_view(plan.z, materialized);
    WeightView fused;
    const std::array<const WeightView*, 4> all{&q, &k, &v, &z};
    if (try_coalesce_rows(all, fused)) {
        return FusedGdnInputProjectionPayload{.query_key_value_z = native_weight(fused)};
    }
    WeightView qk;
    WeightView vz;
    const std::array<const WeightView*, 2> qk_parts{&q, &k};
    const std::array<const WeightView*, 2> vz_parts{&v, &z};
    if (!try_coalesce_rows(qk_parts, qk) || !try_coalesce_rows(vz_parts, vz)) {
        throw std::runtime_error("gdn q/k/v/z layout is neither fused nor (q,k)+(v,z) split");
    }
    return SplitGdnInputProjectionPayload{
        .query_key = native_weight(qk),
        .value_z   = native_weight(vz),
    };
}

void bind_text_layers(artifact::Binder& binder, BindingPlan& out) {
    const auto bind_dev = [&](std::string_view name, std::initializer_list<std::uint64_t> shape,
                              std::optional<QType> exact_format = {}) {
        return binder.parameter(name, shape, artifact::Residency::Device, exact_format);
    };
    for (std::size_t layer = 0; layer < kTextLayers; ++layer) {
        TextLayerPlan& target    = out.text_layers[layer];
        const std::string prefix = "text/layers/" + std::to_string(layer) + "/";
        target.input_norm        = bind_dev(prefix + "input_norm", {5120}, QType::BF16_CTRL);
        target.is_full_attention = is_full_layer(layer);
        if (target.is_full_attention) {
            target.attention.query      = bind_dev(prefix + "attention/query", {6144, 5120});
            target.attention.key        = bind_dev(prefix + "attention/key", {1024, 5120});
            target.attention.gate       = bind_dev(prefix + "attention/gate", {6144, 5120});
            target.attention.value      = bind_dev(prefix + "attention/value", {1024, 5120});
            target.attention.query_norm = bind_dev(prefix + "attention/query_norm", {256}, QType::BF16_CTRL);
            target.attention.key_norm   = bind_dev(prefix + "attention/key_norm", {256}, QType::BF16_CTRL);
            target.attention.output     = bind_dev(prefix + "attention/output", {5120, 6144});
        } else {
            target.gdn.a_log   = bind_dev(prefix + "gdn/a_log", {48}, QType::FP32_CTRL);
            target.gdn.dt_bias = bind_dev(prefix + "gdn/dt_bias", {48}, QType::FP32_CTRL);
            target.gdn.convolution =
                bind_dev(prefix + "gdn/convolution", {4, 10240}, QType::BF16_CTRL);
            target.gdn.a_projection =
                bind_dev(prefix + "gdn/a_projection", {48, 5120}, QType::BF16_CTRL);
            target.gdn.b_projection =
                bind_dev(prefix + "gdn/b_projection", {48, 5120}, QType::BF16_CTRL);
            target.gdn.query  = bind_dev(prefix + "gdn/query", {2048, 5120});
            target.gdn.key    = bind_dev(prefix + "gdn/key", {2048, 5120});
            target.gdn.value  = bind_dev(prefix + "gdn/value", {6144, 5120});
            target.gdn.z      = bind_dev(prefix + "gdn/z", {6144, 5120});
            target.gdn.norm   = bind_dev(prefix + "gdn/norm", {128}, QType::BF16_CTRL);
            target.gdn.output = bind_dev(prefix + "gdn/output", {5120, 6144});
        }
        target.post_attention_norm =
            bind_dev(prefix + "post_attention_norm", {5120}, QType::BF16_CTRL);
        target.mlp.gate = bind_dev(prefix + "mlp/gate", {17408, 5120});
        target.mlp.up   = bind_dev(prefix + "mlp/up", {17408, 5120});
        target.mlp.down = bind_dev(prefix + "mlp/down", {5120, 17408});
    }
}

void validate_draft_ids(artifact::Binder& binder, const artifact::ParameterReference& token_ids) {
    constexpr std::size_t kDraftVocab     = 131072;
    constexpr std::size_t kTokenizerVocab = 248077;
    const auto values                     = binder.values(token_ids.binding, QType::I32_CTRL);
    const auto ids                        = values.integers();
    if (ids.size() != kDraftVocab) {
        throw artifact::ArtifactError("draft-head token id count mismatch");
    }
    std::vector<bool> seen(kTokenizerVocab, false);
    for (const std::int32_t id : ids) {
        if (id < 0 || static_cast<std::size_t>(id) >= kTokenizerVocab) {
            throw artifact::ArtifactError("draft-head token id is outside tokenizer domain");
        }
        if (seen[static_cast<std::size_t>(id)]) {
            throw artifact::ArtifactError("draft-head token ids are not unique");
        }
        seen[static_cast<std::size_t>(id)] = true;
    }
}

} // namespace

ArtifactLoadPlan bind_artifact(artifact::Binder& binder, WeightsProfile weights_profile,
                               qwen3_6::StartupFeatures features) {
    const auto bind_dev = [&](std::string_view name, std::initializer_list<std::uint64_t> shape,
                              std::optional<QType> exact_format = {}) {
        return binder.parameter(name, shape, artifact::Residency::Device, exact_format);
    };

    ArtifactLoadPlan load_plan;
    BindingPlan& out = load_plan.bindings;
    out.frontend     = qwen3_6::bind_frontend_resources(binder);
    out.features     = features;

    const QType vocabulary_format =
        weights_profile == WeightsProfile::GroupwiseIntW8Endpoints ? QType::W8G32_F16S
                                                                  : QType::Q6G64_F16S;
    out.token_embedding =
        bind_dev("text/token_embedding", {248320, 5120}, vocabulary_format);
    bind_text_layers(binder, out);
    out.final_norm  = bind_dev("text/final_norm", {5120}, QType::BF16_CTRL);
    out.output_head = bind_dev("text/output_head", {248320, 5120}, vocabulary_format);

    if (features.optimized_proposal()) {
        ProposalPlan proposal;
        proposal.head      = bind_dev("proposal/head", {131072, 5120});
        proposal.token_ids = bind_dev("proposal/token_ids", {131072}, QType::I32_CTRL);
        validate_draft_ids(binder, proposal.token_ids);
        out.proposal = std::move(proposal);
    }

    if (features.mtp()) {
        MtpPlan mtp;
        mtp.input_projection    = bind_dev("mtp/input_projection", {5120, 10240});
        mtp.embedding_norm      = bind_dev("mtp/embedding_norm", {5120}, QType::BF16_CTRL);
        mtp.hidden_norm         = bind_dev("mtp/hidden_norm", {5120}, QType::BF16_CTRL);
        mtp.input_norm          = bind_dev("mtp/layers/0/input_norm", {5120}, QType::BF16_CTRL);
        mtp.query               = bind_dev("mtp/layers/0/attention/query", {6144, 5120});
        mtp.key                 = bind_dev("mtp/layers/0/attention/key", {1024, 5120});
        mtp.gate                = bind_dev("mtp/layers/0/attention/gate", {6144, 5120});
        mtp.value               = bind_dev("mtp/layers/0/attention/value", {1024, 5120});
        mtp.query_norm          = bind_dev("mtp/layers/0/attention/query_norm", {256}, QType::BF16_CTRL);
        mtp.key_norm            = bind_dev("mtp/layers/0/attention/key_norm", {256}, QType::BF16_CTRL);
        mtp.output              = bind_dev("mtp/layers/0/attention/output", {5120, 6144});
        mtp.post_attention_norm = bind_dev("mtp/layers/0/post_attention_norm", {5120}, QType::BF16_CTRL);
        mtp.mlp.gate            = bind_dev("mtp/layers/0/mlp/gate", {17408, 5120});
        mtp.mlp.up              = bind_dev("mtp/layers/0/mlp/up", {17408, 5120});
        mtp.mlp.down            = bind_dev("mtp/layers/0/mlp/down", {5120, 17408});
        mtp.final_norm          = bind_dev("mtp/final_norm", {5120}, QType::BF16_CTRL);
        out.mtp                 = std::move(mtp);
    }

    if (features.vision) {
        out.vision = VisionPlan{
            .backbone        = qwen3_6::bind_vision_backbone(binder),
            .merger_input    = qwen3_6::bind_vision_merger_input(binder),
            .merger_fc2      = bind_dev("vision/merger/fc2", {5120, 4608}),
            .merger_fc2_bias = bind_dev("vision/merger/fc2_bias", {5120}, QType::BF16_CTRL),
            .merger_norm     = qwen3_6::bind_vision_merger_norm(binder),
        };
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
                weight_tensor(artifact::bind_view(source.input_norm, backing), {5120});
            target.projection = load_attention_projection(source.attention, backing);
            target.query_norm =
                weight_tensor(artifact::bind_view(source.attention.query_norm, backing), {256});
            target.key_norm =
                weight_tensor(artifact::bind_view(source.attention.key_norm, backing), {256});
            target.output = native_weight(artifact::bind_view(source.attention.output, backing));
            target.post_attention_norm =
                weight_tensor(artifact::bind_view(source.post_attention_norm, backing), {5120});
            target.post_mixer = load_mlp(source.mlp, backing);
        } else {
            GdnWeights& target = gdn_layers.at(gdn_index++);
            target.input_norm =
                weight_tensor(artifact::bind_view(source.input_norm, backing), {5120});
            target.projection.a_log =
                weight_tensor(artifact::bind_view(source.gdn.a_log, backing), {48});
            target.projection.dt_bias =
                weight_tensor(artifact::bind_view(source.gdn.dt_bias, backing), {48});
            target.convolution =
                weight_tensor(artifact::bind_view(source.gdn.convolution, backing), {10240, 4});
            target.projection.a_projection =
                native_weight(artifact::bind_view(source.gdn.a_projection, backing));
            target.projection.b_projection =
                native_weight(artifact::bind_view(source.gdn.b_projection, backing));
            target.projection.input_projection = load_gdn_input_projection(source.gdn, backing);
            target.norm =
                weight_tensor(artifact::bind_view(source.gdn.norm, backing), {128});
            target.output = native_weight(artifact::bind_view(source.gdn.output, backing));
            target.post_attention_norm =
                weight_tensor(artifact::bind_view(source.post_attention_norm, backing), {5120});
            target.post_mixer = load_mlp(source.mlp, backing);
        }
    }
    if (full_index != full_layers.size() || gdn_index != gdn_layers.size()) {
        throw std::logic_error("text topology binding is incomplete");
    }
    final_norm  = weight_tensor(artifact::bind_view(plan.final_norm, backing), {5120});
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
            weight_tensor(artifact::bind_view(plan.mtp->embedding_norm, backing), {5120});
        mtp.hidden_norm =
            weight_tensor(artifact::bind_view(plan.mtp->hidden_norm, backing), {5120});
        mtp.input_norm =
            weight_tensor(artifact::bind_view(plan.mtp->input_norm, backing), {5120});

        const WeightView q_view = artifact::bind_view(plan.mtp->query, backing);
        const WeightView k_view = artifact::bind_view(plan.mtp->key, backing);
        const WeightView g_view = artifact::bind_view(plan.mtp->gate, backing);
        const WeightView v_view = artifact::bind_view(plan.mtp->value, backing);
        WeightView packed;
        const std::array<const WeightView*, 4> parts{&q_view, &k_view, &g_view, &v_view};
        if (try_coalesce_rows(parts, packed)) {
            mtp.attention.packed = native_weight(packed);
        }
        mtp.attention.query       = native_weight(q_view);
        mtp.attention.key         = native_weight(k_view);
        mtp.attention.output_gate = native_weight(g_view);
        mtp.attention.value       = native_weight(v_view);
        mtp.query_norm =
            weight_tensor(artifact::bind_view(plan.mtp->query_norm, backing), {256});
        mtp.key_norm =
            weight_tensor(artifact::bind_view(plan.mtp->key_norm, backing), {256});
        mtp.output = native_weight(artifact::bind_view(plan.mtp->output, backing));
        mtp.post_attention_norm =
            weight_tensor(artifact::bind_view(plan.mtp->post_attention_norm, backing), {5120});
        mtp.post_mixer = load_mlp(plan.mtp->mlp, backing);
        mtp.final_norm =
            weight_tensor(artifact::bind_view(plan.mtp->final_norm, backing), {5120});
    }

    if (plan.vision) {
        auto& vision  = runtime.vision.emplace();
        vision.common = qwen3_6::materialize_vision_common(
            backing, plan.vision->backbone, plan.vision->merger_input, plan.vision->merger_norm);
        vision.merger_fc2 = native_weight(artifact::bind_view(plan.vision->merger_fc2, backing));
        vision.merger_fc2_bias =
            weight_tensor(artifact::bind_view(plan.vision->merger_fc2_bias, backing), {5120});
    }
}

} // namespace ninfer::targets::qwen3_6_27b::detail
