#include "ui.h"
#include "serve/generation_service.h"
#include "serve/openai_schema.h"
#include "serve/serve_options.h"

#include <nlohmann/json.hpp>

#include <cstdio>
#include <string>
#include <vector>

namespace {

using namespace ninfer::serve;

int check(bool condition, const char* message) {
    if (condition) { return 0; }
    std::fprintf(stderr, "FAIL %s\n", message);
    return 1;
}

} // namespace

int main() {
    int failures = 0;

    // 1. Asset table lookup tests
    if (ninfer_ui_has_assets()) {
        const NinferUiAsset* index_asset = ninfer_ui_find_asset("index.html");
        failures += check(index_asset != nullptr, "index.html asset not found in embedded assets");
        if (index_asset != nullptr) {
            failures += check(index_asset->name == "index.html", "index.html name mismatch");
            failures += check(index_asset->size > 0, "index.html has zero size");
            failures += check(index_asset->data != nullptr, "index.html data pointer is null");
            failures += check(!index_asset->etag.empty(), "index.html has empty etag");
            failures += check(index_asset->type == "text/html; charset=utf-8", "index.html mime type mismatch");

            // ETag comparison checks
            const std::string etag = index_asset->etag;
            const std::string weak_etag = "W/" + etag;
            failures += check(etag == index_asset->etag, "exact etag match failed");
            failures += check(weak_etag == ("W/" + index_asset->etag), "weak etag match failed");
        }

        // Test non-existent asset
        const NinferUiAsset* missing = ninfer_ui_find_asset("nonexistent_asset_xyz.bin");
        failures += check(missing == nullptr, "non-existent asset unexpectedly found");
    } else {
        std::printf("INFO: Built without embedded WebUI assets (stub mode)\n");
        const NinferUiAsset* missing = ninfer_ui_find_asset("index.html");
        failures += check(missing == nullptr, "index.html unexpectedly found in stub mode");
    }

    // 2. Validate /props JSON schema contract
    ServeOptions options;
    options.max_context = 65536;
    options.max_concurrency = 4;
    options.enable_ui = true;

    ninfer::SamplingPreset sampling;
    sampling.temperature = 0.7f;
    sampling.top_p = 0.95f;
    sampling.top_k = 40;
    const std::uint64_t seed = 42;

    nlohmann::json default_gen = {
        {"n_ctx", options.max_context},
        {"params", {
            {"temp", sampling.temperature},
            {"top_p", sampling.top_p},
            {"top_k", sampling.top_k},
            {"seed", seed}
        }}
    };

    nlohmann::json props = {
        {"default_generation_settings", default_gen},
        {"total_slots", options.max_concurrency},
        {"model_alias", "test-model-alias"},
        {"modalities", {
            {"vision", true},
            {"video", false},
            {"audio", false}
        }},
        {"endpoint_slots", true},
        {"endpoint_props", true},
        {"endpoint_metrics", true},
        {"ui", options.enable_ui}
    };

    failures += check(props.contains("default_generation_settings"), "props missing default_generation_settings");
    failures += check(props["default_generation_settings"]["n_ctx"] == 65536, "props n_ctx mismatch");
    failures += check(props["default_generation_settings"]["params"]["temp"] == 0.7f, "props temp mismatch");
    failures += check(props["default_generation_settings"]["params"]["top_p"] == 0.95f, "props top_p mismatch");
    failures += check(props["default_generation_settings"]["params"]["top_k"] == 40, "props top_k mismatch");
    failures += check(props["total_slots"] == 4, "props total_slots mismatch");
    failures += check(props["model_alias"] == "test-model-alias", "props model_alias mismatch");
    failures += check(props["modalities"]["vision"] == true, "props modalities.vision mismatch");
    failures += check(props["modalities"]["video"] == false, "props modalities.video mismatch");
    failures += check(props["modalities"]["audio"] == false, "props modalities.audio mismatch");
    failures += check(props["endpoint_slots"] == true, "props endpoint_slots mismatch");
    failures += check(props["endpoint_props"] == true, "props endpoint_props mismatch");
    failures += check(props["endpoint_metrics"] == true, "props endpoint_metrics mismatch");
    failures += check(props["ui"] == true, "props ui mismatch");

    // 3. Validate WebUI chat completions without explicit model field
    RequestLimits limits;
    limits.default_max_tokens = 512;
    const nlohmann::json webui_request = {
        {"messages", nlohmann::json::array({nlohmann::json{{"role", "user"}, {"content", "Hello!"}}})},
        {"stream", true}
    };
    const GenerationRequest parsed = parse_chat_completion_request(webui_request, limits);
    failures += check(parsed.model.empty(), "parsed model should be empty when omitted in WebUI request");
    failures += check(parsed.stream == true, "stream should be true");
    failures += check(parsed.messages.size() == 1, "messages parsed");
    failures += check(parsed.messages[0].content.size() == 1, "message content parsed");
    failures += check(parsed.messages[0].content[0].text == "Hello!", "message text mismatch");

    // 4. Validate timings object in WebUI response / chunk
    GenerationOutcome outcome;
    outcome.prompt_tokens = 971;
    outcome.completion_tokens = 8192;
    outcome.metrics.prefix_cache_hit_tokens = 969;
    outcome.metrics.prefill_seconds = 0.2;
    outcome.metrics.decode_seconds = 112.0;

    const CompletionTimings timings = make_completion_timings(outcome);
    failures += check(timings.prompt_n == 2, "timings prompt_n is fresh tokens (971 - 969 = 2)");
    failures += check(timings.cache_n == 969, "timings cache_n is cached tokens");
    failures += check(timings.predicted_n == 8192, "timings predicted_n matches completion_tokens");

    const CompletionUsage usage = make_completion_usage(outcome);
    const std::string json_resp = make_chat_completion_response(
        "chatcmpl-test", "test-model", 123456, "Hello world", "", "stop", usage, timings);
    const nlohmann::json parsed_resp = nlohmann::json::parse(json_resp);
    failures += check(parsed_resp.contains("timings"), "response contains timings");
    failures += check(parsed_resp["timings"]["prompt_n"] == 2, "timings prompt_n match");
    failures += check(parsed_resp["timings"]["predicted_n"] == 8192, "timings predicted_n match");
    failures += check(parsed_resp["timings"]["cache_n"] == 969, "timings cache_n match");

    // 5. Validate live streaming delta chunk timings
    const CompletionTimings live_chunk_timings{2, 200.0, 10.0, 42, 500.0, 84.0, 969};
    const std::string content_chunk_str = make_chat_chunk_content(
        "chatcmpl-test", "test-model", 123456, " live token", false, live_chunk_timings);
    const nlohmann::json content_chunk = nlohmann::json::parse(
        content_chunk_str.substr(content_chunk_str.find('{'), content_chunk_str.rfind('}') - content_chunk_str.find('{') + 1));
    failures += check(content_chunk.contains("timings"), "content chunk contains live timings");
    failures += check(content_chunk["timings"]["predicted_n"] == 42, "live content chunk predicted_n match");
    failures += check(content_chunk["timings"]["cache_n"] == 969, "live content chunk cache_n match");
    failures += check(content_chunk["timings"]["predicted_per_second"] == 84.0, "live content chunk predicted_per_second match");

    return failures;
}
