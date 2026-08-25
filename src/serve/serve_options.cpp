#include "serve/serve_options.h"
#include "product/speculative_options.h"

#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>

namespace ninfer::serve {
namespace {

int parse_nonnegative_int(const char* text, const char* label) {
    char* end        = nullptr;
    const long value = std::strtol(text, &end, 10);
    if (end == text || *end != '\0' || value < 0 ||
        value > static_cast<long>(std::numeric_limits<int>::max())) {
        throw std::invalid_argument(std::string("invalid ") + label + ": " + text);
    }
    return static_cast<int>(value);
}

float parse_float_in(const char* text, const char* label, float lo, float hi) {
    char* end          = nullptr;
    const double value = std::strtod(text, &end);
    if (end == text || *end != '\0' || !(value >= lo) || !(value <= hi)) {
        throw std::invalid_argument(std::string("invalid ") + label + ": " + text);
    }
    return static_cast<float>(value);
}

std::uint64_t parse_u64(const char* text, const char* label) {
    if (text == nullptr || *text == '\0' || *text == '-') {
        throw std::invalid_argument(std::string("invalid ") + label + ": " +
                                    (text == nullptr ? "" : text));
    }
    errno                          = 0;
    char* end                      = nullptr;
    const unsigned long long value = std::strtoull(text, &end, 10);
    if (errno == ERANGE || end == text || *end != '\0') {
        throw std::invalid_argument(std::string("invalid ") + label + ": " + text);
    }
    return static_cast<std::uint64_t>(value);
}

KvCacheStorage parse_kv_dtype(const char* text) {
    const std::string value(text);
    if (value == "bf16") { return KvCacheStorage::BFloat16; }
    if (value == "int8") { return KvCacheStorage::Int8Group64; }
    if (value == "rk8v4") { return KvCacheStorage::RotatedInt8KeyInt4ValueGroup64; }
    if (value == "rk4v4") { return KvCacheStorage::RotatedInt4KeyInt4ValueGroup64; }
    if (value == "rk4v4-e8") { return KvCacheStorage::RK4V4E8; }
    if (value == "rk2v4-e8") { return KvCacheStorage::RK2V4E8; }
    throw std::invalid_argument("invalid kv-dtype: " + value);
}

KvCapacityPolicy parse_kv_capacity(const char* text) {
    if (std::string_view(text) == "auto") { return KvCapacityPolicy::automatic(); }
    const int value = parse_nonnegative_int(text, "kv-capacity");
    if (value == 0) { throw std::invalid_argument("--kv-capacity must be positive"); }
    return KvCapacityPolicy::explicit_capacity(static_cast<std::uint32_t>(value));
}

} // namespace

std::string serve_usage_text(const char* argv0) {
    const std::string headroom_mib =
        std::to_string(kDefaultKvCapacityHeadroomBytes / (1024ULL * 1024ULL));
    const std::string default_max_toks = std::to_string(kDefaultMaxTokens);
    return std::string("usage: ") + argv0 + " <model.ninfer> [options]\n\n"
           "High-performance OpenAI Responses/Chat Completions and Anthropic Messages server\n"
           "for native .ninfer checkpoint artifacts.\n\n"
           "Server & Network:\n"
           "  --host <H>                  HTTP listen address (default: 127.0.0.1)\n"
           "  --port <N>                  HTTP listen port (default: 8080)\n"
           "  --api-key <KEY>             Bearer token authentication key (optional; auth disabled if omitted)\n"
           "  --cors                      Enable Cross-Origin Resource Sharing (CORS) headers for web clients\n"
           "  --ui / --no-ui              Enable or disable the embedded Web UI dashboard (default: enabled)\n"
           "  --model-id <ID>             Override model identifier in /v1/models (default: artifact identity.model_id)\n"
           "  --max-request-mib <N>       Maximum incoming request payload size in MiB (default: 384, enforced before parsing)\n"
           "  --request-log-jsonl <FILE>  Append full-precision server request and telemetry records to JSONL file\n"
           "  --log-stats-interval-ms <N> Periodic throughput and engine stats logging interval in ms (default: 5000; 0 disables)\n\n"
           "Concurrency & Ingress:\n"
           "  --max-concurrency <N>       Maximum active parallel decode slots (1 to 8, default: 1)\n"
           "  --max-pending-requests <N>  Maximum capacity of incoming request FIFO queue (default: 16)\n"
           "  --pending-timeout-ms <N>    Queue wait timeout before returning 503 Service Unavailable (default: 30000)\n\n"
           "Stateful Conversations & Response Store:\n"
           "  --response-store-max-records <N>  Maximum completed response records in process-local store (default: 1024)\n"
           "  --response-store-max-mib <N>      Memory budget in MiB for response store records (default: 256)\n\n"
           "Model & Hardware Configuration:\n"
           "  --device <N>                CUDA device ordinal (default: 0)\n"
           "  --max-context <N>           Maximum sequence context length in tokens (prompt + completion, default: 8192)\n"
           "  --kv-capacity <N|auto>      KV cache capacity in tokens, or 'auto' (allocates available VRAM\n"
           "                              reserving " + headroom_mib + " MiB headroom; default: matches --max-context)\n"
           "  --prefill-chunk <N>         Prefill chunk size in tokens (multiple of 128, default: 1024)\n"
           "  --no-cuda-graph             Disable CUDA Graph capture/replay (executes via standard CUDA streams)\n"
           "  --no-prefix-reuse           Disable KV prefix cache reuse across requests (prefix reuse enabled by default)\n\n"
           "Persistent Prompt Cache (DirectStorage DMA):\n"
           "  --disk-cache / --prompt-cache     Enable persistent multi-turn prompt caching to disk via DirectStorage DMA\n"
           "  --no-disk-cache                   Disable persistent disk prompt cache (default)\n"
           "  --disk-cache-dir <DIR>            Directory path for prompt cache (default: %LOCALAPPDATA%/ninfer/cache/<profile>)\n"
           "  --disk-cache-gb <N>               Disk cache storage quota limit in GiB (default: 30)\n\n"
           "Quantization & Storage Layouts:\n"
           "  --kv-dtype <dtype>          KV cache storage data type and quantization layout:\n"
           "                                bf16       - 16-bit brain floating-point\n"
           "                                int8       - 8-bit integer channel-quantized\n"
           "                                rk8v4      - 8-bit rank-compressed K, 4-bit V\n"
           "                                rk4v4      - 4-bit rank-compressed K, 4-bit V\n"
           "                                rk4v4-e8   - 4-bit E8 lattice Keys, 4-bit Values\n"
           "                                rk2v4-e8   - 2-bit E8 lattice Keys, 4-bit Values\n\n"
           "Speculative Decoding:\n"
           "  --spec <mtp|dflash>         Speculative execution backend (default: none / autoregressive):\n"
           "                                mtp        - Multi-Token Prediction draft heads\n"
           "                                dflash     - Block-parallel draft model (35B-A3B text-only)\n"
           "  --draft-tokens <N>          Speculative draft tokens per verification round (MTP: 1..15, DFlash: 1..15)\n"
           "  --lm-head-draft             Reuse base model LM head weights for draft logits projection (requires --spec)\n\n"
           "Vision & Multimodal:\n"
           "  --vision                    Enable image/video vision encoder and load Vision GPU allocations\n"
           "  --vision-max-tokens <N>     Vision scratchpad token capacity (default: 8192)\n\n"
           "Reasoning & Generation Defaults:\n"
           "  --default-max-tokens <N>    Default maximum output tokens when omitted in client request (default: " + default_max_toks + ")\n"
           "  --no-thinking               Disable reasoning/thinking mode globally by default\n"
           "  --preserve-thinking         Retain closed-turn assistant reasoning in multi-turn conversation history\n"
           "  --reasoning-effort <effort> Default thinking depth preset (low | medium | xhigh) when omitted by client\n\n"
           "Sampler Defaults (overridden by client request parameters):\n"
           "  --temperature <F>           Fallback softmax temperature (0.0 to 2.0)\n"
           "  --top-p <F>                 Fallback nucleus sampling cumulative probability cutoff (0.0 to 1.0)\n"
           "  --top-k <N>                 Fallback top-K vocabulary truncation candidate count (0 to INT32_MAX)\n"
           "  --min-p <F>                 Fallback minimum token probability relative to top token (0.0 to 1.0)\n"
           "  --presence-penalty <F>      Fallback presence penalty (-2.0 to 2.0)\n"
           "  --frequency-penalty <F>     Fallback frequency penalty (-2.0 to 2.0)\n"
           "  --seed <N>                  Fallback 64-bit random seed\n"
           "  --greedy                    Force greedy argmax sampling (equivalent to --temperature 0)\n";
}

ServeOptions parse_serve_options(int argc, char** argv) {
    ServeOptions options;
    options.startup_argv.reserve(static_cast<std::size_t>(argc));
    bool redact_next = false;
    for (int i = 0; i < argc; ++i) {
        if (redact_next) {
            options.startup_argv.emplace_back("<redacted>");
            redact_next = false;
            continue;
        }
        options.startup_argv.emplace_back(argv[i] == nullptr ? "" : argv[i]);
        redact_next = options.startup_argv.back() == "--api-key";
    }
    bool default_max_tokens_explicit = false;
    bool kv_capacity_explicit        = false;
    if (argc >= 2 && (std::string(argv[1]) == "--help" || std::string(argv[1]) == "-h")) {
        options.help_requested = true;
        return options;
    }
    if (argc < 2) { throw std::invalid_argument("artifact path is required"); }
    options.artifact_path = argv[1];
    for (int i = 2; i < argc; ++i) {
        const std::string arg    = argv[i];
        const auto require_value = [&](const char* flag) -> const char* {
            if (++i >= argc) { throw std::invalid_argument(std::string(flag) + " needs a value"); }
            return argv[i];
        };
        if (arg == "--host") {
            options.host = require_value("--host");
        } else if (arg == "--port") {
            options.port = parse_nonnegative_int(require_value("--port"), "port");
        } else if (arg == "--api-key") {
            options.api_key = require_value("--api-key");
        } else if (arg == "--model-id") {
            options.model_id_override = require_value("--model-id");
            if (options.model_id_override->empty()) {
                throw std::invalid_argument("--model-id must not be empty");
            }
        } else if (arg == "--max-context") {
            options.max_context = static_cast<std::uint32_t>(
                parse_nonnegative_int(require_value("--max-context"), "max-context"));
        } else if (arg == "--kv-capacity") {
            options.kv_capacity  = parse_kv_capacity(require_value("--kv-capacity"));
            kv_capacity_explicit = true;
        } else if (arg == "--max-concurrency") {
            options.max_concurrency = static_cast<std::uint32_t>(
                parse_nonnegative_int(require_value("--max-concurrency"), "max-concurrency"));
        } else if (arg == "--max-pending-requests") {
            options.max_pending_requests = static_cast<std::uint32_t>(parse_nonnegative_int(
                require_value("--max-pending-requests"), "max-pending-requests"));
        } else if (arg == "--pending-timeout-ms") {
            options.pending_timeout_ms = static_cast<std::uint32_t>(
                parse_nonnegative_int(require_value("--pending-timeout-ms"), "pending-timeout-ms"));
        } else if (arg == "--prefill-chunk") {
            options.prefill_chunk = static_cast<std::uint32_t>(
                parse_nonnegative_int(require_value("--prefill-chunk"), "prefill-chunk"));
        } else if (arg == "--log-stats-interval-ms") {
            options.log_stats_interval_ms = static_cast<std::uint32_t>(parse_nonnegative_int(
                require_value("--log-stats-interval-ms"), "log-stats-interval-ms"));
        } else if (arg == "--max-request-mib") {
            const std::uint64_t mib =
                parse_u64(require_value("--max-request-mib"), "max-request-mib");
            if (mib == 0 || mib > std::numeric_limits<std::size_t>::max() / (1ULL << 20)) {
                throw std::invalid_argument("--max-request-mib is out of range");
            }
            options.max_request_bytes = static_cast<std::size_t>(mib << 20);
        } else if (arg == "--request-log-jsonl") {
            options.request_log_jsonl = require_value("--request-log-jsonl");
            if (options.request_log_jsonl.empty()) {
                throw std::invalid_argument("--request-log-jsonl must not be empty");
            }
        } else if (arg == "--response-store-max-records") {
            const int records = parse_nonnegative_int(require_value("--response-store-max-records"),
                                                      "response-store-max-records");
            if (records == 0) {
                throw std::invalid_argument("--response-store-max-records must be positive");
            }
            options.response_store_max_records = static_cast<std::size_t>(records);
        } else if (arg == "--response-store-max-mib") {
            const std::uint64_t mib =
                parse_u64(require_value("--response-store-max-mib"), "response-store-max-mib");
            if (mib == 0 || mib > std::numeric_limits<std::size_t>::max() / (1ULL << 20)) {
                throw std::invalid_argument("--response-store-max-mib is out of range");
            }
            options.response_store_max_bytes = static_cast<std::size_t>(mib << 20);
        } else if (arg == "--device") {
            options.device = parse_nonnegative_int(require_value("--device"), "device");
        } else if (arg == "--kv-dtype") {
            options.kv_cache = parse_kv_dtype(require_value("--kv-dtype"));
        } else if (arg == "--spec") {
            options.speculative.backend =
                product::parse_speculative_backend(require_value("--spec"));
        } else if (arg == "--draft-tokens") {
            options.speculative.draft_tokens = static_cast<std::uint32_t>(
                parse_nonnegative_int(require_value("--draft-tokens"), "draft-tokens"));
        } else if (arg == "--default-max-tokens") {
            options.default_max_tokens =
                parse_nonnegative_int(require_value("--default-max-tokens"), "default-max-tokens");
            default_max_tokens_explicit = true;
        } else if (arg == "--vision") {
            options.enable_vision = true;
        } else if (arg == "--vision-max-tokens" || arg == "--vision-limit") {
            const int val = parse_nonnegative_int(require_value(arg.c_str()), "vision-max-tokens");
            if (val <= 0) {
                throw std::invalid_argument(std::string(arg) + " must be positive");
            }
            options.vision_max_tokens = static_cast<std::uint32_t>(val);
            options.enable_vision     = true;
        } else if (arg == "--no-cuda-graph") {
            options.use_cuda_graph = false;
        } else if (arg == "--no-prefix-reuse") {
            options.allow_prefix_reuse = false;
        } else if (arg == "--prompt-cache" || arg == "--disk-cache" || arg == "--enable-prompt-cache") {
            options.enable_prompt_cache = true;
        } else if (arg == "--no-prompt-cache" || arg == "--no-disk-cache") {
            options.enable_prompt_cache = false;
        } else if (arg == "--prompt-cache-dir" || arg == "--disk-cache-dir") {
            options.prompt_cache_dir    = require_value(arg.c_str());
            options.enable_prompt_cache = true;
        } else if (arg == "--prompt-cache-gb" || arg == "--disk-cache-gb") {
            const std::uint64_t gb = parse_u64(require_value(arg.c_str()), arg.c_str());
            options.prompt_cache_max_bytes = static_cast<std::size_t>(gb << 30);
            options.enable_prompt_cache    = true;
        } else if (arg == "--lm-head-draft") {
            options.speculative.proposal_head = ProposalHead::Optimized;
        } else if (arg == "--no-thinking") {
            options.enable_thinking = false;
        } else if (arg == "--preserve-thinking") {
            options.preserve_thinking = true;
        } else if (arg == "--reasoning-effort" || arg == "--thinking-effort") {
            const std::string val = require_value(arg.c_str());
            const auto effort     = parse_requested_reasoning_effort(val);
            if (!effort) {
                throw std::invalid_argument("invalid value for " + arg + ": '" + val +
                                            "' (expected low, medium, or xhigh)");
            }
            options.default_reasoning_effort = *effort;
        } else if (arg == "--cors") {
            options.enable_cors = true;
        } else if (arg == "--ui") {
            options.enable_ui = true;
        } else if (arg == "--no-ui") {
            options.enable_ui = false;
        } else if (arg == "--temperature") {
            options.sampling_overrides.temperature =
                parse_float_in(require_value("--temperature"), "temperature", 0.0f, 2.0f);
        } else if (arg == "--top-p") {
            options.sampling_overrides.top_p =
                parse_float_in(require_value("--top-p"), "top-p", 0.0f, 1.0f);
        } else if (arg == "--top-k") {
            options.sampling_overrides.top_k =
                parse_nonnegative_int(require_value("--top-k"), "top-k");
        } else if (arg == "--min-p") {
            options.sampling_overrides.min_p =
                parse_float_in(require_value("--min-p"), "min-p", 0.0f, 1.0f);
        } else if (arg == "--presence-penalty") {
            options.sampling_overrides.presence_penalty = parse_float_in(
                require_value("--presence-penalty"), "presence-penalty", -2.0f, 2.0f);
        } else if (arg == "--frequency-penalty") {
            options.sampling_overrides.frequency_penalty = parse_float_in(
                require_value("--frequency-penalty"), "frequency-penalty", -2.0f, 2.0f);
        } else if (arg == "--seed") {
            options.sampling_overrides.seed = parse_u64(require_value("--seed"), "seed");
        } else if (arg == "--greedy") {
            options.greedy = true;
        } else {
            throw std::invalid_argument("unknown argument: " + arg);
        }
    }
    if (!kv_capacity_explicit) {
        options.kv_capacity = KvCapacityPolicy::explicit_capacity(options.max_context);
    }
    if (options.port <= 0 || options.port > 65535) {
        throw std::invalid_argument("--port must be in [1,65535]");
    }
    if (options.max_context == 0) { throw std::invalid_argument("--max-context must be positive"); }
    if (options.kv_capacity.mode == KvCapacityMode::Explicit &&
        options.kv_capacity.explicit_tokens < options.max_context) {
        throw std::invalid_argument("--kv-capacity must be at least --max-context");
    }
    if (options.max_concurrency == 0 || options.max_concurrency > kMaximumConcurrency) {
        throw std::invalid_argument("--max-concurrency must be in [1,8]");
    }
    if (options.max_pending_requests == 0) {
        throw std::invalid_argument("--max-pending-requests must be positive");
    }
    if (options.pending_timeout_ms == 0) {
        throw std::invalid_argument("--pending-timeout-ms must be positive");
    }
    if (options.max_request_bytes == 0) {
        throw std::invalid_argument("--max-request-mib must be positive");
    }
    if (options.prefill_chunk == 0 || options.prefill_chunk % 128 != 0) {
        throw std::invalid_argument("--prefill-chunk must be a positive multiple of 128");
    }
    product::validate_speculative_cli_options(options.speculative);
    if (options.speculative.backend == SpeculativeBackend::DFlash && options.enable_vision) {
        throw std::invalid_argument("--spec dflash cannot be combined with --vision");
    }
    if (default_max_tokens_explicit) {
        if (options.default_max_tokens <= 0) {
            throw std::invalid_argument("--default-max-tokens must be positive");
        }
    }
    return options;
}

std::string resolve_public_model_id(const ServeOptions& options,
                                    std::string_view artifact_model_id) {
    if (options.model_id_override.has_value()) { return *options.model_id_override; }
    if (artifact_model_id.empty()) {
        throw std::logic_error("loaded artifact model_id must not be empty");
    }
    return std::string(artifact_model_id);
}

} // namespace ninfer::serve
