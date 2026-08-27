#include "serve/http_server.h"

#include "serve/anthropic_schema.h"
#include "serve/console_log.h"
#include "serve/openai_schema.h"
#include "serve/request_log.h"
#include "serve/translate.h"
#include "ui.h"

#include <nlohmann/json.hpp>

#include <atomic>
#include <chrono>
#include <exception>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

namespace ninfer::serve {
namespace {

struct StreamingRequest {
    explicit StreamingRequest(PreparedRequest request) : prepared(std::move(request)) {}

    PreparedRequest prepared;
    std::atomic<bool> cancelled{false};
    bool started = false;
};

class ClientDisconnected final : public std::exception {
public:
    [[nodiscard]] const char* what() const noexcept override { return "client disconnected"; }
};

void write_stream_item(httplib::DataSink& sink, StreamingRequest& request,
                       const std::string& item) {
    if (request.cancelled.load(std::memory_order_acquire) ||
        (sink.is_writable && !sink.is_writable()) || !sink.write(item.data(), item.size())) {
        request.cancelled.store(true, std::memory_order_release);
        throw ClientDisconnected();
    }
}

void set_owned_content(httplib::Response& response, std::string body,
                       std::shared_ptr<RequestLifetime> lifetime) {
    response.set_content(std::move(body), "application/json");
    response.hold_resource(std::move(lifetime));
}

void write_error(httplib::Response& res, const ApiError& error) {
    res.status = error.status;
    res.set_content(make_error_body(error), "application/json");
}

// Anthropic-shaped error body ({"type":"error","error":{...}}), used by the
// /v1/messages endpoints so Claude clients see the error format they expect.
void write_messages_error(httplib::Response& res, const ApiError& error) {
    res.status = error.status;
    res.set_content(make_messages_error_body(error), "application/json");
}

void write_exception(httplib::Response& res, const std::exception& ex) {
    ApiError error;
    error.status  = 500;
    error.type    = "internal_error";
    error.message = ex.what();
    write_error(res, error);
}

std::string sse_error_event(const ApiError& error) {
    return "data: " + make_error_body(error) + "\n\n";
}

ThroughputReport make_throughput_report(const ninfer::RuntimeStats& previous,
                                        const ninfer::RuntimeStats& current,
                                        double interval_seconds) {
    return ThroughputReport{
        .interval_seconds = interval_seconds,
        .computed_prefill_tokens =
            current.computed_prefill_tokens - previous.computed_prefill_tokens,
        .committed_decode_tokens =
            current.committed_decode_tokens - previous.committed_decode_tokens,
        .decode_rounds     = current.decode_rounds - previous.decode_rounds,
        .decode_row_rounds = current.decode_row_rounds - previous.decode_row_rounds,
        .scheduler         = current,
    };
}

bool report_has_activity(const ThroughputReport& report) {
    return report.computed_prefill_tokens != 0 || report.committed_decode_tokens != 0 ||
           report.decode_rounds != 0 || report.scheduler.running_requests != 0 ||
           report.scheduler.waiting_requests != 0;
}

std::string_view unstreamed_content(const GenerationOutcome& outcome) {
    if (outcome.streamed_content_bytes > outcome.text.size()) {
        throw std::logic_error("streamed content exceeds terminal content");
    }
    return std::string_view(outcome.text).substr(outcome.streamed_content_bytes);
}

} // namespace

HttpServer::HttpServer(ServeOptions options)
    : options_(std::move(options)),
      response_store_(options_.response_store_max_records, options_.response_store_max_bytes),
      request_jsonl_(options_.request_log_jsonl, options_.artifact_path) {
    const std::size_t queued_requests =
        static_cast<std::size_t>(options_.max_concurrency) + options_.max_pending_requests;
    const std::size_t worker_count = queued_requests + 1;
    server_.new_task_queue         = [queued_requests, worker_count] {
        return new httplib::ThreadPool(worker_count, queued_requests);
    };
    server_.set_payload_max_length(options_.max_request_bytes);
    register_routes();
}

void HttpServer::log_line(const std::string& line) {
    write_console_log(ConsoleLogLevel::Info, line);
}

void HttpServer::log_request_start(const RequestLogContext& context) {
    log_line(format_request_start(context));
    request_jsonl_.write_request_start(context);
    metrics_.begin_request(context.id, context.prompt_tokens);
}

void HttpServer::log_request_done(const RequestLogContext& context,
                                  const GenerationOutcome& outcome) {
    log_line(format_request_done(context, outcome));
    request_jsonl_.write_request_done(context, outcome);
    metrics_.end_request(context.id);
    metrics_.record(outcome);
}

void HttpServer::log_request_error(const RequestLogContext& context, const std::string& message) {
    log_line(format_request_error(context, message));
    request_jsonl_.write_request_error(context, message);
    metrics_.end_request(context.id);
}

void HttpServer::log_throughput(const ThroughputReport& report) {
    log_line(format_throughput(report));
    request_jsonl_.write_throughput(report);
}

void HttpServer::run_stats_reporter() {
    using Clock                     = std::chrono::steady_clock;
    ninfer::RuntimeStats previous   = service_->runtime_stats();
    Clock::time_point previous_time = Clock::now();
    const auto interval             = std::chrono::milliseconds(options_.log_stats_interval_ms);

    for (;;) {
        {
            std::unique_lock lock(stats_mutex_);
            if (stats_cv_.wait_for(lock, interval, [this] { return stats_stopping_; })) { break; }
        }

        const ninfer::RuntimeStats current = service_->runtime_stats();
        const Clock::time_point now        = Clock::now();
        const ThroughputReport report      = make_throughput_report(
            previous, current, std::chrono::duration<double>(now - previous_time).count());
        if (report_has_activity(report)) { log_throughput(report); }
        previous      = current;
        previous_time = now;
    }

    const ninfer::RuntimeStats current = service_->runtime_stats();
    const Clock::time_point now        = Clock::now();
    const ThroughputReport tail        = make_throughput_report(
        previous, current, std::chrono::duration<double>(now - previous_time).count());
    if (tail.computed_prefill_tokens != 0 || tail.committed_decode_tokens != 0 ||
        tail.decode_rounds != 0) {
        log_throughput(tail);
    }
}

void HttpServer::stop_stats_reporter() {
    if (!stats_thread_.joinable()) { return; }
    {
        std::lock_guard lock(stats_mutex_);
        stats_stopping_ = true;
    }
    stats_cv_.notify_one();
    stats_thread_.join();
}

void HttpServer::register_routes() {
    server_.set_error_handler([](const httplib::Request& req, httplib::Response& res) {
        // httplib invokes this for EVERY status >= 400, including 413s that
        // route handlers already answered with a specific error body (e.g.
        // media_budget_exceeded). Only synthesize the generic payload-limit
        // error for the bare 413 httplib itself produces on oversized bodies.
        if (res.status != 413 || !res.body.empty()) { return; }
        ApiError error;
        error.status  = 413;
        error.type    = "invalid_request_error";
        error.code    = "request_too_large";
        error.message = "request body exceeds the configured payload limit";
        if (req.path.rfind("/v1/messages", 0) == 0) {
            write_messages_error(res, error);
        } else {
            write_error(res, error);
        }
    });
    if (options_.enable_cors) {
        server_.set_default_headers(
            {{"Access-Control-Allow-Origin", "*"},
             {"Access-Control-Allow-Headers", "Authorization, Content-Type"},
             {"Access-Control-Allow-Methods", "GET, POST, DELETE, OPTIONS"}});
        // CORS preflight: browsers send OPTIONS with no credentials before the real
        // request; answer it without auth so the actual GET/POST can carry the key.
        server_.Options(R"(.*)",
                        [](const httplib::Request&, httplib::Response& res) { res.status = 204; });
    }

    server_.set_pre_routing_handler([this](const httplib::Request& req, httplib::Response& res) {
        if (options_.api_key.empty() || req.path == "/health" || req.method == "OPTIONS") {
            return httplib::Server::HandlerResponse::Unhandled;
        }
        // Accept both the OpenAI-style bearer token and the Anthropic-style
        // x-api-key header so OpenAI clients and Claude Code (ANTHROPIC_API_KEY
        // -> x-api-key, ANTHROPIC_AUTH_TOKEN -> Authorization: Bearer) both work.
        const bool bearer_ok =
            req.get_header_value("Authorization") == ("Bearer " + options_.api_key);
        const bool x_api_key_ok = req.get_header_value("x-api-key") == options_.api_key;
        if (!bearer_ok && !x_api_key_ok) {
            ApiError error;
            error.status  = 401;
            error.type    = "invalid_request_error";
            error.code    = "invalid_api_key";
            error.message = "missing or invalid API key";
            // Render the 401 in the shape the target endpoint speaks.
            if (req.path.rfind("/v1/messages", 0) == 0) {
                write_messages_error(res, error);
            } else {
                write_error(res, error);
            }
            return httplib::Server::HandlerResponse::Handled;
        }
        return httplib::Server::HandlerResponse::Unhandled;
    });

    server_.set_exception_handler(
        [](const httplib::Request&, httplib::Response& res, std::exception_ptr ep) {
            try {
                std::rethrow_exception(ep);
            } catch (const ApiException& e) {
                write_error(res, e.error());
            } catch (const std::exception& e) { write_exception(res, e); } catch (...) {
                ApiError error;
                error.status  = 500;
                error.type    = "internal_error";
                error.message = "unknown error";
                write_error(res, error);
            }
        });

    server_.Get("/health", [](const httplib::Request&, httplib::Response& res) {
        res.set_content(nlohmann::json{{"status", "ok"}}.dump(), "application/json");
    });
    server_.Get("/metrics", [this](const httplib::Request&, httplib::Response& res) {
        res.set_content(metrics_.render(options_.max_concurrency), "text/plain; version=0.0.4");
    });
    // llama.cpp-shaped slot detail. The Engine has no exposed slot table, so
    // the first `max_concurrency` in-flight requests (FIFO order) count as
    // processing and the rest of the table reads idle; per-slot cache detail
    // is unknown mid-flight and reported as zero.
    server_.Get("/slots", [this](const httplib::Request&, httplib::Response& res) {
        const auto active = metrics_.active_snapshot();
        const bool speculative =
            options_.speculative.backend != ninfer::SpeculativeBackend::None;
        nlohmann::json slots = nlohmann::json::array();
        for (std::uint32_t i = 0; i < options_.max_concurrency; ++i) {
            const bool busy = i < active.size();
            slots.push_back({{"id", i},
                             {"is_processing", busy},
                             {"n_ctx", options_.max_context},
                             {"n_prompt_tokens", busy ? active[i].second : 0},
                             {"n_prompt_tokens_cache", 0},
                             {"speculative", speculative}});
        }
        res.set_content(slots.dump(), "application/json");
    });
    server_.Get("/props", [this](const httplib::Request&, httplib::Response& res) {
        ninfer::SamplingPreset sampling;
        if (service_ != nullptr) {
            const auto defaults = service_->sampling_defaults();
            sampling = defaults.for_mode(options_.enable_thinking ? ninfer::SamplingMode::Thinking
                                                                  : ninfer::SamplingMode::NonThinking);
        }
        const float temp = options_.sampling_overrides.temperature.value_or(sampling.temperature);
        const float top_p = options_.sampling_overrides.top_p.value_or(sampling.top_p);
        const int top_k = options_.sampling_overrides.top_k.value_or(sampling.top_k);
        const std::uint64_t seed = options_.sampling_overrides.seed.value_or(0);
        const bool vision = options_.enable_vision;

        nlohmann::json default_gen = {
            {"n_ctx", options_.max_context},
            {"n_predict", -1},
            {"params", {
                {"n_predict", -1},
                {"max_tokens", options_.default_max_tokens},
                {"temp", temp},
                {"top_p", top_p},
                {"top_k", top_k},
                {"seed", seed}
            }}
        };
        nlohmann::json props = {
            {"default_generation_settings", default_gen},
            {"total_slots", options_.max_concurrency},
            {"model_alias", public_model_id_},
            {"modalities", {
                {"vision", vision},
                {"video", false},
                {"audio", false}
            }},
            {"endpoint_slots", true},
            {"endpoint_props", true},
            {"endpoint_metrics", true},
            {"ui", options_.enable_ui}
        };
        res.set_content(props.dump(), "application/json");
    });

    const auto register_get = [&](const std::string& path, httplib::Server::Handler handler) {
        server_.Get(path, handler);
        server_.Get("/v1" + path, handler);
        server_.Get("/v1/v1" + path, handler);
    };
    const auto register_post = [&](const std::string& path, httplib::Server::Handler handler) {
        server_.Post(path, handler);
        server_.Post("/v1" + path, handler);
        server_.Post("/v1/v1" + path, handler);
    };
    const auto register_delete = [&](const std::string& path, httplib::Server::Handler handler) {
        server_.Delete(path, handler);
        server_.Delete("/v1" + path, handler);
        server_.Delete("/v1/v1" + path, handler);
    };

    register_get("/models", [this](const httplib::Request& req, httplib::Response& res) {
        handle_models(req, res);
    });
    register_get(R"(/models/(.+))", [this](const httplib::Request& req, httplib::Response& res) {
        handle_model(req, res);
    });
    register_post("/chat/completions",
                  [this](const httplib::Request& req, httplib::Response& res) {
                      handle_chat_completions(req, res);
                  });
    register_post("/responses", [this](const httplib::Request& req, httplib::Response& res) {
        handle_responses(req, res);
    });
    register_post("/responses/input_tokens",
                  [this](const httplib::Request& req, httplib::Response& res) {
                      handle_response_input_tokens(req, res);
                  });
    register_post("/responses/compact",
                  [this](const httplib::Request& req, httplib::Response& res) {
                      handle_response_compact(req, res);
                  });
    register_post(R"(/responses/([^/]+)/cancel)",
                  [this](const httplib::Request& req, httplib::Response& res) {
                      handle_response_cancel(req, res);
                  });
    register_get(R"(/responses/([^/]+)/input_items)",
                 [this](const httplib::Request& req, httplib::Response& res) {
                     handle_response_input_items(req, res);
                 });
    register_get(R"(/responses/([^/]+))",
                 [this](const httplib::Request& req, httplib::Response& res) {
                     handle_response_get(req, res);
                 });
    register_delete(R"(/responses/([^/]+))",
                    [this](const httplib::Request& req, httplib::Response& res) {
                        handle_response_delete(req, res);
                    });
    register_post("/messages/count_tokens",
                  [this](const httplib::Request& req, httplib::Response& res) {
                      handle_count_tokens(req, res);
                  });
    register_post("/messages", [this](const httplib::Request& req, httplib::Response& res) {
        handle_messages(req, res);
    });

    if (options_.enable_ui && ninfer_ui_has_assets()) {
        auto serve_asset = [](const std::string& name, httplib::Response& res, const httplib::Request& req) {
            const NinferUiAsset* asset = ninfer_ui_find_asset(name);
            if (!asset) {
                asset = ninfer_ui_find_asset("index.html");
            }
            if (!asset) {
                res.status = 404;
                return;
            }
            res.set_header("ETag", asset->etag);
            const std::string inm = req.get_header_value("If-None-Match");
            if (!inm.empty() && (inm == asset->etag || inm == "W/" + asset->etag)) {
                res.status = 304;
                return;
            }
            res.set_header("Cache-Control", asset->name == "index.html" ? "no-cache" : "max-age=31536000, immutable");
            res.set_content(reinterpret_cast<const char*>(asset->data), asset->size, asset->type.c_str());
        };

        server_.Get("/", [serve_asset](const httplib::Request& req, httplib::Response& res) {
            serve_asset("index.html", res, req);
        });
        server_.Get("/index.html", [serve_asset](const httplib::Request& req, httplib::Response& res) {
            serve_asset("index.html", res, req);
        });
        server_.Get(R"(/([^/].*))", [serve_asset](const httplib::Request& req, httplib::Response& res) {
            std::string path = req.matches[1];
            if (path.rfind("v1/", 0) == 0 || path == "health" || path == "metrics" ||
                path == "slots" || path == "props" || path == "models") {
                res.status = 404;
                return;
            }
            const NinferUiAsset* asset = ninfer_ui_find_asset(path);
            if (asset) {
                serve_asset(path, res, req);
            } else if (path.find('.') == std::string::npos) {
                // SPA client route fallback
                serve_asset("index.html", res, req);
            } else {
                res.status = 404;
            }
        });
    }
}

void HttpServer::handle_models(const httplib::Request&, httplib::Response& res) const {
    res.set_content(make_models_list(public_model_id_, unix_time_now(), options_.max_context,
                                     options_.enable_vision),
                    "application/json");
}

void HttpServer::handle_model(const httplib::Request& req, httplib::Response& res) const {
    const std::string id = req.matches.size() > 1 ? req.matches[1].str() : public_model_id_;
    res.set_content(make_model_object(id.empty() ? public_model_id_ : id, unix_time_now(),
                                      options_.max_context, options_.enable_vision),
                    "application/json");
}

void HttpServer::handle_chat_completions(const httplib::Request& req, httplib::Response& res) {
    nlohmann::json body;
    try {
        body = nlohmann::json::parse(req.body);
    } catch (const std::exception&) {
        ApiError error;
        error.status  = 400;
        error.message = "request body is not valid JSON";
        write_error(res, error);
        return;
    }

    GenerationRequest request;
    PreparedRequest prepared;
    try {
        RequestLimits limits;
        limits.default_max_tokens = options_.default_max_tokens;
        request = parse_chat_completion_request(body, limits);
        if (request.model.empty()) {
            request.model = public_model_id_;
        }
        prepared = service_->prepare(
            request, [&req] { return req.is_connection_alive && !req.is_connection_alive(); });
    } catch (const ApiException& e) {
        write_error(res, e.error());
        return;
    }

    const std::string id       = new_chat_completion_id();
    const std::int64_t created = unix_time_now();
    const std::string model    = request.model;

    const std::uint64_t req_id = ++request_seq_;
    const RequestLogContext log_context =
        make_request_log_context(req_id, "openai_chat_completions", request, prepared);
    log_request_start(log_context);

    if (!request.stream) {
        try {
            const GenerationOutcome outcome = service_->run(prepared, nullptr, [&req] {
                return req.is_connection_alive && !req.is_connection_alive();
            });
            log_request_done(log_context, outcome);
            const CompletionUsage usage     = make_completion_usage(outcome);
            const CompletionTimings timings = make_completion_timings(outcome);
            std::string response_body;
            if (!outcome.tool_calls.empty()) {
                response_body = make_chat_completion_tool_response(
                    id, model, created, outcome.text, outcome.reasoning, outcome.tool_calls, usage,
                    timings);
            } else {
                response_body = make_chat_completion_response(
                    id, model, created, outcome.text, outcome.reasoning,
                    finish_reason_wire(outcome.finish_reason), usage, timings);
            }
            set_owned_content(res, std::move(response_body), prepared.lifetime);
        } catch (const std::exception& e) {
            log_request_error(log_context, e.what());
            throw;
        }
        return;
    }

    auto stream              = std::make_shared<StreamingRequest>(std::move(prepared));
    const bool include_usage = stream->prepared.include_usage;
    const bool tool_capable  = stream->prepared.tool_capable;

    // SSE hints: disable client/proxy caching and reverse-proxy response buffering
    // so tokens flush immediately. Content-Type is set by the chunked provider.
    res.set_header("Cache-Control", "no-cache");
    res.set_header("X-Accel-Buffering", "no");

    res.set_chunked_content_provider(
        "text/event-stream",
        [this, stream, id, created, model, include_usage, tool_capable,
         log_context](std::size_t, httplib::DataSink& sink) -> bool {
            if (stream->started) {
                sink.done();
                return true;
            }
            stream->started = true;
            try {
                write_stream_item(sink, *stream,
                                  make_chat_chunk_role(id, model, created, include_usage));
                const bool timings_per_token = stream->prepared.timings_per_token;
                const int prompt_tokens      = stream->prepared.prompt_tokens;
                std::size_t streamed_tokens  = 0;
                std::chrono::steady_clock::time_point decode_start{};
                bool decode_started          = false;

                const auto get_chunk_timings = [&](std::uint32_t tokens) -> std::optional<CompletionTimings> {
                    if (!timings_per_token) { return std::nullopt; }
                    const auto now = std::chrono::steady_clock::now();
                    if (!decode_started) {
                        decode_start   = now;
                        decode_started = true;
                    }
                    streamed_tokens += tokens;
                    const double elapsed_ms =
                        std::chrono::duration<double, std::milli>(now - decode_start).count();
                    CompletionTimings t;
                    t.prompt_n = prompt_tokens;
                    t.predicted_n = static_cast<int>(streamed_tokens);
                    t.predicted_ms = elapsed_ms;
                    t.predicted_per_second = elapsed_ms > 0.0
                        ? (static_cast<double>(streamed_tokens) / (elapsed_ms / 1000.0))
                        : 0.0;
                    return t;
                };

                auto last_activity = std::chrono::steady_clock::now();
                StreamSink output;
                output.on_content = [&](const std::string& text, std::uint32_t tokens) {
                    last_activity = std::chrono::steady_clock::now();
                    write_stream_item(
                        sink, *stream,
                        make_chat_chunk_content(id, model, created, text, include_usage,
                                                get_chunk_timings(tokens)));
                };
                output.on_reasoning = [&](const std::string& text, std::uint32_t tokens) {
                    last_activity = std::chrono::steady_clock::now();
                    write_stream_item(
                        sink, *stream,
                        make_chat_chunk_reasoning(id, model, created, text, include_usage,
                                                  get_chunk_timings(tokens)));
                };
                output.is_cancelled = [&] {
                    if (stream->cancelled.load(std::memory_order_acquire)) {
                        return true;
                    }
                    if (sink.is_writable && !sink.is_writable()) {
                        return true;
                    }
                    const auto now = std::chrono::steady_clock::now();
                    if (std::chrono::duration_cast<std::chrono::milliseconds>(now - last_activity).count() >= 2000) {
                        last_activity = now;
                        try {
                            write_stream_item(sink, *stream, sse_ping());
                        } catch (const ClientDisconnected&) {
                            return true;
                        }
                    }
                    return false;
                };

                const GenerationOutcome outcome = service_->run(stream->prepared, &output);
                log_request_done(log_context, outcome);
                const CompletionTimings timings = make_completion_timings(outcome);
                const std::string_view remaining = unstreamed_content(outcome);
                if (!outcome.tool_calls.empty()) {
                    if (!remaining.empty()) {
                        write_stream_item(sink, *stream,
                                          make_chat_chunk_content(id, model, created,
                                                                  std::string(remaining),
                                                                  include_usage));
                    }
                    write_stream_item(sink, *stream,
                                      make_chat_chunk_tool_calls(
                                          id, model, created, outcome.tool_calls, include_usage));
                    write_stream_item(
                        sink, *stream,
                        make_chat_chunk_final(id, model, created, "tool_calls", include_usage,
                                              timings));
                } else {
                    if (tool_capable && !remaining.empty()) {
                        write_stream_item(sink, *stream,
                                          make_chat_chunk_content(id, model, created,
                                                                  std::string(remaining),
                                                                  include_usage));
                    }
                    write_stream_item(
                        sink, *stream,
                        make_chat_chunk_final(id, model, created,
                                              finish_reason_wire(outcome.finish_reason),
                                              include_usage, timings));
                }
                if (include_usage) {
                    const CompletionUsage usage = make_completion_usage(outcome);
                    write_stream_item(sink, *stream,
                                      make_chat_chunk_usage(id, model, created, usage, timings));
                }
                write_stream_item(sink, *stream, sse_done());
                sink.done();
                return true;
            } catch (const ClientDisconnected& e) {
                log_request_error(log_context, e.what());
                return false;
            } catch (const ApiException& e) {
                log_request_error(log_context, e.error().message);
                try {
                    write_stream_item(sink, *stream, sse_error_event(e.error()));
                    sink.done();
                    return true;
                } catch (const ClientDisconnected&) { return false; }
            } catch (const std::exception& e) {
                log_request_error(log_context, e.what());
                ApiError error;
                error.status  = 500;
                error.type    = "internal_error";
                error.message = e.what();
                try {
                    write_stream_item(sink, *stream, sse_error_event(error));
                    sink.done();
                    return true;
                } catch (const ClientDisconnected&) { return false; }
            }
        },
        [stream](bool) { stream->cancelled.store(true, std::memory_order_release); });
}

void HttpServer::handle_count_tokens(const httplib::Request& req, httplib::Response& res) {
    nlohmann::json body;
    try {
        body = nlohmann::json::parse(req.body);
    } catch (const std::exception&) {
        ApiError error;
        error.status  = 400;
        error.message = "request body is not valid JSON";
        write_messages_error(res, error);
        return;
    }
    try {
        RequestLimits limits;
        limits.default_max_tokens       = options_.default_max_tokens;
        const GenerationRequest request = parse_messages_request(body, limits);
        const int input_tokens          = service_->count_prompt_tokens(
            request, [&req] { return req.is_connection_alive && !req.is_connection_alive(); });
        res.set_content(make_count_tokens_response(input_tokens), "application/json");
    } catch (const ApiException& e) {
        write_messages_error(res, e.error());
    } catch (const std::exception& e) {
        ApiError error;
        error.status  = 500;
        error.type    = "internal_error";
        error.message = e.what();
        write_messages_error(res, error);
    }
}

void HttpServer::handle_messages(const httplib::Request& req, httplib::Response& res) {
    nlohmann::json body;
    try {
        body = nlohmann::json::parse(req.body);
    } catch (const std::exception&) {
        ApiError error;
        error.status  = 400;
        error.message = "request body is not valid JSON";
        write_messages_error(res, error);
        return;
    }

    GenerationRequest request;
    PreparedRequest prepared;
    try {
        RequestLimits limits;
        limits.default_max_tokens = options_.default_max_tokens;
        // The Anthropic endpoint accepts any `model` string (Claude Code sends real
        // Claude model names) and echoes it back; it never 404s on model id.
        request  = parse_messages_request(body, limits);
        prepared = service_->prepare(
            request, [&req] { return req.is_connection_alive && !req.is_connection_alive(); });
    } catch (const ApiException& e) {
        write_messages_error(res, e.error());
        return;
    } catch (const std::exception& e) {
        ApiError error;
        error.status  = 500;
        error.type    = "internal_error";
        error.message = e.what();
        write_messages_error(res, error);
        return;
    }

    const std::string id    = new_message_id();
    const std::string model = request.model; // echo the requested model
    const int input_tokens  = prepared.prompt_tokens;

    const std::uint64_t req_id = ++request_seq_;
    const RequestLogContext log_context =
        make_request_log_context(req_id, "anthropic_messages", request, prepared);
    log_request_start(log_context);

    if (!request.stream) {
        try {
            const GenerationOutcome outcome = service_->run(prepared, nullptr, [&req] {
                return req.is_connection_alive && !req.is_connection_alive();
            });
            log_request_done(log_context, outcome);
            const CompletionUsage usage{outcome.prompt_tokens, outcome.completion_tokens};
            const char* stop_reason =
                messages_stop_reason(outcome.finish_reason, !outcome.tool_calls.empty());
            set_owned_content(res,
                              make_messages_response(id, model, outcome.text, outcome.reasoning,
                                                     outcome.tool_calls, stop_reason, usage),
                              prepared.lifetime);
        } catch (const ApiException& e) {
            log_request_error(log_context, e.error().message);
            write_messages_error(res, e.error());
        } catch (const std::exception& e) {
            log_request_error(log_context, e.what());
            ApiError error;
            error.status  = 500;
            error.type    = "internal_error";
            error.message = e.what();
            write_messages_error(res, error);
        }
        return;
    }

    auto stream             = std::make_shared<StreamingRequest>(std::move(prepared));
    const bool tool_capable = stream->prepared.tool_capable;

    res.set_header("Cache-Control", "no-cache");
    res.set_header("X-Accel-Buffering", "no");

    res.set_chunked_content_provider(
        "text/event-stream",
        [this, stream, id, model, input_tokens, tool_capable,
         log_context](std::size_t, httplib::DataSink& sink) -> bool {
            if (stream->started) {
                sink.done();
                return true;
            }
            stream->started = true;

            int next_index     = 0;
            bool thinking_open = false;
            int thinking_index = -1;
            bool text_open     = false;
            int text_index     = -1;
            try {
                write_stream_item(sink, *stream, make_message_start(id, model, input_tokens));

                auto last_activity = std::chrono::steady_clock::now();
                StreamSink output;
                output.on_reasoning = [&](const std::string& text, std::uint32_t) {
                    last_activity = std::chrono::steady_clock::now();
                    if (!thinking_open) {
                        thinking_index = next_index++;
                        thinking_open  = true;
                        write_stream_item(sink, *stream,
                                          make_content_block_start_thinking(thinking_index));
                    }
                    write_stream_item(sink, *stream,
                                      make_content_block_delta_thinking(thinking_index, text));
                };
                output.on_content = [&](const std::string& text, std::uint32_t) {
                    last_activity = std::chrono::steady_clock::now();
                    if (thinking_open) {
                        write_stream_item(sink, *stream, make_content_block_stop(thinking_index));
                        thinking_open = false;
                    }
                    if (!text_open) {
                        text_index = next_index++;
                        text_open  = true;
                        write_stream_item(sink, *stream, make_content_block_start_text(text_index));
                    }
                    write_stream_item(sink, *stream,
                                      make_content_block_delta_text(text_index, text));
                };
                output.is_cancelled = [&] {
                    if (stream->cancelled.load(std::memory_order_acquire)) {
                        return true;
                    }
                    if (sink.is_writable && !sink.is_writable()) {
                        return true;
                    }
                    const auto now = std::chrono::steady_clock::now();
                    if (std::chrono::duration_cast<std::chrono::milliseconds>(now - last_activity).count() >= 2000) {
                        last_activity = now;
                        try {
                            write_stream_item(sink, *stream, make_messages_ping());
                        } catch (const ClientDisconnected&) {
                            return true;
                        }
                    }
                    return false;
                };

                const GenerationOutcome outcome = service_->run(stream->prepared, &output);
                log_request_done(log_context, outcome);
                const std::string_view remaining = unstreamed_content(outcome);

                if (thinking_open) {
                    write_stream_item(sink, *stream, make_content_block_stop(thinking_index));
                    thinking_open = false;
                }
                if (text_open) {
                    write_stream_item(sink, *stream, make_content_block_stop(text_index));
                    text_open = false;
                }

                if (tool_capable) {
                    if (!remaining.empty()) {
                        const int idx = next_index++;
                        write_stream_item(sink, *stream, make_content_block_start_text(idx));
                        write_stream_item(
                            sink, *stream,
                            make_content_block_delta_text(idx, std::string(remaining)));
                        write_stream_item(sink, *stream, make_content_block_stop(idx));
                    }
                    for (const ToolCall& call : outcome.tool_calls) {
                        const int idx = next_index++;
                        write_stream_item(sink, *stream,
                                          make_content_block_start_tool_use(idx, call));
                        write_stream_item(
                            sink, *stream,
                            make_content_block_delta_tool_json(idx, call.arguments_json));
                        write_stream_item(sink, *stream, make_content_block_stop(idx));
                    }
                }

                if (next_index == 0) {
                    const int idx = next_index++;
                    write_stream_item(sink, *stream, make_content_block_start_text(idx));
                    write_stream_item(sink, *stream, make_content_block_stop(idx));
                }

                const char* stop_reason =
                    messages_stop_reason(outcome.finish_reason, !outcome.tool_calls.empty());
                write_stream_item(sink, *stream,
                                  make_message_delta(stop_reason, outcome.completion_tokens));
                write_stream_item(sink, *stream, make_message_stop());
                sink.done();
                return true;
            } catch (const ClientDisconnected& e) {
                log_request_error(log_context, e.what());
                return false;
            } catch (const ApiException& e) {
                log_request_error(log_context, e.error().message);
                try {
                    write_stream_item(sink, *stream, messages_sse_error_event(e.error()));
                    sink.done();
                    return true;
                } catch (const ClientDisconnected&) { return false; }
            } catch (const std::exception& e) {
                log_request_error(log_context, e.what());
                ApiError error;
                error.status  = 500;
                error.type    = "internal_error";
                error.message = e.what();
                try {
                    write_stream_item(sink, *stream, messages_sse_error_event(error));
                    sink.done();
                    return true;
                } catch (const ClientDisconnected&) { return false; }
            }
        },
        [stream](bool) { stream->cancelled.store(true, std::memory_order_release); });
}

bool HttpServer::bind() { return server_.bind_to_port(options_.host, options_.port); }

void HttpServer::attach(GenerationService& service) {
    if (service_ != nullptr) {
        throw std::logic_error("HTTP generation service is already attached");
    }
    const ninfer::LoadSummary load = service.load_summary();
    public_model_id_               = resolve_public_model_id(options_, load.model_id);
    service_                       = &service;
    request_jsonl_.write_server_start(options_, service.sampling_defaults(), public_model_id_, load,
                                      service.memory_summary());
}

bool HttpServer::listen() {
    if (service_ == nullptr) { throw std::logic_error("HTTP generation service is not attached"); }
    if (public_model_id_.empty()) {
        throw std::logic_error("HTTP public model id is not resolved");
    }
    if (options_.log_stats_interval_ms != 0) {
        stats_stopping_ = false;
        stats_thread_   = std::thread([this] { run_stats_reporter(); });
    }
    try {
        const bool result = server_.listen_after_bind();
        stop_stats_reporter();
        return result;
    } catch (...) {
        stop_stats_reporter();
        throw;
    }
}

void HttpServer::stop() { server_.stop(); }

} // namespace ninfer::serve
