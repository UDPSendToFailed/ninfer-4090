#pragma once

#include <cstdint>
#include <optional>
#include <span>

namespace ninfer::targets::qwen3_6::detail {

struct TurnCheckpointMatch {
    std::uint32_t frontier = 0;
    bool is_rolling        = false;
};

/**
 * Selects the longest resident checkpoint strictly shorter than `prompt_tokens` whose prefix
 * matches the incoming prompt and whose KV/MTP state is valid at that frontier.
 * Prefers the VRAM `turn_checkpoint` on an exact frontier tie to avoid host-to-device copies.
 */
template <class PrefixMatchesFn>
[[nodiscard]] std::optional<TurnCheckpointMatch>
select_best_turn_checkpoint(bool has_vram_checkpoint, std::uint32_t vram_frontier,
                            std::span<const std::uint32_t> rolling_frontiers,
                            std::uint32_t prompt_tokens, std::uint32_t text_kv_valid,
                            bool require_mtp, bool mtp_cache_available, std::uint32_t mtp_kv_valid,
                            bool allow_rolling, PrefixMatchesFn&& prefix_matches_fn) {
    std::optional<TurnCheckpointMatch> best;
    const auto consider = [&](std::uint32_t frontier, bool is_rolling) {
        if (frontier == 0 || frontier >= prompt_tokens || frontier > text_kv_valid) { return; }
        if (require_mtp && (!mtp_cache_available || mtp_kv_valid < frontier - 1U)) { return; }
        if (best && frontier <= best->frontier) { return; }
        if (!prefix_matches_fn(frontier)) { return; }
        best = TurnCheckpointMatch{.frontier = frontier, .is_rolling = is_rolling};
    };

    if (has_vram_checkpoint) { consider(vram_frontier, false); }
    if (allow_rolling) {
        for (auto it = rolling_frontiers.rbegin(); it != rolling_frontiers.rend(); ++it) {
            consider(*it, true);
        }
    }
    return best;
}

} // namespace ninfer::targets::qwen3_6::detail
