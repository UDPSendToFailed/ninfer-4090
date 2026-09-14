#pragma once

#include <algorithm>
#include <cstdint>
#include <span>

namespace ninfer::targets::qwen3_6::detail {

// A planned disk restore whose snapshot cannot be loaded (missing or corrupt journal page,
// checksum or layout mismatch, DirectStorage unavailable) must not prefill from the matched
// base over unrestored KV and GDN state. The request is rewritten into the FullReset shape the
// planner would have produced without a disk match. Turn checkpoint action stays Drop, which the
// planner always assigns to disk restores and which is valid for FullReset.
template <class Plan>
void degrade_disk_restore_to_full_reset(Plan& plan, bool mtp_backend) {
    using Reuse  = decltype(plan.reuse);
    using Bridge = decltype(plan.mtp_bridge);
    plan.reuse      = Reuse::FullReset;
    plan.reuse_base = 0;
    plan.disk_snapshot_path.clear();
    plan.summary.reusable_prompt_tokens = 0;
    if (mtp_backend) {
        plan.prepare_mtp = true;
        plan.mtp_bridge  = Bridge::None;
    }
}

// The vision plan only covers media spans past the reuse base, so a fallback from a base that
// skipped media cannot be prefilled without replanning.
[[nodiscard]] inline bool prefix_has_visual_tokens(std::span<const std::uint8_t> token_types,
                                                   std::uint32_t base) noexcept {
    const auto end = token_types.begin() +
                     static_cast<std::ptrdiff_t>(std::min<std::size_t>(base, token_types.size()));
    return std::any_of(token_types.begin(), end, [](std::uint8_t type) { return type != 0; });
}

} // namespace ninfer::targets::qwen3_6::detail
