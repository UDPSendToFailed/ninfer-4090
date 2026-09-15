#pragma once

// Test-only seams into DiskStateCache. They exist only when the including target defines
// NINFER_DISK_STATE_CACHE_TEST_SEAMS; production builds have neither the declarations nor
// the call sites.
#if defined(NINFER_DISK_STATE_CACHE_TEST_SEAMS)

#include <cstdint>
#include <functional>

namespace ninfer::test_seams {

enum class DiskLoadPoint : std::uint8_t {
    AfterResolvePages, // resolve_pages returned true
    AfterPoolOpen,     // the host load opened the page pool file
};

// Installs `hook`, called on the loading thread at each DiskLoadPoint with no cache or journal
// lock held, so it may call DiskStateCache::compact_pool synchronously. `direct_storage` tells
// the DirectStorage load from the host load. An empty hook removes it.
void set_disk_load_hook(std::function<void(DiskLoadPoint point, bool direct_storage)> hook);

} // namespace ninfer::test_seams

#endif
