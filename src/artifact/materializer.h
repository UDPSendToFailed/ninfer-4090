#pragma once

#include "artifact/schema.h"
#include "core/arena.h"
#include "core/device.h"
#include "core/weight_view.h"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <string_view>
#include <vector>

namespace ninfer::artifact {

class Reader;

struct LoadProgress {
    std::function<void(std::string_view, std::uint64_t, std::uint64_t)> callback;
};

struct DevicePlacement {
    ObjectHandle object;
    std::uint64_t offset    = 0;
    std::uint64_t bytes     = 0;
    std::uint64_t alignment = 256;
};

struct HostPlacement {
    ObjectHandle object;
    // Already-read resources move into final storage without invalidating their byte views.
    std::vector<std::byte> data;
};

struct MaterializationPlan {
    const Reader* source                = nullptr;
    std::size_t object_count            = 0;
    std::uint64_t device_capacity_bytes = 0;
    std::uint64_t prior_read_bytes      = 0;
    std::uint64_t owned_value_bytes     = 0;
    std::vector<DevicePlacement> device_objects;
    std::vector<HostPlacement> host_objects;
};

struct MaterializationStats {
    std::uint64_t file_bytes            = 0; // Declared container file set, including framing.
    std::uint64_t read_bytes            = 0; // Actual payload reads, including direct-I/O alignment.
    std::uint64_t h2d_bytes             = 0;
    std::uint64_t device_capacity_bytes = 0;
    std::uint64_t retained_host_bytes   = 0;
    std::uint64_t owned_value_bytes     = 0;
    std::uint64_t peak_staging_bytes    = 0;
    std::size_t device_object_count     = 0;
    std::size_t host_object_count       = 0;
    double upload_seconds               = 0.0;
};

class MaterializedArtifact {
public:
    MaterializedArtifact()                                           = default;
    ~MaterializedArtifact()                                          = default;
    MaterializedArtifact(MaterializedArtifact&&) noexcept            = default;
    MaterializedArtifact& operator=(MaterializedArtifact&&) noexcept = default;
    MaterializedArtifact(const MaterializedArtifact&)                = delete;
    MaterializedArtifact& operator=(const MaterializedArtifact&)     = delete;

    [[nodiscard]] const WeightParent& device_parent(ObjectHandle handle) const;
    [[nodiscard]] const WeightParent& host_parent(ObjectHandle handle) const;
    [[nodiscard]] std::span<const std::byte> host_bytes(ObjectHandle handle) const;
    [[nodiscard]] std::vector<std::byte> take_host_bytes(ObjectHandle handle);
    [[nodiscard]] bool has_device(ObjectHandle handle) const noexcept;

    [[nodiscard]] const MaterializationStats& stats() const noexcept { return stats_; }

    DeviceArena& device_arena();

private:
    friend MaterializedArtifact materialize(const Reader&, MaterializationPlan, DeviceContext&,
                                            LoadProgress*);

    struct ObjectStorage {
        std::optional<WeightParent> device;
        std::optional<WeightParent> host;
        std::vector<std::byte> host_data;
    };

    std::unique_ptr<DeviceArena> arena_;
    std::vector<ObjectStorage> objects_;
    MaterializationStats stats_;
};

[[nodiscard]] MaterializedArtifact materialize(const Reader& reader, MaterializationPlan plan,
                                               DeviceContext& device,
                                               LoadProgress* progress = nullptr);

} // namespace ninfer::artifact
