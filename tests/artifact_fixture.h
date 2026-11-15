#pragma once

#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#ifdef _WIN32
#include <process.h>
#else
#include <unistd.h>
#endif

namespace ninfer::test::artifact_fixture {

using Json = nlohmann::json;

inline constexpr std::array<std::uint8_t, 8> kV2Magic = {
    'N', 'I', 'N', 'F', 'E', 'R', 0, 2,
};
inline constexpr std::array<std::uint8_t, 8> kMagic = {
    'N', 'I', 'N', 'F', 'E', 'R', 0, 3,
};

inline std::uint64_t align_up(std::uint64_t value, std::uint64_t alignment) {
    return (value + alignment - 1) / alignment * alignment;
}

inline void write_u64_le(std::byte* output, std::uint64_t value) {
    for (unsigned i = 0; i < 8; ++i) { output[i] = std::byte((value >> (i * 8)) & 0xff); }
}

struct TemporaryArtifact {
    std::filesystem::path path;

    ~TemporaryArtifact() {
        std::error_code ignored;
        std::filesystem::remove(path, ignored);
    }
};

inline TemporaryArtifact write_fixture(const Json& directory, std::string_view suffix,
                                       const std::array<std::uint8_t, 8>& magic = kMagic) {
    const std::string json         = directory.dump();
    const auto payload_offset      = align_up(32 + json.size(), 4096);
    const auto nonnegative_integer = [](const Json& value) {
        return value.is_number_unsigned() ||
               (value.is_number_integer() && value.get<std::int64_t>() >= 0);
    };
    std::uint64_t payload_bytes = 0;
    if (directory.contains("files") && directory.at("files").is_array() &&
        !directory.at("files").empty() && directory.at("files")[0].contains("payload_bytes")) {
        payload_bytes = directory.at("files")[0].at("payload_bytes").get<std::uint64_t>();
    } else if (directory.contains("objects") && directory.at("objects").is_array()) {
        for (const auto& object : directory.at("objects")) {
            if (object.contains("offset") && object.contains("bytes") &&
                nonnegative_integer(object.at("offset")) &&
                nonnegative_integer(object.at("bytes"))) {
                payload_bytes =
                    std::max(payload_bytes, object.at("offset").get<std::uint64_t>() +
                                                object.at("bytes").get<std::uint64_t>());
            }
        }
    }

    std::vector<std::byte> file(payload_offset + payload_bytes, std::byte{0});
    for (std::size_t i = 0; i < magic.size(); ++i) { file[i] = std::byte{magic[i]}; }
    write_u64_le(file.data() + 8, json.size());
    for (std::size_t i = 0; i < 16; ++i) { file[16 + i] = std::byte{static_cast<std::uint8_t>(i + 1)}; }
    std::memcpy(file.data() + 32, json.data(), json.size());

    std::uint8_t marker = 1;
    if (directory.contains("objects") && directory.at("objects").is_array()) {
        for (const auto& object : directory.at("objects")) {
            if (object.contains("offset") && object.contains("bytes") &&
                nonnegative_integer(object.at("offset")) &&
                nonnegative_integer(object.at("bytes"))) {
                const auto offset = object.at("offset").get<std::uint64_t>();
                const auto bytes  = object.at("bytes").get<std::uint64_t>();
                if (offset + bytes <= payload_bytes) {
                    std::fill_n(file.data() + payload_offset + offset, bytes, std::byte{marker++});
                }
            }
        }
    }

    static std::atomic<std::uint64_t> counter{0};
#ifdef _WIN32
    const auto pid = _getpid();
#else
    const auto pid = getpid();
#endif
    auto path = std::filesystem::temp_directory_path() /
                ("ninfer_artifact_" + std::string(suffix) + "_" + std::to_string(pid) + "_" +
                 std::to_string(counter.fetch_add(1)) + ".ninfer");
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    output.write(reinterpret_cast<const char*>(file.data()),
                 static_cast<std::streamsize>(file.size()));
    if (!output) { throw std::runtime_error("failed to write artifact fixture"); }
    return {std::move(path)};
}

} // namespace ninfer::test::artifact_fixture
