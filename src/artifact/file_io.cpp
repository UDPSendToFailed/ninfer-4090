#include "artifact/file_io.h"

#include "artifact/framing.h"
#include "artifact/schema.h"

#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <string>
#include <utility>

namespace ninfer::artifact {
namespace {

[[noreturn]] void fail_win32(const std::filesystem::path& path, const char* operation,
                             DWORD error) {
    char* msg       = nullptr;
    const DWORD len = ::FormatMessageA(
        FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS,
        nullptr, error, MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT), reinterpret_cast<LPSTR>(&msg), 0,
        nullptr);
    std::string detail = (len && msg) ? std::string(msg, len) : ("error " + std::to_string(error));
    if (msg) { ::LocalFree(msg); }
    while (!detail.empty() && (detail.back() == '\r' || detail.back() == '\n')) {
        detail.pop_back();
    }
    throw ArtifactError(path.string() + ": " + operation + ": " + detail);
}

HANDLE open_win32(const std::filesystem::path& path, DWORD flags) {
    HANDLE h = ::CreateFileW(path.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING,
                             flags, nullptr);
    if (h == INVALID_HANDLE_VALUE) { fail_win32(path, "CreateFileW", ::GetLastError()); }
    return h;
}

DWORD read_at(HANDLE handle, std::uint64_t offset, void* destination, DWORD bytes,
              const std::filesystem::path& path, const char* op) {
    OVERLAPPED ov{};
    ov.Offset          = static_cast<DWORD>(offset & 0xFFFFFFFFULL);
    ov.OffsetHigh      = static_cast<DWORD>(offset >> 32);
    DWORD bytes_read   = 0;
    const BOOL ok      = ::ReadFile(handle, destination, bytes, &bytes_read, &ov);
    if (!ok) {
        const DWORD err = ::GetLastError();
        if (err == ERROR_HANDLE_EOF) { return bytes_read; }
        fail_win32(path, op, err);
    }
    return bytes_read;
}

} // namespace

InputFile::InputFile(std::filesystem::path path) : path_(std::move(path)) {
    HANDLE h = open_win32(path_, FILE_ATTRIBUTE_NORMAL);
    if (::GetFileType(h) != FILE_TYPE_DISK) {
        ::CloseHandle(h);
        throw ArtifactError(path_.string() + ": expected a regular file");
    }
    LARGE_INTEGER size{};
    if (!::GetFileSizeEx(h, &size)) {
        const DWORD err = ::GetLastError();
        ::CloseHandle(h);
        fail_win32(path_, "GetFileSizeEx", err);
    }
    if (size.QuadPart < 0) {
        ::CloseHandle(h);
        throw ArtifactError(path_.string() + ": negative file size");
    }
    handle_ = h;
    bytes_  = static_cast<std::uint64_t>(size.QuadPart);
}

InputFile::~InputFile() {
    if (direct_handle_) { ::CloseHandle(static_cast<HANDLE>(direct_handle_)); }
    if (handle_) { ::CloseHandle(static_cast<HANDLE>(handle_)); }
}

void InputFile::read_exact(std::uint64_t offset, std::span<std::byte> destination) const {
    if (offset > bytes_ || destination.size() > bytes_ - offset) {
        throw ArtifactError(path_.string() + ": read exceeds file length");
    }
    auto* handle = static_cast<HANDLE>(handle_);
    while (!destination.empty()) {
        const auto count =
            static_cast<DWORD>(std::min<std::size_t>(destination.size(), 64ULL * 1024 * 1024));
        const DWORD got = read_at(handle, offset, destination.data(), count, path_, "ReadFile");
        if (got == 0) { throw ArtifactError(path_.string() + ": unexpected EOF"); }
        offset += static_cast<std::uint64_t>(got);
        destination = destination.subspan(static_cast<std::size_t>(got));
    }
}

std::size_t InputFile::read_direct(std::uint64_t offset, std::span<std::byte> destination) const {
    if (offset % kPayloadAlignment || destination.size() % kPayloadAlignment ||
        reinterpret_cast<std::uintptr_t>(destination.data()) % kPayloadAlignment ||
        destination.size() > static_cast<std::size_t>(std::numeric_limits<DWORD>::max())) {
        throw ArtifactError(path_.string() + ": unaligned or oversized direct read");
    }
    if (destination.empty()) { return 0; }
    if (!direct_handle_) {
        direct_handle_ = open_win32(path_, FILE_ATTRIBUTE_NORMAL | FILE_FLAG_NO_BUFFERING);
    }
    const DWORD got = read_at(static_cast<HANDLE>(direct_handle_), offset, destination.data(),
                              static_cast<DWORD>(destination.size()), path_, "direct ReadFile");
    return static_cast<std::size_t>(got);
}

} // namespace ninfer::artifact
