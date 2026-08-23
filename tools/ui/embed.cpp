#include <algorithm>
#include <cinttypes>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <string>
#include <vector>

namespace {

bool read_file(const std::filesystem::path& path, std::vector<unsigned char>& out) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f.is_open()) {
        return false;
    }
    const auto sz = f.tellg();
    if (sz < 0) {
        return false;
    }
    out.resize(static_cast<std::size_t>(sz));
    f.seekg(0, std::ios::beg);
    if (sz > 0) {
        f.read(reinterpret_cast<char*>(out.data()), sz);
    }
    return f.good() || (sz == 0);
}

bool write_if_different(const std::string& path, const std::string& content) {
    std::ifstream in(path, std::ios::binary);
    if (in.is_open()) {
        std::string existing((std::istreambuf_iterator<char>(in)),
                             std::istreambuf_iterator<char>());
        if (existing == content) {
            return true; // Unchanged
        }
    }
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out.is_open()) {
        fprintf(stderr, "embed: cannot open %s for writing\n", path.c_str());
        return false;
    }
    if (!content.empty()) {
        out.write(content.data(), static_cast<std::streamsize>(content.size()));
    }
    const bool ok = out.good();
    if (ok) {
        printf("embed: wrote output file %s\n", path.c_str());
    }
    return ok;
}

std::uint64_t fnv1a_64(const unsigned char* data, std::size_t size) {
    std::uint64_t h = 14695981039346656037ULL;
    for (std::size_t i = 0; i < size; ++i) {
        h = (h ^ data[i]) * 1099511628211ULL;
    }
    return h;
}

const char* mime_from_ext(const std::string& name) {
    const auto dot = name.rfind('.');
    if (dot == std::string::npos) {
        return "application/octet-stream";
    }
    const std::string ext = name.substr(dot);
    if (ext == ".html" || ext == ".htm") return "text/html; charset=utf-8";
    if (ext == ".js" || ext == ".mjs")   return "application/javascript";
    if (ext == ".css")                   return "text/css; charset=utf-8";
    if (ext == ".json")                  return "application/json";
    if (ext == ".webmanifest")           return "application/manifest+json";
    if (ext == ".svg")                   return "image/svg+xml";
    if (ext == ".png")                   return "image/png";
    if (ext == ".jpg" || ext == ".jpeg") return "image/jpeg";
    if (ext == ".webp")                  return "image/webp";
    if (ext == ".gif")                   return "image/gif";
    if (ext == ".ico")                   return "image/x-icon";
    if (ext == ".woff2")                 return "font/woff2";
    if (ext == ".woff")                  return "font/woff";
    if (ext == ".ttf")                   return "font/ttf";
    if (ext == ".wasm")                  return "application/wasm";
    if (ext == ".txt")                   return "text/plain; charset=utf-8";
    return "application/octet-stream";
}

void append_bytes_hex(std::string& out, const std::vector<unsigned char>& bytes) {
    char buf[16];
    for (std::size_t i = 0; i < bytes.size(); ++i) {
        if (i % 16 == 0) {
            out += "\n    ";
        }
        snprintf(buf, sizeof(buf), "0x%02x, ", bytes[i]);
        out += buf;
    }
    if (!bytes.empty()) {
        out += "\n";
    }
}

std::string fmt(const char* pattern, ...) {
    char tmp[512];
    va_list ap;
    va_start(ap, pattern);
    const int n = vsnprintf(tmp, sizeof(tmp), pattern, ap);
    va_end(ap);
    return (n > 0) ? std::string(tmp, static_cast<std::size_t>(n)) : std::string();
}

struct AssetEntry {
    std::string name;
    std::filesystem::path path;
};

// Locate the directory that directly contains index.html (in case the tar contains a top-level folder)
std::filesystem::path find_asset_root(const std::filesystem::path& base_dir) {
    if (!std::filesystem::exists(base_dir)) {
        return base_dir;
    }
    if (std::filesystem::exists(base_dir / "index.html")) {
        return base_dir;
    }
    std::error_code ec;
    for (const auto& entry : std::filesystem::recursive_directory_iterator(base_dir, ec)) {
        if (entry.is_regular_file() && entry.path().filename() == "index.html") {
            return entry.path().parent_path();
        }
    }
    return base_dir;
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 3 || argc > 4) {
        fprintf(stderr, "usage: %s <out_cpp> <out_h> [<asset_dir>]\n", argv[0]);
        return 1;
    }

    const std::string out_cpp = argv[1];
    const std::string out_h   = argv[2];
    const std::string raw_dir = (argc >= 4) ? argv[3] : std::string();

    std::vector<AssetEntry> assets;
    if (!raw_dir.empty()) {
        const std::filesystem::path asset_root = find_asset_root(raw_dir);
        if (std::filesystem::exists(asset_root)) {
            std::error_code ec;
            for (const auto& entry : std::filesystem::recursive_directory_iterator(asset_root, ec)) {
                if (!entry.is_regular_file()) {
                    continue;
                }
                std::string rel_name = entry.path().lexically_relative(asset_root).generic_string();
                assets.push_back({std::move(rel_name), entry.path()});
            }
            std::sort(assets.begin(), assets.end(),
                      [](const AssetEntry& a, const AssetEntry& b) { return a.name < b.name; });
        }
    }

    const int n_assets = static_cast<int>(assets.size());

    // Generate Header (ui.h)
    std::string h;
    h += "#pragma once\n\n";
    h += "#include <array>\n";
    h += "#include <cstddef>\n";
    h += "#include <string>\n\n";
    h += "namespace ninfer::serve {\n\n";
    h += "struct NinferUiAsset {\n";
    h += "    std::string name;\n";
    h += "    const unsigned char* data;\n";
    h += "    std::size_t size;\n";
    h += "    std::string etag;\n";
    h += "    std::string type;\n";
    h += "};\n\n";
    h += "[[nodiscard]] const NinferUiAsset* ninfer_ui_find_asset(const std::string& name);\n";
    h += "[[nodiscard]] bool ninfer_ui_has_assets();\n";
    h += fmt("[[nodiscard]] const std::array<NinferUiAsset, %d>& ninfer_ui_get_assets();\n\n", n_assets);
    h += "} // namespace ninfer::serve\n";

    // Generate Source (ui.cpp)
    std::string cpp;
    cpp += "#include \"ui.h\"\n\n";
    cpp += "#include <algorithm>\n\n";
    cpp += "namespace ninfer::serve {\n";
    cpp += "namespace {\n\n";

    if (n_assets > 0) {
        for (int i = 0; i < n_assets; ++i) {
            std::vector<unsigned char> bytes;
            if (!read_file(assets[i].path, bytes) || bytes.empty()) {
                fprintf(stderr, "embed: failed reading asset: %s\n", assets[i].path.generic_string().c_str());
                return 1;
            }
            cpp += fmt("const unsigned char asset_%d_data[] = {", i);
            append_bytes_hex(cpp, bytes);
            cpp += "};\n";
            cpp += fmt("const std::size_t asset_%d_size = %zu;\n", i, bytes.size());
            const std::uint64_t hash = fnv1a_64(bytes.data(), bytes.size());
            cpp += fmt("const char asset_%d_etag[] = \"\\\"0x%016" PRIx64 "\\\"\";\n\n", i, hash);
        }

        cpp += fmt("const std::array<NinferUiAsset, %d> g_assets = {{\n", n_assets);
        for (int i = 0; i < n_assets; ++i) {
            const std::string& name = assets[i].name;
            cpp += fmt("    { \"%s\", asset_%d_data, asset_%d_size, asset_%d_etag, \"%s\" },\n",
                       name.c_str(), i, i, i, mime_from_ext(name));
        }
        cpp += "}};\n\n";
        cpp += "} // namespace\n\n";

        cpp += "const NinferUiAsset* ninfer_ui_find_asset(const std::string& name) {\n";
        cpp += "    for (const auto& a : g_assets) {\n";
        cpp += "        if (a.name == name) {\n";
        cpp += "            return &a;\n";
        cpp += "        }\n";
        cpp += "    }\n";
        cpp += "    return nullptr;\n";
        cpp += "}\n\n";

        cpp += "bool ninfer_ui_has_assets() {\n";
        cpp += "    return true;\n";
        cpp += "}\n\n";

        cpp += fmt("const std::array<NinferUiAsset, %d>& ninfer_ui_get_assets() {\n", n_assets);
        cpp += "    return g_assets;\n";
        cpp += "}\n\n";
    } else {
        cpp += "const std::array<NinferUiAsset, 0> g_empty_assets{};\n\n";
        cpp += "} // namespace\n\n";
        cpp += "const NinferUiAsset* ninfer_ui_find_asset(const std::string&) {\n";
        cpp += "    return nullptr;\n";
        cpp += "}\n\n";
        cpp += "bool ninfer_ui_has_assets() {\n";
        cpp += "    return false;\n";
        cpp += "}\n\n";
        cpp += "const std::array<NinferUiAsset, 0>& ninfer_ui_get_assets() {\n";
        cpp += "    return g_empty_assets;\n";
        cpp += "}\n\n";
    }

    cpp += "} // namespace ninfer::serve\n";

    bool ok = true;
    ok = write_if_different(out_h, h) && ok;
    ok = write_if_different(out_cpp, cpp) && ok;
    return ok ? 0 : 1;
}
