#include "attention_reference.h"
#include "ninfer/ops/gqa_attention.h"

using namespace ninfer;
using namespace ninfer::test;
using namespace ninfer::test::attention_reference;

namespace {
int exercise(Format format, int tokens, int keys, int repeats) {
    EncodedCache host(format, keys);
    auto dk = to_device(host.k), dv = to_device(host.v);
    auto dks = to_device(host.ks), dvs = to_device(host.vs), dt = to_device(host.table);
    const bool packed = format == Format::Rk4v4E8;
    PagedKVLayerView cache;
    cache.k_pages = Tensor(dk.p, packed ? DType::U8 : DType::I8, {host.code_dim, 64, kv_heads, host.pages});
    cache.v_pages = Tensor(dv.p, packed ? DType::U8 : DType::I8, {host.code_dim, 64, kv_heads, host.pages});
    cache.k_scale_pages = Tensor(dks.p, DType::FP16, {4, 64, kv_heads, host.pages});
    cache.v_scale_pages = Tensor(dvs.p, DType::FP16, {4, 64, kv_heads, host.pages});
    cache.block_table = Tensor(dt.p, DType::I32, {host.pages});
    cache.head_dim = head_dim; cache.num_kv_heads = kv_heads;
    cache.dtype = DType::I8; cache.quant_group = group;
    cache.packed_k = cache.packed_v = cache.rotate_k = cache.rotate_v = cache.e8_lattice = packed;
    const auto q = queries(tokens);
    std::vector<std::int32_t> positions(tokens);
    std::iota(positions.begin(), positions.end(), keys - tokens);
    const auto expected = oracle(host, q, positions);
    auto dq = to_device(q), dp = to_device(positions);
    DeviceBuffer output(q.size() * 2);
    Tensor tq(dq.p, DType::BF16, {head_dim, q_heads, tokens});
    Tensor tp(dp.p, DType::I32, {tokens});
    Tensor tout(output.p, DType::BF16, {head_dim, q_heads, tokens});
    const ops::GqaExecutionEnvelope envelope{static_cast<std::uint32_t>(keys), static_cast<std::uint32_t>(keys)};
    const auto bytes = ops::gqa_attention_workspace_capacity_bytes(q_heads, DType::I8, envelope, 1, tokens, tokens);
    DeviceBuffer scratch(std::max<std::size_t>(bytes, 256));
    WorkspaceArena workspace({scratch.p, scratch.bytes});
    std::vector<std::uint16_t> first, actual(q.size());
    int failures = 0;
    const auto label = std::string(packed ? "rk4v4-e8" : "int8") + " T=" +
        std::to_string(tokens) + " keys=" + std::to_string(keys);
    for (int repeat = 0; repeat < repeats; ++repeat) {
        ops::gqa_attention_cached(tq, tp, 0.0625f, cache, envelope, workspace, tout, nullptr);
        cuda_synchronize();
        output.copy_to_host(actual.data(), output.bytes);
        if (!repeat) { first = actual; }
        else if (first != actual) {
            std::cerr << label << " bitwise repeat mismatch at repeat=" << repeat << '\n';
            ++failures; break;
        }
    }
    std::vector<double> got(actual.size());
    std::transform(actual.begin(), actual.end(), got.begin(), [](auto bits) { return double(bf16_to_f32(bits)); });
    // INT8 uses the existing registered Q8 criterion. Rotated packed output has
    // an additional BF16 boundary before inverse H64, allowing two BF16 roundings.
    const ReductionCriterion criterion = packed ? ReductionCriterion{6e-3, 2e-3, 4.4e-3} :
                                                  ReductionCriterion{3.15e-3, 1.1e-3, 2.2e-3};
    failures += verify_reduction(label.c_str(), got, expected, criterion);
    std::cout << (failures ? "FAIL " : "PASS ") << label << " repeats=" << repeats << '\n';
    return failures;
}
}
int main(int argc, char** argv) try {
    if (cuda_unavailable()) { return 77; }
    const int repeats = argc == 2 && std::string_view(argv[1]) == "--sanitizer" ? 2 : 64;
    int failures = 0;
    for (auto format : {Format::Int8, Format::Rk4v4E8}) {
        for (int keys : {512, 8199}) {
            for (int tokens : {1, 2, 4}) { failures += exercise(format, tokens, keys, repeats); }
        }
    }
    return failures ? 1 : 0;
} catch (const std::exception& error) {
    std::cerr << "FAIL small-T synchronization regression: " << error.what() << '\n'; return 1;
}
