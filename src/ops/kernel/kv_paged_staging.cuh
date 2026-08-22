#pragma once

#include <cuda_runtime.h>
#include <cstdint>

namespace ninfer::ops {

/**
 * Gather and scatter kernels for paged KV cache staging.
 * Staging layout is strictly Page-Major (each page contains all of its planes contiguously),
 * enabling independent content-addressed page deduplication and direct DMA streaming.
 */

struct PagedKVPlaneDescriptor {
    void* plane_base;
    uint32_t num_heads;
    uint32_t page_head_uint4s;           // page_head_bytes / 16
    uint32_t head_stride_uint4s;         // head_stride_bytes / 16
    uint32_t page_stride_uint4s;         // page_stride_bytes / 16
    uint32_t plane_page_offset_uint4s;   // offset of this plane within a single page
    uint32_t single_page_uint4s;         // total uint4s for one full page across all planes
};

__global__ void gather_paged_kv_kernel(
    const PagedKVPlaneDescriptor* __restrict__ planes,
    uint32_t num_planes,
    const int32_t* __restrict__ d_page_ids,
    uint32_t num_pages,
    uint4* __restrict__ dst_staging) 
{
    const uint32_t in_head_offset = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t head           = blockIdx.y;
    const uint32_t v_page         = blockIdx.z;

    if (v_page >= num_pages) { return; }
    const int32_t physical_page = d_page_ids[v_page];
    if (physical_page < 0) { return; }

    for (uint32_t p = 0; p < num_planes; ++p) {
        const PagedKVPlaneDescriptor plane = planes[p];

        if (head < plane.num_heads && in_head_offset < plane.page_head_uint4s) {
            const auto* src_base = static_cast<const uint4*>(plane.plane_base);
            const uint32_t src_idx = head * plane.head_stride_uint4s + 
                                     static_cast<uint32_t>(physical_page) * plane.page_stride_uint4s + 
                                     in_head_offset;
            const uint32_t dst_idx = v_page * plane.single_page_uint4s + 
                                     plane.plane_page_offset_uint4s + 
                                     head * plane.page_head_uint4s + 
                                     in_head_offset;
            dst_staging[dst_idx] = src_base[src_idx];
        }
    }
}

__global__ void scatter_paged_kv_kernel(
    const PagedKVPlaneDescriptor* __restrict__ planes,
    uint32_t num_planes,
    const int32_t* __restrict__ d_page_ids,
    uint32_t num_pages,
    const uint4* __restrict__ src_staging) 
{
    const uint32_t in_head_offset = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t head           = blockIdx.y;
    const uint32_t v_page         = blockIdx.z;

    if (v_page >= num_pages) { return; }
    const int32_t physical_page = d_page_ids[v_page];
    if (physical_page < 0) { return; }

    for (uint32_t p = 0; p < num_planes; ++p) {
        const PagedKVPlaneDescriptor plane = planes[p];

        if (head < plane.num_heads && in_head_offset < plane.page_head_uint4s) {
            auto* dst_base = static_cast<uint4*>(plane.plane_base);
            const uint32_t src_idx = v_page * plane.single_page_uint4s + 
                                     plane.plane_page_offset_uint4s + 
                                     head * plane.page_head_uint4s + 
                                     in_head_offset;
            const uint32_t dst_idx = head * plane.head_stride_uint4s + 
                                     static_cast<uint32_t>(physical_page) * plane.page_stride_uint4s + 
                                     in_head_offset;
            dst_base[dst_idx] = src_staging[src_idx];
        }
    }
}

} // namespace ninfer::ops
