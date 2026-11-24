// Stage 3C: exact-source MXFP8 GEMV CUDA alternatives for A/B performance.
// Coalesced uint32 packed E4M3 loads and/or four warps per row (Split-K).
// Input: row-major E4M3FN codes [N,K], UE8M0 [N,K/32]. No conversion cache.
// FP32 accumulation. Tested separately from Stage 3B warp-row baseline.

#include "mxfp8_gemv_ab.h"

#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>

namespace ninfer::models::inkling2 {
namespace {
constexpr int kWarp = 32;
constexpr int kWarpsPerBlock = 8;
constexpr int kSplitWarps = 4;
constexpr int kRowsPerSplitBlock = kWarpsPerBlock / kSplitWarps;

__device__ __forceinline__ float fp8_to_float(std::uint8_t word) {
    if ((word & 0x7fu) == 0x7fu) return __uint_as_float(0x7fc00000u);
    const unsigned sign = static_cast<unsigned>(word & 0x80u) << 24;
    const unsigned exp = (word >> 3) & 15u;
    const unsigned mantissa = word & 7u;
    if (exp == 0u) {
        const float magnitude = static_cast<float>(mantissa) * 0x1p-9f;
        return __uint_as_float(__float_as_uint(magnitude) | sign);
    }
    return __uint_as_float(sign | ((exp + 120u) << 23) | (mantissa << 20));
}

__device__ __forceinline__ float ue8m0_to_float(std::uint8_t word) {
    if (word == 0xffu) return __uint_as_float(0x7fc00000u);
    // Crucial: exponent byte 0 means 2^-127, not a zero scale.
    return __uint_as_float(word == 0u ? 0x00400000u :
                           (static_cast<unsigned>(word) << 23));
}

__device__ __forceinline__ float warp_reduce_sum(float value) {
    // Entire warp participates in every caller, including zero-work warps.
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        value += __shfl_down_sync(0xffffffffu, value, offset);
    return value;
}

// Each lane loads four adjacent code bytes as a single 32-bit word and the
// matching float4 input. Four 32-element G32 groups are covered by one warp.
// K32 tail groups are predicated, with all lanes kept for the warp shuffle.
__device__ __forceinline__ float vector4_lane_sum(
    const std::uint8_t* codes,
    const std::uint8_t* scales,
    const float* x,
    std::size_t row_code_base,
    std::size_t row_scale_base,
    int groups,
    int first_tile,
    int tile_stride,
    int lane) {
    float sum = 0.f;
    const int lane_group = lane >> 3;            // 8 lanes per G32 group
    const int in_group_col = (lane & 7) * 4;    // four codes per lane
    const auto* packed_codes = reinterpret_cast<const std::uint32_t*>(codes);
    const auto* packed_x = reinterpret_cast<const float4*>(x);
    const int tiles = (groups + 3) / 4;
    for (int tile = first_tile; tile < tiles; tile += tile_stride) {
        const int group = tile * 4 + lane_group;
        if (group >= groups) continue;
        const std::size_t col = static_cast<std::size_t>(group) * 32 + in_group_col;
        const std::uint32_t words = packed_codes[(row_code_base + col) / 4];
        const float4 in = packed_x[col / 4];
        const float scale = ue8m0_to_float(scales[row_scale_base + group]);
        sum = fmaf(fp8_to_float(static_cast<std::uint8_t>(words)) * scale, in.x, sum);
        sum = fmaf(fp8_to_float(static_cast<std::uint8_t>(words >> 8)) * scale, in.y, sum);
        sum = fmaf(fp8_to_float(static_cast<std::uint8_t>(words >> 16)) * scale, in.z, sum);
        sum = fmaf(fp8_to_float(static_cast<std::uint8_t>(words >> 24)) * scale, in.w, sum);
    }
    return sum;
}

__global__ void gemv_warp_vector4(
    const std::uint8_t* __restrict__ codes,
    const std::uint8_t* __restrict__ scales,
    const float* __restrict__ x,
    float* __restrict__ y,
    int rows,
    int cols) {
    const int row = static_cast<int>(blockIdx.x) * kWarpsPerBlock +
                    static_cast<int>(threadIdx.y);
    if (row >= rows) return; // whole warp exits together
    const int lane = static_cast<int>(threadIdx.x);
    const int groups = cols / 32;
    const auto base = static_cast<std::size_t>(row) * cols;
    const auto scale_base = static_cast<std::size_t>(row) * groups;
    float sum = vector4_lane_sum(codes, scales, x, base, scale_base,
                                 groups, 0, 1, lane);
    sum = warp_reduce_sum(sum);
    if (lane == 0) y[row] = sum;
}

// Two rows per CTA. Four warps independently process every fourth G32 group.
// Warp partials are reduced in shared memory after a full-block barrier.
// No conditional return before __syncthreads(), including the partial last CTA.
__global__ void gemv_split_k4(
    const std::uint8_t* __restrict__ codes,
    const std::uint8_t* __restrict__ scales,
    const float* __restrict__ x,
    float* __restrict__ y,
    int rows,
    int cols) {
    __shared__ float partial[kRowsPerSplitBlock][kSplitWarps];
    const int lane = static_cast<int>(threadIdx.x);
    const int warp = static_cast<int>(threadIdx.y);
    const int local_row = warp / kSplitWarps;
    const int split = warp % kSplitWarps;
    const int row = static_cast<int>(blockIdx.x) * kRowsPerSplitBlock + local_row;
    const int groups = cols / 32;
    float sum = 0.f;
    if (row < rows) {
        const std::size_t base = static_cast<std::size_t>(row) * cols;
        const std::size_t scale_base = static_cast<std::size_t>(row) * groups;
        for (int group = split; group < groups; group += kSplitWarps) {
            const std::uint8_t code = codes[base + static_cast<std::size_t>(group) * 32 + lane];
            const float scale = ue8m0_to_float(scales[scale_base + group]);
            const float input = x[group * 32 + lane];
            sum = fmaf(fp8_to_float(code) * scale, input, sum);
        }
    }
    sum = warp_reduce_sum(sum);
    if (lane == 0) partial[local_row][split] = sum;
    __syncthreads();
    if (lane == 0 && split == 0 && row < rows) {
        y[row] = (partial[local_row][0] + partial[local_row][1]) +
                 (partial[local_row][2] + partial[local_row][3]);
    }
}

__global__ void gemv_split_k4_vector4(
    const std::uint8_t* __restrict__ codes,
    const std::uint8_t* __restrict__ scales,
    const float* __restrict__ x,
    float* __restrict__ y,
    int rows,
    int cols) {
    __shared__ float partial[kRowsPerSplitBlock][kSplitWarps];
    const int lane = static_cast<int>(threadIdx.x);
    const int warp = static_cast<int>(threadIdx.y);
    const int local_row = warp / kSplitWarps;
    const int split = warp % kSplitWarps;
    const int row = static_cast<int>(blockIdx.x) * kRowsPerSplitBlock + local_row;
    const int groups = cols / 32;
    float sum = 0.f;
    if (row < rows) {
        sum = vector4_lane_sum(
            codes, scales, x,
            static_cast<std::size_t>(row) * cols,
            static_cast<std::size_t>(row) * groups,
            groups, split, kSplitWarps, lane);
    }
    sum = warp_reduce_sum(sum);
    if (lane == 0) partial[local_row][split] = sum;
    __syncthreads();
    if (lane == 0 && split == 0 && row < rows) {
        y[row] = (partial[local_row][0] + partial[local_row][1]) +
                 (partial[local_row][2] + partial[local_row][3]);
    }
}
}  // namespace

const char* mxfp8_ab_variant_name(Mxfp8AbVariant variant) noexcept {
    switch (variant) {
        case Mxfp8AbVariant::warp_vector4: return "warp_vector4";
        case Mxfp8AbVariant::split_k4: return "split_k4";
        case Mxfp8AbVariant::split_k4_vector4: return "split_k4_vector4";
        default: return "invalid";
    }
}

cudaError_t launch_mxfp8_gemv_ab(
    const std::uint8_t* codes,
    const std::uint8_t* scales,
    const float* x,
    float* y,
    int rows,
    int cols,
    Mxfp8AbVariant variant,
    cudaStream_t stream) noexcept {
    if (!codes || !scales || !x || !y || rows <= 0 || cols <= 0 || cols % 32 != 0)
        return cudaErrorInvalidValue;
    const bool vector4 = variant == Mxfp8AbVariant::warp_vector4 ||
                         variant == Mxfp8AbVariant::split_k4_vector4;
    if (vector4 && ((reinterpret_cast<std::uintptr_t>(codes) & 3u) != 0 ||
                    (reinterpret_cast<std::uintptr_t>(x) & 15u) != 0))
        return cudaErrorInvalidValue;
    const dim3 threads(kWarp, kWarpsPerBlock);
    switch (variant) {
        case Mxfp8AbVariant::warp_vector4: {
            const dim3 blocks((static_cast<unsigned>(rows) + kWarpsPerBlock - 1u) /
                              kWarpsPerBlock);
            gemv_warp_vector4<<<blocks, threads, 0, stream>>>(
                codes, scales, x, y, rows, cols);
            break;
        }
        case Mxfp8AbVariant::split_k4: {
            const dim3 blocks((static_cast<unsigned>(rows) + kRowsPerSplitBlock - 1u) /
                              kRowsPerSplitBlock);
            gemv_split_k4<<<blocks, threads, 0, stream>>>(
                codes, scales, x, y, rows, cols);
            break;
        }
        case Mxfp8AbVariant::split_k4_vector4: {
            const dim3 blocks((static_cast<unsigned>(rows) + kRowsPerSplitBlock - 1u) /
                              kRowsPerSplitBlock);
            gemv_split_k4_vector4<<<blocks, threads, 0, stream>>>(
                codes, scales, x, y, rows, cols);
            break;
        }
        default:
            return cudaErrorInvalidValue;
    }
    return cudaGetLastError();
}

}  // namespace ninfer::models::inkling2
