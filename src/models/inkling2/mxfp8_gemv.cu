// Stage 3B first native CUDA GEMV baseline for Inkling 2 Mini MXFP8 banks.
// This intentionally uses CUDA FP32 FMA, NOT Blackwell MXFP8 tensor cores.
// No modifications to the existing NInfer model registry/dispatch paths.

#include "mxfp8_gemv.h"

#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>

namespace ninfer::models::inkling2 {
namespace {

constexpr int kWarp = 32;
constexpr int kWarpsPerBlock = 8;

__device__ __forceinline__ float fp8_e4m3fn_to_float(std::uint8_t word) {
    // Reject the two E4M3FN NaN bit patterns. No infinity representation.
    if ((word & 0x7fu) == 0x7fu) {
        return __uint_as_float(0x7fc00000u);
    }
    const unsigned sign = (static_cast<unsigned>(word & 0x80u) << 24);
    const unsigned e = (word >> 3) & 0x0fu;
    const unsigned mantissa = word & 0x07u;
    if (e == 0) {
        // Exact E4M3FN subnormal: mantissa * 2^-9; preserve signed zero.
        const float mag = static_cast<float>(mantissa) * 0x1p-9f;
        return __uint_as_float(__float_as_uint(mag) | sign);
    }
    // E4M3FN normal: significand (1 + mantissa/8) * 2^(e-7).
    // FP32 biased exponent = e-7+127 = e+120; exact reinterpretation.
    return __uint_as_float(sign | ((e + 120u) << 23) | (mantissa << 20));
}

__device__ __forceinline__ float ue8m0_to_float(std::uint8_t word) {
    if (word == 0xffu) {
        return __uint_as_float(0x7fc00000u);
    }
    // 0x00 is 2^-127 (FP32 subnormal bit 22), not 0. All 1..254 map
    // directly to the FP32 biased exponent with zero mantissa.
    return __uint_as_float(word == 0 ? 0x00400000u :
                           (static_cast<unsigned>(word) << 23));
}

__global__ void mxfp8_gemv_warp_rows(
    const std::uint8_t* __restrict__ codes,
    const std::uint8_t* __restrict__ scales,
    const float* __restrict__ x,
    float* __restrict__ y,
    int rows,
    int cols) {
    const int row = static_cast<int>(blockIdx.x) * kWarpsPerBlock +
                    static_cast<int>(threadIdx.y);
    if (row >= rows) return;  // all lanes of this warp return together
    const int lane = static_cast<int>(threadIdx.x);
    const int groups = cols / 32;
    const std::size_t code_row = static_cast<std::size_t>(row) * cols;
    const std::size_t scale_row = static_cast<std::size_t>(row) * groups;

    float sum = 0.f;
    for (int group = 0; group < groups; ++group) {
        // Each warp fetches one aligned block of 32 bytes and one common scale.
        const auto word = codes[code_row + group * 32 + lane];
        const auto scale_word = scales[scale_row + group];
        const float weight = fp8_e4m3fn_to_float(word);
        const float scale = ue8m0_to_float(scale_word);
        const float input = x[group * 32 + lane];
        sum = fmaf(weight * scale, input, sum);
    }
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(0xffffffffu, sum, offset);
    }
    if (lane == 0) y[row] = sum;
}

}  // namespace

cudaError_t launch_mxfp8_gemv(
    const std::uint8_t* codes,
    const std::uint8_t* scales,
    const float* x,
    float* y,
    int rows,
    int cols,
    cudaStream_t stream) noexcept {
    if (codes == nullptr || scales == nullptr || x == nullptr || y == nullptr ||
        rows <= 0 || cols <= 0 || (cols % 32) != 0) {
        return cudaErrorInvalidValue;
    }
    const dim3 threads(kWarp, kWarpsPerBlock);
    const dim3 blocks((static_cast<unsigned>(rows) + kWarpsPerBlock - 1u) /
                      kWarpsPerBlock);
    mxfp8_gemv_warp_rows<<<blocks, threads, 0, stream>>>(
        codes, scales, x, y, rows, cols);
    return cudaGetLastError();
}

}  // namespace ninfer::models::inkling2
