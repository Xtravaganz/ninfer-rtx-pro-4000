#pragma once

// Inkling 2 Mini Stage 3C: opt-in CUDA GEMV microarchitecture experiments.
// Three kernels use the exact Stage-3B E4M3FN/G32/UE8M0 input layout.
// Stage 3B stays the unchanged baseline; none of these kernels are registered
// in NInfer's inference runtime or use Blackwell tensor cores.

#include <cuda_runtime_api.h>
#include <cstdint>

namespace ninfer::models::inkling2 {

enum class Mxfp8AbVariant : int {
    warp_vector4 = 0,      // 8 warps/CTA, 1 warp/row, 4 weights/thread
    split_k4 = 1,          // 8 warps/CTA, 4 warps/row, 1 weight/thread
    split_k4_vector4 = 2,  // 8 warps/CTA, 4 warps/row, 4 weights/thread
};

// Asynchronous launch. W = codes[rows,cols], scales[rows,cols/32],
// x = FP32[cols], y = FP32[rows]. Requires positive rows and cols % 32 == 0.
// All buffers are GPU resident; they must remain alive until stream sync.
// Vector4 modes require codes 4-byte aligned and x 16-byte aligned. CUDA
// allocations and expert-row starts with K % 32 == 0 meet this requirement.
// The scalar split-K mode has no additional vector-alignment requirement.
// Invalid E4M3FN or UE8M0 words propagate NaN; validate on CPU first.
// Returns launch errors; synchronize to diagnose asynchronous failures.
[[nodiscard]] cudaError_t launch_mxfp8_gemv_ab(
    const std::uint8_t* codes,
    const std::uint8_t* scales,
    const float* x,
    float* y,
    int rows,
    int cols,
    Mxfp8AbVariant variant,
    cudaStream_t stream = nullptr) noexcept;

const char* mxfp8_ab_variant_name(Mxfp8AbVariant variant) noexcept;

}  // namespace ninfer::models::inkling2
