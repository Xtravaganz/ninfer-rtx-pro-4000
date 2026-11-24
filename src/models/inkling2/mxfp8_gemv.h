#pragma once

// Inkling 2 Mini Stage 3B: standalone native CUDA scalar/warp baseline.
// Layout: codes[rows, cols] E4M3FN, scales[rows, cols/32] UE8M0.
// No repack/dequantized full matrix, no NInfer-runtime registration.

#include <cuda_runtime_api.h>
#include <cstdint>

namespace ninfer::models::inkling2 {

// Asynchronous launch; buffers must be on device and remain valid until stream
// completion. y is FP32 [rows], x FP32 [cols]. cols must be a positive multiple
// of 32. Invalid E4M3FN/UE8M0 words deliberately propagate NaN; validate once
// before GPU upload using the Stage-3A MatrixView contract.
// Returns CUDA launch/configuration status. Caller must synchronize to detect
// asynchronous failures. Supports non-default CUDA streams.
[[nodiscard]] cudaError_t launch_mxfp8_gemv(
    const std::uint8_t* codes,
    const std::uint8_t* scales,
    const float* x,
    float* y,
    int rows,
    int cols,
    cudaStream_t stream = nullptr) noexcept;

}  // namespace ninfer::models::inkling2
