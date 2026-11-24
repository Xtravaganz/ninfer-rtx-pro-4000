#pragma once
// Inkling 2 Mini Stage 5B: CUDA mathematical primitives, isolated test backend.
// All pointers are device-resident; output has the same type as input.
// This is NOT an executable NInfer model or a registered runtime kernel.
#include <cuda_runtime_api.h>
#include <cstdint>

namespace ninfer::models::inkling2 {

enum class ForwardScalar : int { float32 = 0, bfloat16 = 1 };

// BF16 input/output uses FP32 accumulation. Weight is FP32 (the loader may
// upcast the checkpoint's norm weight once). If weighted, the normalized
// activation is rounded back to BF16 BEFORE weight application, matching HF.
cudaError_t launch_rmsnorm_forward(
    const void* x, const float* weight, void* out,
    int rows, int hidden, float eps, bool weighted, ForwardScalar dtype,
    cudaStream_t stream = nullptr) noexcept;

// [batch, seq, channels] input; [channels,kernel] weight, oldest tap first;
// optional uint8 [batch,seq] mask; convolution sees masked inputs but
// residual remains unmasked. /sqrt(kernel+1), matching Stage-5A.
cudaError_t launch_static_conv_forward(
    const void* x, const float* weight, const std::uint8_t* padding_mask,
    void* out, int batch, int seq, int channels, int kernel,
    ForwardScalar dtype, cudaStream_t stream = nullptr) noexcept;

// Single-query causal/decode GQA. Queries [B,Hq,D]; K,V [B,Hkv,T,D],
// bias [B,Hq,T] (already computed relative bias etc.), optional [B,T]
// valid-key mask (0=masked), sink logits [B,Hq] in FP32. Sink value is ZERO.
// No implicit causal/sliding mask: caller supplies valid_key_mask.
// Stable online softmax, each (batch,query-head) independent; [B,Hq,D] output.
// BF16 Q/K/V/output supported, with FP32 softmax and accumulation.
cudaError_t launch_sink_gqa_decode(
    const void* query, const void* key, const void* value,
    const float* position_bias, const float* sink_logits,
    const std::uint8_t* valid_key_mask, void* out,
    int batch, int query_heads, int kv_heads, int keys, int head_dim,
    ForwardScalar dtype, cudaStream_t stream = nullptr) noexcept;

}  // namespace ninfer::models::inkling2
