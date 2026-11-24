// Stage 5B: deliberately simple, independent native CUDA numerics.
// First correctness baseline for RMSNorm, causal static convolution and
// sink-aware GQA token decode. NOT a fused decoder, model loader or fast
// general prefill attention implementation.
#include "forward_primitives.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace ninfer::models::inkling2 {
namespace {
constexpr int kThreads = 256;

template <typename T> __device__ __forceinline__ float to_f32(T x) {
    return static_cast<float>(x);
}
template <> __device__ __forceinline__ float to_f32<__nv_bfloat16>(__nv_bfloat16 x) {
    return __bfloat162float(x);
}
template <typename T> __device__ __forceinline__ T from_f32(float x) {
    return static_cast<T>(x);
}
template <> __device__ __forceinline__ __nv_bfloat16 from_f32<__nv_bfloat16>(float x) {
    return __float2bfloat16_rn(x);
}

template <typename T, bool Weighted>
__global__ void rmsnorm_kernel(const T* __restrict__ x,
                               const float* __restrict__ weight,
                               T* __restrict__ out, int hidden, float eps) {
    const int row = static_cast<int>(blockIdx.x);
    const int lane = static_cast<int>(threadIdx.x);
    const std::size_t start = static_cast<std::size_t>(row) * hidden;
    __shared__ float partial[kThreads];
    __shared__ float factor;
    float sum = 0.f;
    for (int d=lane; d<hidden; d+=kThreads) {
        const float value = to_f32(x[start+d]);
        sum = fmaf(value, value, sum);
    }
    partial[lane] = sum;
    __syncthreads();
    if (lane == 0) {
        float total = 0.f;
        for (int i=0; i<kThreads; ++i) total += partial[i];
        factor = rsqrtf(total / static_cast<float>(hidden) + eps);
    }
    __syncthreads();
    for (int d=lane; d<hidden; d+=kThreads) {
        float value = to_f32(x[start+d]) * factor;
        if constexpr (Weighted) {
            // The Stage5A oracle calls unweighted_rmsnorm(...).to(x.dtype)
            // before multiplying the learned weight in FP32.
            value = to_f32(from_f32<T>(value)) * weight[d];
        }
        out[start+d] = from_f32<T>(value);
    }
}

template <typename T>
__global__ void static_conv_kernel(const T* __restrict__ x,
                                   const float* __restrict__ weight,
                                   const std::uint8_t* __restrict__ mask,
                                   T* __restrict__ out, int seq, int channels,
                                   int taps, std::size_t elements) {
    const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= elements) return;
    const int channel = static_cast<int>(i % channels);
    const int pos = static_cast<int>((i / channels) % seq);
    const std::size_t batch = i / (static_cast<std::size_t>(seq) * channels);
    const std::size_t time_start = batch * static_cast<std::size_t>(seq);
    const float residual = to_f32(x[i]);
    float sum = 0.f;
    for (int tap=0; tap<taps; ++tap) {
        const int src_pos = pos + tap - (taps - 1);
        if (src_pos < 0) continue;
        if (mask != nullptr && mask[time_start + src_pos] == 0) continue;
        const std::size_t src = (time_start + src_pos) * channels + channel;
        sum = fmaf(to_f32(x[src]), weight[static_cast<std::size_t>(channel)*taps+tap], sum);
    }
    out[i] = from_f32<T>((sum + residual) * rsqrtf(static_cast<float>(taps + 1)));
}

template <typename T>
__global__ void sink_gqa_decode_kernel(
    const T* __restrict__ query, const T* __restrict__ key,
    const T* __restrict__ value, const float* __restrict__ position_bias,
    const float* __restrict__ sinks, const std::uint8_t* __restrict__ valid_mask,
    T* __restrict__ out, int q_heads, int kv_heads, int keys, int head_dim) {
    const int head = static_cast<int>(blockIdx.x);
    const int batch = static_cast<int>(blockIdx.y);
    const int lane = static_cast<int>(threadIdx.x);
    const int kv_head = head / (q_heads / kv_heads);
    const std::size_t q_base = (static_cast<std::size_t>(batch)*q_heads + head)*head_dim;
    const std::size_t kv_base = (static_cast<std::size_t>(batch)*kv_heads + kv_head)*keys*head_dim;
    const std::size_t bias_base = (static_cast<std::size_t>(batch)*q_heads + head)*keys;
    __shared__ float products[kThreads];
    __shared__ float logit_shared;
    float numer = 0.f;
    float maximum = sinks[static_cast<std::size_t>(batch)*q_heads + head];
    float denom = 1.f;  // one additional sink logit with zero value
    const float scaling = rsqrtf(static_cast<float>(head_dim));
    for (int token=0; token<keys; ++token) {
        // Uniform branch for this entire CTA: all heads in a batch share mask.
        if (valid_mask != nullptr && valid_mask[static_cast<std::size_t>(batch)*keys+token] == 0) continue;
        const std::size_t kv_pos = kv_base + static_cast<std::size_t>(token)*head_dim;
        const float product = lane < head_dim
            ? to_f32(query[q_base+lane]) * to_f32(key[kv_pos+lane]) : 0.f;
        products[lane] = product;
        __syncthreads();
        if (lane == 0) {
            float dot = 0.f;
            for (int d=0; d<kThreads; ++d) dot += products[d];
            logit_shared = dot * scaling + (position_bias == nullptr ? 0.f : position_bias[bias_base+token]);
        }
        __syncthreads();
        const float logit = logit_shared;
        const float next_max = fmaxf(maximum, logit);
        const float alpha = expf(maximum - next_max);
        const float beta = expf(logit - next_max);
        denom = denom * alpha + beta;
        if (lane < head_dim) numer = numer * alpha + beta * to_f32(value[kv_pos+lane]);
        maximum = next_max;
        __syncthreads();  // products[] may be written on next token
    }
    if (lane < head_dim) out[q_base+lane] = from_f32<T>(numer / denom);
}

inline bool valid_type(ForwardScalar dtype) {
    return dtype == ForwardScalar::float32 || dtype == ForwardScalar::bfloat16;
}

} // namespace

cudaError_t launch_rmsnorm_forward(const void* x, const float* weight, void* out,
                                   int rows, int hidden, float eps, bool weighted,
                                   ForwardScalar dtype, cudaStream_t stream) noexcept {
    if (!x || !out || (weighted && !weight) || rows <= 0 || hidden <= 0 ||
        !std::isfinite(eps) || eps <= 0 || !valid_type(dtype)) return cudaErrorInvalidValue;
    if (dtype == ForwardScalar::float32) {
        if (weighted) rmsnorm_kernel<float, true><<<rows, kThreads, 0, stream>>>(
            static_cast<const float*>(x),weight,static_cast<float*>(out),hidden,eps);
        else rmsnorm_kernel<float, false><<<rows, kThreads, 0, stream>>>(
            static_cast<const float*>(x),weight,static_cast<float*>(out),hidden,eps);
    } else {
        if (weighted) rmsnorm_kernel<__nv_bfloat16, true><<<rows, kThreads, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(x),weight,static_cast<__nv_bfloat16*>(out),hidden,eps);
        else rmsnorm_kernel<__nv_bfloat16, false><<<rows, kThreads, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(x),weight,static_cast<__nv_bfloat16*>(out),hidden,eps);
    }
    return cudaGetLastError();
}

cudaError_t launch_static_conv_forward(const void* x, const float* weight,
                                       const std::uint8_t* padding_mask, void* out,
                                       int batch, int seq, int channels, int kernel,
                                       ForwardScalar dtype, cudaStream_t stream) noexcept {
    if (!x || !weight || !out || batch <= 0 || seq <= 0 || channels <= 0 ||
        kernel <= 0 || kernel > 64 || !valid_type(dtype)) return cudaErrorInvalidValue;
    const std::size_t total = static_cast<std::size_t>(batch)*seq*channels;
    if (total > static_cast<std::size_t>(std::numeric_limits<int>::max()) * kThreads)
        return cudaErrorInvalidValue;
    const unsigned blocks = static_cast<unsigned>((total + kThreads-1)/kThreads);
    if (dtype == ForwardScalar::float32) {
        static_conv_kernel<float><<<blocks,kThreads,0,stream>>>(
            static_cast<const float*>(x),weight,padding_mask,static_cast<float*>(out),seq,channels,kernel,total);
    } else {
        static_conv_kernel<__nv_bfloat16><<<blocks,kThreads,0,stream>>>(
            static_cast<const __nv_bfloat16*>(x),weight,padding_mask,static_cast<__nv_bfloat16*>(out),seq,channels,kernel,total);
    }
    return cudaGetLastError();
}

cudaError_t launch_sink_gqa_decode(const void* query, const void* key, const void* value,
                                    const float* position_bias, const float* sink_logits,
                                    const std::uint8_t* valid_key_mask, void* out,
                                    int batch, int query_heads, int kv_heads, int keys,
                                    int head_dim, ForwardScalar dtype, cudaStream_t stream) noexcept {
    if (!query || !key || !value || !sink_logits || !out || batch <= 0 ||
        query_heads <= 0 || kv_heads <= 0 || query_heads % kv_heads != 0 ||
        keys <= 0 || head_dim <= 0 || head_dim > kThreads || !valid_type(dtype))
        return cudaErrorInvalidValue;
    const dim3 grid(static_cast<unsigned>(query_heads),static_cast<unsigned>(batch));
    if (dtype == ForwardScalar::float32) {
        sink_gqa_decode_kernel<float><<<grid,kThreads,0,stream>>>(
            static_cast<const float*>(query),static_cast<const float*>(key),
            static_cast<const float*>(value),position_bias,sink_logits,valid_key_mask,
            static_cast<float*>(out),query_heads,kv_heads,keys,head_dim);
    } else {
        sink_gqa_decode_kernel<__nv_bfloat16><<<grid,kThreads,0,stream>>>(
            static_cast<const __nv_bfloat16*>(query),static_cast<const __nv_bfloat16*>(key),
            static_cast<const __nv_bfloat16*>(value),position_bias,sink_logits,valid_key_mask,
            static_cast<__nv_bfloat16*>(out),query_heads,kv_heads,keys,head_dim);
    }
    return cudaGetLastError();
}
} // namespace ninfer::models::inkling2
