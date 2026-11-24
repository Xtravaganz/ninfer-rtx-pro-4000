#include "models/inkling2/conv_cache_stage5c.h"
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>

namespace ninfer::models::inkling2 {
namespace {
constexpr int kThreads = 256;

template <typename T> __device__ __forceinline__ float cvt(T value) {
    return static_cast<float>(value);
}
template <> __device__ __forceinline__ float cvt<__nv_bfloat16>(__nv_bfloat16 value) {
    return __bfloat162float(value);
}
template <typename T> __device__ __forceinline__ T cast(float value) {
    return static_cast<T>(value);
}
template <> __device__ __forceinline__ __nv_bfloat16 cast<__nv_bfloat16>(float value) {
    return __float2bfloat16_rn(value);
}

template <typename T, bool Dynamic>
__global__ void apply_conv_kernel(const T* __restrict__ x,
                                  const float* __restrict__ static_weight,
                                  const float* __restrict__ dynamic_taps,
                                  const std::uint8_t* __restrict__ mask,
                                  const std::int32_t* __restrict__ seq_ids,
                                  const float* __restrict__ history,
                                  const std::int32_t* __restrict__ history_ids,
                                  T* __restrict__ y,
                                  int sequence, int channels, int heads,
                                  int kernel, int stride, std::size_t count) {
    const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= count) return;
    const int ch = static_cast<int>(i % channels);
    const std::size_t global_time = i / channels;
    const int t = static_cast<int>(global_time % sequence);
    const int b = static_cast<int>(global_time / sequence);
    const float residual = cvt(x[i]);
    int head = 0;
    if constexpr (Dynamic) head = ch / (channels / heads);
    const int seq_target = (Dynamic && seq_ids != nullptr) ? seq_ids[global_time] : 0;
    float sum = 0.f;
    for (int lag = 0; lag < kernel; ++lag) {
        const int pos = t - lag;
        float src;
        if (pos >= 0) {
            const std::size_t src_time = static_cast<std::size_t>(b)*sequence + pos;
            if constexpr (Dynamic) {
                if (seq_ids != nullptr && seq_ids[src_time] != seq_target) continue;
            }
            if (mask != nullptr && mask[src_time] == 0) continue;
            src = cvt(x[src_time * channels + ch]);
        } else {
            const int old = (kernel - 1) + pos;
            if constexpr (Dynamic) {
                if (seq_ids != nullptr && history_ids[static_cast<std::size_t>(b)*(kernel-1)+old] != seq_target)
                    continue;
            }
            src = history[(static_cast<std::size_t>(b)*(kernel-1)+old)*stride+ch];
        }
        const int weight_index = Dynamic ? lag : kernel - 1 - lag;
        float w = static_weight[static_cast<std::size_t>(ch)*kernel + weight_index];
        if constexpr (Dynamic) {
            w += dynamic_taps[(global_time*heads+head)*kernel+lag];
        }
        sum = fmaf(src, w, sum);
    }
    const float denom = Dynamic ? (2.f * kernel + 1.f) : (kernel + 1.f);
    y[i] = cast<T>((sum + residual) * rsqrtf(denom));
}

template <typename T>
__global__ void update_history_kernel(const T* __restrict__ x,
                                      const std::uint8_t* __restrict__ mask,
                                      const std::int32_t* __restrict__ seq_ids,
                                      const float* __restrict__ previous,
                                      const std::int32_t* __restrict__ prev_ids,
                                      float* __restrict__ next,
                                      std::int32_t* __restrict__ next_ids,
                                      int sequence, int channels, int kernel,
                                      int stride, std::size_t count) {
    const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= count) return;
    const int ch = static_cast<int>(i % channels);
    const std::size_t total_time = i / channels;
    const int past_idx = static_cast<int>(total_time % (kernel - 1));
    const int b = static_cast<int>(total_time / (kernel - 1));
    const int pos = sequence + past_idx - (kernel - 1);
    float value;
    std::int32_t sid;
    if (pos < 0) {
        const int old_idx = (kernel - 1) + pos;
        const std::size_t old_time = static_cast<std::size_t>(b)*(kernel-1) + old_idx;
        value = previous[old_time*stride+ch];
        sid = prev_ids[old_time];
    } else {
        const std::size_t now = static_cast<std::size_t>(b)*sequence + pos;
        value = (mask != nullptr && mask[now] == 0) ? 0.f
                : cvt(x[now*channels+ch]);
        sid = seq_ids == nullptr ? -1 : seq_ids[now];
    }
    next[total_time*stride+ch] = value;
    if (ch == 0) next_ids[total_time] = sid;
}

inline bool supported(ForwardScalar scalar) {
    return scalar == ForwardScalar::float32 || scalar == ForwardScalar::bfloat16;
}
}  // namespace

int inkling2_cache_slot(int layer, int pass) noexcept {
    if (layer < 0 || layer >= 24 || pass < 0 || pass > 2) return -1;
    if (layer < 6 || layer >= 18) return pass == 0 ? layer : -1;
    return pass == 0 ? layer : 24 + (pass - 1) * 12 + layer - 6;
}

ConvCache48::~ConvCache48() noexcept { release(); }

void ConvCache48::release() noexcept {
    if (history_) cudaFree(history_);
    if (ids_) cudaFree(ids_);
    history_ = nullptr;
    ids_ = nullptr;
    batch_ = max_channels_ = kernel_ = 0;
    std::memset(parity_, 0, sizeof(parity_));
    std::memset(bound_channels_, 0, sizeof(bound_channels_));
    std::memset(bound_heads_, 0, sizeof(bound_heads_));
    std::memset(bound_mode_, 0, sizeof(bound_mode_));
    std::memset(bound_dtype_, 0, sizeof(bound_dtype_));
}

cudaError_t ConvCache48::create(int batch, int max_channels, int kernel,
                                cudaStream_t stream) noexcept {
    if (ready() || ids_ || batch <= 0 || max_channels <= 0 ||
        kernel < 2 || kernel > 32) return cudaErrorInvalidValue;
    const std::size_t max = std::numeric_limits<std::size_t>::max();
    // Compute total alloc bytes without overflowing size_t or a 32-bit grid.
    const std::size_t groups = kGroups;
    if (static_cast<std::size_t>(batch) > max/groups/(kernel-1)/max_channels/sizeof(float))
        return cudaErrorInvalidValue;
    const std::size_t words = groups * static_cast<std::size_t>(batch) * (kernel-1) * max_channels;
    const std::size_t ids = groups * static_cast<std::size_t>(batch) * (kernel-1);
    cudaError_t e = cudaMalloc(reinterpret_cast<void**>(&history_), words*sizeof(float));
    if (e != cudaSuccess) return e;
    e = cudaMalloc(reinterpret_cast<void**>(&ids_), ids*sizeof(std::int32_t));
    if (e != cudaSuccess) { release(); return e; }
    batch_ = batch; max_channels_ = max_channels; kernel_ = kernel;
    e = reset(stream);
    if (e != cudaSuccess) { release(); return e; }
    return cudaSuccess;
}

cudaError_t ConvCache48::reset(cudaStream_t stream) noexcept {
    if (!ready() || !ids_) return cudaErrorInvalidValue;
    std::memset(parity_, 0, sizeof(parity_));
    std::memset(bound_channels_, 0, sizeof(bound_channels_));
    std::memset(bound_heads_, 0, sizeof(bound_heads_));
    std::memset(bound_mode_, 0, sizeof(bound_mode_));
    std::memset(bound_dtype_, 0, sizeof(bound_dtype_));
    const std::size_t elements = static_cast<std::size_t>(kGroups)*batch_*(kernel_-1)*max_channels_;
    const std::size_t nids = static_cast<std::size_t>(kGroups)*batch_*(kernel_-1);
    cudaError_t e = cudaMemsetAsync(history_, 0, elements*sizeof(float), stream);
    if (e != cudaSuccess) return e;
    return cudaMemsetAsync(ids_, 0xff, nids*sizeof(std::int32_t), stream);
}

std::size_t ConvCache48::group_index(int slot, int conv, int copy) const noexcept {
    return static_cast<std::size_t>(((slot * kStreams) + conv)*kCopies+copy);
}
float* ConvCache48::history_group(std::size_t index) const noexcept {
    return history_ + index*static_cast<std::size_t>(batch_)*(kernel_-1)*max_channels_;
}
std::int32_t* ConvCache48::id_group(std::size_t index) const noexcept {
    return ids_ + index*static_cast<std::size_t>(batch_)*(kernel_-1);
}

cudaError_t ConvCache48::run(int physical_layer, int core_pass, int conv_idx,
                             ConvMode5C mode, const void* x,
                             const float* static_weight, const float* dynamic_taps,
                             const std::uint8_t* mask, const std::int32_t* seq_ids,
                             void* y, int sequence, int channels, int heads,
                             ForwardScalar dtype, cudaStream_t stream) noexcept {
    const int slot = inkling2_cache_slot(physical_layer,core_pass);
    const bool dynamic = mode == ConvMode5C::dynamic_kv;
    const bool full_attention = physical_layer % 6 == 5;
    if (!ready() || slot < 0 || conv_idx < 0 || conv_idx >= 4 ||
        !x || !y || x == y || !static_weight || sequence <= 0 ||
        channels <= 0 || channels > max_channels_ || !supported(dtype) ||
        (dynamic && (conv_idx > 1 || !full_attention || heads <= 0 || channels%heads || !dynamic_taps)) ||
        (!dynamic && (mode != ConvMode5C::static_causal ||
                      (conv_idx < 2 && full_attention) ||
                      heads != 0 || dynamic_taps != nullptr || seq_ids != nullptr)))
        return cudaErrorInvalidValue;
    const std::size_t elements = static_cast<std::size_t>(batch_)*sequence*channels;
    const std::size_t history_elements = static_cast<std::size_t>(batch_)*(kernel_-1)*channels;
    const std::size_t max_grid = static_cast<std::size_t>(std::numeric_limits<int>::max())*kThreads;
    if (elements > max_grid || history_elements > max_grid) return cudaErrorInvalidValue;
    const int state = slot*4 + conv_idx;
    if (bound_channels_[state] && (bound_channels_[state] != channels ||
        bound_heads_[state] != heads || bound_mode_[state] != static_cast<int>(mode) ||
        bound_dtype_[state] != static_cast<int>(dtype))) return cudaErrorInvalidValue;
    const int old_copy = parity_[state];
    const std::size_t old_idx = group_index(slot,conv_idx,old_copy);
    const std::size_t new_idx = group_index(slot,conv_idx,1-old_copy);
    const float* old_history = history_group(old_idx);
    const std::int32_t* old_ids = id_group(old_idx);
    float* new_history = history_group(new_idx);
    std::int32_t* new_ids = id_group(new_idx);
    const unsigned grid = static_cast<unsigned>((elements+kThreads-1)/kThreads);
    const unsigned tail_grid = static_cast<unsigned>((history_elements+kThreads-1)/kThreads);
    if (dtype == ForwardScalar::float32) {
        if (dynamic) {
            apply_conv_kernel<float,true><<<grid,kThreads,0,stream>>>(
                static_cast<const float*>(x),static_weight,dynamic_taps,mask,seq_ids,
                old_history,old_ids,static_cast<float*>(y),sequence,channels,heads,kernel_,max_channels_,elements);
        } else {
            apply_conv_kernel<float,false><<<grid,kThreads,0,stream>>>(
                static_cast<const float*>(x),static_weight,nullptr,mask,nullptr,
                old_history,old_ids,static_cast<float*>(y),sequence,channels,1,kernel_,max_channels_,elements);
        }
    } else {
        if (dynamic) {
            apply_conv_kernel<__nv_bfloat16,true><<<grid,kThreads,0,stream>>>(
                static_cast<const __nv_bfloat16*>(x),static_weight,dynamic_taps,mask,seq_ids,
                old_history,old_ids,static_cast<__nv_bfloat16*>(y),sequence,channels,heads,kernel_,max_channels_,elements);
        } else {
            apply_conv_kernel<__nv_bfloat16,false><<<grid,kThreads,0,stream>>>(
                static_cast<const __nv_bfloat16*>(x),static_weight,nullptr,mask,nullptr,
                old_history,old_ids,static_cast<__nv_bfloat16*>(y),sequence,channels,1,kernel_,max_channels_,elements);
        }
    }
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) return e;
    if (dtype == ForwardScalar::float32) {
        update_history_kernel<float><<<tail_grid,kThreads,0,stream>>>(
            static_cast<const float*>(x),mask,dynamic ? seq_ids : nullptr,
            old_history,old_ids,new_history,new_ids,sequence,channels,kernel_,max_channels_,history_elements);
    } else {
        update_history_kernel<__nv_bfloat16><<<tail_grid,kThreads,0,stream>>>(
            static_cast<const __nv_bfloat16*>(x),mask,dynamic ? seq_ids : nullptr,
            old_history,old_ids,new_history,new_ids,sequence,channels,kernel_,max_channels_,history_elements);
    }
    e = cudaGetLastError();
    if (e == cudaSuccess) {
        parity_[state] = static_cast<std::uint8_t>(1-old_copy);
        bound_channels_[state] = channels;
        bound_heads_[state] = heads;
        bound_mode_[state] = static_cast<int>(mode);
        bound_dtype_[state] = static_cast<int>(dtype);
    }
    return e;
}

} // namespace ninfer::models::inkling2
