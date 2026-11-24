#include "models/inkling2/attention_cache_stage5d.h"
#include "models/inkling2/conv_cache_stage5c.h" // inkling2_cache_slot mapping
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cmath>
#include <cstring>
#include <limits>

namespace ninfer::models::inkling2 {
namespace {
constexpr int kThreads = 256;
template<class T> __device__ __forceinline__ float as_float(T v) { return static_cast<float>(v); }
template<> __device__ __forceinline__ float as_float(__nv_bfloat16 v) { return __bfloat162float(v); }
template<class T> __device__ __forceinline__ T as_type(float x) { return static_cast<T>(x); }
template<> __device__ __forceinline__ __nv_bfloat16 as_type(float x) { return __float2bfloat16_rn(x); }

template<class T> __global__ void append_kv_kernel(
    const T* k, const T* v, const std::uint8_t* mask, const std::int32_t* ids,
    T* cached_k, T* cached_v, std::uint8_t* cached_valid, std::int32_t* cached_ids,
    int tokens, int start, int capacity, int kv_heads, int dim,
    std::size_t n_elements) {
    const std::size_t i = static_cast<std::size_t>(blockIdx.x)*blockDim.x+threadIdx.x;
    if (i < n_elements) {
        const std::size_t per_token=static_cast<std::size_t>(kv_heads)*dim;
        const std::size_t pos=i/per_token;
        const int b=static_cast<int>(pos/tokens), t=static_cast<int>(pos%tokens);
        const std::size_t dst=(static_cast<std::size_t>(b)*capacity+start+t)*per_token+i%per_token;
        cached_k[dst]=k[i]; cached_v[dst]=v[i];
    }
    const std::size_t m=static_cast<std::size_t>(blockIdx.x)*blockDim.x+threadIdx.x;
    if (m < n_elements/(static_cast<std::size_t>(kv_heads)*dim)) {
        const int b=static_cast<int>(m/tokens), t=static_cast<int>(m%tokens);
        const std::size_t dst=static_cast<std::size_t>(b)*capacity+start+t;
        cached_valid[dst]=mask?mask[m]:1u;
        cached_ids[dst]=ids?ids[m]:0;
    }
}

template<class T> __global__ void attend_chunk_kernel(
    const T* q, const T* cached_k, const T* cached_v,
    const std::uint8_t* key_valid, const std::int32_t* key_seq,
    const float* rel_states, const float* rel_bank, const float* sink,
    const std::int32_t* current_seq_ids, T* out,
    int batch, int tokens, int start, int capacity, int kv_heads, int dim,
    int q_heads, int rel_dim, int rel_extent, int sliding_window) {
    const int head=static_cast<int>(blockIdx.x);
    const int local_token=static_cast<int>(blockIdx.y);
    const int b=static_cast<int>(blockIdx.z);
    const int lane=static_cast<int>(threadIdx.x);
    const int kv_head=head/(q_heads/kv_heads);
    const int q_pos=start+local_token;
    const std::size_t q_base=((static_cast<std::size_t>(b)*tokens+local_token)*q_heads+head)*dim;
    const std::size_t rel_base=((static_cast<std::size_t>(b)*tokens+local_token)*q_heads+head)*rel_dim;
    const std::size_t valid_base=static_cast<std::size_t>(b)*capacity;
    const std::size_t kv_base=valid_base*kv_heads*dim;
    const std::int32_t this_id=current_seq_ids?current_seq_ids[static_cast<std::size_t>(b)*tokens+local_token]:0;
    __shared__ float products[kThreads];
    __shared__ float current_logit;
    float numerator=0.f;
    float maximum=sink[(static_cast<std::size_t>(b)*tokens+local_token)*q_heads+head];
    float denominator=1.f;
    const float scaling=rsqrtf(static_cast<float>(dim));
    const int lower=q_pos-sliding_window+1;
    const int first=(sliding_window>0&&lower>0)?lower:0;
    for (int key_pos=first; key_pos<=q_pos; ++key_pos) {
        if (!key_valid[valid_base+key_pos] || key_seq[valid_base+key_pos]!=this_id) continue;
        const std::size_t kv_idx=kv_base+(static_cast<std::size_t>(key_pos)*kv_heads+kv_head)*dim;
        products[lane]=(lane<dim)?as_float(q[q_base+lane])*as_float(cached_k[kv_idx+lane]):0.f;
        __syncthreads();
        if (lane==0) {
            float dot=0.f;
            for (int i=0;i<kThreads;++i) dot+=products[i];
            float bias=0.f;
            const int distance=q_pos-key_pos;
            if (rel_dim && distance < rel_extent) {
                for (int r=0;r<rel_dim;++r)
                    bias=fmaf(rel_states[rel_base+r],rel_bank[static_cast<std::size_t>(r)*rel_extent+distance],bias);
            }
            current_logit=dot*scaling+bias;
        }
        __syncthreads();
        const float logit=current_logit;
        const float new_max=fmaxf(maximum,logit);
        const float alpha=expf(maximum-new_max),beta=expf(logit-new_max);
        denominator=denominator*alpha+beta;
        if (lane<dim) numerator=numerator*alpha+beta*as_float(cached_v[kv_idx+lane]);
        maximum=new_max;
        __syncthreads();
    }
    if(lane<dim) out[q_base+lane]=as_type<T>(numerator/denominator);
}

bool valid_type(ForwardScalar dtype) {return dtype==ForwardScalar::float32 || dtype==ForwardScalar::bfloat16;}
}

AttentionKv48::~AttentionKv48() noexcept {release();}
void AttentionKv48::release() noexcept {
    if(keys_) cudaFree(keys_);
    if(values_) cudaFree(values_);
    if(valid_) cudaFree(valid_);
    if(seq_) cudaFree(seq_);
    keys_=values_=nullptr;valid_=nullptr;seq_=nullptr;
    batch_=kv_heads_=dim_=capacity_=0;total_words_=0;
    std::memset(lengths_,0,sizeof(lengths_));
    std::memset(head_count_,0,sizeof(head_count_));
}

cudaError_t AttentionKv48::create(int batch,int kv_heads,int dim,int capacity,
                                 ForwardScalar dtype,cudaStream_t stream) noexcept {
    if(ready()||batch<=0||kv_heads<=0||dim<=0||dim>256||capacity<=0||
       capacity>65536||!valid_type(dtype))return cudaErrorInvalidValue;
    const std::size_t max=std::numeric_limits<std::size_t>::max();
    if(static_cast<std::size_t>(batch)>max/48/capacity/kv_heads/dim/sizeof(float))
        return cudaErrorInvalidValue;
    const std::size_t words=48ull*batch*capacity*kv_heads*dim;
    const std::size_t n_mask=48ull*batch*capacity;
    const std::size_t bytes=words*(dtype==ForwardScalar::float32?sizeof(float):sizeof(__nv_bfloat16));
    cudaError_t e=cudaMalloc(&keys_,bytes);
    if(e!=cudaSuccess){release();return e;}
    e=cudaMalloc(&values_,bytes);if(e!=cudaSuccess){release();return e;}
    e=cudaMalloc(reinterpret_cast<void**>(&valid_),n_mask);if(e!=cudaSuccess){release();return e;}
    e=cudaMalloc(reinterpret_cast<void**>(&seq_),n_mask*sizeof(std::int32_t));if(e!=cudaSuccess){release();return e;}
    batch_=batch;kv_heads_=kv_heads;dim_=dim;capacity_=capacity;dtype_=dtype;total_words_=words;
    e=reset(stream);if(e!=cudaSuccess){release();return e;}
    return cudaSuccess;
}

cudaError_t AttentionKv48::reset(cudaStream_t stream) noexcept {
    if(!ready())return cudaErrorInvalidValue;
    std::memset(lengths_,0,sizeof(lengths_));
    std::memset(head_count_,0,sizeof(head_count_));
    return cudaMemsetAsync(valid_,0,48ull*batch_*capacity_,stream);
}
int AttentionKv48::size(int physical_layer,int core_pass) const noexcept {
    const int slot=inkling2_cache_slot(physical_layer,core_pass);
    return slot<0?-1:lengths_[slot];
}

cudaError_t AttentionKv48::append_and_attend(
    int physical_layer,int core_pass,const void* q,const void* k,const void* v,
    const float* rel_states,const float* rel_bank,const float* sink_logits,
    const std::uint8_t* key_valid,const std::int32_t* seq_ids,void* y,
    int tokens,int q_heads,int rel_dim,int rel_extent,int sliding_window,
    cudaStream_t stream) noexcept {
    const int slot=inkling2_cache_slot(physical_layer,core_pass);
    if(!ready()||slot<0||!q||!k||!v||!y||!sink_logits||q==y||
       tokens<=0||tokens>capacity_-lengths_[slot]||q_heads<=0||q_heads%kv_heads_!=0||
       rel_dim<0||rel_dim>256||rel_extent<0||rel_extent>65536||
       (rel_dim>0&&(!rel_states||!rel_bank||rel_extent==0))||
       (rel_dim==0&&(rel_states||rel_bank||rel_extent!=0))||
       sliding_window<0||
       (head_count_[slot]!=0&&head_count_[slot]!=q_heads))return cudaErrorInvalidValue;
    const int start=lengths_[slot];
    const std::size_t slot_words=static_cast<std::size_t>(batch_)*capacity_*kv_heads_*dim_;
    const std::size_t slot_mask=static_cast<std::size_t>(batch_)*capacity_;
    const std::size_t new_elements=static_cast<std::size_t>(batch_)*tokens*kv_heads_*dim_;
    if(new_elements>(static_cast<std::size_t>(std::numeric_limits<unsigned>::max())*kThreads))return cudaErrorInvalidValue;
    const unsigned blocks=static_cast<unsigned>((new_elements+kThreads-1)/kThreads);
    const dim3 grid(static_cast<unsigned>(q_heads),static_cast<unsigned>(tokens),static_cast<unsigned>(batch_));
    if(dtype_==ForwardScalar::float32) {
        auto* kc=static_cast<float*>(keys_)+slot*slot_words;
        auto* vc=static_cast<float*>(values_)+slot*slot_words;
        append_kv_kernel<float><<<blocks,kThreads,0,stream>>>(static_cast<const float*>(k),static_cast<const float*>(v),key_valid,seq_ids,kc,vc,valid_+slot*slot_mask,seq_+slot*slot_mask,tokens,start,capacity_,kv_heads_,dim_,new_elements);
        cudaError_t e=cudaGetLastError();if(e!=cudaSuccess)return e;
        attend_chunk_kernel<float><<<grid,kThreads,0,stream>>>(static_cast<const float*>(q),kc,vc,valid_+slot*slot_mask,seq_+slot*slot_mask,rel_states,rel_bank,sink_logits,seq_ids,static_cast<float*>(y),batch_,tokens,start,capacity_,kv_heads_,dim_,q_heads,rel_dim,rel_extent,sliding_window);
    } else {
        auto* kc=static_cast<__nv_bfloat16*>(keys_)+slot*slot_words;
        auto* vc=static_cast<__nv_bfloat16*>(values_)+slot*slot_words;
        append_kv_kernel<__nv_bfloat16><<<blocks,kThreads,0,stream>>>(static_cast<const __nv_bfloat16*>(k),static_cast<const __nv_bfloat16*>(v),key_valid,seq_ids,kc,vc,valid_+slot*slot_mask,seq_+slot*slot_mask,tokens,start,capacity_,kv_heads_,dim_,new_elements);
        cudaError_t e=cudaGetLastError();if(e!=cudaSuccess)return e;
        attend_chunk_kernel<__nv_bfloat16><<<grid,kThreads,0,stream>>>(static_cast<const __nv_bfloat16*>(q),kc,vc,valid_+slot*slot_mask,seq_+slot*slot_mask,rel_states,rel_bank,sink_logits,seq_ids,static_cast<__nv_bfloat16*>(y),batch_,tokens,start,capacity_,kv_heads_,dim_,q_heads,rel_dim,rel_extent,sliding_window);
    }
    const cudaError_t e=cudaGetLastError();
    if(e==cudaSuccess){lengths_[slot]+=tokens;head_count_[slot]=q_heads;}
    return e;
}
} // namespace ninfer::models::inkling2
