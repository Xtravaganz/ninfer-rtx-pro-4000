#include "models/inkling2/native_resident_stage6gh.h"
#include <cuda_runtime.h>
#include <math_constants.h>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <new>

// Native multi-token expert operator, not a 48-visit full model forward.
// The source-order NVFP4 W13 rows are interleaved (even=gate, odd=up).
namespace {
constexpr int H=2304,I=1152,E=128,K=8;
__device__ __forceinline__ float fp8e4m3(unsigned w) {
    if((w&0x7fu)==0x7fu)return __uint_as_float(0x7fc00000u);
    const unsigned sign=(w&0x80u)<<24,exp=(w>>3)&15u,man=w&7u;
    if(!exp)return __uint_as_float(__float_as_uint(float(man)*0x1p-9f)|sign);
    return __uint_as_float(sign|((exp+120u)<<23)|(man<<20));
}
__device__ __forceinline__ float scaleue8m0(unsigned w) {
    return w==255 ? CUDART_NAN_F : __uint_as_float(w==0?0x00400000u:w<<23);
}
__device__ __forceinline__ float fp4e2m1(unsigned nib) {
    constexpr float mag[8]={0.f,.5f,1.f,1.5f,2.f,3.f,4.f,6.f};
    return (nib&8)?-mag[nib&7]:mag[nib&7];
}
__global__ void route_batch(const float* x,const float* w,const float* bias,
                            float route_scale,int tokens,int* ids,float* coeff) {
    int t=blockIdx.x,e=threadIdx.x;
    if(t>=tokens)return;
    __shared__ float signal[E], ranked[E];
    float dot=0;
    for(int j=0;j<H;++j)dot=fmaf(x[std::size_t(t)*H+j],w[std::size_t(e)*H+j],dot);
    const float score=1.f/(1.f+expf(-dot));
    signal[e]=score;ranked[e]=score+bias[e];
    __syncthreads();
    if(!e){
        float sum=0;
        for(int r=0;r<K;++r){
            int best=-1;float value=-CUDART_INF_F;
            for(int j=0;j<E;++j)if(ranked[j]>value){best=j;value=ranked[j];}
            ids[t*K+r]=best;coeff[t*K+r]=signal[best];sum+=signal[best];
            ranked[best]=-CUDART_INF_F;
        }
        const float s=route_scale/(sum+1e-20f);
        for(int r=0;r<K;++r)coeff[t*K+r]*=s;
    }
}
__global__ void check_batch(int tokens,const int* ids,const std::uint64_t* ptrs,
                            const int* valid,int* missing) {
    const int t=blockIdx.x;
    if(t>=tokens||threadIdx.x)return;
    int bad=0;
    for(int r=0;r<K;++r) {
        const int e=ids[t*K+r];
        if(e<0||e>=E||valid[e]!=e||!ptrs[e]||!ptrs[E+e]||!ptrs[2*E+e]||!ptrs[3*E+e])bad=1;
    }
    missing[t]=bad;
}
template<int FMT,bool PerRoute> __global__ void matvec_batch(
    const std::uint64_t* ptrs,const float* factors,const float* x,
    const int* ids,const int* valid,float* out,int rows,int cols,int plane) {
    const int row=blockIdx.x,flat=blockIdx.y,lane=threadIdx.x;
    const int token=flat/K,expert=ids[flat];
    if(expert<0||expert>=E||valid[expert]!=expert||
       !ptrs[plane*2*E+expert]||!ptrs[(plane*2+1)*E+expert]){
        if(!lane)out[std::size_t(flat)*rows+row]=CUDART_NAN_F;
        return;
    }
    const auto* code=reinterpret_cast<const std::uint8_t*>(
        static_cast<std::uintptr_t>(ptrs[plane*2*E+expert]));
    const auto* scale=reinterpret_cast<const std::uint8_t*>(
        static_cast<std::uintptr_t>(ptrs[(plane*2+1)*E+expert]));
    const float factor=FMT==1?
        factors[plane*2*E+expert]/factors[(plane*2+1)*E+expert]:1.f;
    float sum=0;
    for(int col=lane;col<cols;col+=128){
        const auto index=std::size_t(row)*cols+col;
        float weight;
        if constexpr(FMT==1){
            const std::uint8_t q=code[index/2];
            weight=fp4e2m1((col&1)?q>>4:q&15)*
                fp8e4m3(scale[std::size_t(row)*(cols/16)+col/16])*factor;
        }else{
            weight=fp8e4m3(code[index])*
                scaleue8m0(scale[std::size_t(row)*(cols/32)+col/32]);
        }
        const auto input=PerRoute?std::size_t(flat)*cols+col:std::size_t(token)*cols+col;
        sum=fmaf(weight,x[input],sum);
    }
    __shared__ float temp[128]; temp[lane]=sum;
    __syncthreads();
    for(int shift=64;shift>0;shift>>=1){
        if(lane<shift)temp[lane]+=temp[lane+shift];
        __syncthreads();
    }
    if(lane==0)out[std::size_t(flat)*rows+row]=temp[0];
}
__global__ void activate_batch(const float* projected,float* activation,int tokens){
    const int idx=blockIdx.x*blockDim.x+threadIdx.x;
    if(idx>=tokens*K*I)return;
    const int route=idx/I,neuron=idx%I;
    const float gate=projected[route*2*I+2*neuron];
    const float up=projected[route*2*I+2*neuron+1];
    const float a=4.f*tanhf(gate/4.f)/(1.f+expf(-gate));
    const float b=25.f*tanhf(up/25.f);
    activation[idx]=a*b;
}
__global__ void merge_batch(const float* down,const float* coeff,
                            const int* missing,float* out,int tokens){
    const int idx=blockIdx.x*blockDim.x+threadIdx.x;
    if(idx>=tokens*H)return;
    const int token=idx/H,row=idx%H;
    if(missing[token]){out[idx]=CUDART_NAN_F;return;}
    float total=0.f;
    for(int r=0;r<K;++r)total=fmaf(coeff[token*K+r],down[(std::size_t(token)*K+r)*H+row],total);
    out[idx]=total;
}
struct BatchWorkspace{
    int max_tokens=0, *ids=nullptr,*missing=nullptr;
    float *coeff=nullptr,*up=nullptr,*act=nullptr,*down=nullptr;
    explicit BatchWorkspace(int n):max_tokens(n){}
    ~BatchWorkspace()noexcept {
        if(ids)cudaFree(ids);if(missing)cudaFree(missing);if(coeff)cudaFree(coeff);
        if(up)cudaFree(up);if(act)cudaFree(act);if(down)cudaFree(down);
    }
    cudaError_t allocate()noexcept{
        const auto c=std::size_t(max_tokens);
        cudaError_t e;
        if((e=cudaMalloc(reinterpret_cast<void**>(&ids),c*K*sizeof(int)))!=cudaSuccess)return e;
        if((e=cudaMalloc(reinterpret_cast<void**>(&missing),c*sizeof(int)))!=cudaSuccess)return e;
        if((e=cudaMalloc(reinterpret_cast<void**>(&coeff),c*K*sizeof(float)))!=cudaSuccess)return e;
        if((e=cudaMalloc(reinterpret_cast<void**>(&up),c*K*2*I*sizeof(float)))!=cudaSuccess)return e;
        if((e=cudaMalloc(reinterpret_cast<void**>(&act),c*K*I*sizeof(float)))!=cudaSuccess)return e;
        return cudaMalloc(reinterpret_cast<void**>(&down),c*K*H*sizeof(float));
    }
};
}
extern "C" {
void* inkling6i_workspace_create(int max_tokens) noexcept {
    if(max_tokens<1||max_tokens>128)return nullptr;
    auto* w=new(std::nothrow) BatchWorkspace(max_tokens);
    if(!w)return nullptr;
    if(w->allocate()!=cudaSuccess){delete w;return nullptr;}
    return w;
}
void inkling6i_workspace_destroy(void* p)noexcept{delete static_cast<BatchWorkspace*>(p);}
int inkling6i_run(void* raw,void* store,int layer,std::uint64_t hidden,
                 std::uint64_t router,std::uint64_t bias,float scale,int tokens,
                 std::uint64_t output,std::uint64_t stream_value) noexcept {
    auto* w=static_cast<BatchWorkspace*>(raw);
    if(!w||tokens<1||tokens>w->max_tokens||!hidden||!router||!bias||!output||
       !std::isfinite(scale)||scale<=0)return int(cudaErrorInvalidValue);
    std::uint64_t ptrs=0,factors=0,valid=0;int fmt=0;
    const int rc=inkling6g_layer_view(store,layer,&ptrs,&factors,&valid,&fmt);
    if(rc)return rc;
    auto* s=reinterpret_cast<cudaStream_t>(stream_value);
    route_batch<<<tokens,E,0,s>>>(reinterpret_cast<const float*>(hidden),
        reinterpret_cast<const float*>(router),reinterpret_cast<const float*>(bias),
        scale,tokens,w->ids,w->coeff);
    if(auto e=cudaGetLastError();e!=cudaSuccess)return int(e);
    check_batch<<<tokens,32,0,s>>>(tokens,w->ids,
        reinterpret_cast<const std::uint64_t*>(ptrs),reinterpret_cast<const int*>(valid),w->missing);
    if(auto e=cudaGetLastError();e!=cudaSuccess)return int(e);
    const dim3 up_grid(2*I,tokens*K),down_grid(H,tokens*K);
    if(fmt==1){
        matvec_batch<1,false><<<up_grid,128,0,s>>>(
            reinterpret_cast<const std::uint64_t*>(ptrs),reinterpret_cast<const float*>(factors),
            reinterpret_cast<const float*>(hidden),w->ids,reinterpret_cast<const int*>(valid),
            w->up,2*I,H,0);
    }else{
        matvec_batch<2,false><<<up_grid,128,0,s>>>(
            reinterpret_cast<const std::uint64_t*>(ptrs),reinterpret_cast<const float*>(factors),
            reinterpret_cast<const float*>(hidden),w->ids,reinterpret_cast<const int*>(valid),
            w->up,2*I,H,0);
    }
    if(auto e=cudaGetLastError();e!=cudaSuccess)return int(e);
    activate_batch<<<(tokens*K*I+255)/256,256,0,s>>>(w->up,w->act,tokens);
    if(auto e=cudaGetLastError();e!=cudaSuccess)return int(e);
    if(fmt==1){
        matvec_batch<1,true><<<down_grid,128,0,s>>>(
            reinterpret_cast<const std::uint64_t*>(ptrs),reinterpret_cast<const float*>(factors),
            w->act,w->ids,reinterpret_cast<const int*>(valid),w->down,H,I,1);
    }else{
        matvec_batch<2,true><<<down_grid,128,0,s>>>(
            reinterpret_cast<const std::uint64_t*>(ptrs),reinterpret_cast<const float*>(factors),
            w->act,w->ids,reinterpret_cast<const int*>(valid),w->down,H,I,1);
    }
    if(auto e=cudaGetLastError();e!=cudaSuccess)return int(e);
    merge_batch<<<(tokens*H+255)/256,256,0,s>>>(w->down,w->coeff,w->missing,
        reinterpret_cast<float*>(output),tokens);
    return int(cudaGetLastError());
}
}
