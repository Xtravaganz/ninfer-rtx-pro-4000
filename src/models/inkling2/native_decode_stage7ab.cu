// Inkling 2 Mini: opt-in single-token MoE, source-order Stage6G banks.
// Stage7A NVFP4 + Stage7B MXFP8 and parallel sigmoid router.
// FP32 activation/accumulation, no new quantization or bank copies.
#include "models/inkling2/native_resident_stage6gh.h"
#include <cuda_runtime.h>
#include <math_constants.h>
#include <cstdint>
#include <cstddef>
#include <cmath>
#include <new>

namespace {
constexpr int H=2304,I=1152,E=128,K=8;
static_assert(H%32==0 && I%32==0 && E==128 && K==8);
struct Scratch { float scores[E]; int ids[K]; float coeff[K]; float act[K*I]; };
static_assert(offsetof(Scratch,act)%16==0);
struct Workspace { Scratch* s=nullptr; int device=-1; };
__device__ __forceinline__ float e4m3(unsigned b) {
    if ((b&127)==127) return CUDART_NAN_F;
    unsigned sign=(b&128)<<24, e=(b>>3)&15, m=b&7;
    return __uint_as_float(sign | (e ? ((e+120)<<23)|(m<<20) : __float_as_uint(float(m)*0x1p-9f)));
}
__device__ __forceinline__ float ue8m0(unsigned b) {
    return b==255 ? CUDART_NAN_F : __uint_as_float(b==0 ? 0x00400000u : b<<23);
}
__device__ __forceinline__ float e2m1(unsigned b) {
    const unsigned magnitude=(0xc8643210u >> (4*(b&7)))&15;
    return (b&8) ? -0.5f*float(magnitude) : 0.5f*float(magnitude);
}
__device__ __forceinline__ float warp_sum(float v) {
    #pragma unroll
    for(int d=16;d;d>>=1) v+=__shfl_down_sync(0xffffffffu,v,d);
    return v;
}
__global__ void router_scores(const float* x,const float* w,Scratch* s) {
    int lane=threadIdx.x&31, expert=blockIdx.x*4+(threadIdx.x>>5);
    const float4* a=reinterpret_cast<const float4*>(x);
    const float4* b=reinterpret_cast<const float4*>(w+expert*H);
    float v=0;
    for(int j=lane;j<H/4;j+=32){
        float4 p=a[j],q=b[j];
        v=fmaf(p.x,q.x,v);v=fmaf(p.y,q.y,v);v=fmaf(p.z,q.z,v);v=fmaf(p.w,q.w,v);
    }
    v=warp_sum(v);
    if(!lane)s->scores[expert]=1.f/(1.f+expf(-v));
}
__device__ __forceinline__ bool better(float a,int ai,float b,int bi){return a>b||(a==b&&ai<bi);}
__global__ void router_topk(const float* bias,float scale,Scratch* s){
    int e=threadIdx.x,lane=e&31,warp=e>>5;
    float score=s->scores[e],rank=score+bias[e];
    // Rank by corrected sigmoid; coefficient uses the uncorrected score.
    __shared__ float maxes[4]; __shared__ int winners[4],chosen;
    for(int r=0;r<K;++r){
        float v=rank;int idx=e;
        #pragma unroll
        for(int d=16;d;d>>=1){
            float a=__shfl_down_sync(0xffffffffu,v,d);int b=__shfl_down_sync(0xffffffffu,idx,d);
            if(lane+d<32&&better(a,b,v,idx)){v=a;idx=b;}
        }
        if(!lane){maxes[warp]=v;winners[warp]=idx;}
        __syncthreads();
        if(!e){int best=0;for(int j=1;j<4;++j)if(better(maxes[j],winners[j],maxes[best],winners[best]))best=j;
            chosen=winners[best];s->ids[r]=chosen;s->coeff[r]=s->scores[chosen];}
        __syncthreads();
        if(e==chosen)rank=-CUDART_INF_F;
        __syncthreads();
    }
    if(!e){float total=0;for(int r=0;r<K;++r)total+=s->coeff[r];float m=scale/(total+1e-20f);
        for(int r=0;r<K;++r)s->coeff[r]*=m;}
}
// Diagnostic only: reproduces Stage6E's serial accumulation and ranking.
__global__ void router_reference(const float* x,const float* w,const float* bias,float scale,int* ids,float* coeff){
    int e=threadIdx.x;__shared__ float scores[E],rank[E];float dot=0;
    for(int j=0;j<H;++j)dot=fmaf(x[j],w[e*H+j],dot);
    scores[e]=1.f/(1.f+expf(-dot));rank[e]=scores[e]+bias[e];__syncthreads();
    if(!e){float total=0;for(int r=0;r<K;++r){int best=-1;float v=-CUDART_INF_F;
        for(int j=0;j<E;++j)if(rank[j]>v){v=rank[j];best=j;}
        ids[r]=best;coeff[r]=scores[best];total+=coeff[r];rank[best]=-CUDART_INF_F;}
        float m=scale/(total+1e-20f);for(int r=0;r<K;++r)coeff[r]*=m;}
}
struct Matrix {const unsigned char* codes;const unsigned char* scales;float factor;};
__device__ __forceinline__ Matrix matrix(const std::uint64_t* ptr,const float* factors,int eid,int plane,int fmt){
    return {reinterpret_cast<const unsigned char*>(ptr[plane*2*E+eid]),
            reinterpret_cast<const unsigned char*>(ptr[(plane*2+1)*E+eid]),
            fmt==1?factors[plane*2*E+eid]/factors[(plane*2+1)*E+eid]:1.f};
}
__device__ __forceinline__ bool valid_route(const std::uint64_t* ptr,const int* valid,int id){
    return id>=0&&id<E&&valid[id]==id&&ptr[id]&&ptr[E+id]&&ptr[2*E+id]&&ptr[3*E+id];
}
// One lane per scale group; packed load, reuse scale for the entire group.
// Native source order, gate/up interleaved. No FP4 activation rounding.
template<int FMT> __device__ __forceinline__ float group_dot(Matrix m,int row,int cols,int group,const float* input){
    constexpr int G=FMT==1?16:32;
    const auto* codes=m.codes+std::size_t(row)*cols/(FMT==1?2:1)+group*G/(FMT==1?2:1);
    const float factor=(FMT==1?e4m3(m.scales[row*(cols/G)+group]):ue8m0(m.scales[row*(cols/G)+group]))*m.factor;
    const float4* x=reinterpret_cast<const float4*>(input+group*G);float sum=0;
    if constexpr(FMT==1){
        uint2 v=*reinterpret_cast<const uint2*>(codes);
        #pragma unroll
        for(int j=0;j<4;++j){unsigned word=j<2?v.x:v.y;word>>=(j&1)*16;float4 a=x[j];
            sum=fmaf(e2m1(word&15),a.x,sum);sum=fmaf(e2m1((word>>4)&15),a.y,sum);
            sum=fmaf(e2m1((word>>8)&15),a.z,sum);sum=fmaf(e2m1((word>>12)&15),a.w,sum);}
    }else{
        const uint4 a=*reinterpret_cast<const uint4*>(codes),b=*reinterpret_cast<const uint4*>(codes+16);
        const unsigned words[8]={a.x,a.y,a.z,a.w,b.x,b.y,b.z,b.w};
        #pragma unroll
        for(int j=0;j<8;++j){unsigned word=words[j];float4 v=x[j];
            sum=fmaf(e4m3(word&255),v.x,sum);sum=fmaf(e4m3((word>>8)&255),v.y,sum);
            sum=fmaf(e4m3((word>>16)&255),v.z,sum);sum=fmaf(e4m3(word>>24),v.w,sum);}
    }
    return sum*factor;
}
template<int FMT> __global__ void gate_up(const float* x,const std::uint64_t* ptr,const float* factors,const int* valid,Scratch* s){
    int lane=threadIdx.x&31,row=blockIdx.x*4+(threadIdx.x>>5),route=blockIdx.y,eid=s->ids[route];
    if(!valid_route(ptr,valid,eid)){if(!lane)s->act[route*I+row]=CUDART_NAN_F;return;}
    const Matrix m=matrix(ptr,factors,eid,0,FMT);
    constexpr int G=FMT==1?16:32;float g=0,u=0;
    for(int group=lane;group<H/G;group+=32){g+=group_dot<FMT>(m,2*row,H,group,x);u+=group_dot<FMT>(m,2*row+1,H,group,x);}
    g=warp_sum(g);u=warp_sum(u);
    if(!lane)s->act[route*I+row]=(4.f*tanhf(g/4.f)/(1.f+expf(-g)))*(25.f*tanhf(u/25.f));
}
template<int FMT,int Rows=4> __global__ void down_reduce(const std::uint64_t* ptr,const float* factors,const int* valid,const Scratch* s,float* out){
    int lane=threadIdx.x&31,route=threadIdx.x>>5,eid=s->ids[route];
    __shared__ float contribution[K][Rows];float sum[Rows]={};
    if(valid_route(ptr,valid,eid)){
        Matrix m=matrix(ptr,factors,eid,1,FMT);constexpr int G=FMT==1?16:32;
        const float* a=s->act+route*I;
        for(int group=lane;group<I/G;group+=32){
            #pragma unroll
            for(int r=0;r<Rows;++r)sum[r]+=group_dot<FMT>(m,blockIdx.x*Rows+r,I,group,a);
        }
    }else{for(int r=0;r<Rows;++r)sum[r]=CUDART_NAN_F;}
    #pragma unroll
    for(int r=0;r<Rows;++r){float v=warp_sum(sum[r]);if(!lane)contribution[route][r]=v;}
    __syncthreads();
    if(threadIdx.x<Rows){float result=0;for(int j=0;j<K;++j)result=fmaf(s->coeff[j],contribution[j][threadIdx.x],result);
        out[blockIdx.x*Rows+threadIdx.x]=result;}
}
__global__ void copy_routes(const Scratch* s,int* ids,float* coeff){int r=threadIdx.x;if(r<K){ids[r]=s->ids[r];coeff[r]=s->coeff[r];}}
inline bool aligned(std::uint64_t p){return p && !(p&15);}
}
extern "C" {
void* inkling7ab_create() noexcept {
    auto* w=new(std::nothrow) Workspace();if(!w)return nullptr;
    if(cudaGetDevice(&w->device)!=cudaSuccess||cudaMalloc(reinterpret_cast<void**>(&w->s),sizeof(Scratch))!=cudaSuccess){delete w;return nullptr;}return w;
}
void inkling7ab_destroy(void* raw) noexcept {
    auto* w=static_cast<Workspace*>(raw);if(!w)return;
    int old=-1;if(cudaGetDevice(&old)==cudaSuccess&&cudaSetDevice(w->device)==cudaSuccess){cudaFree(w->s);cudaSetDevice(old);}delete w;
}
// Caller serializes this workspace on one stream and keeps all input/store data alive.
int inkling7ab_run(void* store,void* raw,int layer,std::uint64_t x,std::uint64_t router,std::uint64_t bias,float scale,std::uint64_t out,std::uint64_t stream) noexcept {
    auto* w=static_cast<Workspace*>(raw);if(!w||!store||!aligned(x)||!aligned(router)||!aligned(out)||!bias||x==out||!std::isfinite(scale)||scale<=0)return int(cudaErrorInvalidValue);
    std::uint64_t ptr=0,fac=0,valid=0;int fmt=0;
    int rc=inkling6g_layer_view(store,layer,&ptr,&fac,&valid,&fmt);if(rc)return rc;
    if(fmt!=1&&fmt!=2)return int(cudaErrorInvalidValue);
    auto s=reinterpret_cast<cudaStream_t>(stream);auto* a=reinterpret_cast<const float*>(x);auto* p=reinterpret_cast<const std::uint64_t*>(ptr);auto* f=reinterpret_cast<const float*>(fac);auto* v=reinterpret_cast<const int*>(valid);auto* y=reinterpret_cast<float*>(out);
    router_scores<<<E/4,128,0,s>>>(a,reinterpret_cast<const float*>(router),w->s);auto e=cudaGetLastError();if(e!=cudaSuccess)return int(e);
    router_topk<<<1,E,0,s>>>(reinterpret_cast<const float*>(bias),scale,w->s);e=cudaGetLastError();if(e!=cudaSuccess)return int(e);
    if(fmt==1)gate_up<1><<<dim3(I/4,K),128,0,s>>>(a,p,f,v,w->s);else gate_up<2><<<dim3(I/4,K),128,0,s>>>(a,p,f,v,w->s);
    e=cudaGetLastError();if(e!=cudaSuccess)return int(e);
    if(fmt==1)down_reduce<1><<<H/4,K*32,0,s>>>(p,f,v,w->s,y);else down_reduce<2><<<H/4,K*32,0,s>>>(p,f,v,w->s,y);
    return int(cudaGetLastError());
}
int inkling7ab_routes(void* raw,std::uint64_t ids,std::uint64_t coeff,std::uint64_t stream) noexcept {
    auto* w=static_cast<Workspace*>(raw);if(!w||!ids||!coeff)return int(cudaErrorInvalidValue);
    copy_routes<<<1,K,0,reinterpret_cast<cudaStream_t>(stream)>>>(w->s,reinterpret_cast<int*>(ids),reinterpret_cast<float*>(coeff));return int(cudaGetLastError());
}
int inkling7ab_reference_routes(std::uint64_t x,std::uint64_t router,std::uint64_t bias,float scale,std::uint64_t ids,std::uint64_t coeff,std::uint64_t stream) noexcept {
    if(!x||!router||!bias||!ids||!coeff||!std::isfinite(scale)||scale<=0)return int(cudaErrorInvalidValue);
    router_reference<<<1,E,0,reinterpret_cast<cudaStream_t>(stream)>>>(reinterpret_cast<const float*>(x),reinterpret_cast<const float*>(router),reinterpret_cast<const float*>(bias),scale,reinterpret_cast<int*>(ids),reinterpret_cast<float*>(coeff));return int(cudaGetLastError());
}
}
