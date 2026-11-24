#include "models/inkling2/attention_cache_stage5d.h"
#include "models/inkling2/conv_cache_stage5c.h"
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

using namespace ninfer::models::inkling2;
namespace {
void check(cudaError_t e,const char* s){if(e!=cudaSuccess)throw std::runtime_error(std::string(s)+": "+cudaGetErrorString(e));}
void require(bool ok,const char* s){if(!ok)throw std::runtime_error(s);}
struct Device {
    void* p=nullptr;
    explicit Device(std::size_t bytes){check(cudaMalloc(&p,bytes),"alloc");}
    ~Device(){if(p)cudaFree(p);}
    template<class T> T* ptr(){return static_cast<T*>(p);}
    void copy(const void* data,std::size_t bytes){check(cudaMemcpy(p,data,bytes,cudaMemcpyHostToDevice),"H2D");}
    void get(void* data,std::size_t bytes){check(cudaMemcpy(data,p,bytes,cudaMemcpyDeviceToHost),"D2H");}
};
constexpr int B=2,T=9,HQ=24,HK=8,D=128,RD=4,RE=6,CAP=16;
struct Inputs {
    std::vector<float> q,k,v,rel,sink,bank;
    std::vector<std::uint8_t> mask;
    std::vector<std::int32_t> seq;
    Inputs():q(B*T*HQ*D),k(B*T*HK*D),v(B*T*HK*D),
        rel(B*T*HQ*RD),sink(B*T*HQ),bank(RD*RE),mask(B*T),seq(B*T){
        for(std::size_t i=0;i<q.size();++i)q[i]=std::sin(float(i%113)*0.13f)*0.1f;
        for(std::size_t i=0;i<k.size();++i)k[i]=std::cos(float(i%193)*0.07f)*0.3f;
        for(std::size_t i=0;i<v.size();++i)v[i]=std::sin(float(i%97)*0.22f)*0.4f;
        for(std::size_t i=0;i<rel.size();++i)rel[i]=std::cos(float(i%29)*0.2f)*0.1f;
        for(std::size_t i=0;i<sink.size();++i)sink[i]=0.2f*std::cos(float(i%19));
        for(std::size_t i=0;i<bank.size();++i)bank[i]=std::sin(float(i+1)*0.13f)*0.3f;
        for(int b=0;b<B;++b)for(int t=0;t<T;++t){
            mask[b*T+t]=static_cast<std::uint8_t>(t!=3);
            seq[b*T+t]=t<5?10+b:20+b;
        }
    }
};
void golden(const Inputs& a,int window,std::vector<float>& result){
    result.assign(B*T*HQ*D,0.f);
    for(int b=0;b<B;++b)for(int t=0;t<T;++t)for(int h=0;h<HQ;++h){
        const auto qb=((b*T+t)*HQ+h)*D;
        const auto rb=((b*T+t)*HQ+h)*RD;
        const int kh=h/(HQ/HK);
        double mx=double(a.sink[(b*T+t)*HQ+h]);
        std::vector<double> logits(t+1,-INFINITY);
        for(int z=0;z<=t;++z){
            if((window>0&&t-z>=window)||!a.mask[b*T+z]||a.seq[b*T+z]!=a.seq[b*T+t])continue;
            const auto base=((b*T+z)*HK+kh)*D;
            double dot=0;
            for(int d=0;d<D;++d)dot+=double(a.q[qb+d])*a.k[base+d];
            double bias=0;
            if(t-z<RE)for(int d=0;d<RD;++d)bias+=double(a.rel[rb+d])*a.bank[d*RE+(t-z)];
            const double logit=dot/std::sqrt(double(D))+bias;
            logits[z]=logit;
            mx=std::max(mx,logit);
        }
        double den=std::exp(double(a.sink[(b*T+t)*HQ+h])-mx);
        for(int z=0;z<=t;++z)if(std::isfinite(logits[z]))den+=std::exp(logits[z]-mx);
        for(int d=0;d<D;++d){
            double num=0;
            for(int z=0;z<=t;++z)if(std::isfinite(logits[z]))
                num+=std::exp(logits[z]-mx)*double(a.v[((b*T+z)*HK+kh)*D+d]);
            result[qb+d]=float(num/den);
        }
    }
}
std::vector<float> run(const Inputs& a,const std::vector<int>& chunks,int window,int layer,int pass){
    AttentionKv48 cache;
    check(cache.create(B,HK,D,CAP,ForwardScalar::float32),"create");
    require(cache.size(layer,pass)==0,"initial size");
    Device bank(a.bank.size()*4);bank.copy(a.bank.data(),a.bank.size()*4);
    std::vector<float> out(B*T*HQ*D);
    int begin=0;
    for(int n:chunks){
        std::vector<float> q(B*n*HQ*D),k(B*n*HK*D),v(B*n*HK*D),rel(B*n*HQ*RD),sink(B*n*HQ);
        std::vector<std::uint8_t> mask(B*n);
        std::vector<std::int32_t> ids(B*n);
        for(int b=0;b<B;++b){
            std::copy_n(a.q.begin()+((b*T+begin)*HQ*D),n*HQ*D,q.begin()+b*n*HQ*D);
            std::copy_n(a.k.begin()+((b*T+begin)*HK*D),n*HK*D,k.begin()+b*n*HK*D);
            std::copy_n(a.v.begin()+((b*T+begin)*HK*D),n*HK*D,v.begin()+b*n*HK*D);
            std::copy_n(a.rel.begin()+((b*T+begin)*HQ*RD),n*HQ*RD,rel.begin()+b*n*HQ*RD);
            std::copy_n(a.sink.begin()+((b*T+begin)*HQ),n*HQ,sink.begin()+b*n*HQ);
            std::copy_n(a.mask.begin()+b*T+begin,n,mask.begin()+b*n);
            std::copy_n(a.seq.begin()+b*T+begin,n,ids.begin()+b*n);
        }
        Device dq(q.size()*4),dk(k.size()*4),dv(v.size()*4),dr(rel.size()*4),ds(sink.size()*4),dm(mask.size()),di(ids.size()*4),dy(q.size()*4);
        dq.copy(q.data(),q.size()*4);dk.copy(k.data(),k.size()*4);dv.copy(v.data(),v.size()*4);
        dr.copy(rel.data(),rel.size()*4);ds.copy(sink.data(),sink.size()*4);dm.copy(mask.data(),mask.size());di.copy(ids.data(),ids.size()*4);
        check(cache.append_and_attend(layer,pass,dq.p,dk.p,dv.p,dr.ptr<float>(),bank.ptr<float>(),ds.ptr<float>(),dm.ptr<std::uint8_t>(),di.ptr<std::int32_t>(),dy.p,n,HQ,RD,RE,window),"attention");
        std::vector<float> part(q.size());dy.get(part.data(),part.size()*4);
        for(int b=0;b<B;++b)
            std::copy_n(part.begin()+b*n*HQ*D,n*HQ*D,out.begin()+(b*T+begin)*HQ*D);
        begin+=n;
        require(cache.size(layer,pass)==begin,"cache length");
    }
    require(begin==T,"complete chunks");
    // Core layers must NOT share slots across passes.
    if(layer>=6&&layer<18){
        const int other=(pass+1)%3;
        require(cache.size(layer,other)==0,"independent per-pass K/V cache");
    }
    require(cache.append_and_attend(layer,pass,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,1,HQ,RD,RE,window)==cudaErrorInvalidValue,"invalid contract");
    check(cache.reset(),"reset");
    require(cache.size(layer,pass)==0,"reset length");
    return out;
}


void interleaved_passes(){
    constexpr int dim=8;
    AttentionKv48 cache;
    check(cache.create(1,1,dim,3,ForwardScalar::float32),"interleaved create");
    std::vector<float> q(dim,0.f),k(dim,0.f),v(dim,0.f),sink(1,0.f);
    Device dq(dim*4),dk(dim*4),dv(dim*4),ds(4),dy(dim*4);
    dq.copy(q.data(),dim*4);dk.copy(k.data(),dim*4);ds.copy(sink.data(),4);
    for(int pass=0;pass<3;++pass){
        std::fill(v.begin(),v.end(),float(1+2*pass));dv.copy(v.data(),dim*4);
        check(cache.append_and_attend(6,pass,dq.p,dk.p,dv.p,nullptr,nullptr,ds.ptr<float>(),nullptr,nullptr,dy.p,1,1,0,0,0),"interleaved append");
        std::vector<float> got(dim);dy.get(got.data(),dim*4);
        for(float number:got)require(std::abs(number-0.5f*float(1+2*pass))<1e-5f,"pass values mixed");
    }
    for(int pass=0;pass<3;++pass)require(cache.size(6,pass)==1,"pass length mismatch");
    std::fill(v.begin(),v.end(),7.f);dv.copy(v.data(),dim*4);
    check(cache.append_and_attend(6,0,dq.p,dk.p,dv.p,nullptr,nullptr,ds.ptr<float>(),nullptr,nullptr,dy.p,1,1,0,0,0),"pass0 second token");
    std::vector<float> got(dim);dy.get(got.data(),dim*4);
    for(float number:got)require(std::abs(number-(1.f+7.f)/3.f)<1e-5f,"pass0 state corrupted");
    require(cache.size(6,1)==1&&cache.size(6,2)==1,"other pass length modified");
    std::puts("PASS Stage5D interleaved 3-pass K/V slot isolation");
}

void bf16_case(){
    constexpr int count=3,heads=4,kvheads=2,dim=16;
    AttentionKv48 cache;
    check(cache.create(1,kvheads,dim,4,ForwardScalar::bfloat16),"BF16 cache create");
    std::vector<__nv_bfloat16> q(count*heads*dim),k(count*kvheads*dim),v(count*kvheads*dim);
    for(auto& word:q)word=__float2bfloat16_rn(0.f);
    for(auto& word:k)word=__float2bfloat16_rn(0.f);
    for(int t=0;t<count;++t)for(int i=0;i<kvheads*dim;++i)
        v[t*kvheads*dim+i]=__float2bfloat16_rn(float(t+1));
    std::vector<float> sink(count*heads,0.f);
    Device dq(q.size()*2),dk(k.size()*2),dv(v.size()*2),ds(sink.size()*4),dy(q.size()*2);
    dq.copy(q.data(),q.size()*2);dk.copy(k.data(),k.size()*2);dv.copy(v.data(),v.size()*2);
    ds.copy(sink.data(),sink.size()*4);
    check(cache.append_and_attend(23,0,dq.p,dk.p,dv.p,nullptr,nullptr,ds.ptr<float>(),nullptr,nullptr,dy.p,count,heads,0,0,0),"BF16 append/attend");
    std::vector<__nv_bfloat16> result(q.size());dy.get(result.data(),result.size()*2);
    for(int t=0;t<count;++t)for(int i=0;i<heads*dim;++i){
        const float expected=float((t+1)*(t+2))/float(2*(t+2)); // = (t+1)/2
        const float actual=__bfloat162float(result[t*heads*dim+i]);
        require(std::abs(actual-expected)<1e-5f,"BF16 sink attention values");
    }
    std::puts("PASS Stage5D BF16 cache + zero relative-state geometry");
}

void check_case(int layer,int pass,int window){
    Inputs a;
    std::vector<float> expected;golden(a,window,expected);
    const auto full=run(a,{T},window,layer,pass);
    const auto chunks=run(a,{2,1,3,3},window,layer,pass);
    float max_ref=0.f,max_par=0.f;
    for(std::size_t i=0;i<full.size();++i){
        max_ref=std::max(max_ref,std::abs(expected[i]-full[i]));
        max_par=std::max(max_par,std::abs(full[i]-chunks[i]));
    }
    std::printf("PASS Stage5D layer=%d pass=%d window=%d max_vs_fp64=%.8g chunk_parity=%.8g\n",layer,pass,window,max_ref,max_par);
    require(max_ref<2e-5f&&max_par<1e-6f,"GPU GQA mismatch");
}
}
int main(){
    try{
        int count=0;check(cudaGetDeviceCount(&count),"device count");require(count>0,"requires CUDA GPU");
        cudaDeviceProp prop{};check(cudaGetDeviceProperties(&prop,0),"properties");
        std::printf("Stage5D GPU %s sm_%d%d\n",prop.name,prop.major,prop.minor);
        require(inkling2_cache_slot(6,0)==6&&inkling2_cache_slot(6,1)==24&&inkling2_cache_slot(6,2)==36,"cache slot mapping");
        require(inkling2_cache_slot(23,1)==-1,"invalid core-pass mapping");
        check_case(5,0,0);check_case(11,0,5);check_case(6,0,6);check_case(6,1,6);check_case(6,2,6);bf16_case();interleaved_passes();
        std::puts("OK Stage5D: 48-slot GPU K/V, sink+relative bias, full/sliding, prefill/decode chunk parity");
        return 0;
    }catch(const std::exception& e){std::fprintf(stderr,"FAIL Stage5D: %s\n",e.what());return 1;}
}
