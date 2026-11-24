#include "models/inkling2/moe_top8_stage5e.h"
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

using namespace ninfer::models::inkling2;
namespace {
constexpr int H=kInklingHidden5E,I=kInklingIntermediate5E,E=kInklingExperts5E,R=kInklingTopK5E;
constexpr int RES=R;
void check(cudaError_t e,const char* label){if(e!=cudaSuccess)throw std::runtime_error(std::string(label)+": "+cudaGetErrorString(e));}
void req(bool ok,const char* msg){if(!ok)throw std::runtime_error(msg);}
struct Device {
    void* p=nullptr;
    explicit Device(std::size_t size){check(cudaMalloc(&p,size),"cudaMalloc");}
    ~Device(){if(p)cudaFree(p);}
    void put(const void* data,std::size_t size){check(cudaMemcpy(p,data,size,cudaMemcpyHostToDevice),"H2D");}
    void get(void* data,std::size_t size){check(cudaMemcpy(data,p,size,cudaMemcpyDeviceToHost),"D2H");}
    template<class T> T* cast()const{return static_cast<T*>(p);}
};
// Four nonzero words per output row (different G16 / G32 groups).
// All 2304 output rows and both real Inkling K dimensions are exercised.
int feature(int expert,int row,int tap,int cols){
    const int groups=cols/32;
    return ((expert*7+row*13+tap*19)%groups)*32+((tap*7+row*3+expert)%32);
}
std::uint8_t value_code(bool nv,int tap){
    static constexpr std::uint8_t fp4[]={2,0xb,4,0xd};
    static constexpr std::uint8_t fp8[]={0x38,0xb8,0x40,0xc0};
    return nv?fp4[tap]:fp8[tap];
}
float weight_value(bool nv,int expert,int tap){
    static constexpr float fp4[]={1.f,-1.5f,2.f,-3.f};
    static constexpr float fp8[]={1.f,-1.f,2.f,-2.f};
    return (nv?fp4[tap]*((1.25f+0.05f*expert)/1.5f):fp8[tap]);
}
struct HostBank {
    bool nv;
    int cols;
    std::vector<std::uint8_t> codes,scales;
    std::vector<float> amax,den;
    HostBank(bool is_nv,int width):nv(is_nv),cols(width),
        codes(static_cast<std::size_t>(RES)*H*width/(is_nv?2:1),0),
        scales(static_cast<std::size_t>(RES)*H*(width/(is_nv?16:32)),is_nv?0x38:127),
        amax(RES),den(RES,1.5f){
        for(int e=0;e<RES;++e){
            amax[e]=1.25f+0.05f*e;
            for(int row=0;row<H;++row)for(int tap=0;tap<4;++tap){
                const int col=feature(e,row,tap,cols);
                const std::size_t offset=(static_cast<std::size_t>(e)*H+row)*cols+col;
                const std::uint8_t word=value_code(nv,tap);
                if(nv){
                    const std::size_t idx=offset/2;
                    codes[idx]|=static_cast<std::uint8_t>((col&1)?word<<4:word);
                }else codes[offset]=word;
            }
        }
    }
};
struct GpuBank {
    Device codes,scales,amax,den;
    ExpertBank5E view;
    explicit GpuBank(const HostBank& b):codes(b.codes.size()),scales(b.scales.size()),amax(b.amax.size()*4),den(b.den.size()*4){
        codes.put(b.codes.data(),b.codes.size());scales.put(b.scales.data(),b.scales.size());
        amax.put(b.amax.data(),b.amax.size()*4);den.put(b.den.data(),b.den.size()*4);
        view.codes=codes.cast<std::uint8_t>();view.scales=scales.cast<std::uint8_t>();
        view.amax=amax.cast<float>();view.denominator=den.cast<float>();
        view.format=b.nv?ExpertFormat5E::nvfp4_e2m1_g16:ExpertFormat5E::mxfp8_e4m3_g32;
        view.resident=RES;view.rows=H;view.cols=b.cols;
    }
};
float activation(float gate,float up){
    return (4.f*std::tanh(gate/4.f)/(1.f+std::exp(-gate)))*(25.f*std::tanh(up/25.f));
}
void run_format(bool nv){
    HostBank h13(nv,H),h2(nv,I);
    GpuBank d13(h13),d2(h2);
    std::vector<float> input(H),router(E*H,0.f),correction(E,-3.f);
    std::vector<std::int32_t> mapping(E,-1);
    for(int i=0;i<H;++i)input[i]=0.24f*std::cos(i*0.017f)+0.11f*std::sin(i*0.03f);
    for(int e=0;e<E;++e){router[e*H]=0.005f*e;router[e*H+1]=-0.002f*e;}
    for(int i=0;i<RES;++i){mapping[i]=i;correction[i]=3.f;}
    Device dx(input.size()*4),drouter(router.size()*4),dbias(correction.size()*4),dmap(mapping.size()*4),dy(H*4);
    dx.put(input.data(),input.size()*4);drouter.put(router.data(),router.size()*4);
    dbias.put(correction.data(),correction.size()*4);dmap.put(mapping.data(),mapping.size()*4);
    MoEWorkspace5E work;check(work.create(),"create");
    req(work.run(nullptr,drouter.cast<float>(),dbias.cast<float>(),8.f,d13.view,d2.view,dmap.cast<std::int32_t>(),dy.cast<float>())==cudaErrorInvalidValue,"input contract");
    check(work.run(dx.cast<float>(),drouter.cast<float>(),dbias.cast<float>(),8.f,d13.view,d2.view,dmap.cast<std::int32_t>(),dy.cast<float>()),"run");
    std::vector<float> actual(H),weights(R);std::vector<std::int32_t> indices(R);
    dy.get(actual.data(),actual.size()*4);
    check(cudaMemcpy(indices.data(),work.top_ids(),R*4,cudaMemcpyDeviceToHost),"indices");
    check(cudaMemcpy(weights.data(),work.top_weights(),R*4,cudaMemcpyDeviceToHost),"weights");
    int missing=-1;check(cudaMemcpy(&missing,work.missing_resident_flag(),sizeof(int),cudaMemcpyDeviceToHost),"missing flag");
    req(missing==0,"selected experts not resident");
    for(int e:indices)req(e>=0&&e<RES,"top8 routing selected wrong expert");
    double sum_scores=0.0;
    for(int e:indices){const double dot=double(router[e*H])*input[0]+double(router[e*H+1])*input[1];sum_scores+=1/(1+std::exp(-dot));}
    for(int j=0;j<R;++j){
        const int e=indices[j];const double dot=double(router[e*H])*input[0]+double(router[e*H+1])*input[1];
        const double expected=8/(1+std::exp(-dot))/sum_scores;
        req(std::abs(weights[j]-expected)<2e-5,"router must ignore correction bias for weights");
    }
    std::vector<float> expected(H,0.f);
    for(int j=0;j<R;++j){
        const int e=indices[j];
        std::vector<float> activated(I,0.f);
        for(int neuron=0;neuron<I;++neuron){
            float gate=0.f,up=0.f;
            for(int tap=0;tap<4;++tap){
                gate+=weight_value(nv,e,tap)*input[feature(e,neuron,tap,H)];
                up+=weight_value(nv,e,tap)*input[feature(e,neuron+I,tap,H)];
            }
            activated[neuron]=activation(gate,up);
        }
        for(int row=0;row<H;++row){
            float value=0.f;
            for(int tap=0;tap<4;++tap)
                value+=weight_value(nv,e,tap)*activated[feature(e,row,tap,I)];
            expected[row]+=weights[j]*value;
        }
    }
    float max_err=0.f;
    for(int i=0;i<H;++i){
        const float error=std::abs(actual[i]-expected[i]);
        max_err=std::max(max_err,error);
        if(!(error <= 2e-4f+2e-4f*std::abs(expected[i]))){
            std::fprintf(stderr,"mismatch format=%d row=%d actual=%.8g ref=%.8g\n",nv?1:2,i,actual[i],expected[i]);
            throw std::runtime_error("Top8 MoE independent row oracle failed");
        }
    }
    std::printf("PASS Stage5E %s original W13[2304,2304]/W2[2304,1152], top8 max_error=%.9g\n",nv?"NVFP4":"MXFP8",max_err);
    // Contract: if router selects a nonresident expert, output cannot be used.
    correction[9]=100.f;dbias.put(correction.data(),correction.size()*4);
    check(work.run(dx.cast<float>(),drouter.cast<float>(),dbias.cast<float>(),8.f,d13.view,d2.view,dmap.cast<std::int32_t>(),dy.cast<float>()),"nonresident run");
    dy.get(actual.data(),actual.size()*4);
    check(cudaMemcpy(&missing,work.missing_resident_flag(),sizeof(int),cudaMemcpyDeviceToHost),"missing flag");
    req(missing==1&&std::isnan(actual[0]),"missing resident expert must invalidate output");
    std::printf("PASS Stage5E %s selected missing expert rejected\n",nv?"NVFP4":"MXFP8");
}
}
int main(){
    try{
        int count=0;check(cudaGetDeviceCount(&count),"cudaGetDeviceCount");req(count>0,"CUDA GPU required");
        cudaDeviceProp prop{};check(cudaGetDeviceProperties(&prop,0),"properties");
        std::printf("Stage5E GPU %s sm_%d%d\n",prop.name,prop.major,prop.minor);
        run_format(true);run_format(false);
        std::puts("OK Stage5E: Top8 biased selection + original-score weights + native NVFP4/MXFP8 expert MLP");
        return 0;
    }catch(const std::exception& exc){std::fprintf(stderr,"FAIL Stage5E: %s\n",exc.what());return 1;}
}
