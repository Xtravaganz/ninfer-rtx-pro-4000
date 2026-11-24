// Stage 5G: execute Stage-5E CUDA Top-8 MoE on REAL v3-staged Inkling weights.
// Strict experimental operator test: input is a real embedding row, NOT the
// hidden state produced by previous transformer layers.
#include "models/inkling2/moe_top8_stage5e.h"
#include <cuda_runtime.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

using namespace ninfer::models::inkling2;
namespace {
constexpr int H = 2304, I = 1152, E = 128, TOP = 8;
using byte = std::uint8_t;
void req(bool ok, const char* msg) { if (!ok) throw std::runtime_error(msg); }
void check(cudaError_t status, const char* label) {
    if (status != cudaSuccess) throw std::runtime_error(std::string(label)+": "+cudaGetErrorString(status));
}
std::string file(const std::string& root,const char* name) { return root+"/"+name; }
std::vector<byte> read(const std::string& root,const char* name,std::size_t expected) {
    std::ifstream in(file(root,name),std::ios::binary|std::ios::ate);
    if(!in) throw std::runtime_error(std::string("cannot open ")+name);
    const auto size=in.tellg();
    if(size<0||static_cast<std::uint64_t>(size)!=expected)
        throw std::runtime_error(std::string("unexpected length of ")+name);
    std::vector<byte> out(expected);
    in.seekg(0);
    if(expected&&!in.read(reinterpret_cast<char*>(out.data()),static_cast<std::streamsize>(expected)))
        throw std::runtime_error(std::string("short read of ")+name);
    return out;
}
template<class T> std::vector<T> read_words(const std::string& root,const char* name,std::size_t count) {
    auto raw=read(root,name,count*sizeof(T));
    std::vector<T> out(count);
    std::memcpy(out.data(),raw.data(),raw.size());
    return out;
}
struct GPU {
    void* p=nullptr;
    explicit GPU(std::size_t n){req(n>0,"zero allocation");check(cudaMalloc(&p,n),"cudaMalloc");}
    GPU(const GPU&)=delete;GPU& operator=(const GPU&)=delete;
    ~GPU(){if(p)cudaFree(p);}
    template<class T>T* as()const{return static_cast<T*>(p);}
    void put(const void* src,std::size_t n){check(cudaMemcpy(p,src,n,cudaMemcpyHostToDevice),"copy to GPU");}
    void get(void* dst,std::size_t n){check(cudaMemcpy(dst,p,n,cudaMemcpyDeviceToHost),"copy from GPU");}
};

struct Bank {
    bool nv;int resident,cols;
    std::vector<byte> codes,scales;
    std::vector<float> amax,den;
    Bank(const std::string& root,const char* proj,bool is_nv,int count,int k):
        nv(is_nv),resident(count),cols(k),
        codes(read(root, (std::string(proj)+"_codes.u8").c_str(),
                   static_cast<std::size_t>(count)*H*k/(is_nv?2:1))),
        scales(read(root, (std::string(proj)+"_scales.u8").c_str(),
                    static_cast<std::size_t>(count)*H*k/(is_nv?16:32))) {
        if(nv){
            amax=read_words<float>(root,(std::string(proj)+"_amax.f32").c_str(),count);
            den=read_words<float>(root,(std::string(proj)+"_denominator.f32").c_str(),count);
            for(int i=0;i<count;++i)
                req(std::isfinite(amax[i])&&std::isfinite(den[i])&&amax[i]>0&&den[i]>0,
                    "invalid original NVFP4 scalars");
        }
        for(byte b:scales)req(nv?(!(b&0x80)&&b!=0x7f):(b!=255),"invalid scale word");
        if(!nv)for(byte b:codes)req((b&0x7f)!=0x7f,"invalid MXFP8 code");
    }
};
struct GPUBank {
    GPU codes,scales,amax,den;
    ExpertBank5E view;
    explicit GPUBank(const Bank& b):codes(b.codes.size()),scales(b.scales.size()),
        amax(static_cast<std::size_t>(b.resident)*sizeof(float)),
        den(static_cast<std::size_t>(b.resident)*sizeof(float)) {
        codes.put(b.codes.data(),b.codes.size());scales.put(b.scales.data(),b.scales.size());
        if(b.nv){amax.put(b.amax.data(),b.amax.size()*4);den.put(b.den.data(),b.den.size()*4);}
        view.codes=codes.as<byte>();view.scales=scales.as<byte>();
        view.amax=b.nv?amax.as<float>():nullptr;
        view.denominator=b.nv?den.as<float>():nullptr;
        view.format=b.nv?ExpertFormat5E::nvfp4_e2m1_g16:ExpertFormat5E::mxfp8_e4m3_g32;
        view.resident=b.resident;view.rows=H;view.cols=b.cols;
    }
};

double e4m3(byte w) {
    if((w&127)==127)return std::numeric_limits<double>::quiet_NaN();
    const double sign=(w&128)?-1:1;
    const unsigned ex=(w>>3)&15, fraction=w&7;
    return sign*(ex==0?std::ldexp(double(fraction),-9):std::ldexp(1.+fraction/8.,int(ex)-7));
}
double e2m1(byte c){
    static constexpr double m[8]={0.,.5,1.,1.5,2.,3.,4.,6.};
    return (c&8)?-m[c&7]:m[c&7];
}
double quant(const Bank& b,int resident,int row,int col) {
    const std::size_t i=(static_cast<std::size_t>(resident)*H+row)*b.cols+col;
    const std::size_t scale_i=(static_cast<std::size_t>(resident)*H+row)*(b.cols/(b.nv?16:32))
                              + col/(b.nv?16:32);
    if(b.nv){
        const byte c=b.codes[i/2];
        const byte nib=(col&1)?(c>>4):(c&15);
        return e2m1(nib)*e4m3(b.scales[scale_i])*(double(b.amax[resident])/double(b.den[resident]));
    }
    const byte scale=b.scales[scale_i];
    return e4m3(b.codes[i])*std::ldexp(1.,int(scale)-127);
}

std::array<double,E> independent_router(const std::vector<float>& hidden,
                                        const std::vector<float>& weight,
                                        const std::vector<float>& correction) {
    std::array<double,E> scores;
    for(int e=0;e<E;++e){
        double dot=0.;
        for(int k=0;k<H;++k)dot+=double(weight[e*H+k])*double(hidden[k]);
        const double s=dot>=0?1./(1.+std::exp(-dot)):
                              std::exp(dot)/(1.+std::exp(dot));
        scores[e]=s;
    }
    return scores;
}

void run(const std::string& root) {
    auto hdr=read(root,"header.bin",36);
    const std::array<char,8> magic={'I','2','R','E','A','L','F','1'};
    req(std::memcmp(hdr.data(),magic.data(),8)==0,"unsupported stage bundle magic");
    std::array<std::uint32_t,7> words{};
    std::memcpy(words.data(),hdr.data()+8,28);
    const unsigned version=words[0],layer=words[1],fmt=words[2],resident=words[3];
    req(version==1&&layer<24&&fmt==(layer<20?1u:2u)&&resident>=8&&resident<=128&&
        words[4]==H&&words[5]==I&&words[6]<201024,"invalid stage bundle geometry");
    auto hidden=read_words<float>(root,"hidden.f32",H);
    auto router=read_words<float>(root,"router.f32",E*H);
    auto bias=read_words<float>(root,"bias.f32",E);
    auto map=read_words<std::int32_t>(root,"resident_map.i32",E);
    auto predicted=read_words<std::int32_t>(root,"predicted_ids.i32",TOP);
    auto expected_weights=read_words<float>(root,"predicted_weights.f32",TOP);
    const float route_scale=read_words<float>(root,"route_scale.f32",1)[0];
    req(std::isfinite(route_scale)&&route_scale>0,"invalid route scale");
    std::vector<int> seen(resident,0);
    for(int e=0;e<E;++e)if(map[e]>=0){
        req(map[e]<static_cast<int>(resident)&&++seen[map[e]]==1,"invalid resident map");
    }
    req(std::all_of(seen.begin(),seen.end(),[](int n){return n==1;}),"non-contiguous resident entries");
    for(int e:predicted)req(e>=0&&e<E&&map[e]>=0,"nonresident predicted topk");
    const bool nv=fmt==1;
    Bank bank13(root,"w13",nv,resident,H),bank2(root,"w2",nv,resident,I);
    GPUBank gpu13(bank13),gpu2(bank2);
    GPU d_hidden(H*4),d_router(E*H*4),d_bias(E*4),d_map(E*4),d_out(H*4);
    d_hidden.put(hidden.data(),H*4);d_router.put(router.data(),E*H*4);
    d_bias.put(bias.data(),E*4);d_map.put(map.data(),E*4);
    MoEWorkspace5E workspace;check(workspace.create(),"workspace.create");
    check(workspace.run(d_hidden.as<float>(),d_router.as<float>(),d_bias.as<float>(),
                        route_scale,gpu13.view,gpu2.view,d_map.as<std::int32_t>(),
                        d_out.as<float>()),"workspace.run");
    std::array<std::int32_t,TOP> indices{};
    std::array<float,TOP> weights{};
    std::vector<float> output(H);
    check(cudaMemcpy(indices.data(),workspace.top_ids(),TOP*4,cudaMemcpyDeviceToHost),"indices D2H");
    check(cudaMemcpy(weights.data(),workspace.top_weights(),TOP*4,cudaMemcpyDeviceToHost),"weights D2H");
    d_out.get(output.data(),H*4);
    int missing=-1;
    check(cudaMemcpy(&missing,workspace.missing_resident_flag(),sizeof(int),cudaMemcpyDeviceToHost),"flag D2H");
    req(missing==0,"GPU selected a nonresident expert; rerun Stage5F with --resident 128");
    std::printf("REAL Layer %u %s: token=%u resident=%u GPU top8=",layer,
                nv?"NVFP4":"MXFP8",words[6],resident);
    for(int j=0;j<TOP;++j){
        req(indices[j]>=0&&indices[j]<E&&map[indices[j]]>=0,"invalid GPU top8");
        std::printf("%s%d",j?",":"",indices[j]);
    }
    std::putchar('\n');
    auto scores=independent_router(hidden,router,bias);
    // Independent selection with float32 BF16 weights and double-precision dot.
    std::array<int,E> order{};
    std::iota(order.begin(),order.end(),0);
    std::stable_sort(order.begin(),order.end(),[&](int a,int b){
        return scores[a]+double(bias[a])>scores[b]+double(bias[b]);
    });
    for(int i=0;i<TOP;++i){
        if(order[i]!=indices[i]){
            const double delta=std::abs((scores[order[i]]+bias[order[i]])-
                                        (scores[indices[i]]+bias[indices[i]]));
            if(delta>5e-5)
                throw std::runtime_error("GPU router top8 disagrees with independent selection");
            std::printf("NOTE near-tie router ordering CPU=%d GPU=%d gap=%.3g\n",order[i],indices[i],delta);
        }
    }
    double total_score=0.;
    for(int e:indices)total_score+=scores[e];
    for(int j=0;j<TOP;++j){
        const double expected=route_scale*scores[indices[j]]/(total_score+1e-20);
        if(!(std::abs(double(weights[j])-expected)<2e-3+2e-3*std::abs(expected)))
            throw std::runtime_error("GPU routing weight is inconsistent with original unbiased score");
    }
    // Independent FP64 reference. Decode only 8 experts and one row at a time.
    std::vector<double> reference(H,0.),sum_abs(H,0.);
    for(int j=0;j<TOP;++j){
        const int slot=map[indices[j]];
        std::vector<double> active(I);
        for(int row=0;row<I;++row){
            double gate=0.,up=0.;
            for(int k=0;k<H;++k){
                gate+=quant(bank13,slot,row,k)*double(hidden[k]);
                up+=quant(bank13,slot,row+I,k)*double(hidden[k]);
            }
            active[row]=(4.*std::tanh(gate/4.)/(1.+std::exp(-gate)))*
                        (25.*std::tanh(up/25.));
        }
        for(int row=0;row<H;++row){
            double sum=0.;
            for(int k=0;k<I;++k)sum+=quant(bank2,slot,row,k)*active[k];
            const double value=double(weights[j])*sum;
            reference[row]+=value;
            sum_abs[row]+=std::abs(value);
        }
    }
    double worst_ratio=0.,worst_abs=0.;int worst_row=-1;
    for(int row=0;row<H;++row){
        const double error=std::abs(double(output[row])-reference[row]);
        const double bound=.01+.002*std::max(1.,std::abs(reference[row]))+
                           .001*sum_abs[row];
        if(!(std::isfinite(output[row])&&std::isfinite(reference[row])&&error<=bound)){
            std::fprintf(stderr,"row=%d cuda=%.10g oracle=%.10g abs_error=%.8g bound=%.8g\n",
                         row,output[row],reference[row],error,bound);
            throw std::runtime_error("REAL expert inference does not match FP64 oracle");
        }
        if(error/bound>worst_ratio){worst_ratio=error/bound;worst_row=row;}
        worst_abs=std::max(worst_abs,error);
    }
    std::printf("PASS real NInfer-v3 -> native CUDA Top8 MoE: max_abs_error=%.7g max_error/bound=%.5g worst_row=%d\n",
                worst_abs,worst_ratio,worst_row);
    std::puts("PASS: this checks one real MoE operator; NO full decoder, logits or generated tokens");
}
} // namespace
int main(int argc,char** argv){
    try{
        req(argc==2,"usage: ninfer_inkling2_real_moe_stage5g <bundle-directory>");
        int count=0;check(cudaGetDeviceCount(&count),"cudaGetDeviceCount");req(count>0,"GPU required");
        cudaDeviceProp prop{};check(cudaGetDeviceProperties(&prop,0),"cudaGetDeviceProperties");
        std::printf("GPU: %s sm_%d%d\n",prop.name,prop.major,prop.minor);
        run(argv[1]);
        std::puts("OK Stage5G real-weight GPU MoE smoke");
        return 0;
    }catch(const std::exception& e){std::fprintf(stderr,"FAIL Stage5G: %s\n",e.what());return 1;}
}
