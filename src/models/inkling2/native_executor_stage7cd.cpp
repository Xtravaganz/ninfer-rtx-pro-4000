// Inkling Stage 7C/D: native C++ orchestration with ATen CUDA operators.
// No Python callback in the 48-visit loop. Stage7AB and Stage6L C ABIs reused.
// Not yet the standalone NInfer runtime; ATen supplies dense math and attention.
#include <ATen/ATen.h>
#include <torch/library.h>
#include <cstdint>
#include <cmath>
#include <string>
#include <vector>
#include <tuple>

namespace {
using T=at::Tensor;
using W=c10::Dict<std::string,T>;
constexpr int H=2304,HQ=24,HK=8,D=128,R=16,EXT=1024;
using Moe=int(*)(void*,void*,int,uint64_t,uint64_t,uint64_t,float,uint64_t,uint64_t) noexcept;
using Conv=int(*)(void*,int,int,uint64_t,uint64_t,uint64_t,int,int,int,uint64_t) noexcept;
uint64_t ptr(const T& t){return reinterpret_cast<uint64_t>(t.data_ptr());}
void* handle(int64_t p){return reinterpret_cast<void*>(static_cast<uintptr_t>(p));}
T get(const W& w,const std::string& key){return w.at(key);}
T lin(const T& x,const T& w){return at::linear(x,w);}
T rms(const T& x,double eps){return x*at::rsqrt(x.square().mean(-1,true)+eps);}
T clip(const T& x,double threshold){return x*(threshold/(at::linalg_vector_norm(x,2,{-1},true)+1e-6)).clamp_max(1.0);}
T residual(const T& blocks,const T& x,const T& q,double eps){
    auto states=rms(at::cat({blocks,x.unsqueeze(-2)},-2),eps);
    return at::matmul((at::matmul(q.reshape({-1,H}),states.transpose(-1,-2))*(1/std::sqrt(double(H)))).softmax(-1),states);
}
T controller(const T& x,const T& msg,const T& down,const T& forget,const T& gain,double eps){
    auto c=lin(at::cat({rms(x,eps),rms(msg,eps)},-1),down);
    return (1-lin(c,forget).tanh())*x+lin(c,gain).tanh()*msg;
}
struct Executor {
    const W& w; T keys,values; int64_t pos; double eps,qclip,rclip,scale,alpha; int64_t floor;
    Moe moe; Conv conv; void *store,*mw,*cs; uint64_t stream;
    T weight(const std::string& s){return get(w,s);}
    T norm(const T& x,const std::string& s){return rms(x,eps)*weight(s);}
    T control(const T& x,const T& msg,const std::string& prefix){return controller(x,msg,weight(prefix+"down_weight"),weight(prefix+"forget_weight"),weight(prefix+"gain_weight"),eps);}
    T convolution(int slot,int path,const T& x,const T& taps,bool dynamic){
        auto in=x.contiguous(),tw=taps.contiguous(),out=at::empty_like(in);
        int rc=conv(cs,slot,path,ptr(in),ptr(tw),ptr(out),int(in.numel()),1,dynamic,stream);
        TORCH_CHECK(rc==0,"Stage7C convolution error ",rc," slot ",slot," path ",path);
        return out;
    }
    T kvconv(int slot,int path,const T& x,const T& input,const std::string& p,bool full){
        if(!full)return convolution(slot,path,x,weight(p+".weight").reshape({HK*D,4}),false);
        auto delta=lin(input,weight(p+".weight_proj")).reshape({HK,1,4});
        auto taps=(weight(p+".static_w").transpose(0,1).reshape({HK,D,4})+delta).reshape({HK*D,4});
        return convolution(slot,path,x,taps,true);
    }
    T layer(T hidden,T& blocks,int layer,int slot,const T& message){
        const bool full=layer%6==5;
        const auto p="model.llm.layers."+std::to_string(layer)+".";const auto a=p+"attn.";
        T qh=hidden,kh=hidden,vh=hidden;
        if(layer){auto h=residual(blocks,hidden,weight("model.llm.attn_res.attn_queries."+std::to_string(layer-1)),eps);
            qh=h.select(-2,0);kh=full?h.select(-2,1):qh;vh=full?h.select(-2,2):qh;}
        if(message.defined()){
            qh=control(qh,message.select(-2,0),p+"controllers.attn.");
            kh=control(kh,message.select(-2,1),p+"controllers.k.");
            vh=control(vh,message.select(-2,2),p+"controllers.v.");
        }
        if(full){qh=norm(qh,p+"q_input_norm.weight");kh=norm(kh,p+"k_input_norm.weight");vh=norm(vh,p+"v_input_norm.weight");}
        else {qh=norm(qh,p+"attn_norm.weight");kh=message.defined()?norm(kh,p+"attn_norm.weight"):qh;vh=message.defined()?norm(vh,p+"attn_norm.weight"):qh;}
        double logscale=full?1+alpha*std::log(std::max(1.0,double(pos+1)/floor)):1;
        auto q=clip(lin(qh,weight(a+"wq_du.weight")).reshape({1,1,HQ,D})*weight(a+"q_l2_clip.scale"),qclip)*logscale;
        auto k=kvconv(slot,0,lin(kh,weight(a+"wk_dv.weight")),kh,a+"k_sconv",full);
        auto v=kvconv(slot,1,lin(vh,weight(a+"wv_dv.weight")),vh,a+"v_sconv",full);
        k=clip(k.reshape({HK,D}),qclip);v=rms(v.reshape({HK,D}),eps);
        auto kb=keys.select(0,slot),vb=values.select(0,slot);
        kb.select(0,pos).copy_(k);vb.select(0,pos).copy_(v);
        const int64_t start=full?0:std::max<int64_t>(0,pos+1-EXT), n=pos+1-start;
        auto kr=at::repeat_interleave(kb.narrow(0,start,n).transpose(0,1),HQ/HK,0);
        auto vr=at::repeat_interleave(vb.narrow(0,start,n).transpose(0,1),HQ/HK,0);
        auto rel=clip(lin(qh,weight(a+"wr_du.weight")).reshape({1,1,HQ,R})*weight(a+"rel_l2_clip.scale"),rclip)*logscale;
        auto profile=at::matmul(rel,weight(a+"rel_logits_proj.proj")).reshape({HQ,EXT});
        auto distances=at::arange(n-1,-1,-1,keys.options().dtype(at::kLong));
        auto bias=profile.index_select(1,distances.clamp_max(EXT-1)).masked_fill(distances.ge(EXT).unsqueeze(0),0);
        auto logits=at::matmul(q.reshape({HQ,1,D}),kr.transpose(-1,-2))*(1/std::sqrt(double(D)))+bias.unsqueeze(1);
        auto sink=lin(qh,weight(a+"attn_sink_proj.weight")).reshape({HQ,1,1});
        auto probs=at::cat({logits,sink},-1).softmax(-1).narrow(-1,0,n);
        auto attended=at::matmul(probs,vr).reshape({1,1,HQ*D});
        auto attn=lin(attended*lin(qh,weight(a+"gate_attn.weight")).sigmoid(),weight(a+"wo_ud.weight"));
        auto attnout=convolution(slot,2,attn,weight(p+"attn_sconv.weight").reshape({H,4}),false);
        T res;
        if(layer%6==0){res=attnout;blocks=at::cat({blocks,hidden.unsqueeze(-2)},-2);}else res=hidden+attnout;
        auto mlp=residual(blocks,res,weight("model.llm.attn_res.mlp_queries."+std::to_string(layer)),eps).select(-2,0);
        if(message.defined())mlp=control(mlp,message.select(-2,3),p+"controllers.mlp.");
        mlp=norm(mlp,p+"mlp_norm.weight").contiguous();
        auto routed=at::empty_like(mlp);
        int rc=moe(store,mw,layer,ptr(mlp),ptr(weight(p+"mlp.gate.weight")),ptr(weight(p+"mlp.gate.bias")),float(scale),ptr(routed),stream);
        TORCH_CHECK(rc==0,"Stage7C MoE error ",rc," layer ",layer," slot ",slot);
        return res+convolution(slot,3,routed,weight(p+"mlp_sconv.weight").reshape({H,4}),false);
    }
};
T forward(const W& w,const T& embedding,T keys,T values,int64_t token,int64_t pos,
          double eps,double qclip,double rclip,double scale,double alpha,int64_t floor,
          int64_t moe_fn,int64_t conv_fn,int64_t store,int64_t mw,int64_t cs,int64_t stream){
    TORCH_CHECK(keys.is_cuda() && values.is_cuda() && embedding.is_cuda(),"CUDA required");
    TORCH_CHECK(keys.sizes()==values.sizes() && keys.dim()==4 && keys.size(0)==48 && keys.size(2)==HK && keys.size(3)==D && pos>=0 && pos<keys.size(1),"invalid KV capacity/position");
    TORCH_CHECK(token>=0 && token<embedding.size(0) && embedding.size(1)==H && floor>0 && moe_fn && conv_fn && store && mw && cs,"invalid native executor inputs");
    Executor e{w,keys,values,pos,eps,qclip,rclip,scale,alpha,floor,reinterpret_cast<Moe>(moe_fn),reinterpret_cast<Conv>(conv_fn),handle(store),handle(mw),handle(cs),uint64_t(stream)};
    auto hidden=e.norm(embedding.select(0,token).to(at::kFloat).reshape({1,1,H}),"model.llm.embed_norm.weight");
    auto blocks=at::empty({1,1,0,H},hidden.options());
    for(int l=0;l<6;++l)hidden=e.layer(hidden,blocks,l,l,T());
    auto prehidden=hidden,preblocks=blocks;
    auto qm=get(w,"model.llm.middle_cycle_cross_pass_queries").reshape({12,2,H});
    auto kv=get(w,"model.llm.middle_cycle_cross_pass_kv_queries").reshape({12,2,H});
    auto queries=at::cat({qm.narrow(1,0,1),kv,qm.narrow(1,1,1)},1);
    for(int pass=0;pass<3;++pass){T previous,previous_blocks;
        if(pass){previous=hidden;previous_blocks=blocks.narrow(-2,2,1);hidden=prehidden;blocks=preblocks;}
        for(int l=6;l<18;++l){auto msg=pass?residual(previous_blocks,previous,queries.select(0,l-6),eps):T();
            hidden=e.layer(hidden,blocks,l,pass?24+(pass-1)*12+l-6:l,msg);}
    }
    for(int l=18;l<24;++l)hidden=e.layer(hidden,blocks,l,l,T());
    TORCH_CHECK(blocks.size(-2)==4,"unexpected residual block count");
    return e.norm(residual(blocks,hidden,get(w,"model.llm.attn_res.output_query"),eps).select(-2,0),"model.llm.norm.weight");
}
std::tuple<T,T> greedy(const T& head,const T& hidden,double cap){
    TORCH_CHECK(head.dim()==2 && hidden.numel()==head.size(1) && head.scalar_type()==at::kFloat && hidden.scalar_type()==at::kFloat && cap>0 && std::isfinite(cap),"invalid greedy head");
    auto logits=at::mv(head,hidden.reshape({head.size(1)}));
    logits=cap*(logits/cap).tanh();
    auto result=logits.max(0); // first ID wins ties, after softcap, as reference
    auto finite=at::isfinite(logits).all();
    auto val=at::where(finite,std::get<0>(result),at::full_like(std::get<0>(result),NAN));
    return {val,std::get<1>(result)};
}
}
TORCH_LIBRARY(inkling7cd,m){
    m.def("forward(Dict(str, Tensor) weights, Tensor embedding, Tensor(a!) keys, Tensor(b!) values, int token, int pos, float eps, float qclip, float rclip, float scale, float alpha, int floor, int moe_fn, int conv_fn, int store, int mw, int cs, int stream) -> Tensor", &forward);
    m.def("greedy", &greedy);
    m.def("residual", &residual);
    m.def("controller", &controller);
}
