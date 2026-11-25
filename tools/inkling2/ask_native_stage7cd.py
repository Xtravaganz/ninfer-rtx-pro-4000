#!/usr/bin/env python3
"""Stage7C/D: C++ ATen executor + GPU resident greedy head; sequential prefill.
Optional reference path shares the same weights, but has independent KV/conv state.
"""
import argparse, ctypes, json, math, statistics, time
from pathlib import Path
import torch

H=2304

def specifications(g, rank):
    yield 'model.llm.embed_norm.weight',(H,)
    yield 'model.llm.norm.weight',(H,)
    yield 'model.llm.attn_res.output_query',(H,)
    for suffix in ('middle_cycle_cross_pass_queries','middle_cycle_cross_pass_kv_queries'):
        yield 'model.llm.'+suffix,(24,H)
    for l in range(24):
        p=f'model.llm.layers.{l}.';a=p+'attn.';full=l%6==5
        if l: yield f'model.llm.attn_res.attn_queries.{l-1}',(3,H) if full else (H,)
        yield f'model.llm.attn_res.mlp_queries.{l}',(H,)
        for suffix in (('q_input_norm.weight','k_input_norm.weight','v_input_norm.weight') if full else ('attn_norm.weight',)):
            yield p+suffix,(H,)
        yield p+'mlp_norm.weight',(H,)
        for suffix,shape in [('wq_du.weight',(3072,H)),('wk_dv.weight',(1024,H)),('wv_dv.weight',(1024,H)),('q_l2_clip.scale',(24,128)),('wr_du.weight',(384,H)),('rel_l2_clip.scale',(24,16)),('rel_logits_proj.proj',(16,1024)),('attn_sink_proj.weight',(24,H)),('gate_attn.weight',(3072,H)),('wo_ud.weight',(H,3072))]:
            yield a+suffix,shape
        for name in ('k_sconv','v_sconv'):
            if full:
                yield a+name+'.static_w',(4,1024)
                yield a+name+'.weight_proj',(32,H)
            else: yield a+name+'.weight',None
        for suffix in ('attn_sconv.weight','mlp_sconv.weight'): yield p+suffix,None
        yield p+'mlp.gate.weight',(128,H)
        yield p+'mlp.gate.bias',(128,)
        if 6<=l<18:
            for path in ('attn','k','v','mlp'):
                c=p+'controllers.'+path+'.'
                yield c+'down_weight',(rank,2*H)
                yield c+'forget_weight',(H,rank)
                yield c+'gain_weight',(H,rank)

def bind(w,rank):
    result={}
    for name,shape in specifications(w.geom,rank):
        if shape is None:
            result[name]=w.conv(name,1024 if '.attn.' in name else H)
        else: result[name]=w.tensor(name,shape)
        t=result[name]
        if t.dtype!=torch.float32 or t.device!=w.device or not t.is_contiguous(): raise ValueError('invalid bound tensor '+name)
    if w.counters['nonexpert_evictions']:
        raise RuntimeError('Nonexpert cache too small for fixed binding; use 4096 MiB or more')
    return result

def load_matrix(artifact,name,rows,*,fp32):
    obj=artifact.object('hf/model.llm.'+name+'.weight')
    if obj.kind!='tensor' or tuple(obj.shape)!=(201024,H) or obj.format not in ('bf16','fp32'):
        raise ValueError('invalid matrix '+name)
    dtype=torch.bfloat16 if obj.format=='bf16' else torch.float32
    target=torch.float32 if fp32 else dtype
    stride=H*(2 if dtype==torch.bfloat16 else 4)
    out=torch.empty((rows,H),device='cuda:0',dtype=target)
    for start in range(0,rows,4096):
        n=min(rows-start,4096);raw=artifact.read_range(obj.offset+start*stride,n*stride)
        if len(raw)!=n*stride: raise ValueError('short matrix read '+name)
        source=torch.frombuffer(bytearray(raw),dtype=dtype).reshape(n,H)
        out[start:start+n].copy_(source)
    return out

def address(fn): return ctypes.cast(fn,ctypes.c_void_p).value

def run(a):
    from transformers import AutoTokenizer
    from tools.artifact.reader import Artifact
    from nonexpert_artifact import Checkpoint
    from real_carrier_stage5f import Carrier5F,validate_complete_carriers
    from real_sliding_stage5h import Geometry
    from native_stage6jkl import PreloadedResidentWeights,ConvState,make_native_conv_cache
    from native_stage7ab import FastMoE
    from streaming_stage5q import forward_one_token
    from real_postlude_stage5no import streaming_unembed_topk
    from run_real_forward_stage5no import full_forward_preflight
    from ask_inkling_stage5r import tokenize_question,stop_ids,render_tokens
    torch.cuda.set_device(0)
    # Preserve FP32 reference rounding policy; no TF32 or new quantization.
    torch.set_float32_matmul_precision('highest')
    torch.ops.load_library(str(a.executor))
    tokenizer=AutoTokenizer.from_pretrained(str(a.source),local_files_only=True,trust_remote_code=False,use_fast=True)
    ids=tokenize_question(tokenizer,a.question)
    if len(ids)>a.capacity-a.max_new_tokens: raise ValueError('prompt + requested output exceeds KV capacity')
    with Artifact(a.nonexpert) as non, Artifact(a.experts) as exp:
        validate_complete_carriers(a.source,non,exp,a.banks)
        with Checkpoint(a.source) as ck:
            full_forward_preflight(ck.spans,ck.config,'greedy');config=ck.config
        g=Geometry.from_hf_config(config); text=config['text_config'];rank=text['loop_config']['gate_rank']
        if (g.hidden,g.query_heads,g.kv_heads,g.head_dim,g.rel_dim,g.rel_extent,g.conv_kernel,g.intermediate,g.experts,g.top_k)!=(2304,24,8,128,16,1024,4,1152,128,8): raise ValueError('Stage7CD geometry unsupported')
        if text.get('unpadded_vocab_size',200058)!=200058: raise ValueError('unexpected vocabulary')
        cap=text.get('final_logit_softcapping',30.0)
        if not isinstance(cap,(int,float)) or not math.isfinite(cap) or cap<=0: raise ValueError('positive finite logit softcap required')
        alpha=text['log_scaling_alpha'];floor=text['log_scaling_n_floor']
        w=fast=conv=refcache=None
        try:
            startup=time.perf_counter();w=PreloadedResidentWeights(Carrier5F(non,exp),'cuda:0',g,library_path=a.library,nonexpert_mib=a.nonexpert_mib,verify_first=False,use_graph=False)
            fast=FastMoE(w.native,a.fast_library);conv=ConvState(a.library)
            bound=bind(w,rank)
            embedding=load_matrix(non,'embed',201024,fp32=False)
            head=load_matrix(non,'unembed',200058,fp32=True) if a.mode=='cd' else None
            keys=torch.empty((48,a.capacity,8,128),device='cuda:0',dtype=torch.float32);values=torch.empty_like(keys)
            s=torch.cuda.current_stream();s.synchronize()
            free,total=torch.cuda.mem_get_info(); print('STAGE7CD STARTUP:',json.dumps(dict(mode=a.mode,seconds=time.perf_counter()-startup,weights=len(bound),free_mib=free/1048576,kv_capacity=a.capacity,head_mib=0 if head is None else head.numel()*4/1048576)),flush=True)
            if a.verify:
                refcache=make_native_conv_cache(a.library)(kernel=4)
                def fast_run(x,router,bias,scale,layer,out,stream): fast.run(x,router,bias,scale,layer,out,stream)
                w.native.run=fast_run
            pos=0;last_ref=None;forward_ms=[];prefill_ms=[];head_ms=[];end_ms=[];generated=[];errors=[];head_errors=[]
            def forward(token,phase):
                nonlocal pos,last_ref
                start=time.perf_counter()
                x=torch.ops.inkling7cd.forward(bound,embedding,keys,values,token,pos,g.norm_eps,g.qk_clip,g.rel_clip,g.route_scale,alpha,floor,address(fast.lib.inkling7ab_run),address(conv.lib.inkling6l_conv_apply),w.native.store,fast.handle,conv.handle,int(s.cuda_stream))
                s.synchronize(); elapsed=(time.perf_counter()-start)*1000
                if not bool(torch.isfinite(x).all()): raise RuntimeError('nonfinite native hidden; abort state')
                (prefill_ms if phase=='prefill' else forward_ms).append(elapsed)
                if a.verify:
                    ref,_=forward_one_token(w,refcache,token,gate_rank=rank,log_scaling_alpha=alpha,log_scaling_n_floor=floor)
                    last_ref=ref
                    err=float((x-ref).abs().max());errors.append(err)
                    if not torch.allclose(x,ref,atol=.005,rtol=.002): raise RuntimeError(f'forward parity failed at position {pos}: {err}')
                print(f'STAGE7C {phase} position={pos} ms={elapsed:.3f}',flush=True);pos+=1
                return x
            with torch.inference_mode():
                for token in ids: x=forward(token,'prefill')
                for step in range(a.max_new_tokens):
                    tick=time.perf_counter()
                    if step: x=forward(generated[-1],'decode')
                    start=time.perf_counter()
                    if head is None:
                        top=streaming_unembed_topk(w.reader,x,softcap=cap,topk=1)[0];idx=top['token_id'];value=top['logit']
                    else:
                        val,ind=torch.ops.inkling7cd.greedy(head,x,cap);idx=int(ind.item());value=float(val.item())
                    head_ms.append((time.perf_counter()-start)*1000)
                    if not math.isfinite(value): raise RuntimeError('nonfinite head')
                    end_ms.append((time.perf_counter()-tick)*1000)
                    if a.verify and head is not None:
                        old=streaming_unembed_topk(w.reader,x,softcap=cap,topk=1)[0]
                        head_errors.append(abs(old['logit']-value))
                        if old['token_id']!=idx or not math.isclose(old['logit'],value,abs_tol=.005,rel_tol=.002): raise RuntimeError(f'head parity failed step {step}: {old} vs {(idx,value)}')
                    if a.verify:
                        if head is not None:
                            _,ref_id=torch.ops.inkling7cd.greedy(head,last_ref,cap);ref_id=int(ref_id.item())
                        else:
                            ref_id=streaming_unembed_topk(w.reader,last_ref,softcap=cap,topk=1)[0]['token_id']
                        if ref_id!=idx: raise RuntimeError(f'greedy ID differs from reference hidden at step {step}: {idx} vs {ref_id}')
                    generated.append(idx);print(f'GEN {step+1}: id={idx} logit={value:.6g} text={render_tokens(tokenizer,generated)!r}',flush=True)
                    if idx in stop_ids(tokenizer,text): break
            def median(x):return round(statistics.median(x),3) if x else None
            print('GENERATED IDS:',generated)
            print('STAGE7CD PERF:',json.dumps(dict(mode=a.mode,verified=a.verify,forward_calls=pos,expected_moe_visits=48*pos,prefill_warm_ms=median(prefill_ms[1:]),decode_forward_ms=median(forward_ms),head_ms=median(head_ms),decode_with_head_ms=None if a.verify else median(end_ms[1:]),max_hidden_error=max(errors,default=None),max_head_error=max(head_errors,default=None),note='ATen C++ executor, greedy only, sequential prefill; verification mode reports no combined time; forward and head times exclude the reference path')))
        finally:
            torch.cuda.synchronize()
            if refcache: refcache.close()
            if conv: conv.close()
            if fast: fast.close()
            if w: w.release()

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('source','nonexpert','experts','banks','library','fast-library','executor'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--question',required=True);p.add_argument('--max-new-tokens',type=int,default=10);p.add_argument('--capacity',type=int,default=128);p.add_argument('--nonexpert-mib',type=int,default=4096);p.add_argument('--mode',choices=['c','cd'],default='cd');p.add_argument('--verify',action='store_true')
    a=p.parse_args();
    if not 1<=a.max_new_tokens<=64 or not 1<=a.capacity<=4096: p.error('invalid token/capacity bounds')
    run(a)
if __name__=='__main__':main()
