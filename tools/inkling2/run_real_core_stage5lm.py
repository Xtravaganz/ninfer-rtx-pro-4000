#!/usr/bin/env python3
"""Stage5L/M original-weight first core pass and learned three-pass loop.

No generated logits, no postlude layers 18..23, no historic token KV caches,
and no native NInfer runtime. The actual execution uses Torch on CPU/CUDA.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
for path in (HERE,ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0,str(path))

from nonexpert_artifact import Checkpoint
from real_carrier_stage5f import Carrier5F, validate_complete_carriers
from runtime_bindings import schedule
from run_real_block_stage5jk import first_block_preflight, mandatory_full_sources


def core_source_shapes(layer: int, config: dict) -> dict[str,tuple[tuple[int,...],...]]:
    """Original HF checkpoint names and shape contract for ONE core layer."""
    if type(layer) is not int or layer not in range(6,18):
        raise ValueError('only physical core layers 6..17')
    t=config['text_config']
    if (t['hidden_size']!=2304 or t['num_attention_heads']!=24 or
        t['num_key_value_heads']!=8 or t['head_dim']!=128 or
        t['d_rel']!=16 or t['rel_extent']!=t['sliding_window_size'] or
        t['sconv_kernel_size']!=4 or t['n_routed_experts']!=128 or
        t['num_experts_per_tok']!=8 or t['intermediate_size']!=1152 or
        t['local_layer_ids'] != [i for i in range(24) if i%6!=5]):
        raise ValueError('unsupported Inkling 2 Mini original core geometry')
    d=t['hidden_size'];hq=t['num_attention_heads'];hk=t['num_key_value_heads']
    head=t['head_dim'];rel=t['d_rel'];extent=t['sliding_window_size']
    taps=t['sconv_kernel_size'];experts=t['n_routed_experts'];p=f'model.llm.layers.{layer}.';a=p+'attn.'
    data={
        p+'mlp_norm.weight':((d,),),
        p+'attn_sconv.weight':((d,taps),(d,1,taps)),
        p+'mlp_sconv.weight':((d,taps),(d,1,taps)),
        p+'mlp.gate.weight':((experts,d),),
        p+'mlp.gate.bias':((experts,),),
        a+'wq_du.weight':((hq*head,d),),
        a+'wk_dv.weight':((hk*head,d),),
        a+'wv_dv.weight':((hk*head,d),),
        a+'wo_ud.weight':((d,hq*head),),
        a+'gate_attn.weight':((hq*head,d),),
        a+'attn_sink_proj.weight':((hq,d),),
        a+'q_l2_clip.scale':((hq,head),),
        a+'rel_l2_clip.scale':((hq,rel),),
        a+'rel_logits_proj.proj':((rel,extent),),
        a+'wr_du.weight':((hq*rel,d),),
        f'model.llm.attn_res.mlp_queries.{layer}':((d,),),
        f'model.llm.attn_res.attn_queries.{layer-1}':(((3,d) if layer%6==5 else (d,)),),
    }
    if layer in (11,17):
        for role in ('q','k','v'):
            data[p+f'{role}_input_norm.weight']=((d,),)
        for role in ('k','v'):
            data[a+role+'_sconv.static_w']=((taps,hk*head),)
            data[a+role+'_sconv.weight_proj']=((hk*taps,d),)
    else:
        data[p+'attn_norm.weight']=((d,),)
        for role in ('k','v'):
            data[a+role+'_sconv.weight']=((hk*head,taps),(hk*head,1,taps))
    return data


def core_preflight(spans: dict, config: dict, *, core_layers: int=6,
                   passes: int=1) -> dict:
    """Preflight only; never reads tensor payloads or imports PyTorch."""
    if (type(core_layers) is not int or core_layers not in (6,12) or
        type(passes) is not int or passes not in (1,2,3) or
        core_layers==6 and passes!=1):
        raise ValueError('Stage5L: 6/12 layers one pass; Stage5M: full 12-layer core multiple passes')
    t=config['text_config']
    steps=schedule(config)
    if (len(steps)!=48 or [(s['physical_layer'],s['core_pass']) for s in steps[6:18]]!=
        [(i,0) for i in range(6,18)]):
        raise ValueError('invalid Stage1 48-visit schedule')
    expected={
        'model.llm.embed.weight':((t['vocab_size'],t['hidden_size']),),
        'model.llm.embed_norm.weight':((t['hidden_size'],),),
    }
    # Reuse already confirmed Stage5J/K prelude contracts, and reject absent
    # weights in the entire six-layer prelude before starting core computation.
    from run_real_sliding_stage5hi import mandatory_sliding_sources
    for layer in range(5):
        expected.update(mandatory_sliding_sources(layer,config))
    expected.update(mandatory_full_sources(config))
    for layer in range(6,6+core_layers):
        expected.update(core_source_shapes(layer,config))
    if passes>1:
        gate_rank=t['loop_config']['gate_rank']
        d=t['hidden_size']
        expected.update({
            'model.llm.middle_cycle_cross_pass_queries':((24,d),),
            'model.llm.middle_cycle_cross_pass_kv_queries':((24,d),),
        })
        for layer in range(6,18):
            for path in ('attn','k','v','mlp'):
                p=f'model.llm.layers.{layer}.controllers.{path}.'
                expected[p+'down_weight']=((gate_rank,2*d),)
                expected[p+'forget_weight']=((d,gate_rank),)
                expected[p+'gain_weight']=((d,gate_rank),)
    missing=[];wrong=[]
    for name,shapes in sorted(expected.items()):
        span=spans.get(name)
        if span is None:missing.append(name)
        elif (tuple(span.shape) not in shapes or
              (span.dtype != 'BF16' if '.controllers.' in name else span.dtype not in ('F32','BF16'))):
            wrong.append((name,str(span.dtype),list(span.shape),[list(s) for s in shapes]))
    if missing or wrong:
        raise ValueError(f'Stage5L/M unsupported HF tensor contract: missing={missing[:7]}, wrong={wrong[:7]}')
    count=6+core_layers*passes
    visits=steps[:6] + [s for s in steps if s['stage']=='core' and s['core_pass']<passes and s['physical_layer']<6+core_layers]
    assert len(visits)==count
    slots=[s['cache_slot'] for s in visits]
    if len(set(slots))!=len(slots):
        raise ValueError('logical cache slots collide across core passes')
    return {
        'required_nonexpert_tensor_names':len(expected),
        'physical_core_layers':list(range(6,6+core_layers)),
        'core_passes':passes,
        'planned_visits':count,
        'cache_slots':slots,
        'controller_tensors':144 if passes>1 else 0,
        'has_cross_pass_messages':passes>1,
        'has_postlude':False,'supports_generated_tokens':False,
        'supports_past_kv_or_conv_states':False,
        'native_ninfer_runtime':False,
    }


def run(args):
    if (type(args.token_id) is not int or args.token_id<0 or
        args.output and (args.output.exists() or args.output.is_symlink())):
        raise ValueError('invalid token or refusal to overwrite report')
    from tools.artifact.reader import Artifact
    with Artifact(args.nonexpert) as non,Artifact(args.experts) as exp:
        bound=validate_complete_carriers(args.source,non,exp,args.banks)
        with Checkpoint(args.source) as checkpoint:
            ready=core_preflight(checkpoint.spans,checkpoint.config,
                core_layers=args.core_layers,passes=args.passes)
            config=checkpoint.config
        print('MODEL: verified NInfer v3 full metadata (1068 nonexpert + 48 expert banks)')
        print('READY:',json.dumps({k:ready[k] for k in ('planned_visits','core_passes','has_cross_pass_messages','has_postlude')},sort_keys=True))
        if args.plan:
            print('PLAN:',json.dumps(ready,sort_keys=True))
            return
        import torch
        from real_sliding_stage5h import Geometry,TorchCarrierWeights
        from real_core_stage5lm import run_real_core
        if args.device.startswith('cuda') and not torch.cuda.is_available():
            raise RuntimeError('CUDA requested but not available')
        g=Geometry.from_hf_config(config)
        if args.token_id>=config['text_config']['vocab_size']:
            raise ValueError('token ID outside vocabulary')
        reader=Carrier5F(non,exp)
        weights=TorchCarrierWeights(reader,args.device,g)
        with torch.inference_mode():
            hidden,blocks,trace=run_real_core(weights,args.token_id,
                core_layers=args.core_layers,passes=args.passes,
                gate_rank=config['text_config']['loop_config']['gate_rank'],
                log_scaling_alpha=config['text_config']['log_scaling_alpha'],
                log_scaling_n_floor=config['text_config']['log_scaling_n_floor'])
        flat=hidden.detach().float().cpu().contiguous().reshape(-1)
        if not bool(torch.isfinite(flat).all()):
            raise ValueError('nonfinite hidden state')
        digest=sha256(flat.numpy().tobytes()).hexdigest()
        if len(trace)!=ready['planned_visits'] or [r['cache_slot'] for r in trace]!=ready['cache_slots']:
            raise ValueError('executed real trace does not match 48-step subset schedule')
        for record in trace:
            if record['stage']=='core':
                print('REAL CORE pass={} layer={} slot={}: top8={} hidden_rms={:.6g}'.format(
                    record['core_pass'],record['physical_layer'],record['cache_slot'],
                    record['top8_expert_ids'],record['hidden_rms']))
        report={
            'contract':'inkling2_stage5lm_real_core_one_token_v1',
            'model_revision':args.source.name,'torch_device':args.device,
            'token_id':args.token_id,'executed_visits':len(trace),
            'core_layers':args.core_layers,'core_passes':args.passes,
            'hidden_rms':float(flat.square().mean().sqrt()),
            'last_hidden_fp32_sha256':digest,
            'block_states':int(blocks.shape[-2]),'trace':trace,
            'nonexpert_artifact_id':non.artifact_id.hex(),
            'experts_artifact_id':exp.artifact_id.hex(),
            'checkpoint_index_sha256':bound['checkpoint']['model_index_sha256'],
            'full_48_step_forward':False,'generated_token':False,
            'HF_end_to_end_numeric_parity':False,'native_ninfer_runtime':False,
            'has_past_cache':False,
        }
        if args.output:
            args.output.parent.mkdir(parents=True,exist_ok=True)
            args.output.write_text(json.dumps(report,indent=2)+'\n')
            print('TRACE:',args.output)
        print(f'OK Stage5L/M real core: visits={len(trace)} passes={args.passes} digest={digest[:20]}')
        print('NOT YET: final 6 postlude layers, HF end-to-end parity, historical KV/conv cache, logits or native NInfer runtime')


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--nonexpert',type=Path,required=True)
    p.add_argument('--experts',type=Path,required=True)
    p.add_argument('--banks',type=Path,required=True)
    p.add_argument('--core-layers',type=int,choices=(6,12),default=6)
    p.add_argument('--passes',type=int,choices=(1,2,3),default=1)
    p.add_argument('--token-id',type=int,default=1000)
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--plan',action='store_true')
    p.add_argument('--output',type=Path,help='optional non-overwriting JSON trace')
    args=p.parse_args(argv)
    try:run(args)
    except (RuntimeError,ValueError,OSError,KeyError,TypeError) as exc:
        print('ERROR:',exc,file=sys.stderr)
        return 1
    return 0

if __name__=='__main__':
    sys.exit(main())
