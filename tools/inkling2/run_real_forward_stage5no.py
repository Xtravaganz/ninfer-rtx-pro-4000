#!/usr/bin/env python3
"""Stage5N/O original NInfer-v3 checkpoint: 48 visits, final norm, candidate logits.

This is single-token position-zero Torch integration, NOT validated generation,
HF end-to-end parity, multi-token inference, or native ninfer-serve.
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
    if str(path) not in sys.path:sys.path.insert(0,str(path))

from nonexpert_artifact import Checkpoint
from real_carrier_stage5f import Carrier5F,validate_complete_carriers
from run_real_core_stage5lm import core_preflight, core_source_shapes
from runtime_bindings import schedule


def postlude_sources(layer:int, config:dict) -> dict[str,tuple[tuple[int,...],...]]:
    """Strict HF source shapes for postlude 18..23. Reuses Stage5L core shapes."""
    if type(layer) is not int or layer not in range(18,24):
        raise ValueError('postlude physical layer must be in 18..23')
    # Geometry matches core. Translate contract index carefully, without
    # broadening any shape or mutating earlier stage code.
    original=core_source_shapes(17 if layer == 23 else 16,config)
    # Full (23) has 3 Q/K/V queries, dynamic K/V; sliding (18..22) has 1.
    parent=17 if layer == 23 else 16
    old=f'model.llm.layers.{parent}.'
    new=f'model.llm.layers.{layer}.'
    result={}
    for name,shapes in original.items():
        if name.startswith(old):
            result[new+name[len(old):]]=shapes
        elif name.startswith('model.llm.attn_res.attn_queries.'):
            result[f'model.llm.attn_res.attn_queries.{layer-1}']=shapes
        elif name.startswith('model.llm.attn_res.mlp_queries.'):
            result[f'model.llm.attn_res.mlp_queries.{layer}']=shapes
        else:raise AssertionError('unexpected core shape group')
    return result


def full_forward_preflight(spans:dict, config:dict, mode:str='hidden') -> dict:
    """No tensor payload access; verifies every required source geometry/dtype."""
    if mode not in ('hidden','greedy'):
        raise ValueError('mode must be hidden or greedy')
    prior=core_preflight(spans,config,core_layers=12,passes=3)
    t=config['text_config'];d=t['hidden_size']
    if t.get('unpadded_vocab_size',200058)!=200058 or t['vocab_size']!=201024:
        raise ValueError('unexpected padded/unpadded vocabulary')
    cap=t.get('final_logit_softcapping',30.)
    if cap is not None and (type(cap) not in (int,float) or cap<=0):
        raise ValueError('unexpected output softcap')
    if t['loop_config'].get('num_passes')!=3 or t['n_layers_per_attn_res_block']!=6:
        raise ValueError('unsupported loop configuration')
    needed={}
    for layer in range(18,24):
        needed.update(postlude_sources(layer,config))
    needed['model.llm.attn_res.output_query']=((d,),)
    needed['model.llm.norm.weight']=((d,),)
    if mode=='greedy':
        needed['model.llm.unembed.weight']=((t['vocab_size'],d),)
    missing=[];wrong=[]
    for name, shapes in sorted(needed.items()):
        src=spans.get(name)
        if src is None:
            missing.append(name)
        elif tuple(src.shape) not in shapes or src.dtype not in ('F32','BF16'):
            wrong.append((name,src.dtype,tuple(src.shape),shapes))
    if missing or wrong:
        raise ValueError(f'Stage5N/O unsupported HF tensor geometry: missing={missing[:6]}, wrong={wrong[:6]}')
    visits=schedule(config)
    slots=[x['cache_slot'] for x in visits]
    if (len(visits)!=48 or [v['physical_layer'] for v in visits[-6:]]!=list(range(18,24)) or
            slots != prior['cache_slots']+list(range(18,24))):
        raise ValueError('execution schedule/slots are not the official 48-step contract')
    return {
        'logical_visits':len(visits),'physical_weights':24,
        'postlude_physical_layers':list(range(18,24)),
        'quantization_postlude':{'nvfp4':[18,19],'mxfp8':[20,21,22,23]},
        'additional_weight_names':len(needed),'cache_slots':slots,
        'unpadded_vocab_size':t.get('unpadded_vocab_size',200058),
        'final_logit_softcapping':cap,
        'mode':mode,'position':0,'supports_historical_kv_cache':False,
        'HF_end_to_end_parity':False,'native_ninfer_runtime':False,
    }


def run(args):
    if (type(args.token_id) is not int or args.token_id < 0 or
        (args.output is not None and (args.output.exists() or args.output.is_symlink()))):
        raise ValueError('invalid token id or refusing to overwrite report')
    from tools.artifact.reader import Artifact
    with Artifact(args.nonexpert) as non, Artifact(args.experts) as exp:
        binding=validate_complete_carriers(args.source,non,exp,args.banks)
        with Checkpoint(args.source) as checkpoint:
            ready=full_forward_preflight(checkpoint.spans,checkpoint.config,args.mode)
            cfg=checkpoint.config
        print('MODEL: complete NInfer-v3 carriers verified: 1068 nonexpert + 48 expert banks')
        print('READY:',json.dumps({k:ready[k] for k in ('logical_visits','mode','unpadded_vocab_size','HF_end_to_end_parity')},sort_keys=True))
        if args.plan:
            print('PLAN:',json.dumps(ready,sort_keys=True))
            print('PLAN ONLY: no weight payloads read, no token generated')
            return
        import torch
        from real_sliding_stage5h import Geometry,TorchCarrierWeights
        from real_postlude_stage5no import run_real_full_one_token,streaming_unembed_topk
        if args.device.startswith('cuda') and not torch.cuda.is_available():
            raise ValueError('CUDA requested, unavailable')
        if args.token_id>=cfg['text_config']['vocab_size']:
            raise ValueError('token outside embeddings')
        reader=Carrier5F(non,exp)
        weights=TorchCarrierWeights(reader,args.device,Geometry.from_hf_config(cfg))
        with torch.inference_mode():
            hidden,blocks,trace=run_real_full_one_token(weights,args.token_id,
                gate_rank=cfg['text_config']['loop_config']['gate_rank'],
                log_scaling_alpha=cfg['text_config']['log_scaling_alpha'],
                log_scaling_n_floor=cfg['text_config']['log_scaling_n_floor'])
            if [r['cache_slot'] for r in trace]!=ready['cache_slots']:
                raise ValueError('real executed visits do not match slot schedule')
            winners=(streaming_unembed_topk(reader,hidden,
                     unpadded_vocab=ready['unpadded_vocab_size'],
                     softcap=ready['final_logit_softcapping'],topk=args.topk,
                     chunk_rows=args.chunk_rows) if args.mode=='greedy' else [])
        flat=hidden.detach().float().cpu().contiguous().reshape(-1)
        digest=sha256(flat.numpy().tobytes()).hexdigest()
        print('REAL POSTLUDE:')
        for r in trace[-6:]:
            print('  layer={} type={} top8={} hidden_rms={:.6g}'.format(
                r['physical_layer'],r['attention_type'],r['top8_expert_ids'],r['hidden_rms']))
        print(f'PASS 48 physical visits; final_hidden_rms={float(flat.square().mean().sqrt()):.6g} digest={digest[:20]}')
        if winners:
            print('GREEDY CANDIDATE (not HF-validated generated text):', winners[0])
            print('TOP LOGITS:', json.dumps(winners))
        report={
            'contract':'inkling2_stage5no_one_token_reference_v1',
            'model_revision':args.source.name,'mode':args.mode,
            'token_id':args.token_id,'last_hidden_fp32_sha256':digest,
            'final_hidden_rms':float(flat.square().mean().sqrt()),
            'logical_visits':48,'postlude_trace':trace[-6:],
            'predicted_next_token_candidate':winners[0]['token_id'] if winners else None,
            'top_logits':winners,
            'nonexpert_artifact_id':non.artifact_id.hex(),
            'experts_artifact_id':exp.artifact_id.hex(),
            'checkpoint_index_sha256':binding['checkpoint']['model_index_sha256'],
            'HF_end_to_end_parity':False,'native_ninfer_runtime':False,
            'multi_token_generation':False,'past_kv_conv_cache':False,
        }
        if args.output is not None:
            args.output.parent.mkdir(parents=True,exist_ok=True)
            args.output.write_text(json.dumps(report,indent=2)+'\n')
            print('REPORT:',args.output)
        print('OK Stage5N/O one-token real-weight integration (NOT production inference)')
        print('NOT YET: HF full-forward parity, position>0, multi-token decode or native ninfer-serve')


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('source','nonexpert','experts','banks'):
        p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--mode',choices=('hidden','greedy'),default='hidden')
    p.add_argument('--token-id',type=int,default=1000)
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--topk',type=int,default=5)
    p.add_argument('--chunk-rows',type=int,default=1024)
    p.add_argument('--plan',action='store_true')
    p.add_argument('--output',type=Path)
    args=p.parse_args(argv)
    try:run(args)
    except (ValueError,RuntimeError,KeyError,OSError,TypeError) as exc:
        print('ERROR:',exc,file=sys.stderr)
        return 1
    return 0

if __name__=='__main__':
    sys.exit(main())
