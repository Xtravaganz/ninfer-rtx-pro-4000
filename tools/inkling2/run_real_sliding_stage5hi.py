#!/usr/bin/env python3
"""Stages 5H/5I: first real sliding decoder + five-layer prelude (position zero).

DOES NOT implement full 48-step Inkling forward, tokenization, logits, or
ninfer-serve. Torch is used for the reference execution (CPU/CUDA).
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for location in (HERE, ROOT):
    if str(location) not in sys.path:
        sys.path.insert(0, str(location))

from runtime_bindings import schedule
from nonexpert_artifact import Checkpoint
from real_carrier_stage5f import Carrier5F, validate_complete_carriers


def mandatory_sliding_sources(layer: int, config: dict) -> dict[str, tuple[tuple[int,...],...]]:
    """An intentionally strict supported-subset manifest for prelude layers 0..4."""
    if type(layer) is not int or layer not in range(5):
        raise ValueError('5H/5I supports only first five sliding layers')
    c=config['text_config']
    d=c['hidden_size']; q=c['num_attention_heads']*c['head_dim']
    kv=c['num_key_value_heads']*c['head_dim']; nh=c['num_attention_heads']
    rel=c['d_rel']; extent=c['sliding_window_size']; k=c['sconv_kernel_size']
    experts=c['n_routed_experts']
    p=f'model.llm.layers.{layer}.'; a=p+'attn.'
    data = {
        p+'attn_norm.weight':((d,),), p+'mlp_norm.weight':((d,),),
        p+'attn_sconv.weight':((d,k),(d,1,k)),
        p+'mlp_sconv.weight':((d,k),(d,1,k)),
        p+'mlp.gate.weight':((experts,d),),
        p+'mlp.gate.bias':((experts,),),
        a+'wq_du.weight':((q,d),), a+'wk_dv.weight':((kv,d),),
        a+'wv_dv.weight':((kv,d),), a+'wo_ud.weight':((d,q),),
        a+'gate_attn.weight':((q,d),), a+'attn_sink_proj.weight':((nh,d),),
        a+'k_sconv.weight':((kv,k),(kv,1,k)),
        a+'v_sconv.weight':((kv,k),(kv,1,k)),
        a+'q_l2_clip.scale':((nh,c['head_dim']),),
        a+'rel_l2_clip.scale':((nh,rel),),
        a+'rel_logits_proj.proj':((rel,extent),),
        a+'wr_du.weight':((nh*rel,d),),
        f'model.llm.attn_res.mlp_queries.{layer}':((d,),),
    }
    if layer > 0:
        data[f'model.llm.attn_res.attn_queries.{layer-1}']=((d,),)
    return data


def readiness_from_spans(spans: dict, cfg: dict, *, layers: int) -> dict:
    """Metadata-only preflight, no payload reads; unknown weights fail closed."""
    if type(layers) is not int or not 1 <= layers <= 5:
        raise ValueError('requested prelude exceeds supported sliding layers 0..4')
    expected = {
        'model.llm.embed.weight':((cfg['text_config']['vocab_size'],cfg['text_config']['hidden_size']),),
        'model.llm.embed_norm.weight':((cfg['text_config']['hidden_size'],),),
    }
    for layer in range(layers):
        expected.update(mandatory_sliding_sources(layer,cfg))
    missing=[]; wrong=[]
    for name,shapes in sorted(expected.items()):
        span=spans.get(name)
        if span is None:
            missing.append(name)
        elif (tuple(span.shape) not in shapes or span.dtype not in ('BF16','F32')):
            wrong.append((name,str(span.dtype),list(span.shape),[list(s) for s in shapes]))
    if missing or wrong:
        raise ValueError(f'5H/5I unsupported source geometry: missing={missing[:5]}, wrong={wrong[:5]}')
    plan=schedule(cfg)
    if len(plan)!=48 or [s['physical_layer'] for s in plan[:layers]]!=list(range(layers)):
        raise ValueError('48-step schedule does not start with expected prelude')
    return {'declared_sources':len(expected),'layers_supported':layers,
            'full_schedule_visits':len(plan),'full_forward_supported':False,
            'first_visit_cache_slots':[s['cache_slot'] for s in plan[:layers]]}


def run(args):
    if args.layers not in range(1,6):
        raise ValueError('--layers must be 1..5; full-attention and loop NOT implemented')
    if args.token_id < 0:
        raise ValueError('token ID must be nonnegative')
    if args.output and (args.output.exists() or args.output.is_symlink()):
        raise FileExistsError(f'no overwrite: {args.output}')
    from tools.artifact.reader import Artifact
    with Artifact(args.nonexpert) as non, Artifact(args.experts) as exp:
        plan=validate_complete_carriers(args.source,non,exp,args.banks)
        with Checkpoint(args.source) as checkpoint:
            ready=readiness_from_spans(checkpoint.spans,checkpoint.config,layers=args.layers)
        print('MODEL: full metadata bindings verified; 1068 non-expert tensors + 48 banks')
        print(f'READY: supported sliding prelude layers 0..{args.layers-1}; full 48-step execution NOT supported')
        if args.plan:
            print('PLAN:',json.dumps(ready,sort_keys=True))
            return
        import torch
        from real_sliding_stage5h import Geometry, TorchCarrierWeights, run_prelude
        if args.device.startswith('cuda') and not torch.cuda.is_available():
            raise RuntimeError('CUDA device requested but unavailable')
        geometry=Geometry.from_hf_config(json.loads((args.source/'config.json').read_text()))
        vocab=json.loads((args.source/'config.json').read_text())['text_config']['vocab_size']
        if args.token_id>=vocab:
            raise ValueError('token ID outside vocabulary')
        reader=Carrier5F(non,exp)
        weights=TorchCarrierWeights(reader,args.device,geometry)
        with torch.inference_mode():
            hidden,blocks,trace=run_prelude(weights,args.token_id,args.layers)
        hidden_f32=hidden.float().detach().cpu().contiguous().reshape(-1)
        if not bool(torch.isfinite(hidden_f32).all()):
            raise ValueError('nonfinite final hidden state')
        digest=sha256(hidden_f32.numpy().tobytes()).hexdigest()
        report={'contract':'inkling2_stage5hi_real_sliding_v1',
                'native_ninfer_model':False,'full_48_step_forward':False,
                'generated_token':False,'torch_device':str(args.device),
                'token_id':args.token_id,'executed_physical_layers':list(range(args.layers)),
                'hidden_fp32_sha256':digest,'hidden_rms':float(hidden_f32.square().mean().sqrt()),
                'block_states':int(blocks.shape[-2]),'layer_trace':trace,
                'nonexpert_artifact_id':non.artifact_id.hex(),
                'experts_artifact_id':exp.artifact_id.hex(),
                'checkpoint_index_sha256':plan['checkpoint']['model_index_sha256']}
        for record in trace:
            print(f"REAL GPU/TORCH LAYER {record['physical_layer']}: top8={record['top8_expert_ids']} hidden_rms={record['hidden_rms']:.6g}")
        if args.output:
            args.output.parent.mkdir(parents=True,exist_ok=True)
            args.output.write_text(json.dumps(report,indent=2)+'\n')
            print('TRACE:',args.output)
        print(f"OK real-weight sliding prelude: {args.layers} layer(s), digest={digest[:20]}")
        print('NOT YET: physical full-attention layer 5, loop messages, 48-step forward, logits and native NInfer runtime')


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',required=True,type=Path,help='full original HF snapshot')
    p.add_argument('--nonexpert',required=True,type=Path)
    p.add_argument('--experts',required=True,type=Path)
    p.add_argument('--banks',required=True,type=Path,help='Stage-2A original bank directory')
    p.add_argument('--layers',type=int,default=1,help='1..5; first five sliding prelude layers only')
    p.add_argument('--token-id',type=int,default=1000)
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--plan',action='store_true',help='only metadata, no Torch kernel or payload read')
    p.add_argument('--output',type=Path,help='optional JSON trace path, refuses overwrite')
    args=p.parse_args(argv)
    try:
        run(args)
    except (RuntimeError,ValueError,OSError,KeyError) as exc:
        print('ERROR:',exc,file=sys.stderr)
        return 1
    return 0

if __name__=='__main__':
    sys.exit(main())
