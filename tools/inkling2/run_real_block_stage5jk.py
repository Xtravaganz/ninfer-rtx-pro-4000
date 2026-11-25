#!/usr/bin/env python3
"""Stage 5J+5K: original-weight FULL attention 5 + first 6-layer block.

Requires the complete Inkling 2 Mini HF snapshot, Stage4A/2B v3 carriers,
Stage5H+5I with installed query-index hotfix, and Torch CPU/CUDA.
This is a real-weight integration reference, NOT `ninfer-serve`, a 48-step
full decoder, an HF numerics-parity certification, or token generation.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for path in (HERE, ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from runtime_bindings import schedule
from nonexpert_artifact import Checkpoint
from real_carrier_stage5f import Carrier5F, validate_complete_carriers
from run_real_sliding_stage5hi import mandatory_sliding_sources, readiness_from_spans


def mandatory_full_sources(config: dict) -> dict[str, tuple[tuple[int, ...], ...]]:
    """Strict original checkpoint source contracts for PHYSICAL layer 5 only."""
    t = config['text_config']
    h = t['hidden_size']; heads = t['num_attention_heads']
    kv = t['num_key_value_heads']; d = t['head_dim']
    rel = t['d_rel']; extent = t['rel_extent']; taps = t['sconv_kernel_size']
    experts = t['n_routed_experts']
    if (h != 2304 or heads != 24 or kv != 8 or d != 128 or
            rel != 16 or taps != 4 or extent != t['sliding_window_size'] or
            experts != 128 or t['num_experts_per_tok'] != 8 or
            5 in t['local_layer_ids'] or t['n_layers_per_attn_res_block'] != 6):
        raise ValueError('Stage 5J requires original full-attention layer-5 geometry')
    p = 'model.llm.layers.5.'
    a = p + 'attn.'
    contract = {
        p + 'q_input_norm.weight': ((h,),),
        p + 'k_input_norm.weight': ((h,),),
        p + 'v_input_norm.weight': ((h,),),
        p + 'mlp_norm.weight': ((h,),),
        p + 'attn_sconv.weight': ((h,taps), (h,1,taps)),
        p + 'mlp_sconv.weight': ((h,taps), (h,1,taps)),
        p + 'mlp.gate.weight': ((experts,h),),
        p + 'mlp.gate.bias': ((experts,),),
        a + 'wq_du.weight': ((heads*d,h),),
        a + 'wk_dv.weight': ((kv*d,h),),
        a + 'wv_dv.weight': ((kv*d,h),),
        a + 'wo_ud.weight': ((h,heads*d),),
        a + 'gate_attn.weight': ((heads*d,h),),
        a + 'attn_sink_proj.weight': ((heads,h),),
        a + 'q_l2_clip.scale': ((heads,d),),
        a + 'rel_l2_clip.scale': ((heads,rel),),
        a + 'rel_logits_proj.proj': ((rel,extent),),
        a + 'wr_du.weight': ((heads*rel,h),),
        a + 'k_sconv.static_w': ((taps,kv*d),),
        a + 'v_sconv.static_w': ((taps,kv*d),),
        a + 'k_sconv.weight_proj': ((kv*taps,h),),
        a + 'v_sconv.weight_proj': ((kv*taps,h),),
        'model.llm.attn_res.attn_queries.4': ((3,h),),
        'model.llm.attn_res.mlp_queries.5': ((h,),),
    }
    return contract


def first_block_preflight(spans: dict, config: dict, *, layers: int) -> dict:
    if type(layers) is not int or layers not in (5,6):
        raise ValueError('Stage 5J/K only implements 5 or 6 prelude layers')
    old = readiness_from_spans(spans, config, layers=5)
    expected = {
        'model.llm.embed.weight': ((config['text_config']['vocab_size'],
                                    config['text_config']['hidden_size']),),
        'model.llm.embed_norm.weight': ((config['text_config']['hidden_size'],),),
    }
    for l in range(5):
        expected.update(mandatory_sliding_sources(l,config))
    if layers == 6:
        expected.update(mandatory_full_sources(config))
    missing=[]; wrong=[]
    for name, shapes in sorted(expected.items()):
        span=spans.get(name)
        if span is None:
            missing.append(name)
        elif tuple(span.shape) not in shapes or span.dtype not in ('BF16','F32'):
            wrong.append((name,str(span.dtype),list(span.shape),[list(s) for s in shapes]))
    if missing or wrong:
        raise ValueError(f'5J/K unsupported full-layer weights: missing={missing[:8]}, wrong={wrong[:8]}')
    visits=schedule(config)
    if (len(visits) != 48 or [s['physical_layer'] for s in visits[:6]] != list(range(6)) or
        [s['cache_slot'] for s in visits[:6]] != list(range(6))):
        raise ValueError('invalid full 6-layer prelude schedule')
    if layers == 6 and 5 in config['text_config']['local_layer_ids']:
        raise ValueError('layer 5 must be full attention, not sliding')
    return {
        'declared_sources':len(expected), 'layers_supported':layers,
        'stage5i_sources':old['declared_sources'],
        'full_attention_source_count':len(mandatory_full_sources(config)) if layers==6 else 0,
        'full_forward_supported':False,
        'executed_physical_layers':list(range(layers)),
        'first_visit_cache_slots':[s['cache_slot'] for s in visits[:layers]],
        'separate_full_attn_qkv_inputs':layers==6,
        'has_dynamic_kv_convolution':layers==6,
    }


def run(args):
    if args.layers not in (5,6):
        raise ValueError('--layers must be 5 (baseline) or 6 (full block)')
    if type(args.token_id) is not int or args.token_id < 0:
        raise ValueError('token id must be nonnegative')
    if args.output and (args.output.exists() or args.output.is_symlink()):
        raise FileExistsError(f'no overwrite: {args.output}')
    from tools.artifact.reader import Artifact
    with Artifact(args.nonexpert) as non, Artifact(args.experts) as exp:
        bound = validate_complete_carriers(args.source,non,exp,args.banks)
        with Checkpoint(args.source) as checkpoint:
            ready = first_block_preflight(checkpoint.spans, checkpoint.config, layers=args.layers)
            config = checkpoint.config
        print('MODEL: complete v3-carrier bindings (1068 non-expert tensors + 48 expert banks)')
        print(f"READY: layers 0..{args.layers-1}, position-zero one-token, no past; full 48-step NOT supported")
        if args.plan:
            print('PLAN:',json.dumps(ready,sort_keys=True))
            return
        import torch
        from real_sliding_stage5h import Geometry, TorchCarrierWeights
        from real_full_stage5j import run_first_attention_block
        if args.device.startswith('cuda') and not torch.cuda.is_available():
            raise RuntimeError('CUDA requested but unavailable')
        geom = Geometry.from_hf_config(config)
        if args.token_id >= config['text_config']['vocab_size']:
            raise ValueError('token ID outside model vocabulary')
        reader=Carrier5F(non,exp)
        weights=TorchCarrierWeights(reader,args.device,geom)
        with torch.inference_mode():
            hidden,blocks,trace=run_first_attention_block(weights,args.token_id,args.layers,
                log_scaling_alpha=config['text_config']['log_scaling_alpha'],
                log_scaling_n_floor=config['text_config']['log_scaling_n_floor'])
        x=hidden.detach().float().cpu().contiguous().reshape(-1)
        if not bool(torch.isfinite(x).all()):
            raise ValueError('nonfinite final block hidden state')
        digest=sha256(x.numpy().tobytes()).hexdigest()
        report={
            'contract':'inkling2_stage5jk_first_real_block_v1',
            'torch_device':str(args.device),'token_id':args.token_id,
            'model_revision':args.source.name,
            'native_ninfer_model':False,'full_48_step_forward':False,
            'generated_token':False,'has_past_kv_cache':False,
            'executed_physical_layers':list(range(args.layers)),
            'last_hidden_fp32_sha256':digest,
            'hidden_rms':float(x.square().mean().sqrt()),
            'block_states':int(blocks.shape[-2]),'layer_trace':trace,
            'nonexpert_artifact_id':non.artifact_id.hex(),
            'experts_artifact_id':exp.artifact_id.hex(),
            'checkpoint_index_sha256':bound['checkpoint']['model_index_sha256'],
        }
        for rec in trace:
            print(f"REAL GPU/TORCH LAYER {rec['physical_layer']}: top8={rec['top8_expert_ids']} hidden_rms={rec['hidden_rms']:.6g}")
        if args.output:
            args.output.parent.mkdir(parents=True,exist_ok=True)
            args.output.write_text(json.dumps(report,indent=2)+'\n')
            print('TRACE:',args.output)
        print(f"OK Stage5J/K real first attention block: {args.layers} layer(s), digest={digest[:20]}")
        print('NOT YET: HF full-layer numerical parity, position>0/KV cache, core 3-pass loop, final norm/logits, native NInfer runtime')


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',required=True,type=Path,help='original HF snapshot')
    p.add_argument('--nonexpert',required=True,type=Path)
    p.add_argument('--experts',required=True,type=Path)
    p.add_argument('--banks',required=True,type=Path)
    p.add_argument('--layers',type=int,default=6,help='5 for previous baseline, 6 for Stage5J+5K')
    p.add_argument('--token-id',type=int,default=1000)
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--plan',action='store_true',help='metadata only, no weight payload reads, no torch needed')
    p.add_argument('--output',type=Path,help='optional JSON report path (no overwrite)')
    args=p.parse_args(argv)
    try:
        run(args)
    except (RuntimeError,ValueError,OSError,KeyError,TypeError) as exc:
        print('ERROR:',exc,file=sys.stderr)
        return 1
    return 0

if __name__=='__main__':
    sys.exit(main())
