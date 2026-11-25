#!/usr/bin/env python3
"""Stage5Q: experimental real-weight 2-token autoregressive cache smoke.

GPU-resident per-slot K/V plus 192 convolution histories; no production claims.
Modes:
  --tokens 1000,1001: two supplied input tokens, inspect final states
  --seed-token 1000 --steps 2 --greedy: produce *unvalidated* token candidates
The first-token hidden should match Stage5N/O up to floating-point arithmetic.
HF full-model parity and true native ninfer-serve integration remain separate.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
for p in (HERE,ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0,str(p))

from nonexpert_artifact import Checkpoint
from real_carrier_stage5f import Carrier5F, validate_complete_carriers
from run_real_forward_stage5no import full_forward_preflight


def run(args) -> dict:
    import torch
    from tools.artifact.reader import Artifact
    from real_sliding_stage5h import Geometry, TorchCarrierWeights
    from real_postlude_stage5no import streaming_unembed_topk
    from streaming_stage5q import IncrementalCache48, forward_one_token
    from real_postlude_stage5no import run_real_full_one_token
    if args.output is not None and (args.output.exists() or args.output.is_symlink()):
        raise FileExistsError('refusing to overwrite an existing report')
    if args.greedy and args.tokens is not None:
        raise ValueError('choose fixed --tokens OR --greedy, not both')
    if not args.greedy and args.tokens is None:
        raise ValueError('fixed input token IDs required when not using --greedy')
    if args.tokens:
        ids = [int(v) for v in args.tokens.split(',')]
        if not 1 <= len(ids) <= 3:
            raise ValueError('real-cache smoke supports 1..3 input tokens')
    else:
        ids = [args.seed_token]
    if not args.greedy and args.steps != 2:
        raise ValueError('--steps only changes the number of --greedy steps')
    if args.greedy and not 1 <= args.steps <= 3:
        raise ValueError('real-cache greedy smoke supports 1..3 steps')
    if not all(0 <= token < 201024 for token in ids):
        raise ValueError('input token outside padded vocabulary')
    if args.device.startswith('cuda') and not torch.cuda.is_available():
        raise ValueError('CUDA requested but unavailable')
    with Artifact(args.nonexpert) as non, Artifact(args.experts) as exp:
        validate_complete_carriers(args.source,non,exp,args.banks)
        with Checkpoint(args.source) as checkpoint:
            ready=full_forward_preflight(checkpoint.spans,checkpoint.config,'greedy' if args.greedy else 'hidden')
            config=checkpoint.config
        g=Geometry.from_hf_config(config)
        w=TorchCarrierWeights(Carrier5F(non,exp),args.device,g)
        c=IncrementalCache48(kernel=g.conv_kernel)
        text=config['text_config']
        record=[]
        n=args.steps if args.greedy else len(ids)
        print(f'REAL CACHE: revision={args.source.name} tokens={n} greedy={args.greedy}; '
              '48 logical K/V slots + 192 convolution paths')
        print('MODEL: verified complete original-weight NInfer-v3 carriers')
        with torch.inference_mode():
            for i in range(n):
                token=ids[i]
                hidden,trace=forward_one_token(w,c,token,
                    gate_rank=text['loop_config']['gate_rank'],
                    log_scaling_alpha=text['log_scaling_alpha'],
                    log_scaling_n_floor=text['log_scaling_n_floor'])
                if i == 0 and args.check_legacy:
                    legacy,_,_ = run_real_full_one_token(w,token,
                        gate_rank=text['loop_config']['gate_rank'],
                        log_scaling_alpha=text['log_scaling_alpha'],
                        log_scaling_n_floor=text['log_scaling_n_floor'])
                    tolerance = 1.0e-5 + 2.0e-5*legacy.float().abs()
                    delta = (hidden.float()-legacy.float()).abs()
                    if bool((delta > tolerance).any()):
                        raise ValueError(f'position-0 stateful vs Stage5N/O mismatch: max_abs={float(delta.max())}')
                    print(f'PARITY position0 vs Stage5N/O: max_abs={float(delta.max()):.9g} PASS')
                    w.clear()
                top=(streaming_unembed_topk(w.reader,hidden,
                    unpadded_vocab=text.get('unpadded_vocab_size',200058),
                    softcap=text.get('final_logit_softcapping',30.0),
                    topk=5) if args.greedy else [])
                h=hidden.detach().float().cpu().contiguous().reshape(-1)
                digest=sha256(h.numpy().tobytes()).hexdigest()
                row={'position':i,'input_token':token,'hidden_sha256':digest,
                     'hidden_rms':float(h.square().mean().sqrt()),
                     'top1_candidate':top[0]['token_id'] if top else None,
                     'top_logits':top,
                     'visits':len(trace),'kv_slots':len(c.kv),
                     'conv_paths':len(c.conv),
                     'cache_slot_schedule':[r['cache_slot'] for r in trace]}
                record.append(row)
                print(f'POSITION {i}: input={token} final_hidden_rms={row["hidden_rms"]:.6g} '
                      f'digest={digest[:20]} kv={len(c.kv)} conv={len(c.conv)} '
                      f'greedy_candidate={row["top1_candidate"]}')
                if args.greedy and i+1<n:
                    ids.append(int(top[0]['token_id']))
                w.clear()
        report={'contract':'inkling2_stage5q_experimental_stateful_48_v1',
                'checkpoint_revision':args.source.name,
                'steps':n,'position0_legacy_digest':'bac20fc5403c224d1f30',
                'results':record,'hf_end_to_end_parity':False,
                'production_inference':False,'native_ninfer_runtime':False,
                'cache_state':{'tokens':c.next_position,'kv_slots':len(c.kv),
                               'conv_paths':len(c.conv)}}
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report,indent=2)+'\n')
            print('REPORT:',args.output)
        print('OK Stage5Q: experimental real-weight sequential cache smoke')
        print('NOT YET: official HF end-to-end parity or registered native NInfer runtime')
        return report


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('source','nonexpert','experts','banks'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--tokens',help='fixed comma-separated input IDs, 1..3 tokens')
    p.add_argument('--seed-token',type=int,default=1000)
    p.add_argument('--steps',type=int,default=2)
    p.add_argument('--greedy',action='store_true',help='generate next token candidates via unembedding')
    p.add_argument('--check-legacy',action='store_true',help='repeat position 0 through Stage5N/O; compare every hidden dimension')
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--output',type=Path)
    args=p.parse_args(argv)
    try:run(args)
    except (OSError,ValueError,RuntimeError,KeyError,TypeError,ImportError) as e:
        print('ERROR:',e,file=sys.stderr)
        return 1
    return 0

if __name__=='__main__':
    raise SystemExit(main())
