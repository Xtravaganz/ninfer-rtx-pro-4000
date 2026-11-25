"""Stage 5S/5T: inspect and optionally correct Inkling W13 gate/up row order.

Official Transformers conversion_mapping.py inkling2_loop_text applies
Interleave(dim=1) to dequantized W13 expert matrices. For an original expert
weight [2*I,H], this maps row 2*j to gate row j and row 2*j+1 to up row j.
This file is additive and never edits existing kernels, artifacts, or models.
"""
from __future__ import annotations

from contextlib import contextmanager
import json
from pathlib import Path
import sys


def interleave_source_rows(data):
    """Independent form of HF Interleave(dim=1), for [2*I,H] matrices."""
    if data.ndim != 2 or data.shape[0] % 2 or data.shape[0] == 0:
        raise ValueError('W13 source must be [2*intermediate, hidden]')
    return data.reshape(data.shape[0] // 2, 2, data.shape[1]).transpose(0, 1).reshape(data.shape)


def split_gate_up_source_order(out):
    """Correct source-order W13 activation partition before deinterleaving."""
    if out.ndim != 1 or not out.numel() or out.numel() % 2:
        raise ValueError('expected even-length W13 output vector')
    return out[0::2], out[1::2]


def previous_split_gate_up(out):
    if out.ndim != 1 or not out.numel() or out.numel() % 2:
        raise ValueError('expected even-length W13 output vector')
    middle = out.numel() // 2
    return out[:middle], out[middle:]


def corrected_moe(self, hidden, layer):
    """Diagnostic replacement of TorchCarrierWeights.moe; same math but corrected W13 rows.

    Does NOT validate all other layers against HF. Retains original FP32 numerics.
    """
    import torch
    import torch.nn.functional as F
    from real_sliding_stage5h import decode_expert

    g = self.geom
    if hidden.shape != (1, 1, g.hidden):
        raise ValueError('MoE expects [1,1,hidden]')
    prefix = f'model.llm.layers.{layer}.mlp.'
    router = self.tensor(prefix + 'gate.weight', (g.experts, g.hidden))
    bias = self.tensor(prefix + 'gate.bias', (g.experts,))
    scores = torch.sigmoid(F.linear(hidden.float(), router).reshape(g.experts))
    ids_tensor = torch.topk(scores + bias, k=g.top_k, sorted=True).indices
    ids = [int(i) for i in ids_tensor.cpu().tolist()]
    multipliers = scores[ids_tensor] / (scores[ids_tensor].sum() + 1e-20) * g.route_scale
    w13 = self.reader.bank(layer, 'w13', ids)
    w2 = self.reader.bank(layer, 'w2', ids)
    if len(w13) != g.top_k or len(w2) != g.top_k:
        raise ValueError('missing selected real experts')
    fmt = 'nvfp4' if layer < 20 else 'mxfp8'
    result = torch.zeros(g.hidden, device=self.device, dtype=torch.float32)
    x = hidden.float().reshape(g.hidden)
    for i in range(g.top_k):
        gate_up_raw = decode_expert(w13[i], rows=2 * g.intermediate,
                                    cols=g.hidden, fmt=fmt, device=self.device) @ x
        gate, up = split_gate_up_source_order(gate_up_raw)
        activated = 4 * torch.tanh(gate / 4) * torch.sigmoid(gate) * 25 * torch.tanh(up / 25)
        down = decode_expert(w2[i], rows=g.hidden, cols=g.intermediate,
                             fmt=fmt, device=self.device) @ activated
        result += multipliers[i] * down
    if not bool(torch.isfinite(result).all()):
        raise ValueError('nonfinite corrected MoE')
    return result.reshape(1, 1, -1), ids


@contextmanager
def temporary_corrected_moe():
    from real_sliding_stage5h import TorchCarrierWeights
    old = TorchCarrierWeights.moe
    if old is corrected_moe:
        raise RuntimeError('cannot stack diagnostic MoE patches')
    TorchCarrierWeights.moe = corrected_moe
    try:
        yield
    finally:
        TorchCarrierWeights.moe = old


def real_expert_audit(weights, layer: int, token_id: int) -> dict:
    """Check W13 row mismatch against HF Interleave with ONE real expert."""
    import torch
    from real_sliding_stage5h import decode_expert
    g = weights.geom
    hidden = weights.embedding(token_id).reshape(-1)
    fmt = 'nvfp4' if layer < 20 else 'mxfp8'
    expert_id = 0
    raw = weights.reader.bank(layer, 'w13', [expert_id])[0]
    matrix = decode_expert(raw, rows=2*g.intermediate, cols=g.hidden, fmt=fmt, device=weights.device)
    actual = matrix @ hidden
    gate, up = split_gate_up_source_order(actual)
    rearranged = interleave_source_rows(matrix) @ hidden
    correct_gate, correct_up = rearranged[:g.intermediate], rearranged[g.intermediate:]
    if not torch.allclose(gate, correct_gate, atol=1e-4, rtol=1e-5) or not torch.allclose(up,correct_up,atol=1e-4,rtol=1e-5):
        raise RuntimeError('independent Interleave oracle failed')
    old_gate, old_up = previous_split_gate_up(actual)
    delta = float((old_gate - correct_gate).abs().max())
    return {'layer':layer,'format':fmt,'expert':expert_id,
            'max_abs_gate_partition_difference':delta,
            'hf_interleave_oracle':'PASS',
            'previous_source_half_split_matches_hf':bool(torch.equal(old_gate, correct_gate) and torch.equal(old_up, correct_up)),
            'correction':'gate=out[0::2], up=out[1::2]'}


def setup(args):
    import torch
    from tools.artifact.reader import Artifact
    from real_carrier_stage5f import Carrier5F, validate_complete_carriers
    from real_sliding_stage5h import Geometry, TorchCarrierWeights
    from nonexpert_artifact import Checkpoint
    from run_real_forward_stage5no import full_forward_preflight
    from contextlib import ExitStack
    stack=ExitStack()
    try:
        non=stack.enter_context(Artifact(args.nonexpert))
        exp=stack.enter_context(Artifact(args.experts))
        validate_complete_carriers(args.source, non, exp, args.banks)
        with Checkpoint(args.source) as check:
            full_forward_preflight(check.spans, check.config, 'greedy')
            cfg=check.config
        if args.device.startswith('cuda') and not torch.cuda.is_available():
            raise ValueError('CUDA requested but unavailable')
        weights=TorchCarrierWeights(Carrier5F(non,exp),args.device,Geometry.from_hf_config(cfg))
        return stack, weights, cfg
    except BaseException:
        stack.close()
        raise


def run_audit(args):
    stack,w,cfg=setup(args)
    try:
        import torch
        with torch.inference_mode():
            for layer in (0,20):
                rec=real_expert_audit(w,layer,args.token)
                print('REAL W13 AUDIT:',json.dumps(rec),flush=True)
                w.clear()
        print('RESULT: HF W13 row-order audit finished. This does not establish full-model parity.')
    finally:
        stack.close()


def run_compare(args):
    """Same real-weight token through unmodified and corrected MoE; report first differing visit."""
    import torch
    from parity_stage5p import TraceRecorder, _capture_student, execution_layers
    from real_postlude_stage5no import run_real_full_one_token, streaming_unembed_topk
    stack,w,cfg=setup(args)
    try:
        text=cfg['text_config']
        kwargs={'gate_rank':text['loop_config']['gate_rank'],
                'log_scaling_alpha':text['log_scaling_alpha'],
                'log_scaling_n_floor':text['log_scaling_n_floor']}
        results=[]
        with torch.inference_mode():
            for corrected in (False,True):
                rec=TraceRecorder([],[])
                patch=temporary_corrected_moe() if corrected else __import__('contextlib').nullcontext()
                with patch, _capture_student(rec):
                    hidden,_,visits=run_real_full_one_token(w,args.token,**kwargs)
                    top=streaming_unembed_topk(w.reader,hidden,topk=5,
                        unpadded_vocab=text.get('unpadded_vocab_size',200058),
                        softcap=text.get('final_logit_softcapping',30.0))
                results.append((rec.matrix(), hidden.detach().float().cpu().reshape(-1).clone(), top))
                w.clear()
                print(f'FORWARD {"HF-W13-interleave-corrected" if corrected else "existing-student"}: '
                      f'top1={top[0]["token_id"]} top5={[x["token_id"] for x in top]}',flush=True)
        a,b=results
        diffs=abs(a[0]-b[0]); magnitudes=diffs.max(axis=1)
        first=next((i for i,v in enumerate(magnitudes) if v>args.threshold),None)
        report={'test':'same original weights, different W13 row order',
                'threshold':args.threshold,'first_different_visit':first,
                'first_different_physical':execution_layers()[first] if first is not None else None,
                'first_different_max_abs':float(magnitudes[first]) if first is not None else 0.0,
                'max_abs_final_hidden':float((a[1]-b[1]).abs().max()),
                'old_top1':int(a[2][0]['token_id']), 'corrected_top1':int(b[2][0]['token_id']),
                'hf_end_to_end_validated':False}
        print('DIFFERENTIAL REPORT:',json.dumps(report,indent=2),flush=True)
        if args.output:
            if args.output.exists() or args.output.is_symlink():
                raise FileExistsError(args.output)
            args.output.parent.mkdir(parents=True,exist_ok=True)
            args.output.write_text(json.dumps(report,indent=2)+'\n')
        print('NOTE: corrected row ordering matches published HF conversion; other model mismatches may remain.')
    finally:
        stack.close()


def run_question(args):
    import argparse
    import ask_inkling_stage5r
    a=argparse.Namespace(source=args.source,nonexpert=args.nonexpert,experts=args.experts,
       banks=args.banks,question=args.question,raw=args.raw,tokenize_only=False,
       max_prompt_tokens=64,max_new_tokens=args.max_new_tokens,device=args.device)
    with temporary_corrected_moe():
        return ask_inkling_stage5r.run(a)


def main(argv=None):
    import argparse
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=('audit','compare','question'))
    for n in ('source','nonexpert','experts','banks'):
        p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--token',type=int,default=1000)
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--threshold',type=float,default=1e-4)
    p.add_argument('--question',default='What is 2 + 2?')
    p.add_argument('--max-new-tokens',type=int,default=6)
    p.add_argument('--raw',action='store_true')
    p.add_argument('--output',type=Path)
    a=p.parse_args(argv)
    if not 0<=a.token<201024 or not 1<=a.max_new_tokens<=64 or a.threshold<0:
        p.error('invalid token/count/threshold')
    try:
        if a.mode=='audit':run_audit(a)
        elif a.mode=='compare':run_compare(a)
        else: return run_question(a)
    except (OSError,RuntimeError,ValueError,TypeError,ImportError,KeyError) as exc:
        print('ERROR:',exc,file=sys.stderr)
        return 1
    return 0


if __name__=='__main__':
    raise SystemExit(main())
