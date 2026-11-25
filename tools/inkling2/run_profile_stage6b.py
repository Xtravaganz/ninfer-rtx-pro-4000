#!/usr/bin/env python3
"""Instrument real Stage5W question without editing the existing runtime.

--sync-per-operation is diagnostic and intentionally changes runtime timings.
Use baseline Stage5W for end-to-end throughput comparisons.
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for p in (HERE,ROOT):
    if str(p) not in sys.path: sys.path.insert(0,str(p))

from profile_stage6b import SpanProfiler, PatchScope


def run(args):
    import torch
    import ask_native_stage5w as native
    import real_sliding_stage5h as source
    import streaming_stage5q as stream
    import ask_inkling_stage5r as ask
    import native_moe_stage5w as moe
    import quantized_store_stage5v as quant
    from tools.artifact import reader as artifact_reader

    profiler = SpanProfiler(synchronize_cuda=args.sync_per_operation)
    state = {'w':None, 'prompt':None}
    original_init = moe.NativeQuantizedCarrierWeights.__init__
    original_run = native.run
    original_tokenize = ask.tokenize_question
    original_forward = stream.forward_one_token

    def weight_init(self, *a, **kw):
        original_init(self,*a,**kw)
        state['w'] = self
        for meth,label in [('tensor','nonexpert_tensor'),('conv','convolution_weight'),('embedding','embedding'),('moe','moe_total')]:
            profiler.wrap(self,meth,label,device_attr='device' if args.sync_per_operation else None)
        profiler.wrap(self.quant,'select','expert_select_and_upload')
        profiler.wrap(self.quant,'_upload','expert_cpu_to_gpu_upload')
        profiler.wrap(self.quant.reader,'bank','expert_artifact_bank_select')
        profiler.wrap(self.native,'run','native_cuda_launch')

    def tokenize(*a, **kw):
        with profiler.span('tokenization'):
            ids=original_tokenize(*a,**kw)
        state['prompt']=len(ids)
        return ids

    def forward(*a, **kw):
        device=getattr(a[0],'device',None) if a else None
        with profiler.span('forward_48', device=device):
            return original_forward(*a,**kw)

    with PatchScope() as patch:
        patch.patch(moe.NativeQuantizedCarrierWeights,'__init__',weight_init)
        patch.patch(ask,'tokenize_question',tokenize)
        patch.patch(stream,'forward_one_token',forward)
        # Existing Stage5W monkey-patches source.TorchCarrierWeights inside native.run;
        # wrapping its constructor instruments the instantiated NativeQuantizedCarrierWeights.
        settings=argparse.Namespace(**vars(args))
        result=None
        try:
            result=original_run(settings)
        finally:
            report=profiler.result()
            w=state['w']
            if w is not None:
                try: report['cache']=w.metrics()
                except Exception: pass
            report['prompt_tokens']=state['prompt']
            report['note']='Stage5W itself synchronizes after each native MoE and for each full token.'
            path=args.output
            if path:
                path.parent.mkdir(parents=True,exist_ok=True)
                if path.exists(): raise FileExistsError(path)
                path.write_text(json.dumps(report,indent=2,sort_keys=True))
            print('STAGE6B PROFILE:',json.dumps(report,sort_keys=True),flush=True)
    return result or 0


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('source','nonexpert','experts','banks','library'):
        p.add_argument('--'+name,required=True,type=Path)
    p.add_argument('--question',default='Hi')
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--max-new-tokens',type=int,default=1)
    p.add_argument('--max-prompt-tokens',type=int,default=64)
    p.add_argument('--nonexpert-cache-mib',type=int,default=4096)
    p.add_argument('--quantized-cache-mib',type=int,default=8192)
    p.add_argument('--verify-first',action='store_true')
    p.add_argument('--per-token-profile',action='store_true')
    p.add_argument('--tokenize-only',action='store_true')
    p.add_argument('--raw',action='store_true')
    p.add_argument('--sync-per-operation',action='store_true')
    p.add_argument('--output',type=Path)
    a=p.parse_args()
    if a.max_new_tokens<1 or a.max_prompt_tokens<1: p.error('positive token counts required')
    return run(a)
if __name__=='__main__': raise SystemExit(main())
