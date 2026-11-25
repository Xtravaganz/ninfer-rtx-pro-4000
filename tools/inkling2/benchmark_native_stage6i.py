#!/usr/bin/env python3
"""Real-carrier native MoE A/B (batch vs one-token, both GPU-side routed).

Measures ONLY MoE (CUDA events), not Inkling's 48-visit forward or tokens/s.
Loads exactly one physical layer's 128 experts. No HF model needed.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import statistics
from time import perf_counter


def run(a):
    import torch
    from tools.artifact.reader import Artifact
    from real_carrier_stage5f import Carrier5F
    from native_resident_stage6ghi import NativeResidentLibrary
    from prefill_cache_stage5u import CachedCarrierWeights
    if not torch.cuda.is_available():
        raise RuntimeError('real CUDA GPU required')
    device = torch.device('cuda:0')
    with Artifact(a.nonexpert) as non, Artifact(a.experts) as exp:
        reader = Carrier5F(non, exp)
        from real_sliding_stage5h import Geometry
        # Reuse the validated model geometry, including route_scale and norm metadata.
        weights = CachedCarrierWeights(reader, device, Geometry(), nonexpert_mib=256, expert_mib=0)
        torch.manual_seed(17)
        x = torch.randn(a.tokens,2304,dtype=torch.float32,device=device)*0.03
        router = weights.tensor(f'model.llm.layers.{a.layer}.mlp.gate.weight',(128,2304)).contiguous()
        bias = weights.tensor(f'model.llm.layers.{a.layer}.mlp.gate.bias',(128,)).contiguous()
        native = NativeResidentLibrary(a.library,max_batch_tokens=max(a.tokens,16))
        try:
            t0=perf_counter()
            native.upload(a.layer,reader)
            cold=perf_counter()-t0
            s=torch.cuda.current_stream(device)
            sequential = torch.empty_like(x)
            batched = torch.empty_like(x)
            def one():
                for i in range(a.tokens):
                    native.run(x[i],router,bias,a.route_scale,a.layer,sequential[i],s)
            def block():
                native.run_batch(x,router,bias,a.route_scale,a.layer,batched,s)
            one();block();s.synchronize()
            if not torch.allclose(sequential,batched,atol=0.0001,rtol=0.0001):
                diff=float(torch.max(torch.abs(sequential-batched)).item())
                raise ValueError(f'Native 6I batch vs 6H tokenwise mismatch: {diff}')
            error=float(torch.max(torch.abs(sequential-batched)).item())
            if not torch.isfinite(batched).all():
                raise ValueError('Nonfinite native MoE result')
            def measure(fn):
                samples=[]
                for _ in range(a.repeats):
                    start=torch.cuda.Event(enable_timing=True)
                    end=torch.cuda.Event(enable_timing=True)
                    start.record(s);fn();end.record(s);end.synchronize()
                    samples.append(float(start.elapsed_time(end))*1000)
                return statistics.median(samples)
            seq=measure(one)
            bat=measure(block)
            print('STAGE6I NATIVE BATCH MOE:',json.dumps({
                'device': str(device),'layer':a.layer,'format':'nvfp4' if a.layer<20 else 'mxfp8',
                'tokens':a.tokens,'repeats':a.repeats,
                'native_tokenwise_us':round(seq,3),'native_batch_us':round(bat,3),
                'moe_operator_speedup':round(seq/bat,3),
                'max_abs_vs_native_tokenwise':error,
                'cold_read_and_upload_s':round(cold,3),
                'route_scale':a.route_scale,
                'disclaimer':'MoE-only CUDA operator; 48-visit model remains sequential Torch',
                'metrics':native.metrics(),
            },sort_keys=True),flush=True)
        finally:
            s.synchronize()
            native.close()
            weights.release()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--library',required=True,type=Path)
    p.add_argument('--nonexpert',required=True,type=Path)
    p.add_argument('--experts',required=True,type=Path)
    p.add_argument('--layer',type=int,default=0)
    p.add_argument('--tokens',type=int,default=16)
    p.add_argument('--repeats',type=int,default=8)
    p.add_argument('--route-scale',type=float,default=8.0)
    a=p.parse_args()
    if not 0<=a.layer<24 or not 1<=a.tokens<=128 or not 1<=a.repeats<=100:
        p.error('layer 0..23, tokens 1..128, repeats 1..100')
    run(a)
if __name__=='__main__':
    main()
