#!/usr/bin/env python3
"""Real bank A/B: unchanged 6H vs Stage7A/B, one physical layer per process."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import statistics


def run(a):
    import torch
    from tools.artifact.reader import Artifact
    from real_carrier_stage5f import Carrier5F
    from real_sliding_stage5h import Geometry
    from prefill_cache_stage5u import CachedCarrierWeights
    from native_resident_stage6ghi import NativeResidentLibrary
    from native_stage7ab import FastMoE
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA GPU required')
    dev=torch.device('cuda:0'); torch.cuda.set_device(dev)
    s=torch.cuda.current_stream(dev)
    with Artifact(a.nonexpert) as non, Artifact(a.experts) as exp:
        reader=Carrier5F(non,exp)
        g=Geometry(); scale=g.route_scale
        w=CachedCarrierWeights(reader,dev,g,nonexpert_mib=256,expert_mib=0)
        native=None; fast=None
        try:
            native=NativeResidentLibrary(a.base,max_batch_tokens=1)
            native.upload(a.layer,reader)
            fast=FastMoE(native,a.library)
            p=f'model.llm.layers.{a.layer}.mlp.'
            router=w.tensor(p+'gate.weight',(128,2304)).contiguous()
            bias=w.tensor(p+'gate.bias',(128,)).contiguous()
            out0=torch.empty(2304,device=dev,dtype=torch.float32); out1=torch.empty_like(out0)
            torch.manual_seed(23)
            inputs=[torch.zeros_like(out0)]+[torch.randn_like(out0)*v for v in (0.025,0.1,0.5,1.0) for _ in range(a.samples)]
            maxerr=maxcoef=maxrelative=0.; mismatches=0
            def baseline(x): native.run(x,router,bias,scale,a.layer,out0,s)
            def candidate(x): fast.run(x,router,bias,scale,a.layer,out1,s)
            for i,x in enumerate(inputs):
                baseline(x); candidate(x)
                ids,coef,ref_ids,ref_coef=fast.routes(x,router,bias,scale,s)
                s.synchronize()
                if not torch.equal(ids,ref_ids):
                    mismatches+=1
                    raise AssertionError(f'route mismatch sample {i}: fast={ids.tolist()} reference={ref_ids.tolist()}')
                maxcoef=max(maxcoef,(coef-ref_coef).abs().max().item())
                if not torch.allclose(coef,ref_coef,atol=2e-5,rtol=2e-5):
                    raise AssertionError(f'coefficient parity sample {i}: {maxcoef}')
                err=(out0-out1).abs().max().item(); maxerr=max(maxerr,err)
                maxrelative=max(maxrelative,err/max(out0.abs().max().item(),1e-12))
                if not torch.isfinite(out1).all() or not torch.allclose(out0,out1,atol=.005,rtol=.002):
                    raise AssertionError(f'output parity sample {i}: max_abs={err}')
            x=inputs[1]
            # Optional graph test on a non-default stream, no stage6K changes.
            graph_error=None
            if a.graph:
                capture_stream=torch.cuda.Stream(device=dev)
                capture_stream.wait_stream(s)
                graph=torch.cuda.CUDAGraph()
                with torch.cuda.stream(capture_stream):
                    fast.run(x,router,bias,scale,a.layer,out1,capture_stream)
                capture_stream.synchronize()
                with torch.cuda.graph(graph,stream=capture_stream):
                    fast.run(x,router,bias,scale,a.layer,out1,capture_stream)
                graph.replay(); s.synchronize()
                baseline(x); s.synchronize()
                graph_error=(out1-out0).abs().max().item()
                if not torch.allclose(out0,out1,atol=.005,rtol=.002):
                    raise AssertionError(f'graph parity: {graph_error}')
                graph.reset()
            # Interleave AB/BA ordering to reduce clock/thermal bias.
            def time(fn):
                start=torch.cuda.Event(enable_timing=True); end=torch.cuda.Event(enable_timing=True)
                start.record(s);fn(x);end.record(s);end.synchronize()
                return start.elapsed_time(end)*1000
            for _ in range(5): baseline(x);candidate(x)
            s.synchronize()
            base=[]; opt=[]
            for i in range(a.repeats):
                if i%2: opt.append(time(candidate));base.append(time(baseline))
                else: base.append(time(baseline));opt.append(time(candidate))
            b,c=statistics.median(base),statistics.median(opt)
            print('STAGE7AB NATIVE DECODE:',json.dumps(dict(layer=a.layer,format='nvfp4' if a.layer<20 else 'mxfp8',
                  device=str(dev),base_us=round(b,3),fast_us=round(c,3),speedup=round(b/c,3),
                  route_samples=len(inputs),route_mismatches=mismatches,max_abs_coeff_error=maxcoef,
                  max_abs_output_error=maxerr,max_relative_to_output_peak=maxrelative,graph_max_abs_vs_6h=graph_error,
                  resident_mib=native.metrics()['native_resident_mib'],repeats=a.repeats,
                  disclaimer='single MoE operator only; synthetic hidden vectors with real weights; not full model or tokens/s'),sort_keys=True),flush=True)
        finally:
            torch.cuda.synchronize(dev)
            if fast is not None: fast.close()
            if native is not None: native.close()
            w.release()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('nonexpert','experts','base','library'): p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--layer',type=int,default=0);p.add_argument('--samples',type=int,default=3)
    p.add_argument('--repeats',type=int,default=25);p.add_argument('--graph',action='store_true')
    a=p.parse_args()
    if not 0<=a.layer<24 or not 1<=a.samples<=32 or not 5<=a.repeats<=200: p.error('invalid layer/samples/repeats')
    run(a)
if __name__=='__main__': main()
