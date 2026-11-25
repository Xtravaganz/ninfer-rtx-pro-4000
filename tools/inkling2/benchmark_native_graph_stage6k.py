#!/usr/bin/env python3
"""GPU0 only: one real Inkling bank, captured graph vs native Stage6H."""
from __future__ import annotations
import argparse,json,statistics
from pathlib import Path
from time import perf_counter


def run(a):
    import torch
    from tools.artifact.reader import Artifact
    from real_carrier_stage5f import Carrier5F
    from real_sliding_stage5h import Geometry
    from prefill_cache_stage5u import CachedCarrierWeights
    from native_resident_stage6ghi import NativeResidentLibrary
    from native_stage6jkl import GraphMoE,require_device0
    require_device0('cuda:0')
    device=torch.device('cuda:0')
    with Artifact(a.nonexpert) as non,Artifact(a.experts) as exp:
        reader=Carrier5F(non,exp)
        w=CachedCarrierWeights(reader,device,Geometry(),nonexpert_mib=256,expert_mib=0)
        native=NativeResidentLibrary(a.library)
        graph=None
        try:
            native.upload(a.layer,reader)
            p=f'model.llm.layers.{a.layer}.mlp.'
            router=w.tensor(p+'gate.weight',(128,2304)).contiguous()
            bias=w.tensor(p+'gate.bias',(128,)).contiguous()
            torch.manual_seed(23)
            x=torch.randn(2304,device=device,dtype=torch.float32)*0.025
            direct=torch.empty_like(x);captured=torch.empty_like(x)
            s=torch.cuda.current_stream(device)
            native.run(x,router,bias,8.0,a.layer,direct,s)
            s.synchronize()
            graph=GraphMoE(native)
            t=perf_counter();graph.capture(a.layer,router,bias,8.0,s)
            capture_s=perf_counter()-t
            graph.run(x,captured,a.layer,s)
            s.synchronize()
            err=float((captured-direct).abs().max().item())
            if not torch.allclose(captured,direct,atol=.0001,rtol=.0001):
                raise AssertionError(f'CUDA Graph vs native resident MoE mismatch: {err}')
            def time_event(fn):
                out=[]
                for i in range(a.repeats+2):
                    st=torch.cuda.Event(enable_timing=True);ed=torch.cuda.Event(enable_timing=True)
                    st.record(s);fn();ed.record(s);ed.synchronize()
                    if i>=2:out.append(st.elapsed_time(ed)*1000)
                return statistics.median(out)
            one=time_event(lambda:native.run(x,router,bias,8.0,a.layer,direct,s))
            fast=time_event(lambda:graph.run(x,captured,a.layer,s))
            print('STAGE6K NATIVE GRAPH:',json.dumps({
                'device':'cuda:0','layer':a.layer,'format':'nvfp4' if a.layer<20 else 'mxfp8',
                'capture_s':round(capture_s,3),'direct_us':round(one,3),
                'graph_us':round(fast,3),'graph_operator_speedup':round(one/fast,3),
                'max_abs_vs_native_resident':err,'graph_counters':graph.metrics(),
                'disclaimer':'MoE-only CUDA Graph, not entire model graph',
            },sort_keys=True),flush=True)
        finally:
            s=torch.cuda.current_stream(device);s.synchronize()
            if graph is not None:graph.close()
            native.close();w.release()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('nonexpert','experts','library'):p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--layer',type=int,default=0)
    p.add_argument('--repeats',type=int,default=20)
    a=p.parse_args()
    if not 0<=a.layer<24 or not 1<=a.repeats<=100:p.error('invalid options')
    run(a)
if __name__=='__main__':main()
