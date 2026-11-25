#!/usr/bin/env python3
"""GPU0 only: native Stage6L causal convolution vs torch and native tokenwise."""
from __future__ import annotations
import argparse
import json
import statistics
from time import perf_counter


def reference(x,taps,dynamic):
    import torch
    T,C=x.shape
    history=torch.zeros(3,C,device=x.device,dtype=x.dtype)
    output=[]
    for t in range(T):
        win=torch.cat((history,x[t:t+1]),dim=0)
        current_taps=taps[t] if dynamic else taps
        if dynamic:
            val=(win.flip(0).T*current_taps).sum(-1)
            out=(val+x[t])/3.0
        else:
            val=(win.T*current_taps).sum(-1)
            out=(val+x[t])/(5.0**0.5)
        output.append(out)
        history=win[1:]
    return torch.stack(output)


def benchmark(a):
    import torch
    from native_stage6jkl import ConvState,require_device0
    require_device0('cuda:0')
    device=torch.device('cuda:0')
    torch.manual_seed(173)
    x=(torch.randn(a.tokens,a.channels,device=device,dtype=torch.float32)*0.2).contiguous()
    taps=(torch.randn((a.tokens,a.channels,4) if a.dynamic else (a.channels,4),
                      device=device,dtype=torch.float32)*0.3).contiguous()
    ref=reference(x,taps,a.dynamic)
    single=ConvState(a.library)
    batch=ConvState(a.library)
    serial=torch.empty_like(x);parallel=torch.empty_like(x)
    try:
        def tokenwise():
            for t in range(a.tokens):
                single.apply(0,0,x[t:t+1],taps[t:t+1] if a.dynamic else taps,
                             serial[t:t+1],dynamic=a.dynamic)
        def batchwise():
            batch.apply(0,0,x,taps,parallel,dynamic=a.dynamic)
        tokenwise();batchwise();torch.cuda.synchronize(device)
        err_serial=float((serial-ref).abs().max().item())
        err_batch=float((parallel-ref).abs().max().item())
        err_pair=float((serial-parallel).abs().max().item())
        if not torch.allclose(serial,ref,rtol=0.0001,atol=0.0001):
            raise AssertionError(f'serial/native vs Torch mismatch: {err_serial}')
        if not torch.allclose(parallel,ref,rtol=0.0001,atol=0.0001):
            raise AssertionError(f'batch/native vs Torch mismatch: {err_batch}')
        # Continuation checks persistent causal history across a second call.
        nextx=torch.randn(2,a.channels,device=device,dtype=torch.float32)*0.2
        nexttaps=(torch.randn(2,a.channels,4,device=device,dtype=torch.float32)*0.3
                  if a.dynamic else taps)
        next_serial=torch.empty_like(nextx);next_batch=torch.empty_like(nextx)
        for t in range(2):
            single.apply(0,0,nextx[t:t+1],nexttaps[t:t+1] if a.dynamic else nexttaps,
                         next_serial[t:t+1],dynamic=a.dynamic)
        batch.apply(0,0,nextx,nexttaps,next_batch,dynamic=a.dynamic)
        torch.cuda.synchronize(device)
        continuation_error=float((next_serial-next_batch).abs().max().item())
        if not torch.allclose(next_serial,next_batch,rtol=0.00001,atol=0.00001):
            raise AssertionError(f'cross-call history mismatch: {continuation_error}')
        s=torch.cuda.current_stream(device)
        def time_event(fn):
            values=[]
            for i in range(a.repeats+2):
                st=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
                st.record(s);fn();end.record(s);end.synchronize()
                if i>=2:values.append(st.elapsed_time(end)*1000)
            return statistics.median(values)
        us_serial=time_event(tokenwise)
        us_batch=time_event(batchwise)
        print('STAGE6L NATIVE CONV:',json.dumps({
            'device':'cuda:0','tokens':a.tokens,'channels':a.channels,
            'dynamic':a.dynamic,'native_tokenwise_us':round(us_serial,3),
            'native_batch_us':round(us_batch,3),
            'operator_speedup':round(us_serial/us_batch,3),
            'max_abs_native_serial_vs_torch':err_serial,
            'max_abs_native_batch_vs_torch':err_batch,
            'max_abs_native_batch_vs_serial':err_pair,
            'continuation_max_abs':continuation_error,
            'disclaimer':'Convolution-only, not 48-visit whole-model batch prefill',
        },sort_keys=True),flush=True)
    finally:
        single.close();batch.close()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--library',required=True)
    p.add_argument('--tokens',type=int,default=16)
    p.add_argument('--channels',type=int,default=2304)
    p.add_argument('--repeats',type=int,default=12)
    p.add_argument('--dynamic',action='store_true')
    a=p.parse_args()
    if not 1<=a.tokens<=256 or not 1<=a.channels<=2304 or not 1<=a.repeats<=100:
        p.error('invalid benchmark geometry')
    benchmark(a)
if __name__=='__main__':main()
