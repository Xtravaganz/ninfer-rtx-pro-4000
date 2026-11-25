#!/usr/bin/env python3
"""Benchmark batched attention + convolution operators, not full model throughput."""
import argparse,json,time
import torch
from batch_prefill_stage6c import attention_prefill,static_prefill,dynamic_prefill
from forward_oracle_stage5a import sink_attention,static_short_convolution_step


def bench(fn,reps,device):
    fn()
    if device.type=='cuda': torch.cuda.synchronize(device)
    if device.type=='cuda':
        start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(reps):fn()
        end.record();end.synchronize()
        return start.elapsed_time(end)*1000/reps
    start=time.perf_counter()
    for _ in range(reps):fn()
    return (time.perf_counter()-start)*1e6/reps


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--tokens',type=int,default=16)
    p.add_argument('--repeats',type=int,default=20)
    p.add_argument('--device',default='cuda:0')
    a=p.parse_args()
    if not 1<=a.tokens<=256 or not 1<=a.repeats<=300:p.error('invalid dimensions')
    device=torch.device(a.device)
    if device.type=='cuda' and not torch.cuda.is_available():p.error('CUDA unavailable')
    torch.manual_seed(43)
    t=a.tokens
    q=torch.randn(1,24,t,128,device=device)*.03
    k=torch.randn(1,8,t,128,device=device)*.03
    v=torch.randn_like(k)
    sinks=torch.randn(1,24,t,1,device=device)*.1
    def batched():return attention_prefill(q,k,v,sink=sinks,sliding_window=1024)
    def sequential():
        out=[]
        for i in range(t):
            out.append(sink_attention(q[:,:,i:i+1],k[:,:,:i+1],v[:,:,:i+1],sinks=sinks[:,:,i:i+1]))
        return torch.cat(out,dim=1)
    with torch.inference_mode():
        torch.testing.assert_close(batched(),sequential(),atol=5e-6,rtol=5e-6)
        attention_batch_us=bench(batched,a.repeats,device)
        attention_sequential_us=bench(sequential,a.repeats,device)
        x=torch.randn(1,t,2304,device=device)*.05
        taps=torch.randn(2304,4,device=device)*.02
        def static_batch():return static_prefill(x,taps)
        def static_seq():
            state=torch.zeros(1,3,2304,device=device)
            ans=[]
            for i in range(t):
                y,state=static_short_convolution_step(x[:,i:i+1],taps,state)
                ans.append(y)
            return torch.cat(ans,dim=1)
        torch.testing.assert_close(static_batch(),static_seq(),atol=5e-6,rtol=5e-6)
        static_batch_us=bench(static_batch,a.repeats,device)
        static_sequential_us=bench(static_seq,a.repeats,device)
    result={'device':str(device),'tokens':t,'repeats':a.repeats,
       'attention_batched_us':round(attention_batch_us,3),
       'attention_sequential_us':round(attention_sequential_us,3),
       'attention_operator_speedup':round(attention_sequential_us/attention_batch_us,3),
       'static_conv_batched_us':round(static_batch_us,3),
       'static_conv_sequential_us':round(static_sequential_us,3),
       'static_conv_operator_speedup':round(static_sequential_us/static_batch_us,3),
       'disclaimer':'Operator-only benchmark; not Inkling 48-step batch-prefill.'}
    print('STAGE6C OPERATOR BENCH:',json.dumps(result,sort_keys=True),flush=True)
if __name__=='__main__':main()
