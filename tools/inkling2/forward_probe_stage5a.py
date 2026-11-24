#!/usr/bin/env python3
"""Run representative Stage-5A text primitive probes on CPU or Torch CUDA.

No full model/checkpoint is loaded. This validates eager Python numerics, not a
native NInfer attention kernel or end-to-end Inkling inference.
"""
from __future__ import annotations

import argparse
import torch
from forward_oracle_stage5a import (
    relative_position_bias, causal_additive_mask, sink_attention,
    static_short_convolution, static_short_convolution_step,
    dynamic_short_convolution, block_attention_residual,
    cross_pass_messages, controller_message_gate,
)


def max_err(a, b):
    return (a.cpu().float() - b.cpu().float()).abs().max().item()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', choices=['cpu','cuda:0'],default='cpu')
    args=parser.parse_args()
    if args.device=='cuda:0' and not torch.cuda.is_available():
        parser.error('CUDA device unavailable; run with --gpus or --device cpu')
    if args.device=='cuda:0':
        print(f'CUDA GPU: {torch.cuda.get_device_name(0)}')
    torch.manual_seed(5210)
    b,t,d,heads,kv,dhead=1,8,16,4,2,8
    q=torch.randn(b,heads,5,dhead)
    k=torch.randn(b,kv,t,dhead)
    v=torch.randn(b,kv,t,dhead)
    mask=causal_additive_mask(torch.tensor([[3,4,5,6,7]]),torch.arange(t)[None],sliding_window=4)
    rel=torch.tensor([[3,4,5,6,7]])[..., :, None] - torch.arange(t)[None,None]
    hidden=torch.randn(b,5,d)
    w=torch.randn(heads*3,d)*.03
    sc=torch.ones(heads,3)
    profiles=torch.randn(3,8)*.05
    sinks=torch.randn(b,heads,5,1)*.1
    relbias=relative_position_bias(hidden,w,sc,profiles,rel,positions=torch.tensor([[3,4,5,6,7]]))
    ref=sink_attention(q,k,v,attention_mask=mask,position_bias=relbias,sinks=sinks)
    if args.device=='cpu':
        actual=ref
    else:
        dev=args.device
        bias=relative_position_bias(hidden.to(dev),w.to(dev),sc.to(dev),profiles.to(dev),rel.to(dev),
                                    positions=torch.tensor([[3,4,5,6,7]],device=dev))
        actual=sink_attention(q.to(dev),k.to(dev),v.to(dev),attention_mask=mask.to(dev),
                              position_bias=bias,sinks=sinks.to(dev))
    error=max_err(ref,actual)
    torch.testing.assert_close(actual.cpu(),ref,rtol=2e-5,atol=5e-6)
    print(f'PASS GQA + relative bias + sliding mask + sink: max_abs_error={error:.9g}')

    x=torch.randn(b,t,d); weight=torch.randn(d,4)*.05
    full=static_short_convolution(x,weight)
    history=torch.zeros(b,3,d)
    pieces=[]
    for i in range(t):
        y,history=static_short_convolution_step(x[:,i:i+1],weight,history)
        pieces.append(y)
    torch.testing.assert_close(torch.cat(pieces,dim=1),full,rtol=1e-5,atol=1e-5)
    if args.device=='cpu':
        actual_static=full
    else:
        actual_static=static_short_convolution(x.to(args.device),weight.to(args.device))
    err=max_err(full,actual_static)
    torch.testing.assert_close(actual_static.cpu(),full,rtol=2e-5,atol=5e-6)
    print(f'PASS static convolution prefill/step: max_abs_error={err:.9g}')

    dyn=torch.randn(b,t,d)
    stat=torch.randn(4,d)*.03
    proj=torch.randn(2*4,d)*.03
    seq=torch.tensor([[0,0,0,0,1,1,1,1]])
    dref=dynamic_short_convolution(x,dyn,stat,proj,num_heads=2,sequence_ids=seq)
    if args.device=='cpu':
        dgot=dref
    else:
        dgot=dynamic_short_convolution(x.to(args.device),dyn.to(args.device),stat.to(args.device),
                                       proj.to(args.device),num_heads=2,sequence_ids=seq.to(args.device))
    err=max_err(dref,dgot)
    torch.testing.assert_close(dgot.cpu(),dref,rtol=2e-5,atol=5e-6)
    print(f'PASS dynamic K/V convolution (boundary mask): max_abs_error={err:.9g}')

    partial=x[:, :5]
    previous=x[:, :5,None,:] * .1
    queries=torch.randn(3,d)
    blocks=block_attention_residual(previous,partial,queries)
    messages=cross_pass_messages(previous,partial,torch.randn(12,4,d),3)
    assert len(blocks)==3 and messages.shape==(b,5,4,d)
    zeros_down=torch.zeros(4,2*d)
    zeros_up=torch.zeros(d,4)
    restored=controller_message_gate(partial,messages[:,:,0],zeros_down,zeros_up,zeros_up)
    torch.testing.assert_close(restored,partial)
    print('PASS block residual Q=3 + four cross-pass messages + controller identity')
    print('OK Stage 5A: PyTorch primitive oracle, not native NInfer runtime')

if __name__=='__main__': main()
