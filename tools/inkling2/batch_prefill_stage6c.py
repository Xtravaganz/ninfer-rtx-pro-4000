"""Stage6C batched *operator* prefill reference, NOT a full Inkling batch forward.

Provides exact GQA+sink, static and dynamic conv on [B,T,*] tensors.
No model-weight loading, router batching, block-residual or 48-visit orchestration.
Use as a parity gate before replacing Stage5Q's sequential model path.
"""
from __future__ import annotations

import torch
from forward_oracle_stage5a import (
    causal_additive_mask, sink_attention,
    static_short_convolution, dynamic_short_convolution,
)


def attention_prefill(q, k, v, *, sink, relative_bias=None,
                      sliding_window=None, query_positions=None,
                      key_positions=None):
    if q.ndim!=4 or k.ndim!=4 or v.shape!=k.shape:
        raise ValueError('q [B,Hq,T,D], k/v [B,Hkv,T,D] required')
    b,hq,t,d=q.shape
    if t<1 or k.shape[0]!=b or k.shape[2]!=t or hq%k.shape[1]:
        raise ValueError('attention dimensions mismatch')
    if query_positions is None: query_positions=torch.arange(t,device=q.device)[None].expand(b,t)
    if key_positions is None: key_positions=query_positions
    if query_positions.shape!=(b,t) or key_positions.shape!=(b,t):
        raise ValueError('invalid position layout')
    if sink.shape!=(b,hq,t,1):
        raise ValueError('sink shape must be [B,Hq,T,1]')
    mask=causal_additive_mask(query_positions,key_positions,sliding_window=sliding_window)
    return sink_attention(q,k,v,attention_mask=mask,position_bias=relative_bias,sinks=sink)


def static_prefill(x,taps,padding_mask=None):
    return static_short_convolution(x,taps,padding_mask=padding_mask)


def dynamic_prefill(x,context,static_taps,projection,heads,*,padding_mask=None,sequence_ids=None):
    return dynamic_short_convolution(x,context,static_taps,projection,
                    num_heads=heads,padding_mask=padding_mask,sequence_ids=sequence_ids)


def vectorized_linear(x,weight):
    if x.ndim!=3 or weight.ndim!=2 or x.shape[-1]!=weight.shape[-1]:
        raise ValueError('linear expects [B,T,K] @ [N,K]')
    return torch.nn.functional.linear(x,weight)
