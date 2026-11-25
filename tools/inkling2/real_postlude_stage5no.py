"""Stage 5N/5O: original-carrier position-zero postlude and bounded logit scan.

Torch FP32 integration reference, NOT HF parity and NOT a native NInfer runtime.
The full 48 visits use 24 physical weight sets, with the core repeated 3x.
Only ONE input token at position zero, with empty attention/conv history.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import torch
import torch.nn.functional as F

from real_core_stage5lm import run_real_core
from forward_oracle_stage5a import (
    weighted_rmsnorm, unweighted_rmsnorm, l2_clip, scaled_l2_clip,
    relative_position_bias, sink_attention, gated_output,
    static_short_convolution, dynamic_short_convolution,
    block_attention_residual,
)

POSTLUDE = tuple(range(18, 24))
FULL_ATTENTION = (23,)


def postlude_one_token_layer(weights: Any, hidden: torch.Tensor,
                             blocks: torch.Tensor, layer: int, *,
                             log_scaling_alpha: float = 0.1,
                             log_scaling_n_floor: int = 8192):
    """One physical postlude layer. Layer 23 has dynamic full K/V convolution."""
    g = weights.geom
    if (type(layer) is not int or layer not in POSTLUDE or
            hidden.shape != (1, 1, g.hidden) or
            blocks.ndim != 4 or blocks.shape[:2] != (1, 1) or
            blocks.shape[-1] != g.hidden or
            blocks.shape[-2] != (3 if layer == 18 else 4)):
        raise ValueError('Stage5N postlude requires physical 18..23 and correct residual block count')
    if (not math.isfinite(log_scaling_alpha) or log_scaling_alpha < 0 or
            type(log_scaling_n_floor) is not int or log_scaling_n_floor <= 0):
        raise ValueError('invalid log scaling')
    full = layer == 23
    p = f'model.llm.layers.{layer}.'
    a = p + 'attn.'
    hq, hk, d, rel = g.query_heads, g.kv_heads, g.head_dim, g.rel_dim

    # HF checkpoint attn_queries.(physical_layer-1). Last full layer uses Q/K/V.
    attn_query = weights.tensor(f'model.llm.attn_res.attn_queries.{layer-1}',
                                (3, g.hidden) if full else (g.hidden,))
    prepared = block_attention_residual(blocks, hidden, attn_query, eps=g.norm_eps)
    if full:
        q_hidden, k_hidden, v_hidden = prepared
        q_hidden = weighted_rmsnorm(q_hidden, weights.tensor(p+'q_input_norm.weight', (g.hidden,)), g.norm_eps)
        k_hidden = weighted_rmsnorm(k_hidden, weights.tensor(p+'k_input_norm.weight', (g.hidden,)), g.norm_eps)
        v_hidden = weighted_rmsnorm(v_hidden, weights.tensor(p+'v_input_norm.weight', (g.hidden,)), g.norm_eps)
    else:
        norm = weights.tensor(p+'attn_norm.weight', (g.hidden,))
        q_hidden = weighted_rmsnorm(prepared[0], norm, g.norm_eps)
        k_hidden = v_hidden = q_hidden

    pos = torch.zeros((1, 1), dtype=torch.long, device=weights.device)
    log_args = dict(positions=pos, log_scaling_alpha=log_scaling_alpha,
                    log_scaling_n_floor=log_scaling_n_floor) if full else {}
    q = F.linear(q_hidden, weights.tensor(a+'wq_du.weight', (hq*d,g.hidden)))
    q = scaled_l2_clip(q.reshape(1,1,hq,d),
          weights.tensor(a+'q_l2_clip.scale', (hq,d)),g.qk_clip,**log_args).transpose(1,2)
    k = F.linear(k_hidden, weights.tensor(a+'wk_dv.weight', (hk*d,g.hidden)))
    v = F.linear(v_hidden, weights.tensor(a+'wv_dv.weight', (hk*d,g.hidden)))
    if full:
        def dynamic(projected, input_hidden, key):
            tap=weights.tensor(a+key+'.static_w',(g.conv_kernel,hk*d))
            projection=weights.tensor(a+key+'.weight_proj',(hk*g.conv_kernel,g.hidden))
            return dynamic_short_convolution(projected,input_hidden,tap,projection,num_heads=hk)
        k=dynamic(k,k_hidden,'k_sconv')
        v=dynamic(v,v_hidden,'v_sconv')
    else:
        k=static_short_convolution(k,weights.conv(a+'k_sconv.weight',hk*d))
        v=static_short_convolution(v,weights.conv(a+'v_sconv.weight',hk*d))
    k=l2_clip(k.reshape(1,1,hk,d),g.qk_clip).transpose(1,2)
    v=unweighted_rmsnorm(v.reshape(1,1,hk,d),g.norm_eps).transpose(1,2)
    bias=relative_position_bias(q_hidden,
          weights.tensor(a+'wr_du.weight',(hq*rel,g.hidden)),
          weights.tensor(a+'rel_l2_clip.scale',(hq,rel)),
          weights.tensor(a+'rel_logits_proj.proj',(rel,g.rel_extent)),
          torch.zeros((1,1),dtype=torch.long,device=weights.device),
          threshold=g.rel_clip,**log_args)
    sinks=F.linear(q_hidden,weights.tensor(a+'attn_sink_proj.weight',(hq,g.hidden)))
    attended=sink_attention(q,k,v,position_bias=bias,
                            sinks=sinks.transpose(1,2).unsqueeze(-1))
    out=gated_output(attended.reshape(1,1,hq*d),q_hidden,
          weights.tensor(a+'gate_attn.weight',(hq*d,g.hidden)),
          weights.tensor(a+'wo_ud.weight',(g.hidden,hq*d)))
    attn_conv=static_short_convolution(out,weights.conv(p+'attn_sconv.weight',g.hidden))
    # First postlude layer starts the fourth residual block; no add of incoming hidden.
    if layer == 18:
        residual=attn_conv.float()
        blocks=torch.cat((blocks,hidden.unsqueeze(-2)),dim=-2)
    else:
        residual=hidden.float()+attn_conv.float()
    mlp_query=weights.tensor(f'model.llm.attn_res.mlp_queries.{layer}',(g.hidden,))
    mlp_input=block_attention_residual(blocks,residual,mlp_query,eps=g.norm_eps)[0]
    mlp_input=weighted_rmsnorm(mlp_input,weights.tensor(p+'mlp_norm.weight',(g.hidden,)),g.norm_eps)
    routed,ids=weights.moe(mlp_input,layer)
    mlp_conv=static_short_convolution(routed,weights.conv(p+'mlp_sconv.weight',g.hidden))
    result=residual+mlp_conv.float()
    if not bool(torch.isfinite(result).all()):
        raise ValueError('nonfinite postlude hidden state')
    return result,blocks,dict(stage='postlude',physical_layer=layer,core_pass=None,
        cache_slot=layer,attention_type='full' if full else 'sliding',
        dynamic_kv_convolution=full,top8_expert_ids=ids,
        hidden_rms=float(result.square().mean().sqrt()),block_states=int(blocks.shape[-2]))


def finish_hidden(weights: Any, hidden: torch.Tensor, blocks: torch.Tensor):
    """Official final-input block-attention residual and weighted RMSNorm."""
    g=weights.geom
    if hidden.shape != (1,1,g.hidden) or blocks.shape != (1,1,4,g.hidden):
        raise ValueError('final residual expects four saved attention blocks')
    query=weights.tensor('model.llm.attn_res.output_query',(g.hidden,))
    hidden=block_attention_residual(blocks,hidden,query,eps=g.norm_eps)[0]
    result=weighted_rmsnorm(hidden,weights.tensor('model.llm.norm.weight',(g.hidden,)),g.norm_eps)
    if not bool(torch.isfinite(result).all()):
        raise ValueError('nonfinite final hidden state')
    return result


def run_real_full_one_token(weights: Any, token_id: int, *,gate_rank: int = 64,
                            log_scaling_alpha: float = 0.1,
                            log_scaling_n_floor: int = 8192):
    """48 visits + final residual/norm; no logits, past or generation."""
    hidden,blocks,trace=run_real_core(weights,token_id,core_layers=12,passes=3,
        gate_rank=gate_rank,log_scaling_alpha=log_scaling_alpha,
        log_scaling_n_floor=log_scaling_n_floor)
    trace=list(trace)
    if len(trace)!=42 or blocks.shape[-2]!=3:
        raise ValueError('core 3-pass preconditions failed')
    for layer in POSTLUDE:
        hidden,blocks,record=postlude_one_token_layer(weights,hidden,blocks,layer,
            log_scaling_alpha=log_scaling_alpha,
            log_scaling_n_floor=log_scaling_n_floor)
        trace.append(record)
        weights.clear()
    if (len(trace)!=48 or [x['physical_layer'] for x in trace[-6:]]!=list(POSTLUDE)
        or len(set(record['cache_slot'] for record in trace))!=48):
        raise ValueError('invalid complete execution schedule')
    return finish_hidden(weights,hidden,blocks),blocks,trace


def _bounded_logit_scan(artifact: Any, hidden: torch.Tensor, *,
                        shape: tuple[int,int], unpadded_vocab: int,
                        softcap: float|None, topk: int, chunk_rows: int) -> list[dict]:
    """Implementation also exercised on small synthetic tensor artifacts."""
    obj=artifact.object('hf/model.llm.unembed.weight')
    if (obj.kind!='tensor' or obj.format not in ('bf16','fp32') or
            tuple(obj.shape)!=shape or
            hidden.numel()!=shape[1] or not 0<unpadded_vocab<=shape[0] or
            type(topk) is not int or not 1<=topk<=10 or
            type(chunk_rows) is not int or not 1<=chunk_rows<=4096 or
            (softcap is not None and (not math.isfinite(softcap) or softcap<=0))):
        raise ValueError('invalid unembedding geometry/dtype/scan parameters')
    stride=shape[1]*(2 if obj.format=='bf16' else 4)
    winners=[]
    x=hidden.reshape(shape[1]).float()
    for start in range(0,unpadded_vocab,chunk_rows):
        rows=min(chunk_rows,unpadded_vocab-start)
        raw=artifact.read_range(obj.offset+start*stride,rows*stride)
        if len(raw)!=rows*stride:
            raise ValueError('short unembedding read')
        dtype=torch.bfloat16 if obj.format=='bf16' else torch.float32
        matrix=torch.frombuffer(bytearray(raw),dtype=dtype).reshape(rows,shape[1])
        matrix=matrix.to(device=x.device,dtype=torch.float32)
        logits=matrix@x
        if softcap is not None:
            logits=softcap*torch.tanh(logits/softcap)
        if not bool(torch.isfinite(logits).all()):
            raise ValueError('nonfinite logits')
        # Stable sort matters when many logits saturate at the softcap.
        # HF torch.argmax selects the first index on ties; keep that ordering.
        indices=torch.argsort(logits,descending=True,stable=True)[:min(rows,topk)]
        val=logits[indices]
        winners.extend((float(v),start+int(i)) for v,i in zip(val.cpu().tolist(),indices.cpu().tolist()))
        winners.sort(key=lambda pair:(-pair[0],pair[1]))
        winners=winners[:topk]
    return [dict(token_id=idx,logit=value) for value,idx in winners]


def streaming_unembed_topk(reader: Any, hidden: torch.Tensor, *,
                           unpadded_vocab: int=200058,
                           softcap: float|None=30., topk: int=5,
                           chunk_rows: int=1024) -> list[dict]:
    """Scan the 201024-row padded unembedding without loading its full weights.

    Exactly the first 200058 output rows are eligible for argmax. This is
    a *candidate*, not a validated text generation pipeline.
    """
    if not isinstance(hidden,torch.Tensor) or hidden.shape!=(1,1,2304) or not bool(torch.isfinite(hidden).all()):
        raise ValueError('expected finite Inkling final hidden [1,1,2304]')
    if type(unpadded_vocab) is not int or unpadded_vocab!=200058:
        raise ValueError('unexpected Inkling output vocabulary')
    return _bounded_logit_scan(reader.nonexpert,hidden,shape=(201024,2304),
        unpadded_vocab=unpadded_vocab,softcap=softcap,topk=topk,chunk_rows=chunk_rows)
