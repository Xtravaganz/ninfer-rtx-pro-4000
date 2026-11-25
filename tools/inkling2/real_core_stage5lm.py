"""Inkling 2 Mini Stage 5L/M: real-weight one-token three-pass loop reference.

This is an FP32 Torch integration smoke on ORIGINAL NInfer-v3 carriers, not
Hugging Face end-to-end parity, a KV/conv cache implementation, native NInfer,
or generated text. The only supported position is 0 (no previous tokens).

Stage 5L: execute one core pass (physical layers 6..11 or 6..17).
Stage 5M: execute the 12-layer core three times, with original learned
cross-pass message queries, tanh controllers, and reset to the prelude state.
"""
from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F

from real_full_stage5j import run_first_attention_block
from forward_oracle_stage5a import (
    weighted_rmsnorm, unweighted_rmsnorm, l2_clip, scaled_l2_clip,
    relative_position_bias, sink_attention, gated_output,
    static_short_convolution, dynamic_short_convolution,
    block_attention_residual, cross_pass_messages, controller_message_gate,
)

CORE_LAYERS = tuple(range(6, 18))
FULL_CORE_LAYERS = (11, 17)
MESSAGE_PATHS = ("attn", "k", "v", "mlp")


def _controller(weights: Any, layer: int, path: str,
                hidden: torch.Tensor, message: torch.Tensor,
                gate_rank: int) -> torch.Tensor:
    if path not in MESSAGE_PATHS or layer not in CORE_LAYERS:
        raise ValueError('unsupported message controller')
    h = weights.geom.hidden
    name = f'model.llm.layers.{layer}.controllers.{path}.'
    down = weights.tensor(name + 'down_weight', (gate_rank, 2 * h))
    forget = weights.tensor(name + 'forget_weight', (h, gate_rank))
    gain = weights.tensor(name + 'gain_weight', (h, gate_rank))
    # HF: two unweighted RMS norms, then low-rank down projection and
    # independent FP32 tanh gates. This is NOT a sigmoid interpolation.
    return controller_message_gate(hidden, message, down, forget, gain,
                                   eps=weights.geom.norm_eps)


def _cross_queries(weights: Any) -> torch.Tensor:
    """Map original [24,D] query/MLP and K/V tensors -> [12,4,D]."""
    d = weights.geom.hidden
    qm = weights.tensor('model.llm.middle_cycle_cross_pass_queries', (24, d))
    kv = weights.tensor('model.llm.middle_cycle_cross_pass_kv_queries', (24, d))
    qm = qm.reshape(12, 2, d)
    kv = kv.reshape(12, 2, d)
    return torch.cat((qm[:, :1, :], kv, qm[:, 1:, :]), dim=1)


def core_one_token_layer(weights: Any, hidden: torch.Tensor,
                         blocks: torch.Tensor, *, layer: int,
                         message: torch.Tensor | None = None,
                         core_pass: int = 0, gate_rank: int = 64,
                         log_scaling_alpha: float = .1,
                         log_scaling_n_floor: int = 8192
                         ) -> tuple[torch.Tensor, torch.Tensor, dict]:
    """One real-weight core layer, sliding except physical 11/17.

    Core pass 0 has no messages. Later passes MUST provide four learned
    cross-pass messages. The caller owns pass resets and the 48-slot map.
    """
    g = weights.geom
    if (type(layer) is not int or layer not in CORE_LAYERS or
            type(core_pass) is not int or core_pass not in (0, 1, 2) or
            type(gate_rank) is not int or gate_rank < 1 or
            hidden.shape != (1, 1, g.hidden) or
            blocks.ndim != 4 or blocks.shape[:2] != (1, 1) or
            blocks.shape[-1] != g.hidden or blocks.shape[-2] not in (1, 2, 3)):
        raise ValueError('only Inkling original core layers, one position-0 token')
    if (core_pass == 0 and message is not None or
            core_pass > 0 and (message is None or
            message.shape != (1, 1, 4, g.hidden))):
        raise ValueError('cross-pass message presence/geometry mismatch')
    if (not math.isfinite(log_scaling_alpha) or log_scaling_alpha < 0 or
            type(log_scaling_n_floor) is not int or log_scaling_n_floor < 1):
        raise ValueError('invalid log-scaling parameters')

    is_full = layer in FULL_CORE_LAYERS
    p = f'model.llm.layers.{layer}.'
    a = p + 'attn.'
    hq, hk, d, rel = g.query_heads, g.kv_heads, g.head_dim, g.rel_dim
    # Before attention use previous attention-residual block queries.
    q_shape = (3, g.hidden) if is_full else (g.hidden,)
    attn_q = weights.tensor(f'model.llm.attn_res.attn_queries.{layer-1}', q_shape)
    prepared = block_attention_residual(blocks, hidden, attn_q, eps=g.norm_eps)
    if is_full:
        q_hidden, k_hidden, v_hidden = prepared
    else:
        q_hidden = k_hidden = v_hidden = prepared[0]
    if message is not None:
        qmsg, kmsg, vmsg, mlpmsg = message.unbind(-2)
        q_hidden = _controller(weights, layer, 'attn', q_hidden, qmsg, gate_rank)
        k_hidden = _controller(weights, layer, 'k', k_hidden, kmsg, gate_rank)
        v_hidden = _controller(weights, layer, 'v', v_hidden, vmsg, gate_rank)
    if is_full:
        q_hidden = weighted_rmsnorm(q_hidden,
            weights.tensor(p+'q_input_norm.weight',(g.hidden,)),g.norm_eps)
        k_hidden = weighted_rmsnorm(k_hidden,
            weights.tensor(p+'k_input_norm.weight',(g.hidden,)),g.norm_eps)
        v_hidden = weighted_rmsnorm(v_hidden,
            weights.tensor(p+'v_input_norm.weight',(g.hidden,)),g.norm_eps)
    else:
        norm = weights.tensor(p+'attn_norm.weight',(g.hidden,))
        q_hidden = weighted_rmsnorm(q_hidden,norm,g.norm_eps)
        k_hidden = weighted_rmsnorm(k_hidden,norm,g.norm_eps)
        v_hidden = weighted_rmsnorm(v_hidden,norm,g.norm_eps)

    positions = torch.zeros((1, 1),dtype=torch.long,device=weights.device)
    q = F.linear(q_hidden, weights.tensor(a+'wq_du.weight',(hq*d,g.hidden)))
    clip_args = dict(positions=positions,log_scaling_alpha=log_scaling_alpha,
                     log_scaling_n_floor=log_scaling_n_floor) if is_full else {}
    q = scaled_l2_clip(q.reshape(1,1,hq,d),
         weights.tensor(a+'q_l2_clip.scale',(hq,d)),g.qk_clip,
         **clip_args).transpose(1,2)
    k = F.linear(k_hidden,weights.tensor(a+'wk_dv.weight',(hk*d,g.hidden)))
    v = F.linear(v_hidden,weights.tensor(a+'wv_dv.weight',(hk*d,g.hidden)))
    if is_full:
        def dynamic_conv(projected, input_hidden, channel):
            tap = weights.tensor(a+channel+'.static_w',(g.conv_kernel,hk*d))
            proj = weights.tensor(a+channel+'.weight_proj',(hk*g.conv_kernel,g.hidden))
            return dynamic_short_convolution(projected,input_hidden,tap,proj,num_heads=hk)
        k = dynamic_conv(k,k_hidden,'k_sconv')
        v = dynamic_conv(v,v_hidden,'v_sconv')
    else:
        k = static_short_convolution(k,weights.conv(a+'k_sconv.weight',hk*d))
        v = static_short_convolution(v,weights.conv(a+'v_sconv.weight',hk*d))
    k = l2_clip(k.reshape(1,1,hk,d),g.qk_clip).transpose(1,2)
    v = unweighted_rmsnorm(v.reshape(1,1,hk,d),g.norm_eps).transpose(1,2)
    bias = relative_position_bias(q_hidden,
        weights.tensor(a+'wr_du.weight',(hq*rel,g.hidden)),
        weights.tensor(a+'rel_l2_clip.scale',(hq,rel)),
        weights.tensor(a+'rel_logits_proj.proj',(rel,g.rel_extent)),
        torch.zeros((1,1),dtype=torch.long,device=weights.device),
        threshold=g.rel_clip,**clip_args)
    sinks = F.linear(q_hidden,weights.tensor(a+'attn_sink_proj.weight',(hq,g.hidden)))
    attended = sink_attention(q,k,v,position_bias=bias,
                              sinks=sinks.transpose(1,2).unsqueeze(-1))
    out = gated_output(attended.reshape(1,1,hq*d),q_hidden,
        weights.tensor(a+'gate_attn.weight',(hq*d,g.hidden)),
        weights.tensor(a+'wo_ud.weight',(g.hidden,hq*d)))
    attn_conv = static_short_convolution(out,weights.conv(p+'attn_sconv.weight',g.hidden))
    # At layer 6 or 12: preserve the incoming block-residual hidden state and
    # DO NOT add it to post-attention output (HF boundary convention).
    if layer % 6 == 0:
        residual = attn_conv.float()
        blocks = torch.cat((blocks,hidden.unsqueeze(-2)),dim=-2)
    else:
        residual = hidden.float() + attn_conv.float()
    query_mlp = weights.tensor(f'model.llm.attn_res.mlp_queries.{layer}',(g.hidden,))
    mlp_input = block_attention_residual(blocks,residual,query_mlp,eps=g.norm_eps)[0]
    if message is not None:
        mlp_input = _controller(weights,layer,'mlp',mlp_input,mlpmsg,gate_rank)
    mlp_input = weighted_rmsnorm(mlp_input,
        weights.tensor(p+'mlp_norm.weight',(g.hidden,)),g.norm_eps)
    routed,ids = weights.moe(mlp_input,layer)
    mlp_conv = static_short_convolution(routed,weights.conv(p+'mlp_sconv.weight',g.hidden))
    result = residual + mlp_conv.float()
    if not bool(torch.isfinite(result).all()):
        raise ValueError('nonfinite core hidden state')
    return result, blocks, {
        'physical_layer':layer,'core_pass':core_pass,
        'attention_type':'full' if is_full else 'sliding',
        'messages_applied':message is not None,
        'top8_expert_ids':ids,'hidden_rms':float(result.square().mean().sqrt()),
        'block_states':int(blocks.shape[-2]),
    }


def run_real_core(weights: Any, token_id: int, *, core_layers: int = 6,
                  passes: int = 1, gate_rank: int = 64,
                  log_scaling_alpha: float = .1,
                  log_scaling_n_floor: int = 8192
                  ) -> tuple[torch.Tensor,torch.Tensor,list[dict]]:
    """6-layer prelude then core first block or all 3 passes, never postlude.

    core_layers 6 => 0..11, exactly ONE core pass.
    core_layers 12 => 0..17, 1..3 passes. Three passes are stage 5M.
    """
    if (type(core_layers) is not int or core_layers not in (6, 12) or
        type(passes) is not int or passes not in (1, 2, 3) or
        passes > 1 and core_layers != 12 or
        type(gate_rank) is not int or gate_rank < 1):
        raise ValueError('Stage5L/M supports 6/12 core layers, three passes only for full core')
    pre_hidden, pre_blocks, trace = run_first_attention_block(
        weights, token_id, 6,
        log_scaling_alpha=log_scaling_alpha,
        log_scaling_n_floor=log_scaling_n_floor)
    trace = [dict(record,stage='prelude',cache_slot=record['physical_layer']) for record in trace]
    if pre_blocks.shape[-2] != 1:
        raise ValueError('prelude must contain exactly one residual-block state')
    hidden,blocks = pre_hidden,pre_blocks
    queries = _cross_queries(weights) if passes > 1 else None
    for core_pass in range(passes):
        if core_pass:
            # HF deliberately discards the first loop-boundary state (at layer6)
            # and attends to the remaining block state(s) + final hidden.
            previous_blocks = blocks[...,pre_blocks.shape[-2]+1:,:]
            if previous_blocks.shape[-2] != 1:
                raise ValueError('previous core pass missing layer12 residual block')
            previous_hidden = hidden
            hidden,blocks=pre_hidden,pre_blocks
        for layer in range(6,6+core_layers):
            message = (cross_pass_messages(previous_blocks,previous_hidden,queries,
                       layer-6,eps=weights.geom.norm_eps) if core_pass else None)
            hidden,blocks,record = core_one_token_layer(
                weights,hidden,blocks,layer=layer,message=message,
                core_pass=core_pass,gate_rank=gate_rank,
                log_scaling_alpha=log_scaling_alpha,
                log_scaling_n_floor=log_scaling_n_floor)
            record['stage']='core'
            record['cache_slot'] = layer if core_pass==0 else 24+(core_pass-1)*12+layer-6
            trace.append(record)
            weights.clear()
        if core_layers==12 and blocks.shape[-2] != 3:
            raise ValueError('full core pass must append exactly two block states')
    return hidden,blocks,trace
