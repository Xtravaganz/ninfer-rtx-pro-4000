"""Stage 5J/5K: position-zero *real-carrier* full-attention layer 5 and 6-layer block.

Torch integration reference only, not an HF parity claim, native NInfer runtime,
or generated token. Stage 5J is restricted to the FIRST full layer (physical 5)
with empty KV/conv history. Stage 5K extends the verified sliding prelude 0..4.
Both use the *original* NInfer-v3 weight carriers; no new weight copy is made.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from real_sliding_stage5h import TorchCarrierWeights, sliding_one_token_layer
from forward_oracle_stage5a import (
    weighted_rmsnorm, unweighted_rmsnorm, l2_clip, scaled_l2_clip,
    relative_position_bias, sink_attention, gated_output,
    static_short_convolution, dynamic_short_convolution,
    block_attention_residual,
)


def full_attention_one_token_layer(
    weights: TorchCarrierWeights,
    hidden: torch.Tensor,
    blocks: torch.Tensor,
    *,
    layer: int = 5,
    position: int = 0,
    log_scaling_alpha: float = 0.1,
    log_scaling_n_floor: int = 8192,
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    """First full-attention layer, single token without past.

    The source `attn_res.attn_queries.4` is [3,D] for physical layer 5;
    `attn_res.mlp_queries.5` is [D]. No sliding shortcut is used for full K/V:
    they require *dynamic* K/V convolution and their own input norms.
    """
    g = weights.geom
    if (type(layer) is not int or layer != 5 or type(position) is not int or position != 0
            or hidden.shape != (1, 1, g.hidden) or blocks.ndim != 4
            or blocks.shape[:2] != (1, 1) or blocks.shape[-1] != g.hidden
            or blocks.shape[-2] != 1):
        raise ValueError('Stage 5J supports ONLY first full layer 5 after blocks 0..4 at position zero')
    if not (math.isfinite(log_scaling_alpha) and log_scaling_alpha >= 0 and
            type(log_scaling_n_floor) is int and log_scaling_n_floor > 0):
        raise ValueError('invalid full-attention log scaling configuration')

    p = 'model.llm.layers.5.'
    a = p + 'attn.'
    hq, hk, d, rel, conv_taps = g.query_heads, g.kv_heads, g.head_dim, g.rel_dim, g.conv_kernel
    pos_ids = torch.zeros((1, 1), dtype=torch.long, device=weights.device)

    attn_queries = weights.tensor('model.llm.attn_res.attn_queries.4', (3, g.hidden))
    q_hidden, k_hidden, v_hidden = block_attention_residual(
        blocks, hidden, attn_queries, eps=g.norm_eps)
    q_hidden = weighted_rmsnorm(
        q_hidden, weights.tensor(p + 'q_input_norm.weight', (g.hidden,)), g.norm_eps)
    k_hidden = weighted_rmsnorm(
        k_hidden, weights.tensor(p + 'k_input_norm.weight', (g.hidden,)), g.norm_eps)
    v_hidden = weighted_rmsnorm(
        v_hidden, weights.tensor(p + 'v_input_norm.weight', (g.hidden,)), g.norm_eps)

    q = F.linear(q_hidden, weights.tensor(a + 'wq_du.weight', (hq * d, g.hidden)))
    q = scaled_l2_clip(q.reshape(1, 1, hq, d),
        weights.tensor(a + 'q_l2_clip.scale', (hq, d)), g.qk_clip,
        positions=pos_ids, log_scaling_alpha=log_scaling_alpha,
        log_scaling_n_floor=log_scaling_n_floor).transpose(1, 2)

    k_projected = F.linear(k_hidden, weights.tensor(a + 'wk_dv.weight', (hk * d, g.hidden)))
    v_projected = F.linear(v_hidden, weights.tensor(a + 'wv_dv.weight', (hk * d, g.hidden)))

    def dynamic_kv(projected: torch.Tensor, input_hidden: torch.Tensor, name: str) -> torch.Tensor:
        static = weights.tensor(a + name + '.static_w', (conv_taps, hk * d))
        projection = weights.tensor(a + name + '.weight_proj', (hk * conv_taps, g.hidden))
        return dynamic_short_convolution(projected, input_hidden, static,
            projection, num_heads=hk)

    k = dynamic_kv(k_projected, k_hidden, 'k_sconv')
    v = dynamic_kv(v_projected, v_hidden, 'v_sconv')
    k = l2_clip(k.reshape(1, 1, hk, d), g.qk_clip).transpose(1, 2)
    v = unweighted_rmsnorm(v.reshape(1, 1, hk, d), g.norm_eps).transpose(1, 2)

    full_extent = weights.tensor(a + 'rel_logits_proj.proj', (rel, g.rel_extent))
    bias = relative_position_bias(q_hidden,
        weights.tensor(a + 'wr_du.weight', (hq * rel, g.hidden)),
        weights.tensor(a + 'rel_l2_clip.scale', (hq, rel)),
        full_extent, torch.zeros((1, 1), dtype=torch.long, device=weights.device),
        threshold=g.rel_clip, positions=pos_ids,
        log_scaling_alpha=log_scaling_alpha, log_scaling_n_floor=log_scaling_n_floor)
    sinks = F.linear(q_hidden, weights.tensor(a + 'attn_sink_proj.weight', (hq, g.hidden)))
    attended = sink_attention(q, k, v, position_bias=bias, sinks=sinks.transpose(1, 2).unsqueeze(-1))
    out = gated_output(attended.reshape(1, 1, hq * d), q_hidden,
        weights.tensor(a + 'gate_attn.weight', (hq * d, g.hidden)),
        weights.tensor(a + 'wo_ud.weight', (g.hidden, hq * d)))
    attn_conv = static_short_convolution(out, weights.conv(p + 'attn_sconv.weight', g.hidden))
    residual = hidden.float() + attn_conv.float()  # layer 5 is NOT a block boundary
    mlp_query = weights.tensor('model.llm.attn_res.mlp_queries.5', (g.hidden,))
    mlp_input = block_attention_residual(blocks, residual, mlp_query, eps=g.norm_eps)[0]
    mlp_input = weighted_rmsnorm(mlp_input,
        weights.tensor(p + 'mlp_norm.weight', (g.hidden,)), g.norm_eps)
    routed, ids = weights.moe(mlp_input, layer)
    mlp_conv = static_short_convolution(routed, weights.conv(p + 'mlp_sconv.weight', g.hidden))
    result = residual + mlp_conv.float()
    if not bool(torch.isfinite(result).all()):
        raise ValueError('nonfinite first full-attention decoder output')
    return result, blocks, {
        'physical_layer': 5, 'attention_type': 'full_attention',
        'dynamic_kv_convolution': True, 'position': position,
        'top8_expert_ids': ids,
        'hidden_rms': float(result.square().mean().sqrt()),
    }


def run_first_attention_block(weights: TorchCarrierWeights, token_id: int,
                              num_layers: int, *, log_scaling_alpha: float = 0.1,
                              log_scaling_n_floor: int = 8192):
    """First real 6-layer attention-residual block (5 sliding + 1 full).

    num_layers=5 is a repeat of Stage 5I, num_layers=6 also executes physical 5.
    Only a SINGLE token at position zero is accepted. It is deliberately NOT
    a full 48-step model, not generated logits, and no KV-state reuse occurs.
    """
    if type(num_layers) is not int or num_layers not in (5, 6):
        raise ValueError('Stage 5J/K can only run 5 or 6 prelude layers')
    g = weights.geom
    hidden = weights.embedding(token_id)
    blocks = hidden.new_zeros((1, 1, 0, g.hidden))
    trace = []
    for physical in range(num_layers):
        if physical < 5:
            hidden, blocks, record = sliding_one_token_layer(weights, hidden, blocks, physical, 0)
        else:
            hidden, blocks, record = full_attention_one_token_layer(
                weights, hidden, blocks, layer=physical,
                log_scaling_alpha=log_scaling_alpha,
                log_scaling_n_floor=log_scaling_n_floor)
        trace.append(record)
        weights.clear()
    return hidden, blocks, trace
