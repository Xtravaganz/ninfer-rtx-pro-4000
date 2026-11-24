"""Inkling 2 Mini Stage 5A: eager Torch mathematical oracles for text primitives.

Reference-only, tensor *inputs* are already projected/loaded. Not a decoder,
checkpoint reader, native CUDA kernel or NInfer runtime implementation.

Derivation: Hugging Face Transformers `modeling_inkling2_loop.py` (2026-11-24),
Inkling2Loop* attention / short convolution / residual / message components.
Full tensor paths execute on CPU and CUDA with Torch; no model weights required.
"""
from __future__ import annotations

import math
from typing import NamedTuple

import torch
import torch.nn.functional as F


def _positive(value: float, name: str):
    if not math.isfinite(float(value)) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")


def unweighted_rmsnorm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    _positive(eps, "eps")
    y = x.float()
    return (y * torch.rsqrt(y.square().mean(-1, keepdim=True) + eps)).to(x.dtype)


def weighted_rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    if weight.ndim != 1 or weight.shape[0] != x.shape[-1]:
        raise ValueError("RMSNorm weight must match last hidden dimension")
    return (unweighted_rmsnorm(x, eps).float() * weight.float()).to(x.dtype)


def l2_clip(x: torch.Tensor, threshold: float) -> torch.Tensor:
    _positive(threshold, "threshold")
    y = x.float()
    norm = torch.linalg.vector_norm(y, dim=-1, keepdim=True)
    return (y * (threshold / (norm + 1e-6)).clamp(max=1)).to(x.dtype)


def scaled_l2_clip(x: torch.Tensor, scales: torch.Tensor, threshold: float,
                   *, positions: torch.Tensor | None = None,
                   log_scaling_alpha: float = 0.1, log_scaling_n_floor: int = 8192) -> torch.Tensor:
    """Per-head learnable scale, clipping, then *optional* full-attention log scale."""
    if x.ndim != 4 or scales.shape != x.shape[-2:]:
        raise ValueError("x must be [B,T,H,D] and scales [H,D]")
    _positive(threshold, "threshold")
    scaled = x.float() * scales.float()
    norm = torch.linalg.vector_norm(scaled, dim=-1, keepdim=True)
    # HF scales/clips in FP32, only casts to input dtype *after* clipping.
    y = (scaled * (threshold / (norm + 1e-6)).clamp(max=1)).to(x.dtype)
    if positions is None:
        return y
    if positions.shape != x.shape[:2] or log_scaling_n_floor <= 0:
        raise ValueError("position ids must be [B,T], and n_floor positive")
    if not math.isfinite(log_scaling_alpha) or log_scaling_alpha < 0:
        raise ValueError("log scaling alpha must be finite and nonnegative")
    t = 1 + log_scaling_alpha * torch.log(((positions.float() + 1) / log_scaling_n_floor).clamp(min=1))
    return (y.float() * t[..., None, None]).to(x.dtype)


def relative_position_bias(
    hidden: torch.Tensor,
    linear_weight: torch.Tensor,
    head_scales: torch.Tensor,
    profiles: torch.Tensor,
    relative_distances: torch.Tensor,
    *,
    threshold: float = 8.0,
    positions: torch.Tensor | None = None,
    log_scaling_alpha: float = 0.1,
    log_scaling_n_floor: int = 8192,
) -> torch.Tensor:
    """Source-conditioned [B,H,T,K] bias; out-of-range backward distances => 0.

    HF `relative_position_bias`: relative states from linear(hidden), head-wise
    scale and L2 clipping; profiles [d_rel,rel_extent] gathered by distance.
    """
    if hidden.ndim != 3:
        raise ValueError("hidden must be [B,T,D]")
    b, t, d = hidden.shape
    if head_scales.ndim != 2:
        raise ValueError("head scales must be [H,d_rel]")
    heads, rel_dim = head_scales.shape
    if (linear_weight.shape != (heads * rel_dim, d) or
        profiles.ndim != 2 or profiles.shape[0] != rel_dim or profiles.shape[1] < 1):
        raise ValueError("invalid relative bias projection geometry")
    if relative_distances.ndim == 2:
        relative_distances = relative_distances[None, :, :].expand(b, -1, -1)
    if relative_distances.ndim != 3 or relative_distances.shape[:2] != (b, t):
        raise ValueError("relative distances must be [T,K] or [B,T,K]")
    states = F.linear(hidden.float(), linear_weight.float()).reshape(b, t, heads, rel_dim)
    states = scaled_l2_clip(states, head_scales, threshold, positions=positions,
                            log_scaling_alpha=log_scaling_alpha,
                            log_scaling_n_floor=log_scaling_n_floor).float()
    logits = (states @ profiles.float()).transpose(1, 2)  # [B,H,T,R]
    extent = profiles.shape[1]
    indices = relative_distances.clamp(0, extent - 1).long()[:, None, :, :].expand(b, heads, -1, -1)
    gathered = logits.gather(-1, indices)
    valid = (relative_distances >= 0) & (relative_distances < extent)
    return gathered.masked_fill(~valid[:, None], 0)


def causal_additive_mask(query_positions: torch.Tensor, key_positions: torch.Tensor,
                         *, sliding_window: int | None = None) -> torch.Tensor:
    """[B,1,Q,K] mask; sliding means current plus previous (window-1) keys."""
    if query_positions.ndim != 2 or key_positions.ndim != 2 or query_positions.shape[0] != key_positions.shape[0]:
        raise ValueError("positions must be [B,Q] and [B,K]")
    if sliding_window is not None and sliding_window < 1:
        raise ValueError("sliding_window must be >=1")
    delta = query_positions[..., :, None] - key_positions[..., None, :]
    allowed = delta >= 0
    if sliding_window is not None:
        allowed &= delta < sliding_window
    mask = torch.zeros_like(delta, dtype=torch.float32)
    return mask.masked_fill(~allowed, float("-inf"))[:, None, :, :]


def sink_attention(
    query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
    *, attention_mask: torch.Tensor | None = None,
    position_bias: torch.Tensor | None = None,
    sinks: torch.Tensor | None = None,
) -> torch.Tensor:
    """GQA attention, extra sink logit with zero-value, FP32 softmax.

    q: [B,Hq,Q,D], k/v: [B,Hkv,K,D], sinks [B,Hq,Q,1].
    Returns [B,Q,Hq,D].  All masked keys are allowed ONLY with a sink.
    """
    if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
        raise ValueError("query,key,value must be rank 4")
    b, hq, q, d = query.shape
    bk, hk, k, dk = key.shape
    if b != bk or d != dk or value.shape != key.shape or hq % hk:
        raise ValueError("GQA shape or head-group mismatch")
    key_repeated = key.repeat_interleave(hq // hk, dim=1).float()
    value_repeated = value.repeat_interleave(hq // hk, dim=1).float()
    logits = query.float() @ key_repeated.transpose(-1, -2) * (d ** -0.5)
    for name, bias in (("position_bias", position_bias), ("attention_mask", attention_mask)):
        if bias is not None:
            try:
                logits = logits + bias.float()
            except RuntimeError as error:
                raise ValueError(f"{name} shape is incompatible") from error
    if sinks is not None:
        if sinks.shape != (b, hq, q, 1):
            raise ValueError("sinks must be [B,Hq,Q,1]")
        probabilities = torch.cat([logits, sinks.float()], dim=-1).softmax(-1)[..., :-1]
    else:
        if bool(torch.isneginf(logits).all(-1).any()):
            raise ValueError("all keys masked without sink: undefined softmax")
        probabilities = logits.softmax(-1)
    return (probabilities @ value_repeated).transpose(1, 2).contiguous().to(query.dtype)


def gated_output(attended: torch.Tensor, input_hidden: torch.Tensor,
                 gate_weight: torch.Tensor, output_weight: torch.Tensor) -> torch.Tensor:
    """Sigmoid(input @ W_gate) * attention_output, followed by W_output."""
    if attended.shape != input_hidden.shape[:-1] + (gate_weight.shape[0],):
        raise ValueError("gated attention output shape mismatch")
    gates = torch.sigmoid(F.linear(input_hidden.float(), gate_weight.float()))
    return F.linear(attended.float() * gates, output_weight.float()).to(attended.dtype)


def _conv_specs(hidden: torch.Tensor, weight: torch.Tensor, mask: torch.Tensor | None):
    if hidden.ndim != 3 or weight.ndim != 2 or weight.shape[0] != hidden.shape[-1] or weight.shape[1] < 1:
        raise ValueError("static conv: hidden [B,T,C], weights [C,kernel]")
    if mask is not None and mask.shape != hidden.shape[:2]:
        raise ValueError("padding mask must be [B,T]")


def static_short_convolution(hidden: torch.Tensor, weights: torch.Tensor,
                             *, padding_mask: torch.Tensor | None = None) -> torch.Tensor:
    """Depthwise causal conv + unmasked residual, divided by sqrt(kernel+1).

    HF `Inkling2LoopShortConvolution` uses float32 hidden prior to convolution;
    the weight is [C,K] with oldest tap at index 0, current tap at K-1.
    """
    _conv_specs(hidden, weights, padding_mask)
    k = weights.shape[1]
    x = hidden.float()
    x_masked = x if padding_mask is None else x * padding_mask[..., None]
    y = F.conv1d(x_masked.transpose(1, 2), weights.float().unsqueeze(1),
                 groups=x.shape[-1], padding=k - 1)[..., :x.shape[1]]
    return ((y.transpose(1, 2) + x) / math.sqrt(k + 1)).to(hidden.dtype)


def static_short_convolution_step(hidden_token: torch.Tensor, weights: torch.Tensor,
                                  history: torch.Tensor,
                                  *, padding_mask: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """One-token equivalent to prefill; history [B,K-1,C] of *masked* inputs."""
    if hidden_token.ndim != 3 or hidden_token.shape[1] != 1:
        raise ValueError("step requires hidden [B,1,C]")
    _conv_specs(hidden_token, weights, padding_mask)
    k = weights.shape[1]
    if history.shape != (hidden_token.shape[0], k - 1, hidden_token.shape[2]):
        raise ValueError("static conv history must be [B,K-1,C]")
    current = hidden_token.float() if padding_mask is None else hidden_token.float() * padding_mask[..., None]
    window = torch.cat([history.float(), current], dim=1)
    y = (window.transpose(1, 2) * weights.float()[None, :, :]).sum(-1)
    out = (y[:, None, :] + hidden_token.float()) / math.sqrt(k + 1)
    return out.to(hidden_token.dtype), window[:, 1:, :].clone()


def dynamic_short_convolution(hidden: torch.Tensor, dynamic_input: torch.Tensor,
                              static_weights: torch.Tensor, projection: torch.Tensor,
                              *, num_heads: int, padding_mask: torch.Tensor | None = None,
                              sequence_ids: torch.Tensor | None = None) -> torch.Tensor:
    """Full-attention key/value dynamic conv, lag 0=current, /sqrt(2K+1).

    hidden[B,T,C], static_weights[K,C], projection[H*K,D_input],
    dynamic_input[B,T,D_input]. Optional per-token sequence ids mask boundaries.
    """
    if hidden.ndim != 3 or num_heads < 1:
        raise ValueError("hidden must be [B,T,C] and heads >=1")
    b, t, c = hidden.shape
    if (c % num_heads or static_weights.ndim != 2 or static_weights.shape[1] != c or
        projection.shape != (num_heads * static_weights.shape[0], dynamic_input.shape[-1]) or
        dynamic_input.shape[:2] != (b, t)):
        raise ValueError("invalid dynamic convolution geometry")
    k = static_weights.shape[0]
    if k < 1 or (padding_mask is not None and padding_mask.shape != (b, t)):
        raise ValueError("invalid kernel or mask")
    if sequence_ids is not None and sequence_ids.shape != (b, t):
        raise ValueError("sequence_ids must be [B,T]")
    delta = F.linear(dynamic_input.float(), projection.float()).reshape(b, t, num_heads, 1, k)
    weights = static_weights.float().T.reshape(num_heads, c // num_heads, k)[None, None] + delta
    residual = hidden.float()
    context = residual if padding_mask is None else residual * padding_mask[..., None]
    # Window lag ordering [current, one step ago, ...]: time dimension is -2.
    padded = F.pad(context.transpose(1, 2), (k - 1, 0))
    windows = padded.unfold(-1, k, 1).flip(-1).transpose(1, 2).reshape(b, t, num_heads, c // num_heads, k)
    if sequence_ids is not None:
        padded_ids = torch.cat([sequence_ids[:, :1].expand(-1, k - 1), sequence_ids], dim=1)
        window_ids = padded_ids.unfold(-1, k, 1).flip(-1)
        windows = windows * (window_ids == sequence_ids[..., None])[:, :, None, None, :]
    result = (windows * weights).sum(-1).reshape(b, t, c)
    return ((result + residual) / math.sqrt(2 * k + 1)).to(hidden.dtype)


def block_attention_residual(block_states: torch.Tensor, partial: torch.Tensor,
                              queries: torch.Tensor, *, eps: float = 1e-6) -> tuple[torch.Tensor, ...]:
    """Attention over unweighted RMS(block history + current), Q=1 or Q=3."""
    if partial.ndim != 3 or block_states.shape[:2] != partial.shape[:2] or block_states.ndim != 4:
        raise ValueError("partial [B,T,D], blocks [B,T,S,D]")
    if block_states.shape[-1] != partial.shape[-1]:
        raise ValueError("block hidden dimension mismatch")
    d = partial.shape[-1]
    if queries.ndim == 1:
        queries = queries[None]
    if queries.ndim != 2 or queries.shape[1] != d:
        raise ValueError("queries must be [D] or [Q,D]")
    states = unweighted_rmsnorm(torch.cat([block_states, partial.unsqueeze(-2)], -2).float(), eps)
    scores = (queries.float() @ states.transpose(-1, -2)) * (d ** -0.5)
    results = scores.softmax(-1) @ states
    return tuple(x.to(partial.dtype) for x in results.unbind(-2))


def cross_pass_messages(previous_blocks: torch.Tensor, previous_hidden: torch.Tensor,
                        queries: torch.Tensor, loop_layer_idx: int,
                        *, eps: float = 1e-6) -> torch.Tensor:
    """Cross-pass messages [B,T,4,D] from previous pass's *block* states."""
    if queries.ndim != 3 or queries.shape[1] != 4 or not 0 <= loop_layer_idx < queries.shape[0]:
        raise ValueError("message queries must be [core_layers,4,D] and index valid")
    return torch.stack(block_attention_residual(
        previous_blocks, previous_hidden, queries[loop_layer_idx], eps=eps), dim=-2)


def controller_message_gate(hidden: torch.Tensor, message: torch.Tensor,
                            down: torch.Tensor, forget: torch.Tensor, gain: torch.Tensor,
                            *, eps: float = 1e-6) -> torch.Tensor:
    """Controller (1-tanh(forget)) * h + tanh(gain) * message (NOT sigmoid)."""
    if hidden.shape != message.shape or hidden.ndim < 2:
        raise ValueError("controller inputs must be matching tensors")
    d = hidden.shape[-1]
    if down.ndim != 2 or down.shape[1] != d * 2:
        raise ValueError("down shape must be [rank,2*hidden]")
    rank = down.shape[0]
    if forget.shape != (d, rank) or gain.shape != (d, rank):
        raise ValueError("forget/gain shape must be [hidden,rank]")
    h, m = hidden.float(), message.float()
    stacked = torch.cat((unweighted_rmsnorm(h, eps), unweighted_rmsnorm(m, eps)), dim=-1)
    compressed = F.linear(stacked.to(down.dtype), down).float()
    f = torch.tanh(F.linear(compressed, forget.float()))
    g = torch.tanh(F.linear(compressed, gain.float()))
    return (1 - f) * h + g * m


class BlockRecord(NamedTuple):
    layer: int
    pass_id: int
    cache_slot: int
    state_before: torch.Tensor


def schedule_trace(schedule: list[dict], initial_hidden: torch.Tensor) -> tuple[BlockRecord, ...]:
    """A *trace of binding scheduling*, not 48 actual decoder-layer computations.

    Produces identities per logical visit to test that physical layers are reused,
    loop passes have disjoint caches, and later passes reset to prelude state.
    """
    if initial_hidden.ndim != 3 or len(schedule) != 48:
        raise ValueError("requires [B,T,D] hidden and 48-step schedule")
    slots: set[int] = set()
    traces = []
    for ordinal, step in enumerate(schedule):
        if step["ordinal"] != ordinal or step["cache_slot"] in slots:
            raise ValueError("nonunique cache slot or incorrect visit ordinal")
        slots.add(step["cache_slot"])
        traces.append(BlockRecord(step["physical_layer"], step["core_pass"], step["cache_slot"], initial_hidden))
    if slots != set(range(48)):
        raise ValueError("invalid cache slot coverage")
    return tuple(traces)


def execute_loop_callbacks(initial_hidden: torch.Tensor, schedule: list[dict],
                           layer_forward, message_forward):
    """Stage 5A *orchestration oracle* matching the physical 6+3*12+6 loop.

    `layer_forward(step, hidden, block_states, messages)` supplies the missing
    decoder operator (there is deliberately NO real decoder here); it must
    return [B,T,D]. `message_forward(prev_core_blocks, prev_hidden, core_index)`
    supplies messages [B,T,4,D]. Each visit owns its unique cache slot as in
    Stage 4B. Block records store the HIDDEN BEFORE a block-boundary layer.

    Returns (last_hidden, block_states, visit_trace). The callback may be a
    synthetic transformation; this is NOT a runnable model or a GPU backend.
    """
    if initial_hidden.ndim != 3 or len(schedule) != 48:
        raise ValueError("expected [B,T,D] hidden and 48 logical visits")
    expected_stages = ([('prelude',i,0) for i in range(6)] +
        [('core',i,p) for p in range(3) for i in range(6,18)] +
        [('postlude',i,0) for i in range(18,24)])
    if any((s.get('stage'),s.get('physical_layer'),s.get('core_pass')) != expected_stages[i]
           or s.get('ordinal') != i for i,s in enumerate(schedule)):
        raise ValueError("unsupported Inkling 6 + 3x12 + 6 visit order")
    if sorted(s.get('cache_slot') for s in schedule) != list(range(48)):
        raise ValueError("cache slots must be a complete one-to-one mapping")

    hidden = initial_hidden
    blocks = hidden.new_zeros((*hidden.shape[:2],0,hidden.shape[-1]))
    pre_hidden = pre_blocks = prev_hidden = prev_blocks = None
    pre_blocks_count = None
    trace = []
    for s in schedule:
        layer = s['physical_layer']
        pass_id = s['core_pass']
        if s['stage'] == 'core' and layer == 6:
            if pass_id == 0:
                pre_hidden, pre_blocks = hidden, blocks
                pre_blocks_count = blocks.shape[-2]
            else:
                prev_hidden = hidden
                # +1 skips previous pass's first core boundary state at layer 6.
                prev_blocks = blocks[...,pre_blocks_count+1:,:]
                hidden, blocks = pre_hidden, pre_blocks
        if s['stage'] == 'core' and pass_id > 0:
            msg = message_forward(prev_blocks, prev_hidden, layer-6)
            if msg.shape != (*hidden.shape[:2],4,hidden.shape[-1]):
                raise ValueError("message callback must return [B,T,4,D]")
        else:
            msg = None
        before = hidden
        updated = layer_forward(s, hidden, blocks, msg)
        if updated.shape != hidden.shape:
            raise ValueError("layer callback must preserve hidden shape")
        if layer % 6 == 0:
            blocks = torch.cat((blocks, before.unsqueeze(-2)),dim=-2)
        trace.append(BlockRecord(layer,pass_id,s['cache_slot'],before))
        hidden = updated
    return hidden, blocks, tuple(trace)
