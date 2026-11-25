"""Inkling 2 Mini Stage5Q: experimental stateful one-token-at-a-time Torch decoder.

Original weights are read from the two verified NInfer-v3 carriers via
TorchCarrierWeights. Runtime state comprises 48 independent logical KV slots
and 4 independent convolution state paths per slot. No HF parity is claimed.

This path is diagnostic only: not the registered native NInfer execution engine.
It deliberately supports batch=1 and sequential positions, refusing speculative
skips, repeated slots and out-of-order input. There is no persistent checkpoint.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any

import torch
import torch.nn.functional as F

from forward_oracle_stage5a import (
    weighted_rmsnorm, unweighted_rmsnorm, l2_clip, scaled_l2_clip,
    relative_position_bias, sink_attention, gated_output,
    block_attention_residual, cross_pass_messages, controller_message_gate,
)
from real_core_stage5lm import _cross_queries, _controller
from real_postlude_stage5no import finish_hidden

_FULL_LAYERS = frozenset((5, 11, 17, 23))
_EXPECTED_SLOTS = frozenset(range(48))


def slot_for(physical: int, core_pass: int = 0) -> int:
    if type(physical) is not int or physical not in range(24):
        raise ValueError('physical layer must be in 0..23')
    if physical not in range(6, 18):
        if core_pass != 0:
            raise ValueError('prelude/postlude have no repeated cache slots')
        return physical
    if core_pass not in (0, 1, 2):
        raise ValueError('core pass must be 0, 1 or 2')
    return physical if core_pass == 0 else 24 + (core_pass - 1) * 12 + physical - 6


@dataclass
class IncrementalCache48:
    """One sequence, GPU resident; no silent fallback to uncached attention."""

    kernel: int = 4
    next_position: int = 0
    conv: dict[tuple[int, int], torch.Tensor] = field(default_factory=dict)
    kv: dict[int, tuple[torch.Tensor, torch.Tensor]] = field(default_factory=dict)
    key_positions: dict[int, list[int]] = field(default_factory=dict)
    _active_position: int | None = None
    _conv_visited: set[tuple[int, int]] = field(default_factory=set)
    _kv_visited: set[int] = field(default_factory=set)
    invalid: bool = False

    def begin(self, position: int) -> None:
        if (self.invalid or self._active_position is not None or
                type(position) is not int or position != self.next_position or
                self.kernel < 1):
            raise ValueError('cache step must be sequential, unique and not invalidated')
        self._active_position = position
        self._conv_visited.clear()
        self._kv_visited.clear()

    def _check_slot(self, slot: int) -> None:
        if self.invalid or self._active_position is None or type(slot) is not int or not 0 <= slot < 48:
            raise ValueError('cache access outside an active step or 48-slot range')

    def _window(self, slot: int, path: int, value: torch.Tensor) -> torch.Tensor:
        self._check_slot(slot)
        if type(path) is not int or not 0 <= path < 4 or (slot, path) in self._conv_visited:
            raise ValueError('convolution path must be used once per logical slot and token')
        if value.ndim != 3 or value.shape[:2] != (1, 1) or not value.is_floating_point():
            raise ValueError('cache convolution token must be [1,1,C] float')
        key = (slot, path)
        history = self.conv.get(key)
        if history is None:
            history = value.new_zeros((1, self.kernel - 1, value.shape[-1]), dtype=torch.float32)
        elif history.shape != (1, self.kernel - 1, value.shape[-1]):
            raise ValueError('cached convolution channel geometry changed')
        window = torch.cat((history, value.float()), dim=1)
        self.conv[key] = window[:, 1:, :].clone()
        self._conv_visited.add(key)
        return window

    def static_conv(self, slot: int, path: int, x: torch.Tensor, taps: torch.Tensor) -> torch.Tensor:
        """HF static conv: oldest to newest tap and residual / sqrt(K+1)."""
        if taps.ndim != 2 or taps.shape != (x.shape[-1], self.kernel):
            raise ValueError('static taps must be [C,K]')
        window = self._window(slot, path, x)
        y = (window.transpose(1, 2) * taps.float()[None, :, :]).sum(-1)
        return ((y[:, None] + x.float()) / math.sqrt(self.kernel + 1)).to(x.dtype)

    def dynamic_conv(self, slot: int, path: int, x: torch.Tensor,
                     dynamic_input: torch.Tensor, taps: torch.Tensor,
                     projection: torch.Tensor, *, heads: int) -> torch.Tensor:
        """HF full-attention K/V conv: lag zero is current, /sqrt(2K+1)."""
        if (x.ndim != 3 or x.shape[:2] != (1, 1) or
                dynamic_input.ndim != 3 or dynamic_input.shape[:2] != (1, 1) or
                type(heads) is not int or heads < 1 or x.shape[-1] % heads or
                taps.shape != (self.kernel, x.shape[-1]) or
                projection.shape != (heads * self.kernel, dynamic_input.shape[-1])):
            raise ValueError('dynamic convolution geometry mismatch')
        # Compute the learned taps for the *current* token before history update.
        delta = F.linear(dynamic_input.float(), projection.float()).reshape(heads, 1, self.kernel)
        all_taps = taps.float().T.reshape(heads, x.shape[-1] // heads, self.kernel) + delta
        window = self._window(slot, path, x)
        lagged = window.flip(dims=[1]).squeeze(0).T.reshape(heads, x.shape[-1] // heads, self.kernel)
        conv = (lagged * all_taps).sum(-1).reshape(1, 1, x.shape[-1])
        return ((conv + x.float()) / math.sqrt(2 * self.kernel + 1)).to(x.dtype)

    def attention(self, slot: int, query: torch.Tensor, key: torch.Tensor,
                  value: torch.Tensor, position: int, *, sliding_window: int | None,
                  bias_from_distances, sinks: torch.Tensor) -> torch.Tensor:
        self._check_slot(slot)
        if slot in self._kv_visited or position != self._active_position:
            raise ValueError('duplicate KV update or wrong token position')
        if (query.ndim != 4 or key.ndim != 4 or value.shape != key.shape or
                query.shape[0] != 1 or query.shape[2] != 1 or key.shape[2] != 1):
            raise ValueError('KV update requires single-token batch=1 Q/K/V')
        if sliding_window is not None and sliding_window < 1:
            raise ValueError('sliding window must be >=1')
        previous = self.kv.get(slot)
        positions = self.key_positions.get(slot, [])
        if previous is None:
            keys, vals = key.float(), value.float()
        else:
            kold, vold = previous
            if kold.shape[:2] != key.shape[:2] or kold.shape[-1] != key.shape[-1]:
                raise ValueError('cached KV geometry changed')
            keys = torch.cat((kold, key.float()), dim=2)
            vals = torch.cat((vold, value.float()), dim=2)
        positions = [*positions, position]
        if len(positions) != keys.shape[2] or positions[-1] != position:
            raise ValueError('cached KV positions are invalid')
        if sliding_window is not None and len(positions) > sliding_window:
            keys = keys[:, :, -sliding_window:].contiguous()
            vals = vals[:, :, -sliding_window:].contiguous()
            positions = positions[-sliding_window:]
        distances = torch.tensor([[position - p for p in positions]], device=query.device,
                                 dtype=torch.long).unsqueeze(0)
        bias = bias_from_distances(distances)
        output = sink_attention(query, keys, vals, position_bias=bias, sinks=sinks)
        self.kv[slot] = (keys, vals)
        self.key_positions[slot] = positions
        self._kv_visited.add(slot)
        return output

    def end(self) -> None:
        if self._active_position is None or self.invalid:
            raise ValueError('no active cache step')
        if (self._kv_visited != _EXPECTED_SLOTS or
                self._conv_visited != {(slot, path) for slot in range(48) for path in range(4)}):
            raise ValueError('incomplete 48-slot KV/conv step: refusing to advance')
        self.next_position += 1
        self._active_position = None
        self._conv_visited.clear()
        self._kv_visited.clear()

    def abort(self) -> None:
        """A failed step cannot be resumed with partially mutated caches."""
        self.invalid = True
        self._active_position = None


def _layer(weights: Any, cache: IncrementalCache48, hidden: torch.Tensor,
           blocks: torch.Tensor, physical: int, *, core_pass: int,
           message: torch.Tensor | None, gate_rank: int,
           log_scaling_alpha: float, log_scaling_n_floor: int) -> tuple[torch.Tensor, torch.Tensor, dict]:
    g = weights.geom
    position = cache._active_position
    if (position is None or type(physical) is not int or not 0 <= physical < 24 or
            hidden.shape != (1, 1, g.hidden) or blocks.shape !=
            (1, 1, (physical + 5) // 6, g.hidden)):
        raise ValueError('invalid physical layer/active cache/block geometry')
    if (message is None) != (core_pass == 0) or (message is not None and message.shape != (1, 1, 4, g.hidden)):
        raise ValueError('cross-pass messages required only for passes 1/2')
    slot = slot_for(physical, core_pass)
    is_full = physical in _FULL_LAYERS
    p = f'model.llm.layers.{physical}.'
    a = p + 'attn.'
    hq, hk, head_dim, rel = g.query_heads, g.kv_heads, g.head_dim, g.rel_dim
    if physical == 0:
        q_hidden = k_hidden = v_hidden = hidden
    else:
        queries = weights.tensor(f'model.llm.attn_res.attn_queries.{physical - 1}',
                                 (3, g.hidden) if is_full else (g.hidden,))
        prepared = block_attention_residual(blocks, hidden, queries, eps=g.norm_eps)
        if is_full:
            q_hidden, k_hidden, v_hidden = prepared
        else:
            q_hidden = k_hidden = v_hidden = prepared[0]
    if message is not None:
        qmsg, kmsg, vmsg, mlpmsg = message.unbind(-2)
        q_hidden = _controller(weights, physical, 'attn', q_hidden, qmsg, gate_rank)
        k_hidden = _controller(weights, physical, 'k', k_hidden, kmsg, gate_rank)
        v_hidden = _controller(weights, physical, 'v', v_hidden, vmsg, gate_rank)
    if is_full:
        q_hidden = weighted_rmsnorm(q_hidden, weights.tensor(p + 'q_input_norm.weight', (g.hidden,)), g.norm_eps)
        k_hidden = weighted_rmsnorm(k_hidden, weights.tensor(p + 'k_input_norm.weight', (g.hidden,)), g.norm_eps)
        v_hidden = weighted_rmsnorm(v_hidden, weights.tensor(p + 'v_input_norm.weight', (g.hidden,)), g.norm_eps)
    else:
        norm = weights.tensor(p + 'attn_norm.weight', (g.hidden,))
        q_hidden = weighted_rmsnorm(q_hidden, norm, g.norm_eps)
        # For sliding layers, HF uses independent controllers then the same norm.
        k_hidden = weighted_rmsnorm(k_hidden, norm, g.norm_eps) if message is not None else q_hidden
        v_hidden = weighted_rmsnorm(v_hidden, norm, g.norm_eps) if message is not None else q_hidden
    pos_tensor = torch.tensor([[position]], dtype=torch.long, device=weights.device)
    clip_args = dict(positions=pos_tensor, log_scaling_alpha=log_scaling_alpha,
                     log_scaling_n_floor=log_scaling_n_floor) if is_full else {}
    query = F.linear(q_hidden, weights.tensor(a + 'wq_du.weight', (hq * head_dim, g.hidden)))
    query = scaled_l2_clip(query.reshape(1, 1, hq, head_dim),
                           weights.tensor(a + 'q_l2_clip.scale', (hq, head_dim)),
                           g.qk_clip, **clip_args).transpose(1, 2)
    key = F.linear(k_hidden, weights.tensor(a + 'wk_dv.weight', (hk * head_dim, g.hidden)))
    value = F.linear(v_hidden, weights.tensor(a + 'wv_dv.weight', (hk * head_dim, g.hidden)))
    if is_full:
        def dyn(x, h, name, path):
            taps = weights.tensor(a + name + '.static_w', (cache.kernel, hk * head_dim))
            proj = weights.tensor(a + name + '.weight_proj', (hk * cache.kernel, g.hidden))
            return cache.dynamic_conv(slot, path, x, h, taps, proj, heads=hk)
        key = dyn(key, k_hidden, 'k_sconv', 0)
        value = dyn(value, v_hidden, 'v_sconv', 1)
    else:
        key = cache.static_conv(slot, 0, key, weights.conv(a + 'k_sconv.weight', hk * head_dim))
        value = cache.static_conv(slot, 1, value, weights.conv(a + 'v_sconv.weight', hk * head_dim))
    key = l2_clip(key.reshape(1, 1, hk, head_dim), g.qk_clip).transpose(1, 2)
    value = unweighted_rmsnorm(value.reshape(1, 1, hk, head_dim), g.norm_eps).transpose(1, 2)

    def make_bias(distances):
        return relative_position_bias(q_hidden,
            weights.tensor(a + 'wr_du.weight', (hq * rel, g.hidden)),
            weights.tensor(a + 'rel_l2_clip.scale', (hq, rel)),
            weights.tensor(a + 'rel_logits_proj.proj', (rel, g.rel_extent)),
            distances, threshold=g.rel_clip, **clip_args)

    sinks = F.linear(q_hidden, weights.tensor(a + 'attn_sink_proj.weight', (hq, g.hidden)))
    attended = cache.attention(slot, query, key, value, position,
        sliding_window=(g.rel_extent if not is_full else None),
        bias_from_distances=make_bias, sinks=sinks.transpose(1, 2).unsqueeze(-1))
    out = gated_output(attended.reshape(1, 1, hq * head_dim), q_hidden,
        weights.tensor(a + 'gate_attn.weight', (hq * head_dim, g.hidden)),
        weights.tensor(a + 'wo_ud.weight', (g.hidden, hq * head_dim)))
    attn_out = cache.static_conv(slot, 2, out, weights.conv(p + 'attn_sconv.weight', g.hidden))
    if physical % 6 == 0:
        residual = attn_out.float()
        blocks = torch.cat((blocks, hidden.unsqueeze(-2)), dim=-2)
    else:
        residual = hidden.float() + attn_out.float()
    mlp_query = weights.tensor(f'model.llm.attn_res.mlp_queries.{physical}', (g.hidden,))
    mlp_input = block_attention_residual(blocks, residual, mlp_query, eps=g.norm_eps)[0]
    if message is not None:
        mlp_input = _controller(weights, physical, 'mlp', mlp_input, mlpmsg, gate_rank)
    mlp_input = weighted_rmsnorm(mlp_input,
                weights.tensor(p + 'mlp_norm.weight', (g.hidden,)), g.norm_eps)
    routed, ids = weights.moe(mlp_input, physical)
    mlp_out = cache.static_conv(slot, 3, routed, weights.conv(p + 'mlp_sconv.weight', g.hidden))
    result = residual + mlp_out.float()
    if not bool(torch.isfinite(result).all()):
        raise ValueError(f'nonfinite hidden at layer {physical}, cache slot {slot}')
    return result, blocks, {
        'physical_layer': physical, 'core_pass': core_pass if 6 <= physical < 18 else None,
        'cache_slot': slot, 'position': position,
        'attention_type': 'full' if is_full else 'sliding',
        'top8_expert_ids': ids, 'hidden_rms': float(result.square().mean().sqrt()),
    }


def forward_one_token(weights: Any, cache: IncrementalCache48, token_id: int, *,
                      gate_rank: int = 64, log_scaling_alpha: float = 0.1,
                      log_scaling_n_floor: int = 8192) -> tuple[torch.Tensor, list[dict]]:
    """Compute complete 48 visits at cache.next_position, in strict causal order."""
    if type(token_id) is not int or token_id < 0 or token_id >= 201024:
        raise ValueError('input token outside padded embedding vocabulary')
    if cache.invalid or cache._active_position is not None:
        raise ValueError('cache is invalid or already in a step')
    pos = cache.next_position
    cache.begin(pos)
    try:
        hidden = weights.embedding(token_id)
        blocks = hidden.new_zeros((1, 1, 0, weights.geom.hidden))
        trace = []
        for physical in range(6):
            hidden, blocks, record = _layer(weights, cache, hidden, blocks,
                physical, core_pass=0, message=None, gate_rank=gate_rank,
                log_scaling_alpha=log_scaling_alpha, log_scaling_n_floor=log_scaling_n_floor)
            trace.append(record)
            weights.clear()
        pre_hidden, pre_blocks = hidden, blocks
        if pre_blocks.shape[-2] != 1:
            raise ValueError('invalid prelude residual block count')
        queries = _cross_queries(weights)
        for core_pass in range(3):
            if core_pass:
                prior_blocks = blocks[..., pre_blocks.shape[-2] + 1:, :]
                if prior_blocks.shape[-2] != 1:
                    raise ValueError('prior pass has wrong residual block count')
                prior_hidden = hidden
                hidden, blocks = pre_hidden, pre_blocks
            for physical in range(6, 18):
                message = (cross_pass_messages(prior_blocks, prior_hidden, queries,
                    physical - 6, eps=weights.geom.norm_eps) if core_pass else None)
                hidden, blocks, record = _layer(weights, cache, hidden, blocks,
                    physical, core_pass=core_pass, message=message, gate_rank=gate_rank,
                    log_scaling_alpha=log_scaling_alpha, log_scaling_n_floor=log_scaling_n_floor)
                trace.append(record)
                weights.clear()
        for physical in range(18, 24):
            hidden, blocks, record = _layer(weights, cache, hidden, blocks,
                physical, core_pass=0, message=None, gate_rank=gate_rank,
                log_scaling_alpha=log_scaling_alpha, log_scaling_n_floor=log_scaling_n_floor)
            trace.append(record)
            weights.clear()
        hidden = finish_hidden(weights, hidden, blocks)
        if ([r['cache_slot'] for r in trace] !=
            list(range(24))[:18] + list(range(24, 48)) + list(range(18, 24)) or
            len(trace) != 48):
            raise ValueError('48-slot execution order mismatch')
        cache.end()
        return hidden, trace
    except BaseException:
        cache.abort()
        raise
