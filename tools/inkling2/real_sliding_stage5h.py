"""Stage 5H: real-weight, one-token Inkling sliding decoder reference.

Executes layers 0..4 on Torch CPU/CUDA from the VERIFIED NInfer-v3 carriers.
This is FP32 numerical integration / real checkpoint smoke, NOT native NInfer
runtime, HF-bitwise-equivalent inference, or model generation.

Only sliding attention at token position zero is supported: there is no past
KV/conv history. A nonzero position or full-attention layer must fail closed.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import sys
from typing import Any

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for location in (HERE, ROOT):
    if str(location) not in sys.path:
        sys.path.insert(0, str(location))

import torch
import torch.nn.functional as F

from forward_oracle_stage5a import (
    weighted_rmsnorm, unweighted_rmsnorm, l2_clip, scaled_l2_clip,
    relative_position_bias, sink_attention, gated_output,
    static_short_convolution, block_attention_residual,
)


@dataclass(frozen=True)
class Geometry:
    hidden: int = 2304
    intermediate: int = 1152
    query_heads: int = 24
    kv_heads: int = 8
    head_dim: int = 128
    rel_dim: int = 16
    rel_extent: int = 1024
    experts: int = 128
    top_k: int = 8
    conv_kernel: int = 4
    qk_clip: float = 22.627416997969522
    rel_clip: float = 8.0
    route_scale: float = 8.0
    norm_eps: float = 1e-6

    def validate(self):
        if (self.query_heads < 1 or self.kv_heads < 1 or self.query_heads % self.kv_heads or
            self.head_dim < 1 or self.hidden < 1 or self.intermediate < 1 or
            self.query_heads * self.head_dim < 1 or self.kv_heads * self.head_dim < 1 or
            self.rel_dim < 1 or self.rel_extent < 1 or self.conv_kernel < 1 or
            self.experts < self.top_k or self.top_k < 1 or self.hidden % 16 or
            self.intermediate % 16 or not math.isfinite(self.route_scale) or
            self.route_scale <= 0):
            raise ValueError('unsupported Stage-5H geometry')

    @classmethod
    def from_hf_config(cls, data: dict):
        raw = data['text_config']
        if (raw['hidden_size'] != 2304 or raw['num_hidden_layers'] != 24 or
            raw['local_layer_ids'] != [i for i in range(24) if i % 6 != 5] or
            raw['num_attention_heads'] != 24 or raw['num_key_value_heads'] != 8 or
            raw['head_dim'] != 128 or raw['intermediate_size'] != 1152 or
            raw['n_routed_experts'] != 128 or raw['num_experts_per_tok'] != 8):
            raise ValueError('requires original Inkling-2-Mini text geometry')
        result = cls(hidden=raw['hidden_size'], intermediate=raw['intermediate_size'],
            query_heads=raw['num_attention_heads'], kv_heads=raw['num_key_value_heads'],
            head_dim=raw['head_dim'], rel_dim=raw['d_rel'],
            rel_extent=raw['sliding_window_size'], experts=raw['n_routed_experts'],
            top_k=raw['num_experts_per_tok'], conv_kernel=raw['sconv_kernel_size'],
            qk_clip=raw['qk_l2_clip_threshold'], rel_clip=raw['rel_l2_clip_threshold'],
            route_scale=raw['route_scale'], norm_eps=raw['rms_norm_eps'])
        result.validate()
        return result


def _u8_tensor(raw: bytes, shape: tuple[int, ...], device: torch.device) -> torch.Tensor:
    if len(raw) != math.prod(shape):
        raise ValueError('expert words have invalid geometry')
    return torch.frombuffer(bytearray(raw), dtype=torch.uint8).reshape(shape).to(device)


def decode_expert(raw: Any, *, rows: int, cols: int, fmt: str,
                  device: torch.device) -> torch.Tensor:
    """Exact source-order decode of one real NInfer-v3 expert to FP32.

    FP8 encoded values reinterpret their bits (never numeric uint8->fp8 cast).
    Intended for reference validation, not a production memory strategy.
    """
    if fmt == 'nvfp4':
        if rows <= 0 or cols <= 0 or cols % 16:
            raise ValueError('NVFP4 requires K16')
        code = _u8_tensor(raw.codes, (rows, cols // 2), device)
        scales = _u8_tensor(raw.scales, (rows, cols // 16), device)
        if bool(((scales & 128) != 0).any() or (scales == 127).any()):
            raise ValueError('nonfinite/negative NVFP4 scale')
        if not all(math.isfinite(v) and v > 0 for v in (raw.amax, raw.denominator)):
            raise ValueError('nonpositive NVFP4 ratio')
        unpacked = torch.empty((rows, cols), dtype=torch.long, device=device)
        unpacked[:, 0::2] = (code & 15).long()
        unpacked[:, 1::2] = (code >> 4).long()
        table = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6,
                              -0., -.5, -1, -1.5, -2, -3, -4, -6],
                             dtype=torch.float32, device=device)
        scale = scales.view(torch.float8_e4m3fn).float().repeat_interleave(16, -1)
        return table[unpacked] * scale * (raw.amax / raw.denominator)
    if fmt == 'mxfp8':
        if rows <= 0 or cols <= 0 or cols % 32:
            raise ValueError('MXFP8 requires K32')
        codes = _u8_tensor(raw.codes, (rows, cols), device)
        scales = _u8_tensor(raw.scales, (rows, cols // 32), device)
        if bool(((codes & 127) == 127).any() or (scales == 255).any()):
            raise ValueError('nonfinite MXFP8 source word')
        # UE8M0 0x00 => 2^-127; 0xff forbidden.
        scale = torch.exp2(scales.float() - 127).repeat_interleave(32, -1)
        return codes.view(torch.float8_e4m3fn).float() * scale
    raise ValueError(f'unsupported format {fmt!r}')


class TorchCarrierWeights:
    """Load per-tensor and only selected experts from the two NInfer carriers."""
    def __init__(self, reader: Any, device: str | torch.device, geometry: Geometry):
        geometry.validate()
        self.reader = reader
        self.device = torch.device(device)
        self.geom = geometry
        self._cache: dict[str, torch.Tensor] = {}

    def clear(self):
        self._cache.clear()

    def tensor(self, name: str, shape: tuple[int, ...]) -> torch.Tensor:
        if name not in self._cache:
            raw, fmt, dims = self.reader.nonexpert_tensor(name)
            if dims != shape or fmt not in ('bf16', 'fp32'):
                raise ValueError(f'{name}: tensor format/shape mismatch; expected {shape}, got {fmt}/{dims}')
            dtype = torch.bfloat16 if fmt == 'bf16' else torch.float32
            value = torch.frombuffer(bytearray(raw), dtype=dtype).reshape(dims)
            self._cache[name] = value.to(device=self.device, dtype=torch.float32)
        result = self._cache[name]
        if tuple(result.shape) != shape:
            raise ValueError(f'{name}: cache geometry mismatch')
        return result

    def conv(self, name: str, channels: int) -> torch.Tensor:
        obj = self.reader.nonexpert_tensor(name)
        shape = tuple(obj[2])
        # HF exported nn.Conv1d weights may have a singleton channel axis.
        if shape not in ((channels, self.geom.conv_kernel),
                         (channels, 1, self.geom.conv_kernel)):
            raise ValueError(f'{name}: unsupported convolution shape {shape}')
        return self.tensor(name, shape).reshape(channels, self.geom.conv_kernel)

    def embedding(self, token: int) -> torch.Tensor:
        raw = self.reader.embedding_row(token)
        if len(raw) != self.geom.hidden:
            raise ValueError('embedding row length mismatch')
        emb = torch.tensor(raw, dtype=torch.float32, device=self.device).reshape(1, 1, -1)
        return weighted_rmsnorm(emb,
            self.tensor('model.llm.embed_norm.weight', (self.geom.hidden,)), self.geom.norm_eps)

    def moe(self, hidden: torch.Tensor, layer: int) -> tuple[torch.Tensor, list[int]]:
        """One-token top-k expert reference; weights originate in both NInfer-v3 carriers."""
        g = self.geom
        if hidden.shape != (1, 1, g.hidden):
            raise ValueError('MoE requires a single token')
        prefix = f'model.llm.layers.{layer}.mlp.'
        router = self.tensor(prefix + 'gate.weight', (g.experts, g.hidden))
        bias = self.tensor(prefix + 'gate.bias', (g.experts,))
        logits = F.linear(hidden.float(), router).reshape(g.experts)
        scores = torch.sigmoid(logits)
        # Bias chooses experts, but is NEVER applied to their aggregation weights.
        ranks = scores + bias
        indices = torch.topk(ranks, k=g.top_k, sorted=True).indices
        ids = [int(i) for i in indices.cpu().tolist()]
        multipliers = scores[indices] / (scores[indices].sum() + 1e-20) * g.route_scale
        w13 = self.reader.bank(layer, 'w13', ids)
        w2 = self.reader.bank(layer, 'w2', ids)
        if len(w13) != g.top_k or len(w2) != g.top_k:
            raise ValueError('missing selected real experts')
        fmt = 'nvfp4' if layer < 20 else 'mxfp8'
        result = torch.zeros(g.hidden, device=self.device, dtype=torch.float32)
        x = hidden.float().reshape(g.hidden)
        for i in range(g.top_k):
            a = decode_expert(w13[i], rows=2 * g.intermediate, cols=g.hidden,
                              fmt=fmt, device=self.device)
            up = a @ x
            gate, val = up[:g.intermediate], up[g.intermediate:]
            activated = (4 * torch.tanh(gate / 4) * torch.sigmoid(gate) *
                         25 * torch.tanh(val / 25))
            down = decode_expert(w2[i], rows=g.hidden, cols=g.intermediate,
                                 fmt=fmt, device=self.device) @ activated
            result += multipliers[i] * down
        if not bool(torch.isfinite(result).all()):
            raise ValueError('nonfinite MoE output')
        return result.reshape(1, 1, -1), ids


def sliding_one_token_layer(weights: TorchCarrierWeights,
                            hidden: torch.Tensor, blocks: torch.Tensor,
                            layer: int, position: int = 0) -> tuple[torch.Tensor, torch.Tensor, dict]:
    """Run the original *sliding* layer 0..4 with empty history, on one token.

    Attention/convolutions/MLP follow Stage-5A's mathematics in FP32. The
    computation is weight-backed, but full-model HF parity is NOT asserted.
    """
    g = weights.geom
    if (type(layer) is not int or layer not in range(5) or
        position != 0 or hidden.shape != (1, 1, g.hidden) or
        blocks.ndim != 4 or blocks.shape[:2] != (1, 1) or blocks.shape[-1] != g.hidden):
        raise ValueError('Stage5H accepts one position-0 token and sliding layers 0..4 only')
    p = f'model.llm.layers.{layer}.'
    # At each six-layer boundary, append the *incoming* hidden state before MLP.
    if layer == 0:
        attention_input = hidden
    else:
        query = weights.tensor(f'model.llm.attn_res.attn_queries.{layer-1}', (g.hidden,))
        attention_input = block_attention_residual(blocks, hidden, query, eps=g.norm_eps)[0]
    attn_input = weighted_rmsnorm(attention_input,
        weights.tensor(p + 'attn_norm.weight', (g.hidden,)), g.norm_eps)
    hq, hk, d, rel = g.query_heads, g.kv_heads, g.head_dim, g.rel_dim
    a = p + 'attn.'
    query = F.linear(attn_input, weights.tensor(a + 'wq_du.weight', (hq*d, g.hidden)))
    query = scaled_l2_clip(query.reshape(1,1,hq,d),
        weights.tensor(a + 'q_l2_clip.scale', (hq,d)), g.qk_clip).transpose(1,2)
    key = F.linear(attn_input, weights.tensor(a + 'wk_dv.weight', (hk*d, g.hidden)))
    value = F.linear(attn_input, weights.tensor(a + 'wv_dv.weight', (hk*d, g.hidden)))
    key = static_short_convolution(key, weights.conv(a + 'k_sconv.weight', hk*d))
    value = static_short_convolution(value, weights.conv(a + 'v_sconv.weight', hk*d))
    key = l2_clip(key.reshape(1,1,hk,d), g.qk_clip).transpose(1,2)
    value = unweighted_rmsnorm(value.reshape(1,1,hk,d),g.norm_eps).transpose(1,2)
    dist = torch.zeros((1,1), dtype=torch.long, device=weights.device)
    bias = relative_position_bias(attn_input,
        weights.tensor(a + 'wr_du.weight', (hq*rel,g.hidden)),
        weights.tensor(a + 'rel_l2_clip.scale', (hq,rel)),
        weights.tensor(a + 'rel_logits_proj.proj', (rel,g.rel_extent)),
        dist, threshold=g.rel_clip)
    sinks = F.linear(attn_input, weights.tensor(a + 'attn_sink_proj.weight', (hq,g.hidden)))
    attended = sink_attention(query,key,value,
        position_bias=bias, sinks=sinks.transpose(1,2).unsqueeze(-1))
    out = gated_output(attended.reshape(1,1,hq*d), attn_input,
        weights.tensor(a + 'gate_attn.weight',(hq*d,g.hidden)),
        weights.tensor(a + 'wo_ud.weight',(g.hidden,hq*d)))
    post_attn = static_short_convolution(out, weights.conv(p+'attn_sconv.weight',g.hidden))
    if layer == 0:
        residual = post_attn
        blocks = torch.cat((blocks, hidden.unsqueeze(-2)), dim=-2)
    else:
        residual = hidden + post_attn
    query_mlp = weights.tensor(f'model.llm.attn_res.mlp_queries.{layer}', (g.hidden,))
    mlp_input = block_attention_residual(blocks, residual, query_mlp, eps=g.norm_eps)[0]
    mlp_input = weighted_rmsnorm(mlp_input,weights.tensor(p+'mlp_norm.weight',(g.hidden,)),g.norm_eps)
    routed, ids = weights.moe(mlp_input,layer)
    out = static_short_convolution(routed,weights.conv(p+'mlp_sconv.weight',g.hidden))
    result = residual + out
    if not bool(torch.isfinite(result).all()):
        raise ValueError('nonfinite decoder output')
    return result,blocks,{'physical_layer': layer,'top8_expert_ids':ids,
                          'hidden_rms':float(result.square().mean().sqrt())}


def run_prelude(weights: TorchCarrierWeights, token: int, num_layers: int):
    """Stage 5I: execute ONLY 1..5 prelude *sliding* layers on real carriers."""
    if type(num_layers) is not int or not 1 <= num_layers <= 5:
        raise ValueError('5I refuses full-attention layer 5 and all unimplemented loop layers')
    hidden = weights.embedding(token)
    blocks = torch.empty((1,1,0,weights.geom.hidden),device=weights.device,dtype=torch.float32)
    trace = []
    for layer in range(num_layers):
        hidden,blocks,record = sliding_one_token_layer(weights,hidden,blocks,layer,0)
        trace.append(record)
        weights.clear()
    return hidden, blocks, trace
