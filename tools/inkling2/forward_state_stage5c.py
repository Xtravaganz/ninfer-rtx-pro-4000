"""Stage 5C: PyTorch reference for native Inkling convolution cache semantics.

This is a *test oracle*, not the native CUDA implementation. The C++ cache
runs on device memory; this module checks equivalent math in readable Torch.
No full-model inference, no KV-attention cache and no checkpoint loading.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch


def cache_slot(layer: int, pass_id: int) -> int:
    if type(layer) is not int or type(pass_id) is not int or not (0 <= layer < 24):
        raise ValueError("physical layer must be in 0..23")
    if layer < 6 or layer >= 18:
        if pass_id != 0:
            raise ValueError("non-core physical layers have no extra pass")
        return layer
    if pass_id not in (0, 1, 2):
        raise ValueError("core pass must be in 0..2")
    return layer if pass_id == 0 else 24 + 12 * (pass_id - 1) + layer - 6


@dataclass
class History:
    values: torch.Tensor  # [B,K-1,C] float32, masked inputs
    ids: torch.Tensor  # [B,K-1] int32, -1 when empty
    mode: str
    heads: int
    dtype: torch.dtype


class ConvolutionCache48:
    def __init__(self, *, kernel: int = 4):
        if type(kernel) is not int or not 2 <= kernel <= 32:
            raise ValueError("kernel must be in 2..32")
        self.kernel = kernel
        self._histories: dict[tuple[int, int], History] = {}

    def reset(self) -> None:
        self._histories.clear()

    def run(self, *, layer: int, pass_id: int, conv_idx: int,
            mode: str, x: torch.Tensor, static_weight: torch.Tensor,
            dynamic_taps: torch.Tensor | None = None,
            padding_mask: torch.Tensor | None = None,
            sequence_ids: torch.Tensor | None = None,
            heads: int = 0) -> torch.Tensor:
        slot = cache_slot(layer, pass_id)
        if type(conv_idx) is not int or not 0 <= conv_idx < 4:
            raise ValueError("conv index must be 0..3")
        full_attention = layer % 6 == 5
        if (mode not in ("static", "dynamic") or
            (mode == "dynamic" and (conv_idx >= 2 or not full_attention)) or
            (mode == "static" and conv_idx < 2 and full_attention)):
            raise ValueError("K/V use dynamic conv only in full-attention layers")
        if x.ndim != 3 or x.dtype not in (torch.float32, torch.bfloat16):
            raise ValueError("x must be FP32/BF16 [B,T,C]")
        b, t, c = x.shape
        k = self.kernel
        if min(b,t,c) < 1 or static_weight.shape != (c,k):
            raise ValueError("invalid geometry")
        if padding_mask is not None and padding_mask.shape != (b,t):
            raise ValueError("invalid padding mask")
        if mode == "dynamic":
            if heads < 1 or c % heads or dynamic_taps is None or dynamic_taps.shape != (b,t,heads,k):
                raise ValueError("invalid dynamic projection")
            if sequence_ids is not None and (sequence_ids.shape != (b,t) or sequence_ids.dtype != torch.int32):
                raise ValueError("sequence ids must be int32 [B,T]")
        elif dynamic_taps is not None or sequence_ids is not None or heads != 0:
            raise ValueError("static convolution forbids dynamic/seq args")
        key = (slot,conv_idx)
        state = self._histories.get(key)
        if state is None:
            state = History(x.new_zeros((b,k-1,c),dtype=torch.float32),
                            torch.full((b,k-1),-1,dtype=torch.int32,device=x.device),
                            mode,heads,x.dtype)
        if (state.mode, state.heads, state.dtype, tuple(state.values.shape),state.values.device) != \
           (mode,heads,x.dtype,(b,k-1,c),x.device):
            raise ValueError("cache stream shape/mode/dtype changed without reset")
        xf = x.float()
        masked = xf if padding_mask is None else xf * padding_mask[...,None].to(xf.dtype)
        full = torch.cat([state.values,masked],dim=1)
        ids = None
        if sequence_ids is not None:
            ids = torch.cat([state.ids,sequence_ids],dim=1)
        taps = static_weight.float()
        result = torch.empty((b,t,c),dtype=torch.float32,device=x.device)
        scale = 1/math.sqrt((2*k+1) if mode == "dynamic" else (k+1))
        for idx in range(t):
            sum_ = torch.zeros((b,c),dtype=torch.float32,device=x.device)
            for lag in range(k):
                pos = k-1+idx-lag
                sample = full[:,pos,:]
                if ids is not None:
                    sample = torch.where((ids[:,pos]==sequence_ids[:,idx])[:,None],sample,0)
                weight = taps[:,lag if mode == "dynamic" else k-1-lag][None,:]
                if mode == "dynamic":
                    weight = weight + dynamic_taps[:,idx,:,lag].float().repeat_interleave(c//heads,dim=-1)
                sum_ += sample*weight
            result[:,idx,:] = (sum_+xf[:,idx,:])*scale
        self._histories[key] = History(full[:,-(k-1):,:].clone(),
            (ids[:,-(k-1):].clone() if ids is not None else state.ids.new_full((b,k-1),-1)),
            mode,heads,x.dtype)
        return result.to(x.dtype)
