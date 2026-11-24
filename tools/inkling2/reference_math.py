"""Reference-only Inkling 2 Mini primitives for numerical oracle tests.

These implement the published Transformers formulas with Torch tensors. They
are NOT a NInfer runtime or optimized CUDA implementation.
"""
from __future__ import annotations

import math
import torch
import torch.nn.functional as F


def unweighted_rmsnorm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    y = x.float()
    return y * torch.rsqrt(y.square().mean(-1, keepdim=True) + eps)


def message_gate(hidden: torch.Tensor, message: torch.Tensor,
                 down_weight: torch.Tensor, forget_weight: torch.Tensor,
                 message_weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """(1 - tanh(W_f d)) * h + tanh(W_m d) * message, with d=W_down [RMS(h), RMS(m)]."""
    h, m = hidden.float(), message.float()
    joint = torch.cat((unweighted_rmsnorm(h, eps),
                       unweighted_rmsnorm(m, eps)), dim=-1)
    rank = F.linear(joint.to(down_weight.dtype), down_weight).float()
    return ((1 - torch.tanh(F.linear(rank, forget_weight.float()))) * h
            + torch.tanh(F.linear(rank, message_weight.float())) * m)


def softcapped_swiglu(gate: torch.Tensor, up: torch.Tensor,
                       gate_cap: float = 4., up_cap: float = 25.) -> torch.Tensor:
    """Softcap(gate)*sigmoid(RAW gate)*Softcap(up), preserving input dtype."""
    g, u = gate.float(), up.float()
    capped_gate = torch.tanh(g / gate_cap) * gate_cap
    capped_up = torch.tanh(u / up_cap) * up_cap
    return (capped_gate * torch.sigmoid(g) * capped_up).to(gate.dtype)


def route_topk(hidden: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor,
               top_k: int = 8, route_scale: float = 8.):
    """Select by sigmoid(logits)+bias, normalize original scores for selected experts."""
    logits = F.linear(hidden.float(), weight.float())
    scores = torch.sigmoid(logits)
    indices = torch.topk(scores + bias.float(), top_k, dim=-1, sorted=False).indices
    weights = scores.gather(-1, indices)
    weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-20)
    return indices, weights * route_scale


def block_message_attention(block_states: torch.Tensor, partial_state: torch.Tensor,
                            queries: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Unmasked per-token attention over RMS-normalized block states and current state.

    `queries` is [hidden] or [num_queries,hidden]. Returns [..., num_queries,hidden].
    """
    stacked = unweighted_rmsnorm(torch.cat((block_states, partial_state.unsqueeze(-2)), -2), eps)
    q = queries.float().reshape(-1, stacked.shape[-1])
    logits = torch.matmul(q, stacked.transpose(-1, -2)) / math.sqrt(stacked.shape[-1])
    p = logits.softmax(-1)
    return torch.matmul(p, stacked)
