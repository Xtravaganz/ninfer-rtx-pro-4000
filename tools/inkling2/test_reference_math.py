"""CPU oracles for the Inkling formulas; no model weights needed."""
from __future__ import annotations

import unittest
try:
    import torch
except ImportError:
    torch = None


@unittest.skipUnless(torch is not None, 'requires torch')
class TestMath(unittest.TestCase):
    def test_swiglu_uses_raw_gate_inside_sigmoid(self):
        from reference_math import softcapped_swiglu
        gate = torch.tensor([[-10., 0., 10.]], dtype=torch.float32)
        up = torch.tensor([[1., 3., 30.]], dtype=torch.float32)
        expected = (4 * torch.tanh(gate / 4)) * torch.sigmoid(gate) * (25 * torch.tanh(up / 25))
        torch.testing.assert_close(softcapped_swiglu(gate, up), expected)

    def test_message_gate_zero_weights_is_identity(self):
        from reference_math import message_gate
        hidden = torch.tensor([[3., -1.]])
        message = torch.tensor([[8., 5.]])
        down = torch.zeros(2, 4)
        forget = torch.zeros(2, 2)
        inject = torch.zeros(2, 2)
        torch.testing.assert_close(message_gate(hidden, message, down, forget, inject), hidden)

    def test_router_selection_bias_does_not_modify_weights(self):
        from reference_math import route_topk
        hidden = torch.tensor([[1., 0.]])
        router = torch.tensor([[1., 0.], [0., 1.], [0., 0.]])
        bias = torch.tensor([0., 1., 0.])
        indices, weights = route_topk(hidden, router, bias, top_k=2, route_scale=8.)
        self.assertIn(1, indices[0].tolist())
        torch.testing.assert_close(weights.sum(-1), torch.tensor([8.]))
        scores = torch.sigmoid(torch.tensor([1., 0., 0.]))
        expected = scores[indices] / scores[indices].sum(-1, keepdim=True) * 8
        torch.testing.assert_close(weights, expected)

    def test_block_attention_single_state(self):
        from reference_math import block_message_attention, unweighted_rmsnorm
        hidden = torch.tensor([[[3., 4.]]])
        blocks = torch.empty(1, 1, 0, 2)
        query = torch.tensor([1., -1.])
        got = block_message_attention(blocks, hidden, query)
        want = unweighted_rmsnorm(hidden).unsqueeze(-2)
        torch.testing.assert_close(got, want)


if __name__ == '__main__':
    unittest.main()
