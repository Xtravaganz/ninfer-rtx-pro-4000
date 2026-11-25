"""Stage5S/T narrow, independent W13 indexing tests."""
import unittest
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
from gateup_audit_stage5st import interleave_source_rows, split_gate_up_source_order, previous_split_gate_up

try:
    import torch
except ImportError:
    torch=None


@unittest.skipIf(torch is None,'torch unavailable (run PyTorch Docker)')
class TestHFInterleave(unittest.TestCase):
    def test_explicit_row_permutation(self):
        m=torch.arange(6,dtype=torch.float32).reshape(6,1)
        self.assertEqual(interleave_source_rows(m)[:,0].tolist(),[0,2,4,1,3,5])

    def test_projection_equivalence(self):
        torch.manual_seed(341)
        for rows, cols in ((4,7),(8,16),(2304,8)):
            m=torch.randn(rows,cols)
            x=torch.randn(cols)
            raw=m@x
            gate,up=split_gate_up_source_order(raw)
            expected=interleave_source_rows(m)@x
            self.assertTrue(torch.allclose(gate,expected[:rows//2],atol=1e-5,rtol=1e-5))
            self.assertTrue(torch.allclose(up,expected[rows//2:],atol=1e-5,rtol=1e-5))

    def test_old_split_is_wrong(self):
        v=torch.arange(8,dtype=torch.float32)
        gate,up=split_gate_up_source_order(v)
        old_gate,old_up=previous_split_gate_up(v)
        self.assertEqual(gate.tolist(),[0,2,4,6])
        self.assertEqual(up.tolist(),[1,3,5,7])
        self.assertNotEqual(gate.tolist(),old_gate.tolist())
        self.assertNotEqual(up.tolist(),old_up.tolist())

    def test_source_order_activation_matches_explicit_hf_matrix(self):
        torch.manual_seed(3)
        m=torch.randn(12,5)
        x=torch.randn(5)
        raw=m@x
        gate,up=split_gate_up_source_order(raw)
        expected=interleave_source_rows(m)@x
        g2,u2=expected.chunk(2)
        act=lambda g,u:4*torch.tanh(g/4)*torch.sigmoid(g)*25*torch.tanh(u/25)
        self.assertTrue(torch.allclose(act(gate,up),act(g2,u2),atol=2e-5))
        og,ou=previous_split_gate_up(raw)
        self.assertGreater(float((act(gate,up)-act(og,ou)).abs().max()),0.001)

    def test_bad_geometries(self):
        for t in (torch.empty(0,4),torch.empty(3,4),torch.empty(4),torch.empty(2,2,2)):
            with self.assertRaises(ValueError):interleave_source_rows(t)
        for t in (torch.empty(0),torch.ones(3),torch.ones(2,2)):
            with self.assertRaises(ValueError):split_gate_up_source_order(t)

    def test_corrected_moe_end_to_end_small_synthetic(self):
        from unittest.mock import patch
        from types import SimpleNamespace
        from gateup_audit_stage5st import corrected_moe
        try:
            from real_sliding_stage5h import TorchCarrierWeights
            import real_sliding_stage5h
        except ImportError:
            self.skipTest('Stage5H not installed here')
        # Explicitly interleaved W13 rows: gate0,up0,gate1,up1.
        w13=torch.tensor([[2.,0.,0.,0.],[0.,2.,0.,0.],
                          [0.,0.,2.,0.],[0.,0.,0.,2.]])
        w2=torch.tensor([[1.,0.],[0.,1.],[1.,1.],[0.,-1.]])
        class Reader:
            def bank(self,layer,projection,ids):
                return [SimpleNamespace(kind=projection)]
        class Weights:
            geom=SimpleNamespace(hidden=4,intermediate=2,experts=2,top_k=1,route_scale=1.0)
            device=torch.device('cpu')
            reader=Reader()
            def tensor(self,name,shape):
                if name.endswith('gate.weight'):return torch.zeros(2,4)
                if name.endswith('gate.bias'):return torch.tensor([1.,0.])
                raise AssertionError(name)
        weights=Weights()
        x=torch.tensor([[[1.,2.,3.,4.]]])
        def decode(raw,*,rows,cols,fmt,device):
            return w13 if raw.kind=='w13' else w2
        with patch.object(real_sliding_stage5h,'decode_expert',side_effect=decode):
            expected_raw=w13@x.reshape(4)
            gate,up=expected_raw[0::2],expected_raw[1::2]
            activate=lambda g,u:4*torch.tanh(g/4)*torch.sigmoid(g)*25*torch.tanh(u/25)
            expected=w2@activate(gate,up)
            got, ids=corrected_moe(weights,x,0)
            self.assertEqual(ids,[0])
            self.assertTrue(torch.allclose(got.reshape(-1),expected,rtol=1e-6,atol=1e-6))
            old, _=TorchCarrierWeights.moe(weights,x,0)
            self.assertGreater(float((old-got).abs().max()),0.1)

    def test_no_global_mutation(self):
        class C:pass
        from gateup_audit_stage5st import temporary_corrected_moe, corrected_moe
        # Import the Torch class from installed Stage5H files only when available.
        try:
            from real_sliding_stage5h import TorchCarrierWeights
        except ImportError:
            self.skipTest('Stage5H modules not installed here')
        old=TorchCarrierWeights.moe
        with temporary_corrected_moe():
            self.assertIs(TorchCarrierWeights.moe,corrected_moe)
        self.assertIs(TorchCarrierWeights.moe,old)

if __name__=='__main__':unittest.main()
