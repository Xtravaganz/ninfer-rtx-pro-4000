"""CPU Torch tests for Stage 5C stateful native conv cache semantics."""
import unittest

try:
    import torch
    from forward_state_stage5c import ConvolutionCache48, cache_slot
    from forward_oracle_stage5a import dynamic_short_convolution, static_short_convolution
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "requires Torch (use existing PyTorch Docker image)")
class Stage5CCacheTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(502)
        self.k = 4

    def test_exact_48_slot_map(self):
        slots = [cache_slot(i,0) for i in range(24)]
        slots += [cache_slot(i,p) for p in (1,2) for i in range(6,18)]
        self.assertEqual(set(slots),set(range(48)))
        self.assertEqual(cache_slot(6,0),6)
        self.assertEqual(cache_slot(6,1),24)
        self.assertEqual(cache_slot(6,2),36)
        self.assertEqual(cache_slot(17,2),47)
        self.assertEqual(cache_slot(23,0),23)

    def test_invalid_slot_rejected(self):
        for args in [(-1,0),(24,0),(5,1),(18,1),(6,3),(6,-1),(1,True)]:
            with self.assertRaises(ValueError):cache_slot(*args)

    def test_static_prefill_parity_and_mask(self):
        b,t,c=2,11,9
        x=torch.randn(b,t,c)
        w=torch.randn(c,4)*0.08
        mask=torch.tensor([[1,1,0,1,1,1,0,1,1,1,1],[1,0,1,1,1,1,1,1,0,1,1]],dtype=torch.uint8)
        full=ConvolutionCache48().run(layer=0,pass_id=0,conv_idx=2,mode="static",x=x,static_weight=w,padding_mask=mask)
        expected=static_short_convolution(x,w,padding_mask=mask)
        torch.testing.assert_close(full,expected,atol=1e-6,rtol=1e-6)
        stream=ConvolutionCache48()
        parts=[]
        for a,z in ((0,2),(2,3),(3,7),(7,11)):
            parts.append(stream.run(layer=0,pass_id=0,conv_idx=2,mode="static",x=x[:,a:z],static_weight=w,padding_mask=mask[:,a:z]))
        torch.testing.assert_close(torch.cat(parts,dim=1),full,atol=1e-6,rtol=1e-6)

    def test_dynamic_prefill_matches_stage5a_oracle(self):
        b,t,c,h,inp=2,11,16,4,7
        x=torch.randn(b,t,c)
        dynamic_input=torch.randn(b,t,inp)
        static=torch.randn(4,c)*0.12
        project=torch.randn(h*4,inp)*0.11
        tap=torch.nn.functional.linear(dynamic_input.float(),project).reshape(b,t,h,4)
        mask=torch.tensor([[1,1,0,1,1,1,1,0,1,1,1],[1,1,1,1,0,1,1,1,1,1,1]],dtype=torch.uint8)
        ids=torch.tensor([[0,0,0,0,2,2,2,2,2,5,5],[7,7,7,9,9,9,9,9,9,9,9]],dtype=torch.int32)
        expected=dynamic_short_convolution(x,dynamic_input,static,project,num_heads=h,padding_mask=mask,sequence_ids=ids)
        full=ConvolutionCache48().run(layer=5,pass_id=0,conv_idx=0,mode="dynamic",x=x,static_weight=static.T.contiguous(),dynamic_taps=tap,padding_mask=mask,sequence_ids=ids,heads=h)
        torch.testing.assert_close(full,expected,atol=1e-6,rtol=1e-6)
        stream=ConvolutionCache48()
        parts=[]
        for a,z in ((0,1),(1,3),(3,7),(7,8),(8,11)):
            parts.append(stream.run(layer=5,pass_id=0,conv_idx=0,mode="dynamic",x=x[:,a:z],static_weight=static.T.contiguous(),dynamic_taps=tap[:,a:z],padding_mask=mask[:,a:z],sequence_ids=ids[:,a:z],heads=h))
        torch.testing.assert_close(torch.cat(parts,dim=1),full,atol=1e-6,rtol=1e-6)

    def test_dynamic_no_ids(self):
        x=torch.randn(1,13,24)
        h=3
        din=torch.randn(1,13,5)
        w=torch.randn(4,24)*0.05
        p=torch.randn(h*4,5)*0.05
        tap=torch.nn.functional.linear(din,p).reshape(1,13,h,4)
        exp=dynamic_short_convolution(x,din,w,p,num_heads=h)
        got=ConvolutionCache48().run(layer=23,pass_id=0,conv_idx=1,mode="dynamic",x=x,static_weight=w.T.contiguous(),dynamic_taps=tap,heads=h)
        torch.testing.assert_close(got,exp,rtol=1e-6,atol=1e-6)

    def test_sliding_kv_static_causal(self):
        x=torch.randn(1,6,16)
        w=torch.randn(16,4)*0.09
        ref=static_short_convolution(x,w)
        got=ConvolutionCache48().run(layer=6,pass_id=2,conv_idx=0,mode="static",x=x,static_weight=w)
        torch.testing.assert_close(got,ref,rtol=1e-6,atol=1e-6)

    def test_bf16_parity(self):
        x=torch.randn(1,9,128,dtype=torch.float32).bfloat16()
        w=torch.randn(128,4)*0.15
        d=torch.randn(1,9,2,4)*0.04
        full=ConvolutionCache48().run(layer=11,pass_id=0,conv_idx=1,mode="dynamic",x=x,static_weight=w,dynamic_taps=d,heads=2)
        cache=ConvolutionCache48()
        step=torch.cat([cache.run(layer=11,pass_id=0,conv_idx=1,mode="dynamic",x=x[:,i:i+1],static_weight=w,dynamic_taps=d[:,i:i+1],heads=2) for i in range(9)],dim=1)
        torch.testing.assert_close(step,full,atol=0,rtol=0)

    def test_core_pass_cache_isolation(self):
        cache=ConvolutionCache48()
        w=torch.zeros(4,4);w[:,2]=0.7
        x=torch.ones(1,1,4)
        cache.run(layer=6,pass_id=0,conv_idx=2,mode="static",x=x,static_weight=w)
        zero=torch.zeros_like(x)
        empty=cache.run(layer=6,pass_id=1,conv_idx=2,mode="static",x=zero,static_weight=w)
        populated=cache.run(layer=6,pass_id=0,conv_idx=2,mode="static",x=zero,static_weight=w)
        self.assertGreater(populated.abs().max().item(),0)
        self.assertEqual(empty.abs().max().item(),0)

    def test_conv_stream_separation_and_reset(self):
        cache=ConvolutionCache48()
        x=torch.ones(1,1,4)
        z=torch.zeros_like(x)
        w=torch.ones(4,4)
        cache.run(layer=6,pass_id=0,conv_idx=2,mode="static",x=x,static_weight=w)
        self.assertEqual(cache.run(layer=6,pass_id=0,conv_idx=3,mode="static",x=z,static_weight=w).abs().max().item(),0)
        cache.reset()
        self.assertEqual(cache.run(layer=6,pass_id=0,conv_idx=2,mode="static",x=z,static_weight=w).abs().max().item(),0)

    def test_mode_dtype_and_geometry_guards(self):
        c=ConvolutionCache48()
        x=torch.ones(1,2,8)
        w=torch.ones(8,4)
        with self.assertRaises(ValueError):c.run(layer=11,pass_id=0,conv_idx=0,mode="static",x=x,static_weight=w)
        with self.assertRaises(ValueError):c.run(layer=11,pass_id=0,conv_idx=2,mode="dynamic",x=x,static_weight=w,heads=2,dynamic_taps=torch.zeros(1,2,2,4))
        with self.assertRaises(ValueError):c.run(layer=11,pass_id=0,conv_idx=0,mode="dynamic",x=x,static_weight=w,heads=3,dynamic_taps=torch.zeros(1,2,3,4))
        c.run(layer=6,pass_id=0,conv_idx=2,mode="static",x=x,static_weight=w)
        with self.assertRaises(ValueError):c.run(layer=6,pass_id=0,conv_idx=2,mode="static",x=x.bfloat16(),static_weight=w)

if __name__ == "__main__":unittest.main()
