"""CPU shape/numerics fixtures for the Stage5B native CUDA acceptance test.

These check the independent scalar formulas used in C++ test against the
Stage-5A PyTorch oracle. They do NOT execute the native CUDA kernel.
"""
from __future__ import annotations

import math
import unittest

try:
    import torch
except ImportError:
    torch = None

if torch is not None:
    try:
        from .forward_oracle_stage5a import (
            weighted_rmsnorm, unweighted_rmsnorm, static_short_convolution, sink_attention
        )
    except ImportError:
        from forward_oracle_stage5a import (
            weighted_rmsnorm, unweighted_rmsnorm, static_short_convolution, sink_attention
        )


@unittest.skipUnless(torch is not None, "requires PyTorch; run in existing Docker image")
class TestStage5BNativeGolden(unittest.TestCase):
    def test_rms_scalar_2304_and_bfloat16_rounding(self):
        for dtype in (torch.float32, torch.bfloat16):
            rows, hidden = 2, 2304
            x = (((torch.arange(rows * hidden) * 17) % 79).float() - 39).reshape(rows, hidden)
            x = (x * .041).to(dtype)
            w = .6 + .025 * (torch.arange(hidden) % 23)
            torch_out = weighted_rmsnorm(x, w, 1e-6)
            manual = torch.empty_like(torch_out)
            for row in range(rows):
                values = x[row].float()
                denom = math.sqrt(sum(float(z) ** 2 for z in values) / hidden + 1e-6)
                # The initial norm rounds to BF16 before learned weights.
                norm = (values / denom).to(dtype).float()
                manual[row] = (norm * w).to(dtype)
            torch.testing.assert_close(torch_out.float(),manual.float(),atol=.012 if dtype==torch.bfloat16 else 2e-5,rtol=.01 if dtype==torch.bfloat16 else 3e-5)

    def test_unweighted_rms_128(self):
        x = torch.tensor([[1.,2.,3.,4.]]).repeat(1,32)
        expected = x / math.sqrt((x*x).mean().item()+1e-6)
        torch.testing.assert_close(unweighted_rmsnorm(x),expected,atol=1e-6,rtol=1e-6)

    def test_static_causal_conv_mask_residual_and_bfloat16(self):
        for dtype in (torch.float32,torch.bfloat16):
            b,t,c,k = 2,9,19,4
            x=(((torch.arange(b*t*c)*11)%91).float()-45).reshape(b,t,c).mul(.032).to(dtype)
            w=(((torch.arange(c*k)*13)%19).float()-9).reshape(c,k).mul(.025)
            mask=torch.tensor([[1,1,0,1,1,1,1,0,1],[0,1,1,1,0,1,1,1,1]],dtype=torch.uint8)
            reference=static_short_convolution(x,w,padding_mask=mask)
            manual=torch.empty_like(x)
            for batch in range(b):
                for pos in range(t):
                    for ch in range(c):
                        accum=0.
                        for tap in range(k):
                            src=pos+tap-(k-1)
                            if src>=0 and mask[batch,src]:
                                accum+=float(x[batch,src,ch])*float(w[ch,tap])
                        manual[batch,pos,ch]=(accum+float(x[batch,pos,ch]))/math.sqrt(k+1)
            torch.testing.assert_close(reference.float(),manual.float(),atol=.009 if dtype==torch.bfloat16 else 2e-6,rtol=.01 if dtype==torch.bfloat16 else 3e-6)

    def test_sink_gqa_scales_and_mask_against_scalar(self):
        for dtype in (torch.float32,torch.bfloat16):
            b,hq,hkv,k,d = 2,24,8,9,128
            q=(((torch.arange(b*hq*d)*17)%61).float()-30).reshape(b,hq,1,d).mul(.016).to(dtype)
            key=(((torch.arange(b*hkv*k*d)*19)%47).float()-23).reshape(b,hkv,k,d).mul(.017).to(dtype)
            value=(((torch.arange(b*hkv*k*d)*23)%59).float()-29).reshape(b,hkv,k,d).mul(.021).to(dtype)
            bias=(((torch.arange(b*hq*k)*7)%31).float()-15).reshape(b,hq,1,k).mul(.014)
            sink=(torch.arange(b*hq).float()%9).reshape(b,hq,1,1)*.06-.1
            allowed=torch.tensor([[1,1,0,1,1,1,0,1,1],[0,1,1,0,1,1,1,0,1]],dtype=torch.bool)
            mask=torch.zeros((b,1,1,k),dtype=torch.float32).masked_fill(~allowed[:,None,None,:],-float('inf'))
            actual=sink_attention(q,key,value,position_bias=bias,attention_mask=mask,sinks=sink)
            expected=torch.zeros_like(actual)
            for bi in range(b):
                for head in range(hq):
                    kh=head//(hq//hkv)
                    logits=[]
                    for token in range(k):
                        if allowed[bi,token]:
                            dot=sum(float(q[bi,head,0,dim])*float(key[bi,kh,token,dim]) for dim in range(d))
                            logits.append((token,dot/math.sqrt(d)+float(bias[bi,head,0,token])))
                    max_logit=max([float(sink[bi,head,0,0])]+[x[1] for x in logits])
                    denominator=math.exp(float(sink[bi,head,0,0])-max_logit)+sum(math.exp(x[1]-max_logit) for x in logits)
                    for dim in range(d):
                        numer=sum(math.exp(score-max_logit)*float(value[bi,kh,token,dim]) for token,score in logits)
                        expected[bi,0,head,dim]=numer/denominator
            torch.testing.assert_close(actual.float(),expected.float(),atol=.004 if dtype==torch.bfloat16 else 2e-6,rtol=.007 if dtype==torch.bfloat16 else 1e-5)

    def test_all_keys_masked_sink_zero_value(self):
        q=torch.ones(1,6,1,16)
        k=torch.ones(1,2,5,16)
        v=torch.ones(1,2,5,16)
        mask=torch.full((1,1,1,5),-float('inf'))
        result=sink_attention(q,k,v,attention_mask=mask,sinks=torch.zeros(1,6,1,1))
        self.assertTrue(bool((result==0).all()))

    def test_one_key_and_sink_half(self):
        q=torch.zeros(1,6,1,16)
        k=torch.zeros(1,2,1,16)
        v=torch.ones(1,2,1,16)
        result=sink_attention(q,k,v,sinks=torch.zeros(1,6,1,1))
        self.assertTrue(bool((result==.5).all()))


if __name__ == '__main__':
    unittest.main()
