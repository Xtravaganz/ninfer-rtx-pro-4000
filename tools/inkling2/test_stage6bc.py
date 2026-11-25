"""Offline CPU tests, with optional CUDA via PyTorch image."""
from __future__ import annotations
import sys, unittest
from pathlib import Path
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))

from profile_stage6b import SpanProfiler,PatchScope

try:
 import torch
 from batch_prefill_stage6c import attention_prefill,static_prefill,dynamic_prefill,vectorized_linear
 from forward_oracle_stage5a import sink_attention,static_short_convolution_step
except ImportError:
 torch=None


class TestProfiler(unittest.TestCase):
 def test_nested_exclusive(self):
    p=SpanProfiler()
    with p.span('outside'):
     with p.span('inside'): sum(range(100))
    r=p.result()['sections']
    self.assertEqual((r['outside']['calls'],r['inside']['calls']),(1,1))
    self.assertGreaterEqual(r['outside']['inclusive_s'],r['outside']['exclusive_s'])
    self.assertLessEqual(r['outside']['exclusive_s'],r['outside']['inclusive_s'])
 def test_patch_restores_on_exception(self):
    class X:
     def f(self):return 2
    x=X()
    with self.assertRaisesRegex(RuntimeError,'stop'):
     with PatchScope() as p:
      p.patch(x,'f',lambda:3)
      self.assertEqual(x.f(),3)
      raise RuntimeError('stop')
    self.assertEqual(x.f(),2)


@unittest.skipIf(torch is None,'torch unavailable')
class TestBatchOps(unittest.TestCase):
 def _device(self):return 'cuda' if torch.cuda.is_available() else 'cpu'
 def test_attention_causal_sink_matches_incremental_oracle(self):
    torch.manual_seed(13);d=self._device()
    b,t,hq,hk,kdim=1,16,6,2,8
    q=torch.randn((b,hq,t,kdim),device=d)*.1
    k=torch.randn((b,hk,t,kdim),device=d)*.1
    v=torch.randn_like(k)
    sinks=torch.randn((b,hq,t,1),device=d)*.1
    out=attention_prefill(q,k,v,sink=sinks)
    parts=[]
    for step in range(t):
     parts.append(sink_attention(q[:,:,step:step+1],k[:,:,:step+1],v[:,:,:step+1],sinks=sinks[:,:,step:step+1]))
    sequential=torch.cat(parts,dim=1)
    torch.testing.assert_close(out,sequential,atol=3e-6,rtol=3e-6)
 def test_attention_sliding_and_relative_bias(self):
    torch.manual_seed(10);d=self._device()
    q=torch.randn((1,6,12,8),device=d)*.2
    k=torch.randn((1,2,12,8),device=d)*.2
    v=torch.randn_like(k)
    sinks=torch.randn((1,6,12,1),device=d)*.1
    bias=torch.randn((1,6,12,12),device=d)*.03
    out=attention_prefill(q,k,v,sink=sinks,relative_bias=bias,sliding_window=4)
    seq=[]
    for pos in range(12):
     first=max(0,pos-3)
     seq.append(sink_attention(q[:,:,pos:pos+1],k[:,:,first:pos+1],v[:,:,first:pos+1],
                position_bias=bias[:,:,pos:pos+1,first:pos+1],sinks=sinks[:,:,pos:pos+1]))
    torch.testing.assert_close(out,torch.cat(seq,dim=1),atol=3e-6,rtol=3e-6)
 def test_static_conv_batch_matches_incremental_with_padding(self):
    torch.manual_seed(7);d=self._device()
    x=torch.randn((2,17,24),device=d)
    taps=torch.randn((24,4),device=d)*.04
    mask=torch.ones((2,17),device=d);mask[1,:3]=0
    batch=static_prefill(x,taps,mask)
    hist=torch.zeros((2,3,24),device=d)
    items=[]
    for i in range(17):
     y,hist=static_short_convolution_step(x[:,i:i+1],taps,hist,padding_mask=mask[:,i:i+1]);items.append(y)
    torch.testing.assert_close(batch,torch.cat(items,dim=1),atol=4e-6,rtol=4e-6)
 def test_dynamic_conv_independent_explicit_windows(self):
    import math
    torch.manual_seed(14);d=self._device()
    b,t,h,c,k=1,13,2,8,4
    x=torch.randn((b,t,c),device=d)*.2
    context=torch.randn((b,t,7),device=d)*.2
    taps=torch.randn((k,c),device=d)*.1
    proj=torch.randn((h*k,7),device=d)*.2
    seq_ids=torch.tensor([[0]*5+[1]*8],device=d)
    out=dynamic_prefill(x,context,taps,proj,h,sequence_ids=seq_ids)
    ref=torch.empty_like(out)
    for p in range(t):
     delta=(proj@context[0,p]).reshape(h,k)
     for channel in range(c):
      head=channel//(c//h)
      acc=0.
      for lag in range(k):
       previous=p-lag
       if previous>=0 and seq_ids[0,p]==seq_ids[0,previous]:
        acc+=float(x[0,previous,channel])*(float(taps[lag,channel])+float(delta[head,lag]))
      ref[0,p,channel]=(acc+float(x[0,p,channel]))/math.sqrt(2*k+1)
    torch.testing.assert_close(out,ref,atol=2e-6,rtol=2e-6)
 def test_vectorized_linear_matches_token_loop(self):
    torch.manual_seed(23);d=self._device()
    x=torch.randn((1,23,64),device=d)
    w=torch.randn((96,64),device=d)
    out=vectorized_linear(x,w)
    seq=torch.stack([w@x[0,i] for i in range(23)],dim=0).unsqueeze(0)
    torch.testing.assert_close(out,seq,atol=1e-4,rtol=1e-4)
 def test_input_contract(self):
    with self.assertRaises(ValueError):
     attention_prefill(torch.zeros(1,4,3,8),torch.zeros(1,3,3,8),torch.zeros(1,3,3,8),sink=torch.zeros(1,4,3,1))

if __name__=='__main__':unittest.main()
