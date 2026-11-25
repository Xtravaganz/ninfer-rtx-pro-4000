"""Stage5U regression: same HF-corrected math, bounded caching, no model downloads."""
from __future__ import annotations
import math
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

HERE=Path(__file__).resolve().parent
for path in (str(HERE),str(HERE.parents[1])):
    if path not in sys.path:
        sys.path.insert(0,path)

try:
    import torch
except ImportError:
    torch=None


@unittest.skipUnless(torch is not None, 'requires torch')
class TestStage5U(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from real_sliding_stage5h import Geometry
        cls.geometry=Geometry(hidden=32,intermediate=16,experts=4,top_k=2,
                              query_heads=2,kv_heads=1,head_dim=8,rel_dim=4,
                              rel_extent=8,conv_kernel=4)

    def reader(self):
        return ToyReader()

    def test_corrected_moe_is_equal_to_uncached_reference(self):
        from gateup_audit_stage5st import corrected_moe
        from real_sliding_stage5h import TorchCarrierWeights
        from prefill_cache_stage5u import CachedCarrierWeights
        x=torch.arange(1,33,dtype=torch.float32).reshape(1,1,32)/128
        ref=TorchCarrierWeights(self.reader(),'cpu',self.geometry)
        fast=CachedCarrierWeights(self.reader(),'cpu',self.geometry,
                                  nonexpert_mib=1,expert_mib=1)
        with torch.inference_mode():
            expected,ei=corrected_moe(ref,x,0)
            actual,ai=fast.moe(x,0)
        self.assertEqual(ai,ei)
        self.assertTrue(torch.equal(actual,expected))
        self.assertGreater(float(actual.abs().max()),0)

    def test_reuses_selected_experts_and_nonexperts(self):
        from prefill_cache_stage5u import CachedCarrierWeights
        source=self.reader()
        fast=CachedCarrierWeights(source,'cpu',self.geometry,
                                  nonexpert_mib=1,expert_mib=1)
        x=torch.arange(1,33,dtype=torch.float32).reshape(1,1,32)/128
        with torch.inference_mode():
            y,ids=fast.moe(x,0)
            fast.clear()  # 5Q calls this 48 times per token
            z,ids2=fast.moe(x,0)
        self.assertEqual(ids,ids2)
        self.assertTrue(torch.equal(y,z))
        self.assertEqual(source.bank_calls,2)
        self.assertEqual(source.tensor_calls,2)
        stats=fast.metrics()
        self.assertEqual(stats['expert_misses'],4)
        self.assertEqual(stats['expert_hits'],4)
        self.assertEqual(stats['clear_requests_ignored'],1)
        self.assertGreaterEqual(stats['nonexpert_hits'],2)
        self.assertEqual(stats['resident_expert_matrices'],4)
        fast.release()
        self.assertEqual(fast.metrics()['resident_expert_matrices'],0)

    def test_zero_budget_keeps_math_but_reloads(self):
        from prefill_cache_stage5u import CachedCarrierWeights
        source=self.reader()
        fast=CachedCarrierWeights(source,'cpu',self.geometry,
                                  nonexpert_mib=0,expert_mib=0)
        x=torch.arange(1,33,dtype=torch.float32).reshape(1,1,32)/128
        with torch.inference_mode():
            y,_=fast.moe(x,0)
            fast.clear()
            z,_=fast.moe(x,0)
        self.assertTrue(torch.equal(y,z))
        self.assertEqual(source.bank_calls,4)
        self.assertEqual(source.tensor_calls,4)
        self.assertEqual(fast.metrics()['resident_expert_matrices'],0)
        self.assertEqual(fast.metrics()['resident_nonexpert_tensors'],0)

    def test_tensor_read_once_even_after_clear(self):
        from prefill_cache_stage5u import CachedCarrierWeights
        source=self.reader()
        fast=CachedCarrierWeights(source,'cpu',self.geometry,
                                  nonexpert_mib=1,expert_mib=0)
        name='model.llm.layers.0.mlp.gate.weight'
        a=fast.tensor(name,(4,32))
        fast.clear()
        b=fast.tensor(name,(4,32))
        self.assertIs(a,b)
        self.assertEqual(source.tensor_calls,1)
        with self.assertRaises(ValueError):fast.tensor(name,(32,4))

    def test_convolution_metadata_does_not_read_weights_twice(self):
        from prefill_cache_stage5u import CachedCarrierWeights
        source=self.reader()
        source.add_conv('model.llm.layers.0.attn.k_sconv.weight', 8,4)
        fast=CachedCarrierWeights(source,'cpu',self.geometry,
                                  nonexpert_mib=1,expert_mib=0)
        a=fast.conv('model.llm.layers.0.attn.k_sconv.weight',8)
        b=fast.conv('model.llm.layers.0.attn.k_sconv.weight',8)
        self.assertTrue(torch.equal(a,b))
        self.assertEqual(source.tensor_calls,1)
        self.assertEqual(source.nonexpert.object_calls,2)
        with self.assertRaises(ValueError):fast.conv('model.llm.layers.0.attn.k_sconv.weight',9)

    def test_budget_and_source_validation(self):
        from prefill_cache_stage5u import CachedCarrierWeights
        for bad in (-1, 16385, 1.25, True):
            with self.assertRaises(ValueError):
                CachedCarrierWeights(self.reader(),'cpu',self.geometry,
                                     nonexpert_mib=bad,expert_mib=1)
        source=self.reader()
        fast=CachedCarrierWeights(source,'cpu',self.geometry,
                                  nonexpert_mib=1,expert_mib=1)
        with self.assertRaises(ValueError):fast._selected_experts(0,'w13',[2,2])
        with self.assertRaises(ValueError):fast._selected_experts(0,'w13',[9])
        source.bad_bank=True
        with self.assertRaises(ValueError):fast._selected_experts(0,'w13',[3])

    def test_real_w13_interleave_not_half_split(self):
        from gateup_audit_stage5st import split_gate_up_source_order, previous_split_gate_up
        from real_sliding_stage5h import decode_expert
        raw=self.reader().bank(0,'w13',[3])[0]
        mat=decode_expert(raw,rows=32,cols=32,fmt='nvfp4',device=torch.device('cpu'))
        x=torch.arange(1,33,dtype=torch.float32)/128
        y=mat@x
        gate,up=split_gate_up_source_order(y)
        old_gate,old_up=previous_split_gate_up(y)
        self.assertTrue(torch.equal(gate,y[0::2]))
        self.assertTrue(torch.equal(up,y[1::2]))
        self.assertFalse(torch.equal(gate,old_gate))
        self.assertFalse(torch.equal(up,old_up))


class ToyReader:
    def __init__(self):
        if torch is None:raise RuntimeError('requires torch')
        self.tensor_calls=0
        self.bank_calls=0
        self.bad_bank=False
        weight=torch.zeros(4,32,dtype=torch.float32)
        bias=torch.tensor([0.,0.1,0.2,0.3],dtype=torch.float32)
        self._tensors={
            'model.llm.layers.0.mlp.gate.weight': (weight.numpy().tobytes(), 'fp32',(4,32)),
            'model.llm.layers.0.mlp.gate.bias': (bias.numpy().tobytes(),'fp32',(4,)),
        }
        parent=self
        class Directory:
            object_calls=0
            def object(self,name):
                self.object_calls+=1
                if not name.startswith('hf/') or name[3:] not in parent._tensors:
                    raise ValueError(name)
                return SimpleNamespace(shape=parent._tensors[name[3:]][2])
        self.nonexpert=Directory()

    def add_conv(self,name,channels,kernel):
        value=torch.ones(channels,1,kernel,dtype=torch.float32)
        self._tensors[name]=(value.numpy().tobytes(),'fp32',tuple(value.shape))

    def nonexpert_tensor(self,name):
        self.tensor_calls+=1
        return self._tensors[name]

    def bank(self,layer,projection,ids):
        self.bank_calls+=1
        if self.bad_bank:return []
        rows=32
        cols=32 if projection=='w13' else 16
        codes=bytearray(rows*cols//2)
        for row in range(rows):
            start=row*cols//2
            code=0x21 if row%2==0 else 0x43
            codes[start:start+cols//2]=bytes([code])*(cols//2)
        scales=bytes([0x38])*(rows*cols//16)
        return [SimpleNamespace(codes=bytes(codes),scales=scales,
                                amax=1.0,denominator=1.0) for _ in ids]


@unittest.skipUnless(torch is not None, 'requires torch')
class TestStage5UOptIn(unittest.TestCase):
    def _args(self, mode):
        from argparse import Namespace
        return Namespace(source=Path('/fake/source'),nonexpert=Path('/fake/nonexpert'),
            experts=Path('/fake/experts'),banks=Path('/fake/banks'),mode=mode,
            nonexpert_cache_mib=1,expert_cache_mib=1,device='cpu',question='Test',
            max_prompt_tokens=4,max_new_tokens=1,raw=True,tokenize_only=True,
            per_token_profile=False)

    def test_opt_in_runner_restores_runtime_for_both_modes(self):
        from unittest.mock import patch
        import real_sliding_stage5h,streaming_stage5q,ask_inkling_stage5r
        from ask_prefill_stage5u import run
        before=(real_sliding_stage5h.TorchCarrierWeights,
                streaming_stage5q.forward_one_token,ask_inkling_stage5r.tokenize_question)
        for mode in ('baseline','cached'):
            names=[]
            def fake_run(_args):
                names.append(real_sliding_stage5h.TorchCarrierWeights.__name__)
                return 0
            with patch.object(ask_inkling_stage5r,'run',fake_run):
                self.assertEqual(run(self._args(mode)),0)
            self.assertEqual(names,['BaselineWeights' if mode=='baseline' else 'CachedWeights'])
            self.assertIs(real_sliding_stage5h.TorchCarrierWeights,before[0])
            self.assertIs(streaming_stage5q.forward_one_token,before[1])
            self.assertIs(ask_inkling_stage5r.tokenize_question,before[2])

    def test_opt_in_runner_restores_runtime_after_failure(self):
        from unittest.mock import patch
        import real_sliding_stage5h,streaming_stage5q,ask_inkling_stage5r
        from ask_prefill_stage5u import run
        before=(real_sliding_stage5h.TorchCarrierWeights,streaming_stage5q.forward_one_token)
        with patch.object(ask_inkling_stage5r,'run',side_effect=ValueError('synthetic error')):
            with self.assertRaisesRegex(ValueError,'synthetic error'):
                run(self._args('cached'))
        self.assertIs(real_sliding_stage5h.TorchCarrierWeights,before[0])
        self.assertIs(streaming_stage5q.forward_one_token,before[1])


if __name__=='__main__':unittest.main()
