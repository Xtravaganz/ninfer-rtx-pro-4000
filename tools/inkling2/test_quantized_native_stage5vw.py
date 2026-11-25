"""Stage5V/W fast CPU contracts; CUDA correctness is a separate on-device gate."""
from __future__ import annotations
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from quantized_store_stage5v import expert_geometry, validate_expert_words, QuantizedGPUStore


@dataclass
class Words:
    codes: bytes
    scales: bytes
    amax: float = 1.0
    denominator: float = 1.0


def generate(layer, proj, hid=64, inter=32, *, bad=False):
    n,k,fmt=expert_geometry(layer,proj,hid,inter)
    codes = bytes([0x12 if fmt=='nvfp4' else 0x38])*(n*k//(2 if fmt=='nvfp4' else 1))
    scales = bytes([0x38 if fmt=='nvfp4' else 127])*(n*k//(16 if fmt=='nvfp4' else 32))
    if bad:
        codes = bytes([0x7f])+codes[1:] if fmt=='mxfp8' else codes
        scales = bytes([0x7f])+scales[1:] if fmt=='nvfp4' else scales
    return Words(codes,scales)


class FakeReader:
    def __init__(self): self.calls=[]
    def bank(self,layer,proj,ids):
        self.calls.append((layer,proj,tuple(ids)))
        return [generate(layer,proj) for _ in ids]


class TestGeometry(unittest.TestCase):
    def test_dimensions_and_format(self):
        self.assertEqual(expert_geometry(0,'w13'), (2304,2304,'nvfp4'))
        self.assertEqual(expert_geometry(20,'w2'), (2304,1152,'mxfp8'))
        self.assertEqual(expert_geometry(17,'w2'), (2304,1152,'nvfp4'))

    def test_invalid_geometry(self):
        for layer in (-1,24,True):
            with self.assertRaises(ValueError): expert_geometry(layer,'w13')
        with self.assertRaises(ValueError): expert_geometry(0,'unknown')

    def test_valid_words(self):
        for layer in (0,20):
            for proj in ('w13','w2'):
                w=generate(layer,proj)
                self.assertEqual(validate_expert_words(layer,proj,w,64,32),
                                 len(w.codes)+len(w.scales)+(8 if layer<20 else 0))

    def test_invalid_quant_words_and_scalars(self):
        for layer in (0,20):
            with self.assertRaises(ValueError):
                validate_expert_words(layer,'w13',generate(layer,'w13',bad=True),64,32)
        with self.assertRaises(ValueError):
            validate_expert_words(0,'w13',Words(b'abc',b'def'),64,32)
        w=generate(0,'w13');w.amax=-1
        with self.assertRaises(ValueError): validate_expert_words(0,'w13',w,64,32)


class TestLRU(unittest.TestCase):
    def new_store(self):
        import torch
        # Bypass CUDA-only init for *CPU-only cache policy* unit testing.
        obj=object.__new__(QuantizedGPUStore)
        obj.reader=FakeReader();obj.device=torch.device('cpu')
        obj.geometry=type('G',(),dict(hidden=64,intermediate=32,experts=16))()
        obj.items=OrderedDict();obj.used_bytes=0
        obj.budget_bytes=2*(validate_expert_words(0,'w13',generate(0,'w13'),64,32)
                            +validate_expert_words(0,'w2',generate(0,'w2'),64,32))
        obj.stats=dict(hits=0,misses=0,evictions=0,bank_reads=0,bank_read_bytes=0,uploads=0,read_seconds=0.)
        return obj

    def test_lru_hits_and_evictions(self):
        store=self.new_store()
        first=store.select(0,[2,4]);self.assertEqual(len(first),2)
        self.assertEqual(store.stats['bank_reads'],2)
        again=store.select(0,[4,2]);self.assertIs(first[0],again[1])
        self.assertEqual(store.stats['hits'],2)
        store.select(0,[6,7]);self.assertEqual(store.stats['evictions'],2)
        self.assertEqual(set(store.items),{(0,6),(0,7)})
        self.assertLessEqual(store.used_bytes,store.budget_bytes)
        store.release();self.assertEqual(store.used_bytes,0)

    def test_protected_current_selection(self):
        store=self.new_store()
        store.select(0,[0,1])
        with self.assertRaisesRegex(ValueError,'selected top-k'):
            store.select(0,[0,1,2])

    def test_dupes_and_expert_ids(self):
        store=self.new_store()
        for ids in ([2,2],[-1],[16],[]):
            with self.assertRaises(ValueError):store.select(0,ids)

    def test_invalid_budget(self):
        import torch
        with self.assertRaises(ValueError):
            QuantizedGPUStore(None,'cuda:0',None,budget_mib=0)
        with self.assertRaises(ValueError):
            QuantizedGPUStore(None,'cpu',None,budget_mib=4)


class TestCorrectedNativeSource(unittest.TestCase):
    def test_w13_cuda_interleave_contract(self):
        source=Path(__file__).resolve().parents[2]/'src/models/inkling2/moe_top8_stage5w.cu'
        # tests run from worktree; verify corrected native code really differs.
        if not source.is_file(): self.skipTest('source outside installed test tree')
        text=source.read_text()
        self.assertIn('const float gate=up[route*2*I+2*neuron];',text)
        self.assertIn('const float value=up[route*2*I+2*neuron+1];',text)
        self.assertNotIn('const float gate=up[route*2*I+neuron];',text)


if __name__=='__main__':unittest.main()
