"""CPU contract tests for Stage6E direct resident pointer directory."""
from collections import OrderedDict, defaultdict
from dataclasses import dataclass
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from direct_experts_stage6e import build_host_tables, DirectEncodedGPUStore


class FakeTensor:
    def __init__(self, addr): self.addr=addr
    def data_ptr(self): return self.addr


@dataclass
class FakeEntry:
    w13_codes: object
    w13_scales: object
    w2_codes: object
    w2_scales: object
    w13_amax: float = 2.0
    w13_denominator: float = 4.0
    w2_amax: float = 3.0
    w2_denominator: float = 4.0
    byte_count: int = 4096


def entry(e):
    return FakeEntry(*(FakeTensor(0x10000+e*0x100+i*0x20) for i in range(4)))


class FakeGPUCache:
    def __init__(self, capacity=10):
        self.items=OrderedDict()
        self.capacity=capacity
        self.selected=[]
    def select(self, layer, ids):
        selected=set((layer,e) for e in ids)
        for eid in ids:
            key=(layer,eid)
            if key not in self.items:
                while len(self.items)>=self.capacity:
                    victim=next((k for k in self.items if k not in selected),None)
                    if victim is None: raise ValueError('protected capacity exceeded')
                    self.items.pop(victim)
                self.items[key]=entry(layer*128+eid)
            self.items.move_to_end(key)
        return [self.items[(layer,eid)] for eid in ids]
    def release(self):self.items.clear()
    def metrics(self):return {'resident':len(self.items)}


class TestDirectTables(unittest.TestCase):
    def test_tables_are_physical_expert_indices(self):
        items=OrderedDict([((0,91),entry(91)),((0,3),entry(3)),((1,91),entry(219))])
        ptr,factor,valid=build_host_tables(items,0)
        self.assertEqual(ptr.shape,(4,128))
        self.assertEqual(valid[91],91)
        self.assertEqual(valid[3],3)
        self.assertEqual(valid[0],-1)
        self.assertEqual(ptr[0,91],entry(91).w13_codes.addr)
        self.assertEqual(ptr[1,91],entry(91).w13_scales.addr)
        self.assertEqual(ptr[2,3],entry(3).w2_codes.addr)
        self.assertEqual(ptr[3,3],entry(3).w2_scales.addr)
        self.assertEqual(ptr[0,0],0)
        self.assertEqual(factor[0,91]/factor[1,91],0.5)
        self.assertEqual(factor[2,3]/factor[3,3],0.75)
    def test_bad_factor_and_address_rejected(self):
        a=entry(0);a.w2_codes=FakeTensor(0)
        with self.assertRaisesRegex(ValueError,'address'):build_host_tables({(0,0):a},0)
        a=entry(0);a.w13_denominator=0
        with self.assertRaisesRegex(ValueError,'divisor'):build_host_tables({(0,0):a},0)
    def test_wrong_model_geometry_rejected(self):
        for l in (-1,24,1.0):
            with self.assertRaises(ValueError):build_host_tables({},l)
        with self.assertRaises(ValueError):build_host_tables({},0,127)
    def test_maintain_tables_and_invalidate_on_evict(self):
        import torch
        d=DirectEncodedGPUStore.__new__(DirectEncodedGPUStore)
        d.cache=FakeGPUCache(capacity=9)
        d.device=torch.device('cpu')
        d.generation=defaultdict(int)
        d.tables={}
        d.table_uploads=d.table_hits=d.pointer_bytes=d.avoided_gather_bytes=0
        ids=list(range(8))
        first=d.select_table(0,ids)
        self.assertEqual(d.table_uploads,1)
        self.assertEqual(first.generation,8)
        self.assertEqual(first.valid[:8].tolist(),ids)
        self.assertEqual(first.pointers.shape,(4,128))
        second=d.select_table(0,ids)
        self.assertIs(first,second)
        self.assertEqual(d.table_hits,1)
        # Force one eviction from another layer, then reload layer 0's directory.
        d.select_table(1,ids)
        again=d.select_table(0,ids)
        self.assertIsNot(first,again)
        self.assertTrue(all(int(again.valid[i])==i for i in ids))
        self.assertEqual(d.avoided_gather_bytes,32*4096)
        d.release()
        self.assertEqual(len(d.tables),0)
    def test_selected_ids_contract(self):
        import torch
        d=DirectEncodedGPUStore.__new__(DirectEncodedGPUStore)
        d.cache=FakeGPUCache()
        d.device=torch.device('cpu')
        d.generation=defaultdict(int)
        d.tables={}
        d.table_uploads=d.table_hits=d.pointer_bytes=d.avoided_gather_bytes=0
        with self.assertRaises(ValueError):d.select_table(0,[0,1])
        with self.assertRaises(ValueError):d.select_table(0,[0,0,1,2,3,4,5,6])


if __name__ == '__main__': unittest.main()
