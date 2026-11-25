"""Host-only checks for the new packing assumptions and wrapper ABI (not CUDA execution)."""
import ctypes
import math
import random
import struct
import unittest
from unittest.mock import patch
from types import SimpleNamespace
from native_stage7ab import FastMoE


def unpack_nvfp4(words):
    return [(-1 if n&8 else 1)*((0xc8643210>>(4*(n&7)))&15)*.5
            for w in words for n in ((w>>(4*i))&15 for i in range(8))]


class Stage7ABHost(unittest.TestCase):
    def test_nvfp4_packed_word(self):
        self.assertEqual(unpack_nvfp4((0x76543210,0xfedcba98)),[0,.5,1,1.5,2,3,4,6,0,-.5,-1,-1.5,-2,-3,-4,-6])
    def test_mxfp8_word_order(self):
        codes=bytes(range(32))
        words=struct.unpack('<8I',codes)
        decoded=bytes((w>>(8*i))&255 for w in words for i in range(4))
        self.assertEqual(decoded,codes)
    def test_source_group_offsets_alignment(self):
        for cols in (2304,1152):
            for g,bytes_per_group in ((16,8),(32,32)):
                stride=cols//g*bytes_per_group
                for row in (0,1,2,1151,2303):
                    self.assertEqual(row*stride%bytes_per_group,0)
                    self.assertEqual(row*stride+(cols//g-1)*bytes_per_group+bytes_per_group,(row+1)*stride)
    def test_group_scaling_equivalence(self):
        rng=random.Random(17)
        for g in (16,32):
            for _ in range(20):
                vals=[rng.uniform(-6,6) for _ in range(g)]
                x=[rng.uniform(-2,2) for _ in range(g)]
                scale=rng.uniform(.01,20)
                ref=sum(w*scale*a for w,a in zip(vals,x))
                actual=sum(w*a for w,a in zip(vals,x))*scale
                self.assertTrue(math.isclose(ref,actual,rel_tol=1e-12,abs_tol=1e-10))
    def test_ctypes_registration_and_close(self):
        class Fn:
            def __init__(self,value=None): self.value=value;self.calls=[]
            def __call__(self,*a): self.calls.append(a);return self.value
        lib=SimpleNamespace(**{'inkling7ab_'+n:Fn(123 if n=='create' else None) for n in ('create','destroy','run','routes','reference_routes')})
        with patch('native_stage7ab.C.CDLL',return_value=lib):
            f=FastMoE(SimpleNamespace(),'fake.so')
            self.assertEqual(f.handle,123)
            self.assertEqual(lib.inkling7ab_run.argtypes,[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_int,ctypes.c_uint64,ctypes.c_uint64,ctypes.c_uint64,ctypes.c_float,ctypes.c_uint64,ctypes.c_uint64])
            f.close();f.close()
            self.assertEqual(lib.inkling7ab_destroy.calls,[(123,)])
    def test_status_propagation(self):
        FastMoE.check(0,'test')
        with self.assertRaisesRegex(RuntimeError,'CUDA/result 2'):
            FastMoE.check(2,'test')

if __name__=='__main__': unittest.main()
