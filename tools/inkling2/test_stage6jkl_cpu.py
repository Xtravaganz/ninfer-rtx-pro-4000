"""Stage6J/K/L CPU-only contract regressions (CUDA parity runs separately)."""
from __future__ import annotations
import unittest
from unittest.mock import patch

import torch
import native_stage6jkl as tested
from streaming_stage5q import IncrementalCache48


class FakeConvState:
    """Independent CPU simulation of the native C++ channel-owned history."""
    def __init__(self,libpath):
        self.history={}

    def apply(self,slot,path,x,taps,out,*,dynamic=False):
        c=x.shape[1]
        old=self.history.get((slot,path))
        if old is None:old=torch.zeros(3,c,dtype=torch.float32)
        result=[]
        for t in range(x.shape[0]):
            w=taps[t] if dynamic else taps
            value=torch.cat([old,x[t:t+1]],dim=0)
            if dynamic:
                y=(value.flip(0).T*w).sum(-1)
                result.append((y+x[t])/3.)
            else:
                y=(value.T*w).sum(-1)
                result.append((y+x[t])/(5.**.5))
            old=value[1:]
        out.copy_(torch.stack(result))
        self.history[(slot,path)]=old

    def close(self):pass


class Contracts(unittest.TestCase):
    def test_gpu_zero_only(self):
        with self.assertRaisesRegex(ValueError,'cuda:0'):
            tested.require_device0('cuda:1')
        with self.assertRaisesRegex(ValueError,'cuda:0'):
            tested.require_device0('cpu')

    def test_static_cache_parity_for_multiple_positions(self):
        channels=64
        torch.manual_seed(100)
        taps=torch.randn(channels,4)*.15
        with patch.object(tested,'ConvState',FakeConvState):
            implementation=tested.make_native_conv_cache('mock.so')(kernel=4)
        reference=IncrementalCache48(kernel=4)
        try:
            for t in range(5):
                x=torch.randn(1,1,channels)*.3
                # Directly set the active position; end() expects all 48 slots,
                # so a single-slot unit test cannot call it.
                reference._active_position=t
                implementation._active_position=t
                reference._conv_visited.clear()
                implementation._conv_visited.clear()
                original=reference.static_conv(13,2,x,taps)
                optimized=implementation.static_conv(13,2,x,taps)
                self.assertTrue(torch.allclose(original,optimized,atol=2e-6,rtol=2e-6))
                with self.assertRaises(ValueError):
                    implementation.static_conv(13,2,x,taps)
        finally:
            implementation.close()

    def test_dynamic_cache_parity_for_multiple_positions(self):
        channels=32;heads=4;d=64
        torch.manual_seed(123)
        taps=torch.randn(4,channels)*.15
        projection=torch.randn(heads*4,d)*.01
        with patch.object(tested,'ConvState',FakeConvState):
            implementation=tested.make_native_conv_cache('mock.so')(kernel=4)
        reference=IncrementalCache48(kernel=4)
        try:
            for t in range(5):
                x=torch.randn(1,1,channels)*.3
                cond=torch.randn(1,1,d)*.3
                reference._active_position=t
                implementation._active_position=t
                reference._conv_visited.clear()
                implementation._conv_visited.clear()
                original=reference.dynamic_conv(17,0,x,cond,taps,projection,heads=heads)
                optimized=implementation.dynamic_conv(17,0,x,cond,taps,projection,heads=heads)
                self.assertTrue(torch.allclose(original,optimized,atol=2e-6,rtol=2e-6))
        finally:
            implementation.close()

    def test_invalid_geometry_fails_closed(self):
        with patch.object(tested,'ConvState',FakeConvState):
            with self.assertRaisesRegex(ValueError,'K=4'):
                tested.make_native_conv_cache('mock.so')(kernel=3)


if __name__=='__main__':unittest.main()
