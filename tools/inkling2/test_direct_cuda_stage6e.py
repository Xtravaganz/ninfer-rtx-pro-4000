"""On-GPU parity: new direct pointer kernel vs corrected Stage5W packed kernel.

Must run inside user's CUDA+PyTorch Docker with Stage5W and Stage6E libraries.
Does not modify original weights. Includes missing-expert failure semantics.
"""
from pathlib import Path
import os
import sys
import unittest
import numpy as np
import torch

sys.path.insert(0,str(Path(__file__).resolve().parent))
from direct_experts_stage6e import NativeLibrary6E
from native_moe_stage5w import NativeLibrary5W


class TestDirectCUDA(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise RuntimeError('Stage6E GPU parity requires real CUDA device')
        cls.device=torch.device('cuda:0')
        cls.old=NativeLibrary5W(os.environ['INKLING_STAGE5W_LIB'])
        cls.new=NativeLibrary6E(os.environ['INKLING_STAGE6E_LIB'])

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls,'old'):cls.old.close()
        if hasattr(cls,'new'):cls.new.close()

    def exercise(self,fmt):
        # Router selects physical expert IDs 0..7 in deterministic order.
        device=self.device
        router=torch.zeros((128,2304),dtype=torch.float32,device=device)
        bias=torch.full((128,),-10.0,dtype=torch.float32,device=device)
        bias[:8]=torch.arange(8,0,-1,dtype=torch.float32,device=device)
        x=(torch.arange(2304,device=device,dtype=torch.float32)%13-6)/200
        multiplier=2 if fmt==1 else 1
        group=16 if fmt==1 else 32
        if fmt==1:
            w13_codes=torch.full((8,2304,2304//multiplier),0x12,dtype=torch.uint8,device=device)
            w2_codes=torch.full((8,2304,1152//multiplier),0x21,dtype=torch.uint8,device=device)
            scale_byte=0x38
        else:
            w13_codes=torch.full((8,2304,2304),0x38,dtype=torch.uint8,device=device)
            w2_codes=torch.full((8,2304,1152),0x30,dtype=torch.uint8,device=device)
            scale_byte=127
        w13_scales=torch.full((8,2304,2304//group),scale_byte,dtype=torch.uint8,device=device)
        w2_scales=torch.full((8,2304,1152//group),scale_byte,dtype=torch.uint8,device=device)
        amax=torch.ones((8,),dtype=torch.float32,device=device)
        den=torch.ones_like(amax)
        mapping=torch.full((128,),-1,dtype=torch.int32,device=device)
        mapping[:8]=torch.arange(8,dtype=torch.int32,device=device)
        pointers=np.zeros((4,128),dtype=np.int64)
        for i in range(8):
            pointers[:,i]=[w13_codes[i].data_ptr(), w13_scales[i].data_ptr(),
                            w2_codes[i].data_ptr(), w2_scales[i].data_ptr()]
        pointers_gpu=torch.from_numpy(pointers).to(device)
        factors_gpu=torch.ones((4,128),dtype=torch.float32,device=device)
        expected=torch.empty((2304,),dtype=torch.float32,device=device)
        got=torch.empty_like(expected)
        stream=torch.cuda.current_stream(device)
        self.old.run(x,router,bias,8.,fmt,8,
                     (w13_codes,w13_scales,amax if fmt==1 else None,den if fmt==1 else None),
                     (w2_codes,w2_scales,amax if fmt==1 else None,den if fmt==1 else None),
                     mapping,expected,stream.cuda_stream)
        self.new.run(x,router,bias,8.,fmt,
                     type('T',(),{'pointers':pointers_gpu,'factors':factors_gpu,'valid':mapping}),
                     got,stream.cuda_stream)
        stream.synchronize()
        self.assertTrue(bool(torch.isfinite(got).all()))
        maxabs=float((expected-got).abs().max().item())
        self.assertTrue(torch.allclose(expected,got,rtol=1e-6,atol=1e-5),f'fmt={fmt} maxabs={maxabs}')
        print(f'PASS Stage6E fmt={fmt} native direct-pointer vs native Stage5W max_abs={maxabs:.9g}')
        # A missing top-8 physical expert must result in NaNs, not an invalid dereference.
        bad=mapping.clone();bad[0]=-1
        self.new.run(x,router,bias,8.,fmt,
                     type('T',(),{'pointers':pointers_gpu,'factors':factors_gpu,'valid':bad}),
                     got,stream.cuda_stream)
        stream.synchronize()
        self.assertTrue(bool(torch.isnan(got).all()))

    def test_nvfp4_direct_vs_stage5w(self):self.exercise(1)
    def test_mxfp8_direct_vs_stage5w(self):self.exercise(2)


if __name__=='__main__':unittest.main()
