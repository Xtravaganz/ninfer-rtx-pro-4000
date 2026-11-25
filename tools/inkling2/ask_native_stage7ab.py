#!/usr/bin/env python3
"""Opt-in full Torch-controlled 48-visit forward with the fast 7A/B MoE.
Not a native model executor. The original Stage6J/K/L runner is unchanged.
"""
from __future__ import annotations
import os
import sys
from pathlib import Path


def main():
    import torch
    import native_stage6jkl as stage6
    import ask_native_stage6jkl as runner
    from native_stage7ab import FastMoE
    if '--graph' in sys.argv:
        raise SystemExit('Stage7AB full-forward runner requires graph=0; use the isolated graph test first')
    path=Path(os.environ.get('INKLING_STAGE7AB_LIBRARY','/artifacts/native7ab/libinkling_native_stage7ab.so'))
    if not path.is_file(): raise SystemExit(f'Missing Stage7AB DSO: {path}')
    if not torch.cuda.is_available(): raise SystemExit('CUDA required')
    torch.cuda.set_device(0)
    free,total=torch.cuda.mem_get_info(0)
    print(f'STAGE7AB VRAM BEFORE PRELOAD: free_MiB={free/1048576:.1f} total_MiB={total/1048576:.1f}',flush=True)
    original=stage6.PreloadedResidentWeights

    class FastWeights(original):
        def __init__(self,*args,**kwargs):
            self.fast=None
            super().__init__(*args,**kwargs)
            try:
                self.fast=FastMoE(self.native,path)
                def fast_run(x,router,bias,scale,layer,out,stream):
                    self.fast.run(x,router,bias,scale,layer,out,stream)
                    self.native.calls_single+=1
                self.native.run=fast_run
            except BaseException:
                self.release();raise
        def metrics(self):
            result=super().metrics()
            result['native_stage7ab_calls']=self.fast.calls if self.fast is not None else 0
            return result
        def release(self):
            if self.fast is not None:
                torch.cuda.synchronize(self.device)
                self.fast.close();self.fast=None
            super().release()

    stage6.PreloadedResidentWeights=FastWeights
    try:
        print('STAGE7AB: opt-in fast NVFP4/MXFP8 single-token MoE; Torch 48-visit executor; graph=0',flush=True)
        return runner.main()
    finally:
        stage6.PreloadedResidentWeights=original

if __name__=='__main__': raise SystemExit(main())
