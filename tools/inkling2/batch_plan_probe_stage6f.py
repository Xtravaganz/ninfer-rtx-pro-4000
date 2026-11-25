#!/usr/bin/env python3
"""Read the native 48-step batch DISPATCH PLAN (not a forward-model benchmark)."""
from __future__ import annotations
import argparse
import ctypes
import json
from pathlib import Path

class Visit(ctypes.Structure):
    _fields_ = [(x,ctypes.c_uint32) for x in (
        'ordinal','physical_layer','cache_slot','core_pass','stage',
        'block_boundary','reads_messages','reset_to_prelude','tokens')]


def inspect(path, token_count):
    if not 1<=token_count<=4096: raise ValueError('token_count must be 1..4096')
    lib=ctypes.CDLL(str(Path(path).resolve()))
    lib.inkling6f_plan.argtypes=[ctypes.POINTER(Visit),ctypes.c_uint32,ctypes.c_uint32]
    lib.inkling6f_plan.restype=ctypes.c_int
    visits=(Visit*48)()
    result=lib.inkling6f_plan(visits,48,token_count)
    if result!=48:raise RuntimeError(f'invalid Stage6F schedule ({result})')
    steps=list(visits)
    assert sorted(x.cache_slot for x in steps)==list(range(48))
    assert sum(x.reads_messages for x in steps)==24
    assert sum(x.reset_to_prelude for x in steps)==2
    return {
        'logical_visits': len(steps),
        'cache_slots': len({x.cache_slot for x in steps}),
        'cross_pass_message_visits':sum(x.reads_messages for x in steps),
        'prelude_resets':sum(x.reset_to_prelude for x in steps),
        'full_attention_visits':sum(x.physical_layer%6==5 for x in steps),
        'tokens_in_batch':token_count,
        'sequential_python_visits_for_same_tokens':48*token_count,
        'potential_native_batch_dispatch_visits':48,
        'execute_full_forward':False,
        'batch_moe_integrated':False,
        'measured_speedup':None,
    }


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--library',type=Path,required=True)
    p.add_argument('--tokens',type=int,default=16)
    a=p.parse_args()
    print('STAGE6F NATIVE BATCH PLAN:',json.dumps(inspect(a.library,a.tokens),sort_keys=True))
    print('PLAN ONLY: C++ dispatch is NOT a full NInfer Inkling executor; no model tokens computed')

if __name__=='__main__':main()
