#!/usr/bin/env python3
"""Stage6H native fully resident expert MoE + experimental Torch full forward. Preserves installed Stage5R/5Q files.

Modes: 'baseline' means Stage5S corrected W13 but no persistent weights;
'cached' means *same W13 order* with bounded nonexpert and expert FP32 GPU LRU.
This is still sequential Torch prefill, NOT a parallel prefill kernel.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import statistics
import sys
from time import perf_counter

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for location in (HERE, ROOT):
    if str(location) not in sys.path:
        sys.path.insert(0, str(location))


def run(args) -> int:
    import torch
    import real_sliding_stage5h as source_mod
    import streaming_stage5q as stream_mod
    import ask_inkling_stage5r as ask_mod
    from gateup_audit_stage5st import corrected_moe
    from native_resident_stage6ghi import ResidentCarrierWeights

    original_class = source_mod.TorchCarrierWeights
    original_forward = stream_mod.forward_one_token
    original_tokenize = ask_mod.tokenize_question
    state = {'prompt_tokens': None, 'weights': None, 'timings': []}

    class NativeWeights(ResidentCarrierWeights):
        def __init__(self, reader, device, geometry):
            super().__init__(reader, device, geometry,
                             library_path=args.library,
                             nonexpert_mib=args.nonexpert_cache_mib,
                             verify_first=args.verify_first)
            state['weights'] = self

    def capture_prompt(*items, **kwargs):
        ids = original_tokenize(*items, **kwargs)
        state['prompt_tokens'] = len(ids)
        return ids

    def timed_forward(weights, cache, token, **kwargs):
        index = len(state['timings']) + 1
        if torch.cuda.is_available() and weights.device.type == 'cuda':
            torch.cuda.synchronize(weights.device)
        start = perf_counter()
        output = original_forward(weights, cache, token, **kwargs)
        if torch.cuda.is_available() and weights.device.type == 'cuda':
            torch.cuda.synchronize(weights.device)
        seconds = perf_counter() - start
        phase = 'PREFILL' if index <= (state['prompt_tokens'] or 0) else 'DECODE'
        state['timings'].append((phase, seconds))
        if (args.per_token_profile or index == 1 or index == state['prompt_tokens'] or
                index % 4 == 0):
            print(f'PERF {phase} position={index-1} seconds={seconds:.3f} '
                  f'cache_mode=native_resident', flush=True)
        return output

    settings = argparse.Namespace(
        source=args.source, nonexpert=args.nonexpert, experts=args.experts,
        banks=args.banks, question=args.question, raw=args.raw,
        tokenize_only=args.tokenize_only,
        max_prompt_tokens=args.max_prompt_tokens,
        max_new_tokens=args.max_new_tokens, device=args.device,
    )
    result = 1
    source_mod.TorchCarrierWeights = NativeWeights
    stream_mod.forward_one_token = timed_forward
    ask_mod.tokenize_question = capture_prompt
    try:
        print(f'STAGE6H: mode=native_resident, math=HF-W13-interleave, '
              f'nonexpert_cache={args.nonexpert_cache_mib}MiB, '
              f'fully_resident_experts=True, '
              f'still_sequential_prefill=True', flush=True)
        result = ask_mod.run(settings)
        return int(result or 0)
    finally:
        # Never leave opt-in patches installed in a caller's Python process.
        source_mod.TorchCarrierWeights = original_class
        stream_mod.forward_one_token = original_forward
        ask_mod.tokenize_question = original_tokenize
        times = state['timings']
        p = [s for phase, s in times if phase == 'PREFILL']
        d = [s for phase, s in times if phase == 'DECODE']
        def summary(a):
            return ({'calls': len(a), 'total_s': round(sum(a), 3),
                     'median_s': round(statistics.median(a), 3),
                     'last_s': round(a[-1], 3)} if a else None)
        stats = {'mode': 'native_resident', 'prompt_tokens': state['prompt_tokens'],
                 'prefill_forward': summary(p), 'decode_forward': summary(d),
                 'note': ('Entire token timed with CUDA sync; NO sync inside native MoE; cold bank loading included; excludes tokenization, '
                          'logit scanning and Docker startup; first-token cold start included')}
        w = state['weights']
        if hasattr(w, 'metrics'):
            stats['weight_cache'] = w.metrics()
        if torch.cuda.is_available() and w is not None and w.device.type == 'cuda':
            stats['peak_torch_allocated_mib'] = round(torch.cuda.max_memory_allocated(w.device)/1048576, 2)
        print('PERF SUMMARY:', json.dumps(stats, sort_keys=True), flush=True)
        if w is not None and hasattr(w, 'release'):
            w.release()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source','nonexpert','experts','banks'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--library', type=Path, required=True)
    parser.add_argument('--verify-first', action='store_true')
    parser.add_argument('--question', required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--max-new-tokens', type=int, default=3)
    parser.add_argument('--max-prompt-tokens', type=int, default=64)
    parser.add_argument('--nonexpert-cache-mib', type=int, default=2048)
    parser.add_argument('--raw', action='store_true')
    parser.add_argument('--tokenize-only', action='store_true')
    parser.add_argument('--per-token-profile', action='store_true')
    args = parser.parse_args(argv)
    if (not 1 <= args.max_new_tokens <= 64 or not 1 <= args.max_prompt_tokens <= 512 or
            not isinstance(args.question, str) or not args.question.strip() or
            not (0 <= args.nonexpert_cache_mib <= 16384 )):
        parser.error('invalid token count, question or cache budget')
    try:
        return run(args)
    except (OSError, ValueError, TypeError, RuntimeError, ImportError, KeyError) as exc:
        print('ERROR Stage6H:', exc, file=sys.stderr)
        return 1

if __name__ == '__main__':
    raise SystemExit(main())
