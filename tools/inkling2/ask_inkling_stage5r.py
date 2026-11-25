#!/usr/bin/env python3
"""Stage 5R: interactive *experimental* text prompt on the existing Stage 5Q runtime.

No official HF full-model load; no native ninfer-serve implementation.
This script runs the existing PyTorch NInfer-v3 student one position at a time.
A plausible answer DOES NOT establish agreement with the published model.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for location in (HERE, ROOT):
    if str(location) not in sys.path:
        sys.path.insert(0, str(location))


def _ids_from_tokenizer(value) -> list[int]:
    """Normalize tokenizer output, but never coerce invalid token IDs silently.

    Fast tokenizers may return list[int], a one-batch list[list[int]],
    BatchEncoding/mapping with input_ids, or an array-like object. If the
    structure is unfamiliar, fail with its type instead of a generic message.
    """
    from collections.abc import Mapping
    from numbers import Integral

    if isinstance(value, Mapping):
        if 'input_ids' not in value:
            raise ValueError(
                f'tokenizer returned {type(value).__name__} without input_ids; '
                f'keys={list(value)[:12]!r}'
            )
        value = value['input_ids']
    if not isinstance(value, (list, tuple)) and hasattr(value, 'tolist'):
        value = value.tolist()
    if (isinstance(value, (tuple, list)) and len(value) == 1
            and isinstance(value[0], (list, tuple))):
        value = value[0]
    if not isinstance(value, (list, tuple)):
        raise ValueError(
            f'tokenizer returned unsupported input_ids type '
            f'{type(value).__name__}; expected a sequence of integers'
        )
    if not value:
        raise ValueError('tokenizer returned empty input_ids')
    result = []
    for i, token in enumerate(value):
        if isinstance(token, bool) or not isinstance(token, Integral):
            raise ValueError(
                f'tokenizer input_ids[{i}] has type '
                f'{type(token).__name__}, value={repr(token)[:160]}; expected int'
            )
        token = int(token)
        if not 0 <= token < 201024:
            raise ValueError(
                f'invalid tokenizer input_ids[{i}]={token} outside embedding rows '
                '[0, 201024); cannot feed this ID to Inkling 2 Mini'
            )
        result.append(token)
    return result


def tokenize_question(tokenizer, question: str, *, raw: bool = False) -> list[int]:
    """Use official chat template unless --raw explicitly requested."""
    if not isinstance(question, str) or not question.strip():
        raise ValueError('question must not be empty')
    if raw:
        result = tokenizer.encode(question, add_special_tokens=False)
    else:
        result = tokenizer.apply_chat_template(
            [{'role': 'user', 'content': question}],
            tokenize=True, add_generation_prompt=True,
        )
    return _ids_from_tokenizer(result)


def stop_ids(tokenizer, text_config: dict) -> set[int]:
    result: set[int] = set()
    for source in (getattr(tokenizer, 'eos_token_id', None), text_config.get('eos_token_id')):
        if isinstance(source, int) and not isinstance(source, bool):
            result.add(source)
        elif isinstance(source, (tuple, list)):
            result.update(v for v in source if isinstance(v, int) and not isinstance(v, bool))
    return result


def render_tokens(tokenizer, ids: list[int]) -> str:
    return tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)


def run(args) -> int:
    from transformers import AutoTokenizer

    if not args.source.is_dir():
        raise ValueError(f'model snapshot directory missing: {args.source}')
    tokenizer = AutoTokenizer.from_pretrained(
        str(args.source), local_files_only=True, trust_remote_code=False,
        use_fast=True,
    )
    ids = tokenize_question(tokenizer, args.question, raw=args.raw)
    if len(ids) > args.max_prompt_tokens:
        raise ValueError(
            f'prompt contains {len(ids)} tokens, limit is {args.max_prompt_tokens}; '
            'shorten question or increase --max-prompt-tokens (higher cost)'
        )
    print(f'PROMPT token_count={len(ids)} mode={"raw" if args.raw else "chat-template"}; IDs={ids}', flush=True)
    if args.tokenize_only:
        print('TOKENIZE ONLY: no model weights loaded, no inference', flush=True)
        return 0

    import torch
    from tools.artifact.reader import Artifact
    from nonexpert_artifact import Checkpoint
    from real_carrier_stage5f import Carrier5F, validate_complete_carriers
    from real_sliding_stage5h import Geometry, TorchCarrierWeights
    from real_postlude_stage5no import streaming_unembed_topk
    from run_real_forward_stage5no import full_forward_preflight
    from streaming_stage5q import IncrementalCache48, forward_one_token

    if args.device.startswith('cuda') and not torch.cuda.is_available():
        raise ValueError('CUDA requested but torch.cuda.is_available() is false')
    print('STAGE5R: experimental NInfer-v3 Torch model (NOT HF-validated or ninfer-serve)', flush=True)
    with Artifact(args.nonexpert) as non, Artifact(args.experts) as exp:
        validate_complete_carriers(args.source, non, exp, args.banks)
        with Checkpoint(args.source) as checkpoint:
            full_forward_preflight(checkpoint.spans, checkpoint.config, 'greedy')
            config = checkpoint.config
        g = Geometry.from_hf_config(config)
        text = config['text_config']
        w = TorchCarrierWeights(Carrier5F(non, exp), args.device, g)
        cache = IncrementalCache48(kernel=g.conv_kernel)
        forward_args = {
            'gate_rank': text['loop_config']['gate_rank'],
            'log_scaling_alpha': text['log_scaling_alpha'],
            'log_scaling_n_floor': text['log_scaling_n_floor'],
        }
        eos = stop_ids(tokenizer, text)
        generated: list[int] = []
        with torch.inference_mode():
            # Prefill for the entire question is deliberately sequential and expensive.
            for i, token in enumerate(ids):
                hidden, trace = forward_one_token(w, cache, int(token), **forward_args)
                if len(trace) != 48:
                    raise ValueError(f'position {i}: expected 48 visits, got {len(trace)}')
                w.clear()
                print(f'PREFILL {i + 1}/{len(ids)}: token={token} slots={len(cache.kv)}', flush=True)
            for step in range(args.max_new_tokens):
                top = streaming_unembed_topk(
                    w.reader, hidden,
                    unpadded_vocab=text.get('unpadded_vocab_size', 200058),
                    softcap=text.get('final_logit_softcapping', 30.0), topk=5,
                )
                if not top or type(top[0]['token_id']) is not int:
                    raise ValueError('unembedding returned no valid token ID')
                next_token = top[0]['token_id']
                generated.append(next_token)
                print(f'GEN {step + 1}/{args.max_new_tokens}: id={next_token} '
                      f'logit={top[0]["logit"]:.6g} '
                      f'text={render_tokens(tokenizer, generated)!r}', flush=True)
                if next_token in eos:
                    print(f'STOP: EOS token {next_token}', flush=True)
                    break
                if step + 1 < args.max_new_tokens:
                    w.clear()
                    hidden, trace = forward_one_token(w, cache, next_token, **forward_args)
                    if len(trace) != 48:
                        raise ValueError(f'generated position {cache.next_position-1}: incomplete forward')
                    w.clear()
        print('\nFRAGE:', args.question)
        print('ANTWORT (EXPERIMENTELL):', render_tokens(tokenizer, generated))
        print('GENERATED IDS:', generated)
        print('NOT VALIDATED: official HF parity, output quality, native NInfer runtime')
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('source', 'nonexpert', 'experts', 'banks'):
        p.add_argument('--' + key, type=Path, required=True)
    p.add_argument('--question', required=True)
    p.add_argument('--raw', action='store_true', help='no chat template; raw plain text prompt')
    p.add_argument('--tokenize-only', action='store_true', help='inspect token IDs, do not load weights')
    p.add_argument('--max-new-tokens', type=int, default=12)
    p.add_argument('--max-prompt-tokens', type=int, default=64)
    p.add_argument('--device', default='cuda:0')
    args = p.parse_args(argv)
    if not (1 <= args.max_new_tokens <= 64 and 1 <= args.max_prompt_tokens <= 256):
        p.error('token limits must satisfy 1<=new<=64 and 1<=prompt<=256')
    try:
        return run(args)
    except (ValueError, RuntimeError, OSError, TypeError, KeyError, ImportError) as err:
        print('ERROR:', err, file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
