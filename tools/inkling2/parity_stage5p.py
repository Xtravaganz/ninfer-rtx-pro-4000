#!/usr/bin/env python3
"""Stage5P: capture official Transformers vs NInfer-v3 Torch traces, compare honestly.

Modes:
    student --source ... --nonexpert ... --experts ... --banks ... --output student.npz
    hf      --source ... --output teacher.npz [--device-map auto]
    compare --student student.npz --teacher teacher.npz --output parity.json

The official-HF run is optional, heavy, and *not* simulated. On hardware where
its dequantized model cannot be loaded, the result is INCONCLUSIVE, not PASS.
No vLLM/ninfer native runtime is claimed. No weights are copied to the ZIP.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import dataclass
from hashlib import sha256
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for base in (HERE, ROOT):
    if str(base) not in sys.path:
        sys.path.insert(0, str(base))


def execution_layers() -> list[int]:
    return [*range(18), *range(6, 18), *range(6, 18), *range(18, 24)]


def execution_slots() -> list[int]:
    return [*range(18), *range(24, 48), *range(18, 24)]


def _flatten_hidden(value: Any) -> np.ndarray:
    import torch
    if isinstance(value, (tuple, list)):
        value = value[0]
    if not isinstance(value, torch.Tensor):
        raise ValueError('HF/Student layer hook produced a non-tensor')
    if value.shape != (1, 1, 2304):
        raise ValueError(f'expected [1,1,2304] layer hidden, got {tuple(value.shape)}')
    result = value.detach().float().cpu().contiguous().reshape(-1).numpy().copy()
    if not np.isfinite(result).all():
        raise ValueError('nonfinite reference/stage layer output')
    return result


@dataclass
class TraceRecorder:
    layers: list[int]
    values: list[np.ndarray]

    def add(self, layer: int, hidden: Any) -> None:
        if len(self.layers) >= 48:
            raise ValueError('too many layer visits')
        expected = execution_layers()[len(self.layers)]
        if layer != expected:
            raise ValueError(f'layer visit #{len(self.layers)} expected {expected}, got {layer}')
        value = _flatten_hidden(hidden)
        self.layers.append(layer)
        self.values.append(value)

    def matrix(self) -> np.ndarray:
        if self.layers != execution_layers():
            raise ValueError(f'expected exactly 48 visits, got {self.layers}')
        result = np.stack(self.values, axis=0)
        if result.shape != (48, 2304):
            raise ValueError('invalid trace matrix')
        return result


@contextmanager
def _capture_student(recorder: TraceRecorder):
    """Wrap original Stage5H/N layer calls, preserving their numerical path."""
    import real_full_stage5j as prelude
    import real_core_stage5lm as core
    import real_postlude_stage5no as postlude

    patches = []

    def patch(module, symbol):
        previous = getattr(module, symbol)
        def observed(*args, **kwargs):
            result = previous(*args, **kwargs)
            record = result[2]
            recorder.add(int(record['physical_layer']), result[0])
            return result
        setattr(module, symbol, observed)
        patches.append((module, symbol, previous))

    patch(prelude, 'sliding_one_token_layer')
    patch(prelude, 'full_attention_one_token_layer')
    patch(core, 'core_one_token_layer')
    patch(postlude, 'postlude_one_token_layer')
    try:
        yield
    finally:
        for module, symbol, original in reversed(patches):
            setattr(module, symbol, original)


def _save(path: Path, *, role: str, revision: str, token: int,
          visits: np.ndarray, final_hidden: np.ndarray,
          ids: np.ndarray, logits: np.ndarray, framework: str) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(f'refusing to overwrite trace {path}')
    if (role not in ('hf', 'student') or visits.shape != (48, 2304) or
            final_hidden.shape != (2304,) or ids.ndim != 1 or logits.shape != ids.shape or
            not np.isfinite(visits).all() or not np.isfinite(final_hidden).all() or
            not np.isfinite(logits).all()):
        raise ValueError('invalid parity capture tensors')
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path,
        schema=np.array('inkling2_stage5p_parity_v1'),
        role=np.array(role),revision=np.array(revision),token=np.array(token, dtype=np.int64),
        framework=np.array(framework),layers=np.array(execution_layers(), dtype=np.int32),
        slots=np.array(execution_slots(), dtype=np.int32),
        visits=np.asarray(visits, dtype=np.float32),
        final_hidden=np.asarray(final_hidden, dtype=np.float32),
        top_ids=np.asarray(ids, dtype=np.int32),
        top_logits=np.asarray(logits, dtype=np.float32))


def _read(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as archive:
        required = {'schema','role','revision','token','framework','layers','slots','visits',
                    'final_hidden','top_ids','top_logits'}
        if set(archive.files) != required:
            raise ValueError(f'{path}: unknown/missing parity capture fields')
        data = {name: archive[name].copy() for name in required}
    for name in ('schema', 'role', 'revision', 'framework', 'token'):
        if data[name].shape != ():
            raise ValueError(f'{name}: expected scalar')
    if (str(data['schema']) != 'inkling2_stage5p_parity_v1' or
        data['role'].item() not in ('hf', 'student') or
        data['visits'].shape != (48,2304) or data['final_hidden'].shape != (2304,) or
        data['layers'].tolist() != execution_layers() or
        data['slots'].tolist() != execution_slots() or
        data['top_ids'].ndim != 1 or data['top_logits'].shape != data['top_ids'].shape or
        data['top_ids'].size == 0 or data['top_ids'].size > 10 or
        not all(np.isfinite(data[n]).all() for n in ('visits','final_hidden','top_logits'))):
        raise ValueError('invalid 48-step parity report shape/contents')
    return data


def compare(student: Path, teacher: Path, *, atol: float, rtol: float) -> dict:
    if (not math.isfinite(atol) or atol < 0 or not math.isfinite(rtol) or rtol < 0):
        raise ValueError('nonnegative finite comparison tolerances required')
    s, h = _read(student), _read(teacher)
    if s['role'].item() != 'student' or h['role'].item() != 'hf':
        raise ValueError('comparison requires one student and one real HF capture')
    if (s['revision'].item() != h['revision'].item() or
            int(s['token']) != int(h['token'])):
        raise ValueError('checkpoint revision or input token differs: cannot compare')
    failures = []
    layer_reports = []
    for index, physical in enumerate(execution_layers()):
        got, ref = s['visits'][index], h['visits'][index]
        error = np.abs(got.astype(np.float64) - ref.astype(np.float64))
        bound = atol + rtol * np.abs(ref.astype(np.float64))
        max_abs = float(error.max())
        fail = bool((error > bound).any())
        if fail:
            failures.append(index)
        layer_reports.append({
            'visit':index,'physical_layer':physical,'cache_slot':execution_slots()[index],
            'max_abs_error':max_abs,'rmse':float(np.sqrt(np.mean(error**2))),
            'fraction_outside_tolerance':float((error > bound).mean()),
            'within_tolerance':not fail,
        })
    error = np.abs(s['final_hidden'].astype(np.float64) - h['final_hidden'].astype(np.float64))
    bound = atol + rtol * np.abs(h['final_hidden'].astype(np.float64))
    same_top1 = int(s['top_ids'][0]) == int(h['top_ids'][0])
    final_ok = not (error > bound).any()
    passed = not failures and final_ok and same_top1
    return {
        'status':'PASS' if passed else 'MISMATCH',
        'source_revision':str(s['revision']), 'token':int(s['token']),
        'student_framework':str(s['framework']),'reference_framework':str(h['framework']),
        'first_failing_visit':failures[0] if failures else None,
        'final_hidden_max_abs_error':float(error.max()),
        'final_hidden_within_tolerance':bool(final_ok),
        'student_top1':int(s['top_ids'][0]),'hf_top1':int(h['top_ids'][0]),
        'same_top1':bool(same_top1),
        'atol':atol,'rtol':rtol,'visits':layer_reports,
        'explanation':'HF BF16 vs Student FP32 differences can be expected; MISMATCH is not proof of source corruption.',
    }


def student_capture(args) -> None:
    from tools.artifact.reader import Artifact
    from nonexpert_artifact import Checkpoint
    from real_carrier_stage5f import Carrier5F, validate_complete_carriers
    from real_sliding_stage5h import Geometry, TorchCarrierWeights
    from real_postlude_stage5no import run_real_full_one_token, streaming_unembed_topk
    from run_real_forward_stage5no import full_forward_preflight
    import torch

    with Artifact(args.nonexpert) as non, Artifact(args.experts) as exp:
        validate_complete_carriers(args.source, non, exp, args.banks)
        with Checkpoint(args.source) as check:
            full_forward_preflight(check.spans, check.config, mode='greedy')
            cfg = check.config
        if not torch.cuda.is_available() and str(args.device).startswith('cuda'):
            raise ValueError('CUDA requested but unavailable')
        weights = TorchCarrierWeights(Carrier5F(non, exp), args.device, Geometry.from_hf_config(cfg))
        recorder = TraceRecorder([], [])
        t = cfg['text_config']
        with torch.inference_mode(), _capture_student(recorder):
            final, blocks, trace = run_real_full_one_token(weights, args.token,
                gate_rank=t['loop_config']['gate_rank'],
                log_scaling_alpha=t['log_scaling_alpha'],
                log_scaling_n_floor=t['log_scaling_n_floor'])
            winners = streaming_unembed_topk(weights.reader, final, topk=5,
                unpadded_vocab=t.get('unpadded_vocab_size', 200058),
                softcap=t.get('final_logit_softcapping', 30.0))
        if [item['cache_slot'] for item in trace] != execution_slots():
            raise ValueError('student trace diverged from official cache-slot schedule')
        _save(args.output,role='student',revision=args.source.name,token=args.token,
              visits=recorder.matrix(),final_hidden=_flatten_hidden(final),
              ids=np.array([x['token_id'] for x in winners],dtype=np.int32),
              logits=np.array([x['logit'] for x in winners],dtype=np.float32),
              framework=f'NInfer-v3 PyTorch FP32 integration / torch {torch.__version__}')
        print(f'STUDENT CAPTURE OK: 48 visits; final token candidate={winners[0]["token_id"]}; {args.output}')
        print('NOT VALIDATED: compare against independently loaded official HF reference')


def hf_capture(args) -> None:
    """Use official Transformers class, not Stage5A's copied math or our carriers.

    This may require much more memory than the quantized 20-GB checkpoint: the
    standard Transformers Inkling2Loop loader can dequantize expert tensors.
    """
    try:
        import torch
        import transformers
        from transformers import Inkling2LoopForCausalLM
    except (ImportError, AttributeError) as error:
        raise RuntimeError('official HF Inkling2LoopForCausalLM unavailable: install a compatible transformers build') from error
    recorder = TraceRecorder([], [])
    final = []
    hooks = []
    try:
        print('Loading OFFICIAL Transformers model; may dequantize experts and require >48 GiB RAM/VRAM.')
        model = Inkling2LoopForCausalLM.from_pretrained(
            str(args.source), dtype=torch.bfloat16,
            device_map=args.device_map, low_cpu_mem_usage=True)
        model.eval()
        layers = model.model.layers
        if len(layers) != 24:
            raise ValueError('official HF model has unexpected physical layer count')
        for i, module in enumerate(layers):
            def callback(mod, inp, out, index=i):
                recorder.add(index, out)
            hooks.append(module.register_forward_hook(callback))
        def capture_final(mod, inp, out):
            final.append(_flatten_hidden(out))
        hooks.append(model.model.norm.register_forward_hook(capture_final))
        embed = model.model.embed_tokens.weight
        device = embed.device
        if device.type == 'meta':
            raise RuntimeError('HF embedding offloaded to meta: input routing requires a concrete execution device')
        ids = torch.tensor([[args.token]], dtype=torch.long, device=device)
        with torch.inference_mode():
            lm = model(input_ids=ids,use_cache=False,logits_to_keep=1)
        if len(final) != 1 or recorder.layers != execution_layers():
            raise RuntimeError('official HF hook did not observe exactly 48 layer visits/final norm')
        logits = lm.logits.detach().float().reshape(-1)
        if logits.numel() != 200058:
            raise ValueError(f'HF returned unexpected number of logits: {logits.numel()}')
        top_ids = torch.argsort(logits,descending=True,stable=True)[:5]
        top = top_ids.cpu().numpy().astype(np.int32)
        top_logits = logits[top_ids].cpu().numpy().astype(np.float32)
        _save(args.output,role='hf',revision=args.source.name,token=args.token,
              visits=recorder.matrix(),final_hidden=final[0],
              ids=top,logits=top_logits,
              framework=f'official HuggingFace transformers {transformers.__version__} / torch {torch.__version__} BF16')
        print(f'OFFICIAL HF CAPTURE OK: 48 visits; HF token={int(top[0])}; {args.output}')
    finally:
        for handle in hooks:
            handle.remove()


def cli(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='mode', required=True)
    student = sub.add_parser('student', help='record NInfer-v3 Torch 48-layer student')
    for name in ('source','nonexpert','experts','banks'):
        student.add_argument('--'+name, required=True, type=Path)
    student.add_argument('--token', type=int, default=1000)
    student.add_argument('--device', default='cuda:0')
    student.add_argument('--output', required=True, type=Path)
    hf = sub.add_parser('hf', help='record official full Transformers BF16 reference; high RAM usage')
    hf.add_argument('--source', required=True, type=Path)
    hf.add_argument('--token', type=int, default=1000)
    hf.add_argument('--device-map', default='auto')
    hf.add_argument('--output', required=True, type=Path)
    cmp = sub.add_parser('compare', help='compare BOTH captures; nonzero exit on mismatch')
    cmp.add_argument('--student', type=Path, required=True)
    cmp.add_argument('--teacher', type=Path, required=True)
    cmp.add_argument('--atol', type=float, default=0.01)
    cmp.add_argument('--rtol', type=float, default=0.01)
    cmp.add_argument('--output', type=Path)
    args = parser.parse_args(argv)
    if getattr(args, 'token', 0) < 0:
        parser.error('token must be nonnegative')
    try:
        if args.mode == 'student':
            student_capture(args)
        elif args.mode == 'hf':
            hf_capture(args)
        else:
            result = compare(args.student,args.teacher,atol=args.atol,rtol=args.rtol)
            if args.output:
                if args.output.exists() or args.output.is_symlink():
                    raise FileExistsError('refusing to overwrite parity JSON report')
                args.output.parent.mkdir(parents=True,exist_ok=True)
                args.output.write_text(json.dumps(result,indent=2)+'\n')
            print('PARITY',result['status'],'first_failing_visit=',result['first_failing_visit'],
                  'same_top1=',result['same_top1'], 'final_max_abs_error=',result['final_hidden_max_abs_error'])
            if result['status'] != 'PASS':
                print('MISMATCH: inspect per-visit diagnostic report; NO HF parity claim')
                return 2
        return 0
    except (OSError,ValueError,RuntimeError,KeyError,ImportError) as error:
        print('ERROR:',error,file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(cli())
