"""Stage 3A independent CPU, C++ golden and optional Torch/CUDA MXFP8 checks."""
from __future__ import annotations

import math
import random
from hashlib import sha256
import json
from pathlib import Path
import struct
import tempfile
import unittest

from mxfp8_reference import (
    Matrix, cpu_gemv, decode_e4m3fn, decode_ue8m0,
    read_stage2a_rows, read_stage2b_rows, synthetic_matrix,
    torch_decode, torch_gemv,
)


class TestScalarMxFp8(unittest.TestCase):
    def test_e8m0_goldens_and_sentinel(self):
        self.assertEqual(decode_ue8m0(0x00), 2.0**-127)  # NOT zero
        self.assertEqual(decode_ue8m0(0x01), 2.0**-126)
        self.assertEqual(decode_ue8m0(0x7e), 0.5)
        self.assertEqual(decode_ue8m0(0x7f), 1.0)
        self.assertEqual(decode_ue8m0(0x80), 2.0)
        self.assertEqual(decode_ue8m0(0xfe), 2.0**127)
        for bad in (0xff, -1, 256, 1.0):
            with self.assertRaises(ValueError):
                decode_ue8m0(bad)

    def test_e4m3_goldens_subnormal_signed_zero(self):
        goldens = {0x38: 1.0, 0xb8: -1.0, 0x01: 2.0**-9,
                   0x07: 7 * 2.0**-9, 0x08: 2.0**-6,
                   0x7e: 448.0, 0xfe: -448.0, 0x00: 0.0}
        for word, expected in goldens.items():
            self.assertEqual(decode_e4m3fn(word), expected)
        self.assertTrue(math.copysign(1, decode_e4m3fn(0x80)) < 0)
        for bad in (0x7f, 0xff, 256, -1):
            with self.assertRaises(ValueError):
                decode_e4m3fn(bad)

    def test_all_finite_fp8_words_match_pytorch_when_available(self):
        try:
            import torch
        except ImportError:
            self.skipTest('torch unavailable')
        raw = bytes(w for w in range(256) if (w & 127) != 127)
        words = torch.frombuffer(bytearray(raw), dtype=torch.uint8)
        values = words.view(torch.float8_e4m3fn).float().tolist()
        for word, value in zip(raw, values):
            self.assertEqual(decode_e4m3fn(word), value)

    def test_block_and_row_major_mapping(self):
        codes = b'\x38'*32 + b'\xb8'*32 + b'\x38'*32 + b'\xb8'*32
        m = Matrix(2, 64, codes, bytes([127, 128, 128, 126]))
        self.assertEqual(m.at(0, 0), 1)
        self.assertEqual(m.at(0, 31), 1)
        self.assertEqual(m.at(0, 32), -2)
        self.assertEqual(m.at(1, 0), 2)
        self.assertEqual(m.at(1, 32), -0.5)
        self.assertEqual(cpu_gemv(m, [1.0]*64), [-32.0, 48.0])
        with self.assertRaises(ValueError):
            cpu_gemv(m, [1.0]*63)
        with self.assertRaises(IndexError):
            m.at(0, 64)

    def test_scale_zero_is_nonzero_smallest(self):
        m = Matrix(1, 32, b'\x38'*32, b'\x00')
        self.assertEqual(m.at(0, 11), 2**-127)
        self.assertEqual(cpu_gemv(m, [1.0]*32), [32 * 2**-127])

    def test_validation_catches_corruption(self):
        for codes, scales in ((b'\x7f'*32, b'\x7f'),
                              (b'\xff'*32, b'\x7f'),
                              (b'\x38'*32, b'\xff'),
                              (b'\x38'*31, b'\x7f'),
                              (b'\x38'*32, b'\x7f\x80')):
            with self.assertRaises(ValueError):
                Matrix(1, 32, codes, scales)
        for rows, cols in ((0, 32), (1, 31), (1, 33), (True, 32)):
            with self.assertRaises(ValueError):
                Matrix(rows, cols, b'\x38'*max(0, cols*int(rows)), b'\x7f')


class TestTorchMxFp8(unittest.TestCase):
    def test_cpu_torch_decoder_exact_ordinary_scales(self):
        try:
            import torch
        except ImportError:
            self.skipTest('torch unavailable')
        m = synthetic_matrix(9, 160)
        decoded = torch_decode(m, device='cpu')
        self.assertEqual(tuple(decoded.shape), (9, 160))
        for row in range(m.rows):
            for k in range(m.cols):
                self.assertEqual(float(decoded[row, k]), m.at(row, k))

    def test_real_inkling_k_geometry(self):
        # Matches the last four physical-layer expert projection K dimensions.
        for cols in (1152, 2304):
            matrix = synthetic_matrix(2, cols)
            self.assertEqual(len(matrix.codes), 2 * cols)
            self.assertEqual(len(matrix.scales), 2 * (cols // 32))
            self.assertTrue(math.isfinite(matrix.at(1, cols - 1)))

    def test_cpu_torch_gemv_vs_independent_reference(self):
        try:
            import torch
        except ImportError:
            self.skipTest('torch unavailable')
        m = synthetic_matrix(17, 224)
        rng = random.Random(62)
        x = [struct.unpack('<f', struct.pack('<f', rng.uniform(-1.0, 1.0)))[0] for _ in range(m.cols)]
        expect = cpu_gemv(m, x)
        actual = torch_gemv(m, x, device='cpu').tolist()
        for a, b in zip(expect, actual):
            self.assertAlmostEqual(a, b, delta=max(1e-5, 4e-6 * abs(a)))

    def test_cuda_torch_dequant_and_gemv(self):
        try:
            import torch
        except ImportError:
            self.skipTest('torch unavailable')
        if not torch.cuda.is_available():
            self.skipTest('CUDA not exposed to current process (use Docker --gpus)')
        # This is vectorized GPU dequant/GEMV via Torch, NOT native MXFP8 tensor cores.
        m = synthetic_matrix(11, 288)
        rng = random.Random(3)
        x = [struct.unpack('<f', struct.pack('<f', rng.uniform(-1.0, 1.0)))[0] for _ in range(m.cols)]
        actual_w = torch_decode(m, device='cuda:0').cpu().tolist()
        for r in range(m.rows):
            for k in range(m.cols):
                self.assertAlmostEqual(actual_w[r][k], m.at(r, k), delta=1e-6)
        reference = cpu_gemv(m, x)
        actual = torch_gemv(m, x, device='cuda:0').cpu().tolist()
        for a, b in zip(reference, actual):
            self.assertAlmostEqual(a, b, delta=max(1e-5, 5e-6 * abs(a)))


class TestExistingBankAndArtifactReaders(unittest.TestCase):
    def test_stage2a_row_window_and_expert_id(self):
        from native_experts import HEADER, MAGIC, VERSION, FMT_MXFP8
        rows, cols, experts = 128, 64, (2, 8)
        code_stride, scale_stride = rows * cols, rows * (cols // 32)
        leading = HEADER.pack(MAGIC, VERSION, FMT_MXFP8, 20, 1, len(experts), rows, cols,
                              32, 0, code_stride, scale_stride, 0)
        data = bytearray(leading + struct.pack('<2I', *experts))
        expected = {}
        for eid in experts:
            codes = bytes([0x38 if eid == 2 else 0xb8]) * code_stride
            scales = bytes([126 if eid == 2 else 128, 127]) * rows
            data.extend(codes + scales)
            expected[eid] = (codes, scales)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'layer20_w13.i2bank'
            path.write_bytes(data)
            m = read_stage2a_rows(path, expert_id=8, first=7, count=3)
            self.assertEqual(m.rows, 3)
            self.assertEqual(m.cols, cols)
            self.assertEqual(m.codes, expected[8][0][7*cols:10*cols])
            self.assertEqual(m.scales, expected[8][1][7*2:10*2])
            self.assertEqual(m.at(0, 0), -2.0)
            self.assertEqual(m.at(0, 32), -1.0)
            with self.assertRaises(ValueError):
                read_stage2a_rows(path, expert_id=3, first=0, count=1)
            with self.assertRaises(ValueError):
                read_stage2a_rows(path, expert_id=8, first=127, count=2)

    def test_stage2b_real_ninfer_reader(self):
        try:
            from tools.artifact.reader import Artifact
        except ImportError:
            self.skipTest('NInfer v3 repository required for integration test')
        from native_experts import MiniGeometry
        from ninfer_artifact_bridge import pack_banks
        from test_ninfer_artifact_bridge import make_banks, SMALL
        from mxfp8_reference import read_stage2b_rows
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            banks_dir = root / 'banks'
            banks_dir.mkdir()
            original = make_banks(banks_dir)
            config_path = Path(__file__).resolve().parents[2] / 'tests/models/inkling2/fixtures/config.json'
            if not config_path.is_file():
                self.skipTest('Stage 1 config fixture required')
            target = root / 'carrier.ninfer'
            pack_banks(banks_dir, target, json.loads(config_path.read_text()), geometry=SMALL,
                       max_file_bytes=32768, chunk_bytes=31)
            first, count = 15, 4
            matrix = read_stage2b_rows(target, layer=20, projection='w13', expert_id=1,
                                        first=first, count=count)
            weights, scales, _ = original[(20,'w13')][1]
            self.assertEqual(matrix.codes, weights[first*64:(first+count)*64])
            self.assertEqual(matrix.scales, scales[first*2:(first+count)*2])
            self.assertEqual(matrix.at(0, 0), 1.0)
            self.assertEqual(matrix.at(0, 1), -1.0)
            self.assertEqual(matrix.at(0, 32), 2.0)
            with self.assertRaises(ValueError):
                read_stage2b_rows(target, layer=0, projection='w13', expert_id=1, first=0, count=1)
            with self.assertRaises(ValueError):
                read_stage2b_rows(target, layer=20, projection='w13', expert_id=50, first=0, count=1)
            with self.assertRaises(ValueError):
                read_stage2b_rows(target, layer=20, projection='w13', expert_id=1, first=127, count=2)


if __name__ == '__main__':
    unittest.main()
