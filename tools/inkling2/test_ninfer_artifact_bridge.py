"""Stage-2B regression tests: NInfer v3 Inkling2 expert-only carrier.

Small synthetic banks use real NVFP4 K16 / MXFP8 K32 byte layouts. The full
NInfer artifact writer/reader test runs when executed inside a NInfer checkout.
"""
from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path
import struct
import tempfile
import unittest

from native_experts import HEADER, MAGIC, VERSION, MiniGeometry, FMT_NVFP4, FMT_MXFP8
from ninfer_artifact_bridge import (
    _weight_divisor, _write_bank, load_manifest, pack_banks, swizzle_nvfp4_128_rows,
    unswizzle_nvfp4_128_rows, verify_artifact,
)

SMALL = MiniGeometry(experts=2, rows=128, gate_up_cols=64, down_cols=64)


def make_banks(root: Path):
    bank_records = []
    source = {}
    for layer in (0, 20):
        for projection in ('w13', 'w2'):
            fmt = FMT_NVFP4 if layer == 0 else FMT_MXFP8
            e, n, k = SMALL.experts, SMALL.rows, SMALL.cols(projection)
            code_stride = n * (k // 2 if fmt == FMT_NVFP4 else k)
            scale_stride = n * (k // (16 if fmt == FMT_NVFP4 else 32))
            header = HEADER.pack(MAGIC, VERSION, fmt, layer, 1 if projection == 'w13' else 2,
                                 e, n, k, 16 if fmt == FMT_NVFP4 else 32, 0,
                                 code_stride, scale_stride, 8 if fmt == FMT_NVFP4 else 0)
            out = bytearray(header + struct.pack('<II', 0, 1))
            records = []
            for expert in range(e):
                codes = (b'\x21' * code_stride if fmt == FMT_NVFP4
                         else (b'\x38\xb8' * (code_stride//2)))
                if fmt == FMT_NVFP4:
                    # Different scale for each row and K group; exercise swizzle.
                    scales = bytes((0x30 + (row + group + expert) % 7)
                                   for row in range(n) for group in range(k//16))
                    original = struct.pack('<ff', 2.0 * (expert + 1), 8.0)
                else:
                    scales = bytes((127 + (row + group + expert) % 2)
                                   for row in range(n) for group in range(k//32))
                    original = b''
                out.extend(codes + scales + original)
                records.append((codes, scales, original))
            filename = f'layer{layer:02d}_{projection}.i2bank'
            (root / filename).write_bytes(out)
            bank_records.append({
                'file': filename, 'layer': layer, 'projection': projection,
                'expert_ids': [0, 1], 'bytes': len(out), 'sha256': sha256(out).hexdigest(),
            })
            source[(layer, projection)] = records
    (root / 'manifest.json').write_text(json.dumps({
        'layout': 'inkling2_native_expert_bank_v1', 'version': 1, 'banks': bank_records,
    }))
    return source


class CaptureWriter:
    """Small, strict object-plane oracle independent of NInfer internals."""
    def __init__(self, banks):
        self.data = {}
        for bank in banks:
            e = len(bank.ids)
            self.data[bank.ids_object] = bytearray(e*4)
            if bank.fmt == FMT_NVFP4:
                code = e*bank.code_stride
                offset = (code+255)//256*256
                self.data[bank.tensor_object] = bytearray(offset+e*bank.scale_stride+4*e)
                self.data[bank.scalars_object] = bytearray(e*8)
            else:
                self.data[bank.code_object] = bytearray(e*bank.code_stride)
                self.data[bank.scale_object] = bytearray(e*bank.scale_stride)
        self.cover = {id: set() for id in self.data}

    def write_region(self, id, offset, data):
        data = bytes(data)
        assert 0 <= offset <= offset+len(data) <= len(self.data[id])
        assert all(i not in self.cover[id] for i in range(offset,offset+len(data)))
        self.cover[id].update(range(offset,offset+len(data)))
        self.data[id][offset:offset+len(data)] = data

    def write_zeros(self, id, offset, size):
        self.write_region(id, offset, bytes(size))

    def assert_complete(self):
        for id, data in self.data.items():
            assert len(self.cover[id]) == len(data), id


class TestNativeV3Bridge(unittest.TestCase):
    def test_swizzle_matches_independent_layout_index(self):
        cols = 192
        stride = cols // 16
        original = bytes((r + k*3) % 126 for r in range(128) for k in range(stride))
        native = swizzle_nvfp4_128_rows(original, cols)
        self.assertEqual(original, unswizzle_nvfp4_128_rows(native, cols))
        self.assertNotEqual(original, native)
        for row in range(128):
            for k_group in range(stride):
                k_tile, g = divmod(k_group, 4)
                group, r32 = divmod(row, 32)
                native_pos = (((k_tile*32+r32)*4+group)*4+g)
                self.assertEqual(native[native_pos], original[row*stride+k_group])
        with self.assertRaises(ValueError):
            swizzle_nvfp4_128_rows(b'\x00'*12, 96)

    def test_divisors_and_invalid_words(self):
        self.assertEqual(_weight_divisor(struct.pack('<ff',2,8)), struct.pack('<f',4))
        for words in (struct.pack('<ff', 0, 1), struct.pack('<ff', 1, 0),
                      struct.pack('<ff', float('nan'), 1)):
            with self.assertRaises(ValueError):
                _weight_divisor(words)

    def test_native_planes_preserve_source_and_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            originals = make_banks(root)
            banks = load_manifest(root, geometry=SMALL)
            writer = CaptureWriter(banks)
            for bank in banks:
                _write_bank(writer, bank, chunk_bytes=47)
                rows = originals[(bank.layer,bank.projection)]
                self.assertEqual(writer.data[bank.ids_object], struct.pack('<II',0,1))
                if bank.fmt == FMT_NVFP4:
                    code, scale, orig = zip(*rows)
                    self.assertEqual(writer.data[bank.tensor_object][:2*bank.code_stride], b''.join(code))
                    scale_offset = 2*bank.code_stride
                    self.assertEqual(writer.data[bank.tensor_object][scale_offset:scale_offset+2*bank.scale_stride],
                                     b''.join(swizzle_nvfp4_128_rows(x,bank.cols) for x in scale))
                    self.assertEqual(writer.data[bank.scalars_object], b''.join(orig))
                    self.assertEqual(writer.data[bank.tensor_object][-8:],
                                     b''.join(_weight_divisor(x) for x in orig))
                else:
                    code, scale, _ = zip(*rows)
                    self.assertEqual(writer.data[bank.code_object], b''.join(code))
                    self.assertEqual(writer.data[bank.scale_object], b''.join(scale))
            writer.assert_complete()

    def test_rejects_bad_manifest_and_changed_quantization(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            make_banks(root)
            manifest = json.loads((root/'manifest.json').read_text())
            banks = load_manifest(root, geometry=SMALL)
            # SHA mismatch is caught after streaming to temporary output; writer aborts.
            first = banks[0]
            raw = bytearray(first.path.read_bytes())
            raw[-1] ^= 0x01
            first.path.write_bytes(raw)
            with self.assertRaisesRegex(ValueError, 'SHA256'):
                _write_bank(CaptureWriter(banks), first, chunk_bytes=31)
            first.path.write_bytes(bytes(raw[:-1])+bytes([raw[-1]^1]))
            manifest['banks'][0]['expert_ids'] = [0]
            (root/'manifest.json').write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, 'identity'):
                load_manifest(root, geometry=SMALL)

    def test_real_ninfer_writer_and_reader(self):
        try:
            from tools.artifact.schema import TensorSpec, ResourceSpec
            from tools.artifact.writer import ArtifactWriter
            from tools.artifact.reader import Artifact
        except ImportError:
            self.skipTest('run integration test inside the NInfer checkout')
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            make_banks(root)
            stage1 = Path(__file__).resolve().parents[2] / 'tests/models/inkling2/fixtures/config.json'
            if not stage1.is_file():
                self.skipTest('install Stage 1 fixture first')
            config = json.loads(stage1.read_text())
            artifact = root.parent / (root.name + '-carrier.ninfer')
            try:
                result = pack_banks(root, artifact, config, geometry=SMALL,
                                    max_file_bytes=32 * 1024, chunk_bytes=47)
                self.assertEqual(result['banks'], 4)
                with Artifact(artifact) as reader:
                    self.assertEqual(reader.directory.metadata['runtime_loadable'], False)
                    self.assertEqual(sum(obj.format == 'nvfp4' for obj in reader.objects
                                         if hasattr(obj,'format')), 2)
                self.assertEqual(verify_artifact(root, artifact, geometry=SMALL, chunk_bytes=41)['banks'],4)
                with self.assertRaises(FileExistsError):
                    pack_banks(root, artifact, config, geometry=SMALL)
            finally:
                artifact.unlink(missing_ok=True)
                for path in artifact.parent.glob(artifact.name+'.part-*'):
                    path.unlink(missing_ok=True)


if __name__ == '__main__':
    unittest.main()
