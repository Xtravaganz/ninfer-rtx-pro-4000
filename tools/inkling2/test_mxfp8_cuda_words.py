"""Stage 3B host-side gold checks for the exact FP32 bit construction in CUDA.

These are mathematical/encoding tests; they do NOT prove nvcc builds or a GPU
launch succeeds. That is the responsibility of test_mxfp8_gemv.cu.
"""
from __future__ import annotations

import math
import struct
import unittest

from mxfp8_reference import decode_e4m3fn, decode_ue8m0


def _float32(bits: int) -> float:
    return struct.unpack('<f', struct.pack('<I', bits))[0]


def _cuda_weight_bits(word: int) -> int:
    if (word & 0x7f) == 0x7f:
        return 0x7fc00000
    sign = (word & 0x80) << 24
    exponent = (word >> 3) & 15
    mantissa = word & 7
    if exponent == 0:
        value = _float32(struct.unpack('<I', struct.pack('<f', mantissa * 2**-9))[0])
        bits = struct.unpack('<I', struct.pack('<f', value))[0]
        return bits | sign
    return sign | ((exponent + 120) << 23) | (mantissa << 20)


def _cuda_scale_bits(word: int) -> int:
    if word == 255:
        return 0x7fc00000
    return 0x00400000 if word == 0 else word << 23


class TestCudaBitContract(unittest.TestCase):
    def test_every_finite_e4m3fn_word(self):
        for word in range(256):
            decoded = _float32(_cuda_weight_bits(word))
            if (word & 0x7f) == 0x7f:
                self.assertTrue(math.isnan(decoded))
            else:
                ref = decode_e4m3fn(word)
                self.assertEqual(decoded, ref, f'weight word {word:#04x}')
                if ref == 0:
                    self.assertEqual(math.copysign(1.0, decoded),
                                     math.copysign(1.0, ref))

    def test_every_ue8m0_byte(self):
        for word in range(256):
            decoded = _float32(_cuda_scale_bits(word))
            if word == 255:
                self.assertTrue(math.isnan(decoded))
            else:
                self.assertEqual(decoded, decode_ue8m0(word),
                                 f'scale word {word:#04x}')
        self.assertEqual(_float32(_cuda_scale_bits(0)), 2**-127)

    def test_group32_indexing_and_large_inkling_geometry(self):
        for rows, cols in ((1, 32), (7, 96), (2304, 1152), (2304, 2304)):
            self.assertEqual((rows * cols) // 32, rows * (cols // 32))
            for row in (0, rows - 1):
                for col in (0, 31, cols // 2, cols - 1):
                    src = row * (cols // 32) + col // 32
                    self.assertLess(src, rows * (cols // 32))
                    self.assertGreaterEqual(src, 0)


if __name__ == '__main__':
    unittest.main()
