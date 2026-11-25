"""Stage 6D: verify exact NVFP4 K16 scale words for original Inkling widths."""
import unittest
import random
import time
from ninfer_artifact_bridge import unswizzle_nvfp4_128_rows, swizzle_nvfp4_128_rows


def scalar_reference(scales, cols):
    if cols <= 0 or cols % 64 or len(scales) != 128 * cols // 16:
        raise ValueError('NVFP4 unswizzle requires a complete 128-row tile')
    stride = cols // 16
    out = bytearray(len(scales))
    cursor = 0
    for k_tile in range(cols // 64):
        for r32 in range(32):
            for group in range(4):
                begin = (group * 32 + r32) * stride + k_tile * 4
                out[begin:begin + 4] = scales[cursor:cursor + 4]
                cursor += 4
    return bytes(out)


class TestFastScaleUnswizzle(unittest.TestCase):
    def test_independent_scalar_byte_equality(self):
        for cols in (64, 1152, 2304):
            for seed in (1, 999, 2901):
                with self.subTest(cols=cols, seed=seed):
                    raw = random.Random(seed).randbytes(128 * cols // 16)
                    self.assertEqual(unswizzle_nvfp4_128_rows(raw, cols), scalar_reference(raw, cols))

    def test_unsigned_words_and_roundtrip(self):
        for cols in (1152, 2304):
            # Input need not be valid quantization for byte-layout permutation.
            raw = bytes((x * 17 + x // 3) % 256 for x in range(128 * cols // 16))
            self.assertEqual(unswizzle_nvfp4_128_rows(swizzle_nvfp4_128_rows(raw, cols), cols), raw)

    def test_invalid_geometry(self):
        for cols, raw in ((0,b''),(63,b''),(1151,b''),(2304,b''),(2304,b'abc')):
            with self.subTest(cols=cols), self.assertRaises(ValueError):
                unswizzle_nvfp4_128_rows(raw, cols)

    def test_multiple_expert_tiles(self):
        for cols in (1152,2304):
            tile = random.Random(cols).randbytes(128 * cols // 16)
            expected = b''.join(scalar_reference(tile, cols) for _ in range(18))
            actual = b''.join(unswizzle_nvfp4_128_rows(tile, cols) for _ in range(18))
            self.assertEqual(actual, expected)


if __name__ == '__main__':
    unittest.main()
