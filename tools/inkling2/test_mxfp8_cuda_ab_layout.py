"""Stage 3C CPU-only warp/K32 lane-map proofs (CUDA correctness still needs GPU).

These tests cover the address mapping in all three CUDA candidates for full and
partial 128-column tiles. They cannot replace nvcc compilation or device tests.
"""
from __future__ import annotations

import math
import unittest

from mxfp8_reference import (
    cpu_gemv,
    decode_e4m3fn,
    decode_ue8m0,
    synthetic_matrix,
)


def candidate_columns(cols: int, variant: str):
    if cols <= 0 or cols % 32:
        raise ValueError("MXFP8 requires K32")
    groups = cols // 32
    for split in range(4 if variant.startswith("split") else 1):
        for lane in range(32):
            if variant == "split_k4":
                for group in range(split, groups, 4):
                    yield (group * 32 + lane, group)
            else:
                tile_count = (groups + 3) // 4
                stride = 4 if variant == "split_k4_vector4" else 1
                for tile in range(split, tile_count, stride):
                    group = tile * 4 + (lane // 8)
                    if group >= groups:
                        continue
                    for part in range(4):
                        col = group * 32 + (lane % 8) * 4 + part
                        yield (col, group)


class TestCudaAbLaneMapping(unittest.TestCase):
    def test_exactly_one_visit_per_column_and_scale_group(self):
        for cols in (32, 64, 96, 128, 160, 256, 1152, 2304):
            for variant in ("warp_vector4", "split_k4", "split_k4_vector4"):
                with self.subTest(cols=cols, variant=variant):
                    visits = list(candidate_columns(cols, variant))
                    self.assertEqual(len(visits), cols)
                    self.assertEqual(sorted(col for col, _ in visits), list(range(cols)))
                    self.assertTrue(all(group == col // 32 for col, group in visits))

    def test_per_row_byte_map_matches_independent_mxfp8_oracle(self):
        for cols in (32, 96, 1152, 2304):
            matrix = synthetic_matrix(3, cols)
            x = [((i * 17) % 47 - 23) * 0.03125 for i in range(cols)]
            expected = cpu_gemv(matrix, x)
            for variant in ("warp_vector4", "split_k4", "split_k4_vector4"):
                mapping = list(candidate_columns(cols, variant))
                for row in range(matrix.rows):
                    value = math.fsum(
                        decode_e4m3fn(matrix.codes[row * cols + col])
                        * decode_ue8m0(matrix.scales[row * (cols // 32) + group])
                        * x[col]
                        for col, group in mapping
                    )
                    with self.subTest(cols=cols, variant=variant, row=row):
                        self.assertEqual(value, expected[row])

    def test_partial_block_output_rows(self):
        for rows in (1, 2, 7, 8, 9, 129, 2304):
            vector_rows = [cta * 8 + warp for cta in range((rows + 7) // 8)
                           for warp in range(8) if cta * 8 + warp < rows]
            split_rows = [cta * 2 + warp // 4 for cta in range((rows + 1) // 2)
                          for warp in range(8) if cta * 2 + warp // 4 < rows]
            with self.subTest(rows=rows):
                self.assertEqual(vector_rows, list(range(rows)))
                self.assertEqual(sorted(set(split_rows)), list(range(rows)))
                self.assertTrue(all(split_rows.count(r) == 4 for r in range(rows)))

    def test_reject_invalid_k(self):
        for cols in (0, -32, 31, 33, 2305):
            with self.assertRaises(ValueError):
                list(candidate_columns(cols, "warp_vector4"))


if __name__ == "__main__":
    unittest.main()
