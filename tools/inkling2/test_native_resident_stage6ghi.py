"""CPU contracts for the new native G/H/I bridge (no CUDA required)."""
from __future__ import annotations
import unittest
from dataclasses import dataclass
from unittest.mock import patch

import numpy as np
import native_resident_stage6ghi as under_test


@dataclass
class Raw:
    codes: bytes
    scales: bytes
    amax: float = 1.0
    denominator: float = 1.0


class TestResidentContracts(unittest.TestCase):
    def test_native_geometry_24_layers(self):
        self.assertEqual(under_test.plane_sizes(0), (339738624, 42467328, 169869312, 21233664))
        self.assertEqual(under_test.plane_sizes(20), (679477248, 21233664, 339738624, 10616832))
        for layer in (-1, 24, 1.0, True):
            with self.assertRaises(ValueError): under_test.plane_sizes(layer)

    def test_complete_and_source_order(self):
        first=[Raw(bytes([i]),bytes([255-i]),i+1.,i+2.) for i in range(128)]
        second=[Raw(bytes([255-i]),bytes([i]),i+3.,i+4.) for i in range(128)]
        with patch.object(under_test,'plane_sizes',return_value=(128,128,128,128)):
            planes,factors=under_test.pack_full_bank(0,first,second,validate=False)
        self.assertEqual(list(planes[0]),list(range(128)))
        self.assertEqual(list(planes[1]),list(range(255,127,-1)))
        self.assertEqual(list(planes[2]),list(range(255,127,-1)))
        self.assertEqual(list(planes[3]),list(range(128)))
        self.assertEqual(factors.shape,(4,128))
        self.assertEqual(tuple(factors[:,3]),(4.,5.,6.,7.))
        self.assertEqual(factors.dtype,np.float32)

    def test_banks_cannot_have_missing_experts(self):
        with self.assertRaisesRegex(ValueError, 'all 128'):
            under_test.pack_full_bank(0,[Raw(b'0',b'0')]*127,[Raw(b'0',b'0')]*128,validate=False)

    def test_mismatched_payload_refused(self):
        raw=[Raw(b'12',b'0') for _ in range(128)]
        with patch.object(under_test,'plane_sizes',return_value=(128,128,128,128)):
            with self.assertRaisesRegex(ValueError,'expected 1 bytes'):
                under_test.pack_full_bank(0,raw,raw,validate=False)

    def test_nonfinite_amplitude_rejected(self):
        first=[Raw(b'1',b'1') for _ in range(128)]
        second=[Raw(b'1',b'1') for _ in range(128)]
        first[3].amax=float('nan')
        with patch.object(under_test,'plane_sizes',return_value=(128,128,128,128)):
            with self.assertRaisesRegex(ValueError,'scale'):
                under_test.pack_full_bank(0,first,second,validate=False)

    def test_address_conforms_to_no_copy_from_buffer(self):
        raw=bytearray(b'\x01\x02\x03')
        import ctypes
        p=under_test._ptr(raw)
        self.assertEqual(ctypes.string_at(p,3),bytes(raw))
        raw[2]=0x55
        self.assertEqual(ctypes.string_at(p,3),b'\x01\x02\x55')


if __name__=='__main__':unittest.main()
