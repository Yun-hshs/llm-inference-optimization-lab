from __future__ import annotations

import unittest

import numpy as np

from llm_opt_lab.quantization import (
    dequantize_int,
    quantization_error,
    quantize_symmetric,
)


class QuantizationTest(unittest.TestCase):
    def test_int8_quantization_preserves_shape_and_metadata(self) -> None:
        values = np.array([-1.0, -0.5, 0.0, 0.5, 1.0], dtype=np.float32)

        quantized = quantize_symmetric(values, bits=8)

        self.assertEqual(quantized.bits, 8)
        self.assertEqual(quantized.values.shape, values.shape)
        self.assertEqual(quantized.values.dtype, np.int8)
        self.assertGreater(quantized.scale, 0)

    def test_int4_quantization_roundtrip_error_is_bounded(self) -> None:
        values = np.linspace(-1.0, 1.0, num=33, dtype=np.float32)

        quantized = quantize_symmetric(values, bits=4)
        restored = dequantize_int(quantized)
        error = quantization_error(values, restored)

        self.assertLess(error["mae"], 0.04)
        self.assertLess(error["max_abs"], 0.08)

    def test_quantization_rejects_unsupported_bits(self) -> None:
        with self.assertRaisesRegex(ValueError, "only 4-bit and 8-bit"):
            quantize_symmetric(np.array([1.0], dtype=np.float32), bits=3)


if __name__ == "__main__":
    unittest.main()
