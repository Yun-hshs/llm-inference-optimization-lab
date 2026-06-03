from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class QuantizedTensor:
    values: np.ndarray
    scale: float
    bits: int


def _signed_range(bits: int) -> tuple[int, int]:
    if bits not in {4, 8}:
        raise ValueError("only 4-bit and 8-bit symmetric quantization are supported")
    qmax = (2 ** (bits - 1)) - 1
    qmin = -(2 ** (bits - 1))
    return qmin, qmax


def quantize_symmetric(values: np.ndarray, bits: int) -> QuantizedTensor:
    qmin, qmax = _signed_range(bits)
    array = np.asarray(values, dtype=np.float32)
    max_abs = float(np.max(np.abs(array))) if array.size else 0.0
    scale = max_abs / qmax if max_abs > 0 else 1.0
    quantized = np.clip(np.round(array / scale), qmin, qmax).astype(np.int8)
    return QuantizedTensor(values=quantized, scale=scale, bits=bits)


def dequantize_int(tensor: QuantizedTensor) -> np.ndarray:
    _signed_range(tensor.bits)
    return tensor.values.astype(np.float32) * tensor.scale


def quantization_error(original: np.ndarray, restored: np.ndarray) -> dict[str, float]:
    original_array = np.asarray(original, dtype=np.float32)
    restored_array = np.asarray(restored, dtype=np.float32)
    diff = original_array - restored_array
    return {
        "mae": float(np.mean(np.abs(diff))),
        "max_abs": float(np.max(np.abs(diff))) if diff.size else 0.0,
        "rmse": float(np.sqrt(np.mean(diff * diff))),
    }
