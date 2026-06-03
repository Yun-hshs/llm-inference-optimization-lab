from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from time import perf_counter
from typing import TypeVar

T = TypeVar("T")


@dataclass(frozen=True)
class BenchmarkResult:
    name: str
    runs: int
    mean_ms: float
    min_ms: float
    max_ms: float


def benchmark(
    name: str,
    fn: Callable[[], T],
    *,
    warmup: int = 3,
    runs: int = 10,
) -> BenchmarkResult:
    if warmup < 0:
        raise ValueError("warmup must be non-negative")
    if runs <= 0:
        raise ValueError("runs must be positive")

    for _ in range(warmup):
        fn()

    durations_ms: list[float] = []
    for _ in range(runs):
        start = perf_counter()
        fn()
        durations_ms.append((perf_counter() - start) * 1000)

    return BenchmarkResult(
        name=name,
        runs=runs,
        mean_ms=sum(durations_ms) / len(durations_ms),
        min_ms=min(durations_ms),
        max_ms=max(durations_ms),
    )
