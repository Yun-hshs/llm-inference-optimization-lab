from __future__ import annotations

import unittest

from llm_opt_lab.benchmark import benchmark


class BenchmarkTest(unittest.TestCase):
    def test_benchmark_returns_timing_summary(self) -> None:
        result = benchmark("constant-work", lambda: sum(range(10)), warmup=1, runs=2)

        self.assertEqual(result.name, "constant-work")
        self.assertEqual(result.runs, 2)
        self.assertGreaterEqual(result.mean_ms, 0)
        self.assertLessEqual(result.min_ms, result.max_ms)


if __name__ == "__main__":
    unittest.main()
