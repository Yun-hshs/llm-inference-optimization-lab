from __future__ import annotations

import unittest

from llm_opt_lab.mini_engine.benchmark import BenchmarkCase
from llm_opt_lab.mini_engine.scheduler import ActiveRequest, GenerationRequest
from llm_opt_lab.mini_engine.serving_loop import ServingLoopResult


class BenchmarkRecordTest(unittest.TestCase):
    def test_builds_record_from_case_config_and_serving_result_metrics(self) -> None:
        request = GenerationRequest("req-1", prompt=[1, 2], max_new_tokens=2)
        finished = ActiveRequest(
            request=request,
            generated_tokens=[7, 8],
            finished=True,
        )
        result = ServingLoopResult(
            finished_requests=[finished],
            final_status={
                "waiting_count": 0,
                "active_count": 0,
                "has_work": False,
                "is_idle": True,
                "is_blocked_by_kv_budget": False,
                "active_kv_cache_entries": 0,
                "active_kv_cache_memory_bytes": 0,
                "remaining_kv_cache_budget_bytes": 128,
                "max_kv_cache_memory_bytes": 128,
            },
        )
        benchmark_case = BenchmarkCase(
            name="batch-1-budget-128",
            max_batch_size=1,
            hidden_size=4,
            bytes_per_element=2,
            result=result,
        )

        self.assertEqual(
            benchmark_case.record(),
            {
                "case_name": "batch-1-budget-128",
                "max_batch_size": 1,
                "hidden_size": 4,
                "bytes_per_element": 2,
                "finished_request_count": 1,
                "generated_token_count": 2,
                "output_token_count": 4,
                "waiting_count": 0,
                "active_count": 0,
                "active_kv_cache_entries": 0,
                "active_kv_cache_memory_bytes": 0,
                "remaining_kv_cache_budget_bytes": 128,
                "max_kv_cache_memory_bytes": 128,
                "stopped_reason": "idle",
            },
        )

    def test_record_preserves_blocked_stop_reason(self) -> None:
        result = ServingLoopResult(
            finished_requests=[],
            final_status={
                "waiting_count": 1,
                "active_count": 0,
                "has_work": True,
                "is_idle": False,
                "is_blocked_by_kv_budget": True,
                "active_kv_cache_entries": 0,
                "active_kv_cache_memory_bytes": 0,
                "remaining_kv_cache_budget_bytes": 15,
                "max_kv_cache_memory_bytes": 15,
            },
        )
        benchmark_case = BenchmarkCase(
            name="blocked-small-budget",
            max_batch_size=1,
            hidden_size=4,
            bytes_per_element=2,
            result=result,
        )

        self.assertEqual(
            benchmark_case.record(),
            {
                "case_name": "blocked-small-budget",
                "max_batch_size": 1,
                "hidden_size": 4,
                "bytes_per_element": 2,
                "finished_request_count": 0,
                "generated_token_count": 0,
                "output_token_count": 0,
                "waiting_count": 1,
                "active_count": 0,
                "active_kv_cache_entries": 0,
                "active_kv_cache_memory_bytes": 0,
                "remaining_kv_cache_budget_bytes": 15,
                "max_kv_cache_memory_bytes": 15,
                "stopped_reason": "blocked_by_kv_budget",
            },
        )


if __name__ == "__main__":
    unittest.main()
