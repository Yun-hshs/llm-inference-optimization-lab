from __future__ import annotations

import unittest

from llm_opt_lab.mini_engine.scheduler import ActiveRequest, GenerationRequest
from llm_opt_lab.mini_engine.serving_loop import ServingLoopResult


class ServingLoopMetricsTest(unittest.TestCase):
    def test_summarizes_finished_requests_and_idle_status(self) -> None:
        req_1 = GenerationRequest("req-1", prompt=[1, 2], max_new_tokens=2)
        req_2 = GenerationRequest("req-2", prompt=[3], max_new_tokens=1)
        active_1 = ActiveRequest(request=req_1, generated_tokens=[7, 8], finished=True)
        active_2 = ActiveRequest(request=req_2, generated_tokens=[9], finished=True)
        result = ServingLoopResult(
            finished_requests=[active_1, active_2],
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

        self.assertEqual(
            result.metrics_summary(),
            {
                "finished_request_count": 2,
                "generated_token_count": 3,
                "output_token_count": 6,
                "waiting_count": 0,
                "active_count": 0,
                "active_kv_cache_entries": 0,
                "active_kv_cache_memory_bytes": 0,
                "remaining_kv_cache_budget_bytes": 128,
                "max_kv_cache_memory_bytes": 128,
                "stopped_reason": "idle",
            },
        )

    def test_reports_blocked_stop_reason(self) -> None:
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

        self.assertEqual(
            result.metrics_summary(),
            {
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

    def test_reports_running_stop_reason_when_status_is_neither_idle_nor_blocked(self) -> None:
        result = ServingLoopResult(
            finished_requests=[],
            final_status={
                "waiting_count": 1,
                "active_count": 1,
                "has_work": True,
                "is_idle": False,
                "is_blocked_by_kv_budget": False,
                "active_kv_cache_entries": 4,
                "active_kv_cache_memory_bytes": 64,
                "remaining_kv_cache_budget_bytes": 64,
                "max_kv_cache_memory_bytes": 128,
            },
        )

        self.assertEqual(
            result.metrics_summary(),
            {
                "finished_request_count": 0,
                "generated_token_count": 0,
                "output_token_count": 0,
                "waiting_count": 1,
                "active_count": 1,
                "active_kv_cache_entries": 4,
                "active_kv_cache_memory_bytes": 64,
                "remaining_kv_cache_budget_bytes": 64,
                "max_kv_cache_memory_bytes": 128,
                "stopped_reason": "running",
            },
        )


if __name__ == "__main__":
    unittest.main()
