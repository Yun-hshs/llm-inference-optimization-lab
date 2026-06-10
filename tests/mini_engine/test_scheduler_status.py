from __future__ import annotations

import unittest

from llm_opt_lab.mini_engine.scheduler import GenerationRequest, RequestScheduler


class SchedulerStatusTest(unittest.TestCase):
    def test_reports_counts_lifecycle_and_kv_budget_usage(self) -> None:
        scheduler = RequestScheduler(
            max_batch_size=2,
            num_kv_layers=2,
            max_kv_cache_memory_bytes=80,
        )
        scheduler.add_request(GenerationRequest("active", prompt=[1], max_new_tokens=3))
        active = scheduler.activate_next_batch()[0]
        active.kv_cache.append(layer_index=0, key=[0.1], value=[0.2])
        active.kv_cache.append(layer_index=1, key=[0.3], value=[0.4])
        scheduler.add_request(GenerationRequest("waiting", prompt=[2, 3], max_new_tokens=1))

        status = scheduler.status(hidden_size=4, bytes_per_element=2)

        self.assertEqual(
            status,
            {
                "waiting_count": 1,
                "active_count": 1,
                "has_work": True,
                "is_idle": False,
                "is_blocked_by_kv_budget": False,
                "active_kv_cache_entries": 2,
                "active_kv_cache_memory_bytes": 32,
                "remaining_kv_cache_budget_bytes": 48,
                "max_kv_cache_memory_bytes": 80,
            },
        )

    def test_reports_blocked_state_when_waiting_front_request_exceeds_budget(self) -> None:
        scheduler = RequestScheduler(
            max_batch_size=1,
            num_kv_layers=1,
            max_kv_cache_memory_bytes=15,
        )
        scheduler.add_request(GenerationRequest("blocked", prompt=[1], max_new_tokens=1))

        status = scheduler.status(hidden_size=4, bytes_per_element=2)

        self.assertEqual(
            status,
            {
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

    def test_reports_unbounded_budget_as_none(self) -> None:
        scheduler = RequestScheduler(max_batch_size=1, num_kv_layers=1)
        scheduler.add_request(GenerationRequest("req-1", prompt=[1], max_new_tokens=1))

        status = scheduler.status(hidden_size=4, bytes_per_element=2)

        self.assertEqual(
            status,
            {
                "waiting_count": 1,
                "active_count": 0,
                "has_work": True,
                "is_idle": False,
                "is_blocked_by_kv_budget": False,
                "active_kv_cache_entries": 0,
                "active_kv_cache_memory_bytes": 0,
                "remaining_kv_cache_budget_bytes": None,
                "max_kv_cache_memory_bytes": None,
            },
        )


if __name__ == "__main__":
    unittest.main()
