from __future__ import annotations

import unittest

from llm_opt_lab.mini_engine.scheduler import GenerationRequest, RequestScheduler


class SchedulerBlockedStateTest(unittest.TestCase):
    def test_reports_blocked_when_waiting_front_request_exceeds_budget_and_no_active_requests(self) -> None:
        scheduler = RequestScheduler(
            max_batch_size=1,
            num_kv_layers=1,
            max_kv_cache_memory_bytes=15,
        )
        scheduler.add_request(GenerationRequest("blocked", prompt=[1], max_new_tokens=1))

        self.assertTrue(
            scheduler.is_blocked_by_kv_budget(hidden_size=4, bytes_per_element=2)
        )
        self.assertTrue(scheduler.has_work())
        self.assertFalse(scheduler.is_idle())

    def test_reports_not_blocked_when_scheduler_is_idle(self) -> None:
        scheduler = RequestScheduler(
            max_batch_size=1,
            num_kv_layers=1,
            max_kv_cache_memory_bytes=15,
        )

        self.assertFalse(
            scheduler.is_blocked_by_kv_budget(hidden_size=4, bytes_per_element=2)
        )
        self.assertFalse(scheduler.has_work())
        self.assertTrue(scheduler.is_idle())

    def test_reports_not_blocked_when_active_request_can_continue_decoding(self) -> None:
        scheduler = RequestScheduler(
            max_batch_size=1,
            num_kv_layers=1,
            max_kv_cache_memory_bytes=15,
        )
        scheduler.add_request(GenerationRequest("active", prompt=[1], max_new_tokens=2))
        scheduler.activate_next_batch()

        self.assertFalse(
            scheduler.is_blocked_by_kv_budget(hidden_size=4, bytes_per_element=2)
        )

    def test_reports_not_blocked_when_front_waiting_request_is_admissible(self) -> None:
        scheduler = RequestScheduler(
            max_batch_size=1,
            num_kv_layers=1,
            max_kv_cache_memory_bytes=16,
        )
        scheduler.add_request(GenerationRequest("admissible", prompt=[1], max_new_tokens=1))

        self.assertFalse(
            scheduler.is_blocked_by_kv_budget(hidden_size=4, bytes_per_element=2)
        )

    def test_unbounded_scheduler_is_not_blocked_by_budget(self) -> None:
        scheduler = RequestScheduler(max_batch_size=1, num_kv_layers=1)
        scheduler.add_request(GenerationRequest("req-1", prompt=[1, 2, 3], max_new_tokens=1))

        self.assertFalse(
            scheduler.is_blocked_by_kv_budget(hidden_size=4, bytes_per_element=2)
        )


if __name__ == "__main__":
    unittest.main()
