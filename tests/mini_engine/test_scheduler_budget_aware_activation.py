from __future__ import annotations

import unittest

from llm_opt_lab.mini_engine.scheduler import GenerationRequest, RequestScheduler


class SchedulerBudgetAwareActivationTest(unittest.TestCase):
    def test_activates_waiting_request_when_budget_allows_it(self) -> None:
        scheduler = RequestScheduler(
            max_batch_size=1,
            num_kv_layers=1,
            max_kv_cache_memory_bytes=16,
        )
        scheduler.add_request(GenerationRequest("req-1", prompt=[1], max_new_tokens=2))

        activated = scheduler.activate_next_admissible_batch(
            hidden_size=4,
            bytes_per_element=2,
        )

        self.assertEqual([active.request.request_id for active in activated], ["req-1"])
        self.assertEqual(scheduler.active_count(), 1)
        self.assertEqual(scheduler.waiting_count(), 0)

    def test_keeps_front_request_waiting_when_budget_rejects_it(self) -> None:
        scheduler = RequestScheduler(
            max_batch_size=1,
            num_kv_layers=1,
            max_kv_cache_memory_bytes=15,
        )
        scheduler.add_request(GenerationRequest("req-1", prompt=[1], max_new_tokens=2))

        activated = scheduler.activate_next_admissible_batch(
            hidden_size=4,
            bytes_per_element=2,
        )

        self.assertEqual(activated, [])
        self.assertEqual(scheduler.active_count(), 0)
        self.assertEqual(scheduler.waiting_count(), 1)
        self.assertEqual(scheduler.request_queue[0].request_id, "req-1")

    def test_does_not_skip_rejected_front_request_to_activate_later_request(self) -> None:
        scheduler = RequestScheduler(
            max_batch_size=2,
            num_kv_layers=1,
            max_kv_cache_memory_bytes=16,
        )
        scheduler.add_request(GenerationRequest("large", prompt=[1, 2], max_new_tokens=2))
        scheduler.add_request(GenerationRequest("small", prompt=[3], max_new_tokens=2))

        activated = scheduler.activate_next_admissible_batch(
            hidden_size=4,
            bytes_per_element=2,
        )

        self.assertEqual(activated, [])
        self.assertEqual(scheduler.active_count(), 0)
        self.assertEqual([request.request_id for request in scheduler.request_queue], ["large", "small"])

    def test_stops_when_active_batch_reaches_capacity(self) -> None:
        scheduler = RequestScheduler(
            max_batch_size=2,
            num_kv_layers=1,
            max_kv_cache_memory_bytes=64,
        )
        scheduler.add_request(GenerationRequest("req-1", prompt=[1], max_new_tokens=2))
        scheduler.add_request(GenerationRequest("req-2", prompt=[2], max_new_tokens=2))
        scheduler.add_request(GenerationRequest("req-3", prompt=[3], max_new_tokens=2))

        activated = scheduler.activate_next_admissible_batch(
            hidden_size=4,
            bytes_per_element=2,
        )

        self.assertEqual([active.request.request_id for active in activated], ["req-1", "req-2"])
        self.assertEqual(scheduler.active_count(), 2)
        self.assertEqual([request.request_id for request in scheduler.request_queue], ["req-3"])


if __name__ == "__main__":
    unittest.main()
