from __future__ import annotations

import unittest

from llm_opt_lab.mini_engine.scheduler import GenerationRequest, RequestScheduler


class SchedulerBudgetAwareRefillTest(unittest.TestCase):
    def test_refill_removes_finished_request_and_activates_admissible_waiting_request(self) -> None:
        scheduler = RequestScheduler(
            max_batch_size=1,
            num_kv_layers=1,
            max_kv_cache_memory_bytes=16,
        )
        scheduler.add_request(GenerationRequest("req-1", prompt=[1], max_new_tokens=1))
        scheduler.add_request(GenerationRequest("req-2", prompt=[2], max_new_tokens=2))
        active = scheduler.activate_next_batch()[0]
        active.kv_cache.append(layer_index=0, key=[0.1], value=[0.2])
        active.append_token(7)

        activated = scheduler.refill_active_batch_admissible(
            hidden_size=4,
            bytes_per_element=2,
        )

        self.assertEqual([item.request.request_id for item in activated], ["req-2"])
        self.assertEqual(scheduler.active_count(), 1)
        self.assertEqual(scheduler.active_requests[0].request.request_id, "req-2")
        self.assertEqual(scheduler.waiting_count(), 0)

    def test_refill_keeps_front_request_waiting_when_budget_rejects_it(self) -> None:
        scheduler = RequestScheduler(
            max_batch_size=1,
            num_kv_layers=1,
            max_kv_cache_memory_bytes=15,
        )
        scheduler.add_request(GenerationRequest("req-1", prompt=[1], max_new_tokens=1))
        scheduler.add_request(GenerationRequest("req-2", prompt=[2], max_new_tokens=2))
        active = scheduler.activate_next_batch()[0]
        active.append_token(7)

        activated = scheduler.refill_active_batch_admissible(
            hidden_size=4,
            bytes_per_element=2,
        )

        self.assertEqual(activated, [])
        self.assertEqual(scheduler.active_count(), 0)
        self.assertEqual(scheduler.waiting_count(), 1)
        self.assertEqual(scheduler.request_queue[0].request_id, "req-2")

    def test_refill_does_not_skip_rejected_front_request_to_activate_later_request(self) -> None:
        scheduler = RequestScheduler(
            max_batch_size=1,
            num_kv_layers=1,
            max_kv_cache_memory_bytes=16,
        )
        scheduler.add_request(GenerationRequest("finished", prompt=[0], max_new_tokens=1))
        scheduler.add_request(GenerationRequest("large", prompt=[1, 2], max_new_tokens=2))
        scheduler.add_request(GenerationRequest("small", prompt=[3], max_new_tokens=2))
        active = scheduler.activate_next_batch()[0]
        active.append_token(7)

        activated = scheduler.refill_active_batch_admissible(
            hidden_size=4,
            bytes_per_element=2,
        )

        self.assertEqual(activated, [])
        self.assertEqual(scheduler.active_count(), 0)
        self.assertEqual(
            [request.request_id for request in scheduler.request_queue],
            ["large", "small"],
        )

    def test_refill_respects_existing_active_request_capacity(self) -> None:
        scheduler = RequestScheduler(
            max_batch_size=2,
            num_kv_layers=1,
            max_kv_cache_memory_bytes=64,
        )
        scheduler.add_request(GenerationRequest("active", prompt=[1], max_new_tokens=3))
        scheduler.activate_next_batch()
        scheduler.add_request(GenerationRequest("waiting", prompt=[2], max_new_tokens=2))

        activated = scheduler.refill_active_batch_admissible(
            hidden_size=4,
            bytes_per_element=2,
        )

        self.assertEqual([item.request.request_id for item in activated], ["waiting"])
        self.assertEqual(scheduler.active_count(), 2)
        self.assertEqual(scheduler.waiting_count(), 0)


if __name__ == "__main__":
    unittest.main()
