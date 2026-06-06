from __future__ import annotations

import unittest

from llm_opt_lab.mini_engine.scheduler import GenerationRequest, RequestScheduler


class SchedulerBudgetAwareStepTest(unittest.TestCase):
    def test_step_applies_tokens_returns_finished_and_refills_admissible_request(self) -> None:
        scheduler = RequestScheduler(
            max_batch_size=1,
            num_kv_layers=1,
            max_kv_cache_memory_bytes=16,
        )
        scheduler.add_request(GenerationRequest("req-1", prompt=[1], max_new_tokens=1))
        scheduler.add_request(GenerationRequest("req-2", prompt=[2], max_new_tokens=2))
        scheduler.activate_next_batch()

        finished = scheduler.step_admissible(
            {"req-1": 7},
            hidden_size=4,
            bytes_per_element=2,
        )

        self.assertEqual([active.request.request_id for active in finished], ["req-1"])
        self.assertEqual(finished[0].output_tokens(), [1, 7])
        self.assertEqual([active.request.request_id for active in scheduler.active_requests], ["req-2"])
        self.assertEqual(scheduler.waiting_count(), 0)

    def test_step_keeps_rejected_waiting_request_queued_after_finished_request_removal(self) -> None:
        scheduler = RequestScheduler(
            max_batch_size=1,
            num_kv_layers=1,
            max_kv_cache_memory_bytes=15,
        )
        scheduler.add_request(GenerationRequest("req-1", prompt=[1], max_new_tokens=1))
        scheduler.add_request(GenerationRequest("req-2", prompt=[2], max_new_tokens=2))
        scheduler.activate_next_batch()

        finished = scheduler.step_admissible(
            {"req-1": 7},
            hidden_size=4,
            bytes_per_element=2,
        )

        self.assertEqual([active.request.request_id for active in finished], ["req-1"])
        self.assertEqual(scheduler.active_count(), 0)
        self.assertEqual(scheduler.waiting_count(), 1)
        self.assertEqual(scheduler.request_queue[0].request_id, "req-2")

    def test_step_rejects_missing_token_before_refill(self) -> None:
        scheduler = RequestScheduler(
            max_batch_size=1,
            num_kv_layers=1,
            max_kv_cache_memory_bytes=16,
        )
        scheduler.add_request(GenerationRequest("req-1", prompt=[1], max_new_tokens=1))
        scheduler.activate_next_batch()

        with self.assertRaisesRegex(KeyError, "missing token for active request req-1"):
            scheduler.step_admissible(
                {},
                hidden_size=4,
                bytes_per_element=2,
            )


if __name__ == "__main__":
    unittest.main()
