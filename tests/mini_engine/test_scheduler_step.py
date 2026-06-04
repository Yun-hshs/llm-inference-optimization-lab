from __future__ import annotations

import unittest

from llm_opt_lab.mini_engine.scheduler import GenerationRequest, RequestScheduler


class SchedulerStepTest(unittest.TestCase):
    def test_step_applies_tokens_returns_finished_and_refills_batch(self) -> None:
        scheduler = RequestScheduler(max_batch_size=2)
        scheduler.add_request(GenerationRequest("req-1", prompt=[1], max_new_tokens=1))
        scheduler.add_request(GenerationRequest("req-2", prompt=[2], max_new_tokens=3))
        scheduler.add_request(GenerationRequest("req-3", prompt=[3], max_new_tokens=2))
        scheduler.activate_next_batch()

        finished = scheduler.step({"req-1": 9, "req-2": 8})

        self.assertEqual([active.request.request_id for active in finished], ["req-1"])
        self.assertEqual(finished[0].output_tokens(), [1, 9])
        self.assertEqual([active.request.request_id for active in scheduler.active_requests], ["req-2", "req-3"])
        self.assertEqual(scheduler.active_requests[0].output_tokens(), [2, 8])
        self.assertEqual(scheduler.active_requests[1].output_tokens(), [3])
        self.assertEqual(scheduler.active_count(), 2)
        self.assertEqual(scheduler.waiting_count(), 0)


if __name__ == "__main__":
    unittest.main()
