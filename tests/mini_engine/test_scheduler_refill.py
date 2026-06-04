from __future__ import annotations

import unittest

from llm_opt_lab.mini_engine.scheduler import GenerationRequest, RequestScheduler


class SchedulerRefillTest(unittest.TestCase):
    def test_refill_removes_finished_requests_and_fills_open_slots(self) -> None:
        scheduler = RequestScheduler(max_batch_size=2)
        scheduler.add_request(GenerationRequest("req-1", prompt=[1], max_new_tokens=1))
        scheduler.add_request(GenerationRequest("req-2", prompt=[2], max_new_tokens=3))
        scheduler.add_request(GenerationRequest("req-3", prompt=[3], max_new_tokens=2))

        active_batch = scheduler.activate_next_batch()
        active_batch[0].append_token(9)

        added = scheduler.refill_active_batch()

        self.assertEqual([active.request.request_id for active in scheduler.active_requests], ["req-2", "req-3"])
        self.assertEqual([active.request.request_id for active in added], ["req-3"])
        self.assertEqual(scheduler.active_count(), 2)
        self.assertEqual(scheduler.waiting_count(), 0)


if __name__ == "__main__":
    unittest.main()
