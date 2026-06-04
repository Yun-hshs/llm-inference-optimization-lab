from __future__ import annotations

import unittest

from llm_opt_lab.mini_engine.scheduler import ActiveRequest, GenerationRequest, RequestScheduler


class ActiveBatchSchedulerTest(unittest.TestCase):
    def test_activate_next_batch_moves_waiting_requests_to_active_requests(self) -> None:
        scheduler = RequestScheduler(max_batch_size=2)
        scheduler.add_request(GenerationRequest("req-1", prompt=[1, 2], max_new_tokens=3))
        scheduler.add_request(GenerationRequest("req-2", prompt=[3], max_new_tokens=2))
        scheduler.add_request(GenerationRequest("req-3", prompt=[4], max_new_tokens=1))

        active_batch = scheduler.activate_next_batch()

        self.assertEqual([active.request.request_id for active in active_batch], ["req-1", "req-2"])
        self.assertTrue(all(isinstance(active, ActiveRequest) for active in active_batch))
        self.assertEqual(scheduler.active_count(), 2)
        self.assertEqual(scheduler.waiting_count(), 1)


if __name__ == "__main__":
    unittest.main()
