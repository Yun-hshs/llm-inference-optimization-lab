from __future__ import annotations

import unittest

from llm_opt_lab.mini_engine.scheduler import GenerationRequest, RequestScheduler


class SchedulerLifecycleTest(unittest.TestCase):
    def test_has_work_tracks_waiting_and_active_requests(self) -> None:
        scheduler = RequestScheduler(max_batch_size=2)

        self.assertFalse(scheduler.has_work())
        self.assertTrue(scheduler.is_idle())

        scheduler.add_request(GenerationRequest("req-1", prompt=[1], max_new_tokens=1))
        self.assertTrue(scheduler.has_work())
        self.assertFalse(scheduler.is_idle())

        scheduler.activate_next_batch()
        self.assertTrue(scheduler.has_work())
        self.assertFalse(scheduler.is_idle())

        finished = scheduler.step({"req-1": 7})

        self.assertEqual([active.request.request_id for active in finished], ["req-1"])
        self.assertFalse(scheduler.has_work())
        self.assertTrue(scheduler.is_idle())


if __name__ == "__main__":
    unittest.main()
