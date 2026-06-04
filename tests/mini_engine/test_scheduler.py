from __future__ import annotations

import unittest

from llm_opt_lab.mini_engine.scheduler import GenerationRequest, RequestScheduler


class RequestSchedulerTest(unittest.TestCase):
    def test_rejects_non_positive_batch_size(self) -> None:
        with self.assertRaisesRegex(ValueError, "max_batch_size must be positive"):
            RequestScheduler(max_batch_size=0)

    def test_next_batch_returns_requests_in_fifo_order(self) -> None:
        scheduler = RequestScheduler(max_batch_size=2)
        first = GenerationRequest(request_id="req-1", prompt=[1, 2], max_new_tokens=4)
        second = GenerationRequest(request_id="req-2", prompt=[3], max_new_tokens=2)
        third = GenerationRequest(request_id="req-3", prompt=[4, 5], max_new_tokens=1)

        scheduler.add_request(first)
        scheduler.add_request(second)
        scheduler.add_request(third)

        batch = scheduler.next_batch()

        self.assertEqual([request.request_id for request in batch], ["req-1", "req-2"])
        self.assertEqual(scheduler.waiting_count(), 1)


if __name__ == "__main__":
    unittest.main()
