from __future__ import annotations

import unittest

from llm_opt_lab.mini_engine.scheduler import GenerationRequest, RequestScheduler


class DecodeStepSchedulerTest(unittest.TestCase):
    def test_apply_tokens_updates_each_active_request(self) -> None:
        scheduler = RequestScheduler(max_batch_size=2)
        scheduler.add_request(GenerationRequest("req-1", prompt=[1], max_new_tokens=2))
        scheduler.add_request(GenerationRequest("req-2", prompt=[2], max_new_tokens=1))
        scheduler.activate_next_batch()

        scheduler.apply_tokens_to_active_requests({"req-1": 7, "req-2": 8})

        self.assertEqual(
            [active.output_tokens() for active in scheduler.active_requests],
            [[1, 7], [2, 8]],
        )
        self.assertFalse(scheduler.active_requests[0].finished)
        self.assertTrue(scheduler.active_requests[1].finished)

    def test_apply_tokens_rejects_missing_request_token(self) -> None:
        scheduler = RequestScheduler(max_batch_size=1)
        scheduler.add_request(GenerationRequest("req-1", prompt=[1], max_new_tokens=2))
        scheduler.activate_next_batch()

        with self.assertRaisesRegex(KeyError, "missing token for active request req-1"):
            scheduler.apply_tokens_to_active_requests({})


if __name__ == "__main__":
    unittest.main()
