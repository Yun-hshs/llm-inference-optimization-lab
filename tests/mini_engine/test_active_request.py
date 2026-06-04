from __future__ import annotations

import unittest

from llm_opt_lab.mini_engine.scheduler import ActiveRequest, GenerationRequest


class ActiveRequestTest(unittest.TestCase):
    def test_tracks_generated_tokens_and_marks_finished_at_limit(self) -> None:
        request = GenerationRequest(request_id="req-1", prompt=[1, 2], max_new_tokens=2)
        active = ActiveRequest.from_request(request)

        active.append_token(7)
        self.assertFalse(active.finished)
        self.assertEqual(active.output_tokens(), [1, 2, 7])

        active.append_token(8)
        self.assertTrue(active.finished)
        self.assertEqual(active.output_tokens(), [1, 2, 7, 8])

    def test_marks_finished_when_eos_token_is_generated(self) -> None:
        request = GenerationRequest(
            request_id="req-2",
            prompt=[3],
            max_new_tokens=5,
            eos_token_id=2,
        )
        active = ActiveRequest.from_request(request)

        active.append_token(2)

        self.assertTrue(active.finished)
        self.assertEqual(active.output_tokens(), [3, 2])


if __name__ == "__main__":
    unittest.main()
