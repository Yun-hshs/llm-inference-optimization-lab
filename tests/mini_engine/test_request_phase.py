from __future__ import annotations

import unittest

from llm_opt_lab.mini_engine.scheduler import ActiveRequest, GenerationRequest


class RequestPhaseTest(unittest.TestCase):
    def test_new_active_request_starts_in_prefill_phase(self) -> None:
        active = ActiveRequest.from_request(
            GenerationRequest("req-1", prompt=[1, 2, 3], max_new_tokens=2)
        )

        self.assertEqual(active.phase(), "prefill")

    def test_active_request_moves_to_decode_phase_after_first_generated_token(self) -> None:
        active = ActiveRequest.from_request(
            GenerationRequest("req-1", prompt=[1, 2, 3], max_new_tokens=2)
        )

        active.append_token(7)

        self.assertEqual(active.phase(), "decode")
        self.assertEqual(active.output_tokens(), [1, 2, 3, 7])


if __name__ == "__main__":
    unittest.main()
