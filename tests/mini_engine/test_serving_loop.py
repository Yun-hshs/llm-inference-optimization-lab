from __future__ import annotations

import unittest

from llm_opt_lab.mini_engine.scheduler import ActiveRequest, GenerationRequest, RequestScheduler
from llm_opt_lab.mini_engine.serving_loop import ServingLoop


class ScriptedTokenProvider:
    def __init__(self, scripted_tokens: dict[str, list[int]]) -> None:
        self.scripted_tokens = {request_id: list(tokens) for request_id, tokens in scripted_tokens.items()}

    def __call__(self, active_requests: list[ActiveRequest]) -> dict[str, int]:
        tokens: dict[str, int] = {}
        for active in active_requests:
            request_id = active.request.request_id
            tokens[request_id] = self.scripted_tokens[request_id].pop(0)
        return tokens


class ServingLoopTest(unittest.TestCase):
    def test_run_until_idle_returns_all_finished_requests(self) -> None:
        scheduler = RequestScheduler(max_batch_size=2)
        scheduler.add_request(GenerationRequest("req-1", prompt=[1], max_new_tokens=1))
        scheduler.add_request(GenerationRequest("req-2", prompt=[2], max_new_tokens=2))
        scheduler.add_request(GenerationRequest("req-3", prompt=[3], max_new_tokens=1))
        token_provider = ScriptedTokenProvider(
            {
                "req-1": [7],
                "req-2": [8, 9],
                "req-3": [6],
            }
        )
        loop = ServingLoop(scheduler=scheduler, token_provider=token_provider)

        finished = loop.run_until_idle()

        self.assertEqual([active.request.request_id for active in finished], ["req-1", "req-2", "req-3"])
        self.assertEqual([active.output_tokens() for active in finished], [[1, 7], [2, 8, 9], [3, 6]])
        self.assertTrue(scheduler.is_idle())


if __name__ == "__main__":
    unittest.main()
