from __future__ import annotations

import unittest

from llm_opt_lab.mini_engine.scheduler import ActiveRequest, GenerationRequest, RequestScheduler
from llm_opt_lab.mini_engine.serving_loop import ServingLoop, ServingLoopResult


class ScriptedTokenProvider:
    def __init__(self, scripted_tokens: dict[str, list[int]]) -> None:
        self.scripted_tokens = {request_id: list(tokens) for request_id, tokens in scripted_tokens.items()}
        self.call_count = 0

    def __call__(self, active_requests: list[ActiveRequest]) -> dict[str, int]:
        self.call_count += 1
        tokens: dict[str, int] = {}
        for active in active_requests:
            request_id = active.request.request_id
            tokens[request_id] = self.scripted_tokens[request_id].pop(0)
        return tokens


class ServingLoopResultTest(unittest.TestCase):
    def test_returns_finished_requests_and_idle_final_status(self) -> None:
        scheduler = RequestScheduler(
            max_batch_size=1,
            num_kv_layers=1,
            max_kv_cache_memory_bytes=16,
        )
        scheduler.add_request(GenerationRequest("req-1", prompt=[1], max_new_tokens=1))
        token_provider = ScriptedTokenProvider({"req-1": [7]})
        loop = ServingLoop(scheduler=scheduler, token_provider=token_provider)

        result = loop.run_until_blocked_or_idle_admissible_with_status(
            hidden_size=4,
            bytes_per_element=2,
        )

        self.assertIsInstance(result, ServingLoopResult)
        self.assertEqual([active.request.request_id for active in result.finished_requests], ["req-1"])
        self.assertEqual([active.output_tokens() for active in result.finished_requests], [[1, 7]])
        self.assertEqual(
            result.final_status,
            {
                "waiting_count": 0,
                "active_count": 0,
                "has_work": False,
                "is_idle": True,
                "is_blocked_by_kv_budget": False,
                "active_kv_cache_entries": 0,
                "active_kv_cache_memory_bytes": 0,
                "remaining_kv_cache_budget_bytes": 16,
                "max_kv_cache_memory_bytes": 16,
            },
        )
        self.assertEqual(token_provider.call_count, 1)

    def test_returns_empty_finished_requests_and_blocked_final_status(self) -> None:
        scheduler = RequestScheduler(
            max_batch_size=1,
            num_kv_layers=1,
            max_kv_cache_memory_bytes=15,
        )
        scheduler.add_request(GenerationRequest("blocked", prompt=[1], max_new_tokens=1))
        token_provider = ScriptedTokenProvider({"blocked": [7]})
        loop = ServingLoop(scheduler=scheduler, token_provider=token_provider)

        result = loop.run_until_blocked_or_idle_admissible_with_status(
            hidden_size=4,
            bytes_per_element=2,
        )

        self.assertEqual(result.finished_requests, [])
        self.assertEqual(
            result.final_status,
            {
                "waiting_count": 1,
                "active_count": 0,
                "has_work": True,
                "is_idle": False,
                "is_blocked_by_kv_budget": True,
                "active_kv_cache_entries": 0,
                "active_kv_cache_memory_bytes": 0,
                "remaining_kv_cache_budget_bytes": 15,
                "max_kv_cache_memory_bytes": 15,
            },
        )
        self.assertEqual(token_provider.call_count, 0)


if __name__ == "__main__":
    unittest.main()
