from __future__ import annotations

import unittest

from llm_opt_lab.mini_engine.scheduler import ActiveRequest, GenerationRequest, RequestScheduler
from llm_opt_lab.mini_engine.serving_loop import ServingLoop


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


class BudgetAwareServingLoopTest(unittest.TestCase):
    def test_run_until_blocked_or_idle_finishes_admissible_requests(self) -> None:
        scheduler = RequestScheduler(
            max_batch_size=1,
            num_kv_layers=1,
            max_kv_cache_memory_bytes=16,
        )
        scheduler.add_request(GenerationRequest("req-1", prompt=[1], max_new_tokens=1))
        scheduler.add_request(GenerationRequest("req-2", prompt=[2], max_new_tokens=1))
        token_provider = ScriptedTokenProvider({"req-1": [7], "req-2": [8]})
        loop = ServingLoop(scheduler=scheduler, token_provider=token_provider)

        finished = loop.run_until_blocked_or_idle_admissible(
            hidden_size=4,
            bytes_per_element=2,
        )

        self.assertEqual([active.request.request_id for active in finished], ["req-1", "req-2"])
        self.assertEqual([active.output_tokens() for active in finished], [[1, 7], [2, 8]])
        self.assertTrue(scheduler.is_idle())
        self.assertEqual(token_provider.call_count, 2)

    def test_run_until_blocked_or_idle_stops_when_initial_request_is_not_admissible(self) -> None:
        scheduler = RequestScheduler(
            max_batch_size=1,
            num_kv_layers=1,
            max_kv_cache_memory_bytes=15,
        )
        scheduler.add_request(GenerationRequest("blocked", prompt=[1], max_new_tokens=1))
        token_provider = ScriptedTokenProvider({"blocked": [7]})
        loop = ServingLoop(scheduler=scheduler, token_provider=token_provider)

        finished = loop.run_until_blocked_or_idle_admissible(
            hidden_size=4,
            bytes_per_element=2,
        )

        self.assertEqual(finished, [])
        self.assertEqual(scheduler.active_count(), 0)
        self.assertEqual(scheduler.waiting_count(), 1)
        self.assertEqual(scheduler.request_queue[0].request_id, "blocked")
        self.assertEqual(token_provider.call_count, 0)

    def test_run_until_blocked_or_idle_leaves_rejected_refill_request_waiting(self) -> None:
        scheduler = RequestScheduler(
            max_batch_size=1,
            num_kv_layers=1,
            max_kv_cache_memory_bytes=16,
        )
        scheduler.add_request(GenerationRequest("req-1", prompt=[1], max_new_tokens=1))
        scheduler.add_request(GenerationRequest("large", prompt=[2, 3], max_new_tokens=1))
        token_provider = ScriptedTokenProvider({"req-1": [7], "large": [8]})
        loop = ServingLoop(scheduler=scheduler, token_provider=token_provider)

        finished = loop.run_until_blocked_or_idle_admissible(
            hidden_size=4,
            bytes_per_element=2,
        )

        self.assertEqual([active.request.request_id for active in finished], ["req-1"])
        self.assertEqual(scheduler.active_count(), 0)
        self.assertEqual(scheduler.waiting_count(), 1)
        self.assertEqual(scheduler.request_queue[0].request_id, "large")
        self.assertEqual(token_provider.call_count, 1)


if __name__ == "__main__":
    unittest.main()
