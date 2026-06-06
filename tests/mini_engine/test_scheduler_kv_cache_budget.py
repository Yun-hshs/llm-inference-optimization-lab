from __future__ import annotations

import unittest

from llm_opt_lab.mini_engine.scheduler import GenerationRequest, RequestScheduler


class SchedulerKVCacheBudgetTest(unittest.TestCase):
    def test_reports_not_over_budget_when_usage_equals_budget(self) -> None:
        scheduler = RequestScheduler(
            max_batch_size=1,
            num_kv_layers=1,
            max_kv_cache_memory_bytes=16,
        )
        scheduler.add_request(GenerationRequest("req-1", prompt=[1], max_new_tokens=3))
        active = scheduler.activate_next_batch()[0]
        active.kv_cache.append(layer_index=0, key=[0.1], value=[0.2])

        self.assertFalse(
            scheduler.is_kv_cache_over_budget(hidden_size=4, bytes_per_element=2)
        )
        self.assertEqual(
            scheduler.remaining_kv_cache_budget_bytes(hidden_size=4, bytes_per_element=2),
            0,
        )

    def test_reports_over_budget_when_usage_exceeds_budget(self) -> None:
        scheduler = RequestScheduler(
            max_batch_size=1,
            num_kv_layers=1,
            max_kv_cache_memory_bytes=16,
        )
        scheduler.add_request(GenerationRequest("req-1", prompt=[1], max_new_tokens=3))
        active = scheduler.activate_next_batch()[0]
        active.kv_cache.append(layer_index=0, key=[0.1], value=[0.2])
        active.kv_cache.append(layer_index=0, key=[0.3], value=[0.4])

        self.assertTrue(
            scheduler.is_kv_cache_over_budget(hidden_size=4, bytes_per_element=2)
        )
        self.assertEqual(
            scheduler.remaining_kv_cache_budget_bytes(hidden_size=4, bytes_per_element=2),
            0,
        )

    def test_unbounded_scheduler_never_reports_over_budget(self) -> None:
        scheduler = RequestScheduler(max_batch_size=1, num_kv_layers=1)
        scheduler.add_request(GenerationRequest("req-1", prompt=[1], max_new_tokens=3))
        active = scheduler.activate_next_batch()[0]
        active.kv_cache.append(layer_index=0, key=[0.1], value=[0.2])

        self.assertFalse(
            scheduler.is_kv_cache_over_budget(hidden_size=4, bytes_per_element=2)
        )
        self.assertIsNone(
            scheduler.remaining_kv_cache_budget_bytes(hidden_size=4, bytes_per_element=2)
        )


if __name__ == "__main__":
    unittest.main()
