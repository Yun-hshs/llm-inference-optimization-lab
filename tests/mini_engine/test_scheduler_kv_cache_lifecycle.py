from __future__ import annotations

import unittest

from llm_opt_lab.mini_engine.scheduler import GenerationRequest, RequestScheduler


class SchedulerKVCacheLifecycleTest(unittest.TestCase):
    def test_activate_next_batch_allocates_kv_cache_for_active_requests(self) -> None:
        scheduler = RequestScheduler(max_batch_size=2, num_kv_layers=2)
        scheduler.add_request(GenerationRequest("req-1", prompt=[1, 2], max_new_tokens=2))

        active_batch = scheduler.activate_next_batch()

        self.assertEqual(len(active_batch), 1)
        active = active_batch[0]
        self.assertIsNotNone(active.kv_cache)
        self.assertEqual(active.kv_cache.sequence_length(layer_index=0), 0)
        self.assertEqual(active.kv_cache.sequence_length(layer_index=1), 0)

    def test_refill_active_batch_allocates_kv_cache_for_new_active_request(self) -> None:
        scheduler = RequestScheduler(max_batch_size=1, num_kv_layers=2)
        scheduler.add_request(GenerationRequest("req-1", prompt=[1], max_new_tokens=1))
        scheduler.add_request(GenerationRequest("req-2", prompt=[2], max_new_tokens=1))
        active_batch = scheduler.activate_next_batch()
        active_batch[0].append_token(7)

        scheduler.refill_active_batch()

        self.assertEqual(scheduler.active_count(), 1)
        active = scheduler.active_requests[0]
        self.assertEqual(active.request.request_id, "req-2")
        self.assertIsNotNone(active.kv_cache)
        self.assertEqual(active.kv_cache.sequence_length(layer_index=0), 0)
        self.assertEqual(active.kv_cache.sequence_length(layer_index=1), 0)


if __name__ == "__main__":
    unittest.main()
