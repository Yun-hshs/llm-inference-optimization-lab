from __future__ import annotations

import unittest

from llm_opt_lab.mini_engine.scheduler import GenerationRequest, RequestScheduler


class SchedulerKVCacheAccountingTest(unittest.TestCase):
    def test_active_kv_cache_entries_sums_all_active_request_caches(self) -> None:
        scheduler = RequestScheduler(max_batch_size=2, num_kv_layers=2)
        scheduler.add_request(GenerationRequest("req-1", prompt=[1], max_new_tokens=3))
        scheduler.add_request(GenerationRequest("req-2", prompt=[2], max_new_tokens=3))
        active_requests = scheduler.activate_next_batch()
        active_requests[0].kv_cache.append(layer_index=0, key=[0.1], value=[0.2])
        active_requests[0].kv_cache.append(layer_index=1, key=[0.3], value=[0.4])
        active_requests[1].kv_cache.append(layer_index=0, key=[0.5], value=[0.6])

        self.assertEqual(scheduler.active_kv_cache_entries(), 3)

    def test_active_kv_cache_memory_bytes_sums_active_request_memory(self) -> None:
        scheduler = RequestScheduler(max_batch_size=2, num_kv_layers=2)
        scheduler.add_request(GenerationRequest("req-1", prompt=[1], max_new_tokens=3))
        scheduler.add_request(GenerationRequest("req-2", prompt=[2], max_new_tokens=3))
        active_requests = scheduler.activate_next_batch()
        active_requests[0].kv_cache.append(layer_index=0, key=[0.1], value=[0.2])
        active_requests[1].kv_cache.append(layer_index=0, key=[0.3], value=[0.4])

        memory_bytes = scheduler.active_kv_cache_memory_bytes(
            hidden_size=4,
            bytes_per_element=2,
        )

        self.assertEqual(memory_bytes, 32)

    def test_finished_requests_are_not_counted_after_removal(self) -> None:
        scheduler = RequestScheduler(max_batch_size=2, num_kv_layers=2)
        scheduler.add_request(GenerationRequest("req-1", prompt=[1], max_new_tokens=1))
        scheduler.add_request(GenerationRequest("req-2", prompt=[2], max_new_tokens=3))
        active_requests = scheduler.activate_next_batch()
        active_requests[0].kv_cache.append(layer_index=0, key=[0.1], value=[0.2])
        active_requests[1].kv_cache.append(layer_index=0, key=[0.3], value=[0.4])
        active_requests[0].append_token(7)

        scheduler.remove_finished_requests()

        self.assertEqual(scheduler.active_kv_cache_entries(), 1)


if __name__ == "__main__":
    unittest.main()
