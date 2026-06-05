from __future__ import annotations

import unittest

from llm_opt_lab.mini_engine.scheduler import GenerationRequest, RequestScheduler


class SchedulerKVCacheReleaseTest(unittest.TestCase):
    def test_remove_finished_requests_clears_finished_request_kv_cache(self) -> None:
        scheduler = RequestScheduler(max_batch_size=1, num_kv_layers=2)
        scheduler.add_request(GenerationRequest("req-1", prompt=[1], max_new_tokens=1))
        active = scheduler.activate_next_batch()[0]
        active.kv_cache.append(layer_index=0, key=[0.1], value=[0.2])
        active.kv_cache.append(layer_index=1, key=[0.3], value=[0.4])
        active.append_token(7)

        finished = scheduler.remove_finished_requests()

        self.assertEqual([item.request.request_id for item in finished], ["req-1"])
        self.assertEqual(scheduler.active_count(), 0)
        self.assertEqual(finished[0].kv_cache.sequence_length(layer_index=0), 0)
        self.assertEqual(finished[0].kv_cache.sequence_length(layer_index=1), 0)

    def test_step_releases_finished_request_kv_cache_before_refill(self) -> None:
        scheduler = RequestScheduler(max_batch_size=1, num_kv_layers=2)
        scheduler.add_request(GenerationRequest("req-1", prompt=[1], max_new_tokens=1))
        scheduler.add_request(GenerationRequest("req-2", prompt=[2], max_new_tokens=1))
        active = scheduler.activate_next_batch()[0]
        active.kv_cache.append(layer_index=0, key=[0.1], value=[0.2])

        finished = scheduler.step({"req-1": 7})

        self.assertEqual([item.request.request_id for item in finished], ["req-1"])
        self.assertEqual(finished[0].kv_cache.sequence_length(layer_index=0), 0)
        self.assertEqual(scheduler.active_count(), 1)
        self.assertEqual(scheduler.active_requests[0].request.request_id, "req-2")
        self.assertIsNotNone(scheduler.active_requests[0].kv_cache)


if __name__ == "__main__":
    unittest.main()
