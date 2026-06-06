from __future__ import annotations

import unittest

from llm_opt_lab.mini_engine.scheduler import GenerationRequest, RequestScheduler


class SchedulerRequestAdmissionTest(unittest.TestCase):
    def test_estimates_prefill_kv_cache_memory_for_request_prompt(self) -> None:
        scheduler = RequestScheduler(max_batch_size=1, num_kv_layers=2)
        request = GenerationRequest("req-1", prompt=[1, 2, 3], max_new_tokens=2)

        memory_bytes = scheduler.estimate_request_prefill_kv_memory_bytes(
            request,
            hidden_size=4,
            bytes_per_element=2,
        )

        self.assertEqual(memory_bytes, 96)

    def test_can_admit_request_when_projected_usage_equals_budget(self) -> None:
        scheduler = RequestScheduler(
            max_batch_size=1,
            num_kv_layers=1,
            max_kv_cache_memory_bytes=32,
        )
        scheduler.add_request(GenerationRequest("active-1", prompt=[1], max_new_tokens=3))
        active = scheduler.activate_next_batch()[0]
        active.kv_cache.append(layer_index=0, key=[0.1], value=[0.2])
        next_request = GenerationRequest("req-2", prompt=[2], max_new_tokens=2)

        self.assertTrue(
            scheduler.can_admit_request(
                next_request,
                hidden_size=4,
                bytes_per_element=2,
            )
        )

    def test_rejects_request_when_projected_usage_exceeds_budget(self) -> None:
        scheduler = RequestScheduler(
            max_batch_size=1,
            num_kv_layers=1,
            max_kv_cache_memory_bytes=31,
        )
        scheduler.add_request(GenerationRequest("active-1", prompt=[1], max_new_tokens=3))
        active = scheduler.activate_next_batch()[0]
        active.kv_cache.append(layer_index=0, key=[0.1], value=[0.2])
        next_request = GenerationRequest("req-2", prompt=[2], max_new_tokens=2)

        self.assertFalse(
            scheduler.can_admit_request(
                next_request,
                hidden_size=4,
                bytes_per_element=2,
            )
        )

    def test_unbounded_scheduler_can_admit_request(self) -> None:
        scheduler = RequestScheduler(max_batch_size=1, num_kv_layers=1)
        request = GenerationRequest("req-1", prompt=[1, 2, 3], max_new_tokens=2)

        self.assertTrue(
            scheduler.can_admit_request(
                request,
                hidden_size=4,
                bytes_per_element=2,
            )
        )


if __name__ == "__main__":
    unittest.main()
