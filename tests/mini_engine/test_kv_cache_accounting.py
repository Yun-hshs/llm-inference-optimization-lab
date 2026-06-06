from __future__ import annotations

import unittest

from llm_opt_lab.mini_engine.kv_cache import KVCache


class KVCacheAccountingTest(unittest.TestCase):
    def test_total_entries_counts_cached_tokens_across_layers(self) -> None:
        cache = KVCache(num_layers=2)
        cache.append(layer_index=0, key=[0.1], value=[0.2])
        cache.append(layer_index=0, key=[0.3], value=[0.4])
        cache.append(layer_index=1, key=[0.5], value=[0.6])

        self.assertEqual(cache.total_entries(), 3)

    def test_estimate_memory_bytes_uses_key_and_value_storage(self) -> None:
        cache = KVCache(num_layers=2)
        cache.append(layer_index=0, key=[0.1, 0.2], value=[0.3, 0.4])
        cache.append(layer_index=1, key=[0.5, 0.6], value=[0.7, 0.8])

        memory_bytes = cache.estimate_memory_bytes(hidden_size=4, bytes_per_element=2)

        self.assertEqual(memory_bytes, 32)

    def test_clear_resets_accounting(self) -> None:
        cache = KVCache(num_layers=2)
        cache.append(layer_index=0, key=[0.1], value=[0.2])
        cache.append(layer_index=1, key=[0.3], value=[0.4])

        cache.clear()

        self.assertEqual(cache.total_entries(), 0)
        self.assertEqual(cache.estimate_memory_bytes(hidden_size=4, bytes_per_element=2), 0)


if __name__ == "__main__":
    unittest.main()
