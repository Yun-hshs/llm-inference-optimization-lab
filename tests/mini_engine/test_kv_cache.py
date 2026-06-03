from __future__ import annotations

import unittest

from llm_opt_lab.mini_engine.kv_cache import KVCache


class KVCacheTest(unittest.TestCase):
    def test_append_stores_key_and_value_for_one_layer(self) -> None:
        cache = KVCache(num_layers=2)

        cache.append(layer_index=0, key=[0.1, 0.2], value=[0.3, 0.4])
        cache.append(layer_index=0, key=[0.5, 0.6], value=[0.7, 0.8])

        layer = cache.get_layer(layer_index=0)
        self.assertEqual(layer.keys, [[0.1, 0.2], [0.5, 0.6]])
        self.assertEqual(layer.values, [[0.3, 0.4], [0.7, 0.8]])
        self.assertEqual(cache.sequence_length(layer_index=0), 2)

    def test_layers_are_stored_independently(self) -> None:
        cache = KVCache(num_layers=2)

        cache.append(layer_index=0, key=[0.1], value=[0.2])
        cache.append(layer_index=1, key=[0.3], value=[0.4])

        self.assertEqual(cache.get_layer(layer_index=0).keys, [[0.1]])
        self.assertEqual(cache.get_layer(layer_index=1).keys, [[0.3]])
        self.assertEqual(cache.sequence_length(layer_index=0), 1)
        self.assertEqual(cache.sequence_length(layer_index=1), 1)

    def test_rejects_out_of_range_layer_index(self) -> None:
        cache = KVCache(num_layers=2)

        with self.assertRaisesRegex(IndexError, "layer_index must be in"):
            cache.append(layer_index=2, key=[0.1], value=[0.2])


if __name__ == "__main__":
    unittest.main()
