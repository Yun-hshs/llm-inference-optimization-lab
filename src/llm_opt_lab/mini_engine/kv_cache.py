from __future__ import annotations
from dataclasses import dataclass

# 简写dataclass装饰器，自动生成__init__等方法
@dataclass
class LayerKVCache:
    # 多头注意力机制中每个头的键和值的缓存
    keys: list[list[float]]
    values: list[list[float]]

class KVCache:
    def __init__(self, num_layers: int) -> None:
        # 初始化每层的KV缓存
        self.layers = [LayerKVCache([], []) for _ in range(num_layers)]
    def clear(self) -> None:
        # 清空所有层的KV缓存
        for layer_cache in self.layers:
            layer_cache.keys.clear()
            layer_cache.values.clear()
    def _validate_layer_index(self, layer_index: int) -> None:
        if not (0 <= layer_index < len(self.layers)):
            # 注意这里：改成测试所期待的文案！
            raise IndexError(f"layer_index must be in [0, {len(self.layers)})")

    def append(self, layer_index: int, key: list[float], value: list[float]) -> None:
       self._validate_layer_index(layer_index)
       # 将新的键和值添加到指定层的缓存中
       # 先找到目标层的缓存
       target_layer_cache = self.layers[layer_index]
       # 将新的键和值添加到缓存中
       target_layer_cache.keys.append(key)
       target_layer_cache.values.append(value)

    def get_layer(self, layer_index: int) -> LayerKVCache:
        self._validate_layer_index(layer_index)
        return self.layers[layer_index]

    def sequence_length(self, layer_index: int) -> int:
        self._validate_layer_index(layer_index)
        # 假设每个键对应一个输入token，因此序列长度等于键的数量
        return len(self.layers[layer_index].keys)  

    def total_entries(self) -> int:
        # TODO: Handwrite Milestone 17 core logic here.
        total = 0
        for layer_cache in self.layers:
            # 每个层的KV缓存中，键和值的数量应该相同，因此可以任选其一来计算总条目数
            if len(layer_cache.keys) != len(layer_cache.values):
                raise ValueError("Inconsistent KV cache: keys and values count mismatch.")
        # 计算所有层的KV缓存中的总条目数        
            total += len(layer_cache.keys)
        return total
        # Count cached token entries across all layers

    def estimate_memory_bytes(self, *, hidden_size: int, bytes_per_element: int) -> int:
        # TODO: Handwrite Milestone 17 core logic here.
        # KV cache stores both key and value vectors for each cached entry.
        return self.total_entries() * hidden_size * bytes_per_element * 2
    
