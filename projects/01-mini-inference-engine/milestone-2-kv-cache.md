# Milestone 2: Minimal KV Cache

## Goal

实现一个教学版 KV Cache 数据结构，用来模拟 decoder-only LLM 推理中每一层保存历史 token 的 key/value。

这一阶段先不实现 attention 计算，也不引入真实张量。重点是理解 KV Cache 的组织方式：

- 模型有多层 Transformer block
- 每一层都有自己的 keys 和 values
- 每生成或处理一个 token，就可以向对应层追加一组 key/value
- decode 阶段可以复用这些历史 key/value，避免重复计算历史 token

## What Was Built

新增了 `KVCache` 和 `LayerKVCache`。

支持：

- 初始化指定层数的 KV cache
- 向某一层追加 key/value
- 读取某一层的缓存
- 查询某一层缓存的 token 数量
- 检查 layer index 越界并抛出清晰错误
- 清空所有层的缓存

## Data Shape

当前教学版使用 Python list 表达缓存：

```python
LayerKVCache(
    keys=[
        [0.1, 0.2],
        [0.5, 0.6],
    ],
    values=[
        [0.3, 0.4],
        [0.7, 0.8],
    ],
)
```

其中：

- 外层 list 表示 token 维度
- 内层 list 表示单个 token 的 key 或 value 向量

真实 LLM 中通常会是更高维的张量，例如：

```text
[num_layers, batch_size, num_heads, sequence_length, head_dim]
```

这个里程碑先保留最小形状，方便理解数据流。

## TDD Process

### RED 1: KVCache Constructor

测试输入：

```python
cache = KVCache(num_layers=2)
```

初始失败：

```text
TypeError: KVCache() takes no arguments
```

### GREEN 1: Store Keys and Values

实现：

- `KVCache.__init__(num_layers)`
- `KVCache.append(layer_index, key, value)`
- `KVCache.get_layer(layer_index)`
- `KVCache.sequence_length(layer_index)`

### RED/GREEN 2: Independent Layers

新增测试确认不同层的 key/value 不会互相影响。

### RED/GREEN 3: Clear Error for Invalid Layer

新增测试确认越界 layer index 会抛出清晰错误：

```text
layer_index must be in [0, num_layers)
```

## Validation

Milestone 2 测试命令：

```bash
PYTHONPATH=src python3 -m unittest tests/mini_engine/test_kv_cache.py -v
```

期望结果：

```text
test_append_stores_key_and_value_for_one_layer ... ok
test_layers_are_stored_independently ... ok
test_rejects_out_of_range_layer_index ... ok

Ran 3 tests
OK
```

## Interview Notes

可以这样介绍这个阶段：

> I implemented a minimal KV cache abstraction to model how decoder-only LLM inference stores per-layer key/value states. The cache supports appending key/value vectors, retrieving layer-local cache state, checking sequence length, and validating layer indices. This prepares the engine for explaining prefill/decode separation and later continuous batching.

中文面试表达：

> 我实现了一个教学版 KV Cache，用每层独立的 key/value 列表模拟 decoder-only LLM 推理中的历史状态缓存。通过这个结构可以解释为什么 decode 阶段不需要重复计算历史 token，以及后续 continuous batching 和 paged KV cache 为什么都围绕 KV 管理展开。

## Next Milestone

Milestone 3 将实现 `RequestScheduler` 的最小形状，用来管理等待队列、active batch，以及后续 continuous batching 的基础逻辑。
