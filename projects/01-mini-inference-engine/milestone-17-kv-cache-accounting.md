# Milestone 17: KV Cache Usage Accounting

## Goal

让 `KVCache` 能统计当前缓存中有多少 KV 条目，并估算这些条目占用的显存字节数。

Milestone 15 和 16 已经建立了 KV cache 的分配和释放生命周期。本阶段开始让 cache 具备可观测性，为后续 benchmark、显存分析和 paged KV allocator 做准备。

## Why It Matters

大模型推理的显存占用不只来自模型权重，还来自 KV cache。

KV cache 的大小通常和这些因素相关：

- layer 数
- 当前缓存 token 数
- hidden size 或 head dimension
- key/value 两份向量
- 数据类型字节数，例如 FP16 是 2 bytes，FP32 是 4 bytes

面试中如果能解释 KV cache memory 如何估算，就能自然连接到 vLLM 的 PagedAttention、continuous batching 下的显存压力，以及长上下文推理为什么贵。

## Key Design Boundary

`KVCache` 负责保存和统计缓存内容：

```text
append key/value -> sequence length -> total entries -> memory estimate
```

`RequestScheduler` 负责请求生命周期，不负责计算 cache 显存。

`TokenProvider` 负责 next token 选择，不负责 cache accounting。

## Target API

本阶段希望 `KVCache` 增加：

```python
cache.total_entries()
```

返回所有 layer 中缓存 token 条目的总数。

还希望增加：

```python
cache.estimate_memory_bytes(hidden_size=4096, bytes_per_element=2)
```

估算公式：

```text
total_entries * 2 * hidden_size * bytes_per_element
```

其中 `2` 表示每个 token entry 同时存 key 和 value。

## Target Data Flow

```text
KVCache.layers
      |
      v
sum(len(layer.keys) for each layer)
      |
      v
total_entries
      |
      v
total_entries * key/value factor * hidden_size * bytes_per_element
      |
      v
estimated memory bytes
```

## RED Test

新增测试文件：

```bash
tests/mini_engine/test_kv_cache_accounting.py
```

测试内容：

- `total_entries()` 统计所有 layer 的缓存条目数
- `estimate_memory_bytes()` 按 key/value 双份存储估算字节数
- `clear()` 后 accounting 结果归零

## Code Framework

我已经在：

```bash
src/llm_opt_lab/mini_engine/kv_cache.py
```

搭好了方法框架：

```python
def total_entries(self) -> int:
    ...

def estimate_memory_bytes(self, *, hidden_size: int, bytes_per_element: int) -> int:
    ...
```

你只需要补充核心统计逻辑。

## Your Implementation Task

在 `total_entries()` 中统计所有 layer 的 key 数量。

在 `estimate_memory_bytes()` 中复用 `total_entries()`，按公式计算：

```text
total_entries * 2 * hidden_size * bytes_per_element
```

本阶段先不做复杂校验，保持实现简单清晰。

## Validation

只跑 Milestone 17：

```bash
PYTHONPATH=src python3 -m unittest tests/mini_engine/test_kv_cache_accounting.py -v
```

实现后建议一起跑 KV cache 相关测试：

```bash
PYTHONPATH=src python3 -m unittest \
  tests/mini_engine/test_kv_cache.py \
  tests/mini_engine/test_kv_cache_accounting.py \
  tests/mini_engine/test_scheduler_kv_cache_lifecycle.py \
  tests/mini_engine/test_scheduler_kv_cache_release.py \
  -v
```

最后再跑 mini engine 全量测试：

```bash
PYTHONPATH=src python3 -m unittest discover tests/mini_engine -v
```
