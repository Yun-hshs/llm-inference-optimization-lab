# Milestone 18: Active Batch KV Cache Accounting

## Goal

让 `RequestScheduler` 能统计当前 active batch 中所有请求的 KV cache 使用量。

Milestone 17 已经让单个 `KVCache` 能计算条目数和估算显存。本阶段把这个能力提升到 scheduler 层面，让 serving loop 可以观察当前批次的 cache 压力。

## Why It Matters

真实 LLM serving 系统需要持续关注 active batch 的 KV cache 占用。

原因是：

- active batch 越大，KV cache 压力越高
- prompt 越长，prefill 后产生的 KV cache 越多
- decode 越久，每个请求持有的 cache 越长
- scheduler 需要根据显存压力决定是否接收新请求、暂停请求或触发 eviction

这一步是从“能生成 token”走向“能管理推理资源”的关键过渡。

## Key Design Boundary

`KVCache` 负责单个 cache 的 accounting：

```text
cache.total_entries()
cache.estimate_memory_bytes(...)
```

`RequestScheduler` 负责 active batch 级别的 accounting：

```text
active requests -> each request KV cache -> batch total
```

`TokenProvider` 仍然不参与资源统计。

## Target API

本阶段希望 `RequestScheduler` 增加：

```python
scheduler.active_kv_cache_entries()
```

返回当前 active requests 中所有 KV cache 条目的总数。

还希望增加：

```python
scheduler.active_kv_cache_memory_bytes(hidden_size=4096, bytes_per_element=2)
```

返回当前 active batch 的 KV cache 显存估算值。

## Target Data Flow

```text
RequestScheduler.active_requests
      |
      v
filter active.kv_cache is not None
      |
      v
sum cache.total_entries()
      |
      v
active batch KV cache entries

RequestScheduler.active_requests
      |
      v
sum cache.estimate_memory_bytes(...)
      |
      v
active batch KV cache memory bytes
```

## RED Test

新增测试文件：

```bash
tests/mini_engine/test_scheduler_kv_cache_accounting.py
```

测试内容：

- active batch 中多个请求的 KV cache entries 会被累加
- active batch 的 KV cache memory bytes 会被累加
- finished request 被移出 active batch 后，不再计入 active cache accounting

## Code Framework

我已经在：

```bash
src/llm_opt_lab/mini_engine/scheduler.py
```

搭好了方法框架：

```python
def active_kv_cache_entries(self) -> int:
    ...

def active_kv_cache_memory_bytes(self, *, hidden_size: int, bytes_per_element: int) -> int:
    ...
```

你只需要补充核心聚合逻辑。

## Your Implementation Task

在 `active_kv_cache_entries()` 中：

- 遍历 `self.active_requests`
- 跳过 `kv_cache is None` 的请求
- 累加 `active.kv_cache.total_entries()`

在 `active_kv_cache_memory_bytes()` 中：

- 遍历 `self.active_requests`
- 跳过 `kv_cache is None` 的请求
- 累加 `active.kv_cache.estimate_memory_bytes(...)`

## Validation

只跑 Milestone 18：

```bash
PYTHONPATH=src python3 -m unittest tests/mini_engine/test_scheduler_kv_cache_accounting.py -v
```

实现后建议一起跑 Milestone 17 和 18：

```bash
PYTHONPATH=src python3 -m unittest \
  tests/mini_engine/test_kv_cache_accounting.py \
  tests/mini_engine/test_scheduler_kv_cache_accounting.py \
  -v
```

最后再跑 mini engine 全量测试：

```bash
PYTHONPATH=src python3 -m unittest discover tests/mini_engine -v
```
