# Milestone 19: KV Cache Memory Budget Guard

## Goal

让 `RequestScheduler` 支持 KV cache 显存预算判断。

Milestone 18 已经可以统计 active batch 的 KV cache memory。本阶段在 scheduler 上增加预算配置，让它能判断当前 active batch 是否已经超过 KV cache memory budget。

## Why It Matters

真实 LLM serving 系统不能只看 batch size，还要看 KV cache 占用。

两个请求的 batch size 可能一样，但显存压力完全不同：

- 短 prompt 请求占用较少 KV cache
- 长 prompt 请求会在 prefill 后占用大量 KV cache
- 长时间 decode 的请求会持续增长 KV cache

因此调度器需要具备资源预算意识。这个阶段先做 budget guard，不直接改变调度行为。下一阶段可以在这个基础上做 admission control。

## Key Design Boundary

`KVCache` 负责单个 cache 的 memory estimate：

```text
cache.estimate_memory_bytes(...)
```

`RequestScheduler` 负责 active batch 的 memory estimate 和 budget 判断：

```text
active_kv_cache_memory_bytes(...) -> compare with max_kv_cache_memory_bytes
```

`ServingLoop` 和 `TokenProvider` 不参与预算判断。

## Target API

本阶段扩展 scheduler 初始化参数：

```python
scheduler = RequestScheduler(
    max_batch_size=2,
    num_kv_layers=2,
    max_kv_cache_memory_bytes=1024,
)
```

新增两个方法：

```python
scheduler.is_kv_cache_over_budget(hidden_size=4096, bytes_per_element=2)
```

返回当前 active KV cache memory 是否超过预算。

```python
scheduler.remaining_kv_cache_budget_bytes(hidden_size=4096, bytes_per_element=2)
```

返回剩余预算。如果没有配置预算，返回 `None`。如果已经超预算，返回 `0`。

## Target Data Flow

```text
RequestScheduler.active_requests
      |
      v
active_kv_cache_memory_bytes(...)
      |
      v
compare with max_kv_cache_memory_bytes
      |
      v
over budget / remaining budget
```

## RED Test

新增测试文件：

```bash
tests/mini_engine/test_scheduler_kv_cache_budget.py
```

测试内容：

- usage 等于 budget 时不算 over budget
- usage 超过 budget 时算 over budget
- 未配置 budget 的 scheduler 永远不报告 over budget，并返回 `None` 作为 remaining budget

## Code Framework

我已经在：

```bash
src/llm_opt_lab/mini_engine/scheduler.py
```

搭好了字段和方法框架：

```python
self.max_kv_cache_memory_bytes = max_kv_cache_memory_bytes

def is_kv_cache_over_budget(...):
    ...

def remaining_kv_cache_budget_bytes(...):
    ...
```

你只需要补充核心预算判断逻辑。

## Your Implementation Task

在 `is_kv_cache_over_budget()` 中：

- 如果 `self.max_kv_cache_memory_bytes is None`，返回 `False`
- 否则计算当前 active KV cache memory
- 当 usage 大于 budget 时返回 `True`

在 `remaining_kv_cache_budget_bytes()` 中：

- 如果没有配置 budget，返回 `None`
- 否则返回 `budget - usage`
- 如果结果小于 0，返回 0

## Validation

只跑 Milestone 19：

```bash
PYTHONPATH=src python3 -m unittest tests/mini_engine/test_scheduler_kv_cache_budget.py -v
```

实现后建议一起跑 Milestone 18 和 19：

```bash
PYTHONPATH=src python3 -m unittest \
  tests/mini_engine/test_scheduler_kv_cache_accounting.py \
  tests/mini_engine/test_scheduler_kv_cache_budget.py \
  -v
```

最后再跑 mini engine 全量测试：

```bash
PYTHONPATH=src python3 -m unittest discover tests/mini_engine -v
```
