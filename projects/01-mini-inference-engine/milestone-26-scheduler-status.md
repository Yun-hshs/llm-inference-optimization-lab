# Milestone 26: Scheduler Status Snapshot

## Goal

给 `RequestScheduler` 增加一个只读状态快照方法，把 waiting queue、active batch、KV cache budget 和 blocked state 统一汇总成结构化数据。

本阶段不改变调度行为，只解决“系统现在处于什么状态”这个可观测性问题。

## Why It Matters

前面 Milestone 已经实现了：

- continuous batching
- budget-aware admission
- budget-aware refill
- blocked state 判断

但这些信息现在分散在多个方法里。真实推理服务需要把这些内部状态输出给日志、metrics、dashboard 或 benchmark report，否则很难解释为什么某个请求没有被激活。

状态快照是 serving system 可观测性的第一步。

## Key Design Boundary

`status(...)` 必须是 read-only：

- 不激活 waiting request
- 不移除 finished request
- 不修改 active batch
- 不追加 token
- 不清理 KV cache

它只读取当前 scheduler 状态，并返回一个 dict。

## Target API

新增：

```python
scheduler.status(
    hidden_size=4096,
    bytes_per_element=2,
)
```

返回：

```python
{
    "waiting_count": 1,
    "active_count": 2,
    "has_work": True,
    "is_idle": False,
    "is_blocked_by_kv_budget": False,
    "active_kv_cache_entries": 128,
    "active_kv_cache_memory_bytes": 2097152,
    "remaining_kv_cache_budget_bytes": 1048576,
    "max_kv_cache_memory_bytes": 3145728,
}
```

如果没有配置 KV cache budget：

```python
{
    "remaining_kv_cache_budget_bytes": None,
    "max_kv_cache_memory_bytes": None,
}
```

## Target Data Flow

```text
request_queue
      |
      v
waiting_count

active_requests
      |
      v
active_count / active_kv_cache_entries / active_kv_cache_memory_bytes

max_kv_cache_memory_bytes
      |
      v
remaining_kv_cache_budget_bytes

active_requests + request_queue[0] + can_admit_request(...)
      |
      v
is_blocked_by_kv_budget
```

## RED Test

新增测试文件：

```bash
tests/mini_engine/test_scheduler_status.py
```

测试内容：

- status 可以报告 waiting/active 数量、生命周期状态和 KV cache budget 使用情况
- waiting 队首请求超预算且没有 active request 时，status 报告 blocked
- 未配置 budget 时，budget 字段返回 `None`

## Code Framework

我已经在：

```bash
src/llm_opt_lab/mini_engine/scheduler.py
```

搭好了方法框架：

```python
def status(...):
    ...
```

你只需要补充核心 dict 组装逻辑。

## Your Implementation Task

实现思路：

- 调用 `waiting_count()`
- 调用 `active_count()`
- 调用 `has_work()`
- 调用 `is_idle()`
- 调用 `is_blocked_by_kv_budget(...)`
- 调用 `active_kv_cache_entries()`
- 调用 `active_kv_cache_memory_bytes(...)`
- 调用 `remaining_kv_cache_budget_bytes(...)`
- 直接读取 `self.max_kv_cache_memory_bytes`

注意：不要在 `status(...)` 里调用任何会修改队列的方法，比如：

```python
activate_next_batch()
activate_next_admissible_batch()
remove_finished_requests()
refill_active_batch()
step()
```

## Validation

只跑 Milestone 26：

```bash
PYTHONPATH=src python3 -m unittest tests/mini_engine/test_scheduler_status.py -v
```

实现后建议一起跑 Milestone 24-26：

```bash
PYTHONPATH=src python3 -m unittest \
  tests/mini_engine/test_budget_aware_serving_loop.py \
  tests/mini_engine/test_scheduler_blocked_state.py \
  tests/mini_engine/test_scheduler_status.py \
  -v
```

最后再跑 mini engine 全量测试：

```bash
PYTHONPATH=src python3 -m unittest discover tests/mini_engine -v
```
