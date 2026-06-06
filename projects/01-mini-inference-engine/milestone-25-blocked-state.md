# Milestone 25: KV Budget Blocked State

## Goal

给 `RequestScheduler` 增加一个明确的 blocked state 判断，用来区分：

- `idle`: 没有 waiting request，也没有 active request
- `running`: 有 active request 可以继续 decode
- `blocked`: waiting queue 有请求，但当前 KV cache budget 无法激活队首请求

Milestone 24 已经让 serving loop 可以在 blocked 或 idle 时停止。本阶段让 scheduler 自己能报告 blocked 状态，避免把它和 `has_work()` 混在一起。

## Why It Matters

在 budget-aware serving 中，`has_work()` 的含义不能改。

即使一个请求因为 KV cache budget 暂时无法激活，只要它还在 waiting queue 中，scheduler 仍然有 work：

```text
request_queue = [blocked_request]
active_requests = []
has_work() = True
is_idle() = False
```

但 serving loop 不能继续 decode，因为没有 active request。这个状态应该被单独命名为 blocked。

这也是真实 serving 系统里 backpressure、admission control 和 overload handling 的基础。

## Key Design Boundary

保持旧生命周期方法语义不变：

```python
has_work()
is_idle()
```

新增：

```python
is_blocked_by_kv_budget(...)
```

它只描述 KV cache budget 造成的 blocked 状态，不替代 `has_work()`。

## Target API

新增：

```python
scheduler.is_blocked_by_kv_budget(
    hidden_size=4096,
    bytes_per_element=2,
)
```

返回值语义：

```text
True  -> no active requests + waiting queue nonempty + front request cannot be admitted
False -> idle / running / front request admissible / unbounded scheduler
```

## Target Data Flow

```text
active_requests
      |
      v
if nonempty -> not blocked

request_queue
      |
      v
if empty -> not blocked

request_queue[0]
      |
      v
can_admit_request(...)
      |
      v
not admissible -> blocked
```

## RED Test

新增测试文件：

```bash
tests/mini_engine/test_scheduler_blocked_state.py
```

测试内容：

- waiting 队首请求超预算且没有 active request 时，报告 blocked
- idle 状态不报告 blocked
- active request 还可以 decode 时，不报告 blocked
- waiting 队首请求可接纳时，不报告 blocked
- 未配置预算的 scheduler 不报告 budget blocked

## Code Framework

我已经在：

```bash
src/llm_opt_lab/mini_engine/scheduler.py
```

搭好了方法框架：

```python
def is_blocked_by_kv_budget(...):
    ...
```

你只需要补充核心判断逻辑。

## Your Implementation Task

实现思路：

- 如果 `self.active_requests` 非空，返回 `False`
- 如果 `self.request_queue` 为空，返回 `False`
- 取 `front_request = self.request_queue[0]`
- 返回 `not self.can_admit_request(front_request, ...)`

注意：不要改 `has_work()` 或 `is_idle()`。

## Validation

只跑 Milestone 25：

```bash
PYTHONPATH=src python3 -m unittest tests/mini_engine/test_scheduler_blocked_state.py -v
```

实现后建议一起跑 Milestone 24 和 25：

```bash
PYTHONPATH=src python3 -m unittest \
  tests/mini_engine/test_budget_aware_serving_loop.py \
  tests/mini_engine/test_scheduler_blocked_state.py \
  -v
```

最后再跑 mini engine 全量测试：

```bash
PYTHONPATH=src python3 -m unittest discover tests/mini_engine -v
```
