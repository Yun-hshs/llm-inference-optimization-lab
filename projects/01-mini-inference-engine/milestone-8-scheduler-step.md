# Milestone 8: End-to-End Scheduler Step

## Goal

实现一次完整 continuous batching 调度 tick。

前面已经分别实现了：

- active requests
- token 回写
- finished request 清理
- waiting queue 补位

这一阶段要把它们串成一个统一入口：`RequestScheduler.step()`。

## Target Flow

```text
active_requests + token_by_request_id
        |
        v
apply generated tokens
        |
        v
finished requests become finished=True
        |
        v
remove finished requests
        |
        v
refill open slots from waiting queue
        |
        v
return finished requests
```

## Example

初始状态：

```text
max_batch_size = 2

waiting queue:
  req-1 max_new_tokens=1
  req-2 max_new_tokens=3
  req-3 max_new_tokens=2
```

先激活第一批：

```text
active:
  req-1
  req-2

waiting:
  req-3
```

执行：

```python
finished = scheduler.step({"req-1": 9, "req-2": 8})
```

期望：

```text
finished:
  req-1

active:
  req-2
  req-3

waiting:
  empty
```

## Implementation Notes

推荐给 `RequestScheduler` 新增：

```python
def step(self, token_by_request_id: dict[str, int]) -> list[ActiveRequest]:
    ...
```

行为要求：

- 调用 `apply_tokens_to_active_requests()`
- 移除完成请求
- 从等待队列补位
- 返回本轮完成的请求

注意：如果直接调用 `refill_active_batch()`，它内部已经会调用 `remove_finished_requests()`。为了避免重复清理，后续可以抽出一个 `_fill_open_slots()` 私有方法；当前阶段也可以在 `step()` 里手写补位逻辑。

## TDD Process

### RED: Missing Step API

测试调用：

```python
finished = scheduler.step({"req-1": 9, "req-2": 8})
```

初始失败：

```text
AttributeError: 'RequestScheduler' object has no attribute 'step'
```

### GREEN: Apply, Finish, Refill

实现后测试确认：

- token 会写回 active requests
- 本轮完成的请求会被返回
- 完成请求会从 active batch 移除
- waiting queue 中的新请求会补入空位
- active count 不超过 `max_batch_size`

## Validation

前八阶段测试命令：

```bash
PYTHONPATH=src python3 -m unittest \
  tests/mini_engine/test_greedy_engine.py \
  tests/mini_engine/test_kv_cache.py \
  tests/mini_engine/test_scheduler.py \
  tests/mini_engine/test_active_request.py \
  tests/mini_engine/test_active_batch_scheduler.py \
  tests/mini_engine/test_scheduler_refill.py \
  tests/mini_engine/test_decode_step_scheduler.py \
  tests/mini_engine/test_scheduler_step.py \
  -v
```

验证结果：

```text
Ran 15 tests
OK
```

## Interview Notes

可以这样介绍这个阶段：

> I implemented an end-to-end scheduling step that applies generated tokens to active requests, finalizes completed requests, and refills open batch slots from the waiting queue. This simulates one decode scheduling tick in a continuous batching LLM serving system.

中文面试表达：

> 我实现了 continuous batching 的单步调度循环：先把本轮模型输出 token 回写到 active requests，再清理完成请求，最后从等待队列补齐 batch 空位。这个过程模拟了 LLM serving 中一次 decode scheduling tick。
