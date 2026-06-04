# Milestone 9: Scheduler Lifecycle

## Goal

让调度器能够告诉外层 serving loop：当前系统是否还有未完成的工作。

前面已经实现了 waiting queue、active requests、step 调度和 refill 补位。如果要写完整循环，还需要判断什么时候继续、什么时候停止。

## What Was Built

`RequestScheduler` 新增：

```python
has_work()
is_idle()
```

语义：

- waiting queue 中有请求时，`has_work()` 为 `True`
- active requests 中有请求时，`has_work()` 为 `True`
- waiting queue 和 active requests 都为空时，`has_work()` 为 `False`
- `is_idle()` 与 `has_work()` 相反

## Data Flow

```text
waiting queue count
active request count
        |
        v
has_work()
        |
        v
serving loop decides whether to continue
```

## Why This Matters

真实 serving loop 需要不断执行：

```text
while scheduler has work:
    run model decode
    scheduler step
```

如果没有生命周期判断，外层循环无法知道什么时候应该停止。

## TDD Process

### RED: Missing Lifecycle API

测试调用：

```python
scheduler.has_work()
```

初始失败：

```text
AttributeError: 'RequestScheduler' object has no attribute 'has_work'
```

### GREEN: Track Waiting And Active Work

实现后测试确认：

- 空 scheduler 是 idle
- waiting queue 有请求时不是 idle
- active requests 有请求时不是 idle
- step 完成最后一个请求后重新变成 idle

## Validation

前九阶段测试命令：

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
  tests/mini_engine/test_scheduler_lifecycle.py \
  -v
```

验证结果：

```text
Ran 16 tests
OK
```

## Interview Notes

可以这样介绍这个阶段：

> I added lifecycle state checks to the scheduler so an outer serving loop can decide whether there is still pending or active work. This is a small but important interface for building an end-to-end inference loop.

中文面试表达：

> 我给调度器增加了生命周期判断，用来告诉外层 serving loop 当前是否还有等待请求或正在 decode 的请求。这样才能写出完整的推理循环，而不是只调用单步调度方法。

## Next Milestone

Milestone 10 将实现最小 serving loop，用一个可控 token provider 模拟模型输出，驱动 scheduler 一直运行到 idle。
