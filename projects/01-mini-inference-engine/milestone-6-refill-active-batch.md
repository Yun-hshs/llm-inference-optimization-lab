# Milestone 6: Refill Active Batch

## Goal

实现 active batch refill：清理已经完成的 active requests，并从 waiting queue 补入新请求，让 active batch 尽可能保持满。

这是 continuous batching 的核心动作之一。

## What Was Built

`RequestScheduler` 新增：

- `remove_finished_requests()`: 移除并返回 finished active requests
- `refill_active_batch()`: 清理 finished requests 后，从等待队列补入新 active requests

## Scheduling Flow

```text
before refill

active requests:
  req-1 finished
  req-2 running

waiting queue:
  req-3

        |
        | refill_active_batch()
        v

active requests:
  req-2 running
  req-3 running

waiting queue:
  empty
```

## Why This Matters

普通 batching 通常一批请求一起开始、一起等待结束。LLM serving 中请求长度差异很大，如果短请求完成后空位一直闲置，吞吐会变差。

continuous batching 的思路是：某个请求完成后，马上从等待队列补入新请求，让 GPU 上的 active batch 保持更高利用率。

## TDD Process

### RED: Missing Refill API

测试调用：

```python
added = scheduler.refill_active_batch()
```

初始失败：

```text
AttributeError: 'RequestScheduler' object has no attribute 'refill_active_batch'
```

### GREEN: Remove Finished And Fill Slot

测试场景：

- `max_batch_size=2`
- 初始 active: `req-1`, `req-2`
- `req-1` 生成完成
- waiting queue 中还有 `req-3`

执行 refill 后：

- active requests 变成 `req-2`, `req-3`
- waiting queue 变空
- 返回新补入的 `req-3`

## Validation

前六阶段测试命令：

```bash
PYTHONPATH=src python3 -m unittest \
  tests/mini_engine/test_greedy_engine.py \
  tests/mini_engine/test_kv_cache.py \
  tests/mini_engine/test_scheduler.py \
  tests/mini_engine/test_active_request.py \
  tests/mini_engine/test_active_batch_scheduler.py \
  tests/mini_engine/test_scheduler_refill.py \
  -v
```

验证结果：

```text
Ran 12 tests
OK
```

## Interview Notes

可以这样介绍这个阶段：

> I implemented active batch refill logic for the scheduler. Finished requests are removed from the active batch, and waiting requests are immediately promoted into newly opened slots. This mirrors the key scheduling idea behind continuous batching in LLM serving systems.

中文面试表达：

> 我实现了 active batch 的 refill 逻辑：完成的请求会从 active batch 中清理掉，等待队列中的新请求会马上补入空位。这就是 continuous batching 提升吞吐的关键思想之一，因为它减少了 batch 内空位闲置。

## Next Milestone

Milestone 7 将模拟一次 decode step：把模型生成的新 token 应用到每个 active request 上，并更新请求完成状态。
