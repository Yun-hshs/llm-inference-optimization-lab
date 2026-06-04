# Milestone 5: Active Batch Scheduling

## Goal

让 `RequestScheduler` 不只管理等待队列，还能把等待请求转换成 active batch。

这一步是从普通 FIFO 调度走向 continuous batching 的关键过渡。

## What Was Built

`RequestScheduler` 新增：

- `active_requests`: 当前正在 decode 的请求列表
- `active_count()`: 查询 active request 数量
- `activate_next_batch()`: 从等待队列取出最多 `max_batch_size` 个请求，并转换成 `ActiveRequest`

## Scheduling Flow

```text
waiting queue
  [GenerationRequest, GenerationRequest, GenerationRequest]

        |
        | activate_next_batch()
        v

active requests
  [ActiveRequest, ActiveRequest]

waiting queue
  [GenerationRequest]
```

## TDD Process

### RED: Missing Active Batch API

测试调用：

```python
active_batch = scheduler.activate_next_batch()
```

初始失败：

```text
AttributeError: 'RequestScheduler' object has no attribute 'activate_next_batch'
```

### GREEN: Convert Waiting Requests To Active Requests

实现：

- 初始化 `self.active_requests`
- 新增 `active_count()`
- 新增 `activate_next_batch()`

测试确认：

- 返回的对象是 `ActiveRequest`
- request ID 顺序保持 FIFO
- active count 增加
- waiting count 减少

## Validation

前五阶段测试命令：

```bash
PYTHONPATH=src python3 -m unittest \
  tests/mini_engine/test_greedy_engine.py \
  tests/mini_engine/test_kv_cache.py \
  tests/mini_engine/test_scheduler.py \
  tests/mini_engine/test_active_request.py \
  tests/mini_engine/test_active_batch_scheduler.py \
  -v
```

验证结果：

```text
Ran 11 tests
OK
```

## Interview Notes

可以这样介绍这个阶段：

> I extended the scheduler from a waiting-queue-only design to an active batch scheduler. Waiting generation requests are converted into active request states, which lets the system track decode progress separately from queued work. This mirrors the serving-side distinction between pending requests and active decoding requests.

中文面试表达：

> 我把调度器从简单等待队列扩展成 active batch 调度器。等待请求进入调度后会转换成 `ActiveRequest`，这样后续每一步 decode 都可以更新请求状态，并在完成后从 active batch 中移除。

## Next Milestone

Milestone 6 将实现 refill 逻辑：清理已经完成的 active requests，并从等待队列补入新请求，保持 active batch 尽可能满。
