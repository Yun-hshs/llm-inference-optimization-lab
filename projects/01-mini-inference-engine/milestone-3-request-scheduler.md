# Milestone 3: Request Scheduler

## Goal

实现一个最小请求调度器，用来模拟 LLM serving 中的等待队列和 batch 选择。

这一阶段先不做真正 continuous batching，只建立后续调度系统的基础：

- 新请求进入等待队列
- scheduler 按 FIFO 顺序选择请求
- 每次最多取出 `max_batch_size` 个请求
- 无效 batch size 会被拒绝

## What Was Built

新增了：

- `GenerationRequest`: 表示一次生成请求
- `RequestScheduler`: 管理等待队列并生成下一个 batch

`GenerationRequest` 包含：

- `request_id`: 请求唯一 ID
- `prompt`: 输入 token IDs
- `max_new_tokens`: 最大生成 token 数
- `eos_token_id`: 可选 EOS token

`RequestScheduler` 支持：

- 初始化最大 batch size
- 添加请求
- 查询等待队列长度
- 按 FIFO 顺序取出下一批请求
- 校验 `max_batch_size > 0`

## TDD Process

### RED 1: Missing Request Type

测试导入：

```python
from llm_opt_lab.mini_engine.scheduler import GenerationRequest, RequestScheduler
```

初始失败：

```text
ImportError: cannot import name 'GenerationRequest'
```

### GREEN 1: FIFO Batch Selection

实现 `GenerationRequest` 和 `RequestScheduler.next_batch()`。

测试场景：

```python
scheduler = RequestScheduler(max_batch_size=2)
scheduler.add_request(req_1)
scheduler.add_request(req_2)
scheduler.add_request(req_3)
batch = scheduler.next_batch()
```

期望：

```python
["req-1", "req-2"]
```

等待队列剩余 1 个请求。

### RED/GREEN 2: Reject Invalid Batch Size

新增测试确认：

```python
RequestScheduler(max_batch_size=0)
```

会抛出：

```text
ValueError: max_batch_size must be positive
```

## Validation

前三阶段测试命令：

```bash
PYTHONPATH=src python3 -m unittest \
  tests/mini_engine/test_greedy_engine.py \
  tests/mini_engine/test_kv_cache.py \
  tests/mini_engine/test_scheduler.py \
  -v
```

验证结果：

```text
Ran 8 tests
OK
```

## Interview Notes

可以这样介绍这个阶段：

> I implemented a minimal request scheduler for LLM serving experiments. It keeps pending generation requests in FIFO order and forms batches up to a configured maximum batch size. This prepares the project for continuous batching, where newly arrived requests can be merged with active decoding work.

中文面试表达：

> 我实现了一个最小请求调度器，用 FIFO 队列管理待处理请求，并按最大 batch size 取出下一批请求。这一步是 continuous batching 的基础，因为后续需要同时管理等待队列和正在 decode 的 active requests。

## Next Milestone

Milestone 4 将实现 `ActiveRequest`，用于记录单个请求的生成状态，包括已生成 token、生成步数、是否达到 EOS 或 `max_new_tokens`。
