# Milestone 4: Active Request State

## Goal

实现 `ActiveRequest`，用来记录单个生成请求在 decode 过程中的状态。

前面的 `GenerationRequest` 只描述“用户提交了什么请求”。`ActiveRequest` 则描述“这个请求现在生成到哪一步了”。

## What Was Built

新增了 `ActiveRequest`，支持：

- 从 `GenerationRequest` 创建 active request
- 记录已经生成的 token
- 输出完整 token 序列：`prompt + generated_tokens`
- 生成数量达到 `max_new_tokens` 后标记完成
- 生成 token 等于 `eos_token_id` 后标记完成

## State Model

```text
GenerationRequest
  request_id
  prompt
  max_new_tokens
  eos_token_id

        |
        v

ActiveRequest
  request
  generated_tokens
  finished
```

这个模型是 continuous batching 的基础。真实服务中，一个请求进入 active batch 后，每个 decode step 都会追加一个 token；如果请求完成，就从 active batch 移除，再从等待队列补入新请求。

## TDD Process

### RED 1: Missing ActiveRequest

测试导入：

```python
from llm_opt_lab.mini_engine.scheduler import ActiveRequest, GenerationRequest
```

初始失败：

```text
ImportError: cannot import name 'ActiveRequest'
```

### GREEN 1: Track Generated Tokens

实现：

- `ActiveRequest.from_request(request)`
- `ActiveRequest.append_token(token_id)`
- `ActiveRequest.output_tokens()`

测试确认：

```python
active.append_token(7)
active.output_tokens() == [1, 2, 7]
```

### RED/GREEN 2: Finish on Max New Tokens

当生成数量达到 `max_new_tokens`，`finished` 变成 `True`。

### RED/GREEN 3: Finish on EOS

当生成 token 等于 `eos_token_id`，`finished` 变成 `True`。

## Validation

前四阶段测试命令：

```bash
PYTHONPATH=src python3 -m unittest \
  tests/mini_engine/test_greedy_engine.py \
  tests/mini_engine/test_kv_cache.py \
  tests/mini_engine/test_scheduler.py \
  tests/mini_engine/test_active_request.py \
  -v
```

验证结果：

```text
Ran 10 tests
OK
```

## Interview Notes

可以这样介绍这个阶段：

> I added an active request state object to track per-request decode progress. It stores generated tokens, exposes full output tokens, and marks requests finished when they reach either `max_new_tokens` or EOS. This separates static request metadata from dynamic decode state, which is a key concept in continuous batching.

中文面试表达：

> 我把用户请求和运行时状态拆开了：`GenerationRequest` 表示请求参数，`ActiveRequest` 表示 decode 过程中的动态状态，包括已经生成的 token 和是否完成。这是 continuous batching 的基础，因为调度器需要不断更新 active request，并在完成后补入新的请求。

## Next Milestone

Milestone 5 将让 `RequestScheduler` 管理 active batch：把等待队列中的请求转换成 `ActiveRequest`，并提供 active request 计数。
