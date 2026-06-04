# Milestone 7: Decode Step Token Update

## Goal

模拟一次 decode step 后，把模型输出的新 token 写回每个 active request。

前面已经能管理 active batch，但 active request 本身还不会随着模型输出更新。这个阶段把“模型输出”和“请求状态”对齐。

## What Was Built

`RequestScheduler` 新增：

```python
apply_tokens_to_active_requests(token_by_request_id)
```

输入是一个 request ID 到 token ID 的映射：

```python
{
    "req-1": 7,
    "req-2": 8,
}
```

调度器会遍历当前 `active_requests`，找到每个 active request 对应的新 token，并调用：

```python
active.append_token(token_id)
```

## Data Flow

```text
model decode output
  {"req-1": 7, "req-2": 8}

        |
        v

RequestScheduler.apply_tokens_to_active_requests()

        |
        v

ActiveRequest("req-1").generated_tokens += [7]
ActiveRequest("req-2").generated_tokens += [8]
```

`ActiveRequest.append_token()` 会负责判断：

- 是否达到 `max_new_tokens`
- 是否生成了 `eos_token_id`
- 是否应该标记 `finished=True`

## Error Handling

如果某个 active request 没有对应 token，调度器会抛出清晰错误：

```text
missing token for active request req-1
```

这是必要的，因为一次 decode step 中，每个 active request 都应该得到一个 token。缺失 token 说明模型输出和 scheduler 状态不一致，不能静默跳过。

## TDD Process

### RED: Missing Decode Update API

测试调用：

```python
scheduler.apply_tokens_to_active_requests({"req-1": 7, "req-2": 8})
```

初始失败：

```text
AttributeError: 'RequestScheduler' object has no attribute 'apply_tokens_to_active_requests'
```

### GREEN: Apply Tokens To Active Requests

实现后测试确认：

- active request 的输出 token 被更新
- 达到 `max_new_tokens` 的请求会变成 finished
- 缺失 request token 时抛出 `KeyError`

## Interview Notes

可以这样介绍这个阶段：

> I added decode-step state updates to the scheduler. For each active request, the scheduler applies the token produced by the model and validates that every active request receives an output token. This keeps scheduler state aligned with model outputs.

中文面试表达：

> 我实现了 decode step 的 token 回写逻辑。调度器会把模型本轮生成的 token 按 request ID 写回对应的 active request，并检查每个 active request 都有输出，避免模型输出和调度状态不一致。
