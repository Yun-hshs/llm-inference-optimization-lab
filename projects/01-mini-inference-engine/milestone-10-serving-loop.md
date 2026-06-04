# Milestone 10: Serving Loop

## Goal

把前面实现的 scheduler 串成一个最小推理服务循环。

前九个阶段已经有了 waiting queue、active requests、decode step、refill 和 lifecycle 判断。Milestone 10 的目标是让外层循环可以持续执行 decode step，直到所有请求都完成。

## What Was Built

新增：

```python
TokenProvider
ServingLoop
run_until_idle()
```

`TokenProvider` 是一个可调用对象，输入当前 active requests，输出每个 request 本轮生成的 token：

```text
active_requests -> token_provider -> token_by_request_id
```

`ServingLoop.run_until_idle()` 负责：

- 初始激活一批 waiting requests
- 调用 token provider 生成本轮 tokens
- 调用 scheduler.step() 更新请求状态
- 收集 finished requests
- 重复执行直到 scheduler idle

## Data Flow

```text
waiting queue
      |
      v
activate_next_batch()
      |
      v
active_requests
      |
      v
token_provider(active_requests)
      |
      v
token_by_request_id
      |
      v
scheduler.step(token_by_request_id)
      |
      v
finished requests + refilled active requests
      |
      v
repeat until scheduler.is_idle()
```

## Why This Matters

真实 LLM serving 系统一般不会只执行一次 decode step，而是持续处理请求：

```text
prefill/decode -> update state -> remove finished -> refill batch -> decode again
```

这个阶段把 scheduler 从“单步组件”推进成“可以被服务循环驱动的组件”，是理解 vLLM/SGLang continuous batching 的关键过渡。

## TDD Process

### RED: Missing Serving Loop

测试期望创建一个 `ServingLoop`，并用可控的 `ScriptedTokenProvider` 模拟模型输出。

初始失败点是缺少 serving loop 模块或对应 API。

### GREEN: Run Until Idle

实现后测试确认：

- 多个请求可以被循环驱动直到完成
- finished requests 会被收集并返回
- scheduler 最终进入 idle 状态
- refill 后的新请求也能继续参与后续 decode step

## Validation

包含 Milestone 10 的 mini engine 测试命令：

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
  tests/mini_engine/test_serving_loop.py \
  -v
```

验证结果：

```text
Ran 17 tests
OK
```

## Interview Notes

可以这样介绍这个阶段：

> I implemented a minimal serving loop that repeatedly asks a token provider for one decode token per active request, applies the scheduler step, collects finished requests, and refills the active batch until the system becomes idle.

中文面试表达：

> 我实现了一个最小 serving loop，把 active requests、token provider 和 scheduler.step 串起来。每一轮 decode 后会更新请求状态、移除完成请求、从等待队列补入新请求，直到整个调度器 idle。

## Next Milestone

Milestone 11 将把测试中的 scripted token provider 替换成 model-backed token provider，让 serving loop 真正从 `DecoderModel.forward()` 的 logits 中选择下一个 token。
