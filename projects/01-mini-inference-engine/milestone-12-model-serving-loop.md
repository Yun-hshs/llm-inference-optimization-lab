# Milestone 12: Model-Driven Serving Loop

## Goal

把 `ServingLoop`、`RequestScheduler` 和 `ModelBackedTokenProvider` 串成一个真正由模型 logits 驱动的端到端 decode loop。

Milestone 11 已经让 token provider 可以从 `DecoderModel.forward()` 中选择下一个 token。本阶段要验证它接入 serving loop 后，状态更新只发生在 scheduler 中。

## Key Design Boundary

`ModelBackedTokenProvider` 的职责：

```text
active request -> model.forward(output_tokens) -> choose next token -> return token map
```

`RequestScheduler` 的职责：

```text
token map -> append token -> mark finished -> remove finished -> refill batch
```

因此 token provider 不应该修改 `ActiveRequest`，否则 serving loop 调用 `scheduler.step()` 时会再次追加同一个 token。

## Target Data Flow

```text
ServingLoop.run_until_idle()
      |
      v
active_requests
      |
      v
ModelBackedTokenProvider
      |
      v
token_by_request_id
      |
      v
RequestScheduler.step()
      |
      v
updated active/finished requests
```

## RED Test

本阶段新增端到端测试：

- 一个请求需要生成 2 个 token
- 模型第一次返回 token 7
- 模型第二次返回 token 8
- serving loop 最终输出 `[prompt, 7, 8]`
- 模型第二次 forward 的输入必须包含上一轮由 scheduler 追加的 token

## Your Implementation Task

如果测试失败，并且出现重复 token，重点检查：

```text
ModelBackedTokenProvider.__call__()
```

它只需要返回 token 字典，不要在 provider 里调用 `active.append_token()`。

## Validation

只跑 Milestone 12：

```bash
PYTHONPATH=src python3 -m unittest tests/mini_engine/test_model_serving_loop.py -v
```

修复后再跑 Milestone 11 和 12：

```bash
PYTHONPATH=src python3 -m unittest \
  tests/mini_engine/test_model_token_provider.py \
  tests/mini_engine/test_model_serving_loop.py \
  -v
```
