# Milestone 11: Model-Backed Token Provider

## Goal

把 serving loop 中的 token 来源从手写脚本推进到模型输出。

前一个阶段的 `ScriptedTokenProvider` 只用于验证调度流程。本阶段要新增 `ModelBackedTokenProvider`，它接收 `DecoderModel`，对每个 active request 调用模型，并从最后一个位置的 logits 中选择分数最高的 token。

## Target Data Flow

```text
active_requests
      |
      v
active.output_tokens()
      |
      v
model.forward(tokens)
      |
      v
logits[-1]
      |
      v
argmax
      |
      v
{request_id: next_token_id}
```

## RED Test

本阶段先创建失败测试，核心实现留在：

```text
src/llm_opt_lab/mini_engine/token_provider.py
```

测试关注两件事：

- provider 会为每个 active request 返回一个 token
- provider 使用的是 `active.output_tokens()`，也就是 prompt 加上已经生成的 tokens

## Your Implementation Task

你需要在 `ModelBackedTokenProvider.__call__()` 中手写核心逻辑：

```text
for each active request:
    tokens = active.output_tokens()
    logits = model.forward(tokens)
    last_token_logits = logits[-1]
    next_token_id = argmax(last_token_logits)
    save request_id -> next_token_id
return token_by_request_id
```

注意：当前项目的 logits 类型是 Python `list[list[float]]`，不依赖 numpy。你可以用列表遍历或 `max(..., key=...)` 找最大值下标。

## Validation

只跑 Milestone 11 的 RED 测试：

```bash
PYTHONPATH=src python3 -m unittest tests/mini_engine/test_model_token_provider.py -v
```

预期当前会失败，因为 `ModelBackedTokenProvider.__call__()` 还没有实现核心逻辑。
