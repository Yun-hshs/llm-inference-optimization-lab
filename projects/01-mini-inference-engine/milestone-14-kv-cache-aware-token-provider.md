# Milestone 14: KV-Cache-Aware Token Provider

## Goal

让 token provider 根据 `ActiveRequest.phase()` 选择不同的 model input：

- `prefill`: 使用完整 prompt / output tokens 调用模型
- `decode`: 只使用最新生成的 token 调用模型

这个阶段不要求你实现真实 Transformer KV cache，只先把推理框架里的调用边界搭出来。

## Why It Matters

在没有 KV cache 的 greedy decoding 中，每一步都把完整序列重新喂给模型：

```text
[prompt tokens + generated tokens] -> model.forward()
```

真实 LLM serving 会在 prefill 阶段计算完整 prompt 的 KV cache。进入 decode 后，历史 token 的 key/value 已经在 cache 中，所以每一步通常只需要处理最新 token，并读取已有 KV cache。

因此本阶段的核心价值是：

- 把 prefill 和 decode 的计算路径区分开
- 为后续真实 KV cache 接入做接口准备
- 让 serving loop 更接近 vLLM / SGLang 的推理数据流

## Key Design Boundary

`KVCacheAwareTokenProvider` 负责：

```text
active request -> choose model input by phase -> model.forward() -> choose next token
```

`RequestScheduler` 仍然负责：

```text
token map -> append token -> mark finished -> remove finished -> refill batch
```

所以本阶段仍然保持一个重要规则：token provider 不修改 `ActiveRequest`。

## Target Data Flow

```text
prefill request
      |
      v
active.output_tokens()
      |
      v
model.forward(full tokens)

decode request
      |
      v
active.generated_tokens[-1]
      |
      v
model.forward([latest token])
```

## RED Test

新增测试文件：

```bash
tests/mini_engine/test_kv_cache_aware_token_provider.py
```

测试内容：

- `prefill` 阶段应该把完整 prompt 传给模型
- `decode` 阶段应该只把最新 generated token 传给模型
- provider 返回 request id 到 next token 的字典
- provider 不直接修改 `ActiveRequest`

## Your Implementation Task

你需要在：

```bash
src/llm_opt_lab/mini_engine/token_provider.py
```

新增 `KVCacheAwareTokenProvider`。

建议复用 `ModelBackedTokenProvider` 的 greedy token 选择逻辑，但 model input 的选择要根据 `active.phase()` 区分。

## Validation

只跑 Milestone 14：

```bash
PYTHONPATH=src python3 -m unittest tests/mini_engine/test_kv_cache_aware_token_provider.py -v
```

实现后建议一起跑 Milestone 13 和 14：

```bash
PYTHONPATH=src python3 -m unittest \
  tests/mini_engine/test_request_phase.py \
  tests/mini_engine/test_kv_cache_aware_token_provider.py \
  -v
```

最后再跑 mini engine 全量测试：

```bash
PYTHONPATH=src python3 -m unittest discover tests/mini_engine -v
```
