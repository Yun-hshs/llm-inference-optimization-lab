# Milestone 13: Prefill / Decode Phase Tracking

## Goal

给 `ActiveRequest` 增加阶段判断能力，让 mini inference engine 能区分请求当前处于 `prefill` 还是 `decode`。

这是连接真实 LLM serving 系统的重要一步：推理服务不会把 prompt 处理和逐 token 生成看成完全一样的阶段。

## Why It Matters

在 Transformer decoder 推理中：

- `prefill`: 第一次处理完整 prompt，计算量通常和 prompt length 强相关。
- `decode`: 每次只追加并生成一个新 token，核心瓶颈逐渐转向 KV cache 访问、batch 调度和显存带宽。

vLLM、SGLang 等 serving 框架都会围绕这两个阶段做调度、batching 和 cache 管理。

## Target API

本阶段希望 `ActiveRequest` 提供：

```python
active.phase()
```

返回值：

```text
"prefill"  -> 还没有生成任何 token
"decode"   -> 已经生成过至少一个 token
```

## Target Data Flow

```text
GenerationRequest
      |
      v
ActiveRequest.from_request()
      |
      v
generated_tokens == []      -> prefill
generated_tokens is nonempty -> decode
```

## RED Test

新增测试文件：

```bash
tests/mini_engine/test_request_phase.py
```

测试内容：

- 新激活的请求应该处于 `prefill`
- 调用 `append_token()` 后，请求应该进入 `decode`
- `output_tokens()` 仍然返回 prompt + generated tokens

## Your Implementation Task

你只需要在 `ActiveRequest` 中补充阶段判断逻辑。

建议先不要引入复杂 enum，当前项目用字符串就够清晰，也方便单测阅读。

## Validation

只跑 Milestone 13：

```bash
PYTHONPATH=src python3 -m unittest tests/mini_engine/test_request_phase.py -v
```

实现后再跑 mini engine 全量测试：

```bash
PYTHONPATH=src python3 -m unittest discover tests/mini_engine -v
```
