# Milestone 1: Single-Request Greedy Decoding

## Goal

复现一个最小 LLM decode loop：给定 prompt token IDs，模型每一步返回 logits，推理引擎选择最后一个位置 logits 中分数最高的 token，并追加到输出序列。

这一阶段不接真实大模型权重，也不引入 GPU。重点是先把推理引擎的最小控制流写清楚。

## What Was Built

新增了 `GreedyEngine`，支持：

- 单请求 greedy decoding
- 连续生成多个 token
- 遇到 `eos_token_id` 时提前停止
- 不修改原始 prompt，返回新的 token 序列

核心流程：

```text
prompt tokens
  -> model.forward(current tokens)
  -> take logits[-1]
  -> argmax over vocabulary
  -> append next token
  -> repeat until max_new_tokens or EOS
```

## Why `logits[-1]`

decoder-only LLM 在每次 decode 时只关心“最后一个位置”预测下一个 token 的分布。

如果模型返回形状为 `[sequence_length, vocab_size]` 的 logits，那么：

- `logits[0]` 表示第 0 个位置预测下一个 token 的分布
- `logits[1]` 表示第 1 个位置预测下一个 token 的分布
- `logits[-1]` 表示当前序列最后一个位置预测下一个 token 的分布

所以 greedy decoding 应该只对 `logits[-1]` 做 argmax。

## TDD Process

### RED 1: Generate One Token

测试输入：

```python
generated = engine.generate([1, 2, 3], max_new_tokens=1)
```

期望输出：

```python
[1, 2, 3, 7]
```

初始失败原因：

```text
NotImplementedError: Handwrite the greedy decoding loop here.
```

### GREEN 1: Minimal Greedy Loop

实现了：

- 复制 prompt
- 调用模型 forward
- 读取最后一行 logits
- 找到最大分数 token
- 追加 token

### RED/GREEN 2: Generate Multiple Tokens

新增测试覆盖连续 decode：

```python
generated = engine.generate([1, 2, 3], max_new_tokens=3)
```

期望输出：

```python
[1, 2, 3, 7, 8, 9]
```

这个测试确认每生成一个 token 后，下一轮 forward 会使用更新后的 token 序列。

### RED/GREEN 3: Stop on EOS

新增测试覆盖 EOS 提前停止：

```python
generated = engine.generate([1, 2, 3], max_new_tokens=3, eos_token_id=2)
```

期望输出：

```python
[1, 2, 3, 7, 2]
```

这个测试确认生成 EOS 后不会继续生成后续 token。

## Validation

Milestone 1 测试命令：

```bash
PYTHONPATH=src python3 -m unittest tests/mini_engine/test_greedy_engine.py -v
```

验证结果：

```text
test_generate_appends_highest_logit_token_once ... ok
test_generate_appends_multiple_tokens_one_step_at_a_time ... ok
test_generate_stops_when_eos_token_is_generated ... ok

Ran 3 tests
OK
```

## Interview Notes

可以这样介绍这个阶段：

> I implemented a minimal greedy decoding engine to reproduce the core control flow of decoder-only LLM inference. The engine repeatedly calls the model forward pass, reads the final-position logits, selects the highest-scoring token, appends it to the sequence, and stops on either `max_new_tokens` or EOS. I used TDD to validate single-token generation, multi-step decoding, and early EOS termination.

中文面试表达：

> 我先复现了 decoder-only LLM 推理中最小的 greedy decode loop。每一步使用当前 token 序列调用模型 forward，取最后一个位置的 logits，在词表维度上做 argmax 得到下一个 token，然后追加到序列中。这个阶段用单元测试覆盖了单 token 生成、多步生成和 EOS 提前停止。

## Next Milestone

Milestone 2 将实现 KV Cache 的最小数据结构，用来解释 prefill 和 decode 阶段为什么可以避免重复计算历史 token。
