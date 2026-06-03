# Step-by-Step Guide

这个项目会按 TDD 方式复现一个教学版推理引擎。你手写核心代码，我负责准备文件、测试和每一步的验收方式。

## Current Step: Milestone 1.1

目标：实现单请求 greedy decoding。

你要实现的行为：

1. 输入 prompt token IDs，例如 `[1, 2, 3]`。
2. 调用模型的 `forward(tokens)` 得到 logits。
3. 取最后一个位置的 logits。
4. 选择分数最大的 token ID。
5. 把这个 token 追加到输出序列。
6. 返回完整 token 序列。

示例：

```python
prompt = [1, 2, 3]
model.forward(prompt)  # 最后一个位置最喜欢 token 7
generated = [1, 2, 3, 7]
```

## Files You Will Edit

第一步你只需要手写这个文件：

```text
src/llm_opt_lab/mini_engine/engine.py
```

## Command

每写完一点，运行：

```bash
PYTHONPATH=src python -m unittest tests/mini_engine/test_greedy_engine.py -v
```

如果你本机的命令不是 `python`，就用：

```bash
PYTHONPATH=src python3 -m unittest tests/mini_engine/test_greedy_engine.py -v
```

## Learning Notes

greedy decoding 是推理引擎最小的 decode loop。真实 LLM 中 logits 来自 Transformer forward，这里先用 toy model 固定 logits，让你先把推理流程写对。

## Next Step: Milestone 2.1

目标：实现 KV Cache 的最小数据结构。

这一步先不做真正 attention，只模拟真实推理框架里的缓存形状：

- 每一层都有一组 `keys`
- 每一层都有一组 `values`
- 每 decode 一个新 token，就向某一层追加一个 key 和 value
- 可以查询某一层当前缓存了多少个 token

你要手写这个文件：

```text
src/llm_opt_lab/mini_engine/kv_cache.py
```

测试命令：

```bash
PYTHONPATH=src python3 -m unittest tests/mini_engine/test_kv_cache.py -v
```

建议实现一个小 dataclass：

```python
LayerKVCache:
    keys: list[list[float]]
    values: list[list[float]]
```

然后 `KVCache(num_layers=2)` 内部维护两个 layer cache。
