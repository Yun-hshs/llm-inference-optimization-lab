# 01 Mini Inference Engine

## Goal

实现一个轻量级 Transformer decoder 推理引擎，覆盖 prefill、decode、KV cache、continuous batching 和 Paged KV 的核心概念。

## Why It Matters

这是理解 vLLM、SGLang 等推理框架的基础项目。面试时可以从请求调度、显存管理、吞吐和延迟权衡展开。

## Suggested Milestones

1. 实现单请求 greedy decoding。
2. 加入 KV cache，比较有无 cache 的 decode latency。
3. 加入 batch decoding。
4. 实现 continuous batching scheduler。
5. 模拟 Paged KV block allocator。
6. 输出不同 batch size 和 context length 的 benchmark。

## Deliverables

- `engine.py`: 推理引擎入口。
- `scheduler.py`: 请求调度逻辑。
- `kv_cache.py`: KV cache 和 block allocator。
- `benchmark.py`: latency、tokens/s、memory benchmark。
- `report.md`: 实验结论。

## Interview Talking Points

- prefill 和 decode 为什么性能特征不同？
- KV cache 如何降低重复计算？
- continuous batching 为什么能提升吞吐？
- PagedAttention / Paged KV 解决了什么显存碎片问题？
