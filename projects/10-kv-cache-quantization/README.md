# 10 KV Cache Quantization

## Goal

实现 FP16、INT8、INT4 KV cache 方案，对比长上下文推理下的显存占用、误差和延迟。

## Suggested Milestones

1. 计算 KV cache 显存公式。
2. 实现 FP16 KV cache baseline。
3. 实现 INT8 KV cache quant/dequant。
4. 实现 INT4 KV cache packed storage。
5. 评估不同 context length 的 memory saving 和误差。

## Deliverables

- `kv_cache.py`
- `kv_quantize.py`
- `memory_model.py`
- `benchmark.py`
- `report.md`

## Interview Talking Points

- KV cache 为什么是长上下文推理的主要显存压力？
- KV cache 量化和权重量化有什么区别？
- 如何权衡精度、显存和 dequant 开销？
