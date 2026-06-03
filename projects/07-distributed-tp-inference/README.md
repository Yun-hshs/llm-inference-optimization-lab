# 07 Distributed TP Inference

## Goal

实现一个最小 Tensor Parallel 推理 Demo，理解多 GPU 推理中的算子切分和通信开销。

## Suggested Milestones

1. 实现单卡 Linear baseline。
2. 实现 column parallel linear。
3. 实现 row parallel linear。
4. 使用 PyTorch distributed 做 all-reduce / all-gather。
5. 统计计算时间和通信时间。

## Deliverables

- `single_gpu.py`
- `tp_linear.py`
- `distributed_benchmark.py`
- `report.md`

## Interview Talking Points

- TP、PP、DP、EP 分别解决什么问题？
- column parallel 和 row parallel 的通信模式有什么不同？
- 通信带宽如何限制多 GPU 扩展效率？
