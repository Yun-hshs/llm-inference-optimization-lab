# 05 MoE Router Optimization

## Goal

实现 MoE top-k routing、expert batching 和负载均衡统计，理解 MoE 推理中的 dispatch 和 combine 瓶颈。

## Suggested Milestones

1. 实现 top-1 和 top-2 expert routing。
2. 统计 token 到 expert 的分布。
3. 实现 expert batching，减少小 batch kernel 调用。
4. 加入 capacity factor 和 dropped token 统计。
5. 比较不同 routing 策略的延迟和负载均衡。

## Deliverables

- `router.py`
- `expert_batcher.py`
- `metrics.py`
- `benchmark.py`
- `report.md`

## Interview Talking Points

- MoE 推理为什么容易出现负载不均？
- expert parallelism 和 tensor parallelism 如何配合？
- dispatch/combine 通信为什么会成为瓶颈？
