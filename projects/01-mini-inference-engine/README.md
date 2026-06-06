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

## Milestone Reports

- [Milestone 1: Single-Request Greedy Decoding](./milestone-1-greedy-decoding.md)
- [Milestone 2: Minimal KV Cache](./milestone-2-kv-cache.md)
- [Milestone 3: Request Scheduler](./milestone-3-request-scheduler.md)
- [Milestone 4: Active Request State](./milestone-4-active-request.md)
- [Milestone 5: Active Batch Scheduling](./milestone-5-active-batch.md)
- [Milestone 6: Refill Active Batch](./milestone-6-refill-active-batch.md)
- [Milestone 7: Decode Step Token Update](./milestone-7-decode-step-update.md)
- [Milestone 8: End-to-End Scheduler Step](./milestone-8-scheduler-step.md)
- [Milestone 9: Scheduler Lifecycle](./milestone-9-scheduler-lifecycle.md)
- [Milestone 10: Serving Loop](./milestone-10-serving-loop.md)
- [Milestone 11: Model-Backed Token Provider](./milestone-11-model-token-provider.md)
- [Milestone 12: Model-Driven Serving Loop](./milestone-12-model-serving-loop.md)
- [Milestone 13: Prefill / Decode Phase Tracking](./milestone-13-prefill-decode-phase.md)
- [Milestone 14: KV-Cache-Aware Token Provider](./milestone-14-kv-cache-aware-token-provider.md)
- [Milestone 15: Per-Request KV Cache Lifecycle](./milestone-15-kv-cache-lifecycle.md)
- [Milestone 16: KV Cache Release on Completion](./milestone-16-kv-cache-release.md)
- [Milestone 17: KV Cache Usage Accounting](./milestone-17-kv-cache-accounting.md)
- [Milestone 18: Active Batch KV Cache Accounting](./milestone-18-active-kv-cache-accounting.md)
- [Milestone 19: KV Cache Memory Budget Guard](./milestone-19-kv-cache-budget.md)
- [Milestone 20: Budget-Aware Request Admission Check](./milestone-20-request-admission.md)
- [Milestone 21: Budget-Aware Batch Activation](./milestone-21-budget-aware-activation.md)
- [Milestone 22: Budget-Aware Refill](./milestone-22-budget-aware-refill.md)
- [Milestone 23: Budget-Aware Scheduler Step](./milestone-23-budget-aware-step.md)
- [Milestone 24: Budget-Aware Serving Loop](./milestone-24-budget-aware-serving-loop.md)
- [Milestone 25: KV Budget Blocked State](./milestone-25-blocked-state.md)

## Interview Talking Points

- prefill 和 decode 为什么性能特征不同？
- KV cache 如何降低重复计算？
- continuous batching 为什么能提升吞吐？
- PagedAttention / Paged KV 解决了什么显存碎片问题？
