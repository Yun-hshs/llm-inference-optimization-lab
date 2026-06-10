# Milestone 28: Serving Loop Metrics Summary

## Goal

给 `ServingLoopResult` 增加一个指标汇总方法，把 finished requests 和 final status 转换成适合日志、benchmark 和报告展示的 summary。

本阶段不重新执行 serving loop，也不读取 scheduler。它只基于已经返回的结果做统计。

## Why It Matters

Milestone 27 已经让 serving loop 返回：

- `finished_requests`
- `final_status`

但真实推理服务还需要把结果转换成可读、可比较的指标。例如：

- 本轮完成了多少请求
- 生成了多少 token
- 最终是 idle 还是 blocked
- KV cache 还剩多少 budget

这些指标可以直接用于后续 benchmark、性能报告和面试讲解。

## Key Design Boundary

`metrics_summary()` 必须只从 `ServingLoopResult` 自身读数据：

- 不调用 token provider
- 不调用 scheduler
- 不修改 finished requests
- 不修改 final status

它是一个纯汇总函数。

## Target API

新增：

```python
summary = result.metrics_summary()
```

返回：

```python
{
    "finished_request_count": 2,
    "generated_token_count": 3,
    "output_token_count": 6,
    "waiting_count": 0,
    "active_count": 0,
    "active_kv_cache_entries": 0,
    "active_kv_cache_memory_bytes": 0,
    "remaining_kv_cache_budget_bytes": 128,
    "max_kv_cache_memory_bytes": 128,
    "stopped_reason": "idle",
}
```

## Target Data Flow

```text
finished_requests
      |
      v
finished_request_count / generated_token_count / output_token_count

final_status
      |
      v
waiting_count / active_count / KV cache metrics / stopped_reason

finished_requests + final_status
      |
      v
metrics_summary
```

## Stopped Reason

根据 final status 推导：

```text
is_idle == True
      -> "idle"

is_blocked_by_kv_budget == True
      -> "blocked_by_kv_budget"

otherwise
      -> "running"
```

## RED Test

新增测试文件：

```bash
tests/mini_engine/test_serving_loop_metrics.py
```

测试内容：

- finished requests 和 idle status 可以汇总成完整 metrics
- blocked status 可以输出 `blocked_by_kv_budget`
- 既不是 idle 也不是 blocked 时输出 `running`

## Code Framework

我已经在：

```bash
src/llm_opt_lab/mini_engine/serving_loop.py
```

搭好了：

```python
def metrics_summary(...):
    ...
```

你只需要补充核心统计逻辑。

## Your Implementation Task

实现思路：

- `finished_request_count`: `len(self.finished_requests)`
- `generated_token_count`: 所有 `active.generated_tokens` 的长度之和
- `output_token_count`: 所有 `active.output_tokens()` 的长度之和
- 从 `self.final_status` 中读取 waiting、active 和 KV cache 字段
- 根据 `is_idle` 和 `is_blocked_by_kv_budget` 推导 `stopped_reason`

注意：这里不要调用 serving loop，也不要调用 scheduler。

## Validation

只跑 Milestone 28：

```bash
PYTHONPATH=src python3 -m unittest tests/mini_engine/test_serving_loop_metrics.py -v
```

实现后建议一起跑 Milestone 27 和 28：

```bash
PYTHONPATH=src python3 -m unittest \
  tests/mini_engine/test_serving_loop_result.py \
  tests/mini_engine/test_serving_loop_metrics.py \
  -v
```

最后再跑 mini engine 全量测试：

```bash
PYTHONPATH=src python3 -m unittest discover tests/mini_engine -v
```
