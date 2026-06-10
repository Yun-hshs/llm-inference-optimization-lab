# Milestone 27: Serving Loop Result

## Goal

给 budget-aware serving loop 增加一个结构化返回结果，同时包含：

- 本次 loop 完成的 requests
- loop 停止时的最终 scheduler status

这样调用方可以直接判断本轮是 `idle` 结束，还是因为 KV cache budget `blocked` 结束。

## Why It Matters

Milestone 24 让 serving loop 能在 blocked 或 idle 时停止。

Milestone 26 让 scheduler 可以输出状态快照。

本阶段把两者连接起来：serving loop 不只返回完成请求，也返回最终状态。这是工程落地里很常见的接口设计，因为上层服务需要根据结果决定：

- 是否继续等待新请求
- 是否触发 backpressure
- 是否记录 blocked metric
- 是否把状态写入 benchmark/report

## Key Design Boundary

保留旧接口：

```python
run_until_blocked_or_idle_admissible(...)
```

新增一个带 status 的接口：

```python
run_until_blocked_or_idle_admissible_with_status(...)
```

旧接口继续只返回 `list[ActiveRequest]`，避免破坏前面 Milestone 的测试和调用方式。

## Target API

新增结果类型：

```python
@dataclass
class ServingLoopResult:
    finished_requests: list[ActiveRequest]
    final_status: dict[str, int | bool | None]
```

新增方法：

```python
result = loop.run_until_blocked_or_idle_admissible_with_status(
    hidden_size=4096,
    bytes_per_element=2,
)
```

使用方式：

```python
result.finished_requests
result.final_status["is_idle"]
result.final_status["is_blocked_by_kv_budget"]
```

## Target Data Flow

```text
run_until_blocked_or_idle_admissible(...)
      |
      v
finished_requests

scheduler.status(...)
      |
      v
final_status

finished_requests + final_status
      |
      v
ServingLoopResult
```

## RED Test

新增测试文件：

```bash
tests/mini_engine/test_serving_loop_result.py
```

测试内容：

- 请求全部完成时，返回 finished requests 和 idle final status
- 队首请求因为 KV budget 被阻塞时，返回空 finished requests 和 blocked final status

## Code Framework

我已经在：

```bash
src/llm_opt_lab/mini_engine/serving_loop.py
```

搭好了：

```python
@dataclass
class ServingLoopResult:
    ...

def run_until_blocked_or_idle_admissible_with_status(...):
    ...
```

你只需要补充核心逻辑。

## Your Implementation Task

实现思路：

- 调用已有的 `run_until_blocked_or_idle_admissible(...)` 得到 `finished_requests`
- 调用 `self.scheduler.status(...)` 得到 `final_status`
- 返回 `ServingLoopResult(finished_requests=..., final_status=...)`

注意：不要复制一遍 serving loop 主循环。复用已有方法，保持行为一致。

## Validation

只跑 Milestone 27：

```bash
PYTHONPATH=src python3 -m unittest tests/mini_engine/test_serving_loop_result.py -v
```

实现后建议一起跑 Milestone 26 和 27：

```bash
PYTHONPATH=src python3 -m unittest \
  tests/mini_engine/test_scheduler_status.py \
  tests/mini_engine/test_serving_loop_result.py \
  -v
```

最后再跑 mini engine 全量测试：

```bash
PYTHONPATH=src python3 -m unittest discover tests/mini_engine -v
```
