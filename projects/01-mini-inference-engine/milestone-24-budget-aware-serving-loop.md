# Milestone 24: Budget-Aware Serving Loop

## Goal

给 `ServingLoop` 增加一条 budget-aware 运行路径，把 budget-aware activation、budget-aware step 和 token provider 串成端到端流程。

Milestone 23 已经让 scheduler 拥有 `step_admissible()`。本阶段让 serving loop 能使用这条预算感知路径执行 decode。

## Why It Matters

真实 LLM serving loop 不只是不断生成 token，还要在每轮 decode 后根据资源情况决定是否 refill 新请求。

普通 serving loop 的目标是：

```text
run until scheduler idle
```

budget-aware serving loop 还需要处理一种新状态：

```text
waiting queue still has work, but front request cannot be admitted
```

这时系统不是 idle，而是 blocked。服务循环应该返回已经完成的请求，并把被 budget 拒绝的请求留在 waiting queue 中，等待后续资源变化或调度策略处理。

## Key Design Boundary

旧方法保持不变：

```python
run_until_idle()
```

它仍然使用普通 scheduler path。

新增方法负责 budget-aware serving：

```python
run_until_blocked_or_idle_admissible(hidden_size=4096, bytes_per_element=2)
```

它复用：

```text
activate_next_admissible_batch(...)
step_admissible(...)
```

## Target API

新增：

```python
finished = loop.run_until_blocked_or_idle_admissible(
    hidden_size=4096,
    bytes_per_element=2,
)
```

返回已经完成的 `ActiveRequest` 列表。

如果因为 KV cache budget 无法激活 waiting request，方法会停止，并保留 waiting queue。

## Target Data Flow

```text
ServingLoop.run_until_blocked_or_idle_admissible()
      |
      v
activate_next_admissible_batch(...)
      |
      v
while active requests exist:
    token_provider(active_requests)
    scheduler.step_admissible(...)
      |
      v
return finished requests
```

## RED Test

新增测试文件：

```bash
tests/mini_engine/test_budget_aware_serving_loop.py
```

测试内容：

- budget 允许时，serving loop 能跑完所有 admissible 请求
- 初始 waiting request 超预算时，loop 不调用 token provider，并保留 waiting request
- refill 阶段遇到超预算请求时，loop 返回已完成请求，并保留被拒绝的 waiting request

## Code Framework

我已经在：

```bash
src/llm_opt_lab/mini_engine/serving_loop.py
```

搭好了方法框架：

```python
def run_until_blocked_or_idle_admissible(...):
    ...
```

你只需要补充核心 loop 逻辑。

## Your Implementation Task

实现思路：

- 如果当前没有 active request，先调用 `activate_next_admissible_batch(...)`
- 创建 `finished_requests = []`
- 当 `self.scheduler.active_requests` 非空时循环
- 调用 `token_provider`
- 调用 `scheduler.step_admissible(...)`
- 累加 finished requests
- 循环结束后返回 finished requests

注意：这里不要用 `while self.scheduler.has_work()`，因为 waiting request 可能因为 budget 被 blocked，导致没有 active request 但仍然有 waiting work。

## Validation

只跑 Milestone 24：

```bash
PYTHONPATH=src python3 -m unittest tests/mini_engine/test_budget_aware_serving_loop.py -v
```

实现后建议一起跑 Milestone 23 和 24：

```bash
PYTHONPATH=src python3 -m unittest \
  tests/mini_engine/test_scheduler_budget_aware_step.py \
  tests/mini_engine/test_budget_aware_serving_loop.py \
  -v
```

最后再跑 mini engine 全量测试：

```bash
PYTHONPATH=src python3 -m unittest discover tests/mini_engine -v
```
