# Milestone 23: Budget-Aware Scheduler Step

## Goal

新增一个 budget-aware scheduler step，把 token update、finished request cleanup 和 budget-aware refill 串成一个完整调度步骤。

Milestone 22 已经让 refill 可以遵守 KV cache budget。本阶段继续补上 decode loop 每一步需要调用的调度入口。

## Why It Matters

真实 LLM serving loop 每轮 decode 都会做类似流程：

```text
model produces next tokens
      |
      v
append tokens to active requests
      |
      v
remove finished requests
      |
      v
refill open slots if resources allow
```

如果 refill 阶段不考虑 KV cache budget，系统仍然可能在长时间运行中把无法接纳的请求放进 active batch。

本阶段让 scheduler 拥有一条完整的 budget-aware step 路径，为后续 budget-aware serving loop 做准备。

## Key Design Boundary

旧方法保持不变：

```python
step(token_by_request_id)
```

它仍然使用普通 `refill_active_batch()`。

新增方法负责 budget-aware step：

```python
step_admissible(token_by_request_id, hidden_size=4096, bytes_per_element=2)
```

它复用已有方法：

```text
apply_tokens_to_active_requests()
remove_finished_requests()
activate_next_admissible_batch()
```

## Target API

新增：

```python
finished = scheduler.step_admissible(
    token_by_request_id,
    hidden_size=4096,
    bytes_per_element=2,
)
```

返回本轮完成的 `ActiveRequest` 列表。

## Target Data Flow

```text
token_by_request_id
      |
      v
apply_tokens_to_active_requests()
      |
      v
remove_finished_requests()
      |
      v
activate_next_admissible_batch(...)
      |
      v
return finished requests
```

## RED Test

新增测试文件：

```bash
tests/mini_engine/test_scheduler_budget_aware_step.py
```

测试内容：

- step 会应用 token、返回 finished request，并激活 budget 允许的 waiting request
- 如果 waiting request 超预算，它会继续留在 waiting queue
- 如果 token map 缺少 active request 的 token，仍然抛出 `KeyError`

## Code Framework

我已经在：

```bash
src/llm_opt_lab/mini_engine/scheduler.py
```

搭好了方法框架：

```python
def step_admissible(...):
    ...
```

你只需要补充核心调度流程。

## Your Implementation Task

实现思路：

- 调用 `self.apply_tokens_to_active_requests(token_by_request_id)`
- 调用 `finished = self.remove_finished_requests()`
- 调用 `self.activate_next_admissible_batch(...)`
- 返回 `finished`

注意：这里不要再调用 `refill_active_batch_admissible()`，否则会重复执行一次 `remove_finished_requests()`。虽然通常不会造成结果错误，但数据流会不够清晰。

## Validation

只跑 Milestone 23：

```bash
PYTHONPATH=src python3 -m unittest tests/mini_engine/test_scheduler_budget_aware_step.py -v
```

实现后建议一起跑 Milestone 21 到 23：

```bash
PYTHONPATH=src python3 -m unittest \
  tests/mini_engine/test_scheduler_budget_aware_activation.py \
  tests/mini_engine/test_scheduler_budget_aware_refill.py \
  tests/mini_engine/test_scheduler_budget_aware_step.py \
  -v
```

最后再跑 mini engine 全量测试：

```bash
PYTHONPATH=src python3 -m unittest discover tests/mini_engine -v
```
