# Milestone 22: Budget-Aware Refill

## Goal

新增一个 budget-aware refill 方法，让 scheduler 在清理 finished request 后，只激活 KV cache budget 允许的新请求。

Milestone 21 已经实现了 budget-aware batch activation。本阶段把这个能力接到 refill 场景里，让 continuous batching 更接近真实 serving 系统。

## Why It Matters

Continuous batching 的核心不是只在服务启动时组 batch，而是在 decode 过程中不断：

```text
remove finished requests -> refill open slots with waiting requests
```

如果 refill 不考虑 KV cache budget，系统仍然可能在运行中把过大的 prompt 请求塞进 active batch，导致显存压力失控。

本阶段让 refill 也具备 admission control 的入口，为后续 budget-aware serving loop 做准备。

## Key Design Boundary

旧方法保持不变：

```python
refill_active_batch()
```

它仍然只按 batch slot 补位。

新增方法负责 budget-aware refill：

```python
refill_active_batch_admissible(hidden_size=4096, bytes_per_element=2)
```

它复用：

```python
remove_finished_requests()
activate_next_admissible_batch(...)
```

## FIFO Rule

和 Milestone 21 一样，本阶段保持严格 FIFO：

```text
if front waiting request cannot be admitted:
    stop refill
    do not skip it
```

这样 scheduler 的行为容易解释，也避免提前引入公平性和优先级策略。

## Target API

新增：

```python
activated = scheduler.refill_active_batch_admissible(
    hidden_size=4096,
    bytes_per_element=2,
)
```

返回本次 refill 新激活的 `ActiveRequest` 列表。

## Target Data Flow

```text
active_requests
      |
      v
remove_finished_requests()
      |
      v
open active slots
      |
      v
activate_next_admissible_batch(...)
      |
      v
newly activated requests
```

## RED Test

新增测试文件：

```bash
tests/mini_engine/test_scheduler_budget_aware_refill.py
```

测试内容：

- refill 会移除完成请求，并激活 budget 允许的 waiting request
- budget 拒绝队首请求时，请求继续留在 waiting queue
- 不跳过被拒绝的队首请求去激活后面的请求
- 已有 active request 占用 batch slot 时，只填充剩余容量

## Code Framework

我已经在：

```bash
src/llm_opt_lab/mini_engine/scheduler.py
```

搭好了方法框架：

```python
def refill_active_batch_admissible(...):
    ...
```

你只需要补充核心调用逻辑。

## Your Implementation Task

实现思路：

- 调用 `self.remove_finished_requests()`
- 调用并返回 `self.activate_next_admissible_batch(...)`

注意：不要直接操作 waiting queue。复用 Milestone 21 的方法，保持 admission 逻辑只有一个入口。

## Validation

只跑 Milestone 22：

```bash
PYTHONPATH=src python3 -m unittest tests/mini_engine/test_scheduler_budget_aware_refill.py -v
```

实现后建议一起跑 Milestone 21 和 22：

```bash
PYTHONPATH=src python3 -m unittest \
  tests/mini_engine/test_scheduler_budget_aware_activation.py \
  tests/mini_engine/test_scheduler_budget_aware_refill.py \
  -v
```

最后再跑 mini engine 全量测试：

```bash
PYTHONPATH=src python3 -m unittest discover tests/mini_engine -v
```
