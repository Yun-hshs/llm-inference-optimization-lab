# Milestone 21: Budget-Aware Batch Activation

## Goal

新增一个 budget-aware 的 batch activation 方法，让 scheduler 在激活 waiting request 之前先做 KV cache admission check。

Milestone 20 已经能判断单个请求是否可以被当前 KV cache budget 接纳。本阶段把这个判断接入一个新的调度入口，但暂时不替换旧的 `activate_next_batch()`，从而保持已有 milestone 的行为稳定。

## Why It Matters

真实 LLM serving scheduler 不应该只按照 batch size 激活请求。它还需要考虑新请求 prefill 后会带来的 KV cache 显存压力。

如果不做 admission-aware activation，系统可能出现：

- batch slot 还有空位，但显存预算已经不足
- 长 prompt 请求被激活后导致 KV cache 超预算
- scheduler 无法对上游请求施加 backpressure

本阶段开始把资源预算从“观测指标”变成“调度决策输入”。

## Key Design Boundary

旧方法保持不变：

```python
activate_next_batch()
```

它仍然只按 batch size 激活请求。

新增方法负责 budget-aware activation：

```python
activate_next_admissible_batch(hidden_size=4096, bytes_per_element=2)
```

它使用 Milestone 20 的：

```python
can_admit_request(...)
```

来决定队首请求是否可以进入 active batch。

## FIFO Rule

本阶段保持严格 FIFO：

```text
if front request cannot be admitted:
    stop activation
    do not skip it
```

原因是：如果跳过队首请求，会引入更复杂的公平性策略。后续可以单独做 priority scheduling 或 shortest-prompt-first 策略。

## Target API

新增：

```python
activated = scheduler.activate_next_admissible_batch(
    hidden_size=4096,
    bytes_per_element=2,
)
```

返回本次被激活的 `ActiveRequest` 列表。

## Target Data Flow

```text
request_queue[0]
      |
      v
can_admit_request(...)
      |
      v
yes -> pop from waiting queue -> ActiveRequest.from_request(...) -> active_requests
no  -> keep request in queue -> stop activation
```

## RED Test

新增测试文件：

```bash
tests/mini_engine/test_scheduler_budget_aware_activation.py
```

测试内容：

- budget 允许时激活队首请求
- budget 不允许时队首请求继续留在 waiting queue
- 不跳过被拒绝的队首请求去激活后面的请求
- active batch 达到容量后停止激活

## Code Framework

我已经在：

```bash
src/llm_opt_lab/mini_engine/scheduler.py
```

搭好了方法框架：

```python
def activate_next_admissible_batch(...):
    ...
```

你只需要补充核心 FIFO admission activation 逻辑。

## Your Implementation Task

实现思路：

- 创建 `activated = []`
- 当 active batch 未满并且 waiting queue 非空时循环
- 只查看 `self.request_queue[0]`
- 如果 `can_admit_request(...)` 返回 `False`，立即 `break`
- 否则从队首 `pop(0)`，创建 `ActiveRequest`
- 追加到 `self.active_requests` 和 `activated`
- 返回 `activated`

注意：创建 `ActiveRequest` 时要传入 `self.num_kv_layers`。

## Validation

只跑 Milestone 21：

```bash
PYTHONPATH=src python3 -m unittest tests/mini_engine/test_scheduler_budget_aware_activation.py -v
```

实现后建议一起跑 Milestone 20 和 21：

```bash
PYTHONPATH=src python3 -m unittest \
  tests/mini_engine/test_scheduler_request_admission.py \
  tests/mini_engine/test_scheduler_budget_aware_activation.py \
  -v
```

最后再跑 mini engine 全量测试：

```bash
PYTHONPATH=src python3 -m unittest discover tests/mini_engine -v
```
