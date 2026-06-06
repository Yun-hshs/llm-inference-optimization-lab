# Milestone 20: Budget-Aware Request Admission Check

## Goal

让 `RequestScheduler` 能在真正激活请求之前，预估这个请求的 prefill KV cache 显存，并判断当前 KV cache budget 是否还能接纳它。

Milestone 19 已经能判断 active batch 当前是否超预算。本阶段继续向真实 serving scheduler 靠近：调度器不仅要知道当前用了多少资源，还要能预估新请求加入后的资源压力。

## Why It Matters

真实 LLM serving 系统在接收请求时，需要考虑新请求的 prompt 长度。

两个请求可能都只生成 1 个 token，但 prompt 长度不同会导致 prefill KV cache 占用完全不同：

- 短 prompt 请求可能很容易接纳
- 长 prompt 请求可能在 prefill 后立刻吃掉大量 KV cache
- 如果没有 admission check，scheduler 可能把请求放入 active batch 后才发现显存不够

这一步对应 serving 系统里的 admission control / backpressure 的前置能力。

## Key Design Boundary

`KVCache` 负责已经存在的 cache accounting。

`RequestScheduler` 负责 admission decision：

```text
current active KV memory + projected request prefill KV memory <= budget
```

本阶段不改变 `activate_next_batch()` 的行为，只新增查询方法。下一阶段可以把 admission check 接入真正的调度流程。

## Target API

新增 request prefill memory 估算：

```python
scheduler.estimate_request_prefill_kv_memory_bytes(
    request,
    hidden_size=4096,
    bytes_per_element=2,
)
```

估算公式：

```text
len(request.prompt) * num_kv_layers * 2 * hidden_size * bytes_per_element
```

其中 `2` 表示 key 和 value 两份向量。

新增 admission check：

```python
scheduler.can_admit_request(
    request,
    hidden_size=4096,
    bytes_per_element=2,
)
```

如果没有配置 `max_kv_cache_memory_bytes`，默认可以接纳。

## Target Data Flow

```text
GenerationRequest.prompt
      |
      v
estimate request prefill KV memory
      |
      v
active_kv_cache_memory_bytes(...)
      |
      v
current usage + projected usage
      |
      v
compare with max_kv_cache_memory_bytes
      |
      v
can admit / cannot admit
```

## RED Test

新增测试文件：

```bash
tests/mini_engine/test_scheduler_request_admission.py
```

测试内容：

- 能按 prompt length 和 layer 数预估 request prefill KV memory
- projected usage 等于 budget 时可以接纳
- projected usage 超过 budget 时拒绝
- 未配置 budget 的 scheduler 默认可以接纳请求

## Code Framework

我已经在：

```bash
src/llm_opt_lab/mini_engine/scheduler.py
```

搭好了方法框架：

```python
def estimate_request_prefill_kv_memory_bytes(...):
    ...

def can_admit_request(...):
    ...
```

你只需要补充核心公式和判断逻辑。

## Your Implementation Task

在 `estimate_request_prefill_kv_memory_bytes()` 中：

- 如果 `self.num_kv_layers is None`，返回 0
- 否则按公式计算 request prefill KV memory

在 `can_admit_request()` 中：

- 如果没有配置 `max_kv_cache_memory_bytes`，返回 `True`
- 计算当前 active KV cache memory
- 计算 request prefill projected memory
- 当两者之和小于等于 budget 时返回 `True`

## Validation

只跑 Milestone 20：

```bash
PYTHONPATH=src python3 -m unittest tests/mini_engine/test_scheduler_request_admission.py -v
```

实现后建议一起跑 Milestone 19 和 20：

```bash
PYTHONPATH=src python3 -m unittest \
  tests/mini_engine/test_scheduler_kv_cache_budget.py \
  tests/mini_engine/test_scheduler_request_admission.py \
  -v
```

最后再跑 mini engine 全量测试：

```bash
PYTHONPATH=src python3 -m unittest discover tests/mini_engine -v
```
