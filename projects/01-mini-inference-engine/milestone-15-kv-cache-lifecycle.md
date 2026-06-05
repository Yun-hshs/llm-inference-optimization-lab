# Milestone 15: Per-Request KV Cache Lifecycle

## Goal

让 `RequestScheduler` 在请求进入 active batch 时，为每个 `ActiveRequest` 分配独立的 KV cache。

Milestone 14 已经让 token provider 按 `prefill` / `decode` 阶段选择不同 model input。本阶段继续补上推理系统里的另一个关键状态：每个请求自己的 KV cache 生命周期。

## Why It Matters

在真实 LLM serving 系统中，请求进入推理队列后通常会拥有自己的 KV cache 状态：

- `prefill` 阶段写入 prompt 对应的 key/value
- `decode` 阶段复用历史 key/value，只追加新 token 的 key/value
- 请求完成后释放或回收 cache 资源

如果所有请求共享一份 cache，batch serving 会很难维护请求边界，也无法安全地做 continuous batching、cache eviction 或 paged KV。

## Key Design Boundary

`RequestScheduler` 负责 active request 的生命周期：

```text
waiting request -> active request -> finished request
```

所以本阶段让 scheduler 在 request 被激活时分配 KV cache：

```text
GenerationRequest
      |
      v
ActiveRequest + KVCache
      |
      v
active batch
```

`KVCacheAwareTokenProvider` 仍然只负责选择 model input 和 next token，不负责创建 active request。

## Target API

本阶段目标接口：

```python
scheduler = RequestScheduler(max_batch_size=2, num_kv_layers=2)
```

当 `num_kv_layers` 被设置时：

```text
activate_next_batch() -> ActiveRequest.kv_cache is not None
refill_active_batch() -> newly activated ActiveRequest.kv_cache is not None
```

为了保持旧测试兼容，如果没有传 `num_kv_layers`，可以继续不分配 KV cache。

## Target Data Flow

```text
RequestScheduler.add_request()
      |
      v
request_queue
      |
      v
activate_next_batch() / refill_active_batch()
      |
      v
ActiveRequest.from_request(..., num_kv_layers=2)
      |
      v
ActiveRequest.kv_cache = KVCache(num_layers=2)
```

## RED Test

新增测试文件：

```bash
tests/mini_engine/test_scheduler_kv_cache_lifecycle.py
```

测试内容：

- scheduler 激活首批请求时，为 active request 分配 KV cache
- scheduler refill 新请求时，也为新 active request 分配 KV cache
- 新分配的 cache 初始 sequence length 为 0

## Your Implementation Task

你需要改两个位置：

```bash
src/llm_opt_lab/mini_engine/scheduler.py
```

建议实现方向：

- 给 `ActiveRequest` 增加可选字段 `kv_cache`
- 让 `ActiveRequest.from_request()` 可选接收 `num_kv_layers`
- 让 `RequestScheduler.__init__()` 可选接收 `num_kv_layers`
- 在 `activate_next_batch()` 和 `refill_active_batch()` 创建 `ActiveRequest` 时传入 `num_kv_layers`

注意：本阶段不需要在 `append_token()` 里写 cache，也不需要改 token provider。

## Validation

只跑 Milestone 15：

```bash
PYTHONPATH=src python3 -m unittest tests/mini_engine/test_scheduler_kv_cache_lifecycle.py -v
```

实现后建议一起跑 Milestone 13 到 15：

```bash
PYTHONPATH=src python3 -m unittest \
  tests/mini_engine/test_request_phase.py \
  tests/mini_engine/test_kv_cache_aware_token_provider.py \
  tests/mini_engine/test_scheduler_kv_cache_lifecycle.py \
  -v
```

最后再跑 mini engine 全量测试：

```bash
PYTHONPATH=src python3 -m unittest discover tests/mini_engine -v
```
