# Milestone 16: KV Cache Release on Completion

## Goal

让 `RequestScheduler` 在请求完成并离开 active batch 时，释放这个请求持有的 KV cache。

Milestone 15 已经让 scheduler 在请求激活时分配 per-request KV cache。本阶段补上生命周期的另一半：请求结束后要清理 cache 资源。

## Why It Matters

真实 LLM serving 系统中，KV cache 通常占用大量显存。请求完成后如果不释放 cache，会导致：

- active batch 可用容量下降
- 长时间服务后显存碎片和泄漏风险增加
- 后续请求无法复用已经空闲的 cache 资源

vLLM 的 paged KV、SGLang 的 cache 管理，本质上都围绕一个问题：如何让请求的 KV cache 分配、复用和释放足够高效。

## Key Design Boundary

`RequestScheduler` 负责请求生命周期：

```text
active request -> finished request -> remove from active batch
```

所以 scheduler 也是当前最合适的 cache release 触发点。

`KVCache` 负责具体清理动作：

```text
kv_cache.clear()
```

`TokenProvider` 不负责释放资源，因为它只负责模型输入和 next token 选择。

## Target API

本阶段不新增外部公开 API，只在 scheduler 内部增加资源释放流程：

```python
remove_finished_requests()
      |
      v
_release_finished_request_resources(finished_requests)
      |
      v
active.kv_cache.clear()
```

## Target Data Flow

```text
ActiveRequest.finished == True
      |
      v
RequestScheduler.remove_finished_requests()
      |
      v
clear finished request KV cache
      |
      v
remove request from active_requests
      |
      v
return finished requests
```

## RED Test

新增测试文件：

```bash
tests/mini_engine/test_scheduler_kv_cache_release.py
```

测试内容：

- `remove_finished_requests()` 会清空 finished request 的 KV cache
- `step()` 在移除完成请求并 refill 前，也会清空 finished request 的 KV cache
- 新 refill 进来的请求仍然会正常分配自己的 KV cache

## Code Framework

我已经在：

```bash
src/llm_opt_lab/mini_engine/scheduler.py
```

搭好了内部方法框架：

```python
def _release_finished_request_resources(self, finished_requests: list[ActiveRequest]) -> None:
    ...
```

你只需要补充核心清理逻辑。

## Your Implementation Task

在 `_release_finished_request_resources()` 中，对每个带有 `kv_cache` 的 finished request 调用：

```python
active.kv_cache.clear()
```

注意：不要把 `active.kv_cache` 直接设成 `None`。本阶段测试希望 finished request 仍然能被返回，并且可以观察到它的 cache 已经被清空。

## Validation

只跑 Milestone 16：

```bash
PYTHONPATH=src python3 -m unittest tests/mini_engine/test_scheduler_kv_cache_release.py -v
```

实现后建议一起跑 Milestone 15 和 16：

```bash
PYTHONPATH=src python3 -m unittest \
  tests/mini_engine/test_scheduler_kv_cache_lifecycle.py \
  tests/mini_engine/test_scheduler_kv_cache_release.py \
  -v
```

最后再跑 mini engine 全量测试：

```bash
PYTHONPATH=src python3 -m unittest discover tests/mini_engine -v
```
