# Mini Inference Engine Report

## Project Goal

本项目实现一个轻量级 LLM inference engine，用来拆解 vLLM / SGLang 这类推理框架里的核心机制：

- greedy decoding
- prefill / decode phase
- KV cache
- continuous batching
- budget-aware admission control
- serving loop
- benchmark record and report table
- Paged KV block allocator

项目重点不是训练模型，而是理解推理服务如何在请求调度、显存管理和吞吐之间做工程权衡。

## Current Status

Project 1 已完成 32 个 Milestones。当前实现已经从最小 greedy decode loop 扩展到一个教学版 LLM serving pipeline，并包含最终报告和 Paged KV block allocator。

最新验证结果：mini engine 全量测试 `79` 个用例通过。

最新验证命令：

```bash
PYTHONPATH=src python3 -m unittest discover tests/mini_engine -v
```

## Architecture

```text
GenerationRequest
      |
      v
RequestScheduler
      |
      v
ActiveRequest
      |
      v
TokenProvider / ModelBackedTokenProvider
      |
      v
ServingLoop
      |
      v
ServingLoopResult
      |
      v
BenchmarkCase / Markdown report

PagedKVBlockAllocator
      |
      v
request_id -> block table
```

## Data Flow

1. 用户请求先进入 waiting queue。
2. Scheduler 根据 batch size 和 KV cache budget 激活请求。
3. ActiveRequest 记录 prompt、generated tokens、finished state 和 KV cache。
4. TokenProvider 每轮为 active batch 生成一个 token。
5. Scheduler 应用 token，移除完成请求，并尝试 refill。
6. ServingLoop 持续运行，直到 idle 或 blocked。
7. ServingLoopResult 汇总 finished requests 和 final status。
8. BenchmarkCase 把实验配置和 metrics 合并成 record。
9. Markdown formatter 把 records 输出成报告表格。
10. PagedKVBlockAllocator 管理固定大小 KV blocks，并维护 request block table。

## Implemented Milestones

| Area | Milestones | Result |
| --- | --- | --- |
| Basic decoding | 1-2 | 单请求 greedy decoding 和基础 KV cache |
| Scheduling | 3-10 | FIFO request queue、active batch、refill、step、serving loop |
| Model integration | 11-14 | model-backed token provider、prefill/decode 输入区分 |
| KV lifecycle | 15-18 | per-request KV cache、完成释放、active KV 统计 |
| Budget control | 19-25 | 显存预算、admission、blocked state、budget-aware serving loop |
| Observability | 26-30 | status snapshot、metrics summary、benchmark record、Markdown table |
| Project report | 31 | 最终项目报告、数据流、测试和面试讲解整理 |
| Paged KV | 32 | block-level KV allocation、release、reuse 和错误处理 |

## Key Behaviors

### Continuous Batching

Scheduler 不等待整个 batch 完成。每轮 decode 后会：

1. 给 active requests 应用新 token。
2. 移除 finished requests。
3. 从 waiting queue 中 refill 新请求。

这模拟了真实 LLM serving 中提升吞吐的 continuous batching 机制。

### KV Cache Accounting

每个 active request 可以持有自己的 KV cache。Scheduler 可以统计：

- active KV entries
- active KV memory bytes
- remaining KV budget
- over-budget state

内存估算公式：

```text
entries * hidden_size * bytes_per_element * 2
```

其中 `2` 表示 key 和 value。

### Budget-Aware Admission

Scheduler 在激活请求前估算 prompt prefill 需要的 KV cache memory。如果激活后会超过预算，请求保留在 waiting queue。

重要边界：

- 不跳过队首被拒绝请求去激活后面的请求。
- blocked state 和 idle state 分开表示。
- `has_work()` 仍然表示 waiting 或 active 中还有工作。

### Serving Metrics

ServingLoopResult 可以汇总：

- finished request count
- generated token count
- output token count
- final queue state
- KV cache memory state
- stopped reason

停止原因包括：

- `idle`
- `blocked_by_kv_budget`
- `running`

### Paged KV Block Allocation

PagedKVBlockAllocator 用固定数量的 blocks 模拟 Paged KV memory pool：

```text
request_id -> [block_id_0, block_id_1, ...]
```

核心行为：

- `allocate(request_id, num_blocks)` 为请求分配 free blocks。
- `free(request_id)` 释放请求持有的 blocks。
- `block_table(request_id)` 返回请求的 block table copy。
- allocator 统计 free / allocated block count。
- free 后的 blocks 可以被后续请求复用。
- 当 free blocks 不足时抛出 `MemoryError`，并保持已有状态不变。

这对应真实 PagedAttention 的核心思想：请求不需要连续显存，只需要维护 block table。

## Benchmark Table Example

| case_name | max_batch_size | hidden_size | bytes_per_element | finished_request_count | generated_token_count | output_token_count | waiting_count | active_count | active_kv_cache_memory_bytes | remaining_kv_cache_budget_bytes | max_kv_cache_memory_bytes | stopped_reason |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| batch-1 | 1 | 4 | 2 | 1 | 2 | 4 | 0 | 0 | 0 | 128 | 128 | idle |
| blocked | 1 | 4 | 2 | 0 | 0 | 0 | 1 | 0 | 0 | 15 | 15 | blocked_by_kv_budget |

## Testing

当前 mini engine 测试覆盖：

- greedy decoding
- KV cache append / clear / accounting
- request scheduling
- active request lifecycle
- serving loop
- model-backed token provider
- budget-aware admission / refill / step
- blocked state
- metrics summary
- benchmark markdown formatting
- Paged KV allocator behavior

当前测试规模：

```text
Ran 79 tests
OK
```

推荐验证命令：

```bash
PYTHONPATH=src python3 -m unittest discover tests/mini_engine -v
```

## Interview Talking Points

- Prefill 是大矩阵计算，decode 是逐 token 增量计算，二者性能瓶颈不同。
- KV cache 用显存换计算，避免重复计算历史 token 的 key/value。
- Continuous batching 通过动态 refill 提高 GPU 利用率。
- Budget-aware admission control 可以防止 KV cache 显存超过上限。
- Block-level Paged KV 可以降低连续显存分配压力，为 PagedAttention 打基础。
- Paged KV 的 block table 让 request 和物理 KV block 解耦，便于复用和减少碎片。

## Limitations

当前项目是教学型 mini engine，不包含：

- 真实 Transformer attention kernel
- CUDA / HIP / OpenCL 算子
- FlashAttention
- int4 / int8 quantized GEMM
- tensor parallel / pipeline parallel
- NCCL communication
- 真实 vLLM / SGLang runtime integration

这些能力建议放到后续算子优化和量化项目中实现。
