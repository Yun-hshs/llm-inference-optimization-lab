# Milestone 32: Paged KV Block Allocator

## Goal

实现一个简化版 Paged KV block allocator，用固定大小的 KV blocks 管理请求的 block table。

这一步不是实现真实 PagedAttention kernel，而是先模拟 vLLM 中最核心的内存管理思想：

```text
request_id -> [block_id_0, block_id_1, ...]
```

## Why It Matters

普通 KV cache 可以理解为每个请求持有自己的连续缓存。真实 LLM serving 中，请求长度不同、生命周期不同，如果要求连续显存，很容易产生碎片和浪费。

Paged KV 把 KV cache 拆成固定大小 blocks：

- 请求只保存 block table
- block 可以从全局池分配
- 请求结束后 block 回到 free pool
- 后续请求可以复用释放的 block

这是理解 vLLM PagedAttention 的基础。

## Target API

新增文件：

```bash
src/llm_opt_lab/mini_engine/paged_kv.py
```

新增：

```python
@dataclass
class PagedKVBlock:
    block_id: int
    owner_request_id: str | None = None


class PagedKVBlockAllocator:
    def __init__(self, num_blocks: int) -> None:
        ...

    def allocate(self, request_id: str, num_blocks: int) -> list[int]:
        ...

    def free(self, request_id: str) -> list[int]:
        ...

    def block_table(self, request_id: str) -> list[int]:
        ...

    def free_block_count(self) -> int:
        ...

    def allocated_block_count(self) -> int:
        ...
```

## Target Data Flow

```text
free blocks
      |
      v
allocate(request_id, num_blocks)
      |
      v
block_tables[request_id] = [block ids]
      |
      v
free(request_id)
      |
      v
released blocks return to free pool
```

## RED Test

新增测试文件：

```bash
tests/mini_engine/test_paged_kv_allocator.py
```

测试内容：

- 分配 free blocks，并记录 request block table
- free 后 block 可以被后续请求复用
- free blocks 不足时抛出 `MemoryError`，并保持已有状态不变
- 拒绝重复 request id
- unknown request free / block_table 抛出 `KeyError`
- 拒绝非正数 block 数量

## Code Framework

代码位于：

```bash
src/llm_opt_lab/mini_engine/paged_kv.py
```

核心逻辑已经实现，包括：

- free block 统计
- allocated block 统计
- request block table 分配
- request block table 查询
- block 释放和复用
- duplicate request 校验
- free blocks 不足时保持状态不变
- unknown request 错误处理

## Implementation Notes

- `free_block_count`: 统计 `block.is_free()` 的 blocks
- `allocated_block_count`: 统计已被 request 持有的 blocks
- `allocate`:
  - 校验 `num_blocks > 0`
  - 校验 `request_id` 不在 `self.block_tables`
  - 找到前 `num_blocks` 个 free block
  - 不足则抛 `MemoryError`
  - 设置每个 block 的 `owner_request_id`
  - 写入 `self.block_tables[request_id]`
  - 返回 block id list
- `free`:
  - 如果 request 不存在，抛 `KeyError`
  - 取出 block ids
  - 把对应 block 的 owner 清空
  - 删除 block table
  - 返回释放的 block ids
- `block_table`:
  - 如果 request 不存在，抛 `KeyError`
  - 返回 block id list 的 copy

## Result

Milestone 32 已完成。Paged KV allocator 当前支持：

```python
allocator = PagedKVBlockAllocator(num_blocks=4)
allocator.allocate("req-1", num_blocks=2)  # [0, 1]
allocator.block_table("req-1")             # [0, 1]
allocator.free("req-1")                    # [0, 1]
```

释放后的 blocks 会重新进入 free pool，可被后续请求复用。

## Validation

Milestone 32 单测已通过：

```text
Ran 6 tests
OK
```

只跑 Milestone 32：

```bash
PYTHONPATH=src python3 -m unittest tests/mini_engine/test_paged_kv_allocator.py -v
```

最后跑 mini engine 全量测试：

当前全量验证结果：

```text
Ran 79 tests
OK
```

```bash
PYTHONPATH=src python3 -m unittest discover tests/mini_engine -v
```
