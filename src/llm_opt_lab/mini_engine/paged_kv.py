from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PagedKVBlock:
    block_id: int
    owner_request_id: str | None = None

    def is_free(self) -> bool:
        return self.owner_request_id is None


class PagedKVBlockAllocator:
    def __init__(self, num_blocks: int) -> None:
        if num_blocks <= 0:
            raise ValueError("num_blocks must be positive")

        self.blocks = [PagedKVBlock(block_id=i) for i in range(num_blocks)]
        self.block_tables: dict[str, list[int]] = {}

    def free_block_count(self) -> int:
        # TODO: Handwrite Milestone 32 core logic here.
        # 遍历blocks观察是否有free
        count = 0
        for block in self.blocks:
            if block.is_free():
                count += 1
        return count

    def allocated_block_count(self) -> int:
        # TODO: Handwrite Milestone 32 core logic here.
        count = 0
        for block in self.blocks:
            if not block.is_free():
                count += 1
        return count

    def allocate(self, request_id: str, num_blocks: int) -> list[int]:
        # TODO: Handwrite Milestone 32 core logic here.
        # Allocate free blocks to request_id and return allocated block IDs.
        # Raise ValueError for duplicate request_id or non-positive num_blocks.
        # Raise MemoryError when there are not enough free blocks.
        if num_blocks <= 0:
            raise ValueError("num_blocks must be positive")
        if request_id in self.block_tables:
            raise ValueError(f"request {request_id} already has allocated blocks")
        if self.free_block_count() < num_blocks:
            raise MemoryError("not enough free KV blocks")

        allocated: list[int] = []
        for block in self.blocks:
            if block.is_free():
                block.owner_request_id = request_id
                allocated.append(block.block_id)
                if len(allocated) == num_blocks:
                    break

        self.block_tables[request_id] = allocated
        return allocated

    def free(self, request_id: str) -> list[int]:
        # TODO: Handwrite Milestone 32 core logic here.
        # Free all blocks owned by request_id and return released block IDs.
        # Raise KeyError when request_id has no block table.
        if request_id not in self.block_tables:
            raise KeyError(f"request {request_id} has no allocated blocks")

        released = self.block_tables.pop(request_id)
        for block_id in released:
            self.blocks[block_id].owner_request_id = None
        return released

    def block_table(self, request_id: str) -> list[int]:
        # TODO: Handwrite Milestone 32 core logic here.
        # Return a copy of the request block table.
        # Raise KeyError when request_id has no block table.
        if request_id not in self.block_tables:
            raise KeyError(f"request {request_id} has no allocated blocks")
        return list(self.block_tables[request_id])
