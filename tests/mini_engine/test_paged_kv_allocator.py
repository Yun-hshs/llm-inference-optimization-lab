from __future__ import annotations

import unittest

from llm_opt_lab.mini_engine.paged_kv import PagedKVBlockAllocator


class PagedKVBlockAllocatorTest(unittest.TestCase):
    def test_allocate_assigns_free_blocks_and_records_block_table(self) -> None:
        allocator = PagedKVBlockAllocator(num_blocks=4)

        block_ids = allocator.allocate("req-1", num_blocks=2)

        self.assertEqual(block_ids, [0, 1])
        self.assertEqual(allocator.block_table("req-1"), [0, 1])
        self.assertEqual(allocator.free_block_count(), 2)
        self.assertEqual(allocator.allocated_block_count(), 2)
        self.assertEqual(allocator.blocks[0].owner_request_id, "req-1")
        self.assertEqual(allocator.blocks[1].owner_request_id, "req-1")

    def test_free_releases_blocks_and_makes_them_reusable(self) -> None:
        allocator = PagedKVBlockAllocator(num_blocks=3)
        allocator.allocate("req-1", num_blocks=2)

        released = allocator.free("req-1")
        reused = allocator.allocate("req-2", num_blocks=2)

        self.assertEqual(released, [0, 1])
        self.assertEqual(reused, [0, 1])
        self.assertEqual(allocator.block_table("req-2"), [0, 1])
        self.assertEqual(allocator.free_block_count(), 1)

    def test_allocate_raises_when_not_enough_free_blocks_and_keeps_state_unchanged(self) -> None:
        allocator = PagedKVBlockAllocator(num_blocks=2)
        allocator.allocate("req-1", num_blocks=1)

        with self.assertRaisesRegex(MemoryError, "not enough free KV blocks"):
            allocator.allocate("req-2", num_blocks=2)

        self.assertEqual(allocator.block_table("req-1"), [0])
        self.assertEqual(allocator.free_block_count(), 1)
        self.assertEqual(allocator.allocated_block_count(), 1)

    def test_rejects_duplicate_request_id(self) -> None:
        allocator = PagedKVBlockAllocator(num_blocks=2)
        allocator.allocate("req-1", num_blocks=1)

        with self.assertRaisesRegex(ValueError, "request req-1 already has allocated blocks"):
            allocator.allocate("req-1", num_blocks=1)

    def test_rejects_unknown_request_on_free_and_block_table(self) -> None:
        allocator = PagedKVBlockAllocator(num_blocks=2)

        with self.assertRaisesRegex(KeyError, "request missing has no allocated blocks"):
            allocator.free("missing")

        with self.assertRaisesRegex(KeyError, "request missing has no allocated blocks"):
            allocator.block_table("missing")

    def test_rejects_non_positive_sizes(self) -> None:
        with self.assertRaisesRegex(ValueError, "num_blocks must be positive"):
            PagedKVBlockAllocator(num_blocks=0)

        allocator = PagedKVBlockAllocator(num_blocks=2)

        with self.assertRaisesRegex(ValueError, "num_blocks must be positive"):
            allocator.allocate("req-1", num_blocks=0)


if __name__ == "__main__":
    unittest.main()
