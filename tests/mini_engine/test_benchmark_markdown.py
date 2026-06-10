from __future__ import annotations

import unittest

from llm_opt_lab.mini_engine.benchmark import format_benchmark_records_as_markdown


class BenchmarkMarkdownTest(unittest.TestCase):
    def test_formats_benchmark_records_as_stable_markdown_table(self) -> None:
        records = [
            {
                "case_name": "batch-1",
                "max_batch_size": 1,
                "hidden_size": 4,
                "bytes_per_element": 2,
                "finished_request_count": 1,
                "generated_token_count": 2,
                "output_token_count": 4,
                "waiting_count": 0,
                "active_count": 0,
                "active_kv_cache_entries": 0,
                "active_kv_cache_memory_bytes": 0,
                "remaining_kv_cache_budget_bytes": 128,
                "max_kv_cache_memory_bytes": 128,
                "stopped_reason": "idle",
            },
            {
                "case_name": "blocked",
                "max_batch_size": 1,
                "hidden_size": 4,
                "bytes_per_element": 2,
                "finished_request_count": 0,
                "generated_token_count": 0,
                "output_token_count": 0,
                "waiting_count": 1,
                "active_count": 0,
                "active_kv_cache_entries": 0,
                "active_kv_cache_memory_bytes": 0,
                "remaining_kv_cache_budget_bytes": 15,
                "max_kv_cache_memory_bytes": 15,
                "stopped_reason": "blocked_by_kv_budget",
            },
        ]

        self.assertEqual(
            format_benchmark_records_as_markdown(records),
            "\n".join(
                [
                    "| case_name | max_batch_size | hidden_size | bytes_per_element | finished_request_count | generated_token_count | output_token_count | waiting_count | active_count | active_kv_cache_memory_bytes | remaining_kv_cache_budget_bytes | max_kv_cache_memory_bytes | stopped_reason |",
                    "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
                    "| batch-1 | 1 | 4 | 2 | 1 | 2 | 4 | 0 | 0 | 0 | 128 | 128 | idle |",
                    "| blocked | 1 | 4 | 2 | 0 | 0 | 0 | 1 | 0 | 0 | 15 | 15 | blocked_by_kv_budget |",
                ]
            ),
        )

    def test_formats_none_values_as_empty_cells(self) -> None:
        records = [
            {
                "case_name": "unbounded",
                "max_batch_size": 2,
                "hidden_size": 4,
                "bytes_per_element": 2,
                "finished_request_count": 1,
                "generated_token_count": 1,
                "output_token_count": 2,
                "waiting_count": 0,
                "active_count": 0,
                "active_kv_cache_memory_bytes": 0,
                "remaining_kv_cache_budget_bytes": None,
                "max_kv_cache_memory_bytes": None,
                "stopped_reason": "idle",
            }
        ]

        self.assertEqual(
            format_benchmark_records_as_markdown(records),
            "\n".join(
                [
                    "| case_name | max_batch_size | hidden_size | bytes_per_element | finished_request_count | generated_token_count | output_token_count | waiting_count | active_count | active_kv_cache_memory_bytes | remaining_kv_cache_budget_bytes | max_kv_cache_memory_bytes | stopped_reason |",
                    "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
                    "| unbounded | 2 | 4 | 2 | 1 | 1 | 2 | 0 | 0 | 0 |  |  | idle |",
                ]
            ),
        )

    def test_formats_empty_records_as_header_only_table(self) -> None:
        self.assertEqual(
            format_benchmark_records_as_markdown([]),
            "\n".join(
                [
                    "| case_name | max_batch_size | hidden_size | bytes_per_element | finished_request_count | generated_token_count | output_token_count | waiting_count | active_count | active_kv_cache_memory_bytes | remaining_kv_cache_budget_bytes | max_kv_cache_memory_bytes | stopped_reason |",
                    "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
                ]
            ),
        )


if __name__ == "__main__":
    unittest.main()
