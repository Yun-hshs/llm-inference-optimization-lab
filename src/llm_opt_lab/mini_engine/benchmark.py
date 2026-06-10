from __future__ import annotations

from dataclasses import dataclass

from llm_opt_lab.mini_engine.serving_loop import ServingLoopResult


BENCHMARK_REPORT_COLUMNS = [
    "case_name",
    "max_batch_size",
    "hidden_size",
    "bytes_per_element",
    "finished_request_count",
    "generated_token_count",
    "output_token_count",
    "waiting_count",
    "active_count",
    "active_kv_cache_memory_bytes",
    "remaining_kv_cache_budget_bytes",
    "max_kv_cache_memory_bytes",
    "stopped_reason",
]


@dataclass
class BenchmarkCase:
    name: str
    max_batch_size: int
    hidden_size: int
    bytes_per_element: int
    result: ServingLoopResult

    def record(self) -> dict[str, int | str | bool | None]:
        record = {
            "case_name": self.name,
            "max_batch_size": self.max_batch_size,
            "hidden_size": self.hidden_size,
            "bytes_per_element": self.bytes_per_element,
    }
        record.update(self.result.metrics_summary())
        return record


def format_benchmark_records_as_markdown(
    records: list[dict[str, int | str | bool | None]],
) -> str:
    header = "| " + " | ".join(BENCHMARK_REPORT_COLUMNS) + " |"
    separator = "| " + " | ".join("---" for _ in BENCHMARK_REPORT_COLUMNS) + " |"

    lines = [header, separator]

    for record in records:
        cells = []
        for column in BENCHMARK_REPORT_COLUMNS:
            value = record.get(column)
            if value is None:
                cells.append("")
            else:
                cells.append(str(value))
        lines.append("| " + " | ".join(cells) + " |")

    return "\n".join(lines)
