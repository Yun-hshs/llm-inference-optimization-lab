from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from llm_opt_lab.mini_engine.scheduler import ActiveRequest, RequestScheduler


@dataclass
class ServingLoopResult:
    finished_requests: list[ActiveRequest]
    final_status: dict[str, int | bool | None]

    def metrics_summary(self) -> dict[str, int | str | bool | None]:
        # TODO: Handwrite Milestone 28 core logic here.
        # Summarize finished request count, generated/output token counts,
        if self.final_status["is_idle"]:
            stop_reason = "idle"
        elif self.final_status["is_blocked_by_kv_budget"]:
            stop_reason = "blocked_by_kv_budget"
        else:
            stop_reason = "running"
        # final queue state, KV memory usage, and stopped reason.
        return {
            "finished_request_count": len(self.finished_requests),
            "generated_token_count": sum(len(active.generated_tokens) for active in self.finished_requests),
            "output_token_count": sum(len(active.output_tokens()) for active in self.finished_requests),
            "waiting_count": self.final_status["waiting_count"],
            "active_count": self.final_status["active_count"],
            "active_kv_cache_entries": self.final_status["active_kv_cache_entries"],
            "active_kv_cache_memory_bytes": self.final_status["active_kv_cache_memory_bytes"],
            "remaining_kv_cache_budget_bytes": self.final_status["remaining_kv_cache_budget_bytes"],
            "max_kv_cache_memory_bytes": self.final_status["max_kv_cache_memory_bytes"],
            "stopped_reason": stop_reason
        }

class TokenProvider(Protocol):
    def __call__(self, active_requests: list[ActiveRequest]) -> dict[str, int]:
        """Return one generated token for each active request ID."""



class ServingLoop:
    def __init__(self, scheduler: RequestScheduler, token_provider: TokenProvider) -> None:
        self.scheduler = scheduler
        self.token_provider = token_provider

    def run_until_idle(self) -> list[ActiveRequest]:
        #如果活跃请求为空，则申请next_batch
        if not self.scheduler.active_requests:
            self.scheduler.activate_next_batch()

        # 创建一个list来存储完成的请求
        finished_requests: list[ActiveRequest] = []
        # 循环直到所有请求都完成
        # 否则进入循环，直到所有请求都完成，生命活跃请求列表不为空
        while self.scheduler.has_work():
            # 调用token_provider，获取每个活跃请求的下一个生成token
            token_dict = self.token_provider(self.scheduler.active_requests)

            finished = self.scheduler.step(token_dict)
            finished_requests.extend(finished)
        # 最后返回所有完成的请求
        return finished_requests

    def run_until_blocked_or_idle_admissible(
        self,
        *,
        hidden_size: int,
        bytes_per_element: int,
    ) -> list[ActiveRequest]:
        # TODO: Handwrite Milestone 24 core logic here.
        # Use budget-aware activation/step methods. Stop when there are no active
        if not self.scheduler.active_requests:
            self.scheduler.activate_next_admissible_batch(
                hidden_size=hidden_size,
                bytes_per_element=bytes_per_element,
            )
        finished_requests: list[ActiveRequest] = []
        #不能使用hsawork，因为可能存在请求被budget阻塞的情况，所以需要使用has_admissible_work来判断是否有可执行的请求
        while self.scheduler.active_requests:
            token_dict = self.token_provider(self.scheduler.active_requests)
            finished = self.scheduler.step_admissible(
                token_by_request_id=token_dict,
                hidden_size=hidden_size,
                bytes_per_element=bytes_per_element,
            )
            finished_requests.extend(finished)
        # requests, because remaining waiting requests may be blocked by budget.
        return finished_requests

    def run_until_blocked_or_idle_admissible_with_status(
        self,
        *,
        hidden_size: int,
        bytes_per_element: int,
    ) -> ServingLoopResult:
        # TODO: Handwrite Milestone 27 core logic here.
        # Run the budget-aware serving loop, then return both finished requests
        finshed_requests = self.run_until_blocked_or_idle_admissible(
            hidden_size=hidden_size,
            bytes_per_element=bytes_per_element,
        )
        # and the final scheduler status snapshot.
        final_status = self.scheduler.status(
            hidden_size=hidden_size,
            bytes_per_element=bytes_per_element,
        )
        return ServingLoopResult(finished_requests=finshed_requests, final_status=final_status)

           
