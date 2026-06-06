from __future__ import annotations

from typing import Protocol

from llm_opt_lab.mini_engine.scheduler import ActiveRequest, RequestScheduler


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

           
