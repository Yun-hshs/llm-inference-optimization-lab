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

           
