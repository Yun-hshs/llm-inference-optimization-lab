from __future__ import annotations
from dataclasses import dataclass

@dataclass
class GenerationRequest:
    request_id: str
    prompt: list[int]
    max_new_tokens: int
    eos_token_id: int | None = None


class RequestScheduler:
    def __init__(self, max_batch_size: int) -> None:
        if max_batch_size <= 0:
            raise ValueError("max_batch_size must be positive")
        self.max_batch_size = max_batch_size
        self.request_queue: list[GenerationRequest] = []

    def add_request(self, request: GenerationRequest) -> None:
        self.request_queue.append(request)

    def waiting_count(self) -> int:
        return len(self.request_queue)

    def next_batch(self) -> list[GenerationRequest]:
        if not self.request_queue:
            return []
        # 从队列中取出最多max_batch_size个请求组成一个批次
        batch = self.request_queue[:self.max_batch_size]
        # 将这些请求从队列中移除
        self.request_queue = self.request_queue[self.max_batch_size:]
        return batch