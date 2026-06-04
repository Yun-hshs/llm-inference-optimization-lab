from __future__ import annotations
from dataclasses import dataclass

@dataclass
class GenerationRequest:
    request_id: str
    prompt: list[int]
    max_new_tokens: int
    eos_token_id: int | None = None

#处理生成请求的类，包含请求的ID、输入提示、最大生成长度和可选的结束标记ID
#工厂模式的类方法from_request用于从GenerationRequest对象创建ActiveRequest对象
#请求对象统一转换为ActiveRequest对象，方便后续处理和调度
@dataclass
class ActiveRequest:
    request: GenerationRequest
    generated_tokens: list[int]
    finished: bool = False

    @classmethod
    def from_request(cls, request: GenerationRequest) -> "ActiveRequest":
        return cls(request=request, 
                   generated_tokens=[], 
                   finished=False)
        
    def append_token(self, token_id: int) -> None:
        #添加token_id到生成的token列表中
        self.generated_tokens.append(token_id)
        #如果当前生成的token_id是eos_token_id，则标记请求为完成
        if self.request.eos_token_id is not None and token_id == self.request.eos_token_id:
            self.finished = True
        #如果生成的token数量已经达到max_new_tokens，也标记请求为完成
        elif len(self.generated_tokens) >= self.request.max_new_tokens:
            self.finished = True

    def output_tokens(self) -> list[int]:
        return self.request.prompt + self.generated_tokens

class RequestScheduler:
    
    def __init__(self, max_batch_size: int) -> None:
        if max_batch_size <= 0:
            raise ValueError("max_batch_size must be positive")
        self.max_batch_size = max_batch_size
        self.request_queue: list[GenerationRequest] = []
        self.active_requests: list[ActiveRequest] = []

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
    def active_count(self) -> int:
        return len(self.active_requests)

    def activate_next_batch(self) -> list[ActiveRequest]:
    #将下一个批次的请求从等待队列中取出并转换为ActiveRequest对象，加入到active_requests列表中
        batch = self.next_batch()
        active_batch = [ActiveRequest.from_request(req) for req in batch]
        self.active_requests.extend(active_batch)
        return active_batch
    
    def remove_finished_requests(self) -> list[ActiveRequest]:
    #从active_requests列表中移除已完成的请求，并返回这些完成的请求
        finished_requests = [req for req in self.active_requests if req.finished]
        self.active_requests = [req for req in self.active_requests if not req.finished]
        return finished_requests

    def refill_active_batch(self) -> list[ActiveRequest]:
    #清楚当前完成的请求后，尝试从等待队列中激活新的请求以填充active_requests列表
        self.remove_finished_requests()
        #先进先出，计算还有多少空位可以激活新的请求
        empty_slots = self.max_batch_size - len(self.active_requests)
        add_request = []

        for _ in range(empty_slots):
            if not self.request_queue:
                break
            req = self.request_queue.pop(0)
            active_req = ActiveRequest.from_request(req)    
            self.active_requests.append(active_req)
            add_request.append(active_req)
        return add_request
        