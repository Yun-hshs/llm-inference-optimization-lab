from __future__ import annotations
from dataclasses import dataclass

from llm_opt_lab.mini_engine.kv_cache import KVCache

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
    #将生成请求转换为活跃请求，包含生成的token列表和完成状态
    request: GenerationRequest
    generated_tokens: list[int]
    finished: bool = False
    kv_cache: KVCache | None = None
    #工厂方法，从GenerationRequest创建ActiveRequest，初始化生成的token列表为空，完成状态为False
    @classmethod
    def from_request(
        cls,
        request: GenerationRequest,
        *,
        num_kv_layers: int | None = None,
    ) -> "ActiveRequest":
        # TODO: Handwrite Milestone 15 core logic here.
        # If num_kv_layers is provided, create a KVCache for this active request.
        #在请求队列被激活的时候，如果提供了num_kv_layers参数，就为这个活跃请求创建一个KVCache对象，并将其赋值给kv_cache属性
        kv_cache = None
        if num_kv_layers is not None:
            kv_cache = KVCache(num_kv_layers)

        return cls(request=request, 
                   generated_tokens=[], 
                   finished=False,
                   kv_cache=kv_cache)
        
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
    
    def phase(self) -> str:
        if not self.generated_tokens:
            return "prefill"
        return "decode"

class RequestScheduler:
    
    def __init__(
        self,
        max_batch_size: int,
        num_kv_layers: int | None = None,
        max_kv_cache_memory_bytes: int | None = None,
    ) -> None:
        if max_batch_size <= 0:
            raise ValueError("max_batch_size must be positive")
        self.max_batch_size = max_batch_size
        self.num_kv_layers = num_kv_layers
        self.max_kv_cache_memory_bytes = max_kv_cache_memory_bytes
        #初始化请求队列和活跃请求列表
        self.request_queue: list[GenerationRequest] = []
        self.active_requests: list[ActiveRequest] = []

    def add_request(self, request: GenerationRequest) -> None:
        self.request_queue.append(request)

    def waiting_count(self) -> int:
        return len(self.request_queue)
    
    #批处理调度器的核心方法，负责从等待队列中取出请求组成批次，并将这些请求从等待队列中移除
    #从等待队列中取出下一个批次的请求，组成一个新的批次返回，并将这些请求从等待队列中移除
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
        active_batch = [
            ActiveRequest.from_request(req, num_kv_layers=self.num_kv_layers)
            for req in batch
        ]
        self.active_requests.extend(active_batch)
        return active_batch
    
    def remove_finished_requests(self) -> list[ActiveRequest]:
    #从active_requests列表中移除已完成的请求，并返回这些完成的请求
        finished_requests = [req for req in self.active_requests if req.finished]
        self._release_finished_request_resources(finished_requests)
        self.active_requests = [req for req in self.active_requests if not req.finished]
        return finished_requests

    def _release_finished_request_resources(self, finished_requests: list[ActiveRequest]) -> None:
        for active in finished_requests:
            if active.kv_cache is None:
                continue
            # TODO: Handwrite Milestone 16 core logic here.
            # Clear this finished request's KV cache before it leaves the active batch.
            active.kv_cache.clear()

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
            active_req = ActiveRequest.from_request(
                req,
                num_kv_layers=self.num_kv_layers,
            )    
            self.active_requests.append(active_req)
            add_request.append(active_req)
        return add_request

    def refill_active_batch_admissible(
        self,
        *,
        hidden_size: int,
        bytes_per_element: int,
    ) -> list[ActiveRequest]:
        # TODO: Handwrite Milestone 22 core logic here.
        # First remove finished requests, then fill open slots using
        self.remove_finished_requests()
        return self.activate_next_admissible_batch(
            hidden_size=hidden_size,
            bytes_per_element=bytes_per_element,
        )
        # activate_next_admissible_batch(...).

    def apply_tokens_to_active_requests(self, token_by_request_id: dict[str, int]) -> None:
        for active in self.active_requests:
            request_id = active.request.request_id

            if request_id not in token_by_request_id:
                raise KeyError(f"missing token for active request {request_id}")

            token_id = token_by_request_id[request_id]
            active.append_token(token_id)
    def step(self, token_by_request_id: dict[str, int]) -> list[ActiveRequest]:
        ''' 1. 给当前 active requests 应用本轮生成 token
            2. 清理 finished requests
            3. 从 waiting queue 补入新请求
            4. 返回本轮完成的请求'''
        self.apply_tokens_to_active_requests(token_by_request_id)
        finished = self.remove_finished_requests()
        self.refill_active_batch()
        return finished

    def step_admissible(
        self,
        token_by_request_id: dict[str, int],
        *,
        hidden_size: int,
        bytes_per_element: int,
    ) -> list[ActiveRequest]:
        # TODO: Handwrite Milestone 23 core logic here.
        # Apply tokens to active requests, collect finished requests, then refill
        self.apply_tokens_to_active_requests(token_by_request_id)
        finished = self.remove_finished_requests()
        self.refill_active_batch_admissible(
            hidden_size=hidden_size,
            bytes_per_element=bytes_per_element,
        )
        return finished
        # open slots using budget-aware admission.
    
    #建立生命周期
    def has_work(self) -> bool:
        return bool(self.request_queue or self.active_requests)

    def is_idle(self) -> bool:
        return not self.has_work()

    def is_blocked_by_kv_budget(
        self,
        *,
        hidden_size: int,
        bytes_per_element: int,
    ) -> bool:
        # TODO: Handwrite Milestone 25 core logic here.
        # Return True only when there are no active requests, the waiting queue
        if  self.active_requests :
            return False
        if not self.request_queue:
            return False
        # has a front request, and that front request cannot be admitted by the
        front_request = self.request_queue[0]
        return not self.can_admit_request(
            front_request,
            hidden_size=hidden_size,
            bytes_per_element=bytes_per_element,
        )
        # configured KV cache budget.

    
    #计算activate里面的k_v cache条目数
    def active_kv_cache_entries(self) -> int:
        total_entries = 0
        for active in self.active_requests:
            if active.kv_cache is None:
                continue
             # Assuming keys and values have the same length
            total_entries += active.kv_cache.total_entries()
        return total_entries
    
    #计算activate里面的k_v cache占用的内存大小
    def active_kv_cache_memory_bytes(
        self,
        *,
        hidden_size: int,
        bytes_per_element: int,
    ) -> int:
        total_memory = 0
        for active in self.active_requests:
            if active.kv_cache is None:
                continue
             # Assuming keys and values have the same length
            total_memory += active.kv_cache.estimate_memory_bytes(
                hidden_size = hidden_size, 
                bytes_per_element = bytes_per_element) # Multiply by 2 for keys and values
        return total_memory
    #计算当前活跃请求的KV缓存是否超过预算，如果没有配置预算则返回False，否则比较当前KV缓存的内存使用与配置的预算
    def is_kv_cache_over_budget(
        self,
        *,
        hidden_size: int,
        bytes_per_element: int,
    ) -> bool:
        # TODO: Handwrite Milestone 19 core logic here.
        # Return False when no max_kv_cache_memory_bytes budget is configured.
        if self.max_kv_cache_memory_bytes is None:
            return False
        # Otherwise compare active KV cache memory against the configured budget.
        current_memory = self.active_kv_cache_memory_bytes(
            hidden_size=hidden_size,
            bytes_per_element=bytes_per_element,
        )        
        if current_memory > self.max_kv_cache_memory_bytes:
            return True
        return False
    

    #计算剩余的KV缓存预算，如果没有配置预算则返回None，否则返回剩余预算，超过预算时返回0
    def remaining_kv_cache_budget_bytes(
        self,
        *,
        hidden_size: int,
        bytes_per_element: int,
    ) -> int | None:
        # TODO: Handwrite Milestone 19 core logic here.
        # Return None when the scheduler is unbounded.
        if self.max_kv_cache_memory_bytes is None:
            return None
        # Otherwise return remaining budget, clamped at 0 when usage exceeds budget.
        current_memory = self.active_kv_cache_memory_bytes(
            hidden_size=hidden_size,
            bytes_per_element=bytes_per_element,
        )
        remaining_budget = self.max_kv_cache_memory_bytes - current_memory
        return max(remaining_budget, 0)
    

    #估算请求需要的KV缓存内存，如果没有配置预算则返回0，否则根据请求的提示长度、KV层数、隐藏层大小和每个元素的字节数计算预估的KV缓存内存使用
    def estimate_request_prefill_kv_memory_bytes(
        self,
        request: GenerationRequest,
        *,
        hidden_size: int,
        bytes_per_element: int,
    ) -> int:
        # TODO: Handwrite Milestone 20 core logic here.
        # Estimate prefill KV memory for this request's prompt.
        if self.num_kv_layers is None:
            return 0
        
            
        # Formula: prompt_length * num_kv_layers * 2 * hidden_size * bytes_per_element.
        return len(request.prompt) * self.num_kv_layers * 2 * hidden_size * bytes_per_element

    def can_admit_request(
        self,
        request: GenerationRequest,
        *,
        hidden_size: int,
        bytes_per_element: int,
    ) -> bool:
        # TODO: Handwrite Milestone 20 core logic here.
        # Return True when no max_kv_cache_memory_bytes budget is configured.
        if self.max_kv_cache_memory_bytes is None:
            return True
        # Otherwise compare current active usage + projected request prefill usage
        current_memory = self.active_kv_cache_memory_bytes(
            hidden_size=hidden_size,
            bytes_per_element=bytes_per_element,
        )
        property_memory = self.estimate_request_prefill_kv_memory_bytes(
            request,
            hidden_size=hidden_size,
            bytes_per_element=bytes_per_element,
        )
        # with the configured budget.
        return current_memory + property_memory <= self.max_kv_cache_memory_bytes

    def activate_next_admissible_batch(
        self,
        *,
        hidden_size: int,
        bytes_per_element: int,
    ) -> list[ActiveRequest]:
        # 循环检查等待队列中的请求，按照FIFO顺序激活满足KV缓存预算的请求，直到当前批次满员或者等待队列空了或者下一个请求不满足预算为止
        # TODO: Handwrite Milestone 21 core logic here.
        # Activate waiting requests in FIFO order while:
        activate_requests = []
        while self.request_queue and len(self.active_requests) < self.max_batch_size:
            next_request = self.request_queue[0]
            if not self.can_admit_request(
                next_request,
                hidden_size=hidden_size,
                bytes_per_element=bytes_per_element,
            ):
                break
            self.request_queue.pop(0)
            active_req = ActiveRequest.from_request(
                next_request,
                num_kv_layers=self.num_kv_layers,
            )    
            self.active_requests.append(active_req)
            activate_requests.append(active_req)
        return activate_requests
        # - active batch has free slots
        # - waiting queue is not empty
        # - the front request can be admitted under the KV cache budget
        #
        # Important: do not skip a rejected front request to activate later requests.
