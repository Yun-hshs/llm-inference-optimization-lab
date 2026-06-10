# Step-by-Step Guide Archive

## Current Project Status

Project 1 已完成 32 个 Milestones。当前最终入口文档是：

- [README.md](./README.md)
- [Mini Inference Engine Report](./report.md)
- [Milestone 32: Paged KV Block Allocator](./milestone-32-paged-kv-allocator.md)

当前验证结果：mini engine 全量测试 `79` 个用例通过。

最终验证命令：

```bash
PYTHONPATH=src python3 -m unittest discover tests/mini_engine -v
```

下面内容保留为早期逐步复现记录，用于回看每个阶段的学习路径。

这个项目会按 TDD 方式复现一个教学版推理引擎。你手写核心代码，我负责准备文件、测试和每一步的验收方式。

## Current Step: Milestone 1.1

目标：实现单请求 greedy decoding。

你要实现的行为：

1. 输入 prompt token IDs，例如 `[1, 2, 3]`。
2. 调用模型的 `forward(tokens)` 得到 logits。
3. 取最后一个位置的 logits。
4. 选择分数最大的 token ID。
5. 把这个 token 追加到输出序列。
6. 返回完整 token 序列。

示例：

```python
prompt = [1, 2, 3]
model.forward(prompt)  # 最后一个位置最喜欢 token 7
generated = [1, 2, 3, 7]
```

## Files You Will Edit

第一步你只需要手写这个文件：

```text
src/llm_opt_lab/mini_engine/engine.py
```

## Command

每写完一点，运行：

```bash
PYTHONPATH=src python -m unittest tests/mini_engine/test_greedy_engine.py -v
```

如果你本机的命令不是 `python`，就用：

```bash
PYTHONPATH=src python3 -m unittest tests/mini_engine/test_greedy_engine.py -v
```

## Learning Notes

greedy decoding 是推理引擎最小的 decode loop。真实 LLM 中 logits 来自 Transformer forward，这里先用 toy model 固定 logits，让你先把推理流程写对。

## Next Step: Milestone 2.1

目标：实现 KV Cache 的最小数据结构。

这一步先不做真正 attention，只模拟真实推理框架里的缓存形状：

- 每一层都有一组 `keys`
- 每一层都有一组 `values`
- 每 decode 一个新 token，就向某一层追加一个 key 和 value
- 可以查询某一层当前缓存了多少个 token

你要手写这个文件：

```text
src/llm_opt_lab/mini_engine/kv_cache.py
```

测试命令：

```bash
PYTHONPATH=src python3 -m unittest tests/mini_engine/test_kv_cache.py -v
```

建议实现一个小 dataclass：

```python
LayerKVCache:
    keys: list[list[float]]
    values: list[list[float]]
```

然后 `KVCache(num_layers=2)` 内部维护两个 layer cache。

## Current Step: Milestone 3.1

目标：实现 Request Scheduler 的最小数据结构。

这一步先不做真正的并发推理，只实现等待队列和 batch 选择：

- 可以创建一个请求 `GenerationRequest`
- 可以把请求加入 `RequestScheduler`
- 可以查询等待队列长度
- 可以按 FIFO 顺序取出最多 `max_batch_size` 个请求

你要手写这个文件：

```text
src/llm_opt_lab/mini_engine/scheduler.py
```

测试命令：

```bash
PYTHONPATH=src python3 -m unittest tests/mini_engine/test_scheduler.py -v
```

建议实现：

```python
@dataclass
class GenerationRequest:
    request_id: str
    prompt: list[int]
    max_new_tokens: int
    eos_token_id: int | None = None


class RequestScheduler:
    def __init__(self, max_batch_size: int) -> None:
        ...

    def add_request(self, request: GenerationRequest) -> None:
        ...

    def waiting_count(self) -> int:
        ...

    def next_batch(self) -> list[GenerationRequest]:
        ...
```

## Current Step: Milestone 4.1

目标：实现 ActiveRequest，用来记录一个请求在 decode 过程中的状态。

现在 `GenerationRequest` 只描述“用户想要什么”。下一步我们需要 `ActiveRequest` 描述“这个请求已经生成到哪一步了”。

你要手写这个文件：

```text
src/llm_opt_lab/mini_engine/scheduler.py
```

测试命令：

```bash
PYTHONPATH=src python3 -m unittest tests/mini_engine/test_active_request.py -v
```

建议实现：

```python
@dataclass
class ActiveRequest:
    request: GenerationRequest
    generated_tokens: list[int]
    finished: bool = False

    @classmethod
    def from_request(cls, request: GenerationRequest) -> "ActiveRequest":
        ...

    def append_token(self, token_id: int) -> None:
        ...

    def output_tokens(self) -> list[int]:
        ...
```

行为要求：

- 初始 `generated_tokens` 为空
- `output_tokens()` 返回 `prompt + generated_tokens`
- 每次 `append_token()` 追加一个新 token
- 如果生成数量达到 `max_new_tokens`，设置 `finished=True`
- 如果生成 token 等于 `eos_token_id`，设置 `finished=True`

## Current Step: Milestone 5.1

目标：让 `RequestScheduler` 管理 active batch。

现在 scheduler 只能从等待队列取出 `GenerationRequest`。下一步要让它把等待请求转换成 `ActiveRequest`，表示这些请求已经进入 decode 阶段。

你要继续手写这个文件：

```text
src/llm_opt_lab/mini_engine/scheduler.py
```

测试命令：

```bash
PYTHONPATH=src python3 -m unittest tests/mini_engine/test_active_batch_scheduler.py -v
```

建议在 `RequestScheduler` 中新增：

```python
self.active_requests: list[ActiveRequest] = []
```

以及方法：

```python
def active_count(self) -> int:
    ...

def activate_next_batch(self) -> list[ActiveRequest]:
    ...
```

行为要求：

- 从等待队列取出最多 `max_batch_size` 个请求
- 把它们转换成 `ActiveRequest`
- 保存到 `self.active_requests`
- 返回这批 active requests
- 等待队列数量对应减少

## Current Step: Milestone 6.1

目标：实现 active batch refill。

continuous batching 的一个核心动作是：

1. 某些 active request 生成结束
2. scheduler 把这些 finished request 清理掉
3. 从 waiting queue 补入新请求
4. 让 active batch 尽可能保持满

你要继续手写这个文件：

```text
src/llm_opt_lab/mini_engine/scheduler.py
```

测试命令：

```bash
PYTHONPATH=src python3 -m unittest tests/mini_engine/test_scheduler_refill.py -v
```

建议新增方法：

```python
def remove_finished_requests(self) -> list[ActiveRequest]:
    ...

def refill_active_batch(self) -> list[ActiveRequest]:
    ...
```

行为要求：

- `remove_finished_requests()` 从 `active_requests` 中移除 finished request，并返回被移除的请求
- `refill_active_batch()` 先清理 finished request
- 然后根据剩余 active 数量，从等待队列补入新请求
- active request 数量不超过 `max_batch_size`

## Current Step: Milestone 10.1

目标：实现最小 serving loop。

现在 scheduler 已经能完成一次 continuous batching step。下一步要写一个外层循环，用可控的 token provider 模拟模型输出，一直运行到 scheduler idle。

你要手写这个文件：

```text
src/llm_opt_lab/mini_engine/serving_loop.py
```

测试命令：

```bash
PYTHONPATH=src python3 -m unittest tests/mini_engine/test_serving_loop.py -v
```

建议实现：

```python
def run_until_idle(self) -> list[ActiveRequest]:
    ...
```

行为要求：

- 如果没有 active requests，先调用 `scheduler.activate_next_batch()`
- 每轮根据当前 active requests 调用 `token_provider`
- 调用 `scheduler.step(tokens)`
- 收集每轮完成的 requests
- 直到 `scheduler.is_idle()`
- 返回所有完成的 active requests

## Current Step: Milestone 9.1

目标：实现 scheduler lifecycle 状态判断。

现在 scheduler 已经可以执行一次 decode scheduling tick。下一步需要让外层 serving loop 知道“系统里是否还有工作没完成”。

你要继续手写这个文件：

```text
src/llm_opt_lab/mini_engine/scheduler.py
```

测试命令：

```bash
PYTHONPATH=src python3 -m unittest tests/mini_engine/test_scheduler_lifecycle.py -v
```

建议新增方法：

```python
def has_work(self) -> bool:
    ...

def is_idle(self) -> bool:
    ...
```

行为要求：

- waiting queue 里有请求时，`has_work()` 返回 `True`
- active requests 里有请求时，`has_work()` 返回 `True`
- waiting 和 active 都为空时，`has_work()` 返回 `False`
- `is_idle()` 和 `has_work()` 相反

## Current Step: Milestone 7.1

目标：模拟一次 decode step 后的状态更新。

真实 LLM serving 中，每个 decode step 会给 active batch 中的每个请求生成一个 token。调度器随后要把这些 token 写回对应的 active request，并更新它们是否完成。

你要继续手写这个文件：

```text
src/llm_opt_lab/mini_engine/scheduler.py
```

测试命令：

```bash
PYTHONPATH=src python3 -m unittest tests/mini_engine/test_decode_step_scheduler.py -v
```

建议新增方法：

```python
def apply_tokens_to_active_requests(self, token_by_request_id: dict[str, int]) -> None:
    ...
```

行为要求：

- 遍历 `self.active_requests`
- 根据 `active.request.request_id` 找到对应 token
- 调用 `active.append_token(token_id)`
- 如果某个 active request 没有对应 token，抛出清晰错误
- token 追加后，`ActiveRequest` 自己负责判断是否 finished

## Current Step: Milestone 8.1

目标：实现一次完整 scheduler step。

这一步要把前面的动作串起来：

1. 对 active requests 应用本轮生成 token
2. 清理完成请求
3. 从 waiting queue 补齐 active batch 空位
4. 返回本轮完成的请求

你要继续手写这个文件：

```text
src/llm_opt_lab/mini_engine/scheduler.py
```

测试命令：

```bash
PYTHONPATH=src python3 -m unittest tests/mini_engine/test_scheduler_step.py -v
```

建议新增方法：

```python
def step(self, token_by_request_id: dict[str, int]) -> list[ActiveRequest]:
    ...
```

行为要求：

- `step()` 返回本轮完成的 active requests
- 完成请求从 `self.active_requests` 移除
- waiting queue 中的新请求补入空位
- active request 数量不超过 `max_batch_size`
