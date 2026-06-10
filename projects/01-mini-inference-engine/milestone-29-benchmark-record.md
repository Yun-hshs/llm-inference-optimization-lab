# Milestone 29: Benchmark Record

## Goal

新增一个轻量级 `BenchmarkCase`，把一次 serving loop 的结果和实验配置合并成稳定的 benchmark record。

本阶段不做真实计时，也不跑模型。它只解决“如何把一次实验结果记录下来”的问题。

## Why It Matters

前面 Milestone 已经可以得到：

- serving loop 的 finished requests
- scheduler final status
- metrics summary

但 benchmark/report 需要的不只是 metrics，还需要实验配置，例如：

- case name
- max batch size
- hidden size
- bytes per element

Milestone 29 把配置和结果合并成一条结构化 record，为后续 markdown table、CSV、JSON report 做准备。

## Key Design Boundary

`BenchmarkCase.record()` 只做数据合并：

- 不执行 serving loop
- 不调用 token provider
- 不修改 result
- 不做真实 wall-clock timing

它应该复用 `ServingLoopResult.metrics_summary()`。

## Target API

新增文件：

```bash
src/llm_opt_lab/mini_engine/benchmark.py
```

新增类型：

```python
@dataclass
class BenchmarkCase:
    name: str
    max_batch_size: int
    hidden_size: int
    bytes_per_element: int
    result: ServingLoopResult

    def record(self) -> dict[str, int | str | bool | None]:
        ...
```

使用方式：

```python
benchmark_case = BenchmarkCase(
    name="batch-1-budget-128",
    max_batch_size=1,
    hidden_size=4096,
    bytes_per_element=2,
    result=result,
)

record = benchmark_case.record()
```

## Target Data Flow

```text
BenchmarkCase config
      |
      v
case_name / max_batch_size / hidden_size / bytes_per_element

ServingLoopResult
      |
      v
metrics_summary()

config + metrics
      |
      v
benchmark record
```

## RED Test

新增测试文件：

```bash
tests/mini_engine/test_benchmark_record.py
```

测试内容：

- record 合并 case config 和 serving result metrics
- blocked case 可以保留 `blocked_by_kv_budget` stop reason

## Code Framework

我已经在：

```bash
src/llm_opt_lab/mini_engine/benchmark.py
```

搭好了：

```python
@dataclass
class BenchmarkCase:
    ...

    def record(...):
        ...
```

你只需要补充核心合并逻辑。

## Your Implementation Task

实现思路：

- 先创建配置部分 dict：

```python
record = {
    "case_name": self.name,
    "max_batch_size": self.max_batch_size,
    "hidden_size": self.hidden_size,
    "bytes_per_element": self.bytes_per_element,
}
```

- 再调用：

```python
record.update(self.result.metrics_summary())
```

- 最后返回 `record`

## Validation

只跑 Milestone 29：

```bash
PYTHONPATH=src python3 -m unittest tests/mini_engine/test_benchmark_record.py -v
```

实现后建议一起跑 Milestone 28 和 29：

```bash
PYTHONPATH=src python3 -m unittest \
  tests/mini_engine/test_serving_loop_metrics.py \
  tests/mini_engine/test_benchmark_record.py \
  -v
```

最后再跑 mini engine 全量测试：

```bash
PYTHONPATH=src python3 -m unittest discover tests/mini_engine -v
```
