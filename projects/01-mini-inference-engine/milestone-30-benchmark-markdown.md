# Milestone 30: Benchmark Markdown Table

## Goal

把多条 benchmark record 渲染成稳定的 Markdown 表格，作为实验报告输出的第一版。

本阶段不跑模型、不做计时、不写文件，只把已经生成的 records 格式化成可复制到 report 的表格文本。

## Why It Matters

Milestone 29 已经能把一次实验变成结构化 record。

真实推理优化项目不只要有代码，还要能展示实验结果。Markdown table 是最轻量的报告格式，可以直接放到 README、技术文档或实验报告里。

这个阶段把 benchmark 数据从“程序内部 dict”推进到“人能读的结果表”。

## Key Design Boundary

`format_benchmark_records_as_markdown(...)` 只做格式化：

- 不执行 serving loop
- 不创建 benchmark case
- 不修改 records
- 不写文件
- 不推导新指标

它只按固定列顺序输出 Markdown 表格。

## Target API

新增：

```python
format_benchmark_records_as_markdown(records)
```

使用方式：

```python
markdown = format_benchmark_records_as_markdown(records)
```

输出示例：

```markdown
| case_name | max_batch_size | hidden_size | bytes_per_element | finished_request_count | generated_token_count | output_token_count | waiting_count | active_count | active_kv_cache_memory_bytes | remaining_kv_cache_budget_bytes | max_kv_cache_memory_bytes | stopped_reason |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| batch-1 | 1 | 4 | 2 | 1 | 2 | 4 | 0 | 0 | 0 | 128 | 128 | idle |
```

## Target Data Flow

```text
records
      |
      v
fixed column order
      |
      v
header row + separator row + data rows
      |
      v
Markdown table string
```

## RED Test

新增测试文件：

```bash
tests/mini_engine/test_benchmark_markdown.py
```

测试内容：

- 多条 records 可以输出稳定 Markdown 表格
- `None` budget 字段输出为空单元格
- 空 records 仍然输出 header-only table

## Code Framework

我已经在：

```bash
src/llm_opt_lab/mini_engine/benchmark.py
```

搭好了：

```python
BENCHMARK_REPORT_COLUMNS = [...]

def format_benchmark_records_as_markdown(...):
    ...
```

你只需要补充核心格式化逻辑。

## Your Implementation Task

实现思路：

- 用 `BENCHMARK_REPORT_COLUMNS` 生成 header 行
- 生成 separator 行
- 遍历 records，按列顺序取值
- 如果值是 `None`，输出空字符串
- 其他值用 `str(value)`
- 用 `"\n".join(lines)` 返回

注意：不要使用 dict 的自然遍历顺序，要用固定列顺序，保证报告稳定。

## Validation

只跑 Milestone 30：

```bash
PYTHONPATH=src python3 -m unittest tests/mini_engine/test_benchmark_markdown.py -v
```

实现后建议一起跑 Milestone 29 和 30：

```bash
PYTHONPATH=src python3 -m unittest \
  tests/mini_engine/test_benchmark_record.py \
  tests/mini_engine/test_benchmark_markdown.py \
  -v
```

最后再跑 mini engine 全量测试：

```bash
PYTHONPATH=src python3 -m unittest discover tests/mini_engine -v
```
