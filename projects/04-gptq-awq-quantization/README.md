# 04 GPTQ/AWQ Quantization

## Goal

基于开源量化工具对 LLaMA/Qwen 风格模型做 GPTQ/AWQ 量化，评估精度、显存和推理速度。

## Suggested Milestones

1. 选择一个小模型作为实验对象。
2. 准备 calibration dataset。
3. 运行 GPTQ 和 AWQ 量化。
4. 在 vLLM 或 SGLang 中加载量化模型。
5. 比较 perplexity、tokens/s、显存占用。

## Deliverables

- `quantize_gptq.py`
- `quantize_awq.py`
- `eval_perplexity.py`
- `benchmark_serving.py`
- `report.md`

## Interview Talking Points

- GPTQ 和 AWQ 的核心差异。
- calibration 数据如何影响量化质量。
- 为什么 4-bit 量化不一定总是更快？
- 如何判断精度损失是否可接受？
