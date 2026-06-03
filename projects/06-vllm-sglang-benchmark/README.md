# 06 vLLM/SGLang Benchmark

## Goal

搭建统一 benchmark，对 vLLM、SGLang、llama.cpp 等推理框架做可复现实验。

## Suggested Milestones

1. 选择统一模型、prompt 集合和采样参数。
2. 测试不同 batch size、input length、output length。
3. 记录 TTFT、TPOT、throughput、显存。
4. 对比 FP16、INT8、INT4 模式。
5. 输出框架调研和实验报告。

## Deliverables

- `configs/`
- `run_vllm.py`
- `run_sglang.py`
- `run_llamacpp.py`
- `analyze_results.py`
- `report.md`

## Interview Talking Points

- TTFT 和 TPOT 分别代表什么？
- serving 框架如何做请求调度？
- 为什么高吞吐和低延迟经常互相冲突？
- vLLM 和 SGLang 的设计侧重点有什么不同？
