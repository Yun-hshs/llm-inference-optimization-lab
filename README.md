# LLM Inference Optimization Lab

面向大模型推理、算子优化、模型量化和分布式部署的开源学习与实践仓库。

这个仓库的目标不是堆 Demo，而是围绕一个真实推理优化岗位，逐步补齐可以写进简历、可以在面试中展开讲、也可以继续向开源社区贡献的工程项目。

## Why This Repo

目标岗位关注：

- vLLM / SGLang 等前沿推理框架调研
- Transformer、MoE、FlashAttention 等算法与性能优化
- GPTQ / AWQ / INT4 / INT8 量化算子开发
- CUDA / HIP / OpenCL GPU 并行计算
- 多 GPU / 多节点分布式推理
- Triton、ONNX Runtime 等模型服务化落地
- 性能评估、技术文档、论文或专利输出

本仓库把这些能力拆成 10 个可独立推进的子项目。

## Project Map

| ID | Project | Focus |
| --- | --- | --- |
| 01 | Mini Inference Engine | LLaMA/Qwen 风格推理、KV cache、continuous batching、Paged KV |
| 02 | FlashAttention Kernel | naive attention、tiled attention、FlashAttention 复现与 benchmark |
| 03 | INT4/INT8 GEMM | weight-only quantization、pack/unpack、dequant、GEMM kernel |
| 04 | GPTQ/AWQ Quantization | 量化校准、精度评估、速度/显存对比 |
| 05 | MoE Router Optimization | top-k routing、expert batching、负载均衡 |
| 06 | vLLM/SGLang Benchmark | 框架调研、统一压测、吞吐/延迟分析 |
| 07 | Distributed TP Inference | Tensor Parallel、通信耗时、NCCL/RCCL profiling |
| 08 | OpenCL/HIP Kernels | 异构 GPU matmul、tiling、vectorized load |
| 09 | Triton + vLLM Serving | Triton backend、服务化、metrics、压测 |
| 10 | KV Cache Quantization | FP16/INT8/INT4 KV cache、长上下文显存优化 |

## Repository Layout

```text
.
├── docs/
│   ├── roadmap.md
│   ├── resume-bullets.md
│   └── project-template.md
├── projects/
│   ├── 01-mini-inference-engine/
│   ├── 02-flash-attention-kernel/
│   ├── 03-int4-int8-gemm/
│   ├── 04-gptq-awq-quantization/
│   ├── 05-moe-router-optimization/
│   ├── 06-vllm-sglang-benchmark/
│   ├── 07-distributed-tp-inference/
│   ├── 08-opencl-hip-kernels/
│   ├── 09-triton-vllm-serving/
│   └── 10-kv-cache-quantization/
├── scripts/
├── src/llm_opt_lab/
└── tests/
```

## Quick Start

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e ".[dev]"
python scripts/check_environment.py
python -m unittest discover -s tests
```

## First Milestone

建议优先完成前三个项目，它们最贴合大模型推理优化面试：

1. `01-mini-inference-engine`: 建立推理系统全局理解。
2. `03-int4-int8-gemm`: 展示量化和底层算子能力。
3. `06-vllm-sglang-benchmark`: 快速产出可读报告和性能数据。

## Benchmark Philosophy

每个项目都应至少包含：

- baseline 实现
- optimized 实现
- benchmark 脚本
- 结果表格
- profiling 记录
- 技术总结
- 简历描述

## License

MIT
