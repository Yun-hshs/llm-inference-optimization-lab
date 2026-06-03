# Roadmap

## Stage 0: Repository Foundation

- Create project structure.
- Add reproducible Python environment.
- Add CI, tests, and shared utility code.
- Write project README files with deliverables and interview talking points.

## Stage 1: Fast, Visible Output

Focus on projects that quickly produce measurable results:

1. `06-vllm-sglang-benchmark`
2. `04-gptq-awq-quantization`
3. `01-mini-inference-engine`

Expected output:

- benchmark tables
- latency/throughput charts
- memory usage comparison
- framework notes

## Stage 2: Kernel and Quantization Depth

Focus on low-level engineering:

1. `03-int4-int8-gemm`
2. `02-flash-attention-kernel`
3. `10-kv-cache-quantization`
4. `08-opencl-hip-kernels`

Expected output:

- baseline and optimized kernels
- profiling traces
- numerical error analysis
- memory bandwidth discussion

## Stage 3: Distributed and Serving Systems

Focus on production deployment:

1. `07-distributed-tp-inference`
2. `09-triton-vllm-serving`
3. `05-moe-router-optimization`

Expected output:

- multi-GPU inference demo
- service deployment guide
- communication bottleneck analysis
- MoE routing benchmark

## Stage 4: Open Source Contribution

Choose one upstream project:

- vLLM
- SGLang
- FlashInfer
- llama.cpp
- GPTQModel

Start with documentation, benchmark reproduction, or small bug fixes. Then move toward kernel, quantization, or serving improvements.
