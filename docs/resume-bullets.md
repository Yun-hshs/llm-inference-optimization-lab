# Resume Bullets

Use these as raw material. Replace numbers with measured results from your own benchmark.

## Inference Engine

- Implemented a lightweight LLM inference engine with KV cache management, request scheduling, and continuous batching for Transformer decoder models.
- Built benchmark scripts to measure tokens/s, latency, and memory usage under different batch sizes and context lengths.

## Attention Optimization

- Reimplemented baseline attention and tiled attention kernels, then compared memory footprint and latency against FlashAttention-style computation.
- Analyzed GPU memory access patterns and reduced intermediate activation storage during attention computation.

## Quantization

- Developed INT4/INT8 weight-only quantization utilities including packing, dequantization, and error evaluation.
- Evaluated GPTQ/AWQ quantization on LLaMA/Qwen-style models and compared perplexity, throughput, and GPU memory usage.

## Distributed Inference

- Implemented Tensor Parallel inference for Transformer linear layers and profiled communication overhead across multi-GPU execution.
- Designed benchmark cases to analyze bandwidth bottlenecks in distributed LLM serving.

## Serving

- Deployed LLM inference service with Triton and vLLM backend, adding throughput, latency, and concurrency evaluation.
- Built reproducible Docker and benchmark workflows for model serving experiments.
