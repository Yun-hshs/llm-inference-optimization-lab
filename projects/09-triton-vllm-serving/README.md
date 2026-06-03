# 09 Triton + vLLM Serving

## Goal

用 Triton 和 vLLM backend 搭建可压测的大模型服务化部署方案。

## Suggested Milestones

1. 编写 Triton model repository 配置。
2. 部署 vLLM backend。
3. 写并发请求压测脚本。
4. 采集 latency、throughput 和 GPU metrics。
5. 输出部署文档。

## Deliverables

- `model_repository/`
- `docker-compose.yml`
- `load_test.py`
- `metrics.md`
- `deployment-guide.md`

## Interview Talking Points

- Triton 的 model repository 和 backend 是什么？
- 在线服务中 batch size 如何影响 TTFT 和 throughput？
- 如何定位 serving 的瓶颈在模型、网络还是调度？
