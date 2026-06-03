# 08 OpenCL/HIP Kernels

## Goal

基于 OpenCL 或 HIP 实现矩阵乘和简单量化算子，面向异构 GPU 或非 NVIDIA GPU 做算子优化。

## Suggested Milestones

1. 实现 naive matmul kernel。
2. 加入 tiling。
3. 加入 vectorized load。
4. 加入 shared/local memory 优化。
5. 比较不同 tile size 的性能。

## Deliverables

- `matmul_naive.cl` 或 `matmul_naive.hip`
- `matmul_tiled.cl` 或 `matmul_tiled.hip`
- `host.cpp`
- `benchmark.py`
- `report.md`

## Interview Talking Points

- OpenCL/HIP 和 CUDA 编程模型的相同点与差异。
- work-group、local memory、global memory 如何影响性能？
- 如何把 NVIDIA 上的优化思路迁移到其他 GPU？
