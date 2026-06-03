# 03 INT4/INT8 GEMM

## Goal

实现 weight-only INT4/INT8 量化矩阵乘，包括量化、pack/unpack、dequant 和 GEMM benchmark。

## Suggested Milestones

1. 实现 symmetric INT8 量化和反量化。
2. 实现 INT4 packed storage。
3. 实现 CPU baseline GEMM。
4. 实现 CUDA/HIP/OpenCL kernel。
5. 比较 FP16、INT8、INT4 的速度、误差和内存占用。

## Deliverables

- `quantize.py`
- `pack_int4.py`
- `gemm_cpu.py`
- `kernels/`
- `benchmark.py`
- `report.md`

## Interview Talking Points

- weight-only quantization 和 activation quantization 的区别。
- INT4 pack 后如何减少内存带宽压力。
- scale 粒度为什么会影响精度和速度。
- kernel 中 dequant 放在哪里更合适。
