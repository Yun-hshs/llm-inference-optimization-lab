# 02 FlashAttention Kernel

## Goal

复现 attention 的 baseline 和 tiled 版本，理解 FlashAttention 通过分块计算降低 HBM 访问和中间显存占用的思路。

## Suggested Milestones

1. 用 NumPy 或 PyTorch 实现 naive attention。
2. 实现 block-wise attention，验证数值一致性。
3. 用 Triton 或 CUDA 实现 tiled attention kernel。
4. 比较 sequence length 从 512 到 8192 的显存和延迟。
5. 写一篇 FlashAttention 原理笔记。

## Deliverables

- `attention_baseline.py`
- `attention_tiled.py`
- `kernel_triton.py` 或 `kernel_cuda/`
- `benchmark.py`
- `report.md`

## Interview Talking Points

- 标准 attention 的显存瓶颈在哪里？
- online softmax 如何避免存完整 attention matrix？
- block size 如何影响 occupancy、shared memory 和吞吐？
