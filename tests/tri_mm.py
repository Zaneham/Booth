"""Matmul-shape kernel for shape-inference testing. Exercises
rank-1 tiles from arange plus [:, None] / [None, :] broadcasts
into rank-2 tiles. Sema clean; lowering is a later sitting."""
import triton
import triton.language as tl

@triton.jit
def mm_shape(a_ptr, c_ptr,
             M, N, K,
             stride_am, stride_ak,
             stride_cm, stride_cn,
             BLOCK_M: tl.constexpr,
             BLOCK_N: tl.constexpr,
             BLOCK_K: tl.constexpr):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, acc)
