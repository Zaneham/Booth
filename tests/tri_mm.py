"""Minimal matmul-shaped kernel for shape-inference testing.
Not a real matmul; we just want the AST to exercise rank-1 tiles
plus the [:, None] / [None, :] broadcast pattern that produces
rank-2 tiles. Lowering does not need to succeed; sema does."""
import triton
import triton.language as tl

@triton.jit
def mm_shape(a_ptr, c_ptr,
             M, N, K,
             stride_am, stride_ak,
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
