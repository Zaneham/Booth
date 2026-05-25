"""Matmul with a runtime K-loop: fixed BLOCK_M x BLOCK_N output tile,
contraction dimension K swept BLOCK_K at a time. Addresses are recomputed
from the loop variable each iteration; acc lives across iterations."""
import triton
import triton.language as tl

@triton.jit
def matmul_k(a_ptr, b_ptr, c_ptr, K,
             stride_am, stride_ak, stride_bk, stride_bn,
             stride_cm, stride_cn,
             BLOCK_M: tl.constexpr = 4,
             BLOCK_N: tl.constexpr = 4,
             BLOCK_K: tl.constexpr = 2):
    offs_m = tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        a_ptrs = a_ptr + offs_m[:, None] * stride_am + (k + offs_k[None, :]) * stride_ak
        b_ptrs = b_ptr + (k + offs_k[:, None]) * stride_bk + offs_n[None, :] * stride_bn
        acc += tl.dot(tl.load(a_ptrs), tl.load(b_ptrs))
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, acc)
