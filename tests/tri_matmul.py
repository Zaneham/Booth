"""Single-tile matmul: C = A @ B for one BLOCK_M x BLOCK_N output tile.
No grid offset, no K-loop tiling (K == BLOCK_K), no masking. Concrete
constexpr block sizes so dims resolve statically for AOT scratch sizing.
The simplest real tl.dot kernel, for bringing up rank-2 CPU lowering."""
import triton
import triton.language as tl

@triton.jit
def matmul(a_ptr, b_ptr, c_ptr,
           stride_am, stride_ak,
           stride_bk, stride_bn,
           stride_cm, stride_cn,
           BLOCK_M: tl.constexpr = 4,
           BLOCK_N: tl.constexpr = 4,
           BLOCK_K: tl.constexpr = 4):
    offs_m = tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
    a = tl.load(a_ptrs)
    b = tl.load(b_ptrs)
    acc = tl.dot(a, b)
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, acc)
