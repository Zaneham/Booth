# tri_where_mixed.py - tl.where numeric promotion coverage.
import triton
import triton.language as tl

@triton.jit
def tri_where_mixed(a_ptr, b_ptr, out_ptr, BLOCK: tl.constexpr = 256):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    a = tl.load(a_ptr + offsets)
    b = tl.load(b_ptr + offsets)
    out = tl.where(a > b, a, 0)
    tl.store(out_ptr + offsets, out)
