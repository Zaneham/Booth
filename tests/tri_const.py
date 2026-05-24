"""Constexpr default values: BLOCK is known at compile time, so
shape inference produces vec[256] instead of vec[?]."""
import triton
import triton.language as tl

@triton.jit
def const_vadd(x_ptr, y_ptr, out_ptr, n_elements,
               BLOCK: tl.constexpr = 256):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK
    offsets = block_start + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, x + y, mask=mask)
