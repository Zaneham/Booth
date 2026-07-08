# tri_where_bad_cond.py - tl.where rejects non-mask conditions.
import triton
import triton.language as tl

@triton.jit
def tri_where_bad_cond(a, b):
    out = tl.where(1.0, a, b)
