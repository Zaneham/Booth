# tri_where_int.py - tl.where integer select lowering coverage.
import triton
import triton.language as tl

@triton.jit
def tri_where_int(a, b):
    out = tl.where(a > b, 1, 0)
