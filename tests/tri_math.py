# tri_math.py - scalar tl.math lowering coverage.
import triton
import triton.language as tl

@triton.jit
def tri_math(in_ptr, out_ptr):
    pid = tl.program_id(axis=0)
    x = tl.load(in_ptr + pid)
    e = tl.exp(x)
    l = tl.log(e)
    s = tl.sin(l)
    c = tl.cos(s)
    t = tl.tan(c)
    h = tl.tanh(t)
    q = tl.sqrt(h)
    r = tl.rsqrt(q)
    a = tl.abs(r)
    f = tl.floor(a)
    g = tl.ceil(f)
    e2 = tl.exp2(g)
    l2 = tl.log2(e2)
    hi = tl.maximum(l2, x)
    lo = tl.minimum(hi, e2)
    y = tl.fdiv(lo, 2.0)
    tl.store(out_ptr + pid, y)
