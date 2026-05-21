# tri_simple.py - smallest Triton kernel that survives DCE.
# We need the arithmetic chain to flow into a tl.store so the
# optimisation passes can not declare the kernel as a no-op.
import triton
import triton.language as tl

@triton.jit
def tri_simple(out_ptr, BLOCK: tl.constexpr):
    pid = tl.program_id(axis=0)
    nbr = tl.num_programs(axis=0)
    a = pid + 1
    b = a * 2
    c = b - pid
    d = c % nbr
    e = -d
    tl.store(out_ptr + pid, e)
