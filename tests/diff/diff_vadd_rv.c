/* diff_vadd_rv.c -- freestanding RV64 runner for the differential harness.
 *
 * qemu-riscv64 user-mode does not hand newlib its syscalls and does not
 * propagate the guest exit code, so we skip libc entirely: our own
 * _start, run the kernel, and write the raw output buffer to stdout. The
 * x86 comparator reads those bytes and does the actual diffing, which is
 * why this stays about thirty lines. One backend's only job here is to
 * produce its answer; judging it happens somewhere with a printf.
 *
 * Build (Linux / WSL):
 *   ./kath --triton --rv64 -o rv.o tests/tri_vadd.py
 *   riscv64-unknown-elf-gcc -nostdlib -static -march=rv64imfd -mabi=lp64d \
 *       -Itests/diff tests/diff/diff_vadd_rv.c rv.o -o rv_runner
 *   qemu-riscv64 ./rv_runner > rvout.bin
 */
#include "vadd_io.h"

/* x, y, out, n_elements, BLOCK_SIZE, then the hidden nthreads. */
extern void vector_add(float *x, float *y, float *out,
                       int n_elements, int block_size, int nthreads);

/* BSS. The loader zero-fills it; we overwrite anyway. */
static float x[VADD_N], y[VADD_N], out[VADD_N];

/* Linux/riscv syscall numbers: write = 64, exit = 93. */
static long sys_write(long fd, const void *buf, long n)
{
    register long a0 asm("a0") = fd;
    register long a1 asm("a1") = (long)buf;
    register long a2 asm("a2") = n;
    register long a7 asm("a7") = 64;
    asm volatile("ecall" : "+r"(a0) : "r"(a1), "r"(a2), "r"(a7) : "memory");
    return a0;
}

static void sys_exit(long code)
{
    register long a0 asm("a0") = code;
    register long a7 asm("a7") = 93;
    asm volatile("ecall" : : "r"(a0), "r"(a7) : "memory");
    __builtin_unreachable();
}

void _start(void)
{
    /* Set up the global pointer before touching any global. The linker
     * relaxes data loads to gp-relative, and without a valid gp they land
     * in the weeds, which is a segfault dressed up as a mystery. A real
     * crt0 does exactly this; we have no crt0, so we do it ourselves. */
    asm volatile(
        ".option push\n"
        ".option norelax\n"
        "la gp, __global_pointer$\n"
        ".option pop\n"
        ::: "memory");

    vadd_inputs(x, y);
    for (int i = 0; i < VADD_N; i++) out[i] = -123.0f;

    vector_add(x, y, out, VADD_N, VADD_BLOCK, VADD_N);

    sys_write(1, out, (long)sizeof(out));
    sys_exit(0);
}
