/* diff_xbackend.c -- the cross-backend comparator for vector_add.
 *
 * This is the real differential test: it runs the kernel on x86 (the
 * --cpu backend, linked straight in), reads the rv64 backend's output
 * from a file the qemu runner produced, and diffs both against the host
 * oracle and against each other. Same BIR, two independent codegen paths.
 * If cpu and rv64 disagree, one of them has a codegen bug and you find
 * out without owning a GPU.
 *
 * The comparison lives here, on the libc side, because exit codes and
 * printf actually work here. The rv64 side just hands us bytes.
 *
 * Build + run is wired into run_diff.sh. Pass --inject to corrupt the
 * rv64 buffer on purpose, which proves the cpu-vs-rv64 diff has teeth.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "bc_diff.h"
#include "vadd_io.h"

extern void vector_add(float *x, float *y, float *out,
                       int n_elements, int block_size, int nthreads);

int main(int argc, char **argv)
{
    const char *rvpath = NULL;
    int inject = 0;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--inject") == 0) inject = 1;
        else rvpath = argv[i];
    }
    if (!rvpath) {
        fprintf(stderr, "usage: %s [--inject] <rv64-output.bin>\n", argv[0]);
        return 2;
    }

    float x[VADD_N], y[VADD_N], cpu[VADD_N], rv[VADD_N], ref[VADD_N];
    vadd_inputs(x, y);
    for (int i = 0; i < VADD_N; i++) { cpu[i] = -123.0f; ref[i] = x[i] + y[i]; }

    /* x86 backend, run in-process. */
    vector_add(x, y, cpu, VADD_N, VADD_BLOCK, VADD_N);

    /* rv64 backend, read the bytes qemu's runner wrote. */
    FILE *f = fopen(rvpath, "rb");
    if (!f) { perror("open rv64 output"); return 2; }
    size_t got = fread(rv, sizeof(float), VADD_N, f);
    fclose(f);
    if (got != (size_t)VADD_N) {
        fprintf(stderr, "short read: %zu/%d floats from %s\n",
                got, VADD_N, rvpath);
        return 2;
    }

    if (inject) rv[VADD_N / 3] += 1.0f;

    bc_diff_report_t r;
    int bad = 0;
    bc_diff_f32(cpu, ref, VADD_N, 1e-6f, 1e-5f, &r); bad |= bc_diff_print("cpu vs host", &r);
    bc_diff_f32(rv,  ref, VADD_N, 1e-6f, 1e-5f, &r); bad |= bc_diff_print("rv64 vs host", &r);
    bc_diff_f32(cpu, rv,  VADD_N, 1e-6f, 1e-5f, &r); bad |= bc_diff_print("cpu vs rv64", &r);

    return bad;
}
