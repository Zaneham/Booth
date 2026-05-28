/* diff_vadd.c -- worked differential test for the Triton vector_add kernel.
 *
 * Subject: the kernel as compiled by `barracuda --triton --cpu`, called
 * the usual way (its own params, then the hidden nthreads). Oracle: the
 * same sum worked out in plain host C. We diff the two.
 *
 * Right now the oracle is a host reference, which is enough to catch
 * frontend, lowering and CPU-codegen bugs and runs on any machine. Swap
 * the oracle for a second backend (rv64 under qemu, or a GPU run through
 * bc_dispatch) and the comparison below does not change one line: that is
 * the whole idea, the compare core does not care who produced the bytes.
 *
 * Build + run lives in run_diff.sh. Pass --inject to corrupt one output
 * on purpose, which is how we check the harness fails when it should.
 *
 * Build (Linux / WSL):
 *   ./barracuda --triton --cpu -o vadd.o tests/tri_vadd.py
 *   gcc -no-pie -Itests/diff tests/diff/diff_vadd.c vadd.o -o diff_vadd -lm
 *   ./diff_vadd            # PASS
 *   ./diff_vadd --inject   # FAIL (proves the harness has teeth)
 */
#include <stdio.h>
#include <string.h>
#include "bc_diff.h"
#include "vadd_io.h"

/* x, y, out, n_elements, BLOCK_SIZE, then the hidden nthreads. */
extern void vector_add(float *x, float *y, float *out,
                       int n_elements, int block_size, int nthreads);

int main(int argc, char **argv)
{
    int inject = (argc > 1 && strcmp(argv[1], "--inject") == 0);

    float x[VADD_N], y[VADD_N], got[VADD_N], ref[VADD_N];
    vadd_inputs(x, y);
    for (int i = 0; i < VADD_N; i++) got[i] = -123.0f;

    /* Subject: run the compiled kernel. One block, nthreads = N. */
    vector_add(x, y, got, VADD_N, VADD_BLOCK, VADD_N);

    /* Oracle: the answer we already know. */
    for (int i = 0; i < VADD_N; i++) ref[i] = x[i] + y[i];

    /* Simulate a backend that disagrees, to prove FAIL is reachable. */
    if (inject) got[VADD_N / 3] += 1.0f;

    bc_diff_report_t rep;
    bc_diff_f32(got, ref, VADD_N, 1e-6f, 1e-5f, &rep);
    int bad = bc_diff_print("cpu vs host", &rep);

    return bad;
}
