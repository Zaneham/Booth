/* bc_diff.h -- differential testing core for BarraCUDA backends.
 *
 * One IR, many targets. Run the same kernel through two of them, diff the
 * output buffers, and you find out fast whether a wrong answer is your
 * maths or the codegen. The CPU backend makes a fine oracle: it is the
 * simplest target we have, stack-everything and no clever scheduling, so
 * when CPU and GPU disagree the GPU is usually the one telling fibs.
 *
 * This header is just the compare half: tolerance, a tidy report, a
 * verdict. It knows nothing about how either buffer was produced, which
 * is the point. The CPU runner, an rv64-under-qemu runner, and a GPU
 * runner (bc_dispatch) all feed the same comparison.
 *
 * Float is not exact across backends and that is fine. Transcendentals
 * and approximate ops (rsqrt, sin, the usual suspects) drift a little
 * between an x86 FPU and a GPU's fast-math unit, so this is a tolerance
 * check, not a bit-for-bit one. Pick the tolerance to match the kernel.
 */
#ifndef BC_DIFF_H
#define BC_DIFF_H

#include <stddef.h>
#include <math.h>
#include <stdio.h>

typedef struct {
    size_t n;            /* elements compared                         */
    size_t n_mismatch;   /* how many fell outside tolerance           */
    size_t first_bad;    /* index of the first mismatch (n if none)   */
    double max_abs_err;  /* largest |got - ref| seen                  */
    double max_rel_err;  /* largest |got - ref| / |ref| seen          */
    double got_at_first; /* the offending pair, for the report        */
    double ref_at_first;
} bc_diff_report_t;

/* A single element passes if it is within EITHER the absolute or the
 * relative tolerance. Two NaNs count as agreement (both backends gave
 * up in the same place); a NaN facing a number does not. */
static int bc_diff_ok_f32(float got, float ref, float abstol, float reltol)
{
    if (isnan(got) || isnan(ref)) return isnan(got) && isnan(ref);
    double d = fabs((double)got - (double)ref);
    if (d <= (double)abstol) return 1;
    double r = fabs((double)ref);
    return r > 0.0 && d <= (double)reltol * r;
}

/* Compare two float buffers. Returns the mismatch count (0 == clean) and
 * fills rep if you hand it one. */
static size_t bc_diff_f32(const float *got, const float *ref, size_t n,
                          float abstol, float reltol,
                          bc_diff_report_t *rep)
{
    bc_diff_report_t r;
    r.n = n; r.n_mismatch = 0; r.first_bad = n;
    r.max_abs_err = 0.0; r.max_rel_err = 0.0;
    r.got_at_first = 0.0; r.ref_at_first = 0.0;

    for (size_t i = 0; i < n; i++) {
        double ad = fabs((double)got[i] - (double)ref[i]);
        double rd = fabs((double)ref[i]) > 0.0
                  ? ad / fabs((double)ref[i]) : 0.0;
        if (ad > r.max_abs_err) r.max_abs_err = ad;
        if (rd > r.max_rel_err) r.max_rel_err = rd;
        if (!bc_diff_ok_f32(got[i], ref[i], abstol, reltol)) {
            if (r.first_bad == n) {
                r.first_bad = i;
                r.got_at_first = (double)got[i];
                r.ref_at_first = (double)ref[i];
            }
            r.n_mismatch++;
        }
    }
    if (rep) *rep = r;
    return r.n_mismatch;
}

/* Print a one-glance verdict. label names what was compared, e.g.
 * "cpu vs host" or "cpu vs gpu". */
static int bc_diff_print(const char *label, const bc_diff_report_t *r)
{
    if (r->n_mismatch == 0) {
        printf("PASS  %-16s  %zu elems, max abs %.3g, max rel %.3g\n",
               label, r->n, r->max_abs_err, r->max_rel_err);
        return 0;
    }
    printf("FAIL  %-16s  %zu/%zu mismatched, first at [%zu] "
           "got %.9g vs ref %.9g (max abs %.3g, max rel %.3g)\n",
           label, r->n_mismatch, r->n, r->first_bad,
           r->got_at_first, r->ref_at_first, r->max_abs_err, r->max_rel_err);
    return 1;
}

#endif /* BC_DIFF_H */
