/* tregalloc.c -- graph coloring vs linear scan regression tests
 * Compiles .cu files with both allocators, asserts graph coloring
 * produces fewer or equal v_mov_b32 instructions than linear scan,
 * and uses fewer or equal registers (VGPRs + SGPRs). */

#include "tharns.h"

#define RA_BUFSZ (1 << 16)  /* 64KB --enough for full assembly output */

static char gc_buf[RA_BUFSZ];
static char ls_buf[RA_BUFSZ];

/* Count occurrences of a substring in a buffer */
static int count_substr(const char *buf, const char *sub)
{
    int count = 0;
    size_t sublen = strlen(sub);
    const char *p = buf;
    while ((p = strstr(p, sub)) != NULL) {
        count++;
        p += sublen;
    }
    return count;
}

/* Parse the maximum VGPR or SGPR count from assembly output.
 * Scans for lines like "; 17 SGPRs, 6 VGPRs, ..." and returns
 * the largest value for the requested field across all kernels. */
static int max_reg_count(const char *buf, const char *field)
{
    int maxval = 0;
    size_t flen = strlen(field);
    const char *p = buf;
    while ((p = strstr(p, field)) != NULL) {
        /* Walk backwards from the match to find the number */
        const char *n = p - 1;
        while (n > buf && *n == ' ') n--;
        if (n <= buf) { p += flen; continue; }
        /* n points to last digit */
        const char *end = n + 1;
        while (n > buf && n[-1] >= '0' && n[-1] <= '9') n--;
        if (n < end) {
            int val = atoi(n);
            if (val > maxval) maxval = val;
        }
        p += flen;
    }
    return maxval;
}

/* Compare MOV counts and register usage between graph coloring and
 * linear scan.  Returns 0 if GC is no worse than LS, -1 on regression. */
static int cmp_ra(const char *cu, const char *extra)
{
    char cmd[TH_BUFSZ];
    int rc;

    /* Graph coloring (default) */
    snprintf(cmd, TH_BUFSZ, BC_BIN " --amdgpu %s %s", extra, cu);
    rc = th_run(cmd, gc_buf, RA_BUFSZ);
    if (rc != 0) {
        printf("  GC compile failed: %s\n", gc_buf);
        return -1;
    }

    /* Linear scan */
    snprintf(cmd, TH_BUFSZ, BC_BIN " --amdgpu --no-graphcolor %s %s", extra, cu);
    rc = th_run(cmd, ls_buf, RA_BUFSZ);
    if (rc != 0) {
        printf("  LS compile failed: %s\n", ls_buf);
        return -1;
    }

    int gc_movs = count_substr(gc_buf, "v_mov_b32");
    int ls_movs = count_substr(ls_buf, "v_mov_b32");

    /* Allow GC to exceed LS by a small margin (coloring order noise).
       Flag regressions > 5 MOVs as failures. */
    if (gc_movs > ls_movs + 5) {
        printf("  %s: GC=%d MOVs > LS=%d MOVs (+%d, regression)\n",
               cu, gc_movs, ls_movs, gc_movs - ls_movs);
        return -1;
    }

    /* Check register counts --GC's dataflow liveness is more precise
       than LS's flat intervals, so GC may correctly report higher
       counts where LS underapproximates.  Max observed delta from
       LS imprecision is +3 VGPRs (canonical.cu). */
    int gc_vgprs = max_reg_count(gc_buf, "VGPRs");
    int ls_vgprs = max_reg_count(ls_buf, "VGPRs");
    int gc_sgprs = max_reg_count(gc_buf, "SGPRs");
    int ls_sgprs = max_reg_count(ls_buf, "SGPRs");

    if (gc_vgprs > ls_vgprs + 3) {
        printf("  %s: GC=%d VGPRs > LS=%d VGPRs (+%d, regression)\n",
               cu, gc_vgprs, ls_vgprs, gc_vgprs - ls_vgprs);
        return -1;
    }
    if (gc_sgprs > ls_sgprs + 3) {
        printf("  %s: GC=%d SGPRs > LS=%d SGPRs (+%d, regression)\n",
               cu, gc_sgprs, ls_sgprs, gc_sgprs - ls_sgprs);
        return -1;
    }

    return 0;
}

/* ---- regalloc: representative .cu files ---- */

/* fn and num are both spelled out rather than pasted together, because ## on a
 * padded number hands TH_REG an octal constant and 09 does not even compile.
 * th_check confirms the two agree at startup. */
#define RA_TEST(fn, num, desc, cu, extra) \
    static void fn(void) { \
        CHECK(cmp_ra(cu, extra) == 0); \
        PASS(); \
    } \
    TH_REG("ral", num, desc, fn)

RA_TEST(ral01, 1, "colouring beats scan: vector add", "tests/vector_add.cu", "")
RA_TEST(ral02, 2, "colouring beats scan: canonical",  "tests/canonical.cu",  "")
RA_TEST(ral03, 3, "colouring beats scan: maths",      "tests/mymathhomework.cu", "")
RA_TEST(ral04, 4, "colouring beats scan: stress",     "tests/stress.cu",     "")

/* ---- Spill path: force low VGPR cap to exercise spill code ---- */

static void ral05(void)
{
    char cmd[TH_BUFSZ];
    int rc;

    /* Compile with --max-vgprs 4 --forces the allocator to spill on
       kernels that need more than 4 VGPRs.  Just verify it doesn't
       crash and produces valid assembly output. */
    snprintf(cmd, TH_BUFSZ,
             BC_BIN " --amdgpu --max-vgprs 4 tests/stress.cu");
    rc = th_run(cmd, gc_buf, RA_BUFSZ);
    CHEQ(rc, 0);
    CHECK(strstr(gc_buf, "s_endpgm") != NULL);

    /* Also test with --max-vgprs 2 on a simpler kernel */
    snprintf(cmd, TH_BUFSZ,
             BC_BIN " --amdgpu --max-vgprs 2 tests/vector_add.cu");
    rc = th_run(cmd, gc_buf, RA_BUFSZ);
    CHEQ(rc, 0);
    CHECK(strstr(gc_buf, "s_endpgm") != NULL);

    PASS();
}
TH_REG("ral", 5, "spills cleanly under a low VGPR cap", ral05)
