/* tra_ssa.c -- divergence-aware SSA register allocator (--ssa-ra)
 *
 * ra_ssa.c had no coverage at all before this file: the allocator is gated
 * behind --ssa-ra and nothing passed the flag. These drive it end to end and
 * check the two things an allocator must never get wrong -- leaving virtual
 * registers behind, and using more registers than the kernel descriptor
 * declares. A kernel that reads past its declared count reads whatever the
 * previous wave left there.
 *
 * Six fixtures still leak vregs (see rss05). They are pinned
 * here rather than skipped, so fixing the allocator fails this file and makes
 * whoever fixes it move the fixture into the clean list. */

#include "tharns.h"

#define SR_BUFSZ (1 << 19)   /* 512KB, canonical.cu emits a lot */

static char sr_buf[SR_BUFSZ];

/* Highest register index referenced on a line, for 'v' or 's'.
 * Handles both v1 and the v[0:1] pair form. Returns -1 for none. */
static int line_max_reg(const char *ln, char pfx)
{
    int max = -1;
    const char *p = ln;

    while (*p) {
        /* a register only starts where the previous char isn't identifier-ish,
         * otherwise s_waitcnt and lgkmcnt0 both look like registers */
        int boundary = (p == ln) ||
                       !((p[-1] >= 'a' && p[-1] <= 'z') ||
                         (p[-1] >= 'A' && p[-1] <= 'Z') ||
                         (p[-1] >= '0' && p[-1] <= '9') || p[-1] == '_');
        if (*p == pfx && boundary) {
            if (p[1] == '[') {              /* v[lo:hi] -- hi is what matters */
                const char *c = strchr(p, ':');
                if (c && c[1] >= '0' && c[1] <= '9') {
                    int hi = atoi(c + 1);
                    if (hi > max) max = hi;
                }
            } else if (p[1] >= '0' && p[1] <= '9') {
                int n = atoi(p + 1);
                if (n > max) max = n;
            }
        }
        p++;
    }
    return max;
}

/* Walk the asm and check every kernel's declared counts actually bound the
 * registers its body touches. Returns 0 if clean, -1 on an overrun. */
static int check_bounds(const char *buf)
{
    const char *p = buf;
    int dv = -1, ds = -1, maxv = -1, maxs = -1;
    char ln[1024];

    while (*p) {
        const char *nl = strchr(p, '\n');
        size_t len = nl ? (size_t)(nl - p) : strlen(p);
        if (len >= sizeof ln) len = sizeof ln - 1;
        memcpy(ln, p, len);
        ln[len] = '\0';

        /* "; 14 SGPRs, 5 VGPRs, ..." opens a new function */
        const char *sg = strstr(ln, " SGPRs");
        if (sg) {
            int a = 0, b = 0;
            if (sscanf(ln, "; %d SGPRs, %d VGPRs", &a, &b) == 2) {
                ds = a; dv = b; maxv = -1; maxs = -1;
            }
        } else {
            int v = line_max_reg(ln, 'v');
            int s = line_max_reg(ln, 's');
            if (v > maxv) maxv = v;
            if (s > maxs) maxs = s;
        }

        if (strstr(ln, "s_endpgm") && dv >= 0) {
            if (maxv >= dv) {
                printf("  VGPR overrun: declared %d, used v%d\n", dv, maxv);
                return -1;
            }
            if (maxs >= ds) {
                printf("  SGPR overrun: declared %d, used s%d\n", ds, maxs);
                return -1;
            }
            dv = -1; ds = -1;
        }

        if (!nl) break;
        p = nl + 1;
    }
    return 0;
}

/* Compile with --ssa-ra. Returns kath's exit code, output in sr_buf. */
static int ssa_compile(const char *cu, const char *extra)
{
    char cmd[TH_BUFSZ];
    snprintf(cmd, TH_BUFSZ, BC_BIN " --amdgpu --ssa-ra %s %s", extra, cu);
    return th_run(cmd, sr_buf, SR_BUFSZ);
}

/* Fixtures the SSA allocator handles today. */
static const char *sr_clean[] = {
    "tests/vector_add.cu",
    "tests/tiny.cu",
    "tests/test_vadd.cu",
    "tests/test_loop.cu",
    "tests/test_branch.cu",
    "tests/test_cf.cu",
    NULL
};

/* Fixtures where it still leaves vregs unallocated. */
static const char *sr_leaky[] = {
    "tests/canonical.cu",
    "tests/stress.cu",
    "tests/cuda_features.cu",
    "tests/device_calls.cu",
    "tests/test_struct.cu",
    NULL
};

/* ---- The allocator finishes its job ---- */

/* Every vreg must be gone by the time RA is done. verify.c says so too, but
 * asserting it here names the actual failure instead of an exit code. */
static void rss01(void)
{
    int i;
    for (i = 0; sr_clean[i]; i++) {
        int rc = ssa_compile(sr_clean[i], "");
        if (rc != 0) {
            printf("  %s: exit %d\n  %s\n", sr_clean[i], rc, sr_buf);
            CHECK(0);
        }
        if (strstr(sr_buf, "still present after RA")) {
            printf("  %s: allocator left virtual regs behind\n", sr_clean[i]);
            CHECK(0);
        }
    }
    PASS();
}
TH_REG("rss", 1, "no vreg leaks into the output", rss01)

/* ---- Declared registers bound the ones actually used ---- */

static void rss02(void)
{
    int i;
    for (i = 0; sr_clean[i]; i++) {
        CHEQ(ssa_compile(sr_clean[i], ""), 0);
        if (check_bounds(sr_buf) != 0) {
            printf("  in %s\n", sr_clean[i]);
            CHECK(0);
        }
    }
    PASS();
}
TH_REG("rss", 2, "within declared", rss02)

/* Same check with the budget squeezed enough to force spilling. */
static void rss03(void)
{
    int i;
    for (i = 0; sr_clean[i]; i++) {
        int rc = ssa_compile(sr_clean[i], "--max-vgprs 8");
        if (rc != 0) {
            printf("  %s: exit %d\n", sr_clean[i], rc);
            CHECK(0);
        }
        CHECK(strstr(sr_buf, "s_endpgm") != NULL);
        if (check_bounds(sr_buf) != 0) {
            printf("  in %s\n", sr_clean[i]);
            CHECK(0);
        }
    }
    PASS();
}
TH_REG("rss", 3, "within declared spilling", rss03)

/* Squeeze harder and the spill path leaks vregs too, on kernels the default
 * allocator handles down to --max-vgprs 2. Same deal as sr_leaky: pinned so a
 * fix trips this and gets folded into the test above. */
static void rss04(void)
{
    const char *caps[] = { "--max-vgprs 4", "--max-vgprs 2", NULL };
    int c;

    for (c = 0; caps[c]; c++) {
        int rc = ssa_compile("tests/vector_add.cu", caps[c]);
        if (rc == 0) {
            printf("  vector_add now survives %s, fold it back in\n", caps[c]);
            CHECK(0);
        }
        if (!strstr(sr_buf, "still present after RA")) {
            printf("  %s: failed but not on a vreg leak\n", caps[c]);
            CHECK(0);
        }
    }
    PASS();
}
TH_REG("rss", 4, "tight cap rejects cleanly", rss04)

/* ---- Known-broken fixtures fail loudly ---- */

/* These leak vregs. What matters until that is fixed is that verify catches
 * it and we exit non-zero, rather than quietly emitting a broken kernel.
 * Fix the allocator and this test fails: move the fixture to sr_clean. */
static void rss05(void)
{
    int i;
    for (i = 0; sr_leaky[i]; i++) {
        int rc = ssa_compile(sr_leaky[i], "");
        if (rc == 0) {
            printf("  %s now passes --ssa-ra, move it to sr_clean\n", sr_leaky[i]);
            CHECK(0);
        }
        if (!strstr(sr_buf, "still present after RA")) {
            printf("  %s: failed but not on a vreg leak\n", sr_leaky[i]);
            CHECK(0);
        }
    }
    PASS();
}
TH_REG("rss", 5, "rejects cleanly", rss05)

/* ---- The allocator doesn't take the compiler down ---- */

/* Bad output is one thing, a crash is another. Everything above runs through
 * here with the budget squeezed hard, and a clean rejection is fine. */
static void rss06(void)
{
    const char *caps[] = { "", "--max-vgprs 4", "--max-vgprs 2", NULL };
    int c, i;

    for (c = 0; caps[c]; c++) {
        for (i = 0; sr_leaky[i]; i++) {
            int rc = ssa_compile(sr_leaky[i], caps[c]);
            if (rc != 0 && rc != 1) {
                printf("  %s %s: exit %d, not a clean rejection\n",
                       sr_leaky[i], caps[c], rc);
                CHECK(0);
            }
        }
        for (i = 0; sr_clean[i]; i++) {
            int rc = ssa_compile(sr_clean[i], caps[c]);
            if (rc != 0 && rc != 1) {
                printf("  %s %s: exit %d, not a clean rejection\n",
                       sr_clean[i], caps[c], rc);
                CHECK(0);
            }
        }
    }
    PASS();
}
TH_REG("rss", 6, "survives pressure", rss06)

/* ---- Both allocators agree on the shape of what they emit ---- */

/* Not a text diff: the whole point is that they allocate differently. But a
 * kernel is still a kernel, so the entry label and terminator must survive. */
static void rss07(void)
{
    char cmd[TH_BUFSZ];
    int i;

    for (i = 0; sr_clean[i]; i++) {
        CHEQ(ssa_compile(sr_clean[i], ""), 0);
        CHECK(strstr(sr_buf, "s_endpgm") != NULL);
        CHECK(strstr(sr_buf, ".amdgcn_target") != NULL);

        snprintf(cmd, TH_BUFSZ, BC_BIN " --amdgpu %s", sr_clean[i]);
        CHEQ(th_run(cmd, sr_buf, SR_BUFSZ), 0);
        CHECK(strstr(sr_buf, "s_endpgm") != NULL);
    }
    PASS();
}
TH_REG("rss", 7, "the allocator emits a kernel", rss07)

/* ---- The allocator succeeds on things that fit ---- */

/* rss06 accepts a clean rejection, which is right for a squeezed budget and
 * wrong as the only assertion: an allocator that rejected everything would
 * pass every test above it. These fixtures fit inside the default ceiling and
 * have to actually compile. */
static void rss08(void)
{
    int i;

    for (i = 0; sr_clean[i]; i++) {
        int rc = ssa_compile(sr_clean[i], "");
        if (rc != 0) {
            printf("  %s: exit %d, should have fitted\n", sr_clean[i], rc);
            CHECK(0);
        }
    }
    PASS();
}
TH_REG("rss", 8, "kernels that fit are allocated", rss08)
