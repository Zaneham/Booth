/* tcomp.c -- compile matrix
 * Every .cu file, every target. The brute-force approach to confidence. */

#include "tharns.h"

static char obuf[TH_BUFSZ];

/* ---- Helpers ---- */

static const char *targets[] = { "--gfx1030", "", "--gfx1200" };
static const char *tnames[]  = { "gfx1030", "gfx1100", "gfx1200" };

static int compile_cu(const char *cu, const char *extra)
{
    char cmd[TH_BUFSZ];
    const char *out = "test_out.hsaco";

    for (int t = 0; t < 3; t++) {
        snprintf(cmd, TH_BUFSZ,
                 BC_BIN " --amdgpu-bin %s %s %s -o %s",
                 targets[t], extra, cu, out);
        int rc = th_run(cmd, obuf, TH_BUFSZ);
        if (rc != 0) {
            printf("  target %s failed: %s\n", tnames[t], obuf);
            return -1;
        }
        if (!th_exist(out)) {
            printf("  target %s: output missing\n", tnames[t]);
            return -1;
        }
        remove(out);
    }
    return 0;
}

/* ---- compile: individual .cu files ---- */

#define COMP_TEST(name, cu, extra) \
    static void name(void) { \
        CHECK(compile_cu(cu, extra) == 0); \
        PASS(); \
    } \
    TH_REG("compile", name)

COMP_TEST(cmp_vecadd,  "tests/vector_add.cu",         "")
COMP_TEST(cmp_canon,   "tests/canonical.cu",           "")
COMP_TEST(cmp_feat,    "tests/cuda_features.cu",       "")
COMP_TEST(cmp_tier12,  "tests/test_tier12.cu",         "")
COMP_TEST(cmp_notgpt,  "tests/notgpt.cu",              "")
COMP_TEST(cmp_stress,  "tests/stress.cu",              "")
COMP_TEST(cmp_math,    "tests/mymathhomework.cu",      "")
COMP_TEST(cmp_launch,  "tests/test_launch_bounds.cu",  "")
COMP_TEST(cmp_coop,    "tests/test_coop_groups.cu",    "")
COMP_TEST(cmp_preproc, "tests/test_preproc.cu",        "")
COMP_TEST(cmp_includ,  "tests/test_include.cu",        "-I tests")
COMP_TEST(cmp_tmpl,    "tests/templates.cu",           "")
COMP_TEST(cmp_unsign,  "tests/test_unsigned.cu",       "")
COMP_TEST(cmp_shr2d,   "tests/test_shared2d.cu",      "")
COMP_TEST(cmp_tgadd,   "tests/tinygrad_add.cu",       "")
COMP_TEST(cmp_tgmul,   "tests/tinygrad_matmul.cu",    "")
COMP_TEST(cmp_tgelm,   "tests/tg_elem.cu",            "")
COMP_TEST(cmp_tgvc4,   "tests/tg_vec4.cu",            "")
COMP_TEST(cmp_tgfil,   "tests/tg_fill.cu",            "")
COMP_TEST(cmp_tgred,   "tests/tg_reduce.cu",          "")
COMP_TEST(cmp_shfl,    "tests/test_shfl.cu",           "")
