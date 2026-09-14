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

/* fn and num are both spelled out rather than pasted together, because ## on a
 * padded number hands TH_REG an octal constant and 09 does not even compile.
 * th_check confirms the two agree at startup. */
#define COMP_TEST(fn, num, desc, cu, extra) \
    static void fn(void) { \
        CHECK(compile_cu(cu, extra) == 0); \
        PASS(); \
    } \
    TH_REG("cmp", num, desc, fn)

COMP_TEST(cmp01,  1, "vector add",         "tests/vector_add.cu",        "")
COMP_TEST(cmp02,  2, "canonical kernel",   "tests/canonical.cu",         "")
COMP_TEST(cmp03,  3, "CUDA feature sweep", "tests/cuda_features.cu",     "")
COMP_TEST(cmp04,  4, "tier 1 and 2 ops",   "tests/test_tier12.cu",       "")
COMP_TEST(cmp05,  5, "overloaded operators", "tests/notgpt.cu",           "")
COMP_TEST(cmp06,  6, "register stress",    "tests/stress.cu",            "")
COMP_TEST(cmp07,  7, "maths intrinsics",   "tests/mymathhomework.cu",    "")
COMP_TEST(cmp08,  8, "launch bounds",      "tests/test_launch_bounds.cu", "")
COMP_TEST(cmp09,  9, "cooperative groups", "tests/test_coop_groups.cu",  "")
COMP_TEST(cmp10, 10, "preprocessor",       "tests/test_preproc.cu",      "")
COMP_TEST(cmp11, 11, "include path",       "tests/test_include.cu",      "-I tests")
COMP_TEST(cmp12, 12, "templates",          "tests/templates.cu",         "")
COMP_TEST(cmp13, 13, "unsigned arithmetic", "tests/test_unsigned.cu",    "")
COMP_TEST(cmp14, 14, "2D shared memory",   "tests/test_shared2d.cu",     "")
COMP_TEST(cmp15, 15, "tinygrad add",       "tests/tinygrad_add.cu",      "")
COMP_TEST(cmp16, 16, "tinygrad matmul",    "tests/tinygrad_matmul.cu",   "")
COMP_TEST(cmp17, 17, "tinygrad elementwise", "tests/tg_elem.cu",         "")
COMP_TEST(cmp18, 18, "tinygrad vec4",      "tests/tg_vec4.cu",           "")
COMP_TEST(cmp19, 19, "tinygrad fill",      "tests/tg_fill.cu",           "")
COMP_TEST(cmp20, 20, "tinygrad reduce",    "tests/tg_reduce.cu",         "")
COMP_TEST(cmp21, 21, "warp shuffle",       "tests/test_shfl.cu",         "")
COMP_TEST(cmp22, 22, "bit counting",       "tests/test_bitcount.cu",     "")
