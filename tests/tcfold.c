/* tcfold.c -- Constant folding tests.
 * Verify that constant folding evaluates constant expressions. */

#include "tharns.h"

static char obuf[TH_BUFSZ];

/* ---- Helpers ---- */

static const char *strnstr_range(const char *start, const char *end,
                                 const char *needle)
{
    size_t nlen = strlen(needle);
    for (const char *p = start; p + nlen <= end; p++) {
        if (memcmp(p, needle, nlen) == 0) return p;
    }
    return NULL;
}

/* ---- cf: integer arithmetic folded ---- */

static void cfd01(void)
{
    int rc = th_run(BC_BIN " --ir tests/test_cf.cu", obuf, TH_BUFSZ);
    CHEQ(rc, 0);
    const char *fn = strstr(obuf, "cf_int_arith");
    CHECK(fn != NULL);
    const char *body = strchr(fn, '\n');
    CHECK(body != NULL);
    const char *fn_end = strstr(body, "\n}");
    CHECK(fn_end != NULL);
    /* The constant add (3+4) must be folded — add with param survives */
    CHECK(strnstr_range(body, fn_end, "= add") != NULL);
    /* The constant 7 (= 3+4) should appear as an operand */
    CHECK(strnstr_range(body, fn_end, " 7") != NULL);
    PASS();
}
TH_REG("cfd", 1, "integer arithmetic folds", cfd01)

/* ---- cf: chained constants fold ---- */

static void cfd02(void)
{
    int rc = th_run(BC_BIN " --ir tests/test_cf.cu", obuf, TH_BUFSZ);
    CHEQ(rc, 0);
    const char *fn = strstr(obuf, "cf_chain");
    CHECK(fn != NULL);
    const char *body = strchr(fn, '\n');
    CHECK(body != NULL);
    const char *fn_end = strstr(body, "\n}");
    CHECK(fn_end != NULL);
    /* mul (5*4=20) must be folded away */
    CHECK(strnstr_range(body, fn_end, "= mul") == NULL);
    /* The constant 20 should appear as an operand */
    CHECK(strnstr_range(body, fn_end, " 20") != NULL);
    PASS();
}
TH_REG("cfd", 2, "folding follows a chain", cfd02)

/* ---- cf: icmp + select with constant condition ---- */

static void cfd03(void)
{
    int rc = th_run(BC_BIN " --ir tests/test_cf.cu", obuf, TH_BUFSZ);
    CHEQ(rc, 0);
    const char *fn = strstr(obuf, "cf_icmp_select");
    CHECK(fn != NULL);
    const char *body = strchr(fn, '\n');
    CHECK(body != NULL);
    const char *fn_end = strstr(body, "\n}");
    CHECK(fn_end != NULL);
    /* icmp and select must be folded away */
    CHECK(strnstr_range(body, fn_end, "= icmp") == NULL);
    CHECK(strnstr_range(body, fn_end, "= select") == NULL);
    PASS();
}
TH_REG("cfd", 3, "icmp feeding select folds", cfd03)

/* ---- cf: integer division by zero not folded ---- */

static void cfd04(void)
{
    int rc = th_run(BC_BIN " --ir tests/test_cf.cu", obuf, TH_BUFSZ);
    CHEQ(rc, 0);
    const char *fn = strstr(obuf, "cf_divzero");
    CHECK(fn != NULL);
    const char *body = strchr(fn, '\n');
    CHECK(body != NULL);
    const char *fn_end = strstr(body, "\n}");
    CHECK(fn_end != NULL);
    /* sdiv by zero must survive — folding it would be UB */
    CHECK(strnstr_range(body, fn_end, "= sdiv") != NULL);
    PASS();
}
TH_REG("cfd", 4, "sdiv by zero survives, folding it is UB", cfd04)

/* ---- cf: integer/float conversions folded ---- */

static void cfd05(void)
{
    int rc = th_run(BC_BIN " --ir tests/test_cf.cu", obuf, TH_BUFSZ);
    CHEQ(rc, 0);
    const char *fn = strstr(obuf, "cf_conv");
    CHECK(fn != NULL);
    const char *body = strchr(fn, '\n');
    CHECK(body != NULL);
    const char *fn_end = strstr(body, "\n}");
    CHECK(fn_end != NULL);
    /* sitofp and fptosi must be folded away */
    CHECK(strnstr_range(body, fn_end, "= sitofp") == NULL);
    CHECK(strnstr_range(body, fn_end, "= fptosi") == NULL);
    PASS();
}
TH_REG("cfd", 5, "conversions fold", cfd05)

/* ---- cf: float arithmetic folded ---- */

static void cfd06(void)
{
    int rc = th_run(BC_BIN " --ir tests/test_cf.cu", obuf, TH_BUFSZ);
    CHEQ(rc, 0);
    const char *fn = strstr(obuf, "cf_float");
    CHECK(fn != NULL);
    const char *body = strchr(fn, '\n');
    CHECK(body != NULL);
    const char *fn_end = strstr(body, "\n}");
    CHECK(fn_end != NULL);
    /* fadd must be folded away — constant 4 (=1.5+2.5) stored directly */
    CHECK(strnstr_range(body, fn_end, "= fadd") == NULL);
    /* The folded value 4 should appear as an f32 operand */
    CHECK(strnstr_range(body, fn_end, "f32 4") != NULL);
    PASS();
}
TH_REG("cfd", 6, "float arithmetic folds", cfd06)

/* ---- cf: subtraction folds in the right order ---- */

static void cfd07(void)
{
    int rc = th_run(BC_BIN " --ir tests/test_cf.cu", obuf, TH_BUFSZ);
    CHEQ(rc, 0);
    const char *fn = strstr(obuf, "cf_sub");
    CHECK(fn != NULL);
    const char *body = strchr(fn, '\n');
    CHECK(body != NULL);
    const char *fn_end = strstr(body, "\n}");
    CHECK(fn_end != NULL);
    /* 20-6 is 14. Backwards it is -14, which prints with the sign attached,
     * so the leading space is doing real work here. */
    CHECK(strnstr_range(body, fn_end, " 14") != NULL);
    PASS();
}
TH_REG("cfd", 7, "subtraction folds in order", cfd07)

static void cfd08(void)
{
    int rc = th_run(BC_BIN " --ir tests/cf_dead.cu", obuf, TH_BUFSZ);
    CHEQ(rc, 0);
    const char *fn = strstr(obuf, "cf_deadarm");
    CHECK(fn != NULL);
    const char *body = strchr(fn, '\n');
    CHECK(body != NULL);
    const char *fn_end = strstr(body, "\n}");
    CHECK(fn_end != NULL);
    CHECK(strnstr_range(body, fn_end, "br_cond") == NULL);
    CHECK(strnstr_range(body, fn_end, "store i32 99") == NULL);
    CHECK(strnstr_range(body, fn_end, "unreachable") != NULL);
    PASS();
}
TH_REG("cfd", 8, "a branch on two literals drops its dead arm", cfd08)
