/* tsmoke.c -- CLI smoke tests
 * Does the binary do anything? Anything at all? Let's find out. */

#include "tharns.h"

static char obuf[TH_BUFSZ];

/* ---- smoke: help ---- */

static void smk_help(void)
{
    int rc = th_run(BC_BIN " --help", obuf, TH_BUFSZ);
    CHEQ(rc, 0);
    CHECK(strstr(obuf, "Usage") != NULL);
    PASS();
}
TH_REG("smoke", smk_help)

/* ---- smoke: no args ---- */

static void smk_noarg(void)
{
    int rc = th_run(BC_BIN, obuf, TH_BUFSZ);
    CHNE(rc, 0);
    PASS();
}
TH_REG("smoke", smk_noarg)

/* ---- smoke: version ---- */

static void smk_vers(void)
{
    SKIP("not implemented");
}
TH_REG("smoke", smk_vers)

/* ---- smoke: bad flag ---- */

static void smk_badf(void)
{
    int rc = th_run(BC_BIN " --nonsense", obuf, TH_BUFSZ);
    CHNE(rc, 0);
    PASS();
}
TH_REG("smoke", smk_badf)

/* ---- smoke: backend gates ---- */

/* Exit 0 with nothing on disk is the worst failure we can hand a build
 * system: it believes us. Every emitting mode gets checked both ways. */

static int smk_emit(const char *args, const char *out)
{
    char cmd[512];
    FILE *f;
    int rc, got;

    remove(out);
    snprintf(cmd, sizeof cmd, "%s %s -o %s", BC_BIN, args, out);
    rc = th_run(cmd, obuf, TH_BUFSZ);
    f = fopen(out, "rb");
    got = f != NULL;
    if (f) fclose(f);
    remove(out);
    return rc == 0 && got;
}

static void smk_tcpu(void)
{
    CHECK(smk_emit("--triton --cpu tests/tri_vadd.py", "smk_tc.o"));
    PASS();
}
TH_REG("smoke", smk_tcpu)

static void smk_trv64(void)
{
    CHECK(smk_emit("--triton --rv64 tests/tri_vadd.py", "smk_tr.o"));
    PASS();
}
TH_REG("smoke", smk_trv64)

static void smk_tptx(void)
{
    CHECK(smk_emit("--triton --nvidia-ptx tests/tri_vadd.py", "smk_tp.ptx"));
    PASS();
}
TH_REG("smoke", smk_tptx)

static void smk_ccpu(void)
{
    CHECK(smk_emit("--cpu tests/canonical.cu", "smk_cc.o"));
    PASS();
}
TH_REG("smoke", smk_ccpu)

/* The Triton fallthrough returned the lexer's status, so a mode it could
 * not honour still exited 0. --pp on Python is nonsense and must say so. */
static void smk_tbad(void)
{
    int rc = th_run(BC_BIN " --triton --pp tests/tri_vadd.py", obuf, TH_BUFSZ);
    CHNE(rc, 0);
    PASS();
}
TH_REG("smoke", smk_tbad)
