/* tphase.c -- phase output tests
 * Does each pipeline stage produce something resembling output? */

#include "tharns.h"

static char obuf[TH_BUFSZ];

/* ---- phase: preprocessor ---- */

static void pha_pp(void)
{
    int rc = th_run(BC_BIN " --pp tests/vector_add.cu", obuf, TH_BUFSZ);
    CHEQ(rc, 0);
    CHECK(strlen(obuf) > 0);
    PASS();
}
TH_REG("phase", pha_pp)

static void pha_pp_comment_opaque(void)
{
    int rc = th_run(BC_BIN " --pp tests/comment_macro_skip.cu", obuf, TH_BUFSZ);
    CHEQ(rc, 0);
    CHECK(strstr(obuf, "COMMENT_MACRO should remain literal") != NULL);
    CHECK(strstr(obuf, "23 should remain literal") == NULL);
    PASS();
}
TH_REG("phase", pha_pp_comment_opaque)

static void pha_lex_comment_quotes(void)
{
    int rc = th_run(BC_BIN " --lex tests/comment_quotes.cu", obuf, TH_BUFSZ);
    CHEQ(rc, 0);
    CHECK(strstr(obuf, "0 error(s)") != NULL);
    PASS();
}
TH_REG("phase", pha_lex_comment_quotes)

/* ---- phase: parser (AST dump) ---- */

static void pha_ast(void)
{
    int rc = th_run(BC_BIN " --parse tests/vector_add.cu", obuf, TH_BUFSZ);
    CHEQ(rc, 0);
    PASS();
}
TH_REG("phase", pha_ast)

/* ---- phase: IR ---- */

static void pha_ir(void)
{
    int rc = th_run(BC_BIN " --ir tests/vector_add.cu", obuf, TH_BUFSZ);
    CHEQ(rc, 0);
    CHECK(strstr(obuf, "func") != NULL);
    PASS();
}
TH_REG("phase", pha_ir)

/* ---- phase: semantic analysis ---- */

static void pha_sema(void)
{
    int rc = th_run(BC_BIN " --sema tests/vector_add.cu", obuf, TH_BUFSZ);
    CHEQ(rc, 0);
    PASS();
}
TH_REG("phase", pha_sema)
