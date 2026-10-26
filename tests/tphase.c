/* tphase.c -- phase output tests
 * Does each pipeline stage produce something resembling output? */

#include "tharns.h"

static char obuf[TH_BUFSZ];

/* ---- phase 2: line splicing ---- */

/* Fixtures are written here rather than committed. .gitattributes checks the
 * tree out as eol=lf, so a CRLF file in tests/ would arrive as an LF one and
 * every CRLF test below would pass without testing anything. */

#define SP_LF   "tests/spl_lf.cu"
#define SP_CRLF "tests/spl_crlf.cu"

static int sp_wr(const char *path, const char *body, int crlf)
{
    FILE *fp = fopen(path, "wb");
    if (!fp) return 0;
    for (const char *p = body; *p != '\0'; p++) {
        if (*p == '\n' && crlf) fputc('\r', fp);
        fputc(*p, fp);
    }
    fclose(fp);
    return 1;
}

/* Token count out of the "N tokens, M error(s)" tail, -1 if it isn't there. */
static int sp_ntok(const char *out)
{
    const char *p = strstr(out, " tokens,");
    if (p == NULL) return -1;
    const char *q = p;
    while (q > out && q[-1] >= '0' && q[-1] <= '9') q--;
    if (q == p) return -1;
    return atoi(q);
}

/* Lex body under both line endings. n[i] is the token count where the lexer
 * reported none of its own errors, -1 where it did. */
static void sp_both(const char *body, int n[2])
{
    static const char *const path[2] = { SP_LF, SP_CRLF };
    char cmd[512];
    for (int i = 0; i < 2; i++) {
        n[i] = -1;
        if (!sp_wr(path[i], body, i)) continue;
        snprintf(cmd, sizeof cmd, "%s --lex %s", BC_BIN, path[i]);
        th_run(cmd, obuf, TH_BUFSZ);
        if (strstr(obuf, "0 error(s)") != NULL)
            n[i] = sp_ntok(obuf);
    }
}

/* ---- phase: preprocessor ---- */

static void pha01(void)
{
    int rc = th_run(BC_BIN " --pp tests/vector_add.cu", obuf, TH_BUFSZ);
    CHEQ(rc, 0);
    CHECK(strlen(obuf) > 0);
    PASS();
}
TH_REG("pha", 1, "preprocessor runs", pha01)

static void pha02(void)
{
    int rc = th_run(BC_BIN " --pp tests/comment_macro_skip.cu", obuf, TH_BUFSZ);
    CHEQ(rc, 0);
    CHECK(strstr(obuf, "COMMENT_MACRO should remain literal") != NULL);
    CHECK(strstr(obuf, "23 should remain literal") == NULL);
    PASS();
}
TH_REG("pha", 2, "preprocessor comment opaque", pha02)

static void pha03(void)
{
    int rc = th_run(BC_BIN " --lex tests/comment_quotes.cu", obuf, TH_BUFSZ);
    CHEQ(rc, 0);
    CHECK(strstr(obuf, "0 error(s)") != NULL);
    PASS();
}
TH_REG("pha", 3, "lexer comment quotes", pha03)

/* ---- phase: parser (AST dump) ---- */

static void pha04(void)
{
    int rc = th_run(BC_BIN " --parse tests/vector_add.cu", obuf, TH_BUFSZ);
    CHEQ(rc, 0);
    PASS();
}
TH_REG("pha", 4, "parser dumps an AST", pha04)

/* A function pointer declarator used to parse as a call expression, which
   cost nothing at parse time and everything later. */
static void pha05(void)
{
    int rc = th_run(BC_BIN " --parse tests/fnptr.cu", obuf, TH_BUFSZ);
    CHEQ(rc, 0);
    CHECK(strstr(obuf, "0 parse error(s)") != NULL);
    CHECK(strstr(obuf, "(ident cb_t)") != NULL);
    CHECK(strstr(obuf, "(ident hook") != NULL);
    CHECK(strstr(obuf, "call") == NULL);
    PASS();
}
TH_REG("pha", 5, "function pointers survive the parser", pha05)

/* Nested structs matter here: the enclosing name is saved and restored,
   so inner's constructor must not be mistaken for outer's. */
static void pha06(void)
{
    int rc = th_run(BC_BIN " --parse tests/ctor.cu", obuf, TH_BUFSZ);
    CHEQ(rc, 0);
    CHECK(strstr(obuf, "0 parse error(s)") != NULL);
    CHECK(strstr(obuf, "(ident vec3)") != NULL);
    CHECK(strstr(obuf, "(ident inner)") != NULL);
    CHECK(strstr(obuf, "(ident sum)") != NULL);
    PASS();
}
TH_REG("pha", 6, "constructors survive the parser", pha06)

/* ---- phase: IR ---- */

static void pha07(void)
{
    int rc = th_run(BC_BIN " --ir tests/vector_add.cu", obuf, TH_BUFSZ);
    CHEQ(rc, 0);
    CHECK(strstr(obuf, "func") != NULL);
    PASS();
}
TH_REG("pha", 7, "IR comes out", pha07)

/* ---- phase: semantic analysis ---- */

static void pha08(void)
{
    int rc = th_run(BC_BIN " --sema tests/vector_add.cu", obuf, TH_BUFSZ);
    CHEQ(rc, 0);
    PASS();
}
TH_REG("pha", 8, "semantic analysis runs", pha08)

/* ---- phase 2: line splicing ---- */

/* A splice is invisible: the same source with and without one has to give the
 * same tokens, and CRLF has to agree with LF on both. */
static void pha09(void)
{
    int spl[2], ref[2];
    sp_both("__device__ int f(int v)\n{\n    v = v \\\n        * 2;\n"
            "    return v;\n}\n", spl);
    sp_both("__device__ int f(int v)\n{\n    v = v * 2;\n"
            "    return v;\n}\n", ref);
    CHECK(ref[0] > 0);
    CHEQ(spl[0], ref[0]);
    CHEQ(spl[1], ref[0]);
    PASS();
}
TH_REG("pha", 9, "a splice leaves the token stream alone", pha09)

/* Phase 2 runs before tokenising, so it joins an identifier cut in half. */
static void pha10(void)
{
    int spl[2], ref[2];
    sp_both("__device__ int foobar(void) { return 7; }\n"
            "__device__ int g(void) { return foo\\\nbar(); }\n", spl);
    sp_both("__device__ int foobar(void) { return 7; }\n"
            "__device__ int g(void) { return foobar(); }\n", ref);
    CHECK(ref[0] > 0);
    CHEQ(spl[0], ref[0]);
    CHEQ(spl[1], ref[0]);
    PASS();
}
TH_REG("pha", 10, "a splice joins a split identifier", pha10)

/* And a string literal, which is why this cannot live in the lexer. */
static void pha11(void)
{
    int spl[2];
    char cmd[512];
    sp_both("__device__ const char *s(void) { return \"ab\\\ncd\"; }\n", spl);
    CHECK(spl[0] > 0);
    CHEQ(spl[1], spl[0]);
    snprintf(cmd, sizeof cmd, "%s --lex %s", BC_BIN, SP_CRLF);
    th_run(cmd, obuf, TH_BUFSZ);
    CHECK(strstr(obuf, "\"abcd\"") != NULL);
    PASS();
}
TH_REG("pha", 11, "a splice joins a string literal", pha11)

/* The continuation a multi-line macro is actually written with. */
static void pha12(void)
{
    char cmd[512];
    CHECK(sp_wr(SP_CRLF, "#define ADD(a,b) \\\n    ((a) + \\\n     (b))\n"
                         "__device__ int f(int x) { return ADD(x,2); }\n", 1));
    snprintf(cmd, sizeof cmd, "%s --pp %s", BC_BIN, SP_CRLF);
    th_run(cmd, obuf, TH_BUFSZ);
    CHECK(strstr(obuf, "((x) +") != NULL);
    CHECK(strstr(obuf, "(2))") != NULL);
    CHECK(strstr(obuf, "ADD") == NULL);
    PASS();
}
TH_REG("pha", 12, "a CRLF macro keeps its continuations", pha12)

/* The one that stays invisible until it bites. A splice must not eat a line
 * out of the buffer the renderer counts, or every diagnostic below one points
 * at the wrong place. The '@' here sits on physical line 6. */
static void pha13(void)
{
    static const char *const path[2] = { SP_LF, SP_CRLF };
    char cmd[512];
    for (int i = 0; i < 2; i++) {
        CHECK(sp_wr(path[i],
                    "__device__ int f(int v)\n{\n    v = v \\\n"
                    "        * 2 \\\n        + 1;\n    return v @;\n}\n", i));
        snprintf(cmd, sizeof cmd, "%s --lex %s", BC_BIN, path[i]);
        th_run(cmd, obuf, TH_BUFSZ);
        CHECK(strstr(obuf, "spl_") != NULL);
        CHECK(strstr(obuf, ".cu:6:15") != NULL);
        CHECK(strstr(obuf, "return v @;") != NULL);
    }
    PASS();
}
TH_REG("pha", 13, "a diagnostic below a splice keeps its line", pha13)

/* Only a backslash immediately before the newline splices. Trailing space, or
 * a CR with no LF behind it, is left alone and shows up as a stray backslash
 * rather than quietly joining two lines. */
static void pha14(void)
{
    int spc[2];
    char cmd[512];
    sp_both("__device__ int f(int v)\n{\n    v = v \\ \n        * 2;\n"
            "    return v;\n}\n", spc);
    CHEQ(spc[0], -1);
    CHEQ(spc[1], -1);
    CHECK(sp_wr(SP_LF, "__device__ int f(int v)\n{\n    v = v \\\r"
                       "        * 2;\n    return v;\n}\n", 0));
    snprintf(cmd, sizeof cmd, "%s --lex %s", BC_BIN, SP_LF);
    th_run(cmd, obuf, TH_BUFSZ);
    CHECK(strstr(obuf, "E005") != NULL);
    PASS();
}
TH_REG("pha", 14, "a backslash not against a newline is no splice", pha14)
/* ---- phase: declaration specifiers Booth does not model ---- */

/* _Noreturn parsed as a type name, so the void after it was a syntax error,
   and 61 of llama.cpp's 67 CUDA files start with one via GGML_NORETURN. */
static void pha15(void)
{
    int rc = th_run(BC_BIN " --parse tests/noretn.cu", obuf, TH_BUFSZ);
    CHEQ(rc, 0);
    CHECK(strstr(obuf, "0 parse error(s)") != NULL);
    CHECK(strstr(obuf, "(ident nrt_kern)") != NULL);
    CHECK(strstr(obuf, "(ident nrt_bump)") != NULL);
    PASS();
}
TH_REG("pha", 15, "_Noreturn and [[attributes]] are accepted", pha15)

/* ---- phase: preprocessor, again ---- */

/* ppcyc_a and ppcyc_b include each other. Without #pragma once the depth
   guard is all that stops them, and it stopped ggml-cuda too. */
static void pha16(void)
{
    int rc = th_run(BC_BIN " --pp tests/ppcycle.cu", obuf, TH_BUFSZ);
    CHEQ(rc, 0);
    CHECK(strstr(obuf, "error") == NULL);
    CHECK(strstr(obuf, "3 + 4") != NULL);
    PASS();
}
TH_REG("pha", 16, "#pragma once breaks an include cycle", pha16)

/* Three that travel together: a body kept its trailing // comment, an
   argument list that closed on a later line was never joined, and ... in a
   parameter list ended up in the body rather than naming __VA_ARGS__. */
static void pha17(void)
{
    int rc = th_run(BC_BIN " --pp tests/ppvarg.cu", obuf, TH_BUFSZ);
    CHEQ(rc, 0);
    CHECK(strstr(obuf, "...") == NULL);
    CHECK(strstr(obuf, "decl") == NULL);
    CHECK(strstr(obuf, "the hint is dropped") == NULL);
    CHECK(strstr(obuf, "int ppv_add(int a, int b);") != NULL);
    CHECK(strstr(obuf, ">> 2") != NULL);
    CHECK(strstr(obuf, "\"a, b\"") != NULL);
    PASS();
}
TH_REG("pha", 17, "variadic macros and multi-line calls", pha17)

/* The output buffer used to fill and keep going, unterminated, so the lexer
   read on into the source buffer that follows it and reported whatever it
   found there. The symptom was an unterminated block comment. */
static void pha18(void)
{
    int rc = th_run(BC_BIN " --nvidia-ptx tests/ppovfl.cu -o ppovfl_test.ptx",
                    obuf, TH_BUFSZ);
    CHNE(rc, 0);
    CHECK(strstr(obuf, "E053") != NULL);
    CHECK(strstr(obuf, "E002") == NULL);
    CHECK(strstr(obuf, "E020") == NULL);
    remove("ppovfl_test.ptx");
    PASS();
}
TH_REG("pha", 18, "a truncated expansion is diagnosed", pha18)
