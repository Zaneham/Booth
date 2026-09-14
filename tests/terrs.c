/* terrs.c -- error handling
 * Making sure the compiler fails gracefully, not dramatically. */

#include "tharns.h"

static char obuf[TH_BUFSZ];

/* ---- errors: syntax ---- */

static void err01(void)
{
    int rc = th_run(BC_BIN " --amdgpu-bin tests/test_errors.cu -o err_test.hsaco",
                    obuf, TH_BUFSZ);
    CHNE(rc, 0);
    CHECK(strstr(obuf, "error") != NULL);
    remove("err_test.hsaco");
    PASS();
}
TH_REG("err", 1, "a syntax error is reported", err01)

/* ---- errors: missing file ---- */

static void err02(void)
{
    int rc = th_run(BC_BIN " --amdgpu-bin nonexistent_file_42.cu -o err_test.hsaco",
                    obuf, TH_BUFSZ);
    CHNE(rc, 0);
    remove("err_test.hsaco");
    PASS();
}
TH_REG("err", 2, "a missing file is reported", err02)

/* ---- errors: bad output directory ---- */
/* /dev/null is a file, not a directory. You can't mkdir inside it.
 * Works on every Unix. On Windows we use NUL, same idea. */

static void err03(void)
{
    char cmd[TH_BUFSZ];
#ifdef _WIN32
    snprintf(cmd, TH_BUFSZ,
             BC_BIN " --amdgpu-bin tests/vector_add.cu "
             "-o NUL\\impossible\\out.hsaco");
#else
    snprintf(cmd, TH_BUFSZ,
             BC_BIN " --amdgpu-bin tests/vector_add.cu "
             "-o /dev/null/impossible/out.hsaco");
#endif
    int rc = th_run(cmd, obuf, TH_BUFSZ);
    (void)rc;
    CHECK(strstr(obuf, "cannot open") != NULL ||
          strstr(obuf, "error") != NULL);
    PASS();
}
TH_REG("err", 3, "a bad output directory is reported", err03)

/* ---- errors: diagnostic rendering + real token text ----
 * A missing semicolon should render Clang-style (id in brackets, location
 * line, caret) and name the actual token it choked on, not the kind. */

static void err04(void)
{
    int rc = th_run(BC_BIN " --parse tests/test_diag.cu", obuf, TH_BUFSZ);
    (void)rc;
    CHECK(strstr(obuf, "error[E020]") != NULL);   /* id in brackets   */
    CHECK(strstr(obuf, "-->")         != NULL);   /* location line    */
    CHECK(strstr(obuf, "^")           != NULL);   /* the caret        */
    CHECK(strstr(obuf, "got 'p'")     != NULL);   /* real token text  */
    CHECK(strstr(obuf, "got 'IDENT'") == NULL);   /* not the kind name */
    PASS();
}
TH_REG("err", 4, "diagnostic renders real token text", err04)

/* ---- calls: past sixteen arguments ----
 * Jorge Galvez's ocean kernels pass 23. Sema stopped counting at 16 and then
 * reported an arity mismatch against a count it had made up. */

static void err05(void)
{
    int rc = th_run(BC_BIN " --nvidia-ptx tests/many_args.cu -o many_args.ptx",
                    obuf, TH_BUFSZ);
    CHEQ(rc, 0);
    CHECK(strstr(obuf, "error") == NULL);
    remove("many_args.ptx");
    PASS();
}
TH_REG("err", 5, "calls past sixteen arguments", err05)

/* ---- calls: past the cap ----
 * Overflowing has to say so. Silently dropping the tail would emit a call
 * with the wrong operands and nothing to show for it. */

static void err06(void)
{
    FILE *f = fopen("argcap_test.cu", "w");
    CHECK(f != NULL);
    fprintf(f, "__device__ double g(double a){return a;}\n");
    fprintf(f, "__global__ void k(double *o){ o[0] = g(0.0");
    for (int i = 1; i < 70; i++) fprintf(f, ",%d.0", i);
    fprintf(f, "); }\n");
    fclose(f);

    int rc = th_run(BC_BIN " --nvidia-ptx argcap_test.cu -o argcap_test.ptx",
                    obuf, TH_BUFSZ);
    CHNE(rc, 0);
    CHECK(strstr(obuf, "E082") != NULL);
    remove("argcap_test.cu");
    remove("argcap_test.ptx");
    PASS();
}
TH_REG("err", 6, "calls past the cap", err06)

/* ---- errors: sema errors are fatal ----
 * They used to be printed and then ignored by every mode but --sema, so the
 * backend ran on source we had already rejected and the exit status said the
 * compile went fine. */

static void err07(void)
{
    FILE *f = fopen("semafail_test.cu", "w");
    CHECK(f != NULL);
    fprintf(f, "__global__ void k(float *o){ o[0] = nosuchfn(1, 2); }\n");
    fclose(f);

    remove("semafail_test.ptx");
    int rc = th_run(BC_BIN " --nvidia-ptx semafail_test.cu -o semafail_test.ptx",
                    obuf, TH_BUFSZ);
    CHNE(rc, 0);
    /* and nothing written, so a build system cannot pick up a stale artefact */
    FILE *o = fopen("semafail_test.ptx", "r");
    CHECK(o == NULL);
    if (o) fclose(o);
    remove("semafail_test.cu");
    remove("semafail_test.ptx");
    PASS();
}
TH_REG("err", 7, "sema errors are fatal", err07)

/* ---- errors: a backend refusal is numbered ---- */

static void err08(void)
{
    static const char *const cmds[] = {
        BC_BIN " --amdgpu tests/mma16.cu -o build/err08.s",
        BC_BIN " --cpu tests/mma16.cu -o build/err08.o",
        BC_BIN " --rv64 tests/mma16.cu -o build/err08.o",
        BC_BIN " --nvidia-ptx tests/mfrg.cu -o build/err08.ptx",
        NULL
    };
    for (int i = 0; cmds[i] != NULL; i++) {
        CHNE(th_run(cmds[i], obuf, TH_BUFSZ), 0);
        CHECK(strstr(obuf, "E541") != NULL);
    }
    remove("build/err08.s");
    remove("build/err08.o");
    remove("build/err08.ptx");
    PASS();
}
TH_REG("err", 8, "every backend numbers its refusal", err08)
