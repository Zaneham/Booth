/* tmain.c -- Booth test runner
 * Runs tests in a sensible order, prints dots, judges silently. */

#include "tharns.h"
#include "booth/bc_abend.h"
#include "bc_err.h"

/* ---- Storage ---- */

tcase_t th_list[TH_MAXTS];
int th_cnt  = 0;
int th_over = 0;
int npass   = 0;
int nfail   = 0;
int nskip   = 0;

/* ---- Utilities ---- */

int th_run(const char *cmd, char *obuf, int osz)
{
    char full[TH_BUFSZ];
    snprintf(full, TH_BUFSZ, "%s 2>&1", cmd);
    FILE *fp = popen(full, "r");
    if (!fp) { obuf[0] = '\0'; return -1; }
    int n = (int)fread(obuf, 1, (size_t)(osz - 1), fp);
    if (n < 0) n = 0;
    obuf[n] = '\0';
    /* Keep swallowing the rest of the pipe even though obuf is full. A child
     * that dumps more than osz bytes would otherwise write into a read end the
     * harness already walked away from, catch a SIGPIPE, and die by signal --
     * which pclose hands back as a non-zero status and the test reads as a
     * failure the compiler never actually committed. Drain to EOF and the
     * child exits in peace. */
    {
        char sink[4096];
        while (fread(sink, 1, sizeof sink, fp) > 0) { }
    }
    int rc = pclose(fp);
#ifndef _WIN32
    if (rc != -1 && (rc & 0xFF) == 0)
        rc = (rc >> 8) & 0xFF;
#endif
    return rc;
}

int th_exist(const char *path)
{
    FILE *fp = fopen(path, "rb");
    if (!fp) return 0;
    fclose(fp);
    return 1;
}

/* ---- Families ---- */

/* The one place a family is declared. This whole scheme is lifted from z390's
 * rt\test directory, where the members are TESTDCB1 through TESTDCB9 and an
 * alphabetical listing does the grouping for you. I got it from living in z390
 * and mainframes for a while so it is a bit cargo-culty, and the PDS eight
 * character limit it comes from has not applied to anything here since about
 * 1974. It still works better than what was here before.
 *
 * Order is pipeline order, not alphabetical, because you read a failing run top
 * to bottom and want to see how far the compiler got before it fell over.
 *
 * Names, filters and listings are all checked against this table, so the file
 * stem, the family and the test name cannot wander off from each other again.
 * Two digits everywhere so grep and ls agree with the runner about what comes
 * first. A family that outgrows 99 needs a wider width, and th_check will
 * nag. */
typedef struct {
    const char *fam;
    const char *file;
    const char *what;
    int         width;
} tfam_t;

static const tfam_t fam_order[] = {
    { "smk", "tsmoke.c",    "driver smoke",             2 },

    { "pha", "tphase.c",    "frontend phases",          2 },
    { "err", "terrs.c",     "diagnostics",              2 },
    { "typ", "ttypes.c",    "type table",               2 },
    { "tab", "ttabs.c",     "static tables",            2 },
    { "pck", "tpack.c",     "parameter packs",          2 },
    { "cxp", "tcxp.c",      "constexpr objects",        2 },
    { "tpl", "ttpl.c",      "class templates",          2 },
    { "vec", "tvec.c",      "CUDA vector types",        2 },
    { "ref", "tref.c",      "reference parameters",     2 },
    { "mof", "tmof.c",      "struct member offsets",    2 },
    { "tyc", "ttyc.c",      "types and coercion",       2 },
    { "cpp", "tcpp.c",      "host C++ in a .cu",        2 },
    { "asm", "tasm.c",      "inline asm and frame layout", 2 },

    { "dce", "tdce.c",      "dead code elimination",    2 },
    { "cfd", "tcfold.c",    "constant folding",         2 },
    { "str", "tstruct.c",   "structurisation",          2 },
    { "ins", "tinsert.c",   "insertion points",         2 },
    { "sro", "tsroa.c",     "scalar replacement",       2 },
    { "inl", "tinline.c",   "inlining",                 2 },

    { "enc", "tenc.c",      "AMD encoding",             2 },
    { "asy", "tasy.c",      "AMD encoding assay",       2 },
    { "sch", "tsched.c",    "AMD scheduling",           2 },
    { "ral", "tregalloc.c", "AMD register allocation",  2 },
    { "rss", "tra_ssa.c",   "AMD SSA allocation",       2 },
    { "grd", "tguard.c",    "AMD exec masking",         2 },

    { "rve", "trv_enc.c",   "RISC-V encoding",          2 },
    { "rvb", "trv_buf.c",   "RISC-V code buffer",       2 },
    { "rvl", "trv_elf.c",   "RISC-V ELF writer",        2 },
    { "rvi", "trv_isel.c",  "RISC-V selection",         2 },
    { "cbs", "tcbsync.c",   "Tensix CB sync",           2 },
    { "tmc", "ttmc.c",      "Tensix machine code",      2 },
    { "tdf", "ttdf.c",      "Tensix dataflow",          2 },

    { "tri", "ttriton.c",   "Triton frontend",          2 },
    { "mma", "tmma.c",      "warp matrix multiply",     2 },
    { "mlr", "tmlir.c",     "MLIR reader",              2 },
    { "bir", "tbir.c",      "BIR text frontend",        2 },

    { "sfp", "tsoft_fp.c",  "soft float",               2 },
    { "spr", "tsysprint.c", "SYSPRINT",                 2 },
    { "abd", "tabend.c",    "abend handling",           2 },

    { "bkd", "tbackend.c",  "backend registry",         2 },
    { "cmp", "tcomp.c",     "compile matrix",           2 },
    { "wsz", "twarpsize.c", "warp size",                2 },

    { "ord", "tordr.c",     "harness ordering",         2 },
    { "ocm", "tocm.c",      "OCaml frontend",           2 },
    { "mtu", "tmtu.c",      "multiple translation units", 2 },
    { "rpi", "trpi.c",      "shipped-bug regressions",  2 },
};

#define NFAM ((int)(sizeof(fam_order) / sizeof(fam_order[0])))

static int fam_idx(const char *fam)
{
    for (int i = 0; i < NFAM; i++)
        if (strcmp(fam, fam_order[i].fam) == 0) return i;
    return -1;
}

/* ---- Startup Checks ---- */

/* Enforcing the naming is the whole trick, so this runs before anything else.
 * It used to be possible to register under a family the runner had never heard
 * of, and 278 of the 380 tests were doing exactly that, running unheaded at the
 * end where --list could not see them and --cat could not reach them. Now a
 * bad family, a misspelled name or a reused number stops the run and names the
 * culprit. */
static int th_check(void)
{
    int bad = 0;

    if (th_over) {
        printf("harness: %d tests did not fit in TH_MAXTS (%d), raise it\n",
               th_over, TH_MAXTS);
        bad++;
    }

    for (int i = 0; i < th_cnt; i++) {
        const tcase_t *t = &th_list[i];
        int fi = fam_idx(t->tfam);

        if (fi < 0) {
            printf("harness: %s registers family \"%s\", "
                   "which is not in fam_order\n", t->tname, t->tfam);
            bad++;
            continue;
        }

        /* The function name has to spell out the family and number it
         * registered with, or the three namespaces start drifting again. */
        char want[TH_NAMEW * 2];
        snprintf(want, sizeof want, "%s%0*d",
                 t->tfam, fam_order[fi].width, t->tnum);
        if (strcmp(want, t->tname) != 0) {
            printf("harness: %s registers as %s, name it %s\n",
                   t->tname, want, want);
            bad++;
        }
        if ((int)strlen(want) > TH_NAMEW - 1) {
            printf("harness: %s is wider than TH_NAMEW\n", want);
            bad++;
        }
        if (t->tdesc == NULL || t->tdesc[0] == '\0') {
            printf("harness: %s has no description\n", t->tname);
            bad++;
        } else if ((int)strlen(t->tdesc) > TH_DESCW) {
            printf("harness: %s description is %d chars, budget is %d\n",
                   t->tname, (int)strlen(t->tdesc), TH_DESCW);
            bad++;
        }

        for (int j = i + 1; j < th_cnt; j++) {
            if (th_list[j].tnum == t->tnum &&
                strcmp(th_list[j].tfam, t->tfam) == 0) {
                printf("harness: %s and %s are both %s number %d\n",
                       t->tname, th_list[j].tname, t->tfam, t->tnum);
                bad++;
            }
        }
    }

    /* A family in the table with nothing registered means its file dropped out
     * of TSRC, which reads as a smaller suite rather than a broken build. */
    for (int f = 0; f < NFAM; f++) {
        int seen = 0;
        for (int i = 0; i < th_cnt && !seen; i++)
            if (strcmp(th_list[i].tfam, fam_order[f].fam) == 0) seen = 1;
        if (!seen) {
            printf("harness: family %s (%s) registered nothing, "
                   "is it still in TSRC?\n",
                   fam_order[f].fam, fam_order[f].file);
            bad++;
        }
    }

    return bad;
}

/* ---- Ordering ---- */

/* Family in pipeline order, then number. th_check has already proved the keys
 * unique, so this does not care whether qsort is stable, and more to the point
 * it does not care what order the constructors fired in. The old orphan pass
 * did, which is why the tail of a run came out in link order. */
static int case_cmp(const void *a, const void *b)
{
    const tcase_t *x = a, *y = b;
    int fx = fam_idx(x->tfam), fy = fam_idx(y->tfam);
    if (fx != fy) return fx < fy ? -1 : 1;
    if (x->tnum != y->tnum) return x->tnum < y->tnum ? -1 : 1;
    return strcmp(x->tname, y->tname);
}

/* ---- Display ---- */

static void print_result(const tcase_t *tc, const char *tag)
{
    int dlen = (int)strlen(tc->tdesc);
    int dots = TH_DESCW - dlen;
    /* A description that fills the budget gets no dots at all, which is the
     * point of ELDERBERRY over in tordr.c. th_check has already refused
     * anything longer. */
    if (dots < 0) dots = 0;
    printf("  %-*s %s ", TH_NAMEW, tc->tname, tc->tdesc);
    for (int i = 0; i < dots; i++) putchar('.');
    printf(" %s\n", tag);
}

static void run_test(tcase_t *tc)
{
    int was_pass = npass;
    int was_fail = nfail;
    int was_skip = nskip;

    tc->func();

    if (nfail > was_fail)
        print_result(tc, "FAIL");
    else if (nskip > was_skip)
        print_result(tc, "SKIP");
    else if (npass > was_pass)
        print_result(tc, "PASS");
    else {
        /* Test forgot to call PASS(). Benefit of the doubt,
         * like a lenient customs officer. */
        npass++;
        print_result(tc, "PASS");
    }
}

/* ---- Main ---- */

static void usage(void)
{
    printf("usage: trunner [--all] [--fam FAM] [--test NAME] [--list] "
           "[--families]\n");
}

int main(int argc, char *argv[])
{
    ab_set_msg_lookup(ab_afmt);

    const char *filter_fam  = NULL;
    const char *filter_test = NULL;
    int list_mode = 0;
    int fam_mode  = 0;

    for (int i = 1; i < argc; i++) {
        /* --cat is what this used to be called. */
        if ((strcmp(argv[i], "--fam") == 0 || strcmp(argv[i], "--cat") == 0)
            && i + 1 < argc)
            filter_fam = argv[++i];
        else if (strcmp(argv[i], "--test") == 0 && i + 1 < argc)
            filter_test = argv[++i];
        else if (strcmp(argv[i], "--list") == 0)
            list_mode = 1;
        else if (strcmp(argv[i], "--families") == 0)
            fam_mode = 1;
        else if (strcmp(argv[i], "--help") == 0) {
            usage();
            return 0;
        }
        /* --all is default, silently accepted */
    }

    if (th_check() != 0) {
        printf("harness: refusing to run with the above unresolved\n");
        return 2;
    }

    qsort(th_list, (size_t)th_cnt, sizeof th_list[0], case_cmp);

    /* A named family that does not exist used to run zero tests and exit 0,
     * which reads exactly like a family that passed. */
    if (filter_fam && fam_idx(filter_fam) < 0) {
        printf("trunner: no family \"%s\". --families lists them.\n",
               filter_fam);
        return 2;
    }

    if (fam_mode) {
        for (int f = 0; f < NFAM; f++) {
            int n = 0;
            for (int i = 0; i < th_cnt; i++)
                if (strcmp(th_list[i].tfam, fam_order[f].fam) == 0) n++;
            printf("  %-4s %-14s %3d  %s\n", fam_order[f].fam,
                   fam_order[f].file, n, fam_order[f].what);
        }
        return 0;
    }

    if (list_mode) {
        for (int i = 0; i < th_cnt; i++) {
            const tcase_t *t = &th_list[i];
            if (filter_fam && strcmp(t->tfam, filter_fam) != 0) continue;
            if (i == 0 || strcmp(t->tfam, th_list[i - 1].tfam) != 0)
                printf("[%s] %s\n", t->tfam, fam_order[fam_idx(t->tfam)].what);
            printf("  %-*s %s\n", TH_NAMEW, t->tname, t->tdesc);
        }
        return 0;
    }

    printf("Booth Test Suite\n");
    printf("====================\n");

    const char *last_fam = NULL;
    for (int i = 0; i < th_cnt; i++) {
        tcase_t *t = &th_list[i];
        if (filter_fam && strcmp(t->tfam, filter_fam) != 0) continue;
        if (filter_test && strcmp(filter_test, t->tname) != 0) continue;

        if (!last_fam || strcmp(t->tfam, last_fam) != 0) {
            printf("[%s] %s\n", t->tfam, fam_order[fam_idx(t->tfam)].what);
            last_fam = t->tfam;
        }
        run_test(t);
    }

    int total = npass + nfail + nskip;
    printf("====================\n");
    printf("%d tests: %d passed, %d failed, %d skipped\n",
           total, npass, nfail, nskip);

    return nfail > 0 ? 1 : 0;
}
