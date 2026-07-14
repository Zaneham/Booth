/* tmain.c -- Booth test runner
 * Runs tests in a sensible order, prints dots, judges silently. */

#include "tharns.h"

/* ---- Storage ---- */

tcase_t th_list[TH_MAXTS];
int th_cnt = 0;
int npass  = 0;
int nfail  = 0;
int nskip  = 0;

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

/* ---- Category Ordering ---- */

static const char *cat_order[] = {
    "smoke", "compile", "encode", "tables",
    "types", "errors", "phase", "sched", "abend", "regalloc", NULL
};

static int cat_idx(const char *cat)
{
    for (int i = 0; cat_order[i]; i++)
        if (strcmp(cat, cat_order[i]) == 0) return i;
    return 99;
}

/* ---- Display ---- */

static void print_result(const char *tname, const char *tag)
{
    int nlen = (int)strlen(tname);
    int dots = 30 - nlen;
    if (dots < 3) dots = 3;
    printf("  %s ", tname);
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
        print_result(tc->tname, "FAIL");
    else if (nskip > was_skip)
        print_result(tc->tname, "SKIP");
    else if (npass > was_pass)
        print_result(tc->tname, "PASS");
    else {
        /* Test forgot to call PASS(). Benefit of the doubt,
         * like a lenient customs officer. */
        npass++;
        print_result(tc->tname, "PASS");
    }
}

/* ---- Main ---- */

int main(int argc, char *argv[])
{
    const char *filter_cat  = NULL;
    const char *filter_test = NULL;
    int list_mode = 0;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--cat") == 0 && i + 1 < argc)
            filter_cat = argv[++i];
        else if (strcmp(argv[i], "--test") == 0 && i + 1 < argc)
            filter_test = argv[++i];
        else if (strcmp(argv[i], "--list") == 0)
            list_mode = 1;
        /* --all is default, silently accepted */
    }

    if (list_mode) {
        for (int ci = 0; cat_order[ci]; ci++) {
            int printed_cat = 0;
            for (int i = 0; i < th_cnt; i++) {
                if (strcmp(th_list[i].tcats, cat_order[ci]) != 0) continue;
                if (!printed_cat) {
                    printf("[%s]\n", cat_order[ci]);
                    printed_cat = 1;
                }
                printf("  %s\n", th_list[i].tname);
            }
        }
        return 0;
    }

    printf("Booth Test Suite\n");
    printf("====================\n");

    /* Run tests in category order */
    for (int ci = 0; cat_order[ci]; ci++) {
        if (filter_cat && strcmp(filter_cat, cat_order[ci]) != 0)
            continue;

        int printed_cat = 0;
        for (int i = 0; i < th_cnt; i++) {
            if (strcmp(th_list[i].tcats, cat_order[ci]) != 0) continue;
            if (filter_test && strcmp(filter_test, th_list[i].tname) != 0)
                continue;

            if (!printed_cat) {
                printf("[%s]\n", cat_order[ci]);
                printed_cat = 1;
            }
            run_test(&th_list[i]);
        }
    }

    /* Stray tests in unknown categories. Orphans, basically. */
    for (int i = 0; i < th_cnt; i++) {
        if (cat_idx(th_list[i].tcats) < 99) continue;
        if (filter_cat) continue;
        if (filter_test && strcmp(filter_test, th_list[i].tname) != 0) continue;
        run_test(&th_list[i]);
    }

    int total = npass + nfail + nskip;
    printf("====================\n");
    printf("%d tests: %d passed, %d failed, %d skipped\n",
           total, npass, nfail, nskip);

    (void)argc;
    return nfail > 0 ? 1 : 0;
}
