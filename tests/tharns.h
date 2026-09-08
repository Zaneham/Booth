/* tharns.h -- Booth test harness
 * Named after nobody in particular. Definitely not a typo. */
#ifndef THARNS_H
#define THARNS_H

#include <stdio.h>
#include <string.h>
#include <stdlib.h>

typedef void (*tfunc_t)(void);

/* Names are family plus number, PDS member style, so an alphabetical listing
 * groups and orders itself. I got this from living in z390 and mainframes for
 * a while, so it is a bit cargo-culty, but the listing really does sort itself
 * and nothing drifts. The name says nothing, so tdesc says it instead. */
typedef struct {
    const char *tname;
    const char *tfam;
    int         tnum;
    const char *tdesc;
    tfunc_t     func;
} tcase_t;

#define TH_MAXTS 1024
#define TH_BUFSZ 4096

/* Column widths for the result line. TH_DESCW is a budget, not a hint. A
 * description that overruns it is a description trying to become a comment. */
#define TH_NAMEW 8
#define TH_DESCW 46

extern tcase_t th_list[];
extern int th_cnt;
extern int th_over;
extern int npass, nfail, nskip;

/* ---- Self-Registration ---- */

/* Works on gcc and clang, which is everyone who matters
 * and several who don't. MSVC users: you know what you did. */

/* fam has to be in fam_order over in tmain.c and fn has to be spelled fam then
 * num, padded. th_check enforces both, which is VERY IMPORTANT, because a test
 * filed under a family nobody lists is a test that quietly stops running and
 * still reports green. Overflow gets counted rather than swallowed for the
 * same reason. */
#define TH_REG(fam, num, desc, fn) \
    __attribute__((constructor)) static void reg_##fn(void) { \
        if (th_cnt < TH_MAXTS) \
            th_list[th_cnt++] = (tcase_t){#fn, fam, num, desc, fn}; \
        else \
            th_over++; \
    }

/* ---- Assertions ---- */

/* The test stops here. No appeals. No severance package. */
#define CHECK(x) do { if (!(x)) { \
    printf("  FAIL %s:%d: %s\n", __FILE__, __LINE__, #x); \
    nfail++; return; } } while(0)

#define CHEQ(a, b)   CHECK((a) == (b))
#define CHNE(a, b)   CHECK((a) != (b))
#define CHSTR(a, b)  CHECK(strcmp((a),(b)) == 0)
#define PASS()       do { npass++; } while(0)
#define SKIP(r)      do { nskip++; printf("  SKIP: %s\n", r); return; } while(0)

/* ---- Hex check with diagnostic ---- */

#define CHEQX(a, b) do { \
    unsigned _a = (unsigned)(a), _b = (unsigned)(b); \
    if (_a != _b) { \
        printf("  FAIL %s:%d: 0x%08X != 0x%08X\n", __FILE__, __LINE__, _a, _b); \
        nfail++; return; \
    } } while(0)

/* ---- Binary Path ---- */

/* popen on Windows goes through cmd.exe, which has less understanding
 * of Unix paths than Grok has of being a real AI. */
#ifdef _WIN32
#define BC_BIN ".\\kath.exe"
#else
#define BC_BIN "./kath"
#endif

/* ---- Utilities ---- */

/* Run a shell command, capture stdout+stderr. Returns exit code. */
int th_run(const char *cmd, char *obuf, int osz);

/* Check if a file exists */
int th_exist(const char *path);

#endif /* THARNS_H */
