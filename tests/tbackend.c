/* tbackend.c -- backend descriptor contract checks. Broken
 * descriptor at test time beats broken kernel at runtime. */

#include "tharns.h"
#include "backend.h"
#include <string.h>

static void bkd01(void)
{
    uint32_t n = 0;
    for (uint32_t i = 0; be_list[i] != NULL && i < 64; i++) {
        const be_desc_t *b = be_list[i];
        CHECK(b->name != NULL && b->name[0] != '\0');
        CHECK(b->is_on != NULL);
        CHECK(b->isel  != NULL);
        CHECK(b->emit  != NULL);
        n++;
    }
    CHECK(n >= 8);
    PASS();
}
TH_REG("bkd", 1, "list ok", bkd01)

static void bkd02(void)
{
    static const char * const known[] = {
        "amdgpu", "nvptx", "tensix", "tensix-rv32",
        "metal", "intel-spirv", "cpu-x86-64", "cpu-rv64"
    };
    for (uint32_t i = 0; i < sizeof(known)/sizeof(known[0]); i++) {
        const be_desc_t *b = be_find(known[i]);
        CHECK(b != NULL);
        CHSTR(b->name, known[i]);
    }
    PASS();
}
TH_REG("bkd", 2, "find hits", bkd02)

static void bkd03(void)
{
    CHECK(be_find(NULL) == NULL);
    CHECK(be_find("") == NULL);
    CHECK(be_find("this-is-not-a-real-backend") == NULL);
    PASS();
}
TH_REG("bkd", 3, "find missing", bkd03)

static void bkd04(void)
{
    const uint32_t all =
        BE_F_SIMT | BE_F_SCALAR | BE_F_ATOMIC | BE_F_SHARED |
        BE_F_WARP | BE_F_MFMA | BE_F_BARRIER | BE_F_DIV |
        BE_F_SCRATCH | BE_F_TRANSC | BE_F_F16 | BE_F_F64 |
        BE_F_BF16 | BE_F_MULTIOUT | BE_F_NOCALL | BE_F_BYTES;
    for (uint32_t i = 0; be_list[i] != NULL && i < 64; i++) {
        CHEQ(be_list[i]->feats & ~all, 0u);
    }
    PASS();
}
TH_REG("bkd", 4, "feats sane", bkd04)

/* ---- one target per run ----
 * Every backend emits to cfg->output_file, so several at once used to
 * write over each other and leave whichever sorted last in be_list. */

static char be_obuf[TH_BUFSZ];

static void bkd05(void)
{
    int rc = th_run(BC_BIN " --amdgpu-bin --nvidia-ptx --metal "
                    "tests/vector_add.cu -o be_multi.out",
                    be_obuf, TH_BUFSZ);
    CHNE(rc, 0);
    CHECK(strstr(be_obuf, "pick one") != NULL);
    /* and nothing written, so a build system cannot pick up the wrong one */
    FILE *f = fopen("be_multi.out", "r");
    CHECK(f == NULL);
    if (f) fclose(f);
    remove("be_multi.out");
    PASS();
}
TH_REG("bkd", 5, "one target", bkd05)

static void bkd06(void)
{
    int rc = th_run(BC_BIN " --nvidia-ptx tests/vector_add.cu -o be_single.ptx",
                    be_obuf, TH_BUFSZ);
    CHEQ(rc, 0);
    remove("be_single.ptx");
    PASS();
}
TH_REG("bkd", 6, "one target ok", bkd06)

/* ---- no two backends claim the same flag ----
 * Flags are declared in the descriptor rather than discovered, so a
 * collision is a static error we can catch here instead of a
 * first-past-the-post race decided by be_list order. */

static void bkd07(void)
{
    for (uint32_t i = 0; be_list[i] != NULL && i < 64; i++) {
        if (be_list[i]->flags == NULL) continue;
        for (uint32_t a = 0; be_list[i]->flags[a] != NULL; a++) {
            for (uint32_t j = i + 1; be_list[j] != NULL && j < 64; j++) {
                if (be_list[j]->flags == NULL) continue;
                for (uint32_t b = 0; be_list[j]->flags[b] != NULL; b++) {
                    CHECK(strcmp(be_list[i]->flags[a],
                                 be_list[j]->flags[b]) != 0);
                }
            }
        }
    }
    PASS();
}
TH_REG("bkd", 7, "flags unique", bkd07)

/* ---- a declared flag is a parsed flag ----
 * The list and parse() could drift apart, which would leave a flag
 * routed to a backend that then ignores it. Offering each one back
 * through the registry proves the pairing still holds. */

static void bkd08(void)
{
    for (uint32_t i = 0; be_list[i] != NULL && i < 64; i++) {
        if (be_list[i]->flags == NULL) continue;
        CHECK(be_list[i]->parse != NULL);
        CHECK(be_list[i]->opts_size <= BE_OPTS_MAX);
        for (uint32_t a = 0; be_list[i]->flags[a] != NULL; a++) {
            int used = 0;
            /* "1" as the value so a flag wanting one is satisfied */
            CHEQ(be_parse_flag(be_list[i]->flags[a], "1", &used), 1);
        }
    }
    PASS();
}
TH_REG("bkd", 8, "flags parse", bkd08)

/* ---- every backend flag is documented ----
 * Backends own their flags now, so --help can fall behind without
 * anything in the driver noticing. */

static char be_hbuf[TH_BUFSZ];

static void bkd09(void)
{
    th_run(BC_BIN " --help", be_hbuf, TH_BUFSZ);
    for (uint32_t i = 0; be_list[i] != NULL && i < 64; i++) {
        if (be_list[i]->flags == NULL) continue;
        for (uint32_t a = 0; be_list[i]->flags[a] != NULL; a++) {
            if (strstr(be_hbuf, be_list[i]->flags[a]) == NULL) {
                printf("\n    undocumented: %s\n", be_list[i]->flags[a]);
                CHECK(0);
            }
        }
    }
    PASS();
}
TH_REG("bkd", 9, "flags documented", bkd09)
