/* tocm.c -- the OCaml frontend, from .cmt to BIR to a backend.
 *
 * Needs kcomp built, which needs ocamlc, so these skip rather than fail where
 * it is absent. Regenerate the golden files with tests/ocaml/regen.sh. */

#include "tharns.h"

#ifdef _WIN32
#define KCOMP ".\\src\\ocaml\\_build\\default\\kcomp.exe"
#else
#define KCOMP "./src/ocaml/_build/default/kcomp.exe"
#endif

#define CMTDIR "src/ocaml/_build/default/.kernels.objs/byte"

static char got[1 << 16];
static char want[1 << 16];

static int have_kcomp(void)
{
    return th_exist("src/ocaml/_build/default/kcomp.exe") &&
           th_exist(CMTDIR "/vadd_k.cmt");
}

static int slurp(const char *path, char *buf, int sz)
{
    FILE *f = fopen(path, "rb");
    if (f == NULL) return 0;
    size_t n = fread(buf, 1, (size_t)sz - 1, f);
    buf[n] = '\0';
    fclose(f);
    return 1;
}

/* CRLF on one side and LF on the other is not a difference worth failing. */
static void strip_cr(char *s)
{
    char *w = s;
    for (const char *r = s; *r != '\0'; r++)
        if (*r != '\r') *w++ = *r;
    *w = '\0';
}

/* Lower one kernel and leave the text in got. */
static int lower(const char *kern)
{
    char cmd[512], out[256];

    snprintf(out, sizeof out, "build/ocm_%s.bir", kern);
    snprintf(cmd, sizeof cmd, "%s %s/%s.cmt -o %s", KCOMP, CMTDIR, kern, out);
    if (th_run(cmd, got, (int)sizeof got) != 0) return 0;
    if (!slurp(out, got, (int)sizeof got)) return 0;
    strip_cr(got);
    return 1;
}

static int golden(const char *kern)
{
    char path[256];
    snprintf(path, sizeof path, "tests/ocaml/%s.bir", kern);
    if (!slurp(path, want, (int)sizeof want)) return 0;
    strip_cr(want);
    return 1;
}

static void check_kernel(const char *kern)
{
    if (!have_kcomp()) { nskip++; printf("  SKIP: kcomp not built\n"); return; }
    if (!lower(kern))  { printf("  FAIL: %s did not lower\n", kern); nfail++; return; }
    if (!golden(kern)) { printf("  FAIL: no golden for %s\n", kern); nfail++; return; }
    if (strcmp(got, want) != 0) {
        printf("  FAIL: %s differs from tests/ocaml/%s.bir\n", kern, kern);
        nfail++;
        return;
    }
    npass++;
}

static void ocm01(void) { check_kernel("vadd_k"); }
TH_REG("ocm", 1, "vadd lowers to the expected module", ocm01)

static void ocm02(void) { check_kernel("scale_k"); }
TH_REG("ocm", 2, "float literals reach BIR as immediates", ocm02)

static void ocm03(void) { check_kernel("reduce_k"); }
TH_REG("ocm", 3, "a loop and a ref lower to alloca", ocm03)

static void ocm04(void) { check_kernel("clamp_k"); }
TH_REG("ocm", 4, "if/else lowers with an fcmp", ocm04)

/* The frontend numbers values as it emits and the printer numbers by position,
 * so a module that reads back unchanged is the proof the two agree. */
static void ocm05(void)
{
    static const char *const ks[] = { "vadd_k", "scale_k", "reduce_k",
                                     "clamp_k", "ops_k", "tile_k", "rng_k",
                                     "ints_k", "atom_k", NULL };
    char cmd[512];

    if (!have_kcomp()) SKIP("kcomp not built");

    for (int i = 0; ks[i] != NULL; i++) {
        CHNE(lower(ks[i]), 0);
        snprintf(cmd, sizeof cmd,
                 "%s --bir-in --ir --no-mem2reg --no-cfold --no-dce --no-sroa "
                 "build/ocm_%s.bir", BC_BIN, ks[i]);
        CHEQ(th_run(cmd, want, (int)sizeof want), 0);
        strip_cr(want);

        char *tail = strrchr(want, '}');
        CHNE(tail, NULL);
        tail[1] = '\0';

        char *gtail = strrchr(got, '}');
        CHNE(gtail, NULL);
        gtail[1] = '\0';

        CHEQ(strcmp(got, want), 0);
    }
    PASS();
}
TH_REG("ocm", 5, "every kernel round-trips through kath", ocm05)

/* mem2reg has to promote what the frontend emits, or the ref stays in memory
 * on the device and the whole alloca approach is wrong. */
static void ocm06(void)
{
    char cmd[512];

    if (!have_kcomp()) SKIP("kcomp not built");

    CHNE(lower("reduce_k"), 0);
    CHNE(strstr(got, "alloca"), NULL);

    snprintf(cmd, sizeof cmd, "%s --bir-in --ir build/ocm_reduce_k.bir", BC_BIN);
    CHEQ(th_run(cmd, want, (int)sizeof want), 0);
    CHEQ(strstr(want, "alloca"), NULL);
    CHNE(strstr(want, "phi"), NULL);
    PASS();
}
TH_REG("ocm", 6, "mem2reg promotes what the frontend emits", ocm06)

static void ocm07(void)
{
    char cmd[512];

    if (!have_kcomp()) SKIP("kcomp not built");

    CHNE(lower("reduce_k"), 0);
    snprintf(cmd, sizeof cmd,
             "%s --bir-in --nvidia-ptx build/ocm_reduce_k.bir -o build/ocm_reduce.ptx",
             BC_BIN);
    CHEQ(th_run(cmd, want, (int)sizeof want), 0);
    CHNE(th_exist("build/ocm_reduce.ptx"), 0);
    CHNE(slurp("build/ocm_reduce.ptx", want, (int)sizeof want), 0);
    CHNE(strstr(want, ".entry reduce"), NULL);
    PASS();
}
TH_REG("ocm", 7, "an OCaml kernel reaches PTX", ocm07)

/* Division, bitwise, conversion and the transcendentals in one kernel. The
 * float literal also catches OCaml's %g writing e+006 where C writes e+06. */
static void ocm08(void) { check_kernel("ops_k"); }
TH_REG("ocm", 8, "arithmetic and intrinsics lower correctly", ocm08)

/* Shared memory is per block, so mem2reg must leave it where it is even
 * though it looks like any other allocation from the outside. */
static void ocm09(void)
{
    char cmd[512];

    if (!have_kcomp()) SKIP("kcomp not built");
    check_kernel("tile_k");
    if (nfail) return;

    snprintf(cmd, sizeof cmd, "%s --bir-in --ir build/ocm_tile_k.bir", BC_BIN);
    CHEQ(th_run(cmd, want, (int)sizeof want), 0);
    CHNE(strstr(want, "shared_alloc ptr<shared, [256 x f32]>"), NULL);
    CHNE(strstr(want, "barrier"), NULL);
}
TH_REG("ocm", 9, "shared memory survives the passes", ocm09)

/* Device functions plus a kernel that allocates. total_insts was the running
 * module count rather than this function's own, so mem2reg's compaction moved
 * the wrong number of instructions and spliced one body into the next. Metal
 * saw it first because its structurer refuses a block with no terminator. */
static void ocm10(void)
{
    static const char *const backends[] = {
        "--nvidia-ptx", "--cpu", "--amdgpu", "--metal", "--rv64", NULL
    };
    static const char *const refuse[][2] = {
        { "--tensix", "E522" }, { NULL, NULL }
    };
    char cmd[512];

    if (!have_kcomp()) SKIP("kcomp not built");
    check_kernel("rng_k");
    if (nfail) return;

    for (int i = 0; backends[i] != NULL; i++) {
        snprintf(cmd, sizeof cmd, "%s --bir-in %s build/ocm_rng_k.bir -o build/ocm_rng.out",
                 BC_BIN, backends[i]);
        CHEQ(th_run(cmd, want, (int)sizeof want), 0);
    }

    for (int i = 0; refuse[i][0] != NULL; i++) {
        snprintf(cmd, sizeof cmd, "%s --bir-in %s build/ocm_rng_k.bir -o build/ocm_rng.out",
                 BC_BIN, refuse[i][0]);
        th_run(cmd, want, (int)sizeof want);
        CHNE(strstr(want, refuse[i][1]), NULL);
    }
}
TH_REG("ocm", 10, "device functions reach every backend", ocm10)

/* get and set take the element type from the array. They were pinned to f32,
 * so an integer array could be declared and passed but never read. */
static void ocm11(void)
{
    if (!have_kcomp()) SKIP("kcomp not built");
    check_kernel("ints_k");
    if (nfail) return;
    CHNE(strstr(got, "shared_alloc ptr<shared, [64 x i32]>"), NULL);
    CHNE(strstr(got, "gep ptr<shared, i32>"), NULL);
    CHNE(strstr(got, "gep ptr<global, i32>"), NULL);
    CHNE(strstr(got, "gep ptr<global, f32>"), NULL);
}
TH_REG("ocm", 11, "arrays carry their own element type", ocm11)

/* Atomics carry the element type too, and min and max stay out of reach
 * while BIR has one opcode for each and the backends disagree on its sign. */
static void ocm12(void)
{
    if (!have_kcomp()) SKIP("kcomp not built");
    check_kernel("atom_k");
    if (nfail) return;
    CHNE(strstr(got, "atomic_add relaxed i32"), NULL);
    CHNE(strstr(got, "atomic_add relaxed f32"), NULL);
    CHNE(strstr(got, "atomic_xchg relaxed i32"), NULL);
    CHEQ(strstr(got, "atomic_max"), NULL);
    CHEQ(strstr(got, "atomic_min"), NULL);
}
TH_REG("ocm", 12, "atomics lower, min and max stay out", ocm12)
