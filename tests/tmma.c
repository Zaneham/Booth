/* tmma.c -- warp-collective matrix multiply
 * PTX gets mma.sync for every shape it knows, AMD gets MFMA in the register
 * tuple form llvm-mc accepts, and everything else says no out loud. */

#include "tharns.h"

static char obuf[TH_BUFSZ];
static char ptx[131072];
static unsigned char bin[65536];

static int mm_run(const char *args)
{
    char cmd[TH_BUFSZ];
    snprintf(cmd, TH_BUFSZ, BC_BIN " %s", args);
    return th_run(cmd, obuf, TH_BUFSZ);
}

static int mm_txt(const char *args, const char *path, char *buf, size_t cap)
{
    if (mm_run(args) != 0) return -1;
    FILE *fp = fopen(path, "rb");
    if (!fp) return -1;
    size_t n = fread(buf, 1, cap - 1, fp);
    fclose(fp);
    buf[n] = '\0';
    return 0;
}

static int mm_ptx(void)
{
    return mm_txt("--nvidia-ptx tests/mma16.cu -o mma16.ptx", "mma16.ptx",
                  ptx, sizeof ptx);
}

static int mm_cnt(const char *hay, const char *needle)
{
    int c = 0;
    const char *p = hay;
    while ((p = strstr(p, needle)) != NULL) { c++; p++; }
    return c;
}

/* first VOP3P-MAI word in an AMD code object, or -1 */
static int mm_mfop(const char *args, const char *path)
{
    if (mm_run(args) != 0) return -1;
    FILE *fp = fopen(path, "rb");
    if (!fp) return -1;
    size_t n = fread(bin, 1, sizeof bin, fp);
    fclose(fp);
    for (size_t o = 0; o + 8 <= n; o++) {
        unsigned w = (unsigned)bin[o] | (unsigned)bin[o+1] << 8 |
                     (unsigned)bin[o+2] << 16 | (unsigned)bin[o+3] << 24;
        if ((w >> 23) == 0x1A7u) return (int)((w >> 16) & 0x7Fu);
    }
    return -1;
}

/* ---- PTX ---- */

static void mma01(void)
{
    CHEQ(mm_ptx(), 0);
    CHEQ(mm_cnt(ptx, "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"), 2);
    CHEQ(mm_cnt(ptx, "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32"), 2);
    CHEQ(mm_cnt(ptx, "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32"), 2);
    CHEQ(mm_cnt(ptx, "mma.sync.aligned.m16n8k8.row.col.f32.bf16.bf16.f32"), 2);
    PASS();
}

/* k16 carries 4 A and 2 B registers, k8 carries 2 and 1. A tuple that lost
 * a register would still assemble, so count the commas. */
static void mma02(void)
{
    CHEQ(mm_ptx(), 0);
    static const struct { const char *sfx; int commas; } want[] = {
        { "m16n8k16.row.col.f32.f16.f16.f32",   13 },  /* 3 + 3+1+3 inside */
        { "m16n8k8.row.col.f32.f16.f16.f32",    10 },  /* 3 + 1+0+3 inside */
    };
    for (unsigned w = 0; w < sizeof want / sizeof want[0]; w++) {
        const char *p = strstr(ptx, want[w].sfx);
        CHECK(p != NULL);
        if (!p) return;
        const char *e = strchr(p, ';');
        CHECK(e != NULL);
        if (!e) return;
        int commas = 0, braces = 0;
        for (const char *q = p; q < e; q++) {
            if (*q == ',') commas++;
            if (*q == '{') braces++;
        }
        CHEQ(braces, 4);
        CHEQ(commas, want[w].commas);
    }
    PASS();
}

static void mma03(void)
{
    CHEQ(mm_ptx(), 0);
    CHECK(strstr(ptx, "%laneid") != NULL);
    CHECK(strstr(ptx, ".reg .b32") != NULL);
    PASS();
}

/* mma.sync is undefined on a divergent warp, so each kernel converges first. */
static void mma04(void)
{
    CHEQ(mm_ptx(), 0);
    CHEQ(mm_cnt(ptx, "bar.warp.sync 0xffffffff"), 4);
    PASS();
}

/* ---- AMD ---- */

/* CDNA3 ISA s7.1 wants contiguous, size-aligned operands. */
static void mma05(void)
{
    CHEQ(mm_txt("--amdgpu --gfx942 tests/mfrg.cu -o mfrg.s", "mfrg.s",
                ptx, sizeof ptx), 0);
    CHECK(strstr(ptx, "v_mfma_f32_16x16x16f16 v[208:211], v[240:241], "
                      "v[244:245], v[224:227]") != NULL);
    PASS();
}

/* The bf16 opcode moved between CDNA2 and CDNA3: 0x67 on gfx90a, 0x61 on
 * gfx942. One table for both targets emitted the wrong instruction. */
static void mma06(void)
{
    CHEQ(mm_mfop("--amdgpu-bin --gfx90a tests/mfbf.cu -o mfbf.hsaco",
                 "mfbf.hsaco"), 0x67);
    CHEQ(mm_mfop("--amdgpu-bin --gfx942 tests/mfbf.cu -o mfbf.hsaco",
                 "mfbf.hsaco"), 0x61);
    PASS();
}

static void mma07(void)
{
    CHEQ(mm_mfop("--amdgpu-bin --gfx942 tests/mfrg.cu -o mfrg.hsaco",
                 "mfrg.hsaco"), 0x4D);
    CHEQ(mm_mfop("--amdgpu-bin --gfx90a tests/mfrg.cu -o mfrg.hsaco",
                 "mfrg.hsaco"), 0x4D);
    PASS();
}

/* i8 16x16x16 is CDNA2 only; CDNA3 spells it 16x16x32. */
static void mma08(void)
{
    CHNE(mm_run("--amdgpu --gfx942 tests/mfi8.cu -o mfi8.s"), 0);
    CHECK(strstr(obuf, "E541") != NULL);
    CHEQ(mm_mfop("--amdgpu-bin --gfx90a tests/mfi8.cu -o mfi8.hsaco",
                 "mfi8.hsaco"), 0x55);
    PASS();
}

/* ---- Refusals ---- */

static void mma09(void)
{
    CHNE(mm_run("--amdgpu --gfx942 tests/mma16.cu -o x.s"), 0);
    CHECK(strstr(obuf, "E541") != NULL);
    CHNE(mm_run("--cpu tests/mma16.cu -o x.o"), 0);
    CHECK(strstr(obuf, "E541") != NULL);
    PASS();
}

static void mma10(void)
{
    CHNE(mm_run("--nvidia-ptx tests/mfrg.cu -o x.ptx"), 0);
    CHECK(strstr(obuf, "E541") != NULL);
    CHNE(mm_run("--amdgpu --gfx942 tests/test_mfma.cu -o x.s"), 0);
    CHECK(strstr(obuf, "E541") != NULL);
    PASS();
}

TH_REG("mma",  1, "every mma.sync shape reaches PTX",    mma01)
TH_REG("mma",  2, "fragment tuples are the right width", mma02)
TH_REG("mma",  3, "lane picks its own fragment",         mma03)
TH_REG("mma",  4, "the warp converges before mma.sync",  mma04)
TH_REG("mma",  5, "MFMA emits an aligned register tuple", mma05)
TH_REG("mma",  6, "bf16 opcode follows the CDNA target", mma06)
TH_REG("mma",  7, "f16 MFMA encodes the same on both",   mma07)
TH_REG("mma",  8, "i8 16x16x16 is CDNA2 only",           mma08)
TH_REG("mma",  9, "warp-collective mma refuses off PTX", mma09)
TH_REG("mma", 10, "AMD matrix ops refuse off CDNA",      mma10)
