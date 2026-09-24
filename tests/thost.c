#include "tharns.h"

static char obuf[TH_BUFSZ];
static char hbuf[65536];

static int slurp(const char *path)
{
    FILE  *f = fopen(path, "rb");
    size_t n;

    hbuf[0] = '\0';
    if (f == NULL) return 0;
    n = fread(hbuf, 1u, sizeof hbuf - 1u, f);
    fclose(f);
    hbuf[n] = '\0';
    return n != 0u;
}

static int emit(const char *cu, const char *out)
{
    char cmd[512];

    snprintf(cmd, sizeof cmd,
             BC_BIN " --cuda-host -Iruntime/include %s -o %s", cu, out);
    if (th_run(cmd, obuf, TH_BUFSZ) != 0) return 0;
    return slurp(out);
}

static void hst01(void)
{
    CHECK(emit("tests/hvadd.cu", "tests/hst01.cpp"));
    CHECK(strstr(hbuf, "__global__") == NULL);
    CHECK(strstr(hbuf, "void vadd(") != NULL);
    remove("tests/hst01.cpp");
    PASS();
}
TH_REG("hst", 1, "the device definition leaves and a stub stays", hst01)

static void hst02(void)
{
    CHECK(emit("tests/hvadd.cu", "tests/hst02.cpp"));
    CHECK(strstr(hbuf, "__cudaRegisterFatBinary") != NULL);
    CHECK(strstr(hbuf, "__cudaRegisterFunction") != NULL);
    CHECK(strstr(hbuf, "\"vadd\"") != NULL);
    CHECK(strstr(hbuf, "__cudaRegisterFatBinaryEnd") != NULL);
    remove("tests/hst02.cpp");
    PASS();
}
TH_REG("hst", 2, "a constructor registers fatbin and kernel", hst02)

static void hst03(void)
{
    CHECK(emit("tests/hvadd.cu", "tests/hst03.cpp"));
    CHECK(strstr(hbuf, "__cudaPushCallConfiguration") != NULL);
    CHECK(strstr(hbuf, "cudaLaunchKernel") != NULL);
    CHECK(strstr(hbuf, "<<<") == NULL);
    remove("tests/hst03.cpp");
    PASS();
}
TH_REG("hst", 3, "a launch goes through the registration ABI", hst03)

static void hst04(void)
{
    CHECK(emit("tests/hmulti.cu", "tests/hst04.cpp"));
    CHECK(strstr(hbuf, "\"vfill\"") != NULL);
    CHECK(strstr(hbuf, "\"vfill$1\"") != NULL);
    CHECK(strstr(hbuf, "\"vadd\"") != NULL);
    CHECK(strstr(hbuf, "\"vdbl\"") != NULL);
    remove("tests/hst04.cpp");
    PASS();
}
TH_REG("hst", 4, "both template instantiations reach the table", hst04)

static void hst05(void)
{
    CHECK(emit("tests/hvadd.cu", "tests/hst05.cpp"));
    CHECK(strstr(hbuf, "#include <stdio.h>") != NULL);
    remove("tests/hst05.cpp");
    PASS();
}
TH_REG("hst", 5, "an include Booth cannot read stays a directive", hst05)

static void hst06(void)
{
    CHECK(emit("tests/hmulti.cu", "tests/hst06.cpp"));
    CHECK(strstr(hbuf, "__device__") == NULL);
    CHECK(strstr(hbuf, "float twice(float x);") != NULL);
    CHECK(strstr(hbuf, "x + x") == NULL);
    remove("tests/hst06.cpp");
    PASS();
}
TH_REG("hst", 6, "a __device__ helper leaves a declaration", hst06)

static void hst07(void)
{
    CHECK(emit("tests/htfn.cu", "tests/hst07.cpp"));
    CHECK(strstr(hbuf, "template <typename T, mode M>") != NULL);
    CHECK(strstr(hbuf, "apply<T, M_MUL>(o, p, n)") != NULL);
    CHECK(strstr(hbuf, "apply<T, M_ADD>(o, p, n)") != NULL);
    remove("tests/hst07.cpp");
    PASS();
}
TH_REG("hst", 7, "a launch inside a host template stays generic", hst07)

static void hst08(void)
{
    CHECK(emit("tests/htfn.cu", "tests/hst08.cpp"));
    CHECK(strstr(hbuf, "(const char *)apply<float, (mode)1>") != NULL);
    CHECK(strstr(hbuf, "(const char *)apply<float, (mode)0>") != NULL);
    CHECK(strstr(hbuf, "(const char *)apply<int, (mode)1>") != NULL);
    CHECK(strstr(hbuf, "(const char *)apply<int, (mode)0>") != NULL);
    remove("tests/hst08.cpp");
    PASS();
}
TH_REG("hst", 8, "one source launch reaches four instantiations", hst08)

static void hst09(void)
{
    CHECK(emit("tests/hmac.cu", "tests/hst09.cpp"));
    CHECK(strstr(hbuf, "(const char *)fill2<vec2>") != NULL);
    CHECK(strstr(hbuf, "(const char *)fill2<float>") != NULL);
    remove("tests/hst09.cpp");
    PASS();
}
TH_REG("hst", 9, "a struct template argument is spelled out", hst09)

static void hst10(void)
{
    CHECK(emit("tests/hmac.cu", "tests/hst10.cpp"));
    CHECK(strstr(hbuf, ": fill2<vec2>(dp,") != NULL);
    CHECK(strstr(hbuf, ": k(o, v, n)") != NULL);
    remove("tests/hst10.cpp");
    PASS();
}
TH_REG("hst", 10, "a macro launch and a callee argument", hst10)

static void hst11(void)
{
    CHECK(emit("tests/hpack.cu", "tests/hst11.cpp"));
    CHECK(strstr(hbuf, "(const char *)vfan<float, const float *, const float *>") != NULL);
    remove("tests/hst11.cpp");
    PASS();
}
TH_REG("hst", 11, "a template argument pack is written out", hst11)

static void hst12(void)
{
    CHECK(emit("tests/htid.cu", "tests/hst12.cpp"));
    CHECK(strstr(hbuf, "(const char *)cpyk<cpy1<float, float>>") != NULL);
    CHECK(strstr(hbuf, "<<<") == NULL);
    remove("tests/hst12.cpp");
    PASS();
}
TH_REG("hst", 12, "a function template-id argument is spelled", hst12)

static void hst13(void)
{
    CHECK(emit("tests/hqual.cu", "tests/hst13.cpp"));
    CHECK(strstr(hbuf, "__device__") == NULL);
    CHECK(strstr(hbuf, "__forceinline__") == NULL);
    CHECK(strstr(hbuf, "__host__") == NULL);
    CHECK(strstr(hbuf, "__global__") == NULL);
    remove("tests/hst13.cpp");
    PASS();
}
TH_REG("hst", 13, "no device qualifier survives on a member", hst13)

static void hst14(void)
{
    CHECK(emit("tests/hqual.cu", "tests/hst14.cpp"));
    CHECK(strstr(hbuf, "operator()(int x) const") != NULL);
    CHECK(strstr(hbuf, "int dbl(int x)") != NULL);
    CHECK(strstr(hbuf, "int trip(int x) const") != NULL);
    CHECK(strstr(hbuf, "qop(int a) : v(a)") != NULL);
    remove("tests/hst14.cpp");
    PASS();
}
TH_REG("hst", 14, "the member itself stays behind", hst14)

static void hst16(void)
{
    CHECK(emit("tests/hpack.cu", "tests/hst16.cpp"));
    CHECK(strstr(hbuf, "(void *)&ps...") != NULL);
    remove("tests/hst16.cpp");
    PASS();
}
TH_REG("hst", 16, "a stub body expands its argument pack", hst16)

static void hst17(void)
{
    CHECK(emit("tests/hdecl.cu", "tests/hst17.cpp"));
    CHECK(strstr(hbuf, "int64_t __p1") != NULL);
    CHECK(strstr(hbuf, "(void *)&__p1") != NULL);
    CHECK(strstr(hbuf, "(void *)&int64_t") == NULL);
    remove("tests/hst17.cpp");
    PASS();
}
TH_REG("hst", 17, "a commented out parameter name is not a type", hst17)

static void hst18(void)
{
    CHECK(emit("tests/hdecl.cu", "tests/hst18.cpp"));
    CHECK(strstr(hbuf, "float v = 0.5f") != NULL);
    CHECK(strstr(hbuf, "(void *)&v") != NULL);
    CHECK(strstr(hbuf, "float dbl(float x);") != NULL);
    remove("tests/hst18.cpp");
    PASS();
}
TH_REG("hst", 18, "a default argument reaches the stub", hst18)

static void hst15(void)
{
    CHECK(emit("tests/hqual.cu", "tests/hst15.cpp"));
    CHECK(strstr(hbuf, "#include <cuda_runtime.h>") != NULL);
    CHECK(strstr(hbuf, "cudaError_t") == NULL);
    remove("tests/hst15.cpp");
    PASS();
}
TH_REG("hst", 15, "an angle include stays a directive", hst15)

static unsigned char fbuf[512];

static unsigned fent1(unsigned char *p, unsigned kind, const char *s)
{
    unsigned len = (unsigned)strlen(s);
    unsigned pad = (len + 7u) & ~7u;

    memset(p, 0, 64u + pad);
    p[0] = (unsigned char)kind;
    p[4] = 64u;
    p[8] = (unsigned char)pad;
    p[16] = (unsigned char)len;
    p[26] = 1u;
    p[28] = 89u;
    memcpy(p + 64u, s, len);
    return 64u + pad;
}

static int mkfat(const char *path, int cub, int ptx)
{
    unsigned at = 16u;
    FILE    *f;

    memset(fbuf, 0, sizeof fbuf);
    fbuf[0] = 0x50u; fbuf[1] = 0xEDu; fbuf[2] = 0x55u; fbuf[3] = 0xBAu;
    fbuf[4] = 1u;
    fbuf[6] = 16u;
    if (cub) at += fent1(fbuf + at, 2u, "notreallyacubin");
    if (ptx) at += fent1(fbuf + at, 1u, ".version 8.0\n");
    fbuf[8] = (unsigned char)(at - 16u);
    f = fopen(path, "wb");
    if (f == NULL) return 0;
    if (fwrite(fbuf, 1u, at, f) != at) { fclose(f); return 0; }
    return fclose(f) == 0;
}

static int emitf(const char *cu, const char *fat, const char *out)
{
    char cmd[512];

    snprintf(cmd, sizeof cmd,
             BC_BIN " --cuda-host --fatbin %s -Iruntime/include %s -o %s",
             fat, cu, out);
    if (th_run(cmd, obuf, TH_BUFSZ) != 0) return 0;
    return slurp(out);
}

static void hst19(void)
{
    CHECK(mkfat("tests/hst19.fatbin", 0, 1));
    CHECK(emitf("tests/hvadd.cu", "tests/hst19.fatbin", "tests/hst19.cpp"));
    CHECK(strstr(hbuf, "booth-fat: ptx-only hvadd.cu") != NULL);
    remove("tests/hst19.cpp");
    remove("tests/hst19.fatbin");
    PASS();
}
TH_REG("hst", 19, "a PTX-only fatbin says so in the host unit", hst19)

static void hst20(void)
{
    CHECK(mkfat("tests/hst20.fatbin", 1, 1));
    CHECK(emitf("tests/hvadd.cu", "tests/hst20.fatbin", "tests/hst20.cpp"));
    CHECK(strstr(hbuf, "booth-fat: cubin+ptx hvadd.cu") != NULL);
    CHECK(strstr(hbuf, "ptx-only") == NULL);
    remove("tests/hst20.cpp");
    remove("tests/hst20.fatbin");
    PASS();
}
TH_REG("hst", 20, "a fatbin with both names both", hst20)

static void hst21(void)
{
    CHECK(mkfat("tests/hst21.fatbin", 1, 0));
    CHECK(emitf("tests/hvadd.cu", "tests/hst21.fatbin", "tests/hst21.cpp"));
    CHECK(strstr(hbuf, "booth-fat: cubin hvadd.cu") != NULL);
    CHECK(strstr(hbuf, "ptx") == NULL);
    remove("tests/hst21.cpp");
    remove("tests/hst21.fatbin");
    PASS();
}
TH_REG("hst", 21, "a fatbin with only a cubin names only that", hst21)

static void hst22(void)
{
    CHECK(emit("tests/hfnb.cu", "tests/hst22.cpp"));
    CHECK(strstr(hbuf, "(const char *)scale<(int)3, (bool)1>") != NULL);
    CHECK(strstr(hbuf, "(const char *)scale<(int)3, (bool)0>") != NULL);
    remove("tests/hst22.cpp");
    PASS();
}
TH_REG("hst", 22, "both branch assignments reach the table", hst22)

static void hst23(void)
{
    CHECK(emit("tests/hfnc.cu", "tests/hst23.cpp"));
    CHECK(strstr(hbuf, "(const char *)scale<(int)3, (bool)1>") != NULL);
    CHECK(strstr(hbuf, "(const char *)scale<(int)3, (bool)0>") != NULL);
    remove("tests/hst23.cpp");
    PASS();
}
TH_REG("hst", 23, "a block scoped constexpr reaches the launch", hst23)

static void hst24(void)
{
    CHECK(emit("tests/hfnd.cu", "tests/hst24.cpp"));
    CHECK(strstr(hbuf, "(const char *)scale<(int)3, (bool)1>") != NULL);
    CHECK(strstr(hbuf, "(const char *)scale<(int)3, (bool)0>") != NULL);
    remove("tests/hst24.cpp");
    PASS();
}
TH_REG("hst", 24, "a kernel pointer crosses two call frames", hst24)

static unsigned char f2[512];
static unsigned char dbuf[1024];

static int mkfat2(const char *path, const unsigned char *img, unsigned len)
{
    unsigned pad = (len + 7u) & ~7u;
    unsigned at = 80u + pad;
    FILE    *f;

    memset(f2, 0, sizeof f2);
    f2[0] = 0x50u; f2[1] = 0xEDu; f2[2] = 0x55u; f2[3] = 0xBAu;
    f2[4] = 1u;
    f2[6] = 16u;
    f2[8] = (unsigned char)(at - 16u);
    f2[9] = (unsigned char)((at - 16u) >> 8);
    f2[16] = 1u;
    f2[20] = 64u;
    f2[24] = (unsigned char)pad;
    f2[25] = (unsigned char)(pad >> 8);
    f2[32] = (unsigned char)len;
    f2[33] = (unsigned char)(len >> 8);
    f2[42] = 1u;
    f2[44] = 89u;
    memcpy(f2 + 80u, img, len);
    f = fopen(path, "wb");
    if (f == NULL) return 0;
    if (fwrite(f2, 1u, at, f) != at) { fclose(f); return 0; }
    return fclose(f) == 0;
}

static unsigned bdec(const char *p, unsigned char *d, unsigned max)
{
    unsigned n = 0;
    int      in = 0;

    for (; *p != '\0' && n < max; p++) {
        if (!in) {
            if (*p == '"') in = 1;
            else if (*p == ';') break;
            continue;
        }
        if (*p == '"') { in = 0; continue; }
        if (*p != '\\') { d[n++] = (unsigned char)*p; continue; }
        p++;
        if (*p == 'n') d[n++] = 10u;
        else if (*p == 't') d[n++] = 9u;
        else if (*p == 'r') d[n++] = 13u;
        else if (*p >= '0' && *p <= '7') {
            unsigned v = 0, k = 0;

            while (k < 3u && *p >= '0' && *p <= '7') {
                v = v * 8u + (unsigned)(*p - '0');
                p++;
                k++;
            }
            p--;
            d[n++] = (unsigned char)v;
        } else d[n++] = (unsigned char)*p;
    }
    return n;
}

static void hst25(void)
{
    unsigned char img[256];
    const char   *at;
    unsigned      n;

    for (unsigned i = 0; i < 256u; i++) img[i] = (unsigned char)i;
    CHECK(mkfat2("tests/hst25.fatbin", img, 256u));
    CHECK(emitf("tests/hvadd.cu", "tests/hst25.fatbin", "tests/hst25.cpp"));
    at = strstr(hbuf, "__bfat[] =");
    CHECK(at != NULL);
    n = bdec(at + 10, dbuf, sizeof dbuf);
    CHECK(n == 336u);
    CHECK(memcmp(dbuf + 80u, img, 256u) == 0);
    remove("tests/hst25.cpp");
    remove("tests/hst25.fatbin");
    PASS();
}
TH_REG("hst", 25, "every byte value survives the fatbin literal", hst25)

static void hst26(void)
{
    CHECK(emit("tests/hfun.cu", "tests/hst26.cpp"));
    CHECK(strstr(hbuf, "\"kern\"") != NULL);
    CHECK(strstr(hbuf, "\"kern$1\"") != NULL);
    remove("tests/hst26.cpp");
    PASS();
}
TH_REG("hst", 26, "a functor's operator() registers its kernel", hst26)
