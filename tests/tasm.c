/* tasm.c -- inline asm operands and the local frame they sit in */

#include "tharns.h"

static char obuf[1 << 16];
static char code[1 << 16];

static int arun(const char *src, const char *args)
{
    char cmd[512];
    FILE *f = fopen("build/tasm.cu", "w");

    if (!f) return -1;
    fputs(src, f);
    fclose(f);
    snprintf(cmd, sizeof cmd, "%s %s build/tasm.cu", BC_BIN, args);
    return th_run(cmd, obuf, (int)sizeof obuf);
}

static int aptx(const char *src)
{
    FILE *fp;
    size_t n;

    if (arun(src, "--nvidia-ptx -o build/tasm.ptx") != 0) return -1;
    fp = fopen("build/tasm.ptx", "rb");
    if (!fp) return -1;
    n = fread(code, 1, sizeof code - 1, fp);
    fclose(fp);
    code[n] = '\0';
    return 0;
}

static const char *const frame =
    "__global__ void k(float *o, int n) {\n"
    "    char c[3];\n"
    "    __align__(16) float t[4];\n"
    "    double d[2];\n"
    "    for (int i = 0; i < n; i++) c[i & 1] = (char)i;\n"
    "    for (int i = 0; i < 4; i++) t[i] = o[i];\n"
    "    for (int i = 0; i < 2; i++) d[i] = (double)o[i];\n"
    "    o[0] = (float)c[0] + t[0] + (float)d[0];\n"
    "}\n";

/* ---- Local frame layout ---- */

static void asm01(void)
{
    CHEQ(aptx(frame), 0);
    CHNE(strstr(code, ".local .align 16 "), NULL);
    PASS();
}
TH_REG("asm", 1, "the frame is declared at its widest member", asm01)

static void asm02(void)
{
    CHEQ(aptx(frame), 0);
    CHNE(strstr(code, "__local+16;"), NULL);
    CHEQ(strstr(code, "__local+3;"), NULL);
    PASS();
}
TH_REG("asm", 2, "an __align__(16) local lands 16-aligned", asm02)

static void asm03(void)
{
    CHEQ(aptx(frame), 0);
    CHNE(strstr(code, "__local+32;"), NULL);
    CHEQ(strstr(code, "__local+19;"), NULL);
    PASS();
}
TH_REG("asm", 3, "a double follows on a multiple of 8, unasked", asm03)

static void asm04(void)
{
    CHEQ(aptx("struct __align__(32) big { float a; };\n"
              "__global__ void k(float *o, int n) {\n"
              "    char c[3];\n"
              "    struct big b[2];\n"
              "    for (int i = 0; i < n; i++) c[i & 1] = (char)i;\n"
              "    for (int i = 0; i < 2; i++) b[i].a = o[i];\n"
              "    o[0] = (float)c[0] + b[0].a;\n"
              "}\n"), 0);
    CHNE(strstr(code, ".local .align 32 "), NULL);
    CHNE(strstr(code, "__local+32;"), NULL);
    PASS();
}
TH_REG("asm", 4, "a struct carries its declared alignment", asm04)

static void asm05(void)
{
    CHEQ(arun("__global__ void k(float *o){ alignas(16) float t[4];\n"
              "                             t[0] = 1.0f; o[0] = t[0]; }\n",
              "--ir"), 0);
    CHNE(strstr(obuf, "align 16"), NULL);
    PASS();
}
TH_REG("asm", 5, "alignas says the same thing as __align__", asm05)

static void asm06(void)
{
    CHNE(arun("__global__ void k(float *o){ __align__(3) float t[4];\n"
              "                             t[0] = 1.0f; o[0] = t[0]; }\n",
              "--ir"), 0);
    CHNE(strstr(obuf, "E220"), NULL);
    CHEQ(strstr(obuf, "E020"), NULL);
    PASS();
}
TH_REG("asm", 6, "an alignment off a power of two is refused", asm06)

/* ---- Inline asm ---- */

static void asm07(void)
{
    CHEQ(aptx("__global__ void k(int *o, int a, int b) {\n"
              "    int r;\n"
              "    asm(\"add.s32 %0, %1, %2;\" : \"=r\"(r) : \"r\"(a), \"r\"(b));\n"
              "    o[0] = r;\n"
              "}\n"), 0);
    CHNE(strstr(code, "add.s32 %r"), NULL);
    CHEQ(strstr(code, "%0"), NULL);
    PASS();
}
TH_REG("asm", 7, "the instruction reaches PTX, operands bound", asm07)

static void asm08(void)
{
    CHEQ(aptx("__global__ void k(int *o, const int *s) {\n"
              "    int xi[2];\n"
              "    const int *xs = s + threadIdx.x;\n"
              "    asm volatile(\"ldmatrix.sync.aligned.m8n8.x2.b16 {%0, %1}, [%2];\"\n"
              "        : \"=r\"(xi[0]), \"=r\"(xi[1]) : \"l\"(xs));\n"
              "    o[0] = xi[0] + xi[1];\n"
              "}\n"), 0);
    CHNE(strstr(code, "ldmatrix.sync.aligned.m8n8.x2.b16 {%r"), NULL);
    CHNE(strstr(code, "}, [%rd"), NULL);
    PASS();
}
TH_REG("asm", 8, "a 64-bit constraint binds the address register", asm08)

static void asm09(void)
{
    const char *p;

    CHEQ(aptx("__global__ void k(int *o, int a) {\n"
              "    int d = 0;\n"
              "    asm(\"mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 %0, %1;\"\n"
              "        : \"+r\"(d) : \"r\"(a));\n"
              "    o[0] = d;\n"
              "}\n"), 0);
    p = strstr(code, "mma.sync.aligned");
    CHNE(p, NULL);
    CHNE(strstr(code, "ld.local.u32"), NULL);
    CHNE(strstr(p, "st.local.u32"), NULL);
    PASS();
}
TH_REG("asm", 9, "a read-write operand goes in and comes back", asm09)

static void asm10(void)
{
    CHEQ(aptx("__global__ void k(float *o){ asm volatile(\"membar.gl;\");\n"
              "                             o[0] = 1.0f; }\n"), 0);
    CHNE(strstr(code, "membar.gl;"), NULL);
    PASS();
}
TH_REG("asm", 10, "a template with no operands is still emitted", asm10)

static void asm11(void)
{
    CHEQ(aptx("__global__ void k(int *o, int a) {\n"
              "    int r;\n"
              "    asm(\"shf.l.wrap.b32 %0, %1, %1, %2;\" : \"=r\"(r)\n"
              "        : \"r\"(a), \"n\"(7));\n"
              "    o[0] = r;\n"
              "}\n"), 0);
    CHNE(strstr(code, ", 7;"), NULL);
    PASS();
}
TH_REG("asm", 11, "an n operand substitutes its literal", asm11)

/* ---- Where the binding stops ---- */

static void asm12(void)
{
    CHNE(arun("__global__ void k(int *o, int a){ int r;\n"
              "    asm(\"x %0, %1;\" : \"=v\"(r) : \"r\"(a)); o[0] = r; }\n",
              "--ir"), 0);
    CHNE(strstr(obuf, "E221"), NULL);
    CHNE(strstr(obuf, "\"=v\""), NULL);
    PASS();
}
TH_REG("asm", 12, "a constraint Booth cannot bind is named", asm12)

static void asm13(void)
{
    CHNE(arun("__global__ void k(int *o, int a){ int r;\n"
              "    asm(\"x %0, %1;\" : \"=r\"(r) : \"0\"(a)); o[0] = r; }\n",
              "--ir"), 0);
    CHNE(strstr(obuf, "E221"), NULL);
    CHNE(strstr(obuf, "\"0\""), NULL);
    PASS();
}
TH_REG("asm", 13, "so is a matching-operand constraint", asm13)

static void asm14(void)
{
    CHNE(arun("__global__ void k(int *o, int a){ int r;\n"
              "    asm(\"x %0, %1;\" : \"=r\"(r) : \"n\"(a)); o[0] = r; }\n",
              "--ir"), 0);
    CHNE(strstr(obuf, "E221"), NULL);
    PASS();
}
TH_REG("asm", 14, "an n operand that is not constant is refused", asm14)

static void asm15(void)
{
    CHNE(arun("__global__ void k(int *o){ asm goto(\"bra %l0;\" :::: L);\n"
              "                           L: o[0] = 1; }\n", "--parse"), 0);
    CHNE(strstr(obuf, "E221"), NULL);
    CHEQ(strstr(obuf, "E020"), NULL);
    PASS();
}
TH_REG("asm", 15, "asm goto is refused without a cascade", asm15)

static void asm16(void)
{
    CHNE(arun("__global__ void k(int *o, int a){ int r;\n"
              "    asm(\"add.s32 %0, %1, %1;\" : \"=r\"(r) : \"r\"(a));\n"
              "    o[0] = r; }\n", "--amdgpu -o build/tasm.s"), 0);
    CHNE(strstr(obuf, "inline asm"), NULL);
    PASS();
}
TH_REG("asm", 16, "a PTX template is refused on an AMD target", asm16)
