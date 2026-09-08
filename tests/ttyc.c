/* ttyc.c -- type resolution, overload selection, literal coercion
 *
 * All three of these compiled before they were fixed, and all three emitted
 * the wrong numbers, so nothing here asserts that a kernel builds. Every
 * check reads the PTX and looks at the constant, or at the width of the
 * instruction the constant went through.
 *
 * The three shapes: an unregistered `using` alias made every aliased name an
 * i32, so float arithmetic emitted integer stores; an ordinary call matched
 * an overload on argument count alone, so sq(3.0f) called sq(int) and
 * reinterpreted the bits; and an integer literal reaching a float slot stored
 * its bit pattern, so 1 read back as 1.4e-45. */

#include "tharns.h"

static char obuf[1 << 16];
static char code[1 << 16];

static int ty_run(const char *src, const char *args)
{
    char cmd[512];
    FILE *f = fopen("build/ttyc.cu", "w");

    if (!f) return -1;
    fputs(src, f);
    fclose(f);
    snprintf(cmd, sizeof cmd, "%s %s build/ttyc.cu", BC_BIN, args);
    return th_run(cmd, obuf, (int)sizeof obuf);
}

static int ty_ptx(const char *src)
{
    FILE *fp;
    size_t n;

    if (ty_run(src, "--nvidia-ptx -o build/ttyc.ptx") != 0) return -1;
    fp = fopen("build/ttyc.ptx", "rb");
    if (!fp) return -1;
    n = fread(code, 1, sizeof code - 1, fp);
    fclose(fp);
    code[n] = '\0';
    return 0;
}

static int ty_err(const char *src, const char *want)
{
    if (ty_run(src, "--ir") == 0) return 0;
    return strstr(obuf, want) != NULL;
}

/* ---- using aliases ---- */

/* 0f40400000 is 3.0f. The alias used to resolve to i32, so 1.5f went into the
 * slot as its bit pattern and came back out through a cvt.rn.f32.s32. */
static void tyc01(void)
{
    CHEQ(ty_ptx("using myf = float;\n"
                "__global__ void k(myf *o) {\n"
                "  myf x = 1.5f;\n"
                "  o[0] = x * 2.0f;\n"
                "}\n"), 0);
    CHNE(strstr(code, "0f40400000"), NULL);
    CHEQ(strstr(code, "cvt.rn.f32.s32"), NULL);
    CHEQ(strstr(code, "st.global.u32"), NULL);
    PASS();
}
TH_REG("tyc", 1, "a using alias names the type it was given", tyc01)

/* An alias declared inside a namespace is still an alias. */
static void tyc02(void)
{
    CHEQ(ty_ptx("namespace nn { using myf = float; }\n"
                "__global__ void k(float *o) {\n"
                "  nn::myf x = 2.5f;\n"
                "  o[0] = x + 0.5f;\n"
                "}\n"), 0);
    CHNE(strstr(code, "0f40400000"), NULL);
    CHEQ(strstr(code, "st.global.u32"), NULL);
    PASS();
}
TH_REG("tyc", 2, "a namespace-scoped using alias registers", tyc02)

/* The star belongs to the alias, so the parameter is a pointer and the store
 * strides by four. Dropping it made o a float and the write went nowhere. */
static void tyc03(void)
{
    CHEQ(ty_ptx("using fptr = float*;\n"
                "__global__ void k(fptr o) {\n"
                "  o[1] = 3.0f;\n"
                "}\n"), 0);
    CHNE(strstr(code, "0f40400000"), NULL);
    CHNE(strstr(code, "st.global.f32"), NULL);
    PASS();
}
TH_REG("tyc", 3, "a using alias carries its pointer depth", tyc03)

/* An alias in a function body is in scope for the declarations under it. */
static void tyc04(void)
{
    CHEQ(ty_ptx("__global__ void k(float *o) {\n"
                "  using myf = float;\n"
                "  myf x = 4.0f;\n"
                "  o[0] = x;\n"
                "}\n"), 0);
    CHNE(strstr(code, "0f40800000"), NULL);
    CHEQ(strstr(code, "st.global.u32"), NULL);
    PASS();
}
TH_REG("tyc", 4, "a using alias inside a body registers", tyc04)

/* ---- an unresolved type name is a refusal, not an i32 ---- */

static void tyc05(void)
{
    CHECK(ty_err("__global__ void k(float *o) {\n"
                 "  widget w;\n"
                 "  (void)w;\n"
                 "  o[0] = 1.0f;\n"
                 "}\n", "E155"));
    PASS();
}
TH_REG("tyc", 5, "an unknown type name is refused by name", tyc05)

/* The refusal turned this up: uint64_t was not a name Booth knew, so it fell
 * to i32 and the shift was 32 bits wide. 0f53800000 is 2^40 as a float, which
 * only appears if the shift ran at 64 bits. The old answer was 0f43800000,
 * 256.0f, from shifting a 32-bit value by 40. */
static void tyc06(void)
{
    CHEQ(ty_ptx("__global__ void k(float *o) {\n"
                "  uint64_t m = 1;\n"
                "  o[0] = (float)(m << 40);\n"
                "}\n"), 0);
    CHNE(strstr(code, "0f53800000"), NULL);
    CHEQ(strstr(code, "0f43800000"), NULL);
    PASS();
}
TH_REG("tyc", 6, "uint64_t is sixty-four bits wide", tyc06)

static void tyc07(void)
{
    CHEQ(ty_ptx("__global__ void k(unsigned char *o) {\n"
                "  uint8_t b = 7;\n"
                "  o[0] = b;\n"
                "}\n"), 0);
    CHNE(strstr(code, "st.global.u8"), NULL);
    PASS();
}
TH_REG("tyc", 7, "uint8_t is one byte wide", tyc07)

/* ---- overload selection ---- */

/* 0f41100000 is 9.0f. Matching on argument count alone reached sq(int) and
 * emitted mul.lo.u32 over 1077936128, the bits of 3.0f read as an integer. */
static void tyc08(void)
{
    CHEQ(ty_ptx("__device__ int sq(int a) { return a * a; }\n"
                "__device__ float sq(float a) { return a * a; }\n"
                "__global__ void k(float *o) {\n"
                "  o[0] = sq(3.0f);\n"
                "}\n"), 0);
    CHNE(strstr(code, "0f41100000"), NULL);
    CHEQ(strstr(code, "1077936128"), NULL);
    PASS();
}
TH_REG("tyc", 8, "a float argument reaches the float overload", tyc08)

/* The same call with the declarations the other way round, so passing is not
 * an accident of which one happened to be first. */
static void tyc09(void)
{
    CHEQ(ty_ptx("__device__ float sq(float a) { return a * a; }\n"
                "__device__ int sq(int a) { return a * a; }\n"
                "__global__ void k(int *o) {\n"
                "  o[0] = sq(5);\n"
                "}\n"), 0);
    CHNE(strstr(code, "mov.u32 %r1, 25"), NULL);
    PASS();
}
TH_REG("tyc", 9, "an int argument reaches the int overload", tyc09)

/* A pointer argument picks by pointee, not by arity. */
static void tyc10(void)
{
    CHEQ(ty_ptx("__device__ float ld(int *p) { return 1.0f; }\n"
                "__device__ float ld(float *p) { return *p; }\n"
                "__global__ void k(float *o, float *p) {\n"
                "  o[0] = ld(p);\n"
                "}\n"), 0);
    CHEQ(strstr(code, "0f3F800000"), NULL);
    PASS();
}
TH_REG("tyc", 10, "a pointer argument picks by pointee", tyc10)

/* Two candidates neither of which is better refuse by name rather than
 * letting declaration order decide. */
static void tyc11(void)
{
    CHECK(ty_err("__device__ int amb(float a, int b) { return b; }\n"
                 "__device__ int amb(int a, float b) { return a; }\n"
                 "__global__ void k(int *o) {\n"
                 "  o[0] = amb(1, 2);\n"
                 "}\n", "E156"));
    PASS();
}
TH_REG("tyc", 11, "an ambiguous call is refused", tyc11)

/* One overload only: nothing to choose between, and the old path stands. */
static void tyc12(void)
{
    CHEQ(ty_ptx("__device__ float dbl(float a) { return a + a; }\n"
                "__global__ void k(float *o) {\n"
                "  o[0] = dbl(2.0f);\n"
                "}\n"), 0);
    CHNE(strstr(code, "0f40800000"), NULL);
    PASS();
}
TH_REG("tyc", 12, "a single candidate still resolves", tyc12)

/* ---- literal coercion ---- */

/* 0f3F800000 is 1.0f. The store used to be the integer 1, which reads back as
 * 1.4e-45. float x = 0 survived only because the bit patterns coincide. */
static void tyc13(void)
{
    CHEQ(ty_ptx("__global__ void k(float *o) {\n"
                "  float x = 1;\n"
                "  o[0] = x;\n"
                "}\n"), 0);
    CHNE(strstr(code, "0f3F800000"), NULL);
    CHEQ(strstr(code, "mov.u32 %r1, 1"), NULL);
    PASS();
}
TH_REG("tyc", 13, "an int literal initialising a float converts", tyc13)

/* 0f40000000 is 2.0f, through the assignment path rather than the decl. */
static void tyc14(void)
{
    CHEQ(ty_ptx("__global__ void k(float *o) {\n"
                "  o[1] = 2;\n"
                "}\n"), 0);
    CHNE(strstr(code, "0f40000000"), NULL);
    CHEQ(strstr(code, "st.global.u32"), NULL);
    PASS();
}
TH_REG("tyc", 14, "an int literal assigned to a float converts", tyc14)

/* An int variable, not a literal, so the conversion is an instruction rather
 * than a folded constant. */
static void tyc15(void)
{
    CHEQ(ty_ptx("__global__ void k(float *o, int n) {\n"
                "  float x = n;\n"
                "  o[0] = x;\n"
                "}\n"), 0);
    CHNE(strstr(code, "cvt.rn.f32.s32"), NULL);
    CHNE(strstr(code, "st.global.f32"), NULL);
    PASS();
}
TH_REG("tyc", 15, "an int variable initialising a float converts", tyc15)

/* Compound assignment took the right-hand side raw, so x += 1 on a float
 * added 0f00000001, which is 1.4e-45 and not 1.0f. */
static void tyc16(void)
{
    CHEQ(ty_ptx("__global__ void k(float *o, float s) {\n"
                "  float x = s;\n"
                "  x += 1;\n"
                "  o[0] = x;\n"
                "}\n"), 0);
    CHNE(strstr(code, "0f3F800000"), NULL);
    CHEQ(strstr(code, "0f00000001"), NULL);
    PASS();
}
TH_REG("tyc", 16, "a compound assignment converts its operand", tyc16)

/* Each element of an array initialiser reaches the element type. */
static void tyc17(void)
{
    CHEQ(ty_ptx("__global__ void k(float *o) {\n"
                "  float a[2] = {1, 2};\n"
                "  o[0] = a[0] + a[1];\n"
                "}\n"), 0);
    CHNE(strstr(code, "0f3F800000"), NULL);
    CHNE(strstr(code, "0f40000000"), NULL);
    CHEQ(strstr(code, "st.local.u32"), NULL);
    PASS();
}
TH_REG("tyc", 17, "an array initialiser converts each element", tyc17)

/* And each field of a struct initialiser. */
static void tyc18(void)
{
    CHEQ(ty_ptx("struct P { float x; float y; };\n"
                "__global__ void k(float *o) {\n"
                "  P p = {1, 2};\n"
                "  o[0] = p.x + p.y;\n"
                "}\n"), 0);
    CHNE(strstr(code, "0f40400000"), NULL);
    PASS();
}
TH_REG("tyc", 18, "a struct initialiser converts each field", tyc18)

/* The other direction: a float literal into an int slot used to keep the
 * float bits. 3.5f truncates to 3, and 0f40400000 is 3.0f coming back. */
static void tyc19(void)
{
    CHEQ(ty_ptx("__global__ void k(float *o) {\n"
                "  int m = 3.5f;\n"
                "  o[0] = (float)m;\n"
                "}\n"), 0);
    CHNE(strstr(code, "0f40400000"), NULL);
    CHEQ(strstr(code, "1080033280"), NULL);
    PASS();
}
TH_REG("tyc", 19, "a float literal initialising an int converts", tyc19)

/* An f64 slot took the raw integer, so the multiply ran on a denormal. */
static void tyc20(void)
{
    CHEQ(ty_ptx("__global__ void k(float *o) {\n"
                "  double d = 2;\n"
                "  o[0] = (float)(d * 2.0);\n"
                "}\n"), 0);
    CHNE(strstr(code, "0f40800000"), NULL);
    CHEQ(strstr(code, "0f00000002"), NULL);
    PASS();
}
TH_REG("tyc", 20, "an int literal initialising a double converts", tyc20)

/* A widening int assignment is a conversion too: the value has to reach the
 * slot as sixty-four bits or the top half is whatever was there. */
static void tyc21(void)
{
    CHEQ(ty_ptx("__global__ void k(unsigned long long *o, int n) {\n"
                "  unsigned long long v = n;\n"
                "  o[0] = v;\n"
                "}\n"), 0);
    CHNE(strstr(code, "cvt"), NULL);
    CHNE(strstr(code, "st.global.u64"), NULL);
    PASS();
}
TH_REG("tyc", 21, "a narrow int widens into a wide slot", tyc21)

/* Same shape at the call boundary: an int literal handed to a float parameter
 * used to arrive as its bit pattern, so half(3) multiplied 4.2e-45 by a half.
 * 0f3FC00000 is 1.5f. */
static void tyc22(void)
{
    CHEQ(ty_ptx("__device__ float half(float a) { return a * 0.5f; }\n"
                "__global__ void k(float *o) {\n"
                "  o[0] = half(3);\n"
                "}\n"), 0);
    CHNE(strstr(code, "0f3FC00000"), NULL);
    CHEQ(strstr(code, "0f00000003"), NULL);
    PASS();
}
TH_REG("tyc", 22, "an int literal argument converts to float", tyc22)

/* And on the way out. The return used to keep the literal's own type, so a
 * float function returned the integer 1 and the caller stored it raw. */
static void tyc23(void)
{
    CHEQ(ty_ptx("__device__ float one(void) { return 1; }\n"
                "__global__ void k(float *o) {\n"
                "  o[0] = one();\n"
                "}\n"), 0);
    CHNE(strstr(code, "0f3F800000"), NULL);
    CHEQ(strstr(code, "st.global.u32"), NULL);
    PASS();
}
TH_REG("tyc", 23, "a returned int literal converts to float", tyc23)
