/* tref.c -- reference parameters, checked in the emitted code */

#include "tharns.h"

static char obuf[1 << 16];
static char code[1 << 16];

static int rf_run(const char *src, const char *args)
{
    char cmd[512];
    FILE *f = fopen("build/tref.cu", "w");

    if (!f) return -1;
    fputs(src, f);
    fclose(f);
    snprintf(cmd, sizeof cmd, "%s %s build/tref.cu", BC_BIN, args);
    return th_run(cmd, obuf, (int)sizeof obuf);
}

static int rf_emit(const char *src, const char *args, const char *path)
{
    FILE *fp;
    size_t n;

    if (rf_run(src, args) != 0) return -1;
    fp = fopen(path, "rb");
    if (!fp) return -1;
    n = fread(code, 1, sizeof code - 1, fp);
    fclose(fp);
    code[n] = '\0';
    return 0;
}

static int rf_err(const char *src, const char *want)
{
    if (rf_run(src, "--ir") == 0) return 0;
    return strstr(obuf, want) != NULL;
}

/* ---- The write reaches the caller ---- */

static void ref01(void)
{
    CHEQ(rf_run("__device__ void bump(float& a, float v) { a += v; }\n"
                "__global__ void k(float *o) {\n"
                "  float a = 1.0f;\n"
                "  bump(a, 3.0f);\n"
                "  o[0] = a;\n"
                "}\n", "--nvidia-ptx --ir"), 0);
    CHNE(strstr(obuf, "store f32 4,"), NULL);
    PASS();
}
TH_REG("ref", 1, "a T& parameter writes the caller's variable", ref01)

/* 0f40800000 is 4.0f. By value it was 1.0f. */
static void ref02(void)
{
    CHEQ(rf_emit("__device__ void bump(float& a, float v) { a += v; }\n"
                 "__global__ void k(float *o) {\n"
                 "  float a = 1.0f;\n"
                 "  bump(a, 3.0f);\n"
                 "  o[0] = a;\n"
                 "}\n", "--nvidia-ptx -o build/tref.ptx", "build/tref.ptx"), 0);
    CHNE(strstr(code, "0f40800000"), NULL);
    CHEQ(strstr(code, "0f3F800000"), NULL);
    PASS();
}
TH_REG("ref", 2, "and it is in the PTX, not only the IR", ref02)

static void ref03(void)
{
    CHEQ(rf_emit("__device__ void bump(float& a, float v) { a += v; }\n"
                 "__global__ void k(float *o) {\n"
                 "  float a = 1.0f;\n"
                 "  bump(a, 3.0f);\n"
                 "  o[0] = a;\n"
                 "}\n", "--amdgpu -o build/tref.s", "build/tref.s"), 0);
    CHNE(strstr(code, "0x40800000"), NULL);
    CHEQ(strstr(code, "0x3f800000"), NULL);
    PASS();
}
TH_REG("ref", 3, "and in the AMD assembly too", ref03)

/* ---- const T& is a reference, not a copy ---- */

/* poke and peek name the same object */
static void ref04(void)
{
    CHEQ(rf_emit("__device__ void poke(float& a) { a = 9.0f; }\n"
                 "__device__ float peek(const float& c, float& a) {\n"
                 "  poke(a);\n"
                 "  return c;\n"
                 "}\n"
                 "__global__ void k(float *o) {\n"
                 "  float x = 1.0f;\n"
                 "  o[0] = peek(x, x);\n"
                 "}\n", "--nvidia-ptx -o build/tref.ptx", "build/tref.ptx"), 0);
    CHNE(strstr(code, "0f41100000"), NULL);
    CHEQ(strstr(code, "0f3F800000"), NULL);
    PASS();
}
TH_REG("ref", 4, "a const T& aliases, it does not copy", ref04)

static void ref05(void)
{
    CHEQ(rf_run("struct V { float x; float y; };\n"
                "__device__ float sum(const V& v) { return v.x + v.y; }\n"
                "__global__ void k(float *o) {\n"
                "  V a; a.x = 1.0f; a.y = 2.0f;\n"
                "  o[0] = sum(a);\n"
                "}\n", "--ir"), 0);
    CHNE(strstr(obuf, "func @sum(ptr<generic,"), NULL);
    CHEQ(strstr(obuf, "func @sum(type_"), NULL);
    PASS();
}
TH_REG("ref", 5, "a const V& parameter is a pointer, not a copy", ref05)

/* ---- The shapes that turn up in real kernels ---- */

static void ref06(void)
{
    CHEQ(rf_run("struct V { float x; float y; };\n"
                "__device__ void addv(V& d, const V& s) {\n"
                "  d.x += s.x; d.y += s.y;\n"
                "}\n"
                "__global__ void k(float *o) {\n"
                "  V a; a.x = 1.0f; a.y = 2.0f;\n"
                "  V b; b.x = 10.0f; b.y = 20.0f;\n"
                "  addv(a, b);\n"
                "  o[0] = a.x; o[1] = a.y;\n"
                "}\n", "--nvidia-ptx --ir"), 0);
    CHNE(strstr(obuf, "store f32 11,"), NULL);
    CHNE(strstr(obuf, "store f32 22,"), NULL);
    PASS();
}
TH_REG("ref", 6, "a V& parameter writes the caller's struct", ref06)

static int rf_cnt(const char *hay, const char *needle)
{
    int c = 0;
    const char *p = hay;

    while ((p = strstr(p, needle)) != NULL) { c++; p++; }
    return c;
}

/* Array slots stay in memory, so caller and callee store to one address */
static void ref07(void)
{
    CHEQ(rf_emit("__device__ void bump(float& a, float v) { a += v; }\n"
                 "__global__ void k(float *o) {\n"
                 "  float b[4];\n"
                 "  b[2] = 5.0f;\n"
                 "  bump(b[2], 1.0f);\n"
                 "  o[0] = b[2];\n"
                 "}\n", "--nvidia-ptx -o build/tref.ptx",
                 "build/tref.ptx"), 0);
    CHEQ(rf_cnt(code, "st.local.f32"), 2);
    CHNE(strstr(code, "add.rn.f32"), NULL);
    PASS();
}
TH_REG("ref", 7, "a reference to an array element writes it", ref07)

static void ref08(void)
{
    CHEQ(rf_run("struct V { float x; float y; };\n"
                "__device__ void bump(float& a, float v) { a += v; }\n"
                "__global__ void k(float *o) {\n"
                "  V a; a.x = 1.0f; a.y = 2.0f;\n"
                "  bump(a.y, 100.0f);\n"
                "  o[0] = a.y;\n"
                "}\n", "--nvidia-ptx --ir"), 0);
    CHNE(strstr(obuf, "store f32 102,"), NULL);
    PASS();
}
TH_REG("ref", 8, "a reference to a member writes that member", ref08)

static void ref09(void)
{
    CHEQ(rf_run("__device__ void bump(float& a, float v) { a += v; }\n"
                "__device__ void chain(float& a, float v) { bump(a, v); }\n"
                "__global__ void k(float *o) {\n"
                "  float a = 1.0f;\n"
                "  chain(a, 10.0f);\n"
                "  o[0] = a;\n"
                "}\n", "--nvidia-ptx --ir"), 0);
    CHNE(strstr(obuf, "store f32 11,"), NULL);
    PASS();
}
TH_REG("ref", 9, "a reference passed on reaches the original", ref09)

/* A reference to a global element stays in global memory */
static void ref10(void)
{
    CHEQ(rf_emit("__device__ void bump(float& a, float v) { a += v; }\n"
                 "__global__ void k(float *o) { bump(o[3], 1.0f); }\n",
                 "--nvidia-ptx -o build/tref.ptx", "build/tref.ptx"), 0);
    {
        const char *e = strstr(code, ".entry");
        CHNE(e, NULL);
        CHEQ(rf_cnt(e, "st.global.f32"), 1);
        CHEQ(rf_cnt(e, "ld.global.f32"), 1);
        CHEQ(rf_cnt(e, ".local"), 0);
    }
    PASS();
}
TH_REG("ref", 10, "a reference to global memory stays global", ref10)

static void ref11(void)
{
    CHEQ(rf_run("__global__ void k(float *o) {\n"
                "  float a = 1.0f;\n"
                "  float& r = a;\n"
                "  r += 2.0f;\n"
                "  o[0] = a;\n"
                "}\n", "--nvidia-ptx --ir"), 0);
    CHNE(strstr(obuf, "store f32 3,"), NULL);
    PASS();
}
TH_REG("ref", 11, "a local T& aliases what it was bound to", ref11)

static void ref12(void)
{
    CHEQ(rf_run("__device__ void nudge(float*& p) { p = p + 1; }\n"
                "__global__ void k(float *o) {\n"
                "  float *q = o;\n"
                "  nudge(q);\n"
                "  q[0] = 7.0f;\n"
                "}\n", "--ir"), 0);
    CHNE(strstr(obuf, "func @nudge(ptr<generic, ptr<global, f32>>"), NULL);
    PASS();
}
TH_REG("ref", 12, "a T*& parameter can move the caller's pointer", ref12)

/* A literal gets a temporary for the const T& to point at */
static void ref13(void)
{
    CHEQ(rf_run("__device__ float dbl(const float& a) { return a * 2.0f; }\n"
                "__global__ void k(float *o) { o[0] = dbl(3.0f); }\n",
                "--ir"), 0);
    CHNE(strstr(obuf, "store f32 3,"), NULL);
    CHNE(strstr(obuf, "call f32 @dbl("), NULL);
    PASS();
}
TH_REG("ref", 13, "a const T& binds a literal through a temporary", ref13)

static void ref14(void)
{
    CHEQ(rf_run("__device__ double dbl(const double& a) { return a * 2.0; }\n"
                "__global__ void k(float *o) {\n"
                "  float x = 3.0f;\n"
                "  o[0] = (float)dbl(x);\n"
                "}\n", "--ir"), 0);
    CHNE(strstr(obuf, "alloca ptr<private, f64>"), NULL);
    CHNE(strstr(obuf, "store f64 3,"), NULL);
    PASS();
}
TH_REG("ref", 14, "a const double& widens the float it was given", ref14)

static void ref15(void)
{
    CHEQ(rf_run("struct v2 { float x; float y; };\n"
                "__device__ v2 operator+(const v2& a, const v2& b) {\n"
                "  v2 r; r.x = a.x + b.x; r.y = a.y + b.y; return r;\n"
                "}\n"
                "__global__ void k(float *o) {\n"
                "  v2 p; p.x = 1.0f; p.y = 2.0f;\n"
                "  v2 q; q.x = 10.0f; q.y = 20.0f;\n"
                "  v2 s = p + q;\n"
                "  o[0] = s.x;\n"
                "}\n", "--ir"), 0);
    CHNE(strstr(obuf, "@operator+(ptr<generic,"), NULL);
    CHEQ(strstr(obuf, "E102"), NULL);
    PASS();
}
TH_REG("ref", 15, "an operator taking const T& is still selected", ref15)

/* ---- What Booth says no to ---- */

static void ref16(void)
{
    CHECK(rf_err("__device__ float& pick(float *p) { return p[0]; }\n"
                 "__global__ void k(float *o) { o[0] = pick(o); }\n",
                 "E145"));
    PASS();
}
TH_REG("ref", 16, "a function returning a reference is refused", ref16)

static void ref17(void)
{
    CHEQ(rf_run("__device__ void set(float&& a) { a = 1.0f; }\n"
                "__global__ void k(float *o) {\n"
                "  float x = 0.0f;\n"
                "  set(std::move(x));\n"
                "  o[0] = x;\n"
                "}\n", "--ir"), 0);
    CHNE(strstr(obuf, "@set(ptr<generic, f32>"), NULL);
    PASS();
}
TH_REG("ref", 17, "an rvalue reference parameter binds", ref17)

static void ref18(void)
{
    CHECK(rf_err("__global__ void k(float& o) { o = 1.0f; }\n", "E147"));
    PASS();
}
TH_REG("ref", 18, "a reference kernel parameter is refused", ref18)

static void ref19(void)
{
    CHECK(rf_err("__device__ void bump(float& a) { a += 1.0f; }\n"
                 "__global__ void k(float *o) {\n"
                 "  float x = 1.0f;\n"
                 "  bump(x + 1.0f);\n"
                 "  o[0] = x;\n"
                 "}\n", "E148"));
    PASS();
}
TH_REG("ref", 19, "a non-const T& will not bind an rvalue", ref19)

/* float to double& is named, not converted */
static void ref20(void)
{
    CHECK(rf_err("__device__ void bump(double& a) { a += 1.0; }\n"
                 "__global__ void k(float *o) {\n"
                 "  float x = 1.0f;\n"
                 "  bump(x);\n"
                 "  o[0] = x;\n"
                 "}\n", "E149"));
    PASS();
}
TH_REG("ref", 20, "a non-const T& will not bind a different type", ref20)

static void ref21(void)
{
    CHECK(rf_err("struct S { float& r; };\n"
                 "__global__ void k(float *o) { o[0] = 1.0f; }\n", "E150"));
    PASS();
}
TH_REG("ref", 21, "a reference member is refused", ref21)

static void ref22(void)
{
    CHECK(rf_err("template<class ... A>\n"
                 "__global__ void pk(float *o, A&... a) {\n"
                 "  o[0] = (0.0f + ... + (float)a);\n"
                 "}\n"
                 "int main(void) {\n"
                 "  float *o; cudaMalloc(&o, 16);\n"
                 "  pk<<<1,1>>>(o, 1.0f, 2.0f);\n"
                 "  return 0;\n"
                 "}\n", "E151"));
    PASS();
}
TH_REG("ref", 22, "a reference in a parameter pack is refused", ref22)

static void ref23(void)
{
    CHECK(rf_err("__global__ void k(float *o) { float& r; o[0] = 1.0f; }\n",
                 "E152"));
    PASS();
}
TH_REG("ref", 23, "a reference with no initialiser is refused", ref23)

static void ref24(void)
{
    CHEQ(rf_emit("__device__ void f(float (&a)[4]) { a[2] = 7.0f; }\n"
                 "__global__ void k(float *o) {\n"
                 "  float b[4];\n"
                 "  for (int i = 0; i < 4; ++i) b[i] = 1.0f;\n"
                 "  f(b);\n"
                 "  o[0] = b[2];\n"
                 "}\n", "--nvidia-ptx -o build/tref.ptx",
                 "build/tref.ptx"), 0);
    CHNE(strstr(code, "0f40E00000"), NULL);
    PASS();
}
TH_REG("ref", 24, "an array reference writes the caller array", ref24)

/* Overloads rank the reference parameter, so 1.0 + 2.0 is a float add */
static void ref25(void)
{
    CHEQ(rf_emit("__device__ void mad(int& a, int v) { a += v; }\n"
                 "__device__ void mad(float& a, float v) { a += v; }\n"
                 "__global__ void k(float *o) {\n"
                 "  float x = 1.0f;\n"
                 "  mad(x, 2.0f);\n"
                 "  o[0] = x;\n"
                 "}\n", "--nvidia-ptx -o build/tref.ptx",
                 "build/tref.ptx"), 0);
    CHNE(strstr(code, "0f40400000"), NULL);
    PASS();
}
TH_REG("ref", 25, "a reference ranks with the other arguments", ref25)

/* ggml_cuda_mad, verbatim */
static void ref26(void)
{
    CHEQ(rf_run("static __device__ __forceinline__ void ggml_cuda_mad("
                "float & acc, const float v, const float u) { acc += v*u; }\n"
                "__global__ void dot4(const float *a, const float *b,"
                " float *out) {\n"
                "  float sum = 0.0f;\n"
                "  for (int i = 0; i < 4; ++i) ggml_cuda_mad(sum, a[i], b[i]);\n"
                "  out[0] = sum;\n"
                "}\n", "--nvidia-ptx --ir"), 0);
    CHNE(strstr(obuf, "fmul f32"), NULL);
    CHNE(strstr(obuf, "fadd f32"), NULL);
    CHEQ(strstr(obuf, "store f32 0,"), NULL);
    PASS();
}
TH_REG("ref", 26, "the ggml accumulator shape accumulates", ref26)

/* The & on a typedef is still a reference */
static void ref27(void)
{
    CHECK(rf_err("typedef float& fref;\n"
                 "__device__ void bump(fref a) { a += 1.0f; }\n"
                 "__global__ void k(float *o) {\n"
                 "  float x = 1.0f;\n"
                 "  bump(x);\n"
                 "  o[0] = x;\n"
                 "}\n", "E154"));
    PASS();
}
TH_REG("ref", 27, "an alias for a reference type is refused", ref27)

/* Overloads differing only in the struct behind a reference resolve */
static void ref30(void)
{
    CHEQ(rf_emit("struct ta { int v; };\n"
                 "struct tb { int v; };\n"
                 "__device__ void fill(ta & t, int s){ t.v = s + 111; }\n"
                 "__device__ void fill(tb & t, int s){ t.v = s + 222; }\n"
                 "__global__ void k(int *o){ ta a; tb b;\n"
                 "  fill(a, 1); fill(b, 1);\n"
                 "  o[0] = a.v; o[1] = b.v; }\n",
                 "--nvidia-ptx -o build/tref.ptx", "build/tref.ptx"), 0);
    CHNE(strstr(code, ", 112;"), NULL);
    CHNE(strstr(code, ", 223;"), NULL);
    PASS();
}
TH_REG("ref", 30, "two struct references pick their own body", ref30)
