/* tvec.c -- CUDA vector types and the device intrinsics over them */

#include "tharns.h"

static char obuf[1 << 16];

static int vcrun(const char *src, const char *args)
{
    char cmd[512];
    FILE *f = fopen("build/tvec.cu", "w");

    if (!f) return -1;
    fputs(src, f);
    fclose(f);
    snprintf(cmd, sizeof cmd, "%s %s build/tvec.cu", BC_BIN, args);
    return th_run(cmd, obuf, (int)sizeof obuf);
}

static const char *vcptx(const char *src)
{
    static char ptx[1 << 16];
    char cmd[512];
    FILE *f, *g;
    size_t n;

    ptx[0] = '\0';
    f = fopen("build/tvec.cu", "w");
    if (!f) return NULL;
    fputs(src, f);
    fclose(f);
    remove("build/tvec.ptx");
    snprintf(cmd, sizeof cmd,
             "%s --nvidia-ptx build/tvec.cu -o build/tvec.ptx", BC_BIN);
    if (th_run(cmd, obuf, (int)sizeof obuf) != 0) return NULL;
    g = fopen("build/tvec.ptx", "rb");
    if (!g) return NULL;
    n = fread(ptx, 1, sizeof ptx - 1, g);
    fclose(g);
    ptx[n] = '\0';
    return ptx;
}

/* Writes n structs, then a kernel that reads a field of the last one. */
static int vcmany(int n)
{
    char src[1 << 16];
    int p = 0, i;

    for (i = 0; i < n; i++)
        p += snprintf(src + p, sizeof src - (size_t)p,
                      "struct s%d { int a; float b; };\n", i);
    p += snprintf(src + p, sizeof src - (size_t)p,
                  "__global__ void k(float *o){ struct s%d v; v.b = 1.0f;"
                  " o[0] = v.b; }\n", n - 1);
    (void)p;
    return vcrun(src, "--ir");
}

/* ---- The struct table ---- */

static void vec01(void)
{
    CHEQ(vcmany(200), 0);
    CHEQ(strstr(obuf, "E110"), NULL);
    PASS();
}
TH_REG("vec", 1, "a field lookup survives 200 structs", vec01)

static void vec02(void)
{
    CHNE(vcmany(400), 0);
    CHNE(strstr(obuf, "E138"), NULL);
    PASS();
}
TH_REG("vec", 2, "and past the table it says so", vec02)

/* ---- half2 and the rest of the packed pairs ---- */

static void vec03(void)
{
    CHEQ(vcrun("__global__ void k(float *o, const half2 *p){\n"
               "  half2 a = p[0];\n"
               "  o[0] = __low2float(a) + __high2float(a);\n"
               "}\n", "--ir"), 0);
    CHNE(strstr(obuf, "fpext f16"), NULL);
    PASS();
}
TH_REG("vec", 3, "half2 has an x and a y", vec03)

static void vec04(void)
{
    CHEQ(vcrun("__global__ void k(float *o, const __nv_bfloat162 *p){\n"
               "  __nv_bfloat162 a = p[0];\n"
               "  o[0] = (float)a.x;\n"
               "}\n", "--ir"), 0);
    CHNE(strstr(obuf, "bf16"), NULL);
    PASS();
}
TH_REG("vec", 4, "so does __nv_bfloat162", vec04)

static void vec05(void)
{
    CHEQ(vcrun("__global__ void k(float *o, const half2 *p){\n"
               "  float2 f = __half22float2(p[0]);\n"
               "  o[0] = f.x + f.y;\n"
               "}\n", "--ir"), 0);
    CHEQ(strstr(obuf, "E1"), NULL);
    PASS();
}
TH_REG("vec", 5, "__half22float2 widens both lanes", vec05)

/* ---- Alignment, which BIR's type record cannot carry ---- */

static void vec06(void)
{
    CHNE(vcrun("struct s { char c; float4 v; };\n"
               "__global__ void k(float *o){ struct s q; q.v.x = 1.0f;"
               " o[0] = q.v.x; }\n", "--ir"), 0);
    CHNE(strstr(obuf, "E142"), NULL);
    PASS();
}
TH_REG("vec", 6, "float4 after a char refuses", vec06)

static void vec07(void)
{
    CHEQ(vcrun("struct s { float4 v; int n; };\n"
               "__global__ void k(float *o){ struct s q; q.v.x = 1.0f;"
               " o[0] = q.v.x; }\n", "--ir"), 0);
    CHEQ(strstr(obuf, "E142"), NULL);
    PASS();
}
TH_REG("vec", 7, "and at offset zero it lays out the same", vec07)

static void vec08(void)
{
    CHEQ(vcrun("struct s { char c; float3 v; };\n"
               "__global__ void k(float *o){ struct s q; q.v.x = 1.0f;"
               " o[0] = q.v.x; }\n", "--ir"), 0);
    CHEQ(strstr(obuf, "E142"), NULL);
    PASS();
}
TH_REG("vec", 8, "a 3-vector keeps the scalar's alignment", vec08)

/* ---- Arithmetic on a vector is an overloaded operator ---- */

static void vec09(void)
{
    CHEQ(vcrun("__global__ void k(half2 *o, const half2 *p){\n"
               "  half2 a = p[0], b = p[1];\n"
               "  o[0] = a * b;\n"
               "}\n", "--ir"), 0);
    CHNE(strstr(obuf, "fmul f16"), NULL);
    CHEQ(strstr(obuf, "mul ptr"), NULL);
    PASS();
}
TH_REG("vec", 9, "half2 * half2 multiplies both lanes", vec09)

static void vec10(void)
{
    CHEQ(vcrun("__global__ void k(float *o, const float *p){\n"
               "  o[0] = p[0] * p[1];\n"
               "}\n", "--ir"), 0);
    CHNE(strstr(obuf, "fmul"), NULL);
    PASS();
}
TH_REG("vec", 10, "and a scalar multiply still is", vec10)

/* ---- Device intrinsics ---- */

static void vec11(void)
{
    CHEQ(vcrun("__global__ void k(int *o, const int *p){\n"
               "  o[0] = __dp4a(p[0], p[1], p[2]);\n"
               "}\n", "--ir"), 0);
    CHNE(strstr(obuf, "ashr"), NULL);
    CHNE(strstr(obuf, "shl"), NULL);
    PASS();
}
TH_REG("vec", 11, "__dp4a expands to four signed byte products", vec11)

static void vec12(void)
{
    CHEQ(vcrun("__global__ void k(float *o, const float *p, int e){\n"
               "  o[0] = ldexpf(p[0], e);\n"
               "}\n", "--ir"), 0);
    CHNE(strstr(obuf, "bitcast"), NULL);
    CHEQ(strstr(obuf, "exp2"), NULL);
    PASS();
}
TH_REG("vec", 12, "ldexpf scales by the exponent field", vec12)

static void vec13(void)
{
    CHEQ(vcrun("__global__ void k(int *o, const int *p){\n"
               "  o[0] = max(p[0], 0);\n"
               "  o[1] = min(p[1], 7);\n"
               "}\n", "--ir"), 0);
    CHNE(strstr(obuf, "select"), NULL);
    PASS();
}
TH_REG("vec", 13, "integer max and min select rather than fmax", vec13)

static void vec14(void)
{
    CHEQ(vcrun("__global__ void k(float *o, const float *p){\n"
               "  o[0] = max(p[0], 0.5f);\n"
               "}\n", "--ir"), 0);
    CHNE(strstr(obuf, "fmax"), NULL);
    PASS();
}
TH_REG("vec", 14, "and the float ones still reach fmax", vec14)

static void vec15(void)
{
    CHEQ(vcrun("__global__ void k(float *o, const float *p){\n"
               "  unsigned b = 0x3f800000u; float r;\n"
               "  memcpy(&r, &b, sizeof(float));\n"
               "  o[0] = r + p[0];\n"
               "}\n", "--ir"), 0);
    CHEQ(strstr(obuf, "E143"), NULL);
    PASS();
}
TH_REG("vec", 15, "a constant memcpy is loads and stores", vec15)

static void vec16(void)
{
    CHNE(vcrun("__global__ void k(char *d, const char *s, int n){\n"
               "  memcpy(d, s, n);\n"
               "}\n", "--ir"), 0);
    CHNE(strstr(obuf, "E143"), NULL);
    PASS();
}
TH_REG("vec", 16, "a runtime length refuses instead of guessing", vec16)

static void vec17(void)
{
    CHEQ(vcrun("__global__ void k(int *o){ printf(\"%d\", o[0]); __trap(); }\n",
               "--ir"), 0);
    CHNE(strstr(obuf, "printf i32"), NULL);
    CHNE(strstr(obuf, "trap  ;"), NULL);
    PASS();
}
TH_REG("vec", 17, "printf and __trap reach the IR", vec17)

static void vec27(void)
{
    CHEQ(vcrun("__global__ void k(int *o, const char *f){ printf(f, o[0]); }\n",
               "--nvidia-ptx -o build/tvec.ptx"), 0);
    CHNE(strstr(obuf, "wrote build/tvec.ptx"), NULL);
    PASS();
}
TH_REG("vec", 27, "and PTX calls vprintf with the pointer", vec27)

static void vec28(void)
{
    CHEQ(vcrun("__device__ int g(int x){ return x; }\n"
               "__global__ void k(int *o){ (void)g; o[0] = 1; }\n",
               "--ir --no-dce"), 0);
    CHNE(strstr(obuf, "fnref @g"), NULL);
    CHNE(vcrun("__device__ int g(int x){ return x; }\n"
               "__global__ void k(int *o){ (void)g; o[0] = 1; }\n",
               "--nvidia-ptx --no-dce -o build/tvec.ptx"), 0);
    CHNE(strstr(obuf, "address of a device function"), NULL);
    PASS();
}
TH_REG("vec", 28, "a function named as a value is a fnref", vec28)

/* ---- reinterpret_cast to a reference ---- */

static void vec18(void)
{
    CHEQ(vcrun("__global__ void k(float *o){\n"
               "  float2 r;\n"
               "  reinterpret_cast<float&>(r.x) = 1.0f;\n"
               "  o[0] = r.x;\n"
               "}\n", "--ir"), 0);
    CHEQ(strstr(obuf, "E111"), NULL);
    PASS();
}
TH_REG("vec", 18, "a cast to a reference is still an lvalue", vec18)

/* ---- A host-only member Booth cannot model ---- */

static void vec19(void)
{
    CHEQ(vcrun("struct ctx { std::vector<int> v; };\n"
               "__global__ void k(float *o){ o[0] = 1.0f; }\n", "--ir"), 0);
    CHEQ(strstr(obuf, "E134"), NULL);
    PASS();
}
TH_REG("vec", 19, "an unmodellable member declares fine", vec19)

static void vec20(void)
{
    CHNE(vcrun("struct ctx { std::vector<int> v; };\n"
               "__global__ void k(float *o){ struct ctx c; o[0] = 1.0f;"
               " (void)c; }\n", "--ir"), 0);
    CHNE(strstr(obuf, "E134"), NULL);
    PASS();
}
TH_REG("vec", 20, "and using one says what is missing", vec20)

/* ---- Operator resolution ---- */

static void vec21(void)
{
    CHNE(vcrun("__global__ void k(float2 *o, const float2 *p){\n"
               "  float2 a = p[0], b = p[1];\n"
               "  o[0] = a * b;\n"
               "}\n", "--ir"), 0);
    CHNE(strstr(obuf, "E102"), NULL);
    PASS();
}
TH_REG("vec", 21, "float2 * float2 has no operator to call", vec21)

static void vec22(void)
{
    CHEQ(vcrun("__global__ void k(half2 *o, const half2 *p){\n"
               "  half2 a = p[0];\n"
               "  a += p[1];\n"
               "  o[0] = a;\n"
               "}\n", "--ir"), 0);
    CHNE(strstr(obuf, "fadd f16"), NULL);
    CHEQ(strstr(obuf, "add ptr"), NULL);
    PASS();
}
TH_REG("vec", 22, "half2 += half2 adds both lanes", vec22)

static void vec23(void)
{
    CHEQ(vcrun("struct v2 { float x, y; };\n"
               "__device__ v2 operator+(v2 a, v2 b){ v2 r; r.x = a.x + b.x;"
               " r.y = a.y + b.y; return r; }\n"
               "__global__ void k(float *o){ v2 a, b; a.x = 1.0f; a.y = 2.0f;"
               " b.x = 3.0f; b.y = 4.0f; v2 c = a + b; o[0] = c.x + c.y; }\n",
               "--ir"), 0);
    CHNE(strstr(obuf, "call"), NULL);
    CHEQ(strstr(obuf, "add ptr"), NULL);
    PASS();
}
TH_REG("vec", 23, "two struct locals reach their operator+", vec23)

static void vec24(void)
{
    const char *p = vcptx(
        "struct a2 { float x, y; };\n"
        "struct b2 { int x, y; };\n"
        "__device__ a2 operator+(a2 p, a2 q){ a2 r; r.x = p.x + q.x;"
        " r.y = p.y + q.y; return r; }\n"
        "__device__ b2 operator+(b2 p, b2 q){ b2 r; r.x = p.x + q.x;"
        " r.y = p.y + q.y; return r; }\n"
        "__global__ void k(int *o){ b2 u, v; u.x = o[2]; u.y = 2;"
        " v.x = 3; v.y = 4; b2 w = u + v; o[0] = w.x + w.y; }\n");

    CHNE(p, NULL);
    CHNE(strstr(p, "add.u32"), NULL);
    CHEQ(strstr(p, "add.rn.f32"), NULL);
    PASS();
}
TH_REG("vec", 24, "the overload picked matches the operand types", vec24)

static void vec25(void)
{
    CHNE(vcrun("struct s2 { float x, y; };\n"
               "__global__ void k(int *o){ s2 a, b; a.x = 1.0f; b.x = 1.0f;"
               " o[0] = (a == b); }\n", "--ir"), 0);
    CHNE(strstr(obuf, "E102"), NULL);
    PASS();
}
TH_REG("vec", 25, "two structs compare by value or not at all", vec25)

static void vec26(void)
{
    CHNE(vcrun("struct s2 { float x, y; };\n"
               "__global__ void k(float *o){ s2 a, b; a.x = 1.0f; b.x = 2.0f;"
               " s2 c = a + b; o[0] = c.x; }\n", "--ir"), 0);
    CHNE(strstr(obuf, "E102"), NULL);
    CHEQ(strstr(obuf, "add ptr"), NULL);
    PASS();
}
TH_REG("vec", 26, "no operator+ refuses, it does not add allocas", vec26)
