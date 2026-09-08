/* tcxp.c -- constexpr objects
 *
 * The old behaviour is what these are really about. A namespace-scope
 * constexpr was collected by nobody, so the declaration went nowhere and
 * the use came back as an undefined variable; an array bound naming one
 * fell through to the scalar path and the bracket became an initialiser,
 * which is a wrong answer rather than a refusal. Both shapes are below. */

#include "tharns.h"

static char obuf[1 << 16];

static int cxrun(const char *src, const char *args)
{
    char cmd[512];
    FILE *f = fopen("build/tcxp.cu", "w");

    if (!f) return -1;
    fputs(src, f);
    fclose(f);
    snprintf(cmd, sizeof cmd, "%s %s build/tcxp.cu", BC_BIN, args);
    return th_run(cmd, obuf, (int)sizeof obuf);
}

/* ---- Binding and folding ---- */

static void cxp01(void)
{
    CHEQ(cxrun("constexpr int N = 4;\n"
               "__device__ int f(){ return N; }\n", "--ir"), 0);
    CHNE(strstr(obuf, "ret i32 4"), NULL);
    PASS();
}
TH_REG("cxp", 1, "a namespace-scope constexpr binds its name", cxp01)

static void cxp02(void)
{
    static const struct { const char *ini; const char *ir; } cs[] = {
        { "2 + 3 * 2",          "ret i32 8"          },
        { "(1 << 10) - 1",      "ret i32 1023"       },
        { "-7 / 2",             "ret i32 -3"         },
        { "17 % 5",             "ret i32 2"          },
        { "~0 & 0xFF",          "ret i32 255"        },
        { "3 > 2 ? 11 : 22",    "ret i32 11"         },
        { "!0 + (2 <= 1)",      "ret i32 1"          },
        { "(int)3.9",           "ret i32 3"          },
        { "'A'",                "ret i32 65"         },
    };

    for (size_t i = 0; i < sizeof cs / sizeof cs[0]; i++) {
        char src[256];
        snprintf(src, sizeof src,
                 "constexpr int N = %s;\n"
                 "__device__ int f(){ return N; }\n", cs[i].ini);
        CHEQ(cxrun(src, "--ir"), 0);
        CHNE(strstr(obuf, cs[i].ir), NULL);
    }
    PASS();
}
TH_REG("cxp", 2, "an initialiser folds the usual operators", cxp02)

static void cxp03(void)
{
    CHEQ(cxrun("constexpr int A = 4;\n"
               "constexpr int B = A * A;\n"
               "constexpr int C = B + A;\n"
               "__device__ int f(){ return C; }\n", "--ir"), 0);
    CHNE(strstr(obuf, "ret i32 20"), NULL);
    PASS();
}
TH_REG("cxp", 3, "one constexpr reads another", cxp03)

static void cxp04(void)
{
    CHEQ(cxrun("constexpr float F = 1.5f;\n"
               "constexpr double D = F * 2;\n"
               "__device__ double f(){ return D; }\n", "--ir"), 0);
    CHNE(strstr(obuf, "ret f64 3"), NULL);
    PASS();
}
TH_REG("cxp", 4, "a float constexpr keeps its declared width", cxp04)

static void cxp05(void)
{
    CHEQ(cxrun("constexpr size_t Z = 1ull << 20;\n"
               "__device__ long f(){ return (long)Z; }\n", "--ir"), 0);
    CHNE(strstr(obuf, "ret i64 1048576"), NULL);
    PASS();
}
TH_REG("cxp", 5, "the declared type sets the constant's width", cxp05)

static void cxp06(void)
{
    CHEQ(cxrun("namespace ns { constexpr int N = 6; }\n"
               "__device__ int f(){ return ns::N; }\n", "--ir"), 0);
    CHNE(strstr(obuf, "ret i32 6"), NULL);

    CHEQ(cxrun("namespace ns { constexpr int N = 6; }\n"
               "__device__ int f(){ return N; }\n", "--ir"), 0);
    CHNE(strstr(obuf, "ret i32 6"), NULL);
    PASS();
}
TH_REG("cxp", 6, "a namespace member answers to either spelling", cxp06)

static void cxp07(void)
{
    CHEQ(cxrun("struct S { static constexpr int qk = 8; };\n"
               "__device__ int f(){ return S::qk; }\n", "--ir"), 0);
    CHNE(strstr(obuf, "ret i32 8"), NULL);
    PASS();
}
TH_REG("cxp", 7, "a static constexpr member binds a value", cxp07)

static void cxp08(void)
{
    CHEQ(cxrun("__device__ int f(){ constexpr int N = 4; return N * 2; }\n",
               "--ir"), 0);
    CHNE(strstr(obuf, "i32 4"), NULL);
    CHEQ(strstr(obuf, "E1"), NULL);
    PASS();
}
TH_REG("cxp", 8, "a block-scope constexpr keeps its storage", cxp08)

static void cxp09(void)
{
    CHEQ(cxrun("const int N = 5;\n"
               "__device__ int f(){ return N; }\n", "--ir"), 0);
    CHNE(strstr(obuf, "ret i32 5"), NULL);
    PASS();
}
TH_REG("cxp", 9, "a const with a constant initialiser folds too", cxp09)

/* ---- Array bounds ----
 *
 * float a[N] used to leave is_array unset and hand the bound to the
 * scalar path as an initialiser, so a became a float holding N and a[0]
 * indexed off that value as though it were an address. The alloca type
 * is what says it is fixed. */

static void cxp10(void)
{
    CHEQ(cxrun("__device__ float f(){\n"
               "    constexpr int N = 4;\n"
               "    float a[N];\n"
               "    a[3] = 1.0f;\n"
               "    return a[3];\n"
               "}\n", "--ir --no-mem2reg --no-cfold"), 0);
    CHNE(strstr(obuf, "alloca ptr<private, [4 x f32]>"), NULL);
    CHEQ(strstr(obuf, "store i32 4, %2"), NULL);
    PASS();
}
TH_REG("cxp", 10, "a constexpr array bound allocates an array", cxp10)

static void cxp11(void)
{
    CHEQ(cxrun("constexpr int R = 2;\n"
               "constexpr int C = 3;\n"
               "__device__ float f(){ float a[R][C]; a[1][2] = 1.0f;\n"
               "                      return a[1][2]; }\n",
               "--ir --no-mem2reg --no-cfold"), 0);
    CHNE(strstr(obuf, "[2 x [3 x f32]]"), NULL);
    PASS();
}
TH_REG("cxp", 11, "both bounds of a two-dimensional array fold", cxp11)

/* ---- Refusals ---- */

static void cxp12(void)
{
    CHNE(cxrun("__device__ int g();\n"
               "constexpr int N = g();\n"
               "__device__ int f(){ return N; }\n", "--ir"), 0);
    CHNE(strstr(obuf, "E128"), NULL);
    PASS();
}
TH_REG("cxp", 12, "an unfoldable initialiser refuses by name", cxp12)

static void cxp13(void)
{
    CHEQ(cxrun("__device__ int g();\n"
               "constexpr int N = g();\n"
               "__device__ int f(){ return 1; }\n", "--ir"), 0);
    CHEQ(strstr(obuf, "E128"), NULL);
    PASS();
}
TH_REG("cxp", 13, "and costs nothing while nothing reads it", cxp13)

static void cxp14(void)
{
    CHNE(cxrun("__device__ float f(int k){ float a[k]; return a[0]; }\n",
               "--ir"), 0);
    CHNE(strstr(obuf, "E131"), NULL);
    PASS();
}
TH_REG("cxp", 14, "a bound that is not constant refuses by name", cxp14)

static void cxp15(void)
{
    CHEQ(cxrun("template<int A> __global__ void f(int *o){\n"
               "    if constexpr (A > 2) o[0] = 111; else o[0] = 222;\n"
               "}\n"
               "int main(void){ int *d = 0; f<4><<<1,1>>>(d); return 0; }\n",
               "--ir"), 0);
    CHNE(strstr(obuf, "111"), NULL);
    CHEQ(strstr(obuf, "222"), NULL);
    PASS();
}
TH_REG("cxp", 15, "if constexpr keeps one arm and drops the other", cxp15)

/* ---- Layout ----
 *
 * A static member counted as a field, so every field after it sat one
 * slot too far along and read back whatever the neighbour held. */

static void cxp16(void)
{
    CHEQ(cxrun("struct S { static constexpr int qk = 8; float x; float y; };\n"
               "__device__ float f(S *s){ s->x = 1.0f; return s->x; }\n",
               "--ir --no-mem2reg --no-cfold"), 0);
    CHNE(strstr(obuf, "gep ptr<global, f32>, %3, 0"), NULL);
    PASS();
}
TH_REG("cxp", 16, "a static member takes no slot in the layout", cxp16)

/* ---- Things that must not have moved ---- */

static void cxp17(void)
{
    CHEQ(cxrun("__device__ float f(){ float a[] = {1.0f, 2.0f};\n"
               "                      return a[1]; }\n", "--ir"), 0);
    CHEQ(strstr(obuf, "E131"), NULL);

    CHEQ(cxrun("__device__ int f(){ int a[3] = {1,2,3}; return a[1]; }\n",
               "--ir --no-mem2reg --no-cfold"), 0);
    CHNE(strstr(obuf, "alloca ptr<private, [3 x i32]>"), NULL);
    PASS();
}
TH_REG("cxp", 17, "a braced initialiser is not read as a bound", cxp17)

static void cxp18(void)
{
    CHEQ(cxrun("__device__ int f(const float *p){ return (int)p[0]; }\n",
               "--ir"), 0);
    CHEQ(strstr(obuf, "E128"), NULL);

    CHEQ(cxrun("__device__ int f(){ const float *p = 0; return p != 0; }\n",
               "--ir"), 0);
    CHEQ(strstr(obuf, "E128"), NULL);
    PASS();
}
TH_REG("cxp", 18, "a const pointer is storage, not a folded value", cxp18)

static void cxp19(void)
{
    CHEQ(cxrun("constexpr int N = 8;\n"
               "__device__ float g[N];\n"
               "__device__ float f(){ return g[3]; }\n", "--ir"), 0);
    CHNE(strstr(obuf, "@g = global [8 x f32]"), NULL);
    PASS();
}
TH_REG("cxp", 19, "a device global takes a constexpr bound", cxp19)

static void cxp20(void)
{
    CHEQ(cxrun("__device__ int x = 5;\n"
               "__device__ int f(){ return x; }\n", "--ir"), 0);
    CHNE(strstr(obuf, "@x = global i32 5"), NULL);
    CHEQ(strstr(obuf, "[5 x i32]"), NULL);
    PASS();
}
TH_REG("cxp", 20, "a global initialiser is not read as a bound", cxp20)

static void cxp21(void)
{
    CHEQ(cxrun("enum E { A = 3 };\n"
               "constexpr int N = A * 4;\n"
               "__device__ int f(){ return N; }\n", "--ir"), 0);
    CHNE(strstr(obuf, "ret i32 12"), NULL);
    PASS();
}
TH_REG("cxp", 21, "an enumerator reaches a constexpr initialiser", cxp21)

static void cxp22(void)
{
    CHNE(cxrun("constexpr unsigned char C = 300;\n"
               "__device__ int f(){ return C; }\n", "--ir"), 0);
    CHNE(strstr(obuf, "E128"), NULL);

    CHEQ(cxrun("constexpr unsigned char C = 200;\n"
               "__device__ unsigned char f(){ return C; }\n", "--ir"), 0);
    CHNE(strstr(obuf, "i8 200"), NULL);
    PASS();
}
TH_REG("cxp", 22, "a value too wide for its type refuses", cxp22)

/* ---- The function's own name ---- */

static void cxp23(void)
{
    CHEQ(cxrun("__device__ const char *w(){ return __FUNCTION__; }\n",
               "--ir"), 0);
    CHEQ(strstr(obuf, "E101"), NULL);
    CHNE(strstr(obuf, ".str.0"), NULL);
    CHEQ(cxrun("__device__ const char *w(){ return __FUNCTION__; }\n",
               "--nvidia-ptx -o build/tcxp.ptx"), 0);
    CHEQ(strstr(obuf, "E110"), NULL);
    PASS();
}
TH_REG("cxp", 23, "__FUNCTION__ is a string, not a stray name", cxp23)

static void cxp24(void)
{
    CHEQ(cxrun("__device__ int q(){ int __func__ = 3; return __func__; }\n",
               "--ir"), 0);
    CHEQ(strstr(obuf, ".str"), NULL);
    PASS();
}
TH_REG("cxp", 24, "a local of that name still wins", cxp24)

/* ---- Discarded statements ----
 *
 * An untaken arm is not code Booth folds away later, it is code Booth never
 * reads. cxp26 is the one that matters: the arm that is not chosen would not
 * compile for the type it was not chosen for. */

static const char *ptxof(const char *src)
{
    static char ptx[1 << 16];
    char cmd[512];
    FILE *f = fopen("build/tcxp.cu", "w");
    FILE *g;
    size_t n;

    ptx[0] = '\0';
    if (!f) return NULL;
    fputs(src, f);
    fclose(f);
    remove("build/tcxp2.ptx");
    snprintf(cmd, sizeof cmd,
             "%s --nvidia-ptx build/tcxp.cu -o build/tcxp2.ptx", BC_BIN);
    if (th_run(cmd, obuf, (int)sizeof obuf) != 0) return NULL;
    g = fopen("build/tcxp2.ptx", "rb");
    if (!g) return NULL;
    n = fread(ptx, 1, sizeof ptx - 1, g);
    fclose(g);
    ptx[n] = '\0';
    return ptx;
}

static const char *const ISSAME =
    "namespace std {\n"
    "template<class A, class B> inline constexpr bool is_same_v = false;\n"
    "template<class A> inline constexpr bool is_same_v<A,A> = true;\n"
    "}\n";

static void cxp25(void)
{
    char src[1024];

    snprintf(src, sizeof src, "%s%s", ISSAME,
             "template<typename T> __device__ float pick(T x){\n"
             "    if constexpr (std::is_same_v<T, float2>) return x.x + 5.0f;\n"
             "    else return x * 3.0f;\n"
             "}\n"
             "__global__ void kf(float *o, float a){ o[0] = pick<float>(a); }\n"
             "__global__ void kv(float *o, float2 a){ o[0] = pick<float2>(a); }\n");
    CHEQ(cxrun(src, "--ir"), 0);
    CHNE(strstr(obuf, "fmul f32"), NULL);
    CHNE(strstr(obuf, "fadd f32"), NULL);
    PASS();
}
TH_REG("cxp", 25, "is_same_v picks an arm per instantiation", cxp25)

static void cxp26(void)
{
    const char *p;
    char src[1024];

    snprintf(src, sizeof src, "%s%s", ISSAME,
             "template<typename T> __device__ float pick(T x){\n"
             "    if constexpr (std::is_same_v<T, float2>) return x.x + 5.0f;\n"
             "    else return nosuchfn(x) + 7.0f;\n"
             "}\n"
             "__global__ void kv(float *o, float2 a){ o[0] = pick<float2>(a); }\n");
    p = ptxof(src);
    CHNE(p, NULL);
    if (p) {
        CHNE(strstr(p, "0f40A00000"), NULL);   /* 5.0f, the taken arm */
        CHEQ(strstr(p, "0f40E00000"), NULL);   /* 7.0f, the arm not taken */
        CHEQ(strstr(p, "nosuchfn"), NULL);
    }
    PASS();
}
TH_REG("cxp", 26, "an arm that would not compile is never read", cxp26)

static void cxp27(void)
{
    CHNE(cxrun("__global__ void k(int *o, int n){\n"
               "    if constexpr (n > 2) o[0] = 1; else o[0] = 0;\n"
               "}\n", "--ir"), 0);
    CHNE(strstr(obuf, "E158"), NULL);
    CHNE(strstr(obuf, "n > 2"), NULL);
    PASS();
}
TH_REG("cxp", 27, "a condition that is not constant refuses", cxp27)

static void cxp28(void)
{
    CHEQ(cxrun("template<typename... A> __device__ int cnt(A... a){\n"
               "    if constexpr (sizeof...(a) > 1) return 111; else return 222;\n"
               "}\n"
               "__global__ void k(int *o){ o[0] = cnt(1,2,3); o[1] = cnt(9); }\n",
               "--ir"), 0);
    CHNE(strstr(obuf, "ret i32 111"), NULL);
    CHNE(strstr(obuf, "ret i32 222"), NULL);
    PASS();
}
TH_REG("cxp", 28, "sizeof... of a pack decides a condition", cxp28)

/* Two aggregates with identical layout are still two types, so the false
 * arm is the answer and the true arm must not be reached. */
static void cxp29(void)
{
    char src[1024];

    snprintf(src, sizeof src, "%s%s", ISSAME,
             "struct pa { float u; float v; };\n"
             "struct pb { float u; float v; };\n"
             "template<typename A, typename B> __device__ int q(A x, B y){\n"
             "    if constexpr (std::is_same_v<A, B>) return 111; else return 222;\n"
             "}\n"
             "__global__ void k(int *o, pa a, pb b){ o[0] = q(a, b); }\n");
    CHEQ(cxrun(src, "--ir"), 0);
    CHNE(strstr(obuf, "ret i32 222"), NULL);
    CHEQ(strstr(obuf, "ret i32 111"), NULL);
    PASS();
}
TH_REG("cxp", 29, "two layout-alike structs are not is_same_v", cxp29)

static void cxp30(void)
{
    CHEQ(cxrun("__global__ void k(long *o){\n"
               "    double d = 1.0; float f = 1.0f; double *p = 0;\n"
               "    o[0] = sizeof(d); o[1] = sizeof(f);\n"
               "    o[2] = sizeof(double *); o[3] = sizeof(p);\n"
               "}\n", "--ir"), 0);
    CHNE(strstr(obuf, "store i64 8"), NULL);
    CHNE(strstr(obuf, "store i64 4"), NULL);
    CHEQ(strstr(obuf, "store i64 4, %7"), NULL);
    PASS();
}
TH_REG("cxp", 30, "sizeof reads the type, not a guess of four", cxp30)

/* ---- Static data members ---- */

/* Every spelling of a read has to reach the same constant, and the proof is
 * the immediate in the .entry rather than anything the IR printer says. */
static void cxp31(void)
{
    const char *p = ptxof(
        "struct sc { static const int va = 9;\n"
        "            static constexpr float vf = 2.5f; };\n"
        "struct dd : sc { };\n"
        "typedef sc TT;\n"
        "template<int N> struct tp { static constexpr int vt = N * 3; };\n"
        "__global__ void k(int *o, float *f){ sc s;\n"
        "  o[0] = sc::va; o[1] = s.va; o[2] = TT::va; o[3] = dd::va;\n"
        "  o[4] = tp<7>::vt; f[0] = sc::vf; }\n");

    CHNE(p, NULL);
    if (p) {
        int n = 0;
        const char *q = p;
        while ((q = strstr(q, "mov.u32 %r")) != NULL) {
            if (strstr(q, ", 9;") == q + 11) n++;
            q += 10;
        }
        CHEQ(n, 4);
        CHNE(strstr(p, ", 21;"), NULL);
        CHNE(strstr(p, "0f40200000"), NULL);
    }
    PASS();
}
TH_REG("cxp", 31, "a static const member reaches PTX as a value", cxp31)

static void cxp32(void)
{
    const char *p = ptxof(
        "template<int N> struct sc { static constexpr int v = N * 2; };\n"
        "__global__ void k(int *o){ o[0] = sc<3>::v; o[1] = sc<5>::v; }\n");

    CHNE(p, NULL);
    if (p) {
        CHNE(strstr(p, ", 6;"), NULL);
        CHNE(strstr(p, ", 10;"), NULL);
    }
    PASS();
}
TH_REG("cxp", 32, "a specialisation gives its own value", cxp32)

static void cxp33(void)
{
    CHEQ(cxrun("struct sc { static constexpr int n = 4; };\n"
               "__global__ void k(int *o){ int a[sc::n];\n"
               "  a[0] = 1; a[3] = 2; o[0] = a[0] + a[3]; }\n", "--ir"), 0);
    CHNE(strstr(obuf, "[4 x i32]"), NULL);
    CHEQ(strstr(obuf, "E131"), NULL);
    PASS();
}
TH_REG("cxp", 33, "a static member sizes an array", cxp33)

static void cxp34(void)
{
    const char *p = ptxof(
        "struct sc { static constexpr int n = 9; };\n"
        "template<int N> __device__ int f(void){ return N + 1; }\n"
        "__global__ void k(int *o){ o[0] = f<sc::n>(); }\n");

    CHNE(p, NULL);
    if (p) CHNE(strstr(p, ", 10;"), NULL);
    PASS();
}
TH_REG("cxp", 34, "a static member is a template argument", cxp34)

static void cxp35(void)
{
    const char *p = ptxof(
        "struct sc { static constexpr int n = 9; };\n"
        "__global__ void k(int *o){\n"
        "  if constexpr (sc::n > 3) o[0] = 111; else o[0] = 222; }\n");

    CHNE(p, NULL);
    if (p) {
        CHNE(strstr(p, ", 111;"), NULL);
        CHEQ(strstr(p, ", 222;"), NULL);
    }
    PASS();
}
TH_REG("cxp", 35, "a static member decides if constexpr", cxp35)

/* The NVIDIA backend drops switch dispatch whatever the label is, literal
 * included, so the constant is only visible in the IR here. */
static void cxp36(void)
{
    CHEQ(cxrun("struct sc { static constexpr int n = 9; };\n"
               "__global__ void k(int *o, int x){\n"
               "  switch (x) { case sc::n: o[0] = 1; break;\n"
               "               default: o[0] = 2; } }\n", "--ir"), 0);
    CHNE(strstr(obuf, "[9: switch.case.0]"), NULL);
    PASS();
}
TH_REG("cxp", 36, "a static member is a case label", cxp36)

/* A declaration without an initialiser has no value at compile time, and a
 * mutable one has no constant value at all. Both say so by name. */
static void cxp37(void)
{
    CHNE(cxrun("struct sc { static const int v; };\n"
               "__global__ void k(int *o){ o[0] = sc::v; }\n", "--ir"), 0);
    CHNE(strstr(obuf, "E400"), NULL);
    CHNE(strstr(obuf, "sc::v"), NULL);
    PASS();
}
TH_REG("cxp", 37, "a member with no initialiser refuses", cxp37)

static void cxp38(void)
{
    CHNE(cxrun("struct sc { static int v; };\n"
               "__global__ void k(int *o){ o[0] = sc::v; }\n", "--ir"), 0);
    CHNE(strstr(obuf, "E401"), NULL);
    CHNE(strstr(obuf, "sc::v"), NULL);
    PASS();
}
TH_REG("cxp", 38, "a mutable static member refuses", cxp38)

static void cxp39(void)
{
    CHNE(cxrun("struct sc { static constexpr int v[3] = {1, 2, 3}; };\n"
               "__global__ void k(int *o){ o[0] = sc::v[1]; }\n", "--ir"), 0);
    CHNE(strstr(obuf, "E402"), NULL);
    CHNE(strstr(obuf, "sc::v"), NULL);
    PASS();
}
TH_REG("cxp", 39, "a static array member will not fold", cxp39)

static void cxp40(void)
{
    CHNE(cxrun("template<int N> struct b { static constexpr int v = N; };\n"
               "struct d : b<4> { };\n"
               "__global__ void k(int *o){ o[0] = d::v; }\n", "--ir"), 0);
    CHNE(strstr(obuf, "E403"), NULL);
    CHNE(strstr(obuf, "d::v"), NULL);
    PASS();
}
TH_REG("cxp", 40, "a base Booth cannot name refuses", cxp40)

static void cxp41(void)
{
    CHEQ(cxrun("struct sc { static constexpr float v = 1.5f; };\n"
               "__global__ void k(float *o){ decltype(sc::v) a = 7.0f;\n"
               "  o[0] = a; }\n", "--ir"), 0);
    CHNE(strstr(obuf, "store f32 7"), NULL);
    CHEQ(strstr(obuf, "E360"), NULL);
    PASS();
}
TH_REG("cxp", 41, "decltype names a static member's type", cxp41)
/* Only a body of one return statement folded, so a constexpr member function
 * that answers with an if-chain was not a constant expression and the arm was
 * never chosen. Each struct must read its own I, not the last one collected. */
static void cxp42(void)
{
    CHEQ(cxrun("struct s8 { static constexpr int I = 8;\n"
               "  static constexpr __device__ bool supp(){"
               " if (I == 8) return true; return false; } };\n"
               "struct s5 { static constexpr int I = 5;\n"
               "  static constexpr __device__ bool supp(){"
               " if (I == 8) return true; return false; } };\n"
               "__global__ void k(int *o){\n"
               "  if constexpr (s8::supp()) o[0] = 11; else o[0] = 22;\n"
               "  if constexpr (s5::supp()) o[1] = 33; else o[1] = 44; }\n",
               "--ir"), 0);
    CHNE(strstr(obuf, "store i32 11"), NULL);
    CHNE(strstr(obuf, "store i32 44"), NULL);
    CHEQ(strstr(obuf, "store i32 22"), NULL);
    CHEQ(strstr(obuf, "store i32 33"), NULL);
    PASS();
}
TH_REG("cxp", 42, "a member function with an if-chain folds", cxp42)

/* A loop is past what the evaluator walks, and a guessed answer would pick an
 * arm at random. The refusal names the condition it could not fold. */
static void cxp43(void)
{
    CHNE(cxrun("__device__ int loopy(int a){"
               " for (int i = 0; i < 3; i++) a++; return a; }\n"
               "__global__ void k(int *o){"
               " if constexpr (loopy(1) == 4) o[0] = 55; else o[0] = 66; }\n",
               "--ir"), 0);
    CHNE(strstr(obuf, "E158"), NULL);
    PASS();
}
TH_REG("cxp", 43, "a loop in a constexpr body is refused", cxp43)

