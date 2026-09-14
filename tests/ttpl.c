/* ttpl.c -- class templates, template-ids and the C++ around them */

#include "tharns.h"

static char obuf[1 << 16];
static char tpx[1 << 16];

static int tprun(const char *src, const char *args)
{
    char cmd[512];
    FILE *f = fopen("build/ttpl.cu", "w");

    if (!f) return -1;
    fputs(src, f);
    fclose(f);
    snprintf(cmd, sizeof cmd, "%s %s build/ttpl.cu", BC_BIN, args);
    return th_run(cmd, obuf, (int)sizeof obuf);
}

static int tpptx(const char *src)
{
    FILE *fp;
    size_t n;

    if (tprun(src, "--nvidia-ptx -o build/ttpl.ptx") != 0) return -1;
    fp = fopen("build/ttpl.ptx", "rb");
    if (!fp) return -1;
    n = fread(tpx, 1, sizeof tpx - 1, fp);
    fclose(fp);
    tpx[n] = '\0';
    return 0;
}

/* ---- Template-ids ---- */

static void tpl01(void)
{
    CHEQ(tprun("template<int W> __device__ int g(int x){ return x + W; }\n"
               "__device__ int h(int a){ return g<32>(a); }\n", "--parse"), 0);
    CHNE(strstr(obuf, "template_args"), NULL);
    CHEQ(strstr(obuf, "binary <"), NULL);
    CHEQ(strstr(obuf, "binary >"), NULL);
    PASS();
}
TH_REG("tpl", 1, "an explicit argument list is not a comparison", tpl01)

static void tpl02(void)
{
    CHEQ(tprun("__device__ int h(int a, int b){ return a < b; }\n",
               "--parse"), 0);
    CHNE(strstr(obuf, "binary <"), NULL);
    PASS();
}
TH_REG("tpl", 2, "and a comparison is still a comparison", tpl02)

static void tpl03(void)
{
    CHEQ(tprun("template<int N> struct unrl { int v; };\n"
               "template<> struct unrl<1> { int w; };\n", "--parse"), 0);
    PASS();
}
TH_REG("tpl", 3, "a full specialisation names its arguments", tpl03)

static void tpl04(void)
{
    CHEQ(tprun("template<int N> struct unrl { };\n"
               "__device__ void g(int n){ unrl<3>{}; (void)n; }\n",
               "--parse"), 0);
    PASS();
}
TH_REG("tpl", 4, "a braced temporary is not a declaration", tpl04)

static void tpl05(void)
{
    CHEQ(tprun("template<typename A> struct pr { A a; };\n"
               "struct S { pr<pr<int>> n; };\n", "--parse"), 0);
    PASS();
}
TH_REG("tpl", 5, "two closing angles come out of one shift token", tpl05)

/* ---- Template parameters ---- */

static void tpl06(void)
{
    CHEQ(tprun("template<int W = 32> __device__ int g(int x){ return x + W; }\n"
               "__device__ int h(int a){ return g<8>(a); }\n", "--parse"), 0);
    PASS();
}
TH_REG("tpl", 6, "a default argument stops at the closing angle", tpl06)

static void tpl07(void)
{
    CHEQ(tprun("template<typename T, typename... Ts>\n"
               "inline constexpr bool anyof = (std::is_same_v<T, Ts> || ...);\n",
               "--parse"), 0);
    CHNE(strstr(obuf, "fold"), NULL);
    PASS();
}
TH_REG("tpl", 7, "a fold over a variable template is a fold", tpl07)

/* ---- Scoped enumerations ---- */

static void tpl08(void)
{
    CHEQ(tprun("enum class brm : int { MAX, SUM, };\n"
               "__device__ int g(void){ return (int)brm::SUM; }\n",
               "--parse"), 0);
    PASS();
}
TH_REG("tpl", 8, "enum class takes a base and a trailing comma", tpl08)

/* ---- Refusals ----
 * A dropped construct is a wrong answer wearing a green tick. */

static void tpl09(void)
{
    CHNE(tprun("struct S { std::vector<int> v; };\n"
               "__device__ int g(void){ S s; return s.v; }\n", "--ir"), 0);
    CHNE(strstr(obuf, "E134"), NULL);
    PASS();
}
TH_REG("tpl", 9, "a std facility refuses by name", tpl09)

static void tpl10(void)
{
    CHEQ(tprun("template<int N> struct unrl {\n"
               "    static constexpr int n = N; int v; };\n"
               "__device__ int g(void){ unrl<2> u;\n"
               "    u.v = unrl<2>::n; return u.v; }\n", "--ir"), 0);
    CHEQ(strstr(obuf, "E133"), NULL);
    CHNE(strstr(obuf, "store i32 2"), NULL);
    PASS();
}
TH_REG("tpl", 10, "a class template instance takes its argument", tpl10)

/* A partial specialisation on an enum reaches its own kernel */

static void tpl28(void)
{
    CHEQ(tpptx("enum class meth { MAX, SUM };\n"
               "template<meth m, typename T> struct pol;\n"
               "template<typename T> struct pol<meth::SUM, T> {\n"
               "    static __device__ T red(T v){ return v + (T)11; } };\n"
               "template<typename T> struct pol<meth::MAX, T> {\n"
               "    static __device__ T red(T v){ return v * (T)29; } };\n"
               "__global__ void k(float *o){\n"
               "    o[0] = pol<meth::SUM, float>::red(o[0]);\n"
               "    o[1] = pol<meth::MAX, float>::red(o[1]); }\n"), 0);
    CHNE(strstr(tpx, "0f41300000"), NULL);
    CHNE(strstr(tpx, "0f41E80000"), NULL);
    PASS();
}
TH_REG("tpl", 28, "two specialisations reach two kernels", tpl28)

/* Ambiguous pairs and declared-only primaries refuse */

static void tpl29(void)
{
    CHNE(tprun("template<typename A, typename B> struct pr;\n"
               "template<typename A> struct pr<A, float> {\n"
               "    static __device__ int v(void){ return 1; } };\n"
               "template<typename B> struct pr<float, B> {\n"
               "    static __device__ int v(void){ return 2; } };\n"
               "__device__ int g(void){ return pr<float, float>::v(); }\n",
               "--ir"), 0);
    CHNE(strstr(obuf, "E320"), NULL);
    PASS();
}
TH_REG("tpl", 29, "an ambiguous specialisation refuses by name", tpl29)

/* Static members belong to the instance that declared them */

static void tpl30(void)
{
    CHEQ(tprun("template<int I, int J> struct tl {\n"
               "    static constexpr int w = I;\n"
               "    static constexpr int n = I * J;\n"
               "    static __device__ int gi(int l){ return l % w + n; } };\n"
               "typedef tl<16, 8> tA;\n"
               "typedef tl<32, 4> tB;\n"
               "__global__ void k(int *o){\n"
               "    o[0] = tA::w + tA::n;\n"
               "    o[1] = tB::w + tB::n;\n"
               "    o[2] = tA::gi(3) + tB::gi(3); }\n", "--ir"), 0);
    CHNE(strstr(obuf, "store i32 144"), NULL);
    CHNE(strstr(obuf, "store i32 160"), NULL);
    CHNE(strstr(obuf, "srem i32 %0, 16"), NULL);
    CHNE(strstr(obuf, "srem i32 %0, 32"), NULL);
    PASS();
}
TH_REG("tpl", 30, "static members follow their own instance", tpl30)

static void tpl31(void)
{
    CHNE(tprun("template<typename T> struct only;\n"
               "__device__ int g(void){ return only<float>::v; }\n",
               "--ir"), 0);
    CHNE(strstr(obuf, "E1"), NULL);
    PASS();
}
TH_REG("tpl", 31, "a declared-only class template refuses", tpl31)

/* The more specialised partial wins, after defaults are filled in */

static void tpl32(void)
{
    CHEQ(tprun("enum dl { IMAJ = 0, JMAJ = 10 };\n"
               "template<int I, int J, typename T, dl d = IMAJ> struct tile {};\n"
               "template<int I, int J, typename T> struct tile<I, J, T, IMAJ> {\n"
               "    static __device__ int gi(int l){ return l % I + 1; } };\n"
               "template<int I, int J> struct tile<I, J, float, IMAJ> {\n"
               "    static __device__ int gi(int l){ return l % I + 2; } };\n"
               "template<int I, int J, typename T> struct tile<I, J, T, JMAJ> {\n"
               "    static __device__ int gi(int l){ return l % I + 3; } };\n"
               "__global__ void k(int *o){\n"
               "    o[0] = tile<16, 8, int>::gi(3);\n"
               "    o[1] = tile<16, 8, float>::gi(3);\n"
               "    o[2] = tile<16, 8, int, JMAJ>::gi(3); }\n", "--ir"), 0);
    CHNE(strstr(obuf, "add i32 %1, 1"), NULL);
    CHNE(strstr(obuf, "add i32 %1, 2"), NULL);
    CHNE(strstr(obuf, "add i32 %1, 3"), NULL);
    PASS();
}
TH_REG("tpl", 32, "the more specialised partial wins", tpl32)

static void tpl11(void)
{
    CHNE(tprun("__device__ int g(int n){\n"
               "    int t = 0;\n"
               "    for (int v : n) t += v;\n"
               "    return t;\n"
               "}\n", "--ir"), 0);
    CHNE(strstr(obuf, "E135"), NULL);
    PASS();
}
TH_REG("tpl", 11, "a range-based for refuses by name", tpl11)

static void tpl12(void)
{
    CHNE(tprun("__device__ int g(void){ dim3 b(4, 1, 1); return (int)b.x; }\n",
               "--ir"), 0);
    CHNE(strstr(obuf, "E136"), NULL);
    PASS();
}
TH_REG("tpl", 12, "a parenthesised initialiser refuses by name", tpl12)

static void tpl13(void)
{
    CHEQ(tprun("__device__ int g(int a){\n"
               "    if constexpr (sizeof(int) > 2) return 111; else return a;\n"
               "}\n"
               "__global__ void k(int *o){ o[0] = g(3); }\n", "--ir"), 0);
    CHNE(strstr(obuf, "ret i32 111"), NULL);
    PASS();
}
TH_REG("tpl", 13, "a sizeof condition picks its arm", tpl13)

static void tpl14(void)
{
    CHEQ(tprun("template<int A> __device__ int g(void){\n"
               "    if constexpr (A > 2) return 1; else return 0;\n"
               "}\n", "--ir"), 0);
    PASS();
}
TH_REG("tpl", 14, "and stays quiet where no code does", tpl14)

static void tpl15(void)
{
    CHEQ(tprun("enum class brm { LO, HI };\n"
               "enum class oth { HI, LO };\n"
               "__global__ void k(int *o){ o[0] = (int)brm::HI;\n"
               "                           o[1] = (int)oth::HI; }\n",
               "--ir"), 0);
    CHNE(strstr(obuf, "store i32 1"), NULL);
    CHNE(strstr(obuf, "store i32 0"), NULL);
    PASS();
}
TH_REG("tpl", 15, "two scoped enums do not share a name", tpl15)

static void tpl16(void)
{
    CHNE(tprun("enum class brm { LO, HI };\n"
               "__global__ void k(int *o){ o[0] = (int)HI; }\n",
               "--ir"), 0);
    CHNE(strstr(obuf, "E101"), NULL);
    PASS();
}
TH_REG("tpl", 16, "and neither reaches the enclosing scope", tpl16)

static void tpl17(void)
{
    CHEQ(tprun("template<int K> __device__ float sc(float x){ return x*K; }\n"
               "typedef float (*fn_t)(float);\n"
               "template<fn_t F> __global__ void k(float *d){ d[0]=F(d[0]); }\n"
               "void ga(float *d){ k<sc<3>><<<1,1>>>(d); }\n"
               "void gb(float *d){ k<sc<7>><<<1,1>>>(d); }\n", "--ir"), 0);
    CHNE(strstr(obuf, "fmul f32 %0, 3"), NULL);
    CHNE(strstr(obuf, "fmul f32 %0, 7"), NULL);
    CHNE(strstr(obuf, "@k$1"), NULL);
    PASS();
}
TH_REG("tpl", 17, "a template-id argument keeps its arguments", tpl17)

static void tpl18(void)
{
    CHEQ(tprun("static constexpr __device__ int w(void){ return 47; }\n"
               "template<int W> __device__ float g(float x){ return x+W; }\n"
               "__global__ void k(float *d){\n"
               "    constexpr int n = w();\n"
               "    d[0] = g<n>(d[0]);\n"
               "}\n", "--ir"), 0);
    CHNE(strstr(obuf, "fadd f32 %0, 47"), NULL);
    PASS();
}
TH_REG("tpl", 18, "a constexpr call folds as a template argument", tpl18)

static void tpl19(void)
{
    CHEQ(tprun("namespace ns {\n"
               "    __device__ float one(float x){ return x*11; }\n"
               "    __device__ float two(float x){ return x*23; }\n"
               "}\n"
               "__global__ void k(float *d){\n"
               "    d[0] = ns::one(d[0]);\n"
               "    using ns::two;\n"
               "    d[1] = two(d[1]);\n"
               "}\n", "--ir"), 0);
    CHNE(strstr(obuf, "fmul f32 %0, 11"), NULL);
    CHNE(strstr(obuf, "fmul f32 %0, 23"), NULL);
    PASS();
}
TH_REG("tpl", 19, "a namespace function is reachable both ways", tpl19)

static void tpl20(void)
{
    CHNE(tprun("__global__ void k(int *o){ o[0] = nosuch(1); }\n", "--ir"), 0);
    CHNE(strstr(obuf, "E105"), NULL);
    CHNE(strstr(obuf, "'nosuch'"), NULL);
    PASS();
}
TH_REG("tpl", 20, "an unknown call names what it could not find", tpl20)

/* ---- Template parameters, arguments and the names they own ---- */

static void tpl21(void)
{
    CHEQ(tprun("template<class op> static void disp(int n){ op()(n); }\n"
               "template<int W> __device__ int g(int x){ return x * W; }\n"
               "__device__ int op(int x){ return g<5>(x); }\n"
               "__global__ void k(int *o){ o[0] = op(o[0]); }\n", "--ir"), 0);
    CHEQ(strstr(obuf, "E155"), NULL);
    CHNE(strstr(obuf, "mul i32 %0, 5"), NULL);
    PASS();
}
TH_REG("tpl", 21, "a type parameter's name ends with its template", tpl21)

static void tpl22(void)
{
    CHEQ(tprun("template<int scale = 7>\n"
               "__global__ void k(int *d, const int *a)"
               " { d[threadIdx.x] = a[threadIdx.x] * scale; }\n"
               "void go(int *d, const int *a){ k<><<<1,32>>>(d, a); }\n",
               "--ir"), 0);
    CHNE(strstr(obuf, "mul i32 %6, 7"), NULL);
    CHEQ(strstr(obuf, "mul i32 %6, 0"), NULL);
    PASS();
}
TH_REG("tpl", 22, "a defaulted argument reaches the body", tpl22)

static void tpl23(void)
{
    CHEQ(tpptx(
        "__device__ float opa(const float a, const float b)"
        " { return a + b * 11.0f; }\n"
        "__device__ float opm(const float a, const float b)"
        " { return a * b + 23.0f; }\n"
        "template <float (*bop)(const float, const float), typename T>\n"
        "__global__ void kb(T *d, const T *a, const T *b)"
        " { int i = threadIdx.x; d[i] = (T) bop((float)a[i], (float)b[i]); }\n"
        "template <float (*bop)(const float, const float), typename T>\n"
        "static void lb(T *d, const T *a, const T *b, int)"
        " { kb<bop, T><<<1,64>>>(d, a, b); }\n"
        "template <float (*op)(const float, const float), int n>\n"
        "static void fu(float *d, const float *a, const float *b)"
        " { lb<op, float>(d, a, b, n); }\n"
        "void ga(float *d, const float *a, const float *b)"
        " { fu<opa, 2>(d, a, b); }\n"
        "void gm(float *d, const float *a, const float *b)"
        " { fu<opm, 3>(d, a, b); }\n"), 0);
    CHNE(strstr(tpx, ".entry kb ("), NULL);
    CHNE(strstr(tpx, ".entry kb$1 ("), NULL);
    CHNE(strstr(tpx, "0f41300000"), NULL);
    CHNE(strstr(tpx, "0f41B80000"), NULL);
    PASS();
}
TH_REG("tpl", 23, "two template arguments make two kernels", tpl23)

static void tpl24(void)
{
    CHEQ(tprun("static void two(int a, int b){ (void)a; (void)b; }\n"
               "static void two(int a, int b, int c, int d)"
               " { (void)a; (void)b; (void)c; (void)d; }\n"
               "__global__ void k(int *o){ o[0] = 1; }\n"
               "void go(void){ two(1,2); two(1,2,3,4); k<<<1,1>>>(0); }\n",
               "--ir"), 0);
    CHEQ(strstr(obuf, "E073"), NULL);
    PASS();
}
TH_REG("tpl", 24, "an overload set is chosen by arity", tpl24)

static void tpl25(void)
{
    CHEQ(tprun("__device__ float sw(float x, float g, float alpha = 1.702f)"
               " { return x * g * alpha; }\n"
               "__global__ void k(float *d, const float *a, const float *b)"
               " { int i = threadIdx.x; d[i] = sw(a[i], b[i]); }\n",
               "--ir"), 0);
    CHNE(strstr(obuf, "@sw(%6, %8, 1.702)"), NULL);
    PASS();
}
TH_REG("tpl", 25, "a default argument arrives at the call", tpl25)

static void tpl26(void)
{
    CHEQ(tprun("template<int n>\n"
               "static __device__ void cp(char *d, const char *s)"
               " { for (int i = 0; i < n; ++i) d[i] = s[i]; }\n"
               "__global__ void k(char *o, const char *a){\n"
               "    int wire[4];\n"
               "    cp<sizeof(wire)>(o, a);\n"
               "    cp<4>(o + 16, a + 16);\n"
               "}\n", "--ir"), 0);
    CHNE(strstr(obuf, "slt i32 %3, 16"), NULL);
    CHNE(strstr(obuf, "slt i32 %3, 4"), NULL);
    PASS();
}
TH_REG("tpl", 26, "an array sizeof gets its own body", tpl26)

static void tpl27(void)
{
    CHNE(tprun("template<typename T, int W>\n"
               "__global__ void k(T *d){ d[0] = (T)W; }\n"
               "void go(float *d){ k<float><<<1,1>>>(d); }\n", "--ir"), 0);
    CHNE(strstr(obuf, "E030"), NULL);
    CHNE(strstr(obuf, "cannot deduce"), NULL);
    PASS();
}
TH_REG("tpl", 27, "a launch names what it could not deduce", tpl27)
