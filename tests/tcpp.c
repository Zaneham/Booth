/* tcpp.c -- the host C++ a .cu file drags in, and where Booth draws the line
 *
 * Every shape here came out of the ggml-cuda tree, where a single unparsed
 * construct produced a page of E020/E022/E024 and the first line of it named
 * a token rather than the thing that was actually wrong. The point of these
 * is that the first diagnostic is the true one, and that a file whose host
 * half Booth skips still hands over its kernels. */

#include "tharns.h"
#include "preproc.h"

static char obuf[1 << 16];
static char code[1 << 16];

static int cprun(const char *src, const char *args)
{
    char cmd[512];
    FILE *f = fopen("build/tcpp.cu", "w");

    if (!f) return -1;
    fputs(src, f);
    fclose(f);
    snprintf(cmd, sizeof cmd, "%s %s build/tcpp.cu", BC_BIN, args);
    return th_run(cmd, obuf, (int)sizeof obuf);
}

static int cpptx(const char *src)
{
    FILE *fp;
    size_t n;

    if (cprun(src, "--nvidia-ptx -o build/tcpp.ptx") != 0) return -1;
    fp = fopen("build/tcpp.ptx", "rb");
    if (!fp) return -1;
    n = fread(code, 1, sizeof code - 1, fp);
    fclose(fp);
    code[n] = '\0';
    return 0;
}

/* ---- What now parses ---- */

static void cpp01(void)
{
    CHEQ(cprun("template <float (*op)(float), typename T>\n"
               "__global__ void k(T *o){ o[0] = op(o[0]); }\n", "--parse"), 0);
    CHNE(strstr(obuf, "template_param"), NULL);
    CHNE(strstr(obuf, "ident op"), NULL);
    PASS();
}
TH_REG("cpp", 1, "a function pointer is a template parameter", cpp01)

static void cpp02(void)
{
    CHEQ(cpptx("__device__ float __forceinline__ hlf(float a){ return a*0.5f; }\n"
               "__global__ void k(float *o){ o[0] = hlf(o[0]); }\n"), 0);
    CHNE(strstr(code, ".entry"), NULL);
    PASS();
}
TH_REG("cpp", 2, "a qualifier may follow the return type", cpp02)

static void cpp03(void)
{
    CHEQ(cpptx("namespace outer { }\n"
               "namespace inr = outer;\n"
               "__global__ void k(int *o){ o[0] = 1; }\n"), 0);
    CHNE(strstr(code, ".entry"), NULL);
    PASS();
}
TH_REG("cpp", 3, "a namespace alias is a declaration", cpp03)

static void cpp04(void)
{
    CHEQ(cprun("namespace std { int min(int, int); }\n"
               "struct dim3 { unsigned x, y, z; };\n"
               "static void host(int n){ const dim3 b(std::min(n, 3), 1, 1);\n"
               "                         (void)b; }\n"
               "__global__ void k(int *o){ o[0] = 1; }\n", "--ir"), 0);
    CHNE(strstr(obuf, "__global__"), NULL);
    PASS();
}
TH_REG("cpp", 4, "a qualified call initialises, not declares", cpp04)

static void cpp05(void)
{
    CHEQ(cprun("namespace ns { struct T { int a; }; }\n"
               "void g(ns::T * p);\n", "--parse"), 0);
    CHNE(strstr(obuf, "func_decl"), NULL);
    PASS();
}
TH_REG("cpp", 5, "and a qualified type still declares", cpp05)

static void cpp06(void)
{
    CHEQ(cpptx("static void host(int n){ auto f = [&](int x){ return x + n; };\n"
               "                         (void)f; }\n"
               "__global__ void k(int *o){ o[0] = 2; }\n"), 0);
    CHNE(strstr(code, ".entry"), NULL);
    PASS();
}
TH_REG("cpp", 6, "a host lambda does not cost the kernel", cpp06)

static void cpp07(void)
{
    CHEQ(cpptx("struct S { int a; };\n"
               "static void host(void){ S * p = new S(); delete p; }\n"
               "__global__ void k(int *o){ o[0] = 3; }\n"), 0);
    CHNE(strstr(code, ".entry"), NULL);
    PASS();
}
TH_REG("cpp", 7, "nor does host new and delete", cpp07)

static void cpp08(void)
{
    CHEQ(cpptx("typedef long long i64;\n"
               "__global__ void k(long long *o, int x){ o[0] = i64(x); }\n"), 0);
    CHNE(strstr(code, ".entry"), NULL);
    PASS();
}
TH_REG("cpp", 8, "a named type called like a function is a cast", cpp08)

static void cpp11(void)
{
    CHEQ(cpptx("__device__ void g(int *x){ asm volatile(\"mov.u32 %0, 1;\"\n"
               "                                        : \"=r\"(x[0])); }\n"
               "__global__ void k(int *o){ g(o); }\n"), 0);
    CHNE(strstr(code, "mov.u32 %r"), NULL);
    PASS();
}
TH_REG("cpp", 11, "inline asm binds its operands", cpp11)

static void cpp12(void)
{
    CHEQ(cpptx("__global__ void k(float *o){ __align__(16) float t[4];\n"
               "                             t[0] = 1.0f; o[0] = t[0]; }\n"), 0);
    CHNE(strstr(code, ".entry"), NULL);
    PASS();
}
TH_REG("cpp", 12, "so does an over-aligned local", cpp12)

/* ---- What is refused, and by what name ---- */

static void cpp09(void)
{
    CHEQ(cprun("__global__ void k(int *o, float *p){\n"
               "  auto f = [](auto x) -> int { return (int)(x * 3); };\n"
               "  o[0] = f(7); o[1] = f(2.5f); (void)p; }\n", "--ir"), 0);
    CHNE(strstr(obuf, "%0, i32 %1) __device__"), NULL);
    CHNE(strstr(obuf, "%0, f32 %1) __device__"), NULL);
    CHNE(strstr(obuf, "fmul"), NULL);
    PASS();
}
TH_REG("cpp", 9, "a generic lambda instantiates per type", cpp09)

static void cpp10(void)
{
    CHNE(cprun("struct S { int a; };\n"
               "__global__ void k(int *o){ S * p = new S(); o[0] = p->a; }\n",
               "--ir"), 0);
    CHNE(strstr(obuf, "E170"), NULL);
    PASS();
}
TH_REG("cpp", 10, "so is new in device code", cpp10)

static void cpp13(void)
{
    CHEQ(cprun("__global__ void k(int *o){ int x = o[0];\n"
               "                           decltype(x) y = x + 1; o[1] = y; }\n",
               "--ir"), 0);
    CHEQ(strstr(obuf, "E020"), NULL);
    CHNE(strstr(obuf, "add i32"), NULL);
    PASS();
}
TH_REG("cpp", 13, "decltype names the type a local already has", cpp13)

static void cpp33(void)
{
    CHNE(cprun("__global__ void k(int *o){ decltype(nowt) y = 1; o[0] = y; }\n",
               "--ir"), 0);
    CHNE(strstr(obuf, "E360"), NULL);
    CHEQ(strstr(obuf, "E020"), NULL);
    PASS();
}
TH_REG("cpp", 33, "and refuses an operand it cannot type, once", cpp33)

static void cpp34(void)
{
    CHEQ(cpptx("template <int D> __global__ void kk(int *o){ o[0] = D * 7 + 5; }\n"
               "static void host(int *o){\n"
               "  auto lz = [&](auto d){ kk<d><<<1, 1>>>(o); };\n"
               "  lz(std::integral_constant<int, 3>{});\n"
               "  lz(std::integral_constant<int, 8>{}); }\n"), 0);
    CHNE(strstr(code, ".entry kk "), NULL);
    CHNE(strstr(code, ".entry kk$1 "), NULL);
    CHNE(strstr(code, "26;"), NULL);
    CHNE(strstr(code, "61;"), NULL);
    PASS();
}
TH_REG("cpp", 34, "two tags through one launcher are two kernels", cpp34)

static void cpp35(void)
{
    CHEQ(cpptx("__global__ void k(int *o){\n"
               "  auto f = [](auto t) -> int { return t * 11 + decltype(t)::value; };\n"
               "  o[0] = f(std::integral_constant<int, 3>{});\n"
               "  o[1] = f(std::integral_constant<int, 5>{});\n"
               "  o[2] = f(std::true_type{}); }\n"), 0);
    CHNE(strstr(code, "36;"), NULL);
    CHNE(strstr(code, "60;"), NULL);
    CHNE(strstr(code, "12;"), NULL);
    PASS();
}
TH_REG("cpp", 35, "a tag parameter carries its constant in", cpp35)

static void cpp37(void)
{
    CHEQ(cpptx("template <int N> __global__ void kk(int *o){ o[0] = N * 7 + 5; }\n"
               "template <int... Ns> static void run(int *o){\n"
               "  auto lz = [&](auto I) -> bool {\n"
               "    constexpr int n = decltype(I)::value;\n"
               "    kk<n><<<1, 1>>>(o); return true; };\n"
               "  if ((lz(std::integral_constant<int, Ns>{}) || ...)) { return; } }\n"
               "void go(int *o){ run<3, 8>(o); }\n"), 0);
    CHNE(strstr(code, ".entry kk "), NULL);
    CHNE(strstr(code, ".entry kk$1 "), NULL);
    CHNE(strstr(code, "26;"), NULL);
    CHNE(strstr(code, "61;"), NULL);
    PASS();
}
TH_REG("cpp", 37, "a fold over a pack drives the launcher", cpp37)

static void cpp36(void)
{
    CHEQ(cprun("__global__ void k(int *o){\n"
               "  auto f = [](auto a, int b) -> int { return (int)a * b; };\n"
               "  o[0] = f(3, 5); o[1] = f(2.0f, 7); }\n", "--ir"), 0);
    CHEQ(strstr(obuf, "E148"), NULL);
    CHNE(strstr(obuf, "%0, i32 %1, i32 %2)"), NULL);
    CHNE(strstr(obuf, "%0, f32 %1, i32 %2)"), NULL);
    PASS();
}
TH_REG("cpp", 36, "a typed parameter beside an auto one", cpp36)

/* A call wider than the argument cap used to expand as the first sixteen, with
   the rest left in the output as though the call had closed early. */
static void mkvar(char *dst, size_t cap, int n)
{
    size_t w = 0;
    int i;

    w = (size_t)snprintf(dst, cap,
                         "#define UNUSED(...) unused_impl(__VA_ARGS__)\n"
                         "static void unused_impl(...);\n"
                         "static void host(int a){ UNUSED(a");
    for (i = 1; i < n && w + 8 < cap; i++) {
        dst[w++] = ',';
        dst[w++] = 'a';
    }
    snprintf(dst + w, cap - w, "); }\n");
}

static void cpp14(void)
{
    static char big[1 << 14];

    mkvar(big, sizeof big, 40);
    CHEQ(cprun(big, "--parse"), 0);
    CHEQ(strstr(obuf, "E174"), NULL);

    mkvar(big, sizeof big, PP_MAX_ARGS + 1);
    CHNE(cprun(big, "--parse"), 0);
    CHNE(strstr(obuf, "E174"), NULL);
    PASS();
}
TH_REG("cpp", 14, "a macro call too long to expand says so", cpp14)

static void cpp20(void)
{
    static char big[1 << 14];
    size_t w;
    int i;

    w = (size_t)snprintf(big, sizeof big, "#define WIDE(p0");
    for (i = 1; i <= PP_MAX_PARAMS && w + 16 < sizeof big; i++)
        w += (size_t)snprintf(big + w, sizeof big - w, ",p%d", i);
    snprintf(big + w, sizeof big - w, ") p0\n");
    CHNE(cprun(big, "--parse"), 0);
    CHNE(strstr(obuf, "E300"), NULL);
    PASS();
}
TH_REG("cpp", 20, "a macro definition too wide to hold says so", cpp20)

static void cpp15(void)
{
    CHNE(cprun("template<typename T> __device__ T id(T v){ return v; }\n"
               "template __device__ float id<float>(float v);\n",
               "--parse"), 0);
    CHNE(strstr(obuf, "E175"), NULL);
    CHEQ(strstr(obuf, "E020"), NULL);
    PASS();
}
TH_REG("cpp", 15, "an explicit instantiation is refused by name", cpp15)

static void cpp16(void)
{
    CHEQ(cprun("template<typename T> __device__ T id(T v){ return v; }\n"
               "extern template __device__ float id<float>(float v);\n"
               "__global__ void k(float *o){ o[0] = 1.0f; }\n", "--ir"), 0);
    CHNE(strstr(obuf, "__global__"), NULL);
    PASS();
}
TH_REG("cpp", 16, "extern template is left to the file with it", cpp16)

/* ---- Lambdas ---- */

static void cpp17(void)
{
    CHEQ(cpptx("__global__ void k(int *o, int n){ int b = n * 7;\n"
               "  auto f = [=](int x) -> int { return b + x; };\n"
               "  o[0] = f(3) + f(4); }\n"), 0);
    CHNE(strstr(code, ", 7;"), NULL);
    CHNE(strstr(code, ", 3;"), NULL);
    CHNE(strstr(code, ", 4;"), NULL);
    PASS();
}
TH_REG("cpp", 17, "a captured value reaches the emitted code", cpp17)

static void cpp18(void)
{
    CHEQ(cpptx("__global__ void k(int *o, int n){ int a = n;\n"
               "  auto f = [&](int x) { a = a + x; };\n"
               "  f(5); o[0] = a; }\n"), 0);
    CHNE(strstr(code, "st.local.u32"), NULL);
    CHNE(strstr(code, "ld.local.u32"), NULL);
    PASS();
}
TH_REG("cpp", 18, "a by-reference capture, and the caller sees it", cpp18)

static void cpp19(void)
{
    CHEQ(cpptx("__global__ void k(int *o){\n"
               "  o[0] = [](int a, int b) -> int { return a - b; }(9, 4); }\n"), 0);
    CHNE(strstr(code, ".entry"), NULL);
    PASS();
}
TH_REG("cpp", 19, "a lambda called on the spot lowers", cpp19)

static void cpp30(void)
{
    CHNE(cprun("template<typename F> __device__ int ap(F f){ return f(1); }\n"
               "__global__ void k(int *o, int n){\n"
               "  auto f = [=](int x) -> int { return x + n; };\n"
               "  o[0] = ap(f); }\n", "--ir"), 0);
    CHNE(strstr(obuf, "E170"), NULL);
    PASS();
}
TH_REG("cpp", 30, "a closure as a value is refused, not guessed", cpp30)

static void cpp21(void)
{
    CHNE(cprun("__global__ void k(int *o, int n){ int t[2]; t[0] = n;\n"
               "  auto f = [=](int x) -> int { return t[x]; };\n"
               "  o[0] = f(1); }\n", "--ir"), 0);
    CHNE(strstr(obuf, "E261"), NULL);
    PASS();
}
TH_REG("cpp", 21, "an aggregate by value is refused by name", cpp21)

static void cpp22(void)
{
    CHEQ(cpptx("template<typename T> __global__ void a(T *o){ o[0] = 1; }\n"
               "static void host(float *f, int *i){\n"
               "  auto go = [&]() { a<float><<<1, 1>>>(f); };\n"
               "  go(); [&]() { a<int><<<1, 1>>>(i); }(); }\n"), 0);
    CHNE(strstr(code, ".entry a "), NULL);
    CHNE(strstr(code, ".entry a$1 "), NULL);
    PASS();
}
TH_REG("cpp", 22, "a launcher lambda still yields its kernels", cpp22)
static void cpp31(void)
{
    CHEQ(cpptx("struct P {\n"
               "    unsigned d;\n"
               "    __device__ unsigned it() const { return d & 0x003FFFFFu; }\n"
               "    __device__ unsigned iex() const { return d >> 22; }\n"
               "};\n"
               "__global__ void k(unsigned *o){\n"
               "    P p; p.d = o[0];\n"
               "    o[1] = p.it(); o[2] = p.iex(); }\n"), 0);
    CHNE(strstr(code, "and.b32"), NULL);
    CHNE(strstr(code, "4194303"), NULL);
    CHNE(strstr(code, "shr.u32"), NULL);
    CHNE(strstr(code, ".local"), NULL);
    PASS();
}
TH_REG("cpp", 31, "a member call reaches the method it named", cpp31)

static void cpp32(void)
{
    CHEQ(cpptx("template <int N> struct T {\n"
               "    static constexpr int value = N * 3;\n"
               "    __device__ static int bump(int a) { return a + N; }\n"
               "};\n"
               "struct U { static constexpr int value = 7; };\n"
               "__global__ void k(int *o){\n"
               "    o[0] = T<5>::value; o[1] = U::value;\n"
               "    o[2] = T<5>::bump(o[2]); o[3] = T<9>::bump(o[3]); }\n"), 0);
    CHNE(strstr(code, ", 15;"), NULL);
    CHNE(strstr(code, "add.u32"), NULL);
    CHNE(strstr(code, ", 5;"), NULL);
    CHNE(strstr(code, ", 9;"), NULL);
    PASS();
}
TH_REG("cpp", 32, "a class template scope answers for itself", cpp32)

static void cpp23(void)
{
    CHEQ(cpptx("template <int n> struct P2 {\n"
               "    static constexpr int value = 2*P2<(n + 1)/2>::value; };\n"
               "template <> struct P2<1> { static constexpr int value = 1; };\n"
               "template <> struct P2<0> { static constexpr int value = 1; };\n"
               "__global__ void k(int *o){ o[0] = P2<5>::value; }\n"), 0);
    CHNE(strstr(code, ", 8;"), NULL);
    PASS();
}
TH_REG("cpp", 23, "an explicit specialisation ends the recursion", cpp23)

static void cpp24(void)
{
    CHEQ(cpptx("__global__ void k(__half2 *s, __half *lo, __half *hi){\n"
               "    __half2 v = s[0];\n"
               "    lo[0] = __low2half(v); hi[0] = __high2half(v); }\n"), 0);
    CHNE(strstr(code, "ld.global.b16"), NULL);
    CHNE(strstr(code, "st.global.b16"), NULL);
    CHNE(strstr(code, ", 2;"), NULL);
    CHEQ(strstr(code, "ld.global.u32 %h"), NULL);
    PASS();
}
TH_REG("cpp", 24, "half lanes load sixteen bits from two offsets", cpp24)

static void cpp25(void)
{
    CHEQ(cprun("static int f(int a){ return a; }\n"
               "static int g(int a){ if (int s = f(a); s) return 1; return 0; }\n"
               "static int h(int a){ switch (int s = f(a); s) { default: return s; } }\n",
               "--parse"), 0);
    CHEQ(strstr(obuf, "E020"), NULL);
    CHEQ(strstr(obuf, "E022"), NULL);
    PASS();
}
TH_REG("cpp", 25, "if and switch take an initialiser", cpp25)

static void cpp26(void)
{
    CHEQ(cprun("__global__ void k(char *o){ const char *s = \"a%\" PRId64 \"b\"; o[0] = s[0]; }\n",
               "--parse"), 0);
    CHEQ(strstr(obuf, "E020"), NULL);
    PASS();
}
TH_REG("cpp", 26, "the inttypes format macros are defined", cpp26)

static void cpp27(void)
{
    CHEQ(cprun("struct W { union { float d4[4]; int i4[4]; };\n"
               "  int a0; int a1; int a2; int a3; int a4; int a5; int a6; int a7;\n"
               "  int b0; int b1; int b2; int b3; int b4; int b5; int b6; int b7;\n"
               "  int c0; };\n"
               "__global__ void k(W *w){ w->d4[0] = 1.0f; w->c0 = w->i4[1]; }\n",
               "--ir"), 0);
    CHEQ(strstr(obuf, "E071"), NULL);
    PASS();
}
TH_REG("cpp", 27, "a wide record keeps its anonymous union", cpp27)

static void cpp28(void)
{
    CHEQ(cprun("template <typename T> __device__ T twice(T v){ return v + v; }\n"
               "struct R { int d;\n"
               "    __device__ int go() const { return twice(d) + 1; } };\n"
               "__global__ void k(int *o){ R r; r.d = o[0]; o[1] = r.go(); }\n",
               "--ir"), 0);
    CHNE(strstr(obuf, "func @twice(i32 %0)"), NULL);
    CHNE(strstr(obuf, "func @R$go(ptr<generic"), NULL);
    PASS();
}
TH_REG("cpp", 28, "a template in a method keeps its shape", cpp28)

static void cpp38(void)
{
    CHEQ(cpptx("struct pa { float u; float v; };\n"
               "struct pb { float u; float v; float w; };\n"
               "__device__ float pick(const pa x){ return x.u + 111.0f; }\n"
               "__device__ float pick(const pb x){ return x.u + 222.0f; }\n"
               "__global__ void k(float *o){ pa a; a.u = 1.0f; a.v = 2.0f;\n"
               "  o[0] = pick(a); }\n"), 0);
    CHNE(strstr(code, "0f42DE0000")
         ? code : strstr(code, "0f42E00000"), NULL);
    CHEQ(strstr(code, "0f435E0000"), NULL);
    CHEQ(strstr(code, "0f435F0000"), NULL);
    PASS();
}
TH_REG("cpp", 38, "an overload is chosen by its struct argument", cpp38)

static void cpp39(void)
{
    CHEQ(cpptx("struct grid { int v;\n"
               "  static __device__ int gidx(const int l){ return l*3 + 41; }\n"
               "  __device__ int inst(const int l) const { return v + l; } };\n"
               "__global__ void k(int *o){ grid g; g.v = 5;\n"
               "  o[0] = g.gidx(2); o[1] = g.inst(1); }\n"), 0);
    CHNE(strstr(code, ", 47;"), NULL);
    PASS();
}
TH_REG("cpp", 39, "a static method takes no receiver", cpp39)

static void cpp40(void)
{
    CHEQ(cpptx("struct boxx { int a; int b; };\n"
               "__global__ void k(int *o){ typedef boxx inner;\n"
               "  inner v; v.a = 7; v.b = 9; o[0] = v.a + v.b; }\n"), 0);
    CHNE(strstr(code, ", 16;"), NULL);
    PASS();
}
TH_REG("cpp", 40, "a typedef inside a body names a type", cpp40)

static void cpp41(void)
{
    CHEQ(cpptx("enum lay { LA = 0, LB = 1 };\n"
               "template <int I, typename T, lay dl = LA> struct tl { T x[I]; };\n"
               "template <int I, typename T>\n"
               "__device__ void fill(tl<I, T, LA> & t, T v){ t.x[I-1] = v + 13; }\n"
               "__global__ void k(int *o){ tl<4, int> a; fill(a, 4); o[0] = a.x[3]; }\n"), 0);
    CHNE(strstr(code, ", 17;"), NULL);
    PASS();
}
TH_REG("cpp", 41, "a class template parameter is deduced", cpp41)

/* arg_type has to unwrap the forwarding call before it resolves a subscript
 * or a member, or a forwarded a[i] types as the wrapper. Deduction is where it
 * shows: sema cannot name a member of a class-template instantiation, so with
 * the unwrap last, T comes back as the wrapper and the reference will not bind.
 * Each pick body carries its own constant, so the .entry says which was run. */
static void cpp42(void)
{
    CHEQ(cpptx("namespace std {\n"
               "  template<class T> T&& forward(T& a){ return (T&&)a; } }\n"
               "struct h2 { float a; float b; };\n"
               "template<int I, typename T> struct tl { T x[I]; };\n"
               "__device__ float pick(const h2 v){ return v.a + 111.0f; }\n"
               "__device__ float pick(const float v){ return v + 222.0f; }\n"
               "template<typename T> __device__ float ap(T&& t)"
               " { return pick(std::forward<T>(t)); }\n"
               "__device__ float use(tl<4, h2> & t)"
               " { return ap(std::forward<h2>(t.x[0])); }\n"
               "__global__ void k(float *o){ tl<4, h2> t; t.x[0].a = 1.0f;\n"
               "  o[0] = use(t); }\n"), 0);
    CHNE(strstr(code, "0f42DE0000")
         ? code : strstr(code, "0f42E00000"), NULL);
    CHEQ(strstr(code, "0f435E0000"), NULL);
    CHEQ(strstr(code, "0f435F0000"), NULL);
    PASS();
}
TH_REG("cpp", 42, "a forwarded member types as what it wraps", cpp42)

/* Scan-time parameter typing must not reach a parameter pack: the ordinary
 * parameters of the enclosing function are typed, the pack is left to the
 * expansion, and the kernel comes out with one parameter per element. */
static void cpp43(void)
{
    CHEQ(cpptx("template <typename... E>\n"
               "static __global__ void ke(int *o, E... e)"
               " { o[0] = (0 + ... + (int)*e); }\n"
               "template <typename... E>\n"
               "static void d(int *o, E... e){ ke<<<1,1>>>(o, e...); }\n"
               "void h(int *o, int *a, float *b){ d(o, a, b); }\n"), 0);
    CHNE(strstr(code, ".param .u64 param1,"), NULL);
    CHNE(strstr(code, ".param .u64 param2"), NULL);
    PASS();
}
TH_REG("cpp", 43, "a pack keeps its elements past param typing", cpp43)
