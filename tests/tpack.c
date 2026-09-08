/* tpack.c -- template parameter packs
 * The grammar, the standard's placement rules, and what a pack lowers to. */

#include "tharns.h"

static char obuf[TH_BUFSZ];

static int wsrc(const char *path, const char *text)
{
    FILE *f = fopen(path, "wb");
    if (!f) return -1;
    fputs(text, f);
    fclose(f);
    return 0;
}

static int run_ir(const char *text)
{
    if (wsrc("pack_tmp.cu", text) != 0) return -1;
    int rc = th_run(BC_BIN " --ir pack_tmp.cu", obuf, TH_BUFSZ);
    remove("pack_tmp.cu");
    return rc;
}

static int run_parse(const char *text)
{
    if (wsrc("pack_tmp.cu", text) != 0) return -1;
    int rc = th_run(BC_BIN " --parse pack_tmp.cu", obuf, TH_BUFSZ);
    remove("pack_tmp.cu");
    return rc;
}

static int run_ptx(const char *text)
{
    if (wsrc("pack_tmp.cu", text) != 0) return -1;
    int rc = th_run(BC_BIN " --nvidia-ptx pack_tmp.cu -o pack_tmp.ptx",
                    obuf, TH_BUFSZ);
    remove("pack_tmp.cu");
    if (rc != 0) return rc;
    FILE *pf = fopen("pack_tmp.ptx", "rb");
    if (!pf) return -1;
    size_t got = fread(obuf, 1, TH_BUFSZ - 1, pf);
    obuf[got] = 0;
    fclose(pf);
    remove("pack_tmp.ptx");
    return 0;
}

/* ---- packs: the fixture compiles ---- */

static void pck01(void)
{
    int rc = th_run(BC_BIN " --ir tests/packs.cu", obuf, TH_BUFSZ);
    CHEQ(rc, 0);
    CHECK(strstr(obuf, "error") == NULL);
    PASS();
}
TH_REG("pck", 1, "the pack fixture reaches BIR", pck01)

/* ---- packs: a pack parameter becomes one BIR parameter per member ---- */

static void pck02(void)
{
    int rc = th_run(BC_BIN " --ir tests/packs.cu", obuf, TH_BUFSZ);
    CHEQ(rc, 0);
    CHECK(strstr(obuf,
        "func @pk_sum(ptr<global, f32> %0, f32 %1, f32 %2, f32 %3, f32 %4)")
        != NULL);
    PASS();
}
TH_REG("pck", 2, "a pack widens the parameter list", pck02)

/* ---- packs: two lengths are two functions ---- */

static void pck03(void)
{
    int rc = th_run(BC_BIN " --ir tests/packs.cu", obuf, TH_BUFSZ);
    CHEQ(rc, 0);
    CHECK(strstr(obuf, "func @pk_sum$1(") != NULL);
    PASS();
}
TH_REG("pck", 3, "two specialisations do not share a symbol", pck03)

/* ---- packs: sizeof... is a constant by lowering ---- */

static void pck04(void)
{
    int rc = run_ir(
        "template<typename... A>\n"
        "__global__ void k(int *o, A... a)\n"
        "{ o[0] = (int)sizeof...(a) + (int)sizeof...(A); }\n"
        "int main(void){ int *o; cudaMalloc(&o,16);"
        " k<<<1,1>>>(o, 1, 2, 3); return 0; }\n");
    CHEQ(rc, 0);
    CHECK(strstr(obuf, "store i32 6,") != NULL);
    PASS();
}
TH_REG("pck", 4, "sizeof... folds to the pack length", pck04)

/* ---- packs: a fold expands to the operator chain ---- */

static void pck05(void)
{
    int rc = run_ir(
        "template<typename... A>\n"
        "__global__ void k(float *o, A... a) { o[0] = (0.0f + ... + (float)a); }\n"
        "int main(void){ float *o; cudaMalloc(&o,16);"
        " k<<<1,1>>>(o, 1.0f, 2.0f, 3.0f); return 0; }\n");
    CHEQ(rc, 0);
    /* seed plus one add per member */
    int n = 0;
    for (const char *p = obuf; (p = strstr(p, "fadd")) != NULL; p++) n++;
    CHEQ(n, 3);
    PASS();
}
TH_REG("pck", 5, "a binary left fold becomes one op per member", pck05)

/* ---- packs: an expansion widens a call ---- */

static void pck06(void)
{
    int rc = th_run(BC_BIN " --ir --no-sroa tests/packs.cu", obuf, TH_BUFSZ);
    CHEQ(rc, 0);
    CHECK(strstr(obuf,
        "func @pk_call(ptr<global, f32> %0, f32 %1, f32 %2, f32 %3)") != NULL);
    PASS();
}
TH_REG("pck", 6, "an expansion in a call supplies every member", pck06)

/* ---- packs: || keeps its short circuit ---- */

static void pck07(void)
{
    int rc = th_run(BC_BIN " --ir tests/packs.cu", obuf, TH_BUFSZ);
    CHEQ(rc, 0);
    const char *f = strstr(obuf, "func @pk_any");
    CHECK(f != NULL);
    CHECK(strstr(f, "br_cond") != NULL);
    PASS();
}
TH_REG("pck", 7, "a logical fold branches rather than chains", pck07)

/* ---- packs: an unexpanded pack is ill-formed ---- */

static void pck08(void)
{
    int rc = run_parse(
        "template<typename... A>\n"
        "__device__ int k(A... a) { return f(a); }\n");
    (void)rc;
    CHECK(strstr(obuf, "error[E083]") != NULL);
    PASS();
}
TH_REG("pck", 8, "a pack used without an expansion is rejected", pck08)

/* ---- packs: [temp.param]/14 placement ---- */

static void pck09(void)
{
    int rc = run_parse(
        "template<typename... A, typename B> struct S { B v; };\n");
    (void)rc;
    CHECK(strstr(obuf, "error[E028]") != NULL);
    PASS();
}
TH_REG("pck", 9, "a class template pack must come last", pck09)

/* ---- packs: no default argument on a pack ---- */

static void pck10(void)
{
    int rc = run_parse("template<typename... A = int> struct S { int v; };\n");
    (void)rc;
    CHECK(strstr(obuf, "error[E029]") != NULL);
    PASS();
}
TH_REG("pck", 10, "a pack takes no default argument", pck10)

/* ---- packs: pack indexing refuses by name ---- */

static void pck11(void)
{
    int rc = run_parse(
        "template<typename... A>\n"
        "__device__ int k(A... a) { return (int)a...[0]; }\n");
    (void)rc;
    CHECK(strstr(obuf, "error[E030]") != NULL);
    CHECK(strstr(obuf, "pack indexing") != NULL);
    PASS();
}
TH_REG("pck", 11, "pack indexing refuses by name", pck11)

/* ---- packs: [dcl.fct]/27 keeps the C ellipsis ---- */

static void pck12(void)
{
    int rc = run_ir(
        "__device__ int v(int a, ...) { return a; }\n"
        "__global__ void k(int *o) { o[0] = v(1); }\n");
    CHEQ(rc, 0);
    CHECK(strstr(obuf, "error") == NULL);
    PASS();
}
TH_REG("pck", 12, "a C ellipsis is still a C ellipsis", pck12)

/* ---- packs: a forwarding-reference pack parses ---- */

static void pck13(void)
{
    int rc = run_parse(
        "template<typename... A>\n"
        "__host__ __device__ constexpr inline void u(A&&...) noexcept {}\n");
    (void)rc;
    CHECK(strstr(obuf, "error[") == NULL);
    PASS();
}
TH_REG("pck", 13, "an unnamed forwarding pack parses", pck13)

/* ---- packs: the pack need not be last in a function template ---- */

static void pck14(void)
{
    int rc = run_parse(
        "template<int... N, typename T>\n"
        "__device__ T k(T x) { return x; }\n");
    (void)rc;
    CHECK(strstr(obuf, "error[") == NULL);
    PASS();
}
TH_REG("pck", 14, "a function template may declare after a pack", pck14)

/* ---- packs: an overload set is chosen by arity ---- */

static void pck15(void)
{
    /* Both overloads inline, so only the emitted constant tells them apart. */
    CHECK(wsrc("pack_tmp.cu",
        "__device__ int g(int a, int b) { return a * 1000 + b; }\n"
        "__device__ int g(int a) { return a + 7; }\n"
        "__global__ void k(int *o) { o[0] = g(5); }\n"
        "int main(void){ int *o; cudaMalloc(&o,16);"
        " k<<<1,1>>>(o); return 0; }\n") == 0);
    int rc = th_run(BC_BIN " --nvidia-ptx pack_tmp.cu -o pack_tmp.ptx",
                    obuf, TH_BUFSZ);
    remove("pack_tmp.cu");
    CHEQ(rc, 0);
    FILE *pf = fopen("pack_tmp.ptx", "rb");
    CHECK(pf != NULL);
    size_t got = fread(obuf, 1, TH_BUFSZ - 1, pf);
    obuf[got] = 0;
    fclose(pf);
    remove("pack_tmp.ptx");
    CHECK(strstr(obuf, "mov.u32 %r1, 12;") != NULL);
    PASS();
}
TH_REG("pck", 15, "a call binds to the overload with its arity", pck15)

/* ---- packs: a struct settles a pack, then the cast refuses ---- */

static void pck16(void)
{
    int rc = run_ir(
        "struct P { float a; };\n"
        "template<typename... A>\n"
        "__global__ void k(float *o, A... a) { o[0] = (0.0f + ... + (float)a); }\n"
        "int main(void){ float *o; struct P p; p.a = 1.0f; cudaMalloc(&o,16);"
        " k<<<1,1>>>(o, p); return 0; }\n");
    (void)rc;
    CHECK(strstr(obuf, "E440") != NULL);
    PASS();
}
TH_REG("pck", 16, "a struct pack member will not cast to float", pck16)

/* ---- packs: a variable settles a pack as well as a literal does ---- */

static void pck17(void)
{
    int rc = run_ir(
        "template<typename... A>\n"
        "__global__ void k(float *o, A... a) { o[0] = (0.0f + ... + (float)a); }\n"
        "int main(void){ float *o; float x = 1.0f; cudaMalloc(&o,16);"
        " k<<<1,1>>>(o, x); return 0; }\n");
    CHEQ(rc, 0);
    CHECK(strstr(obuf, "func @k(ptr<global, f32> %0, f32 %1)") != NULL);
    PASS();
}
TH_REG("pck", 17, "a pack deduced from a variable lowers", pck17)

/* ---- packs: an unnamed forwarding pack still widens the call ---- */

static void pck18(void)
{
    int rc = run_ir(
        "template<typename... A>\n"
        "__host__ __device__ constexpr inline void u(A&&...) noexcept {}\n"
        "__global__ void k(int *o, int a, int b) { u(a, b); o[0] = 7; }\n"
        "int main(void){ int *o; cudaMalloc(&o,16);"
        " k<<<1,1>>>(o, 1, 2); return 0; }\n");
    CHEQ(rc, 0);
    CHECK(strstr(obuf, "func @u(i32 %0, i32 %1)") != NULL);
    CHECK(strstr(obuf, "call void @u(%1, %2)") != NULL);
    PASS();
}
TH_REG("pck", 18, "an unnamed forwarding pack widens the call", pck18)

/* ---- packs: the members arrive in the order they were written ---- */

static void pck19(void)
{
    /* r = r*10 + a over 1, 2, 3 is 123 only if the order survives. */
    int rc = run_ptx(
        "template<typename... A>\n"
        "__device__ int mix(A... a)"
        " { int r = 0; (..., (r = r * 10 + (int)a)); return r; }\n"
        "__global__ void k(int *o) { o[0] = mix(1, 2, 3); }\n"
        "int main(void){ int *o; cudaMalloc(&o,16);"
        " k<<<1,1>>>(o); return 0; }\n");
    CHEQ(rc, 0);
    CHECK(strstr(obuf, "mov.u32 %r1, 123;") != NULL);
    PASS();
}
TH_REG("pck", 19, "the emitted code keeps the pack in order", pck19)

/* ---- packs: two specialisations of one arity do not collide ---- */

static void pck20(void)
{
    int rc = run_ir(
        "template<typename... A>\n"
        "__device__ int wide(A... a)"
        " { int n = 0; (..., (n = n + (int)a)); return n; }\n"
        "__global__ void k(int *o)"
        " { o[0] = wide(1, 2); o[1] = wide(1.0, 2.0); }\n"
        "int main(void){ int *o; cudaMalloc(&o,32);"
        " k<<<1,1>>>(o); return 0; }\n");
    CHEQ(rc, 0);
    CHECK(strstr(obuf, "func @wide(i32 %0, i32 %1)") != NULL);
    CHECK(strstr(obuf, "func @wide$1(f64 %0, f64 %1)") != NULL);
    CHECK(strstr(obuf, "@wide(") != NULL);
    CHECK(strstr(obuf, "@wide$1(") != NULL);
    PASS();
}
TH_REG("pck", 20, "same arity, different types, different symbols", pck20)

/* ---- templates: an explicit argument reaches the specialisation ---- */

static void pck21(void)
{
    int rc = run_ir(
        "template<int N> __global__ void ker(float *d) { d[0] = (float)N; }\n"
        "int main(void){ float *d; cudaMalloc(&d,16);"
        " ker<9><<<1,1>>>(d); return 0; }\n");
    CHEQ(rc, 0);
    CHECK(strstr(obuf, "store f32 9,") != NULL);
    PASS();
}
TH_REG("pck", 21, "an explicit template argument is not dropped", pck21)

/* ---- packs: a launch made through a forwarding wrapper ---- */

static void pck22(void)
{
    int rc = run_ir(
        "struct lp { int a; int b; };\n"
        "template<typename... A>\n"
        "__global__ void ker(int *o, A... a)"
        " { int r = 0; (..., (r = r * 10 + (int)a)); o[0] = r; }\n"
        "template<typename K, typename... A>\n"
        "static inline void kl(K kern, const lp &c, A&&... args)"
        " { kern<<<c.a, c.b, 0, 0>>>(args...); }\n"
        "void go(int *o)"
        " { lp c; c.a = 1; c.b = 1; kl(ker, c, o, 1.0f, 2, 3.0); }\n");
    CHEQ(rc, 0);
    CHECK(strstr(obuf,
        "func @ker(ptr<global, i32> %0, f32 %1, i32 %2, f64 %3)") != NULL);
    PASS();
}
TH_REG("pck", 22, "a wrapper forwards every member to the kernel", pck22)

/* ---- packs: an explicit argument survives the wrapper too ---- */

static void pck23(void)
{
    int rc = run_ir(
        "struct lp { int a; int b; };\n"
        "template<int P, int Q> __global__ void ker(int *o)"
        " { o[0] = P * 10 + Q; }\n"
        "template<typename K, typename... A>\n"
        "static inline void kl(K kern, const lp &c, A&&... args)"
        " { kern<<<c.a, c.b, 0, 0>>>(args...); }\n"
        "void go(int *o) { lp c; c.a = 1; c.b = 1; kl(ker<4, 5>, c, o); }\n");
    CHEQ(rc, 0);
    CHECK(strstr(obuf, "store i32 45,") != NULL);

    rc = run_ptx(
        "struct lp { int a; int b; };\n"
        "template<int P, int Q> __global__ void ker(int *o)"
        " { o[0] = P * 10 + Q; }\n"
        "template<typename K, typename... A>\n"
        "static inline void kl(K kern, const lp &c, A&&... args)"
        " { kern<<<c.a, c.b, 0, 0>>>(args...); }\n"
        "void go(int *o) { lp c; c.a = 1; c.b = 1; kl(ker<4, 5>, c, o); }\n");
    CHEQ(rc, 0);
    CHECK(strstr(obuf, "mov.u32 %r1, 45;") != NULL);
    PASS();
}
TH_REG("pck", 23, "an explicit argument survives a wrapper", pck23)

/* ---- launches: shared memory and stream are not kernel arguments ---- */

static void pck24(void)
{
    int rc = run_ir(
        "template<typename... A>\n"
        "__global__ void ker(int *o, A... a)"
        " { int r = 0; (..., (r = r * 10 + (int)a)); o[0] = r; }\n"
        "int main(void){ int *o; cudaMalloc(&o,16);"
        " ker<<<1,1,0,0>>>(o, 1.0f, 2); return 0; }\n");
    CHEQ(rc, 0);
    CHECK(strstr(obuf,
        "func @ker(ptr<global, i32> %0, f32 %1, i32 %2)") != NULL);
    PASS();
}
TH_REG("pck", 24, "a launch configuration is not an argument list", pck24)

/* ---- packs: a wrapper Booth cannot expand refuses by name ---- */

static void pck25(void)
{
    int rc = run_ir(
        "template<typename K> static inline void kl(K kern, int, ...)"
        " { kern<<<1,1>>>(); }\n"
        "template<int N> __global__ void ker(void) { }\n"
        "void go(void) { kl(ker<3>, 5); }\n");
    (void)rc;
    CHECK(strstr(obuf, "E157") != NULL);
    CHECK(strstr(obuf, "forwarding wrapper") != NULL);
    PASS();
}
TH_REG("pck", 25, "a wrapper that cannot expand refuses by name", pck25)

static void pck26(void)
{
    CHEQ(run_ir(
        "template<typename K> static inline void kl(K kern, int)"
        " { kern<<<1,1>>>(); }\n"
        "template<int N> __global__ void ker(void) { }\n"
        "void go(void) { kl(ker<3>, 5); }\n"), 0);
    CHNE(strstr(obuf, "func @ker() __global__"), NULL);
    PASS();
}
TH_REG("pck", 26, "an unnamed parameter does not stop a wrapper", pck26)

/* ---- packs: a variable template folds its pack ---- */

static const char *const VTSAME =
    "namespace std {\n"
    "template<class A, class B> inline constexpr bool is_same_v = false;\n"
    "}\n"
    "template<typename T, typename... Ts>\n"
    "inline constexpr bool is_any = (std::is_same_v<T, Ts> || ...);\n";

static void pck27(void)
{
    char src[1024];

    snprintf(src, sizeof src, "%s%s", VTSAME,
             "template<typename T> __device__ float pick(float x){\n"
             "    if constexpr (is_any<T, float, int>) return x * 3.0f;\n"
             "    else return x + 7.0f;\n"
             "}\n"
             "__global__ void hit(float *o){ o[0] = pick<float>(o[0]); }\n"
             "__global__ void miss(float *o){ o[0] = pick<double>(o[0]); }\n");
    CHEQ(run_ptx(src), 0);
    CHNE(strstr(obuf, "mul.rn.f32"), NULL);
    CHNE(strstr(obuf, "add.rn.f32"), NULL);
    PASS();
}
TH_REG("pck", 27, "a folded variable template splits the PTX", pck27)

static void pck28(void)
{
    char src[1024];

    snprintf(src, sizeof src, "%s%s", VTSAME,
             "__global__ void k(int *o){\n"
             "    if constexpr (is_any<int, float, char, int>) o[0] = 1;\n"
             "    else o[0] = 2;\n"
             "    if constexpr (is_any<short, float, char, int>) o[1] = 1;\n"
             "    else o[1] = 2;\n"
             "}\n");
    CHEQ(run_ir(src), 0);
    CHNE(strstr(obuf, "store i32 1"), NULL);
    CHNE(strstr(obuf, "store i32 2"), NULL);
    PASS();
}
TH_REG("pck", 28, "a fold reaches every member of its pack", pck28)

static void pck29(void)
{
    CHEQ(run_ir(
        "template<int... N> inline constexpr int suml = (0 + ... + N);\n"
        "template<int... N> inline constexpr int sumr = (N + ... + 0);\n"
        "template<int... N> inline constexpr int subl = (... - N);\n"
        "template<int... N> inline constexpr int subr = (N - ...);\n"
        "template<int... N> inline constexpr int last = (... , N);\n"
        "__global__ void k(int *o){\n"
        "    if constexpr (suml<1,2,3> == 6)   o[0] = 1;\n"
        "    if constexpr (sumr<1,2,3> == 6)   o[1] = 1;\n"
        "    if constexpr (subl<10,3,2> == 5)  o[2] = 1;\n"
        "    if constexpr (subr<10,3,2> == 9)  o[3] = 1;\n"
        "    if constexpr (last<1,2,7> == 7)   o[4] = 1;\n"
        "}\n"), 0);
    CHNE(strstr(obuf, "%0, 0"), NULL);
    CHNE(strstr(obuf, "%0, 1"), NULL);
    CHNE(strstr(obuf, "%0, 2"), NULL);
    CHNE(strstr(obuf, "%0, 3"), NULL);
    CHNE(strstr(obuf, "%0, 4"), NULL);
    PASS();
}
TH_REG("pck", 29, "left and right folds keep their direction", pck29)

static void pck30(void)
{
    CHEQ(run_ir(
        "template<int... N> inline constexpr int anyv = (... || N);\n"
        "template<int... N> inline constexpr int allv = (... && N);\n"
        "__global__ void k(int *o){\n"
        "    if constexpr (anyv<> == 0)  o[0] = 1;\n"
        "    if constexpr (allv<> == 1)  o[1] = 1;\n"
        "    if constexpr (anyv<0,0,2> == 1) o[2] = 1;\n"
        "    if constexpr (allv<1,1,0> == 0) o[3] = 1;\n"
        "}\n"), 0);
    CHNE(strstr(obuf, "%0, 0"), NULL);
    CHNE(strstr(obuf, "%0, 1"), NULL);
    CHNE(strstr(obuf, "%0, 2"), NULL);
    CHNE(strstr(obuf, "%0, 3"), NULL);
    PASS();
}
TH_REG("pck", 30, "an empty pack folds to the operator identity", pck30)

static void pck31(void)
{
    CHEQ(run_ir(
        "template<int... N> __device__ int f(void) { return (N * ...); }\n"
        "__global__ void k(int *o){ o[0] = f<2,3>(); o[1] = f<5,7>(); }\n"), 0);
    CHNE(strstr(obuf, "ret i32 6"), NULL);
    CHNE(strstr(obuf, "ret i32 35"), NULL);
    PASS();
}
TH_REG("pck", 31, "two value packs are two instantiations", pck31)

static void pck32(void)
{
    int rc = run_ir(
        "template<class A, class B> inline constexpr bool sm = false;\n"
        "template<class A> inline constexpr bool sm<A,A> = true;\n"
        "__global__ void k(int *o){ if constexpr (sm<int,int>) o[0] = 1;"
        " else o[0] = 2; }\n");
    (void)rc;
    CHNE(strstr(obuf, "E158"), NULL);
    CHEQ(strstr(obuf, "store i32 1"), NULL);
    PASS();
}
TH_REG("pck", 32, "a specialised variable template is not guessed", pck32)

/* ---- packs: forwarded through a wrapper into a launch ---- */

static void pck33(void)
{
    CHEQ(run_ir(
        "struct lp { dim3 bn; dim3 bd; size_t sh; cudaStream_t st;\n"
        "  lp(const dim3 & a, const dim3 & b, const size_t c,\n"
        "     const cudaStream_t d) : bn(a), bd(b), sh(c), st(d) {} };\n"
        "template <typename T, size_t I> using pick = T;\n"
        "template <typename K, typename... A>\n"
        "static __inline__ void go(K k, const lp & q, A&&... a) {\n"
        "    k<<<q.bn, q.bd, q.sh, q.st>>>(std::forward<A>(a)...); }\n"
        "template <typename T, typename... E>\n"
        "static __global__ void kslot(float * o, T a, int b, double c, E... e) {\n"
        "    o[0] = (float)a; o[1] = (float)b; o[2] = (float)c;\n"
        "    o[3] = (float)(e[0] + ...); }\n"
        "template <typename T, size_t... I>\n"
        "static void drive(float * o, T ** s, cudaStream_t t,\n"
        "                  std::index_sequence<I...>) {\n"
        "    const lp q = lp(dim3(1), dim3(1), 0, t);\n"
        "    go(kslot<T, pick<T *, I>...>, q, o, (T)1, 2, 3.0, (T *)s[I]...); }\n"
        "void host(float * o, float ** s, cudaStream_t t) {\n"
        "    drive<float>(o, s, t, std::make_index_sequence<2>{}); }\n"), 0);
    CHNE(strstr(obuf,
        "func @kslot(ptr<global, f32> %0, f32 %1, i32 %2, f64 %3,"
        " ptr<global, f32> %4, ptr<global, f32> %5)"), NULL);
    PASS();
}
TH_REG("pck", 33, "a forwarded pack reaches the kernel", pck33)

static void pck34(void)
{
    CHEQ(run_ptx(
        "struct lp { dim3 bn; dim3 bd; size_t sh; cudaStream_t st;\n"
        "  lp(const dim3 & a, const dim3 & b, const size_t c,\n"
        "     const cudaStream_t d) : bn(a), bd(b), sh(c), st(d) {} };\n"
        "template <typename T, size_t I> using pick = T;\n"
        "template <typename K, typename... A>\n"
        "static __inline__ void go(K k, const lp & q, A&&... a) {\n"
        "    k<<<q.bn, q.bd, q.sh, q.st>>>(std::forward<A>(a)...); }\n"
        "template <typename T, typename... E>\n"
        "static __global__ void kslot(float * o, T a, int b, double c, E... e) {\n"
        "    o[0] = (float)a; o[1] = (float)b; o[2] = (float)c;\n"
        "    o[3] = (float)(e[0] + ...); }\n"
        "template <typename T, size_t... I>\n"
        "static void drive(float * o, T ** s, cudaStream_t t,\n"
        "                  std::index_sequence<I...>) {\n"
        "    const lp q = lp(dim3(1), dim3(1), 0, t);\n"
        "    go(kslot<T, pick<T *, I>...>, q, o, (T)1, 2, 3.0, (T *)s[I]...); }\n"
        "void host(float * o, float ** s, cudaStream_t t) {\n"
        "    drive<float>(o, s, t, std::make_index_sequence<2>{}); }\n"), 0);
    CHNE(strstr(obuf, ".param .u64 param0,"), NULL);
    CHNE(strstr(obuf, ".param .f32 param1,"), NULL);
    CHNE(strstr(obuf, ".param .u32 param2,"), NULL);
    CHNE(strstr(obuf, ".param .f64 param3,"), NULL);
    CHNE(strstr(obuf, ".param .u64 param4,"), NULL);
    CHNE(strstr(obuf, ".param .u64 param5"), NULL);
    CHNE(strstr(obuf, "ld.param.f32 %f1, [param1];"), NULL);
    CHNE(strstr(obuf, "ld.param.u32 %r1, [param2];"), NULL);
    CHNE(strstr(obuf, "ld.param.f64 %fd1, [param3];"), NULL);
    CHNE(strstr(obuf, "st.global.f32 [%rd4], %f1;"), NULL);
    CHNE(strstr(obuf, "cvt.rn.f32.s32 %f2, %r1;"), NULL);
    CHNE(strstr(obuf, "cvt.rn.f32.f64 %f3, %fd1;"), NULL);
    PASS();
}
TH_REG("pck", 34, "forwarded arguments land in the right slots", pck34)

static void pck35(void)
{
    CHEQ(run_ir(
        "struct lp { dim3 bn; dim3 bd; size_t sh; cudaStream_t st;\n"
        "  lp(const dim3 & a, const dim3 & b, const size_t c,\n"
        "     const cudaStream_t d) : bn(a), bd(b), sh(c), st(d) {} };\n"
        "template <typename T, size_t I> using pick = T;\n"
        "template <typename K, typename... A>\n"
        "static __inline__ void go(K k, const lp & q, A&&... a) {\n"
        "    k<<<q.bn, q.bd, q.sh, q.st>>>(std::forward<A>(a)...); }\n"
        "template <typename T, typename... E>\n"
        "static __global__ void kt(T * o, E... e) { o[0] = (T)0; }\n"
        "template <typename T, size_t... I>\n"
        "static void drive(T * o, cudaStream_t t, std::index_sequence<I...>) {\n"
        "    const lp q = lp(dim3(1), dim3(1), 0, t);\n"
        "    go(kt<T, pick<T *, I>...>, q, o, (T *)(o + I)...); }\n"
        "void host(float * o, cudaStream_t t) {\n"
        "    drive<float>(o, t, std::make_index_sequence<3>{}); }\n"), 0);
    CHNE(strstr(obuf,
        "func @kt(ptr<global, f32> %0, ptr<global, f32> %1,"
        " ptr<global, f32> %2, ptr<global, f32> %3)"), NULL);
    PASS();
}
TH_REG("pck", 35, "a template argument pack reaches a launch", pck35)

static void pck36(void)
{
    CHEQ(run_ir(
        "__device__ void bump(float & v) { v = v + 1.0f; }\n"
        "__global__ void k(float * o, float a) {\n"
        "    bump(std::forward<float &>(a)); o[0] = a; }\n"), 0);
    CHNE(strstr(obuf, "func @bump(ptr<generic, f32> %0)"), NULL);
    CHEQ(strstr(obuf, "E105"), NULL);
    PASS();
}
TH_REG("pck", 36, "std::forward keeps the caller's storage", pck36)

static void pck37(void)
{
    CHEQ(run_ir(
        "struct lp { dim3 bn; dim3 bd; size_t sh; cudaStream_t st;\n"
        "  lp(const dim3 & a, const dim3 & b, const size_t c,\n"
        "     const cudaStream_t d) : bn(a), bd(b), sh(c), st(d) {} };\n"
        "template <typename T, size_t I> using pick = T;\n"
        "template <typename K, typename... A>\n"
        "static __inline__ void go(K k, const lp & q, A&&... a) {\n"
        "    k<<<q.bn, q.bd, q.sh, q.st>>>(std::forward<A>(a)...); }\n"
        "template <typename T, typename... E>\n"
        "static __global__ void kn(T * o, E... e) { o[0] = (T)0; }\n"
        "template <typename T, size_t... I>\n"
        "static void drive(T * o, cudaStream_t t, std::index_sequence<I...>) {\n"
        "    const lp q = lp(dim3(1), dim3(1), 0, t);\n"
        "    go(kn<T, pick<T *, I>...>, q, o, (T *)(o + I)...); }\n"
        "void host(float * o, cudaStream_t t) {\n"
        "    drive<float>(o, t, std::make_index_sequence<1>{});\n"
        "    drive<float>(o, t, std::make_index_sequence<2>{}); }\n"), 0);
    CHNE(strstr(obuf,
        "func @kn(ptr<global, f32> %0, ptr<global, f32> %1)"), NULL);
    CHNE(strstr(obuf,
        "func @kn$1(ptr<global, f32> %0, ptr<global, f32> %1,"
        " ptr<global, f32> %2)"), NULL);
    PASS();
}
TH_REG("pck", 37, "index_sequence fixes the pack length", pck37)
