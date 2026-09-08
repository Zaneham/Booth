/* trpi.c -- one test per bug that got away
 *
 * z390 files these as RPI1540, RPI2001A and so on, named for the problem report
 * they came from, so years later you can still tell why a test exists at all.
 * Ours are numbered in sequence like every other family and carry the issue in
 * the description instead, which greps just as well and means there is only one
 * naming rule to remember.
 *
 * The bar for landing here is that the bug shipped. If a fix has no test, the
 * fix is a coincidence waiting to be undone. */

#include "tharns.h"
#include <stdint.h>
#include "nvidia.h"
#include "barracuda.h"
#include <stdlib.h>

static char obuf[1 << 16];

/* Reads the "; line N" annotation bir_print puts on the instruction matching
 * needle, searching only within the named function. */
static int line_of(const char *fn, const char *needle)
{
    const char *f = strstr(obuf, fn);
    if (!f) return -1;
    const char *end = strstr(f, "\n}");
    if (!end) return -1;

    const char *p = strstr(f, needle);
    if (!p || p > end) return -1;

    const char *tag = strstr(p, "; line ");
    if (!tag || tag > end) return -1;
    return atoi(tag + 7);
}

/* #160: DCE and mem2reg shuffled instructions down over the top of a deletion
 * without moving inst_lines[] along with them, so every instruction past the
 * first thing deleted reported whatever line its old neighbour had. Four sites
 * were fixed and none of them got a test, which is what this is.
 *
 * dce_chain in test_dce.cu is the shape that catches it. Two dead instructions
 * on lines 10 and 11 go away, and the store after them is on line 12. Get the
 * line table wrong and the store starts claiming line 10 or 11. */
static void rpi01(void)
{
    int rc = th_run(BC_BIN " --ir tests/test_dce.cu", obuf, (int)sizeof obuf);
    CHEQ(rc, 0);

    /* int live = a + b; */
    CHEQ(line_of("@dce_chain", "= add "), 9);
    /* out[0] = live; sits two deleted instructions later */
    CHEQ(line_of("@dce_chain", "store "), 12);

    PASS();
}
TH_REG("rpi", 1, "#160 line numbers survive DCE", rpi01)

/* Every backend lists its variant flags next to its on-switch, so a variant on
 * its own was accepted, switched nothing on, and fell through to the AST dump
 * the driver uses when no mode is set. kath printed a parse tree and exited 0
 * having compiled nothing, under whatever -o you asked for. */
static void rpi02(void)
{
    static const char *const variants[] = {
        "--bkhit", "--gfx942", "--snap", "--ssa-ra", NULL
    };
    char cmd[512];

    for (int i = 0; variants[i] != NULL; i++) {
        snprintf(cmd, sizeof cmd, "%s %s examples/cmake/vadd.cu -o build/rpi02.out",
                 BC_BIN, variants[i]);
        CHNE(th_run(cmd, obuf, (int)sizeof obuf), 0);
    }

    /* The same flags alongside their target still work. */
    snprintf(cmd, sizeof cmd,
             "%s --nvidia-ptx --bkhit examples/cmake/vadd.cu -o build/rpi02.ptx",
             BC_BIN);
    CHEQ(th_run(cmd, obuf, (int)sizeof obuf), 0);

    /* And a bare run still prints the tree, which is what the default is for. */
    snprintf(cmd, sizeof cmd, "%s examples/cmake/vadd.cu", BC_BIN);
    CHEQ(th_run(cmd, obuf, (int)sizeof obuf), 0);
    PASS();
}
TH_REG("rpi", 2, "a variant flag alone is not a target", rpi02)

/* The BIR lexer read integers with strtol, and long is 32 bits on Windows, so
 * any constant above INT32_MAX saturated to 2147483647 without a word. A hash
 * multiplier came back as a different number and the kernel still ran. */
static void rpi03(void)
{
    static const char *const mod =
        "; Booth IR\n"
        "\n"
        "func @big(ptr<global, i32> %0, i32 %1) __global__ {\n"
        "entry:\n"
        "    %2 = thread_id.x\n"
        "    %3 = mul i32 %2, 2222261027\n"
        "    %4 = gep ptr<global, i32>, %0, %2\n"
        "    store i32 %3, %4\n"
        "    ret void\n"
        "}\n";
    char cmd[512];

    FILE *f = fopen("build/rpi03.bir", "w");
    CHNE(f, NULL);
    fputs(mod, f);
    fclose(f);

    snprintf(cmd, sizeof cmd, "%s --bir-in --ir --no-cfold build/rpi03.bir", BC_BIN);
    CHEQ(th_run(cmd, obuf, (int)sizeof obuf), 0);
    CHNE(strstr(obuf, "2222261027"), NULL);
    CHEQ(strstr(obuf, "2147483647"), NULL);
    PASS();
}
TH_REG("rpi", 3, "a constant above INT32_MAX is not clamped", rpi03)

/* Counts non-overlapping occurrences of needle in obuf. A dropped statement
 * leaves the IR one instruction short rather than visibly wrong, so the count
 * is what catches it. */
static int occurs(const char *needle)
{
    int n = 0;
    for (const char *p = strstr(obuf, needle); p; p = strstr(p + 1, needle))
        n++;
    return n;
}

/* Writes src to build/<name> and returns the path, so a regression carries its
 * input with it instead of adding a fixture nobody can place later. */
static const char *scratch(const char *name, const char *src)
{
    static char path[256];
    snprintf(path, sizeof path, "build/%s", name);
    FILE *f = fopen(path, "w");
    if (!f) return NULL;
    fputs(src, f);
    fclose(f);
    return path;
}

/* #5 promoted every parameter to an alloca so it could be written to, and
 * shipped without a test. Before it, only struct params were addressable and
 * everything else refused with E108; the promotion is invisible in the IR once
 * mem2reg folds it away, so what is checked here is the behaviour: a parameter
 * is a local initialised from the argument, and reading it after a write sees
 * the write. */
static void rpi04(void)
{
    static const char *const src =
        "__device__ int setc(int x) { x = 100; return x; }\n"
        "__device__ int padd(const int *p, int n) {\n"
        "    p = p + 2; n = n * 3; return p[0] + n;\n"
        "}\n"
        "__device__ int ploop(int a, int b) {\n"
        "    for (int i = 0; i < 3; i++) { a = a + b; b = b * 2; }\n"
        "    return a;\n"
        "}\n"
        "__global__ void kmain(int *out, const int *in) {\n"
        "    int i = threadIdx.x;\n"
        "    i = i + 1;\n"
        "    out[0] = setc(7) + padd(in, 1) + ploop(in[0], 1) + i;\n"
        "}\n";
    char cmd[512];
    const char *path = scratch("rpi04.cu", src);
    CHNE(path, NULL);

    snprintf(cmd, sizeof cmd, "%s --ir %s", BC_BIN, path);
    CHEQ(th_run(cmd, obuf, (int)sizeof obuf), 0);
    CHEQ(strstr(obuf, "E108"), NULL);

    /* x = 100 wins over the incoming argument, so the body is the constant. */
    CHNE(strstr(obuf, "ret i32 100"), NULL);
    /* p = p + 2 has to move the pointer, not be dropped. */
    CHNE(strstr(obuf, "gep ptr<global, i32>, %0, 2"), NULL);
    /* Both params are rewritten every trip, so the loop head carries a phi for
     * each of them on top of the counter's. */
    CHEQ(occurs("phi i32"), 3);
    /* A __global__ entry point's parameter is no different. */
    CHNE(strstr(obuf, "__global__"), NULL);
    PASS();
}
TH_REG("rpi", 4, "#5 a parameter can be assigned to", rpi04)

/* The Triton frontend kept a name's value under the node that first bound it
 * and never moved the binding on, so every assignment after the first was
 * lowered and then dropped on the floor. Reading the name afterwards returned
 * the original value and the compiler said nothing. On an RTX 4060 Ti a kernel
 * doing v = v * 2.0 then v = v * 4.0 wrote x back unchanged. */
static void rpi05(void)
{
    static const char *const src =
        "import triton\n"
        "import triton.language as tl\n"
        "\n"
        "@triton.jit\n"
        "def kloc(x_ptr, out_ptr, n):\n"
        "    offs = tl.program_id(axis=0)\n"
        "    v = tl.load(x_ptr + offs)\n"
        "    v = v * 2.0\n"
        "    v = v * 4.0\n"
        "    tl.store(out_ptr + offs, v)\n";
    char cmd[512];
    const char *path = scratch("rpi05.py", src);
    CHNE(path, NULL);

    snprintf(cmd, sizeof cmd, "%s --triton --ir %s", BC_BIN, path);
    CHEQ(th_run(cmd, obuf, (int)sizeof obuf), 0);
    /* Both multiplies survive. Dropping the rebind leaves the second one dead
     * and DCE takes it, so the count is one before the fix and two after. */
    CHEQ(occurs("fmul f32"), 2);
    PASS();
}
TH_REG("rpi", 5, "a Triton local keeps its latest value", rpi05)

/* Same root cause on a parameter, where it could not be fixed by moving the
 * binding: the name resolved straight back to the incoming argument, so both
 * n = n + 5 and n += 5 vanished. Writing to a parameter makes the name local
 * from that point, as it does in Python and in C. */
static void rpi06(void)
{
    static const char *const tmpl =
        "import triton\n"
        "import triton.language as tl\n"
        "\n"
        "@triton.jit\n"
        "def kpar(x_ptr, out_ptr, n):\n"
        "    offs = tl.program_id(axis=0)\n"
        "    %s\n"
        "    v = tl.load(x_ptr + n)\n"
        "    tl.store(out_ptr + offs, v)\n";
    static const char *const forms[] = { "n = n + 5", "n += 5", NULL };
    char src[1024], cmd[512];

    for (int i = 0; forms[i]; i++) {
        snprintf(src, sizeof src, tmpl, forms[i]);
        const char *path = scratch("rpi06.py", src);
        CHNE(path, NULL);
        snprintf(cmd, sizeof cmd, "%s --triton --ir %s", BC_BIN, path);
        CHEQ(th_run(cmd, obuf, (int)sizeof obuf), 0);
        /* The add is the whole statement. Dropped, it is dead and DCE takes
         * it, and the load indexes off the raw argument instead. */
        CHNE(strstr(obuf, "add i32"), NULL);
    }
    PASS();
}
TH_REG("rpi", 6, "a Triton parameter can be assigned to", rpi06)

/* Rebinding across a loop back-edge needs a phi at the head, and the lowerer
 * builds one only for the counter. The accumulator every reduction is written
 * with therefore read its pre-loop value on every trip, and the sum came back
 * as whatever the last iteration computed, or as the initialiser. Nothing in
 * the pipeline objected. A refusal is the honest answer until the phi exists. */
static void rpi07(void)
{
    static const char *const tmpl =
        "import triton\n"
        "import triton.language as tl\n"
        "\n"
        "@triton.jit\n"
        "def kacc(x_ptr, out_ptr, n):\n"
        "    offs = tl.program_id(axis=0)\n"
        "    acc = 0.0\n"
        "    for i in range(0, 4):\n"
        "        %s\n"
        "    tl.store(out_ptr + offs, acc)\n";
    static const char *const forms[] = {
        "acc = acc + tl.load(x_ptr + i)", "acc += tl.load(x_ptr + i)", NULL
    };
    char src[1024], cmd[512];

    for (int i = 0; forms[i]; i++) {
        snprintf(src, sizeof src, tmpl, forms[i]);
        const char *path = scratch("rpi07.py", src);
        CHNE(path, NULL);
        snprintf(cmd, sizeof cmd, "%s --triton --ir %s", BC_BIN, path);
        CHNE(th_run(cmd, obuf, (int)sizeof obuf), 0);
        CHNE(strstr(obuf, "E141"), NULL);
    }

    /* A name whose whole life is inside the loop body crosses no back-edge and
     * has to keep working, or the refusal has eaten the ordinary case with it.
     * Both statements survive: the load, then the multiply that rebinds it. */
    static const char *const inner =
        "import triton\n"
        "import triton.language as tl\n"
        "\n"
        "@triton.jit\n"
        "def kinner(x_ptr, out_ptr, n):\n"
        "    for i in range(0, 4):\n"
        "        t = tl.load(x_ptr + i)\n"
        "        t = t * 2.0\n"
        "        tl.store(out_ptr + i, t)\n";
    const char *ipath = scratch("rpi07b.py", inner);
    CHNE(ipath, NULL);
    snprintf(cmd, sizeof cmd, "%s --triton --ir %s", BC_BIN, ipath);
    CHEQ(th_run(cmd, obuf, (int)sizeof obuf), 0);
    CHEQ(strstr(obuf, "E141"), NULL);
    CHNE(strstr(obuf, "fmul f32"), NULL);

    /* The rank-2 accumulator in a tl.dot kernel is scratch-backed and unrolled
     * rather than carried in a register, so it must still compile. */
    snprintf(cmd, sizeof cmd, "%s --triton --ir tests/tri_matmul_k.py", BC_BIN);
    CHEQ(th_run(cmd, obuf, (int)sizeof obuf), 0);
    PASS();
}
TH_REG("rpi", 7, "a loop-carried Triton rebind refuses", rpi07)

/* The NVIDIA backend gave every i1 a %p, including values defined by adds,
 * loads, phis and atomics, none of which can write one. `out[0] = (a==0)||(b==0)`
 * came out as `st.global.u32 [%rd2], %p3`, which ptxas and the driver JIT both
 * reject, so the kernel could not load at all. i1esc.cu walks the ways out:
 * store, arithmetic, call, return, shared, atomic, float conversion and a phi. */
static int pbad(const char *ptx, char *where, int wsz)
{
    const char *p = ptx;
    while ((p = strstr(p, "%p")) != NULL) {
        const char *ls = p;
        while (ls > ptx && ls[-1] != '\n') ls--;
        const char *le = strchr(p, '\n');
        if (!le) le = p + strlen(p);

        while (ls < le && (*ls == ' ' || *ls == '\t')) ls++;

        if (strncmp(ls, ".reg", 4) == 0 || strncmp(ls, "@%p", 3) == 0
         || strncmp(ls, "setp.", 5) == 0 || strncmp(ls, "selp.", 5) == 0
         || strncmp(ls, "vote.", 5) == 0 || strncmp(ls, "mov.pred", 8) == 0) {
            p = le;
            continue;
        }
        int n = (int)(le - ls);
        if (n > wsz - 1) n = wsz - 1;
        memcpy(where, ls, (size_t)n);
        where[n] = '\0';
        return 1;
    }
    return 0;
}

static char pbuf[1 << 16];

static void rpi08(void)
{
    static const char *const modes[] = { "", "--no-mem2reg", NULL };
    char cmd[512], bad[192];

    for (int i = 0; modes[i] != NULL; i++) {
        snprintf(cmd, sizeof cmd,
                 "%s --nvidia-ptx %s tests/i1esc.cu -o build/rpi04.ptx",
                 BC_BIN, modes[i]);
        CHEQ(th_run(cmd, obuf, (int)sizeof obuf), 0);

        FILE *f = fopen("build/rpi04.ptx", "r");
        CHNE(f, NULL);
        size_t n = fread(pbuf, 1, sizeof pbuf - 1, f);
        pbuf[n] = '\0';
        fclose(f);

        if (pbad(pbuf, bad, (int)sizeof bad))
            printf("  %s: %s\n", modes[i][0] ? modes[i] : "default", bad);
        CHEQ(pbad(pbuf, bad, (int)sizeof bad), 0);
    }
    PASS();
}
TH_REG("rpi", 8, "a predicate never reaches a wider slot", rpi08)

/* A `bool` is one byte, so the stride between elements of a bool array is one
 * byte, and five size functions disagreed about that. width/8 is zero for i1:
 * the alloca sites clamped the zero to a minimum and the GEP sites did not, so
 * AMD multiplied the index by zero and every element of a bool array resolved
 * to the same address. x86-64 and RV64 substituted 4, Tensix refused.
 *
 * bir_bsz is the one answer now. These pin the answer rather than any one
 * backend's arithmetic. */
static void rpi09(void)
{
    int rc = th_run(BC_BIN " --ir tests/bstride.cu", obuf, (int)sizeof obuf);
    CHEQ(rc, 0);
    CHNE(strstr(obuf, "store i64 1,"), NULL);
    CHEQ(strstr(obuf, "store i64 4,"), NULL);
    PASS();
}
TH_REG("rpi", 9, "sizeof(bool) is one byte", rpi09)

/* The AMD case is the serious one because it was silent. A scaled GEP whose
 * stride is zero collapses a whole array onto element zero, and the shape is
 * a v_mul_lo_u32 against an immediate 0. */
static void rpi10(void)
{
    const char *p;
    int rc = th_run(BC_BIN " --amdgpu tests/bstride.cu", obuf, (int)sizeof obuf);
    CHEQ(rc, 0);

    for (p = obuf; (p = strstr(p, "v_mul_lo_u32")) != NULL; p++) {
        const char *nl = strchr(p, '\n');
        if (nl == NULL) break;
        CHEQ(nl - p >= 3 && nl[-3] == ',' && nl[-1] == '0', 0);
    }

    /* A bool[64] in LDS reserves 64 bytes, not the 4 the clamp used to give. */
    CHNE(strstr(obuf, "64 LDS bytes"), NULL);
    CHNE(strstr(obuf, "8 scratch bytes"), NULL);
    PASS();
}
TH_REG("rpi", 10, "a bool array does not stride by zero on AMD", rpi10)

/* Storage size, array stride and access width are three questions with one
 * answer for a bool. PTX strides by 1 and must therefore touch one byte;
 * a .u32 access at a one-byte stride writes over the next three elements. */
static void rpi11(void)
{
    int rc = th_run(BC_BIN " --nvidia-ptx -o build/rpi11.ptx tests/bstride.cu",
                    obuf, (int)sizeof obuf);
    CHEQ(rc, 0);

    FILE *f = fopen("build/rpi11.ptx", "r");
    CHNE(f, NULL);
    if (f == NULL) return;
    obuf[fread(obuf, 1, sizeof obuf - 1, f)] = '\0';
    fclose(f);

    CHNE(strstr(obuf, ", 1, %rd"), NULL);
    CHEQ(strstr(obuf, ", 4, %rd"), NULL);
    CHNE(strstr(obuf, "ld.global.u8"), NULL);
    CHNE(strstr(obuf, "st.global.u8"), NULL);
    CHNE(strstr(obuf, "ld.shared.u8"), NULL);
    CHNE(strstr(obuf, "st.local.u8"), NULL);
    CHEQ(strstr(obuf, "ld.global.u32"), NULL);
    CHEQ(strstr(obuf, "st.shared.u32"), NULL);
    PASS();
}
TH_REG("rpi", 11, "a bool load reads the byte it strides by", rpi11)

/* Sizing a struct by summing its fields is the same mistake in a different
 * hat: an array of { char; int; } strides by 8 on the host and strode by 5
 * here, so element i past the first landed inside its predecessor. One size
 * function that pads the way C does settles both. */
static void rpi12(void)
{
    int rc = th_run(BC_BIN " --nvidia-ptx -o build/rpi12.ptx tests/bpad.cu",
                    obuf, (int)sizeof obuf);
    CHEQ(rc, 0);

    FILE *f = fopen("build/rpi12.ptx", "r");
    CHNE(f, NULL);
    if (f == NULL) return;
    obuf[fread(obuf, 1, sizeof obuf - 1, f)] = '\0';
    fclose(f);

    CHNE(strstr(obuf, ", 8, %rd"), NULL);
    CHEQ(strstr(obuf, ", 5, %rd"), NULL);
    PASS();
}
TH_REG("rpi", 12, "a struct array strides by its padded size", rpi12)
/* looks_like_cast read `( ident )` before any prefix operator as a cast to a
 * type called ident, without ever asking whether ident named a type. Only `+`
 * and `-` are also infix, so `(a) + (b)` became a cast applied to `+(b)` and
 * the left operand vanished with no diagnostic. `-` left the tell, `sub 0, b`.
 *
 * The fixture stays on disk after a failure, so build/rpi13.cu names the case
 * that broke. */
static void rpi13(void)
{
    static const struct { const char *ex; const char *ir; } cs[] = {
        { "(a) + (b)",        "add i32 %0, %1"     },
        { "(a) - (b)",        "sub i32 %0, %1"     },
        { "(a) + b",          "add i32 %0, %1"     },
        { "(b) + (a)",        "add i32 %1, %0"     },
        { "(a) + (a * 2)",    "add i32 %0, %2"     },
        { "(a) + a * 2",      "add i32 %0, %2"     },
        { "(a) + (a)",        "add i32 %0, %0"     },
        { "((a)) + (b)",      "add i32 %0, %1"     },
        { "(a) * (b)",        "mul i32 %0, %1"     },
        { "(a) & (b)",        "and i32 %0, %1"     },
        { "(a) / (b)",        "sdiv i32 %0, %1"    },
        { "(a) < (b)",        "icmp slt i32 %0, %1"},
        { "(a * 2) + (a)",    "add i32 %2, %0"     },
        { "(a + 1) + (a * 2)","add i32 %2, %3"     },
    };
    char cmd[512];

    for (size_t i = 0; i < sizeof cs / sizeof cs[0]; i++) {
        FILE *f = fopen("build/rpi13.cu", "w");
        CHNE(f, NULL);
        fprintf(f, "__device__ int f(int a, int b){ return %s; }\n", cs[i].ex);
        fclose(f);

        snprintf(cmd, sizeof cmd, "%s --ir build/rpi13.cu", BC_BIN);
        CHEQ(th_run(cmd, obuf, (int)sizeof obuf), 0);
        CHNE(strstr(obuf, cs[i].ir), NULL);
    }
    PASS();
}
TH_REG("rpi", 13, "a parenthesised variable is not a cast", rpi13)

/* The other half of the same test. Tightening it must not cost a real cast,
 * so every shape a typedef name reaches the parser in is checked here, and
 * `(pair){...}` is one the loose rule never recognised at all. */
static void rpi14(void)
{
    static const struct { const char *src; const char *ir; } cs[] = {
        { "typedef int myint;\n"
          "__device__ int f(int a, int b){ return (myint)(a) + b; }\n",
          "add i32 %0, %1" },

        { "typedef int myint;\n"
          "__device__ int f(int a, int b){ (void)a; return (myint) + b; }\n",
          "ret i32 %1" },

        { "typedef int myint;\n"
          "__device__ int f(int a, int b){ (void)a; (void)b; return (myint)-1; }\n",
          "ret i32 4294967295" },

        { "typedef int myint;\n"
          "__device__ int f(void *p){ return *(myint*)p; }\n",
          "bitcast ptr<global, void> %0 to ptr<global, i32>" },

        { "typedef struct { int x; int y; } pair;\n"
          "__device__ int f(int a, int b){ pair p = (pair){ a, b }; return p.y; }\n",
          "store i32 %1, %6" },

        { "__device__ int f(int a, int b){ return (int)(uint32_t)-1 + a + b; }\n",
          "add i32 4294967295, %0" },

        { "using myint = int;\n"
          "__device__ int f(int a, int b){ return (myint)(a) + b; }\n",
          "add i32 %0, %1" },
    };
    char cmd[512];

    for (size_t i = 0; i < sizeof cs / sizeof cs[0]; i++) {
        FILE *f = fopen("build/rpi14.cu", "w");
        CHNE(f, NULL);
        fputs(cs[i].src, f);
        fclose(f);

        snprintf(cmd, sizeof cmd, "%s --ir build/rpi14.cu", BC_BIN);
        CHEQ(th_run(cmd, obuf, (int)sizeof obuf), 0);
        CHNE(strstr(obuf, cs[i].ir), NULL);
    }
    PASS();
}
TH_REG("rpi", 14, "a typedef name still casts", rpi14)

/* A template type parameter is a type name for the body below it. It is not a
 * typedef and never reached the registry, so it survived on the loose rule
 * alone and (T)a + b would have started returning b. */
static void rpi15(void)
{
    static const char *const src =
        "template<typename T> __device__ T g(T a, T b){ return (T)a + b; }\n";
    char cmd[512];

    FILE *f = fopen("build/rpi15.cu", "w");
    CHNE(f, NULL);
    fputs(src, f);
    fclose(f);

    snprintf(cmd, sizeof cmd, "%s --parse build/rpi15.cu", BC_BIN);
    CHEQ(th_run(cmd, obuf, (int)sizeof obuf), 0);
    CHNE(strstr(obuf, "(binary +"), NULL);
    CHNE(strstr(obuf, "(cast"), NULL);
    PASS();
}
TH_REG("rpi", 15, "a template type parameter names a type", rpi15)

/* A variadic callee recorded only its named parameters, so every call that
 * passed anything through the ellipsis came back as the wrong arity. Every
 * GGML_ASSERT in ggml-cuda hit it. */
static void rpi16(void)
{
    static const char *const src =
        "void ab(const char *f, int l, const char *m, ...);\n"
        "__device__ void g(void){ ab(\"x\", 1, \"%s\", \"y\"); }\n";
    char cmd[512];

    FILE *f = fopen("build/rpi16.cu", "w");
    CHNE(f, NULL);
    fputs(src, f);
    fclose(f);

    snprintf(cmd, sizeof cmd, "%s --sema build/rpi16.cu", BC_BIN);
    CHEQ(th_run(cmd, obuf, (int)sizeof obuf), 0);
    CHEQ(strstr(obuf, "E073"), NULL);
    PASS();
}
TH_REG("rpi", 16, "an ellipsis takes as many arguments as given", rpi16)

/* Every ggml-cuda file died at the backend on E110 because the module carried
 * a string literal, so none of them ever reached PTX. NVIDIA lays the bytes in
 * .global now and takes the address with mov.u64 (#94). */
static int cxrn2(const char *src)
{
    char cmd[512];
    FILE *f = fopen("build/rpi25.cu", "w");

    if (!f) return -1;
    fputs(src, f);
    fclose(f);
    snprintf(cmd, sizeof cmd, "%s --ir build/rpi25.cu", BC_BIN);
    return th_run(cmd, obuf, (int)sizeof obuf);
}

static const char *rpi_ptx(const char *src, const char *stem)
{
    static char ptx[1 << 16];
    char cmd[512], cu[128], out[128];
    FILE *f, *g;
    size_t n;

    ptx[0] = '\0';
    snprintf(cu, sizeof cu, "build/%s.cu", stem);
    snprintf(out, sizeof out, "build/%s.ptx", stem);
    f = fopen(cu, "w");
    if (!f) return NULL;
    fputs(src, f);
    fclose(f);
    remove(out);
    snprintf(cmd, sizeof cmd, "%s --nvidia-ptx %s -o %s", BC_BIN, cu, out);
    if (th_run(cmd, obuf, (int)sizeof obuf) != 0) return NULL;
    g = fopen(out, "rb");
    if (!g) return NULL;
    n = fread(ptx, 1, sizeof ptx - 1, g);
    fclose(g);
    ptx[n] = '\0';
    return ptx;
}

static void rpi17(void)
{
    const char *p = rpi_ptx(
        "__device__ const char *w(void){ return \"hi\"; }\n"
        "__global__ void k(const char **o){ o[0] = w(); }\n", "rpi17");

    CHNE(p, NULL);
    CHEQ(strstr(obuf, "E110"), NULL);
    CHNE(strstr(p, ".b8 $_str_0[3] = {104, 105, 0}"), NULL);
    CHNE(strstr(p, "mov.u64 %rd3, $_str_0"), NULL);
    PASS();
}
TH_REG("rpi", 17, "#94 a string literal reaches PTX", rpi17)

/* A union was laid out as a struct, so every field got its own storage: this
 * one measured 16 where the hardware says 8, and everything after it in the
 * enclosing struct sat at the wrong offset. */
static void rpi18(void)
{
    static const char *const src =
        "union U { int i; float f; double d; };\n"
        "struct S { U u; int t; };\n"
        "__global__ void k(int *o){ o[0] = (int)sizeof(U);"
        " o[1] = (int)sizeof(S); }\n";
    char cmd[512];
    FILE *f = fopen("build/rpi18.cu", "w");

    CHNE(f, NULL);
    fputs(src, f);
    fclose(f);
    snprintf(cmd, sizeof cmd, "%s --ir build/rpi18.cu", BC_BIN);
    CHEQ(th_run(cmd, obuf, (int)sizeof obuf), 0);
    CHNE(strstr(obuf, "store i32 8,"), NULL);
    CHNE(strstr(obuf, "store i32 16,"), NULL);
    PASS();
}
TH_REG("rpi", 18, "a union is as big as its largest member", rpi18)

/* ggml-common.h puts the scale pair in an anonymous union, and nothing nested
 * inside a struct body was collected at all, so x[i].dm came back as an unknown
 * field and __half22float2 refused behind it. */
static void rpi19(void)
{
    static const char *const src =
        "typedef struct {\n"
        "  union { struct { half d; half m; } data; half2 dm; };\n"
        "  unsigned char qs[16];\n"
        "} blk;\n"
        "__global__ void k(const blk *x, float2 *o, int *n){\n"
        "  n[0] = (int)sizeof(blk);\n"
        "  o[0] = __half22float2(x[0].dm);\n"
        "  o[1].x = __half2float(x[0].data.m);\n"
        "}\n";
    char cmd[512];
    FILE *f = fopen("build/rpi19.cu", "w");

    CHNE(f, NULL);
    fputs(src, f);
    fclose(f);
    snprintf(cmd, sizeof cmd, "%s --ir build/rpi19.cu", BC_BIN);
    CHEQ(th_run(cmd, obuf, (int)sizeof obuf), 0);
    CHEQ(strstr(obuf, "E110"), NULL);
    CHEQ(strstr(obuf, "E143"), NULL);
    CHNE(strstr(obuf, "store i32 20,"), NULL);
    PASS();
}
TH_REG("rpi", 19, "a member of an anonymous union resolves", rpi19)

/* assert in device code refused outright. It is a trap with a message and the
 * message needs a call to vprintf that no GPU backend here makes, so the
 * condition gets the trap and the text is what we give up. */
static void rpi20(void)
{
    const char *p = rpi_ptx(
        "__global__ void k(int *o, int n){ assert(n > 0); o[0] = n; }\n",
        "rpi20");

    CHNE(p, NULL);
    CHEQ(strstr(obuf, "E143"), NULL);
    CHNE(strstr(p, "setp.gt.s32"), NULL);
    CHNE(strstr(p, "trap;"), NULL);
    PASS();
}
TH_REG("rpi", 20, "assert becomes a conditional trap", rpi20)

/* BIR_GLOBAL_REF was a no-op on the PTX backend, so a kernel reading a
 * __device__ table indexed off whatever the register happened to hold. */
static void rpi21(void)
{
    const char *p = rpi_ptx(
        "__device__ int tbl[4];\n"
        "__global__ void k(int *o, int i){ o[0] = tbl[i]; }\n", "rpi21");

    CHNE(p, NULL);
    CHNE(strstr(p, ".global .align 8 .b8 tbl[16];"), NULL);
    CHNE(strstr(p, ", tbl;"), NULL);
    PASS();
}
TH_REG("rpi", 21, "a __device__ global gets a PTX symbol", rpi21)

static void rpi22(void)
{
    const char *p = rpi_ptx(
        "__global__ void k(float *o, int i){ extern __shared__ float sf[];\n"
        "  sf[i] = 3.0f; o[0] = sf[i+1]; }\n", "rpi22");

    CHNE(p, NULL);
    CHNE(strstr(p, ".extern .shared .align 4 .b8 __dynshmem[];"), NULL);
    CHNE(strstr(p, ", __dynshmem;"), NULL);
    CHNE(strstr(p, ", 4, "), NULL);
    CHEQ(strstr(p, "has no storage size"), NULL);
    PASS();
}
TH_REG("rpi", 22, "extern __shared__ becomes the dynamic block", rpi22)

static void rpi23(void)
{
    const char *p = rpi_ptx(
        "__global__ void k(double *o, int i){ extern __shared__ double sd[];\n"
        "  extern __shared__ char sc[];\n"
        "  sd[i] = 1.0; o[0] = (double)sc[i]; }\n", "rpi23");

    CHNE(p, NULL);
    CHNE(strstr(p, ".extern .shared .align 8 .b8 __dynshmem[];"), NULL);
    CHEQ(strstr(p, ".align 1 .b8 __dynshmem"), NULL);
    CHNE(strstr(p, ", 8, "), NULL);
    CHNE(strstr(p, ", 1, "), NULL);
    PASS();
}
TH_REG("rpi", 23, "two extern __shared__ arrays share one base", rpi23)

static void rpi24(void)
{
    static const char *const src =
        "__global__ void k(float *o, int i){ extern __shared__ float sf[];\n"
        "  sf[i] = 3.0f; o[0] = sf[i+1]; }\n";
    static const char *const hosts[] = { "--amdgpu", "--cpu", "--rv64", NULL };
    const char *path = scratch("rpi24.cu", src);
    char cmd[512];

    CHNE(path, NULL);
    for (int i = 0; hosts[i] != NULL; i++) {
        snprintf(cmd, sizeof cmd, "%s %s %s -o build/rpi24.out",
                 BC_BIN, hosts[i], path);
        CHNE(th_run(cmd, obuf, (int)sizeof obuf), 0);
        CHNE(strstr(obuf, "no storage size"), NULL);
    }
    PASS();
}
TH_REG("rpi", 24, "a host with no dynamic shared refuses", rpi24)

/* A static member was filed under its bare name in the same table as every
 * namespace-scope constant, so a __device__ global of that name never got a
 * look in: the kernel read the struct's 7 instead of the global's 3. */
static void rpi25(void)
{
    const char *p = rpi_ptx(
        "struct a { static constexpr int v = 7; };\n"
        "__device__ int v = 3;\n"
        "__global__ void k(int *o){ o[0] = v; }\n", "rpi25");

    CHNE(p, NULL);
    if (p) {
        CHNE(strstr(p, "ld.global.u32"), NULL);
        CHEQ(strstr(p, ", 7;"), NULL);
    }
    PASS();
}
TH_REG("rpi", 25, "a member name does not leak to file scope", rpi25)

/* Qualified lookup stripped one scope at a time until something matched, so
 * any class whose member name happened to exist elsewhere borrowed it. */
static void rpi26(void)
{
    CHNE(cxrn2("struct a { static constexpr int v = 7; };\n"
               "struct e { };\n"
               "__global__ void k(int *o){ o[0] = e::v; }\n"), 0);
    CHEQ(strstr(obuf, "store i32 7"), NULL);
    PASS();
}
TH_REG("rpi", 26, "another class does not lend its member", rpi26)

static const char *rpi_msl(const char *src, const char *stem)
{
    static char msl[1 << 16];
    char cmd[512], cu[128], out[128];
    FILE *f, *g;
    size_t n;

    msl[0] = '\0';
    snprintf(cu, sizeof cu, "build/%s.cu", stem);
    snprintf(out, sizeof out, "build/%s.metal", stem);
    f = fopen(cu, "w");
    if (!f) return NULL;
    fputs(src, f);
    fclose(f);
    remove(out);
    snprintf(cmd, sizeof cmd, "%s --metal %s -o %s", BC_BIN, cu, out);
    if (th_run(cmd, obuf, (int)sizeof obuf) != 0) return NULL;
    g = fopen(out, "rb");
    if (!g) return NULL;
    n = fread(msl, 1, sizeof msl - 1, g);
    fclose(g);
    msl[n] = '\0';
    return msl;
}

static void rpi27(void)
{
    const char *m = rpi_msl(
        "__global__ void k(float *o, const float *a, int i){\n"
        "  __shared__ float st[32];\n"
        "  __shared__ float4 sv[8];\n"
        "  st[i] = a[i]; sv[i].x = a[i]; __syncthreads();\n"
        "  o[i] = st[0] + sv[0].x; }\n", "rpi27");

    CHNE(m, NULL);
    CHEQ(strstr(m, "TODO"), NULL);
    CHNE(strstr(m, "threadgroup float g"), NULL);
    CHNE(strstr(m, "[32];"), NULL);
    CHNE(strstr(m, "[8];"), NULL);
    CHNE(strstr(m, "threadgroup_barrier(mem_flags::mem_threadgroup);"), NULL);
    CHEQ(strstr(m, "= {}"), NULL);
    PASS();
}
TH_REG("rpi", 27, "a __shared__ array becomes threadgroup", rpi27)

static void rpi28(void)
{
    const char *m = rpi_msl(
        "struct P { float x; int y; };\n"
        "__global__ void k(P *o, int i){ o[i].x = 1.0f; o[i].y = 2; }\n",
        "rpi28");

    CHNE(m, NULL);
    CHEQ(strstr(m, "TODO"), NULL);
    CHNE(strstr(m, "struct s"), NULL);
    CHNE(strstr(m, "float f0;"), NULL);
    CHNE(strstr(m, "int f1;"), NULL);
    PASS();
}
TH_REG("rpi", 28, "a struct reaches Metal as a real struct", rpi28)

static void rpi29(void)
{
    static const char *const src =
        "__global__ void k(int *o){ atomicAdd(o, 1); }\n";
    const char *path = scratch("rpi29.cu", src);
    char cmd[512];
    FILE *g;

    CHNE(path, NULL);
    remove("build/rpi29.metal");
    snprintf(cmd, sizeof cmd, "%s --metal %s -o build/rpi29.metal",
             BC_BIN, path);
    CHNE(th_run(cmd, obuf, (int)sizeof obuf), 0);
    CHNE(strstr(obuf, "E500"), NULL);
    CHNE(strstr(obuf, "atomic_add"), NULL);
    g = fopen("build/rpi29.metal", "rb");
    CHEQ(g, NULL);
    PASS();
}
TH_REG("rpi", 29, "Metal refuses an op it cannot express", rpi29)

static void rpi30(void)
{
    const char *m = rpi_msl(
        "__global__ void k(float *o, int i){ extern __shared__ float sf[];\n"
        "  sf[i] = 3.0f; o[0] = sf[i+1]; }\n", "rpi30");

    CHNE(m, NULL);
    CHEQ(strstr(m, "TODO"), NULL);
    CHNE(strstr(m, "threadgroup uchar * dsh [[threadgroup(0)]]"), NULL);
    CHNE(strstr(m, "(threadgroup float *)dsh"), NULL);
    PASS();
}
TH_REG("rpi", 30, "extern __shared__ binds threadgroup(0)", rpi30)

/* --tensix took the size of a __shared__ array nowhere: shared_alloc got one
 * Dst row like a private alloca and every load and store bound to Dst row 0
 * regardless of the pointer, so extern __shared__ float[] and __shared__
 * float[32] compiled to byte-identical output. Dst is a register file with an
 * immediate row address, not addressable memory, and shared memory on a Tensix
 * tile lives in L1 behind a circular buffer nothing binds. Both forms refuse. */
static void rpi31(void)
{
    static const char *const dyn =
        "__global__ void k(float *o, const float *a, int i){\n"
        "  extern __shared__ float sf[];\n"
        "  sf[i] = a[i]; __syncthreads(); o[i] = sf[0]; }\n";
    static const char *const fix =
        "__global__ void k(float *o, const float *a, int i){\n"
        "  __shared__ float sf[32];\n"
        "  sf[i] = a[i]; __syncthreads(); o[i] = sf[0]; }\n";
    const char *const srcs[2] = { dyn, fix };
    char cmd[512];

    for (int i = 0; i < 2; i++) {
        const char *path = scratch("rpi31.cu", srcs[i]);

        CHNE(path, NULL);
        snprintf(cmd, sizeof cmd, "%s --tensix %s -o build/rpi31_compute.cpp",
                 BC_BIN, path);
        CHNE(th_run(cmd, obuf, (int)sizeof obuf), 0);
        CHNE(strstr(obuf, "E524"), NULL);
    }
    PASS();
}
TH_REG("rpi", 31, "__shared__ on Tensix refuses by name", rpi31)

/* The SFPU isel answered division with a multiply, integer division with a
 * copy of the numerator, atomics and warp collectives with a zero, and a
 * switch with nothing at all, so every one of them compiled clean and lied.
 * Each now names itself and the code says which. */
static void rpi32(void)
{
    static const struct { const char *body; const char *eid; } cases[] = {
        { "o[i] = a[i] / a[i + 1];",                        "E522" },
        { "o[i] = (float)(i / (int)a[0]);",                 "E523" },
        { "atomicAdd(o, a[i]);",                            "E520" },
        { "o[i] = __shfl_down_sync(0xffffffffu, a[i], 1);", "E521" },
        { "switch (i) { case 0: o[0] = 1.0f; break;\n"
          "  default: o[1] = 2.0f; break; }",               "E525" },
    };
    char src[512], cmd[512];

    for (unsigned c = 0; c < sizeof cases / sizeof cases[0]; c++) {
        const char *path;

        snprintf(src, sizeof src,
                 "__global__ void k(float *o, const float *a, int i){ %s }\n",
                 cases[c].body);
        path = scratch("rpi32.cu", src);
        CHNE(path, NULL);
        snprintf(cmd, sizeof cmd, "%s --tensix %s -o build/rpi32_compute.cpp",
                 BC_BIN, path);
        CHNE(th_run(cmd, obuf, (int)sizeof obuf), 0);
        CHNE(strstr(obuf, cases[c].eid), NULL);
    }
    PASS();
}
TH_REG("rpi", 32, "the Tensix SFPU names what it cannot do", rpi32)

/* --rv-elf already refused an unsized shared array, but with a bare stderr
 * line, so nothing downstream could say which layer had stopped. */
static void rpi33(void)
{
    static const char *const src =
        "__global__ void k(float *o, const float *a, int i){\n"
        "  extern __shared__ float sf[];\n"
        "  sf[i] = a[i]; o[i] = sf[0]; }\n";
    const char *path = scratch("rpi33.cu", src);
    char cmd[512];

    CHNE(path, NULL);
    snprintf(cmd, sizeof cmd, "%s --rv-elf %s -o build/rpi33.elf",
             BC_BIN, path);
    CHNE(th_run(cmd, obuf, (int)sizeof obuf), 0);
    CHNE(strstr(obuf, "E535"), NULL);
    PASS();
}
TH_REG("rpi", 33, "a baby-core refusal carries its code", rpi33)

/* The emitted host program hardcoded total_elements = 1024 * 1024 and sized
 * every DRAM buffer off it, so it was a working-looking program that ignored
 * the launch it was generated for. The grid is a launch-time quantity and the
 * host program is the launch site, so it takes it as an argument. */
static void rpi34(void)
{
    static const char *const src =
        "__global__ void k(float *o, const float *a, int i){\n"
        "  o[i] = a[i] + 1.0f; }\n";
    const char *path = scratch("rpi34.cu", src);
    char cmd[512];
    FILE *f;
    size_t n;

    CHNE(path, NULL);
    remove("build/rpi34_host.cpp");
    snprintf(cmd, sizeof cmd, "%s --tensix %s -o build/rpi34_compute.cpp",
             BC_BIN, path);
    CHEQ(th_run(cmd, obuf, (int)sizeof obuf), 0);

    f = fopen("build/rpi34_host.cpp", "rb");
    CHNE(f, NULL);
    if (!f) return;
    n = fread(obuf, 1, sizeof obuf - 1, f);
    fclose(f);
    obuf[n] = '\0';

    CHEQ(strstr(obuf, "1024 * 1024"), NULL);
    CHNE(strstr(obuf, "int main(int argc, char **argv)"), NULL);
    CHNE(strstr(obuf, "std::strtoul(argv[1]"), NULL);
    CHNE(strstr(obuf, "EnqueueWriteBuffer"), NULL);
    CHNE(strstr(obuf, "EnqueueReadBuffer"), NULL);
    PASS();
}
TH_REG("rpi", 34, "the Tensix host is told its element count", rpi34)

/* The x86-64 and RV64 emitters ended their opcode switch with a default arm
 * that stored a zero into the result slot and carried on, so any BIR op they
 * did not implement compiled to "the answer is 0" with a zero exit. A plain
 * switch statement lowers to BIR_SWITCH, which neither of them handles, so
 * both used to emit an object where the switch had simply not happened. */
static void rpi35(void)
{
    static const char *const src =
        "__global__ void k(int *o, int m){ int r = 0;\n"
        "  switch (m) { case 0: r = 7; break; case 1: r = 9; break;\n"
        "               default: r = -1; }\n"
        "  o[0] = r; }\n";
    static const char *const hosts[] = { "--cpu", "--rv64", NULL };
    const char *path = scratch("rpi35.cu", src);
    char cmd[512];

    CHNE(path, NULL);
    for (int i = 0; hosts[i] != NULL; i++) {
        snprintf(cmd, sizeof cmd, "%s %s %s -o build/rpi35.o",
                 BC_BIN, hosts[i], path);
        CHNE(th_run(cmd, obuf, (int)sizeof obuf), 0);
        CHNE(strstr(obuf, "E542"), NULL);
    }
    PASS();
}
TH_REG("rpi", 35, "an unlowerable op is not a stored zero", rpi35)

/* The PTX atomic selector had no case for BIR_ATOMIC_SUB and a default arm
 * that picked atom.add, so atomicSub added instead of subtracting and said
 * nothing. Refuse until there is a real lowering for it. */
static void rpi36(void)
{
    const char *path = scratch("rpi36.cu",
        "__global__ void k(int *o){ atomicSub(o, 3); }\n");
    char cmd[512];

    CHNE(path, NULL);
    snprintf(cmd, sizeof cmd, "%s --nvidia-ptx %s -o build/rpi36.ptx",
             BC_BIN, path);
    CHNE(th_run(cmd, obuf, (int)sizeof obuf), 0);
    CHNE(strstr(obuf, "E543"), NULL);
    PASS();
}
TH_REG("rpi", 36, "atomicSub is not an atomic add on PTX", rpi36)

/* A kernel past NV_MAX_PARAMS had its parameter list truncated at the cap
 * while ld.param kept naming the ones past it, so ggml's im2col_3d_kernel
 * (forty parameters) produced PTX referencing params it never declared. */
static void rpi37(void)
{
    char src[4096], cmd[512];
    int w = snprintf(src, sizeof src, "__global__ void k(int *o");
    for (int i = 0; i < 40; i++)
        w += snprintf(src + w, sizeof src - (size_t)w, ", int a%d", i);
    w += snprintf(src + w, sizeof src - (size_t)w, "){");
    for (int i = 0; i < 40; i++)
        w += snprintf(src + w, sizeof src - (size_t)w, " o[0]+=a%d;", i);
    snprintf(src + w, sizeof src - (size_t)w, " }\n");

    const char *path = scratch("rpi37.cu", src);
    CHNE(path, NULL);
    remove("build/rpi37.ptx");
    snprintf(cmd, sizeof cmd, "%s --nvidia-ptx %s -o build/rpi37.ptx",
             BC_BIN, path);
    CHEQ(th_run(cmd, obuf, (int)sizeof obuf), 0);

    FILE *f = fopen("build/rpi37.ptx", "rb");
    CHNE(f, NULL);
    static char ptx[1 << 16];
    size_t n = fread(ptx, 1, sizeof ptx - 1, f);
    fclose(f);
    ptx[n] = '\0';
    CHNE(strstr(ptx, ".param .u32 param40"), NULL);
    CHNE(strstr(ptx, "[param40]"), NULL);
    PASS();
}
TH_REG("rpi", 37, "a 40-parameter kernel declares them all", rpi37)

/* PTX isel treated BIR_SWITCH as a no-op, so ggml's pool1d and pool2d emitted
 * a kernel with the switch missing entirely and a zero exit to go with it. */
static void rpi38(void)
{
    const char *p = rpi_ptx(
        "__global__ void k(int *o, int m){ int r = 0;\n"
        "  switch (m) { case 0: r = 7; break; case 1: r = 9; break;\n"
        "               default: r = -1; }\n"
        "  o[0] = r; }\n", "rpi38");

    CHNE(p, NULL);
    CHNE(strstr(p, "setp.eq.u32"), NULL);
    CHNE(strstr(p, "@%p"), NULL);
    PASS();
}
TH_REG("rpi", 38, "a switch reaches PTX as a compare chain", rpi38)

static void rpi39(void)
{
    const char *p = rpi_ptx(
        "template <typename T, typename U>\n"
        "static __global__ void kp(T * o, U e) { o[0] = (T)*e; }\n"
        "void h(float * o, float ** s) { kp<float, float *><<<1,1>>>(o, s[0]); }\n", "rpi39");

    CHNE(p, NULL);
    CHNE(strstr(p, ".param .u64 param0,"), NULL);
    CHNE(strstr(p, ".param .u64 param1"), NULL);
    CHEQ(strstr(p, ".param .f32 param1"), NULL);
    CHNE(strstr(p, "ld.global.f32"), NULL);
    PASS();
}
TH_REG("rpi", 39, "an explicit T* template argument stays T*", rpi39)

static void rpi40(void)
{
    const char *p = rpi_ptx(
        "template <typename T, typename... E>\n"
        "static __global__ void ke(T * o, E... e) { o[0] = (T)(*e + ...); }\n"
        "template <typename T, size_t... I>\n"
        "static void d(T * o, T ** s, std::index_sequence<I...>) {\n"
        "    ke<T><<<1,1>>>(o, (const T *)s[I]...); }\n"
        "void h(float * o, float ** s) {\n"
        "    d<float>(o, s, std::make_index_sequence<2>{}); }\n", "rpi40");

    CHNE(p, NULL);
    CHNE(strstr(p, ".param .u64 param1,"), NULL);
    CHNE(strstr(p, ".param .u64 param2"), NULL);
    CHNE(strstr(p, "add.rn.f32"), NULL);
    PASS();
}
TH_REG("rpi", 40, "a pack in launch arguments is expanded", rpi40)

/* A cast from a pointer to a floating type has no conversion, but the cast
 * lowering only refused when one side was an aggregate and the other a scalar.
 * A pointer is neither, so (float)p fell through to the int-to-float rung and
 * emitted sitofp on an address, which assembles and means nothing. */
static void rpi41(void)
{
    const char *path = scratch("rpi41.cu",
        "__global__ void k(float *o, int *p) { o[0] = (float)p; }\n");
    char cmd[512];

    CHNE(path, NULL);
    snprintf(cmd, sizeof cmd, "%s --ir %s", BC_BIN, path);
    th_run(cmd, obuf, (int)sizeof obuf);
    CHNE(strstr(obuf, "E440"), NULL);
    CHEQ(strstr(obuf, "sitofp ptr"), NULL);
    PASS();
}
TH_REG("rpi", 41, "a pointer never converts to a float", rpi41)

/* ---- SFPU functional model, one lane ----
 *
 * Transcribed from WormholeB0/TensixTile/TensixCoprocessor: SFPMOV, SFPLOADI,
 * SFPIADD, SFPSHFT, SFPAND, SFPOR, SFPXOR, SFPSETCC, SFPLOAD and SFPSTORE.
 * The tests below compile a kernel, read the words tensix_emit_binary wrote,
 * seed the Dst rows isel_param loads its parameters from, and run the stream.
 * A wrong Mod1 stops being an encoding nobody reads and becomes a wrong number. */

#define SFP_ROW_P0 2u

static uint32_t sfp_lr[16];
static int      sfp_flag;

static int32_t sfp_i12(uint32_t w)
{
    uint32_t v = (w >> 12) & 0xFFFu;
    return (v & 0x800u) ? (int32_t)(v | 0xFFFFF000u) : (int32_t)v;
}

/* Runs until the first SFPSTORE and reports the value it would have written.
 * Returns 0 on success, -1 if the stream held an opcode this model has no
 * entry for, which is a signal to extend the model rather than to trust it. */
static int sfp_run(const uint32_t *w, int n, const uint32_t *dst, uint32_t ndst,
                   uint32_t *out)
{
    memset(sfp_lr, 0, sizeof sfp_lr);
    sfp_lr[9]  = 0u;
    sfp_lr[10] = 0x3F800000u;
    sfp_lr[15] = 0u;
    sfp_flag   = 1;

    for (int i = 0; i < n; i++) {
        uint32_t word = w[i];
        uint32_t op   = word >> 24;
        uint32_t vc   = (word >> 8) & 0xFu;
        uint32_t vd   = (word >> 4) & 0xFu;
        uint32_t md   = word & 0xFu;

        if (op >= 0xA0u || op == 0x02u || op == 0x8Fu) continue;

        switch (op) {
        case 0x70u: {
            uint32_t lr = (word >> 20) & 0xFu;
            uint32_t rw = word & 0x3FFFu;
            if (lr < 8u) sfp_lr[lr] = rw < ndst ? dst[rw] : 0u;
            break;
        }
        case 0x72u: {
            uint32_t lr = (word >> 20) & 0xFu;
            *out = sfp_lr[lr];
            return 0;
        }
        case 0x71u: {
            uint32_t lr  = (word >> 20) & 0xFu;
            uint32_t md0 = (word >> 16) & 0xFu;
            uint32_t im  = word & 0xFFFFu;
            if (lr >= 8u) break;
            if (md0 == 2u)       sfp_lr[lr] = im;
            else if (md0 == 4u)  sfp_lr[lr] = (uint32_t)(int32_t)(int16_t)im;
            else if (md0 == 8u)  sfp_lr[lr] = (im << 16) | (sfp_lr[lr] & 0xFFFFu);
            else if (md0 == 10u) sfp_lr[lr] = (sfp_lr[lr] & 0xFFFF0000u) | im;
            else return -1;
            break;
        }
        case 0x7Cu:
            if (vd < 8u) sfp_lr[vd] = sfp_lr[vc];
            break;
        case 0x79u: {
            if (vd >= 8u) break;
            if (md & 1u)
                sfp_lr[vd] = sfp_lr[vc] + (uint32_t)sfp_i12(word);
            else if (md & 2u)
                sfp_lr[vd] = sfp_lr[vc] - sfp_lr[vd];
            else
                sfp_lr[vd] = sfp_lr[vc] + sfp_lr[vd];
            if (!(md & 4u)) sfp_flag = ((int32_t)sfp_lr[vd] < 0);
            if (md & 8u)    sfp_flag = !sfp_flag;
            break;
        }
        case 0x7Au: {
            int32_t sa = (md & 1u) ? sfp_i12(word) : (int32_t)sfp_lr[vc];
            if (vd >= 8u) break;
            if (sa >= 0) sfp_lr[vd] <<= (uint32_t)sa & 31u;
            else         sfp_lr[vd] >>= (uint32_t)(-sa) & 31u;
            break;
        }
        case 0x80u: if (vd < 8u) sfp_lr[vd] = ~sfp_lr[vc]; break;
        case 0x7Eu: if (vd < 8u) sfp_lr[vd] &= sfp_lr[vc]; break;
        case 0x7Fu: if (vd < 8u) sfp_lr[vd] |= sfp_lr[vc]; break;
        case 0x8Du: if (vd < 8u) sfp_lr[vd] ^= sfp_lr[vc]; break;
        case 0x7Bu: {
            int32_t c = (int32_t)sfp_lr[vc];
            if (vd >= 12u) break;
            if (md == 0u)      sfp_flag = (c <  0);
            else if (md == 2u) sfp_flag = (c != 0);
            else if (md == 4u) sfp_flag = (c >= 0);
            else if (md == 6u) sfp_flag = (c == 0);
            else return -1;
            break;
        }
        default:
            return -1;
        }
    }
    *out = 0u;
    return -1;
}

static uint32_t sfp_words[4096];

/* Compile src through --tensix and read back the raw Tensix words. */
static int sfp_bld(const char *src, const char *stem)
{
    char cu[128], out[128], bin[128], cmd[512];
    FILE *f;
    int n = 0;

    snprintf(cu,  sizeof cu,  "build/%s.cu", stem);
    snprintf(out, sizeof out, "build/%s_compute.cpp", stem);
    snprintf(bin, sizeof bin, "build/%s_compute.bin", stem);
    f = fopen(cu, "w");
    if (!f) return -1;
    fputs(src, f);
    fclose(f);
    remove(bin);
    snprintf(cmd, sizeof cmd, "%s --tensix %s -o %s", BC_BIN, cu, out);
    if (th_run(cmd, obuf, (int)sizeof obuf) != 0) return -1;
    f = fopen(bin, "rb");
    if (!f) return -1;
    {
        uint8_t b[4];
        while (n < (int)(sizeof sfp_words / sizeof sfp_words[0]) &&
               fread(b, 1, 4, f) == 4)
            sfp_words[n++] = (uint32_t)b[0] | ((uint32_t)b[1] << 8)
                           | ((uint32_t)b[2] << 16) | ((uint32_t)b[3] << 24);
    }
    fclose(f);
    return n;
}

/* Two int parameters land in the Dst rows isel_param picked, one past the
 * pointer. Seed those and run. */
static int sfp_ab(int n, uint32_t a, uint32_t b, uint32_t *out)
{
    uint32_t dst[8];
    memset(dst, 0, sizeof dst);
    dst[SFP_ROW_P0 + 1u] = a;
    dst[SFP_ROW_P0 + 2u] = b;
    return sfp_run(sfp_words, n, dst, 8u, out);
}

/* SFPIADD Mod1 1 is ARG_IMM, so with an immediate of 0 it computes VD = VC:
 * one operand, copied. Every integer subtract on this backend did that, and
 * the compare paths fed the same non-difference into SFPSETCC. The mode wanted
 * is 2 (ARG_2SCOMP_LREG_DST), which is VD = VC - VD_old, so the subtrahend has
 * to be in the destination first. Get the operands the other way round and
 * a - b quietly becomes b - a. */
static void rpi42(void)
{
    static const char *const src =
        "__global__ void k(int *o, int a, int b){ *o = a - b; }\n";
    uint32_t got;
    int n = sfp_bld(src, "rpi42");

    CHECK(n > 0);
    if (n <= 0) return;

    CHEQ(sfp_ab(n, 3u, 5u, &got), 0);
    CHEQ((int32_t)got, -2);
    CHEQ(sfp_ab(n, 5u, 3u, &got), 0);
    CHEQ((int32_t)got, 2);
    CHEQ(sfp_ab(n, 0u, 0x80000000u, &got), 0);
    CHEQ(got, 0x80000000u);
    PASS();
}
TH_REG("rpi", 42, "an integer subtract subtracts", rpi42)

/* The compare paths carried the same ARG_IMM mistake, and on top of it the
 * TT_CC_ table did not match SFPSETCC: GE was 1 and NE was 3, both of which
 * set SFPSETCC_MOD1_IMM_BIT0 and drive LaneFlags from a zero immediate, so
 * those two compares were always false. Underneath both, a compare was the
 * sign of lhs - rhs, which is only the answer while that difference fits: the
 * last six cases are the ones where it does not. SFPSETCC writes no register,
 * so what is checked here is LaneFlags, which is the whole output it has. */
static void rpi43(void)
{
    static const struct {
        const char *expr; uint32_t a; uint32_t b; int want;
    } cases[] = {
        { "a < b",  3u, 5u, 1 }, { "a < b",  5u, 3u, 0 },
        { "a > b",  3u, 5u, 0 }, { "a > b",  5u, 3u, 1 },
        { "a >= b", 3u, 5u, 0 }, { "a >= b", 5u, 3u, 1 },
        { "a <= b", 3u, 5u, 1 }, { "a <= b", 3u, 3u, 1 },
        { "a == b", 3u, 5u, 0 }, { "a == b", 3u, 3u, 1 },
        { "a != b", 3u, 5u, 1 }, { "a != b", 3u, 3u, 0 },
        { "(unsigned)a <  (unsigned)b", 0xFFFFFFFFu, 1u, 0 },
        { "(unsigned)a <  (unsigned)b", 1u, 0xFFFFFFFFu, 1 },
        { "(unsigned)a >= (unsigned)b", 0xFFFFFFFFu, 1u, 1 },
        { "(unsigned)a >  (unsigned)b", 1u, 0xFFFFFFFFu, 0 },
        { "(unsigned)a <= (unsigned)b", 1u, 0xFFFFFFFFu, 1 },
        { "a < b",  0x7FFFFFFFu, 0xFFFFFFFFu, 0 },
        { "a < b",  0x80000000u, 1u, 1 },
        { "a > b",  0x7FFFFFFFu, 0xFFFFFFFFu, 1 },
        { "a >= b", 0x80000000u, 1u, 0 },
        { "a <= b", 0x7FFFFFFFu, 0xFFFFFFFFu, 0 },
        { "a == b", 0x80000000u, 0x80000000u, 1 },
    };
    char src[256];
    uint32_t got;

    for (unsigned c = 0; c < sizeof cases / sizeof cases[0]; c++) {
        int n;

        snprintf(src, sizeof src,
                 "__global__ void k(int *o, int a, int b){ *o = (%s); }\n",
                 cases[c].expr);
        n = sfp_bld(src, "rpi43");
        CHECK(n > 0);
        if (n <= 0) return;
        sfp_ab(n, cases[c].a, cases[c].b, &got);
        CHEQ(sfp_flag, cases[c].want);
    }
    PASS();
}
TH_REG("rpi", 43, "an integer compare compares", rpi43)

/* BIR_MUL went out as SFPMUL, which is FP32, so it multiplied two integer bit
 * patterns as floats. A constant multiplier is the common case and it is a
 * short shift-and-add chain, exact at the full 32 bits. */
static void rpi44(void)
{
    static const struct { const char *k; uint32_t a; uint32_t want; } cases[] = {
        { "0",           7u, 0u },
        { "1",           7u, 7u },
        { "12",          7u, 84u },
        { "1024",        7u, 7168u },
        { "-3",          7u, 0xFFFFFFEBu },
        { "2147483647",  3u, 0x7FFFFFFDu },
        { "65535",  0x10001u, 0xFFFF0000u + 0xFFFFu },
    };
    char src[256];
    uint32_t got;

    for (unsigned c = 0; c < sizeof cases / sizeof cases[0]; c++) {
        int n;

        snprintf(src, sizeof src,
                 "__global__ void k(int *o, int a, int b){ (void)b;\n"
                 "  *o = a * %s; }\n", cases[c].k);
        n = sfp_bld(src, "rpi44");
        CHECK(n > 0);
        if (n <= 0) return;
        CHEQ(sfp_ab(n, cases[c].a, 0u, &got), 0);
        CHEQ(got, cases[c].want);
    }
    PASS();
}
TH_REG("rpi", 44, "a constant multiply is exact at 32 bits", rpi44)

/* Runtime times runtime has no instruction on either part: Wormhole has none
 * at all and Blackhole's SFPMUL24 is 23 by 23. It goes out as a branchless
 * shift-and-add over the 32 bit positions, which is expensive and exactly
 * right modulo 2^32. The last two cases are the ones that would catch a
 * 23-bit answer wearing a 32-bit face. */
static void rpi45(void)
{
    static const struct { uint32_t a, b, want; } cases[] = {
        { 7u, 12u, 84u },
        { 0xFFFFFFFDu, 5u, 0xFFFFFFF1u },
        { 0u, 123u, 0u },
        { 0x00800000u, 3u, 0x01800000u },
        { 0x12345678u, 0x9ABCDEF0u, 0x242D2080u },
        { 0xFFFFFFFFu, 0xFFFFFFFFu, 1u },
    };
    uint32_t got;
    int n = sfp_bld("__global__ void k(int *o, int a, int b){ *o = a * b; }\n",
                    "rpi45");

    CHECK(n > 0);
    if (n <= 0) return;

    for (unsigned c = 0; c < sizeof cases / sizeof cases[0]; c++) {
        CHEQ(sfp_ab(n, cases[c].a, cases[c].b, &got), 0);
        CHEQ(got, cases[c].want);
    }
    PASS();
}
TH_REG("rpi", 45, "a runtime multiply is exact at 32 bits", rpi45)

/* A runtime multiply is 224 instructions, so a kernel that multiplies enough
 * times reaches the end of the minst arena. emit() used to answer that with a
 * bare `return 0`, which drops the rest of the kernel and says nothing. */
static void rpi46(void)
{
    char cmd[512];
    FILE *f = fopen("build/rpi46.cu", "w");

    CHNE(f, NULL);
    if (!f) return;
    fputs("__global__ void k(int *o, int a, int b){\n"
          "  int t = a;\n", f);
    for (int i = 0; i < 1400; i++)
        fputs("  t = t * b;\n", f);
    fputs("  *o = t; }\n", f);
    fclose(f);

    snprintf(cmd, sizeof cmd, "%s --tensix build/rpi46.cu -o "
             "build/rpi46_compute.cpp", BC_BIN);
    CHNE(th_run(cmd, obuf, (int)sizeof obuf), 0);
    CHNE(strstr(obuf, "E537"), NULL);
    PASS();
}
TH_REG("rpi", 46, "a full Tensix arena refuses, not truncates", rpi46)

/* A 32-bit constant went out as two SFPLOADI halves using mod 2 then mod 4,
 * but both of those write the whole register: mod 2 zero-extends a 16-bit
 * immediate and mod 4 sign-extends one, so the second load threw the first
 * away and every constant arrived as the sign-extended low half. The halves
 * are mod 8 and mod 10. */
static void rpi47(void)
{
    static const struct { const char *k; uint32_t a; uint32_t want; } cases[] = {
        { "305419896",  0u, 0x12345678u },
        { "-2000000",   0u, 0xFFE17B80u },
        { "65536",      1u, 0x00010001u },
    };
    char src[256];
    uint32_t got;

    for (unsigned c = 0; c < sizeof cases / sizeof cases[0]; c++) {
        int n;

        snprintf(src, sizeof src,
                 "__global__ void k(int *o, int a, int b){ (void)b;\n"
                 "  *o = a + %s; }\n", cases[c].k);
        n = sfp_bld(src, "rpi47");
        CHECK(n > 0);
        if (n <= 0) return;
        CHEQ(sfp_ab(n, cases[c].a, 0u, &got), 0);
        CHEQ(got, cases[c].want);
    }
    PASS();
}
TH_REG("rpi", 47, "a 32-bit constant loads both halves", rpi47)

static void rpi48(void)
{
    const char *path = scratch("rpi48.cu",
        "__global__ void k(int *o){ o[1] = atomicMin(o, 5); }\n");
    char cmd[512];

    CHNE(path, NULL);
    snprintf(cmd, sizeof cmd, "%s --nvidia-ptx %s -o build/rpi48.ptx",
             BC_BIN, path);
    CHNE(th_run(cmd, obuf, (int)sizeof obuf), 0);
    CHNE(strstr(obuf, "E560"), NULL);
    CHNE(strstr(obuf, "atomicMin"), NULL);
    PASS();
}
TH_REG("rpi", 48, "atomicMin on int refuses, it cannot know sign", rpi48)

static void rpi49(void)
{
    const char *path = scratch("rpi49.cu",
        "__global__ void k(unsigned *o){ o[1] = atomicMax(o, 5u); }\n");
    char cmd[512];

    CHNE(path, NULL);
    snprintf(cmd, sizeof cmd, "%s --nvidia-ptx %s -o build/rpi49.ptx",
             BC_BIN, path);
    CHNE(th_run(cmd, obuf, (int)sizeof obuf), 0);
    CHNE(strstr(obuf, "E560"), NULL);
    CHNE(strstr(obuf, "atomicMax"), NULL);
    PASS();
}
TH_REG("rpi", 49, "atomicMax on unsigned refuses the same way", rpi49)

static void rpi50(void)
{
    const char *p = rpi_ptx(
        "__global__ void k(int *o){ __shared__ int s[64];\n"
        "  int r = atomicAdd(&s[threadIdx.x], 1);\n"
        "  o[0] = r + atomicAdd(&o[1], r); }\n", "rpi50");

    CHNE(p, NULL);
    CHNE(strstr(p, "atom.shared.add.u32"), NULL);
    CHNE(strstr(p, "atom.global.add.u32"), NULL);
    PASS();
}
TH_REG("rpi", 50, "a __shared__ atomic is not atom.global", rpi50)

static void rpi51(void)
{
    const char *p = rpi_ptx(
        "__global__ void k(unsigned long long *q){\n"
        "  q[1] = atomicAdd(q, 7ull); }\n", "rpi51");

    CHNE(p, NULL);
    CHNE(strstr(p, "atom.global.add.u64"), NULL);
    CHEQ(strstr(p, "add.u32"), NULL);
    PASS();
}
TH_REG("rpi", 51, "a 64-bit atomic is not atom.add.u32", rpi51)

static void rpi52(void)
{
    const char *p = rpi_ptx(
        "__global__ void k(int *o){ o[2] = atomicCAS(o, 3, 9); }\n", "rpi52");
    const char *c;

    CHNE(p, NULL);
    c = strstr(p, "atom.global.cas.b32");
    CHNE(c, NULL);
    CHEQ(strstr(c, ", 0;"), NULL);
    PASS();
}
TH_REG("rpi", 52, "atomicCAS swaps its value, not zero", rpi52)

static void rpi53(void)
{
    const char *path = scratch("rpi53.cu",
        "__global__ void k(int *o){ int t[4]; t[0] = 1;\n"
        "  int r = atomicAdd(&t[threadIdx.x & 3], 2);\n"
        "  o[0] = r + t[0]; }\n");
    char cmd[512];

    CHNE(path, NULL);
    snprintf(cmd, sizeof cmd, "%s --nvidia-ptx %s -o build/rpi53.ptx",
             BC_BIN, path);
    CHNE(th_run(cmd, obuf, (int)sizeof obuf), 0);
    CHNE(strstr(obuf, "E561"), NULL);
    PASS();
}
TH_REG("rpi", 53, "an atomic on a local names its refusal", rpi53)

/* RV64 had no BIR_CALL arm at all, so every device function call fell to the
 * top-level default and was quietly lowered to a stored zero until that arm
 * became a refusal. These four cover the call ABI and the branch reach that
 * came with it; all of them read the emitted .text back, because the bugs are
 * in the bytes and nothing upstream of the encoder can see them. */

static unsigned char rtx[1 << 18];
static long rtxn;

static int rv_text(const char *src, const char *stem)
{
    char cmd[512], cu[128], out[128];
    unsigned char h[64], sh[64];
    unsigned long long shoff = 0, off, siz, stroff = 0;
    unsigned nsh, i, nameo;
    FILE *f;

    rtxn = 0;
    snprintf(cu, sizeof cu, "build/%s.cu", stem);
    snprintf(out, sizeof out, "build/%s.o", stem);
    f = fopen(cu, "w");
    if (!f) return 0;
    fputs(src, f);
    fclose(f);
    remove(out);
    snprintf(cmd, sizeof cmd, "%s --rv64 %s -o %s", BC_BIN, cu, out);
    if (th_run(cmd, obuf, (int)sizeof obuf) != 0) return 0;

    f = fopen(out, "rb");
    if (!f) return 0;
    if (fread(h, 1, sizeof h, f) != sizeof h) { fclose(f); return 0; }
    for (i = 0; i < 8; i++) shoff |= (unsigned long long)h[40 + i] << (8 * i);
    nsh = (unsigned)h[60] | ((unsigned)h[61] << 8);
    nameo = (unsigned)h[62] | ((unsigned)h[63] << 8);
    if (nameo >= nsh) { fclose(f); return 0; }

    if (fseek(f, (long)(shoff + 64 * nameo), SEEK_SET) != 0) { fclose(f); return 0; }
    if (fread(sh, 1, sizeof sh, f) != sizeof sh) { fclose(f); return 0; }
    for (i = 0; i < 8; i++) stroff |= (unsigned long long)sh[24 + i] << (8 * i);

    for (i = 0; i < nsh; i++) {
        char nm[16];
        unsigned no = 0, j;
        if (fseek(f, (long)(shoff + 64 * i), SEEK_SET) != 0) break;
        if (fread(sh, 1, sizeof sh, f) != sizeof sh) break;
        for (j = 0; j < 4; j++) no |= (unsigned)sh[j] << (8 * j);
        if (fseek(f, (long)(stroff + no), SEEK_SET) != 0) break;
        if (fread(nm, 1, sizeof nm, f) != sizeof nm) continue;
        nm[sizeof nm - 1] = '\0';
        if (strcmp(nm, ".text") != 0) continue;
        off = 0; for (j = 0; j < 8; j++) off |= (unsigned long long)sh[24 + j] << (8 * j);
        siz = 0; for (j = 0; j < 8; j++) siz |= (unsigned long long)sh[32 + j] << (8 * j);
        if (siz > sizeof rtx) siz = sizeof rtx;
        if (fseek(f, (long)off, SEEK_SET) != 0) break;
        rtxn = (long)fread(rtx, 1, (size_t)siz, f);
        break;
    }
    fclose(f);
    return rtxn > 0;
}

static unsigned rvw(long i)
{
    long b = i * 4;
    return (unsigned)rtx[b] | ((unsigned)rtx[b + 1] << 8)
         | ((unsigned)rtx[b + 2] << 16) | ((unsigned)rtx[b + 3] << 24);
}

/* B and J scatter their immediate across the word (ISA 2.5.1, 2.5.2). */
static int bimm(unsigned w)
{
    int v = (int)(((w >> 8) & 0xF) << 1) | (int)(((w >> 25) & 0x3F) << 5)
          | (int)(((w >> 7) & 1) << 11) | (int)(((w >> 31) & 1) << 12);
    return (w & 0x80000000u) ? v - 8192 : v;
}
static int jimm(unsigned w)
{
    int v = (int)(((w >> 21) & 0x3FF) << 1) | (int)(((w >> 20) & 1) << 11)
          | (int)(((w >> 12) & 0xFF) << 12) | (int)(((w >> 31) & 1) << 20);
    return (w & 0x80000000u) ? v - 2097152 : v;
}
static int iimm(unsigned w)
{
    int v = (int)((w >> 20) & 0xFFF);
    return (v & 0x800) ? v - 4096 : v;
}
static int simm(unsigned w)
{
    int v = (int)((w >> 7) & 0x1F) | (int)(((w >> 25) & 0x7F) << 5);
    return (v & 0x800) ? v - 4096 : v;
}

static void rpi54(void)
{
    static const char *src =
        "__device__ int leaf(int x) { return x * 3 + 1; }\n"
        "__global__ void k(int *o) {\n"
        "    o[threadIdx.x] = leaf((int)threadIdx.x); }\n";
    long i;
    int ncall = 0;

    if (!rv_text(src, "rpi54")) { CHEQ(1, 0); return; }
    CHEQ(strstr(obuf, "E542"), NULL);
    for (i = 0; i < rtxn / 4; i++) {
        unsigned w = rvw(i);
        if ((w & 0x7F) == 0x6F && ((w >> 7) & 0x1F) == 1) ncall++;
    }
    CHNE(ncall, 0);
    PASS();
}
TH_REG("rpi", 54, "a device function call reaches RV64", rpi54)

/* mk_B masked the offset into the field with no range check, so a branch past
 * 4 KiB wrapped and landed somewhere else entirely, silently. The fix inverts
 * the condition and steps over a jal, which reaches 1 MiB, so the generated
 * body below has to be long enough to prove a jump really did need the room. */
static void rpi55(void)
{
    static char src[1 << 16];
    int n = 0, i, far = 0, wide = 0;
    long j;

    n += snprintf(src + n, sizeof src - (size_t)n,
                  "__global__ void k(int *o, int m)\n{\n"
                  "    int t = (int)threadIdx.x; int s = 0;\n"
                  "    if (t < m) {\n");
    for (i = 0; i < 400; i++)
        n += snprintf(src + n, sizeof src - (size_t)n,
                      "        s += (t + %d) * %d;\n", i, i + 1);
    n += snprintf(src + n, sizeof src - (size_t)n, "    }\n    o[t] = s;\n}\n");

    if (!rv_text(src, "rpi55")) { CHEQ(1, 0); return; }
    for (j = 0; j < rtxn / 4; j++) {
        unsigned w = rvw(j);
        if ((w & 0x7F) == 0x63) { int d = bimm(w); if (d > 64 || d < -64) wide++; }
        if ((w & 0x7F) == 0x6F) { int d = jimm(w); if (d > 4096 || d < -4096) far++; }
    }
    CHNE(far, 0);
    CHEQ(wide, 0);
    PASS();
}
TH_REG("rpi", 55, "a long branch does not wrap on RV64", rpi55)

/* The frame pass reserved a slot per parameter index and a second one per
 * BIR_PARAM instruction. Those two only coincide for the first function in
 * the module, whose instruction indices start at zero; in every later one the
 * prologue stored the incoming argument where the body never looked. */
static void rpi56(void)
{
    static const char *src =
        "__device__ int one(int x) { return x + 1; }\n"
        "__device__ int two(int x) { return x + 2; }\n"
        "__global__ void k(int *o) {\n"
        "    o[threadIdx.x] = one((int)threadIdx.x) + two((int)threadIdx.x); }\n";
    long i;
    int nfn = 0, sto = 0, got = 0;

    if (!rv_text(src, "rpi56")) { CHEQ(1, 0); return; }
    for (i = 0; i < rtxn / 4; i++) {
        unsigned w = rvw(i);
        if ((w & 0x7F) == 0x13 && ((w >> 12) & 7) == 0
            && ((w >> 7) & 0x1F) == 2 && ((w >> 15) & 0x1F) == 2 && iimm(w) < 0) {
            if (++nfn > 2) break;
            sto = 0;
        }
        if (nfn != 2) continue;
        if (!sto && (w & 0x7F) == 0x23 && ((w >> 12) & 7) == 3
            && ((w >> 15) & 0x1F) == 8 && ((w >> 20) & 0x1F) == 10) sto = simm(w);
        else if (sto && !got && (w & 0x7F) == 0x03 && ((w >> 12) & 7) == 3
                 && ((w >> 15) & 0x1F) == 8) { got = 1; CHEQ(iimm(w), sto); }
    }
    CHEQ(nfn >= 2, 1);
    CHEQ(got, 1);
    PASS();
}
TH_REG("rpi", 56, "an RV64 callee reads the arg it was given", rpi56)

/* Arguments past the eighth go on the stack, and the psABI wants sp on a
 * 128-bit boundary for the whole of the call, so the outgoing block has to be
 * rounded rather than grown eight bytes at a time. */
static void rpi57(void)
{
    static const char *src =
        "__device__ int many(int a,int b,int c,int d,int e,int f,\n"
        "                    int g,int h,int i,int j,int k,int l)\n"
        "{ return a+b+c+d+e+f+g+h+i+j+k+l; }\n"
        "__global__ void k(int *o) { int t = (int)threadIdx.x;\n"
        "    o[t] = many(t,t+1,t+2,t+3,t+4,t+5,t+6,t+7,t+8,t+9,t+10,t+11); }\n";
    long i;
    int onstk = 0, misal = 0;

    if (!rv_text(src, "rpi57")) { CHEQ(1, 0); return; }
    for (i = 0; i < rtxn / 4; i++) {
        unsigned w = rvw(i);
        if ((w & 0x7F) == 0x23 && ((w >> 12) & 7) == 3
            && ((w >> 15) & 0x1F) == 2 && ((w >> 20) & 0x1F) == 5) onstk++;
        if ((w & 0x7F) == 0x13 && ((w >> 12) & 7) == 0
            && ((w >> 7) & 0x1F) == 2 && ((w >> 15) & 0x1F) == 2 && (iimm(w) & 15)) misal++;
    }
    CHNE(onstk, 0);
    CHEQ(misal, 0);
    PASS();
}
TH_REG("rpi", 57, "RV64 stack arguments keep sp aligned", rpi57)

/* psABI 2.1 passes an aggregate wider than two XLENs by reference, and Boo/* kept every value in one 8-byte slot, so a0 carried eight bytes off the
 * front of the struct. Aggregates now travel as a pointer, so the call lays
 * down; E582 still names the operand shapes RV64 genuinely cannot pass. */
static void rpi58(void)
{
    static const char *src =
        "struct Big { int a, b, c, d, e, f; };\n"
        "__device__ int take(struct Big s) { return s.a + s.f; }\n"
        "__global__ void k(int *o) {\n"
        "    struct Big s;\n"
        "    s.a = (int)threadIdx.x; s.b = 1; s.c = 2;\n"
        "    s.d = 3; s.e = 4; s.f = 5;\n"
        "    o[threadIdx.x] = take(s); }\n";
    char cmd[512];
    FILE *f = fopen("build/rpi58.cu", "w");

    if (!f) { CHEQ(1, 0); return; }
    fputs(src, f);
    fclose(f);
    snprintf(cmd, sizeof cmd, "%s --rv64 build/rpi58.cu -o build/rpi58.o", BC_BIN);
    CHEQ(th_run(cmd, obuf, (int)sizeof obuf), 0);
    CHEQ(strstr(obuf, "E582"), NULL);
    CHEQ(strstr(obuf, "E542"), NULL);
    PASS();
}
TH_REG("rpi", 58, "RV64 passes an aggregate by reference", rpi58)

static int ecnt(const char *hay, const char *ndl)
{
    size_t n = strlen(ndl);
    int c = 0;

    for (const char *p = hay; (p = strstr(p, ndl)) != NULL; p += n) c++;
    return c;
}

static int rpi_err(const char *src, const char *stem)
{
    char cmd[512], cu[128];
    FILE *f;

    snprintf(cu, sizeof cu, "build/%s.cu", stem);
    f = fopen(cu, "w");
    if (!f) return -1;
    fputs(src, f);
    fclose(f);
    snprintf(cmd, sizeof cmd, "%s --ir %s", BC_BIN, cu);
    return th_run(cmd, obuf, (int)sizeof obuf);
}

static void rpi59(void)
{
    int rc = rpi_err(
        "__global__ void k(float *d){\n"
        "  wmma::fragment<wmma::matrix_b, 16, 16, 16, half> f;\n"
        "  d[0] = 1.0f; }\n", "rpi59");

    CHNE(rc, 0);
    CHNE(strstr(obuf, "E903"), NULL);
    CHEQ(strstr(obuf, "E022"), NULL);
    CHEQ(strstr(obuf, "E020"), NULL);
    PASS();
}
TH_REG("rpi", 59, "a multiplicand with no layout refuses", rpi59)

static void rpi60(void)
{
    int rc = rpi_err(
        "__global__ void k(const half *p, float *d, int l){\n"
        "  wmma::fragment<wmma::matrix_a, 16, 16, 16, half,\n"
        "                 wmma::row_major> a;\n"
        "  wmma::fragment<wmma::matrix_b, 16, 16, 16, half,\n"
        "                 wmma::col_major> b;\n"
        "  wmma::fragment<wmma::accumulator, 16, 16, 16, float> c;\n"
        "  wmma::load_matrix_sync(a, p, l);\n"
        "  wmma::load_matrix_sync(b, p, l);\n"
        "  wmma::mma_sync(c, a, b, c);\n"
        "  wmma::store_matrix_sync(d, c, l, wmma::mem_row_major); }\n",
        "rpi60");

    CHEQ(rc, 0);
    CHEQ(ecnt(obuf, "alloca ptr<private, [16 x f16]>"), 2);
    CHEQ(ecnt(obuf, "alloca ptr<private, [8 x f32]>"), 1);
    PASS();
}
TH_REG("rpi", 60, "three wmma decls size to the ISA count", rpi60)

static void rpi61(void)
{
    int rc = rpi_err(
        "__global__ void k(float *d){\n"
        "  cub::BlockScan<float, 128> s;\n"
        "  d[0] = 1.0f; }\n", "rpi61");

    CHNE(rc, 0);
    CHEQ(ecnt(obuf, "E681"), 1);
    CHNE(strstr(obuf, "cub::BlockScan"), NULL);
    CHEQ(strstr(obuf, "E022"), NULL);
    PASS();
}
TH_REG("rpi", 61, "an unknown qualified template-id refuses", rpi61)

static void rpi62(void)
{
    int rc = rpi_err(
        "namespace cooperative_groups { }\n"
        "namespace cg = cooperative_groups;\n"
        "__global__ void k(float *d){\n"
        "  cg::grid_group g = cg::this_grid();\n"
        "  g.sync();\n"
        "  d[0] = 1.0f; }\n", "rpi62");

    CHEQ(rc, 0);
    CHNE(strstr(obuf, "gridbar"), NULL);
    CHEQ(strstr(obuf, "E155"), NULL);
    CHEQ(strstr(obuf, "E105"), NULL);
    PASS();
}
TH_REG("rpi", 62, "a grid barrier lowers to its own op", rpi62)

static void rpi63(void)
{
    const char *p = rpi_ptx(
        "namespace cooperative_groups { }\n"
        "namespace cg = cooperative_groups;\n"
        "__global__ void k(float *d){\n"
        "  cg::thread_block b = cg::this_thread_block();\n"
        "  b.sync();\n"
        "  d[b.thread_rank()] = (float)b.size(); }\n", "rpi63");

    CHNE(p, NULL);
    CHNE(strstr(p, "bar.sync"), NULL);
    CHNE(strstr(p, "%tid.x"), NULL);
    CHNE(strstr(p, "%ntid.x"), NULL);
    PASS();
}
TH_REG("rpi", 63, "a thread_block sync is still bar.sync", rpi63)

static void rpi64(void)
{
    const char *p = rpi_ptx(
        "namespace cooperative_groups { }\n"
        "namespace cg = cooperative_groups;\n"
        "__global__ void k(float *d){\n"
        "  cg::grid_group g = cg::this_grid();\n"
        "  d[g.thread_rank()] = 1.0f; }\n", "rpi64");

    CHNE(p, NULL);
    CHNE(strstr(p, "%ctaid.x"), NULL);
    CHNE(strstr(p, "%ctaid.z"), NULL);
    CHNE(strstr(p, "%nctaid.y"), NULL);
    CHEQ(strstr(p, "bar.sync"), NULL);
    PASS();
}
TH_REG("rpi", 64, "a grid thread_rank spans the whole grid", rpi64)

static void rpi65(void)
{
    const char *p = rpi_ptx(
        "namespace cooperative_groups { }\n"
        "namespace cg = cooperative_groups;\n"
        "__global__ void k(int *d){\n"
        "  cg::grid_group g = cg::this_grid();\n"
        "  d[0] = g.size() + g.num_blocks() + g.block_rank(); }\n", "rpi65");

    CHNE(p, NULL);
    CHNE(strstr(p, "%nctaid.z"), NULL);
    CHNE(strstr(p, "%ntid.z"), NULL);
    CHEQ(strstr(p, "bar.sync"), NULL);
    PASS();
}
TH_REG("rpi", 65, "a grid size multiplies grid by block", rpi65)

static void rpi66(void)
{
    int rc = rpi_err(
        "namespace cooperative_groups { }\n"
        "namespace cg = cooperative_groups;\n"
        "__device__ void f(cg::grid_group g){ g.sync(); }\n"
        "__global__ void k(float *d){ f(cg::this_grid()); d[0] = 1.0f; }\n",
        "rpi66");

    CHEQ(rc, 0);
    CHNE(strstr(obuf, "gridbar"), NULL);
    CHEQ(strstr(obuf, "E155"), NULL);
    PASS();
}
TH_REG("rpi", 66, "a grid group parameter syncs the grid", rpi66)

static void rpi67(void)
{
    const char *p = rpi_ptx(
        "__global__ void k(unsigned *o){\n"
        "  o[0] = __byte_perm(0x03020100u, 0x07060504u, 0x7531u); }\n",
        "rpi67");

    CHNE(p, NULL);
    CHNE(strstr(p, "117768961"), NULL);
    PASS();
}
TH_REG("rpi", 67, "__byte_perm picks bytes out of {y:x}", rpi67)

static void rpi68(void)
{
    const char *p = rpi_ptx(
        "__global__ void k(unsigned *o){\n"
        "  o[0] = __byte_perm(0x800000FFu, 0u, 0x8888u);\n"
        "  o[1] = __byte_perm(0x80FF0100u, 0u, 0x8888u); }\n", "rpi68");

    CHNE(p, NULL);
    CHNE(strstr(p, "mov.u32 %r1, -1;"), NULL);
    CHNE(strstr(p, "mov.u32 %r2, 0;"), NULL);
    PASS();
}
TH_REG("rpi", 68, "__byte_perm nibble 8 smears the sign bit", rpi68)

static void rpi69(void)
{
    const char *p = rpi_ptx(
        "__global__ void k(unsigned *o){\n"
        "  o[0] = __vsubss4(0x80808080u, 0x01010101u); }\n", "rpi69");

    CHNE(p, NULL);
    CHNE(strstr(p, "-2139062144"), NULL);
    PASS();
}
TH_REG("rpi", 69, "__vsubss4 saturates low, not wraps", rpi69)

static void rpi70(void)
{
    const char *p = rpi_ptx(
        "__global__ void k(unsigned *o){\n"
        "  o[0] = __vsubss4(0x7F7F7F7Fu, 0xFFFFFFFFu);\n"
        "  o[1] = __vsubss4(0x0A140000u, 0x05050000u); }\n", "rpi70");

    CHNE(p, NULL);
    CHNE(strstr(p, "2139062143"), NULL);
    CHNE(strstr(p, "84869120"), NULL);
    PASS();
}
TH_REG("rpi", 70, "__vsubss4 saturates high, not wraps", rpi70)

static void rpi71(void)
{
    const char *p = rpi_ptx(
        "__global__ void k(unsigned *o){\n"
        "  o[0] = __vsub4(0x00010203u, 0x03020100u);\n"
        "  o[1] = __vadd4(0xFFFFFFFFu, 0x01010101u); }\n", "rpi71");

    CHNE(p, NULL);
    CHNE(strstr(p, "-33619709"), NULL);
    CHNE(strstr(p, "mov.u32 %r2, 0;"), NULL);
    PASS();
}
TH_REG("rpi", 71, "__vsub4 wraps where __vsubss4 clamps", rpi71)

static void rpi72(void)
{
    const char *p = rpi_ptx(
        "__global__ void k(unsigned *o){\n"
        "  o[0] = __vsubus4(0x00010203u, 0x03020100u);\n"
        "  o[1] = __vaddus4(0xF0F0F0F0u, 0x20202020u); }\n", "rpi72");

    CHNE(p, NULL);
    CHNE(strstr(p, "mov.u32 %r1, 259;"), NULL);
    CHNE(strstr(p, "mov.u32 %r2, -1;"), NULL);
    PASS();
}
TH_REG("rpi", 72, "__vsubus4 stops at zero, not 0xff", rpi72)

static void rpi73(void)
{
    const char *p = rpi_ptx(
        "__global__ void k(unsigned *o){\n"
        "  o[0] = __vcmpne4(0x01020304u, 0x01FF0304u);\n"
        "  o[1] = __vcmpeq4(0x01020304u, 0x01FF0304u); }\n", "rpi73");

    CHNE(p, NULL);
    CHNE(strstr(p, "16711680"), NULL);
    CHNE(strstr(p, "-16711681"), NULL);
    PASS();
}
TH_REG("rpi", 73, "__vcmpne4 answers 0xff for a whole byte", rpi73)

static void rpi74(void)
{
    const char *p = rpi_ptx(
        "__global__ void k(int *o, float f){\n"
        "  o[0] = __float2int_rn(f);\n"
        "  o[1] = __float2int_rz(-2.7f); }\n", "rpi74");

    CHNE(p, NULL);
    CHNE(strstr(p, "cvt.rni.f32.f32"), NULL);
    CHNE(strstr(p, "cvt.rzi.s32.f32"), NULL);
    CHNE(strstr(p, "mov.u32 %r2, -2;"), NULL);
    PASS();
}
TH_REG("rpi", 74, "__float2int_rn rounds before it truncates", rpi74)

static void rpi75(void)
{
    const char *p = rpi_ptx(
        "__global__ void k(int *o, float *f){\n"
        "  o[0] = isinf(f[0]) ? 1 : 0; }\n", "rpi75");

    CHNE(p, NULL);
    CHNE(strstr(p, "abs.f32"), NULL);
    CHNE(strstr(p, "setp.gt.f32"), NULL);
    CHNE(strstr(p, "0f7F7FFFFF"), NULL);
    PASS();
}
TH_REG("rpi", 75, "isinf is an ordered test against FLT_MAX", rpi75)

static void rpi76(void)
{
    const char *p = rpi_ptx(
        "__global__ void k(unsigned *o){ __shared__ int s[4];\n"
        "  s[0] = 1;\n"
        "  o[0] = (unsigned) __cvta_generic_to_shared(&s[1]); }\n", "rpi76");

    CHNE(p, NULL);
    CHNE(strstr(p, "st.shared.u32"), NULL);
    CHNE(strstr(p, "add.u64 %rd5, %rd2, 4;"), NULL);
    CHNE(strstr(p, "cvt.u32.u64 %r2, %rd6;"), NULL);
    PASS();
}
TH_REG("rpi", 76, "__cvta_generic_to_shared keeps the offset", rpi76)

static void rpi77(void)
{
    const char *p = rpi_ptx(
        "__global__ void k(int *o){ __threadfence_system();\n"
        "  __nanosleep(100u); o[0] = 1; }\n", "rpi77");

    CHNE(p, NULL);
    CHNE(strstr(p, "membar.sys"), NULL);
    CHNE(strstr(p, "nanosleep.u32"), NULL);
    CHEQ(strstr(p, "bar.sync"), NULL);
    PASS();
}
TH_REG("rpi", 77, "a system fence is membar.sys, not a bar", rpi77)

static void rpi78(void)
{
    const char *p = rpi_ptx(
        "__global__ void k(float *o){ o[0] = erff(o[1]); }\n", "rpi78");

    CHNE(p, NULL);
    CHNE(strstr(p, "0f3EA7BA05"), NULL);
    CHNE(strstr(p, "ex2.approx.f32"), NULL);
    PASS();
}
TH_REG("rpi", 78, "erff carries its own series, not a zero", rpi78)

static void rpi79(void)
{
    int rc;

    CHNE(scratch("rpi79.cu",
        "struct P { int a; int b; };\n"
        "__device__ P mk(int x) { P p; p.a = x; p.b = x + 1; return p; }\n"
        "__global__ void k(int *o) { o[0] = mk(7).b; }\n"), NULL);
    rc = th_run(BC_BIN " --ir --no-mem2reg build/rpi79.cu",
                obuf, (int)sizeof obuf);
    CHEQ(rc, 0);
    CHEQ(strstr(obuf, "E111"), NULL);
    CHNE(strstr(obuf, ", 1  ; line 3"), NULL);
    PASS();
}
TH_REG("rpi", 79, "a returned struct can be read for a field", rpi79)

/* Compiles src to a SASS listing the same way rpi_ptx does for PTX. The
 * listing is the only text form of the back end, so it is what a regression
 * about encoding can read. */
static const char *rpi_sass(const char *src, const char *stem)
{
    static char txt[1 << 16];
    char cmd[512], cu[128], out[128];
    FILE *f, *g;
    size_t n;

    txt[0] = '\0';
    snprintf(cu, sizeof cu, "build/%s.cu", stem);
    snprintf(out, sizeof out, "build/%s.sass", stem);
    f = fopen(cu, "w");
    if (!f) return NULL;
    fputs(src, f);
    fclose(f);
    remove(out);
    snprintf(cmd, sizeof cmd, "%s --nvidia-sass %s -o %s", BC_BIN, cu, out);
    if (th_run(cmd, obuf, (int)sizeof obuf) != 0) return NULL;
    g = fopen(out, "rb");
    if (!g) return NULL;
    n = fread(txt, 1, sizeof txt - 1, g);
    fclose(g);
    txt[n] = '\0';
    return txt;
}

static void rpi80(void)
{
    const char *path = scratch("rpi80.cu",
        "__global__ void vadd(const float *a, const float *b, float *c,\n"
        "                     int n){ int i = blockIdx.x * blockDim.x\n"
        "  + threadIdx.x; if (i < n) c[i] = a[i] + b[i]; }\n");
    unsigned char hdr[64];
    char cmd[512];
    FILE *f;

    CHNE(path, NULL);
    remove("build/rpi80.cubin");
    snprintf(cmd, sizeof cmd, "%s --nvidia-cubin %s -o build/rpi80.cubin",
             BC_BIN, path);
    CHEQ(th_run(cmd, obuf, (int)sizeof obuf), 0);
    f = fopen("build/rpi80.cubin", "rb");
    CHNE(f, NULL);
    CHEQ(fread(hdr, 1, sizeof hdr, f), sizeof hdr);
    fclose(f);
    CHEQ(hdr[0], 0x7F);
    CHEQ(hdr[1], 'E');
    CHEQ(hdr[18] | (hdr[19] << 8), 190);
    CHEQ(hdr[48], 0x04);
    PASS();
}
TH_REG("rpi", 80, "a cubin Booth built is an ELF for sm_89", rpi80)

static void rpi81(void)
{
    const char *p = rpi_sass(
        "__global__ void k(const int *a, int *o, int n){\n"
        "  int i = threadIdx.x; if (i < n) o[i] = a[i] + 1; }\n", "rpi81");

    CHNE(p, NULL);
    CHNE(strstr(p, "7945 "), NULL);
    CHNE(strstr(p, "7941 "), NULL);
    PASS();
}
TH_REG("rpi", 81, "a divergent if gets its bssy and bsync", rpi81)

static void rpi82(void)
{
    const char *path = scratch("rpi82.cu",
        "__global__ void k(const unsigned *a, unsigned *o, int n){\n"
        "  int i = threadIdx.x; if (i < n) o[i] = a[i] / a[0]; }\n");
    char cmd[512];

    CHNE(path, NULL);
    snprintf(cmd, sizeof cmd, "%s --nvidia-cubin %s -o build/rpi82.cubin",
             BC_BIN, path);
    CHNE(th_run(cmd, obuf, (int)sizeof obuf), 0);
    CHNE(strstr(obuf, "E601"), NULL);
    CHNE(strstr(obuf, "NV_DIV_U32"), NULL);
    PASS();
}
TH_REG("rpi", 82, "an integer divide names its SASS refusal", rpi82)

static void rpi83(void)
{
    const char *path = scratch("rpi83.cu",
        "__global__ void k1(int *o){ o[0] = 1; }\n"
        "__global__ void k2(int *o){ o[1] = 2; }\n");
    char cmd[512];

    CHNE(path, NULL);
    snprintf(cmd, sizeof cmd, "%s --nvidia-cubin %s -o build/rpi83.cubin",
             BC_BIN, path);
    CHNE(th_run(cmd, obuf, (int)sizeof obuf), 0);
    CHNE(strstr(obuf, "E600"), NULL);
    PASS();
}
TH_REG("rpi", 83, "a cubin holds one kernel, not two", rpi83)

static void rpi84(void)
{
    const char *p = rpi_sass(
        "__global__ void k(const int *a, int *o){\n"
        "  __shared__ int s[64]; int t = threadIdx.x;\n"
        "  s[t] = a[t]; __syncthreads(); o[t] = s[63 - t]; }\n", "rpi84");

    CHNE(p, NULL);
    CHNE(strstr(p, "NV_LD_SHR_U32"), NULL);
    CHNE(strstr(p, "NV_ST_SHR_U32"), NULL);
    CHNE(strstr(p, "7984 "), NULL);
    PASS();
}
TH_REG("rpi", 84, "a shared load reaches LDS, not global", rpi84)

static void rpi85(void)
{
    const char *path = scratch("rpi85.cu",
        "__global__ void k(const float *a, float *o){\n"
        "  o[threadIdx.x] = sinf(a[threadIdx.x]); }\n");
    char cmd[512];

    CHNE(path, NULL);
    remove("build/rpi85.cubin");
    snprintf(cmd, sizeof cmd, "%s --nvidia-cubin %s -o build/rpi85.cubin",
             BC_BIN, path);
    CHNE(th_run(cmd, obuf, (int)sizeof obuf), 0);
    CHNE(strstr(obuf, "E601"), NULL);
    CHEQ(th_exist("build/rpi85.cubin"), 0);
    PASS();
}
TH_REG("rpi", 85, "a SASS refusal writes no file at all", rpi85)

static void rpi86(void)
{
    const char *p = rpi_sass(
        "__global__ void k(const int *a, int *o, int n){\n"
        "  int i = threadIdx.x; if (i < n)\n"
        "    o[i] = (int)(((long long)a[i] * 3 + 1) >> 5); }\n", "rpi86");
    int shf = 0;

    CHNE(p, NULL);
    for (const char *q = strstr(p, "7219 "); q; q = strstr(q + 1, "7219 "))
        shf++;
    CHEQ(shf, 2);
    PASS();
}
TH_REG("rpi", 86, "a 64-bit shift is not a 32-bit one", rpi86)

static void rpi87(void)
{
    const char *path = scratch("rpi87.cu",
        "__global__ void k(const double *a, double *o){\n"
        "  o[threadIdx.x] = a[threadIdx.x] + 1.0; }\n");
    char cmd[512];

    CHNE(path, NULL);
    snprintf(cmd, sizeof cmd, "%s --nvidia-cubin %s -o build/rpi87.cubin",
             BC_BIN, path);
    CHNE(th_run(cmd, obuf, (int)sizeof obuf), 0);
    CHNE(strstr(obuf, "E601"), NULL);
    CHNE(strstr(obuf, "NV_ADD_F64"), NULL);
    PASS();
}
TH_REG("rpi", 87, "an f64 add names its SASS refusal", rpi87)

static int nfind(const char *hay, const char *ndl)
{
    const char *p = hay;
    int n = 0;

    while ((p = strstr(p, ndl)) != NULL) { n++; p += strlen(ndl); }
    return n;
}

static void rpi88(void)
{
    const char *p = rpi_ptx(
        "struct pr2 { float x; float y; };\n"
        "template<typename D, typename S>\n"
        "__device__ D conv(S v) {\n"
        "  if constexpr (sizeof(D) == sizeof(S)) { return (D) 0; }\n"
        "  else { return float(v) + 31.0f; }\n"
        "}\n"
        "template<typename T>\n"
        "__device__ float outer(const pr2 * p, T * q) {\n"
        "  if constexpr (sizeof(T) == 4) { return q[0] + 41.0f; }\n"
        "  else { const pr2 t = p[0]; return conv<T>(t) + 43.0f; }\n"
        "}\n"
        "__global__ void k(const pr2 * p, float * o) {\n"
        "  o[0] = outer<float>(p, o); }\n", "rpi88");

    CHNE(p, NULL);
    CHNE(strstr(p, "0f42240000"), NULL);
    CHEQ(strstr(p, "0f422C0000"), NULL);
    CHEQ(strstr(p, "0f41F80000"), NULL);
    PASS();
}
TH_REG("rpi", 88, "a discarded arm is never instantiated", rpi88)

static void rpi89(void)
{
    const char *p = rpi_ptx(
        "template<typename T>\n"
        "__device__ float takex(T v) { return v.x + 53.0f; }\n"
        "template<typename T>\n"
        "__device__ float sel(T v, float f) {\n"
        "  if constexpr (sizeof(T) == 4) { return f + 57.0f; }\n"
        "  else { return takex(v) + 59.0f; }\n"
        "}\n"
        "__global__ void k(float * o) { o[0] = sel<float>(o[1], o[2]); }\n",
        "rpi89");

    CHNE(p, NULL);
    CHNE(strstr(p, "0f42640000"), NULL);
    CHEQ(strstr(p, "0f426C0000"), NULL);
    CHEQ(strstr(p, "0f42540000"), NULL);
    PASS();
}
TH_REG("rpi", 89, "a discarded arm hides its missing member", rpi89)

static void rpi90(void)
{
    const char *p = rpi_ptx(
        "struct pr2 { float x; float y; };\n"
        "template<typename D, typename S>\n"
        "__device__ D conv(S v) {\n"
        "  if constexpr (sizeof(D) == sizeof(S)) { return (D) 0; }\n"
        "  else { return float(v) + 31.0f; }\n"
        "}\n"
        "template<typename T>\n"
        "__device__ float sel2(T v, float f) {\n"
        "  if constexpr (sizeof(T) == 4) { return conv<float>(v) + 73.0f; }\n"
        "  else { return v.x + 79.0f; }\n"
        "}\n"
        "__global__ void k(float * o, pr2 p) { o[0] = sel2<pr2>(p, o[1]); }\n",
        "rpi90");

    CHNE(p, NULL);
    CHNE(strstr(p, "0f429E0000"), NULL);
    CHEQ(strstr(p, "0f42920000"), NULL);
    CHEQ(strstr(p, "0f41F80000"), NULL);
    PASS();
}
TH_REG("rpi", 90, "a discarded then arm goes the same way", rpi90)

static void rpi91(void)
{
    const char *p = rpi_ptx(
        "template<typename T>\n"
        "__device__ float takex(T v) { return v.x + 53.0f; }\n"
        "template<int N>\n"
        "__device__ float deep(float v) {\n"
        "  if constexpr (N == 0) { return v + 61.0f; }\n"
        "  else {\n"
        "    if constexpr (N == 1) { return takex(v) + 67.0f; }\n"
        "    else { return takex(v) + 71.0f; }\n"
        "  }\n"
        "}\n"
        "__global__ void k(float * o) { o[0] = deep<0>(o[1]); }\n", "rpi91");

    CHNE(p, NULL);
    CHNE(strstr(p, "0f42740000"), NULL);
    CHEQ(strstr(p, "0f42860000"), NULL);
    CHEQ(strstr(p, "0f428E0000"), NULL);
    PASS();
}
TH_REG("rpi", 91, "an arm nested in a discarded one is gone", rpi91)

static void rpi92(void)
{
    const char *p = rpi_ptx(
        "struct pr2 { float x; float y; };\n"
        "__global__ void k(float * o) {\n"
        "  pr2 s[4] = {{1.5f, 2.5f}};\n"
        "  o[0] = s[3].x; o[1] = s[3].y; o[2] = s[0].x; }\n", "rpi92");

    CHNE(p, NULL);
    CHNE(strstr(p, "0f3FC00000"), NULL);
    CHNE(strstr(p, "0f40200000"), NULL);
    CHEQ(nfind(p, "0f00000000"), 6);
    PASS();
}
TH_REG("rpi", 92, "an array of structs zeroes its tail", rpi92)

static void rpi93(void)
{
    const char *p = rpi_ptx(
        "struct pr2 { float x; float y; };\n"
        "struct box { pr2 p; float n; };\n"
        "__global__ void k(float * o) {\n"
        "  box b[3] = {{{1.5f, 2.5f}, 7.0f}};\n"
        "  pr2 e[2] = {};\n"
        "  pr2 m[3] = {{1.0f, 2.0f}, {3.0f, 4.0f}};\n"
        "  o[0] = b[2].p.x + e[1].y + m[2].x;\n"
        "  o[1] = b[2].n; }\n", "rpi93");

    CHNE(p, NULL);
    CHNE(strstr(p, "0f3FC00000"), NULL);
    CHNE(strstr(p, "0f40E00000"), NULL);
    CHEQ(nfind(p, "0f00000000"), 12);
    PASS();
}
TH_REG("rpi", 93, "nested braces reach every field", rpi93)

static void rpi94(void)
{
    const char *p = rpi_ptx(
        "struct pr2 { float x; float y; };\n"
        "template<int N>\n"
        "__device__ float last(void) {\n"
        "  pr2 s[N] = {{5.0f, 6.0f}};\n"
        "  return s[N - 1].x + s[N - 1].y;\n"
        "}\n"
        "__global__ void k(float * o) {\n"
        "  o[0] = last<1>(); o[1] = last<4>(); }\n", "rpi94");

    CHNE(p, NULL);
    CHEQ(nfind(p, "0f40A00000"), 2);
    CHEQ(nfind(p, "0f40C00000"), 2);
    CHEQ(nfind(p, "0f00000000"), 6);
    PASS();
}
TH_REG("rpi", 94, "a template extent zeroes its own tail", rpi94)

static void rpi95(void)
{
    const char *path = scratch("rpi95.cu",
        "struct pr2 { float x; float y; };\n"
        "__global__ void k(float * o) {\n"
        "  pr2 s[2] = {{1.0f, 2.0f}, {3.0f, 4.0f}, {5.0f, 6.0f}};\n"
        "  o[0] = s[0].x; }\n");
    char cmd[512];

    CHNE(path, NULL);
    snprintf(cmd, sizeof cmd, "%s --nvidia-ptx %s -o build/rpi95.ptx",
             BC_BIN, path);
    CHNE(th_run(cmd, obuf, (int)sizeof obuf), 0);
    CHNE(strstr(obuf, "E660"), NULL);
    PASS();
}
TH_REG("rpi", 95, "too many initialisers refuses", rpi95)

static void rpi96(void)
{
    const char *p = rpi_ptx(
        "struct vw { float4 v; float n; };\n"
        "__global__ void k(float * o) {\n"
        "  vw a[3] = {{{1.0f, 2.0f, 3.0f, 4.0f}, 5.0f}};\n"
        "  o[0] = a[2].v.x + a[2].v.w;\n"
        "  o[1] = a[2].n; }\n", "rpi96");

    CHNE(p, NULL);
    CHNE(strstr(p, "0f3F800000"), NULL);
    CHNE(strstr(p, "0f40A00000"), NULL);
    CHEQ(nfind(p, "0f00000000"), 10);
    PASS();
}
TH_REG("rpi", 96, "a vector member is filled and zeroed", rpi96)

static void rpi97(void)
{
    const char *path = scratch("rpi97.cu",
        "__global__ void k(int * o) { int a[9000] = {1}; o[0] = a[8999]; }\n");
    char cmd[512];

    CHNE(path, NULL);
    snprintf(cmd, sizeof cmd, "%s --nvidia-ptx %s -o build/rpi97.ptx",
             BC_BIN, path);
    CHNE(th_run(cmd, obuf, (int)sizeof obuf), 0);
    CHNE(strstr(obuf, "E661"), NULL);
    PASS();
}
TH_REG("rpi", 97, "a tail too long to zero is named", rpi97)

static void rpi98(void)
{
    const char *p = rpi_ptx(
        "template <int N> __device__ void f(int *d) { d[0] = N; }\n"
        "__global__ void ka(int *o) { int w[8];   f<sizeof(w)>(o); }\n"
        "__global__ void kb(int *o) { float w[5]; f<sizeof(w)>(o); }\n",
        "rpi98");

    CHNE(p, NULL);
    CHNE(strstr(p, "mov.u32 %r1, 32;"), NULL);
    CHNE(strstr(p, "mov.u32 %r1, 20;"), NULL);
    PASS();
}
TH_REG("rpi", 98, "sizeof of a local array is its extent", rpi98)

static void rpi99(void)
{
    const char *p = rpi_ptx(
        "template <int N> __device__ void f(int *d) { d[0] = N; }\n"
        "__device__ int g8[8];\n"
        "__device__ int g3[3];\n"
        "__global__ void ka(int *o) { f<sizeof(g8)>(o); }\n"
        "__global__ void kb(int *o) { f<sizeof(g3)>(o); }\n",
        "rpi99");

    CHNE(p, NULL);
    CHNE(strstr(p, "mov.u32 %r1, 32;"), NULL);
    CHNE(strstr(p, "mov.u32 %r1, 12;"), NULL);
    PASS();
}
TH_REG("rpi", 99, "a device array sizes as a template arg", rpi99)

static void rpi100(void)
{
    const char *p = rpi_ptx(
        "static constexpr int Q = 4*32;\n"
        "struct A { char qs[Q]; };\n"
        "__global__ void k(int *o)"
        " { __shared__ A a; a.qs[0] = 1; o[0] = a.qs[0]; }\n",
        "rpi100");

    CHNE(p, NULL);
    CHNE(strstr(p, "shmem[128]"), NULL);
    PASS();
}
TH_REG("rpi", 100, "a field bound reads a file-scope constexpr", rpi100)

static void rpi101(void)
{
    const char *p = rpi_ptx(
        "static constexpr int AL = 16;\n"
        "struct alignas(AL) T { int a; };\n"
        "__global__ void k(int *o)"
        " { __shared__ T t; t.a = 1; o[0] = t.a; }\n",
        "rpi101");

    CHNE(p, NULL);
    CHNE(strstr(p, ".shared .align 16"), NULL);
    PASS();
}
TH_REG("rpi", 101, "alignas reads a file-scope constexpr", rpi101)

static void rpi102(void)
{
    CHNE(scratch("rpi102.cu",
        "typedef void (*fp)(int *);\n"
        "template <int J> __device__ void ld(int *p) { p[0] = J; }\n"
        "struct pk { int v; fp f; };\n"
        "template <int J> static constexpr __device__ pk mk()"
        " { return pk{J*10, ld<J>}; }\n"
        "__global__ void k(int *o) { constexpr pk a = mk<3>();\n"
        "  constexpr pk b = mk<7>(); o[0] = a.v; o[1] = b.v; }\n"), NULL);
    CHEQ(th_run(BC_BIN " --ir build/rpi102.cu", obuf, (int)sizeof obuf), 0);
    CHNE(strstr(obuf, "fnref @ld  "), NULL);
    CHNE(strstr(obuf, "fnref @ld$1"), NULL);
    CHNE(strstr(obuf, "store i32 3, "), NULL);
    CHNE(strstr(obuf, "store i32 7, "), NULL);
    PASS();
}
TH_REG("rpi", 102, "a template-id as a value gets its body", rpi102)

static void rpi103(void)
{
    CHNE(scratch("rpi103.cu",
        "template <int N> __device__ void f(int *d) { d[0] = N; }\n"
        "__global__ void k(int *o) { int x = o[1]; f<x>(o); }\n"), NULL);
    CHNE(th_run(BC_BIN " --ir build/rpi103.cu", obuf, (int)sizeof obuf), 0);
    CHNE(strstr(obuf, "E720"), NULL);
    CHNE(strstr(obuf, "'x' is not a constant"), NULL);
    CHEQ(strstr(obuf, "E105"), NULL);
    PASS();
}
TH_REG("rpi", 103, "an unfoldable template argument is named", rpi103)

static void rpi104(void)
{
    CHNE(scratch("rpi104.cu",
        "__global__ void k(int *o)"
        " { int n = o[0]; int a[n]; a[0] = 1; o[1] = a[0]; }\n"), NULL);
    CHNE(th_run(BC_BIN " --ir build/rpi104.cu", obuf, (int)sizeof obuf), 0);
    CHNE(strstr(obuf, "E131"), NULL);
    CHNE(strstr(obuf, "'a[n]'"), NULL);
    PASS();
}
TH_REG("rpi", 104, "an unfoldable array bound names the bound", rpi104)

static void rpi105(void)
{
    const char *p = rpi_ptx(
        "namespace nsx { template <int I_, int J_> struct tile {\n"
        "  static constexpr int I = I_; static constexpr int ne = I_*J_/32; }; }\n"
        "typedef nsx::tile<16,8> tc;\n"
        "__global__ void k(int *o) { __shared__ int a[tc::I][tc::ne];\n"
        "  a[0][0] = 1; o[0] = a[0][0]; }\n",
        "rpi105");

    CHNE(p, NULL);
    CHNE(strstr(p, "shmem[256]"), NULL);
    PASS();
}
TH_REG("rpi", 105, "a qualified template typedef finds ::I", rpi105)

static void rpi106(void)
{
    static const unsigned char want[] = { 0x02u, 0x10u };
    unsigned char buf[8192];
    size_t n, i;
    int hit = 0;
    FILE *f;

    CHEQ(th_run(BC_BIN " --nvidia-cubin examples/cmake/vadd.cu "
                "-o build/rpi106.cubin", obuf, (int)sizeof obuf), 0);
    f = fopen("build/rpi106.cubin", "rb");
    CHNE(f, NULL);
    if (!f) return;
    n = fread(buf, 1, sizeof buf, f);
    fclose(f);
    for (i = 0; i + 1 < n; i++)
        if (buf[i] == want[0] && buf[i + 1] == want[1]) hit++;
    CHNE(hit, 0);
    PASS();
}
TH_REG("rpi", 106, "a kernel symbol is LOCAL and an entry", rpi106)

/* A __device__ function returning a struct declared one type at the call and
 * handed back a pointer to its own private slot, and nothing checked the two
 * agreed. The caller stored eight bytes of pointer into the struct-sized hole
 * and read fields out of it, so mk(7).b came back as the top half of an
 * address. It assembled, it ran, and the number was rubbish.
 *
 * The convention is now sret: the caller owns the slot, passes it as a trailing
 * pointer parameter, and the callee returns void. rpi107 to rpi116 walk the
 * shapes a struct can take. Each one checks the value and, because the old bug
 * was a 64-bit store into a struct slot, that no st.local.u64 survives. */
static void rpi107(void)
{
    const char *p = rpi_ptx(
        "struct P { int a; int b; };\n"
        "__device__ P mk(int x) { P p; p.a = x; p.b = x + 1; return p; }\n"
        "__global__ void k(int *o){ P p = mk(1000);"
        " o[0] = p.a; o[1] = p.b; }\n", "rpi107");

    CHNE(p, NULL);
    CHEQ(strstr(p, "st.local.u64"), NULL);
    CHNE(strstr(p, ", 1000;"), NULL);
    CHNE(strstr(p, ", 1001;"), NULL);
    PASS();
}
TH_REG("rpi", 107, "a returned struct carries both its fields", rpi107)

static void rpi108(void)
{
    int rc;

    CHNE(scratch("rpi108.cu",
        "struct P { int a; int b; };\n"
        "__device__ P mk(int x) { P p; p.a = x; p.b = x + 1; return p; }\n"
        "__global__ void k(int *o){ P p = mk(7); o[0] = p.b; }\n"), NULL);
    rc = th_run(BC_BIN " --ir build/rpi108.cu", obuf, (int)sizeof obuf);
    CHEQ(rc, 0);
    CHNE(strstr(obuf, "func @mk(i32 %0, ptr<private,"), NULL);
    CHNE(strstr(obuf, "ret void"), NULL);
    CHNE(strstr(obuf, "call void @mk("), NULL);
    CHEQ(strstr(obuf, "call {i32, i32}"), NULL);
    PASS();
}
TH_REG("rpi", 108, "a struct return takes a caller slot", rpi108)

static void rpi109(void)
{
    const char *p = rpi_ptx(
        "struct I { int x; int y; };\n"
        "struct O { I i; int z; };\n"
        "__device__ O mk(int n){ O o; o.i.x = n; o.i.y = n + 1;"
        " o.z = n + 2; return o; }\n"
        "__global__ void k(int *o){ O v = mk(1000);"
        " o[0] = v.i.y; o[1] = v.z; }\n", "rpi109");

    CHNE(p, NULL);
    CHEQ(strstr(p, "st.local.u64"), NULL);
    CHNE(strstr(p, ", 1001;"), NULL);
    CHNE(strstr(p, ", 1002;"), NULL);
    PASS();
}
TH_REG("rpi", 109, "a struct inside a struct comes back whole", rpi109)

static void rpi110(void)
{
    const char *p = rpi_ptx(
        "struct M { float2 f; int n; };\n"
        "__device__ M mk(int n){ M m; m.f.x = 1.0f; m.f.y = 2.0f;"
        " m.n = n + 3; return m; }\n"
        "__global__ void k(int *o, float *f){ M m = mk(1000);"
        " f[0] = m.f.y; o[0] = m.n; }\n", "rpi110");

    CHNE(p, NULL);
    CHEQ(strstr(p, "st.local.u64"), NULL);
    CHNE(strstr(p, "0f40000000"), NULL);
    CHNE(strstr(p, ", 1003;"), NULL);
    PASS();
}
TH_REG("rpi", 110, "a returned struct keeps its float2 member", rpi110)

static void rpi111(void)
{
    const char *p = rpi_ptx(
        "struct W { int v[24]; };\n"
        "__device__ W mk(int n){ W w;"
        " for (int i = 0; i < 24; i = i + 1) w.v[i] = n + i; return w; }\n"
        "__global__ void k(int *o){ W w = mk(1000); o[0] = w.v[23]; }\n",
        "rpi111");

    CHNE(p, NULL);
    CHEQ(strstr(p, "st.local.u64"), NULL);
    CHNE(strstr(p, "__local[192]"), NULL);
    PASS();
}
TH_REG("rpi", 111, "a struct past any register still returns", rpi111)

static void rpi112(void)
{
    const char *p = rpi_ptx(
        "struct P { int a; int b; };\n"
        "__device__ P mk(int x){ P p; p.a = x; p.b = x + 1; return p; }\n"
        "__device__ P fwd(int x){ return mk(x + 1); }\n"
        "__global__ void k(int *o){ o[0] = fwd(1000).b; }\n", "rpi112");

    CHNE(p, NULL);
    CHEQ(strstr(p, "st.local.u64"), NULL);
    CHNE(strstr(p, ", 1002;"), NULL);
    PASS();
}
TH_REG("rpi", 112, "two calls deep the value still lands", rpi112)

static void rpi113(void)
{
    const char *p = rpi_ptx(
        "struct E { };\n"
        "__device__ E mk(void){ E e; return e; }\n"
        "__global__ void k(int *o){ E e = mk(); o[0] = 1234; }\n", "rpi113");

    CHNE(p, NULL);
    CHEQ(strstr(p, "st.local.u64"), NULL);
    CHNE(strstr(p, ", 1234;"), NULL);
    PASS();
}
TH_REG("rpi", 113, "an empty struct return compiles clean", rpi113)

static void rpi114(void)
{
    const char *p = rpi_ptx(
        "struct P { int a; int b; };\n"
        "__device__ int bump(P p){ p.a = p.a + 1; return p.a; }\n"
        "__global__ void k(int *o){ P p; p.a = 1000; p.b = 1;"
        " o[0] = bump(p); o[1] = p.a; }\n", "rpi114");

    CHNE(p, NULL);
    CHEQ(strstr(p, "st.local.u64"), NULL);
    CHNE(strstr(p, ", 1001;"), NULL);
    CHNE(strstr(p, ", 1000;"), NULL);
    PASS();
}
TH_REG("rpi", 114, "a struct argument is the callee's copy", rpi114)

static void rpi115(void)
{
    const char *p = rpi_ptx(
        "union U { int i; float f; };\n"
        "struct S { U u; int t; };\n"
        "__device__ S mk(int n){ S s; s.u.i = n; s.t = n + 1; return s; }\n"
        "__global__ void k(int *o){ S s = mk(1000);"
        " o[0] = s.u.i; o[1] = s.t; }\n", "rpi115");

    CHNE(p, NULL);
    CHEQ(strstr(p, "st.local.u64"), NULL);
    CHNE(strstr(p, ", 1000;"), NULL);
    CHNE(strstr(p, ", 1001;"), NULL);
    PASS();
}
TH_REG("rpi", 115, "a union in a returned struct keeps bytes", rpi115)

/* Overloaded function templates are matched by name alone, so a call inside a
 * class template takes the float overload and hands its f32 result back where
 * a float2 was declared. That used to be a silent four-byte store into an
 * eight-byte slot; the sret path now names it rather than lowering it. When
 * template overload resolution learns to read the argument types this becomes
 * a value test. */
static void rpi116(void)
{
    const char *p = rpi_ptx(
        "template<int w = 32> static __device__ float f(float x)"
        " { return x + 3333.0f; }\n"
        "template<int w = 32> static __device__ float2 f(float2 a)"
        " { float2 r; r.x = a.x + 7777.0f; r.y = a.y; return r; }\n"
        "template <typename T> struct pol"
        " { static __device__ T red(T v){ return f(v); } };\n"
        "__global__ void k(float *o){ float2 v; v.x = o[0]; v.y = o[1];"
        " float2 r = pol<float2>::red(v); o[0] = r.x + r.y; }\n", "rpi116");

    CHNE(p, NULL);
    CHNE(strstr(p, "0f45F30800"), NULL);
    CHEQ(strstr(p, "0f45505000"), NULL);
    PASS();
}
TH_REG("rpi", 116, "an overload is picked by argument type", rpi116)

static void rpi117(void)
{
    const char *p = rpi_ptx(
        "struct lp {\n"
        "    dim3 bn;\n"
        "    dim3 bd;\n"
        "    unsigned int sh;\n"
        "    lp(const dim3 &b, const dim3 &d, unsigned int s)\n"
        "        : bn(b), bd(d), sh(s) {}\n"
        "};\n"
        "template<typename K, typename... A>\n"
        "static void klaun(K kern, const lp &p, A&&... a) {\n"
        "    kern<<<p.bn, p.bd, p.sh>>>(std::forward<A>(a)...);\n"
        "}\n"
        "template<int C>\n"
        "__global__ void kk(int *o, int a0, int a1, int a2, int a3, int a4, int a5, int a6, int a7, int a8, int a9, int a10, int a11, int a12, int a13, int a14, int a15, int a16, int a17, int a18, int a19, int a20, int a21, int a22, int a23, int a24, int a25, int a26, int a27, int a28, int a29, int a30, int a31, int a32)\n"
        "    { o[0] = a0 + a5 + C; }\n"
        "void go(int *o, int i) {\n"
        "    int a0 = i + 0;\n"
        "    int a1 = i + 1;\n"
        "    int a2 = i + 2;\n"
        "    int a3 = i + 3;\n"
        "    int a4 = i + 4;\n"
        "    int a5 = i + 5;\n"
        "    int a6 = i + 6;\n"
        "    int a7 = i + 7;\n"
        "    int a8 = i + 8;\n"
        "    int a9 = i + 9;\n"
        "    int a10 = i + 10;\n"
        "    int a11 = i + 11;\n"
        "    int a12 = i + 12;\n"
        "    int a13 = i + 13;\n"
        "    int a14 = i + 14;\n"
        "    int a15 = i + 15;\n"
        "    int a16 = i + 16;\n"
        "    int a17 = i + 17;\n"
        "    int a18 = i + 18;\n"
        "    int a19 = i + 19;\n"
        "    int a20 = i + 20;\n"
        "    int a21 = i + 21;\n"
        "    int a22 = i + 22;\n"
        "    int a23 = i + 23;\n"
        "    int a24 = i + 24;\n"
        "    int a25 = i + 25;\n"
        "    int a26 = i + 26;\n"
        "    int a27 = i + 27;\n"
        "    int a28 = i + 28;\n"
        "    int a29 = i + 29;\n"
        "    int a30 = i + 30;\n"
        "    int a31 = i + 31;\n"
        "    int a32 = i + 32;\n"
        "    lp p = lp(dim3(1,1,1), dim3(32,1,1), 0u);\n"
        "    klaun(kk<77770>, p, o, a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15, a16, a17, a18, a19, a20, a21, a22, a23, a24, a25, a26, a27, a28, a29, a30, a31, a32);\n"
        "    klaun(kk<11117>, p, o, a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15, a16, a17, a18, a19, a20, a21, a22, a23, a24, a25, a26, a27, a28, a29, a30, a31, a32);\n"
        "}\n", "rpi117");

    CHNE(p, NULL);
    CHNE(strstr(p, "77770"), NULL);
    CHNE(strstr(p, "11117"), NULL);
    PASS();
}
TH_REG("rpi", 117, "a long forwarded pack reaches the launch", rpi117)

static void rpi118(void)
{
    const char *p = rpi_ptx(
        "enum qt { QA = 0, QB = 1 };\n"
        "__device__ int dota(const int *v, int i) { return v[i] + 77770; }\n"
        "__device__ int dotb(const int *v, int i) { return v[i] + 11117; }\n"
        "typedef int (*dot_t)(const int *, int);\n"
        "static constexpr __device__ dot_t getdot(qt t) {\n"
        "    switch (t) {\n"
        "    case QA: return dota;\n"
        "    case QB: return dotb;\n"
        "    default: return nullptr;\n"
        "    }\n"
        "}\n"
        "template <qt t>\n"
        "__global__ void k(int *d, const int *v) {\n"
        "    constexpr dot_t dot = getdot(t);\n"
        "    d[threadIdx.x] = dot(v, threadIdx.x);\n"
        "}\n"
        "void g(int *d, const int *v) { k<QB><<<1,32>>>(d, v); }\n", "rpi118");

    CHNE(p, NULL);
    CHNE(strstr(p, "11117"), NULL);
    CHEQ(strstr(p, "77770"), NULL);
    PASS();
}
TH_REG("rpi", 118, "a constexpr pointer picks the callee", rpi118)

static void rpi119(void)
{
    CHNE(scratch("rpi119.cu",
        "enum tb { T_GEN = 0, T_ALT = 1 };\n"
        "static constexpr __host__ __device__ tb gettb() { return T_GEN; }\n"
        "static constexpr __host__ __device__ int rows(int nc, int t,\n"
        "                                              bool sk = false, int nw = 5) {\n"
        "    if (t == T_GEN) {\n"
        "        switch (nc) {\n"
        "        case 1: return sk ? nw : 1;\n"
        "        case 2: return 3;\n"
        "        default: return 1;\n"
        "        }\n"
        "    }\n"
        "    return 1;\n"
        "}\n"
        "template <int nc>\n"
        "__global__ void k(float *d) {\n"
        "    constexpr tb t = gettb();\n"
        "    constexpr int r = rows(nc, t);\n"
        "    float tmp[r];\n"
        "    for (int i = 0; i < r; ++i) tmp[i] = (float) i;\n"
        "    d[0] = tmp[r - 1];\n"
        "}\n"
        "void g(float *d) { k<2><<<1,32>>>(d); }\n"), NULL);
    CHEQ(th_run(BC_BIN " --ir build/rpi119.cu", obuf, (int)sizeof obuf), 0);
    CHEQ(strstr(obuf, "E131"), NULL);
    CHNE(strstr(obuf, "[3 x f32]"), NULL);
    PASS();
}
TH_REG("rpi", 119, "a constexpr switch sizes the array", rpi119)

static void rpi120(void)
{
    const char *p = rpi_ptx(
        "template<typename... A>\n"
        "__host__ __device__ constexpr void unus(A&&...) {}\n"
        "template <int nc>\n"
        "__global__ void k(float *d, int n) {\n"
        "    float s[nc];\n"
        "    for (int j = 0; j < nc; ++j) s[j] = (float) j;\n"
        "    if ((int) threadIdx.x < n) d[0] = s[0] + 4444.0f;\n"
        "    unus(s, n);\n"
        "}\n"
        "void g(float *d, int n) { k<4><<<1,32>>>(d, n); }\n", "rpi120");

    CHNE(p, NULL);
    CHNE(strstr(p, "0f458AE000"), NULL);
    PASS();
}
TH_REG("rpi", 120, "an array binds to a variadic pack", rpi120)

static void rpi121(void)
{
    CHNE(scratch("rpi121.cu",
        "struct buf { int bt; };\n"
        "struct tns { struct tns *v; struct buf *buffer; struct tns *src[4]; };\n"
        "__global__ void k(int *o, struct tns *t, int j) {\n"
        "    o[0] = t->v->buffer->bt;\n"
        "    o[1] = t->src[j]->buffer->bt;\n"
        "}\n"), NULL);
    CHEQ(th_run(BC_BIN " --ir build/rpi121.cu", obuf, (int)sizeof obuf), 0);
    CHEQ(strstr(obuf, "E070"), NULL);
    CHNE(strstr(obuf, "[4 x ptr<global, type_3>]"), NULL);
    PASS();
}
TH_REG("rpi", 121, "a self-referential member stays a pointer", rpi121)

static void rpi122(void)
{
    CHNE(scratch("rpi122.cu",
        "struct pool { int lim; };\n"
        "struct ctx { int dev; pool * mk(int d, int s); };\n"
        "pool * ctx::mk(int d, int s) { return (pool *) (long) (dev + d + s); }\n"
        "struct vec { int d[4]; __device__ int at(int i) { return d[i]; } };\n"
        "__global__ void k(int *o, vec *v) { o[0] = v->at(1) + 5150; }\n"), NULL);
    CHEQ(th_run(BC_BIN " --ir build/rpi122.cu", obuf, (int)sizeof obuf), 0);
    CHEQ(strstr(obuf, "E073"), NULL);
    CHNE(strstr(obuf, "5150"), NULL);
    PASS();
}
TH_REG("rpi", 122, "an out-of-class method keeps calls valid", rpi122)

static void rpi123(void)
{
    const char *p = rpi_ptx(
        "__device__ int mix(int a, int b, int c, int d, int e, int f,\n"
        "                   int g, int h, int i, int j, int k, int l)\n"
        "{\n"
        "    return a + b*2 + c*3 + d*4 + e*5 + f*6\n"
        "         + g*7 + h*8 + i*9 + j*10 + k*11 + l*1000;\n"
        "}\n"
        "__global__ void ker(int *o)\n"
        "{\n"
        "    o[0] = mix(1,1,1,1,1,1,1,1,1,1,1,7) + 100;\n"
        "}\n", "rpi123");

    CHNE(p, NULL);
    CHNE(strstr(p, "7166"), NULL);
    PASS();
}
TH_REG("rpi", 123, "a device call keeps its last argument", rpi123)

static void rpi124(void)
{
    const char *p = rpi_ptx(
        "__device__ int pick(int a, int b, int c, int d, int e, int f, int g)\n"
        "{\n"
        "    int t = a + b + c + d + e + f;\n"
        "    if (g > 0) { t = t * 10; } else { t = t + 5; }\n"
        "    return t;\n"
        "}\n"
        "__global__ void ker(int *o, int n)\n"
        "{\n"
        "    o[0] = pick(1, 2, 3, 4, 5, 7, n);\n"
        "    o[1] = pick(1000, 2000, 3000, 4000, 5000, 6000, n);\n"
        "}\n", "rpi124");

    CHNE(p, NULL);
    CHNE(strstr(p, "220"), NULL);
    CHNE(strstr(p, "210000"), NULL);
    CHNE(strstr(p, "21005"), NULL);
    PASS();
}
TH_REG("rpi", 124, "one callee splices twice without aliasing", rpi124)

static void rpi125(void)
{
    const char *p = rpi_ptx(
        "__device__ int lo(int x) { return x * 3; }\n"
        "__device__ int mid(int x)\n"
        "{ int r = lo(x); if (r > 100) r -= 1; return r + 7; }\n"
        "__device__ int hi(int x)\n"
        "{ int r = mid(x); if (r < 0) r = 0; return r + 11; }\n"
        "__global__ void ker(int *o) { o[0] = hi(5) + 9000; }\n", "rpi125");

    CHNE(p, NULL);
    CHEQ(strstr(p, "call"), NULL);
    CHNE(strstr(p, "9000"), NULL);
    PASS();
}
TH_REG("rpi", 125, "a nested device chain leaves no call", rpi125)

static void rpi126(void)
{
    CHNE(scratch("rpi126.cu",
        "__device__ int rec(int n, int mode)\n"
        "{\n"
        "    if (mode == 0) { return 4242; }\n"
        "    if (n <= 0) { return rec(0, 0); }\n"
        "    return rec(n - 1, mode) + 1;\n"
        "}\n"
        "__global__ void ker(int *o) { o[0] = rec(3, 1); }\n"), NULL);
    remove("build/rpi126.ptx");
    CHNE(th_run(BC_BIN " --nvidia-ptx build/rpi126.cu -o build/rpi126.ptx",
                obuf, (int)sizeof obuf), 0);
    CHNE(strstr(obuf, "E864"), NULL);
    CHNE(strstr(obuf, "names a predecessor that does not branch there"), NULL);
    CHEQ(strstr(obuf, "E541"), NULL);
    CHEQ(th_exist("build/rpi126.ptx"), 0);
    PASS();
}
TH_REG("rpi", 126, "a recursive device call refuses, not lies", rpi126)

static void rpi127(void)
{
    const char *p = rpi_ptx(
        "extern \"C\" {\n"
        "__device__ int addc(int a) { return a + 100; }\n"
        "struct pair { int lo; int hi; };\n"
        "}\n"
        "namespace nm {\n"
        "extern \"C\" {\n"
        "__device__ int mulc(int a) { return a * 7; }\n"
        "}\n"
        "}\n"
        "extern \"C\" __global__ void k(int *o) {\n"
        "    struct pair q;\n"
        "    q.lo = 3;\n"
        "    q.hi = 4;\n"
        "    o[0] = addc(q.lo) + nm::mulc(q.hi);\n"
        "}\n", "rpi127");

    CHNE(p, NULL);
    CHNE(strstr(p, ".entry k"), NULL);
    CHNE(strstr(p, "131"), NULL);
    PASS();
}
TH_REG("rpi", 127, "a linkage block declares at file scope", rpi127)

static void rpi128(void)
{
    CHNE(scratch("rpi128.cu",
        "extern \"C\" int hostf(int a);\n"
        "extern \"C\" {\n"
        "extern \"C++\" {\n"
        "__device__ int deep(int a) { return a + 9; }\n"
        "}\n"
        "}\n"
        "__global__ void k(int *o) { o[0] = deep(4113); }\n"), NULL);
    CHEQ(th_run(BC_BIN " --ir build/rpi128.cu", obuf, (int)sizeof obuf), 0);
    CHEQ(strstr(obuf, "E0"), NULL);
    CHNE(strstr(obuf, "4113"), NULL);
    PASS();
}
TH_REG("rpi", 128, "nested linkage blocks stay transparent", rpi128)

static void rpi129(void)
{
    CHNE(scratch("rpi129.cu",
        "extern \"Fortran\" { int q; }\n"
        "__global__ void k(int *o) { o[0] = q; }\n"), NULL);
    CHNE(th_run(BC_BIN " --ir build/rpi129.cu", obuf, (int)sizeof obuf), 0);
    CHNE(strstr(obuf, "E800"), NULL);
    CHNE(strstr(obuf, "Fortran"), NULL);
    PASS();
}
TH_REG("rpi", 129, "an unknown linkage is refused by name", rpi129)

static void rpi130(void)
{
    CHNE(scratch("rpi130.cu",
        "struct tns { int id; struct buf *b; struct tns *src[4]; };\n"
        "struct buft { int kind; int mag; };\n"
        "typedef struct buft * buft_t;\n"
        "struct buf { int pad; buft_t bt; };\n"
        "__global__ void k(int *o, struct tns *t, int i) {\n"
        "    o[0] = t->src[i]->b->bt->mag;\n"
        "}\n"), NULL);
    CHEQ(th_run(BC_BIN " --sema build/rpi130.cu", obuf, (int)sizeof obuf), 0);
    CHEQ(strstr(obuf, "E070"), NULL);
    CHEQ(strstr(obuf, "E802"), NULL);
    CHNE(strstr(obuf, "member @ 6:21 \xe2\x86\x92 struct buf*"), NULL);
    CHNE(strstr(obuf, "member @ 6:24 \xe2\x86\x92 struct buft*"), NULL);
    PASS();
}
TH_REG("rpi", 130, "a tag used before its definition binds", rpi130)

static void rpi131(void)
{
    CHNE(scratch("rpi131.cu",
        "struct nodeb;\n"
        "struct nodea { int va; struct nodeb *b; };\n"
        "struct nodeb { int vb; struct nodea *a; };\n"
        "__global__ void k(int *o, struct nodea *p) { o[0] = p->b->a->va; }\n"),
        NULL);
    CHEQ(th_run(BC_BIN " --sema build/rpi131.cu", obuf, (int)sizeof obuf), 0);
    CHEQ(strstr(obuf, "E070"), NULL);
    CHEQ(strstr(obuf, "E802"), NULL);
    CHNE(strstr(obuf, "member @ 4:54 \xe2\x86\x92 struct nodeb*"), NULL);
    CHNE(strstr(obuf, "member @ 4:57 \xe2\x86\x92 struct nodea*"), NULL);
    PASS();
}
TH_REG("rpi", 131, "two structs may point at each other", rpi131)

static void rpi132(void)
{
    CHNE(scratch("rpi132.cu",
        "struct held { int v; struct ghost *g; };\n"
        "__global__ void k(int *o, struct held *h) { o[0] = h->g->anything; }\n"),
        NULL);
    CHNE(th_run(BC_BIN " --sema build/rpi132.cu", obuf, (int)sizeof obuf), 0);
    CHNE(strstr(obuf, "E802"), NULL);
    CHNE(strstr(obuf, "ghost"), NULL);
    PASS();
}
TH_REG("rpi", 132, "a tag with no definition refuses by name", rpi132)

static void rpi133(void)
{
    const char *p = rpi_ptx(
        "#ifndef __cplusplus\n"
        "#error no cplusplus\n"
        "#endif\n"
        "#if __cplusplus < 201703L\n"
        "#error too old\n"
        "#endif\n"
        "static_assert(sizeof(int) == 4, \"int is four bytes\");\n"
        "namespace ou::in { struct tag { int v; }; }\n"
        "__global__ void k(int *o) { o[0] = 2317; }\n", "rpi133");

    CHNE(p, NULL);
    CHNE(strstr(p, "2317"), NULL);
    PASS();
}
TH_REG("rpi", 133, "C++17 is announced and its syntax parses", rpi133)

static void rpi134(void)
{
    const char *p = rpi_ptx(
        "struct tsz { int qs; int dm; int sc; };\n"
        "static constexpr __device__ tsz sizes(int i) {\n"
        "    return tsz{i*3, i*2, i};\n"
        "}\n"
        "__global__ void k(float *d) {\n"
        "    constexpr tsz ts = sizes(7);\n"
        "    __shared__ float sm[ts.dm];\n"
        "    sm[threadIdx.x % 14] = 1.0f;\n"
        "    d[0] = sm[0];\n"
        "}\n", "rpi134");

    CHNE(p, NULL);
    CHNE(strstr(p, "shmem[56]"), NULL);
    PASS();
}
TH_REG("rpi", 134, "a constexpr struct member sizes an array", rpi134)

static void rpi135(void)
{
    const char *p = rpi_ptx(
        "struct cfg { int w; int h;\n"
        "    constexpr __device__ int area() const { return w*h; } };\n"
        "static constexpr __device__ cfg mkcfg(int n) { return cfg{n, n+1}; }\n"
        "__global__ void k(float *d) {\n"
        "    __shared__ float sm[mkcfg(4).area()];\n"
        "    sm[threadIdx.x % 20] = 1.0f;\n"
        "    d[0] = sm[0];\n"
        "}\n", "rpi135");

    CHNE(p, NULL);
    CHNE(strstr(p, "shmem[80]"), NULL);
    PASS();
}
TH_REG("rpi", 135, "a constexpr method on a temporary folds", rpi135)

static void rpi136(void)
{
    CHNE(scratch("rpi136.cu",
        "struct cfg { int w; int h;\n"
        "    constexpr __device__ int area() const { return w*h; } };\n"
        "static constexpr __device__ cfg mkcfg(int n) { return cfg{n, n+1}; }\n"
        "static constexpr __device__ int warea(int n) {\n"
        "    return mkcfg(n).area();\n"
        "}\n"
        "__global__ void k(float *d) {\n"
        "    __shared__ float sm[warea(5)];\n"
        "    sm[threadIdx.x % 30] = 1.0f;\n"
        "    d[0] = sm[0];\n"
        "}\n"), NULL);
    CHEQ(th_run(BC_BIN " --ir build/rpi136.cu", obuf, (int)sizeof obuf), 0);
    CHEQ(strstr(obuf, "E131"), NULL);
    CHNE(strstr(obuf, "[30 x f32]"), NULL);
    PASS();
}
TH_REG("rpi", 136, "a struct return folds two calls deep", rpi136)

static void rpi137(void)
{
    const char *p = rpi_ptx(
        "struct sel { int a; int b; };\n"
        "static constexpr __host__ __device__ sel pick(int n) {\n"
        "    return sel{n*11, n*13};\n"
        "}\n"
        "template <int W> __global__ void k(int *d) { d[0] = W + 1000; }\n"
        "void g(int *d) { k<pick(3).b><<<1,32>>>(d); }\n", "rpi137");

    CHNE(p, NULL);
    CHNE(strstr(p, "1039"), NULL);
    CHEQ(strstr(p, "1033"), NULL);
    PASS();
}
TH_REG("rpi", 137, "a struct field picks the instantiation", rpi137)

static void rpi138(void)
{
    const char *p = rpi_ptx(
        "enum lay { LA = 0, LB = 1 };\n"
        "static constexpr __device__ int stride(lay l) {\n"
        "    return l == LA ? 9 : 5;\n"
        "}\n"
        "static constexpr __device__ lay layof(int t) {\n"
        "    return t == 1 ? LA : LB;\n"
        "}\n"
        "static constexpr __device__ int stride(int t, int j) {\n"
        "    return stride(layof(t)) + j;\n"
        "}\n"
        "__global__ void k(int *d) {\n"
        "    __shared__ int sm[stride(1, 2)];\n"
        "    sm[threadIdx.x % 11] = 3;\n"
        "    d[0] = sm[0];\n"
        "}\n", "rpi138");

    CHNE(p, NULL);
    CHNE(strstr(p, "shmem[44]"), NULL);
    PASS();
}
TH_REG("rpi", 138, "a constexpr overload is picked by arity", rpi138)

static void rpi139(void)
{
    const char *p = rpi_ptx(
        "struct sc { float m; int n; };\n"
        "static constexpr __device__ sc mksc() { return sc{2.5f, 6}; }\n"
        "__global__ void k(float *d) {\n"
        "    __shared__ float sm[mksc().n];\n"
        "    sm[threadIdx.x % 6] = mksc().m;\n"
        "    d[0] = sm[0];\n"
        "}\n", "rpi139");

    CHNE(p, NULL);
    CHNE(strstr(p, "shmem[24]"), NULL);
    CHNE(strstr(p, "0f40200000"), NULL);
    PASS();
}
TH_REG("rpi", 139, "a constexpr struct keeps a float member", rpi139)

static void rpi140(void)
{
    CHNE(scratch("rpi140.cu",
        "struct pr { int a; int b; };\n"
        "static constexpr __device__ pr mkpr() { return pr{3, 4}; }\n"
        "__global__ void k(int *d) {\n"
        "    __shared__ int sm[mkpr()];\n"
        "    d[0] = sm[0];\n"
        "}\n"), NULL);
    CHNE(th_run(BC_BIN " --ir build/rpi140.cu", obuf, (int)sizeof obuf), 0);
    CHNE(strstr(obuf, "E131"), NULL);
    PASS();
}
TH_REG("rpi", 140, "a struct value is refused as a bound", rpi140)

static void rpi141(void)
{
    const char *p = rpi_ptx(
        "struct box { int w; int h; int d; };\n"
        "static constexpr __device__ box mkbox(int n) {\n"
        "    return box{n, n+1, n+2};\n"
        "}\n"
        "__global__ void k(int *o) {\n"
        "    constexpr box b = mkbox(4);\n"
        "    __shared__ int sm[b.w + b.h + b.d];\n"
        "    sm[threadIdx.x % 15] = 8642;\n"
        "    o[0] = sm[0];\n"
        "}\n", "rpi141");

    CHNE(p, NULL);
    CHNE(strstr(p, "shmem[60]"), NULL);
    CHNE(strstr(p, "8642"), NULL);
    PASS();
}
TH_REG("rpi", 141, "a constexpr struct outlives its declarator", rpi141)

static void rpi142(void)
{
    const char *p = rpi_ptx(
        "struct v2 { int x; int y; };\n"
        "static constexpr __device__ v2 pickv(int n) {\n"
        "    if (n == 1) { struct anon1; return v2{5, 6}; };\n"
        "    if (n == 2) { struct anon2; return v2{7, 8}; };\n"
        "    return v2{0, 0};\n"
        "}\n"
        "__global__ void k(int *o) {\n"
        "    __shared__ int sm[pickv(2).y];\n"
        "    sm[threadIdx.x % 8] = 4;\n"
        "    o[0] = sm[0];\n"
        "}\n", "rpi142");

    CHNE(p, NULL);
    CHNE(strstr(p, "shmem[32]"), NULL);
    PASS();
}
TH_REG("rpi", 142, "a stray declaration does not stop the fold", rpi142)

static const char *bigcall(const char *stem, int nif)
{
    static char src[24576];
    static char out[128];
    char cmd[512];
    int w, i;

    w = snprintf(src, sizeof src,
                 "__device__ int big1(int n, float f)\n{\n int a = n;\n");
    for (i = 0; i < nif && w > 0 && w < (int)sizeof src - 64; i++)
        w += snprintf(src + w, sizeof src - (size_t)w,
                      " if (a > %d) a += %d; else a -= 1;\n", i, i + 2);
    w += snprintf(src + w, sizeof src - (size_t)w,
                  " return a + (int)f;\n}\n"
                  "__global__ void k(int *o)\n"
                  "{ int t = (int)threadIdx.x; o[t] = big1(t, 2.5f); }\n");
    if (w <= 0 || w >= (int)sizeof src) return NULL;
    if (scratch(stem, src) == NULL) return NULL;

    snprintf(out, sizeof out, "build/%s.ptx", stem);
    remove(out);
    snprintf(cmd, sizeof cmd, "%s --nvidia-ptx build/%s -o %s",
             BC_BIN, stem, out);
    if (th_run(cmd, obuf, (int)sizeof obuf) != 0) return NULL;
    return out;
}

static int ptxhas(const char *path, const char *want)
{
    static char line[4096];
    FILE *f = fopen(path, "rb");
    int hit = 0;
    if (!f) return 0;
    while (fgets(line, (int)sizeof line, f) != NULL) {
        if (strstr(line, want) != NULL) { hit = 1; break; }
    }
    fclose(f);
    return hit;
}

static void rpi143(void)
{
    const char *p = rpi_ptx(
        "__global__ void k(int *o){ printf(\"hi %d\\n\", o[0]); }\n",
        "rpi143");

    CHNE(p, NULL);
    CHNE(strstr(p, ".extern .func (.param .b32 status) vprintf"), NULL);
    CHNE(strstr(p, "call.uni (cr), vprintf, (cp0, cp1);"), NULL);
    CHNE(strstr(p, "cvta.global.u64"), NULL);
    CHNE(strstr(p, "cvta.local.u64"), NULL);
    PASS();
}
TH_REG("rpi", 143, "device printf calls the driver's vprintf", rpi143)

static void rpi144(void)
{
    const char *p = rpi_ptx(
        "__global__ void k(int *o, float *g){\n"
        "  printf(\"%d %f %d\\n\", o[0], g[0], o[1]); }\n", "rpi144");

    CHNE(p, NULL);
    CHNE(strstr(p, ".local .align 8 .b8 __local[24];"), NULL);
    CHNE(strstr(p, "cvt.f64.f32"), NULL);
    CHNE(strstr(p, "st.local.f64"), NULL);
    CHNE(strstr(p, "__local+8"), NULL);
    CHNE(strstr(p, "__local+16"), NULL);
    PASS();
}
TH_REG("rpi", 144, "a printf buffer packs at its alignment", rpi144)

static void rpi145(void)
{
    const char *p = rpi_ptx(
        "__global__ void k(int *o){ __threadfence_block(); o[0] = 1; }\n",
        "rpi145");

    CHNE(p, NULL);
    CHNE(strstr(p, "membar.cta"), NULL);
    CHEQ(strstr(p, "bar.sync"), NULL);
    PASS();
}
TH_REG("rpi", 145, "a block fence is membar.cta, not a bar", rpi145)

static void rpi146(void)
{
    const char *p = rpi_ptx(
        "__global__ void k(int *o){ o[0] = 1; __threadfence();\n"
        "  __syncthreads(); o[1] = 2; }\n", "rpi146");

    CHNE(p, NULL);
    CHNE(strstr(p, "membar.gl"), NULL);
    CHNE(strstr(p, "bar.sync 0"), NULL);
    PASS();
}
TH_REG("rpi", 146, "__threadfence keeps its device scope", rpi146)

static void rpi147(void)
{
    const char *p = rpi_ptx(
        "__global__ void k(int *o){\n"
        "  o[0] = __syncthreads_count((int)threadIdx.x & 1); }\n", "rpi147");

    CHNE(p, NULL);
    CHNE(strstr(p, "bar.red.popc.u32"), NULL);
    CHNE(strstr(p, ", 0, %p"), NULL);
    PASS();
}
TH_REG("rpi", 147, "__syncthreads_count reduces with bar.red", rpi147)

static void rpi148(void)
{
    const char *p = rpi_ptx(
        "__global__ void k(int *o){ int t = (int)threadIdx.x;\n"
        "  o[0] = __syncthreads_and(t < 4) + __syncthreads_or(t == 2); }\n",
        "rpi148");

    CHNE(p, NULL);
    CHNE(strstr(p, "bar.red.and.pred"), NULL);
    CHNE(strstr(p, "bar.red.or.pred"), NULL);
    PASS();
}
TH_REG("rpi", 148, "syncthreads and/or keep the predicate", rpi148)

static void rpi149(void)
{
    const char *out = bigcall("rpi149.cu", 120);

    CHNE(out, NULL);
    CHNE(ptxhas(out, "call.uni (cr), big1, (cp0, cp1);"), 0);
    CHNE(ptxhas(out, "st.param.f32 [cp1], "), 0);
    CHNE(ptxhas(out, ", [cr];"), 0);
    CHNE(ptxhas(out, "st.param.u32 [func_retval0], "), 0);
    PASS();
}
TH_REG("rpi", 149, "an uninlined device call passes .param", rpi149)

static void rpi150(void)
{
    const char *out = bigcall("rpi150.cu", 120);

    CHNE(out, NULL);
    CHNE(ptxhas(out, ".func (.param .b32 func_retval0) big1 ("), 0);
    CHNE(ptxhas(out, "\tld.param.f32 %f1, [param1];"), 0);
    CHNE(ptxhas(out, "\tret;"), 0);
    PASS();
}
TH_REG("rpi", 150, "a called device function keeps its ABI", rpi150)

static int lblchk(const char *path)
{
    static char line[4096];
    FILE *f = fopen(path, "rb");
    long cur = -1;
    int bad = 0;
    if (!f) return -1;
    while (fgets(line, (int)sizeof line, f) != NULL) {
        char *q = line;
        if (line[0] == '.') cur = -1;
        if (line[0] == '$' && line[1] == 'L')
            cur = strtol(line + 2, NULL, 10);
        while ((q = strstr(q, "$L")) != NULL) {
            long p = strtol(q + 2, NULL, 10);
            if (cur >= 0 && p != cur) bad++;
            q += 2;
        }
    }
    fclose(f);
    return bad;
}

static void rpi151(void)
{
    char cmd[512];

    remove("build/rpi151.ptx");
    snprintf(cmd, sizeof cmd,
             "%s --nvidia-ptx tests/callbr.cu -o build/rpi151.ptx", BC_BIN);
    CHEQ(th_run(cmd, obuf, (int)sizeof obuf), 0);
    CHNE(ptxhas("build/rpi151.ptx", "call.uni (cr), brf, (cp0, cp1);"), 0);
    CHNE(ptxhas("build/rpi151.ptx", "$L1_"), 0);
    CHEQ(lblchk("build/rpi151.ptx"), 0);
    PASS();
}
TH_REG("rpi", 151, "labels are scoped to the function they sit in", rpi151)

static void rpi152(void)
{
    CHNE(scratch("rpi152.cu",
        "struct wide { int f0; int f1; int f2; int f3; int f4; int f5;\n"
        "              int f6; int f7; int f8; int f9; int fa; int fb;\n"
        "              int fc; };\n"
        "static constexpr __device__ wide mkw() {\n"
        "    return wide{1,2,3,4,5,6,7,8,9,10,11,12,13};\n"
        "}\n"
        "__global__ void k(int *o) {\n"
        "    __shared__ int sm[mkw().f2];\n"
        "    o[0] = sm[0];\n"
        "}\n"), NULL);
    CHNE(th_run(BC_BIN " --ir build/rpi152.cu", obuf, (int)sizeof obuf), 0);
    CHNE(strstr(obuf, "E820"), NULL);
    CHNE(strstr(obuf, "'wide'"), NULL);
    PASS();
}
TH_REG("rpi", 152, "a wide constexpr struct refuses by name", rpi152)

static void rpi153(void)
{
    const char *p = rpi_ptx(
        "template <int I, typename T> struct box { T x[I]; };\n"
        "template <typename T> static __device__ int pick(box<8, T> &b) {\n"
        "    return b.x[0] + 8000;\n"
        "}\n"
        "template <typename T> static __device__ int pick(box<16, T> &b) {\n"
        "    return b.x[0] + 16000;\n"
        "}\n"
        "__global__ void k(int *o) {\n"
        "    box<16, int> b;\n"
        "    b.x[0] = o[1];\n"
        "    o[0] = pick(b);\n"
        "}\n", "rpi153");

    CHNE(p, NULL);
    CHNE(strstr(p, "16000"), NULL);
    CHEQ(strstr(p, "8000"), NULL);
    PASS();
}
TH_REG("rpi", 153, "a tile width picks among overloads", rpi153)

static void rpi154(void)
{
    const char *p = rpi_ptx(
        "struct blob { int a; };\n"
        "static __device__ int over(blob v) { return v.a + 111; }\n"
        "template <typename T> static __device__ int over(T *p) {\n"
        "    return p[0] + 222;\n"
        "}\n"
        "__global__ void k(int *o, int *p) {\n"
        "    blob v;\n"
        "    v.a = o[1];\n"
        "    o[0] = over(v) + over(p);\n"
        "}\n", "rpi154");

    CHNE(p, NULL);
    CHNE(strstr(p, "111"), NULL);
    CHNE(strstr(p, "222"), NULL);
    PASS();
}
TH_REG("rpi", 154, "a template does not take a real name", rpi154)

static void rpi155(void)
{
    const char *p = rpi_ptx(
        "typedef int (*fn_t)(int *);\n"
        "template <int W> static __device__ int body(int *o) {\n"
        "    return o[0] + W;\n"
        "}\n"
        "template <int W> static __device__ int run(int *o) {\n"
        "    constexpr fn_t f = body<W>;\n"
        "    return f(o);\n"
        "}\n"
        "__global__ void k(int *o) { o[0] = run<5>(o) + run<9>(o); }\n",
        "rpi155");

    CHNE(p, NULL);
    CHNE(strstr(p, ", 5;"), NULL);
    CHNE(strstr(p, ", 9;"), NULL);
    PASS();
}
TH_REG("rpi", 155, "a template-id keeps its own instantiation", rpi155)

static void rpi156(void)
{
    CHNE(scratch("rpi156.cu",
        "struct blob { int a; };\n"
        "static __device__ int only(blob v) { return v.a + 333; }\n"
        "template <typename T> static __device__ int only(T *p) {\n"
        "    return p[0] + 444;\n"
        "}\n"
        "__global__ void k(int *o) {\n"
        "    blob v;\n"
        "    v.a = o[1];\n"
        "    o[0] = only(v);\n"
        "}\n"), NULL);
    CHEQ(th_run(BC_BIN " --ir build/rpi156.cu", obuf, (int)sizeof obuf), 0);
    CHNE(strstr(obuf, "call i32 @only("), NULL);
    CHEQ(strstr(obuf, "call i32 @only$"), NULL);
    PASS();
}
TH_REG("rpi", 156, "a guessed deduction does not take a call", rpi156)

static const char *nvptx(const char *stem, const char *src)
{
    static char out[128];
    char cmd[512];

    if (scratch(stem, src) == NULL) return NULL;
    snprintf(out, sizeof out, "build/%s.ptx", stem);
    remove(out);
    snprintf(cmd, sizeof cmd, "%s --nvidia-ptx build/%s -o %s",
             BC_BIN, stem, out);
    if (th_run(cmd, obuf, (int)sizeof obuf) != 0) return NULL;
    return out;
}

static void rpi157(void)
{
    const char *out = nvptx("rpi157.cu",
        "__global__ void k(float *d, long long n)\n"
        "{ for (long long i = threadIdx.x; i < n; i += 32) d[i] = 1.0f; }\n");

    CHNE(out, NULL);
    CHNE(ptxhas(out, "setp.lt.s64 %p"), 0);
    CHEQ(ptxhas(out, ".s32 %rd"), 0);
    PASS();
}
TH_REG("rpi", 157, "a 64-bit compare keeps 64-bit registers", rpi157)

static void rpi158(void)
{
    const char *out = nvptx("rpi158.cu",
        "__global__ void k(long long *d, long long a, long long b)\n"
        "{ d[0] = a / b; d[1] = a % b; }\n");

    CHNE(out, NULL);
    CHNE(ptxhas(out, "div.s64 %rd"), 0);
    CHNE(ptxhas(out, "rem.s64 %rd"), 0);
    CHEQ(ptxhas(out, "div.s32 %rd"), 0);
    PASS();
}
TH_REG("rpi", 158, "a 64-bit divide is not div.s32", rpi158)

static void rpi159(void)
{
    const char *out = nvptx("rpi159.cu",
        "__global__ void k(float *d)\n"
        "{ float v = d[threadIdx.x];\n"
        "  v += __shfl_xor_sync(0xffffffffu, v, 1);\n"
        "  d[threadIdx.x] = v; }\n");

    CHNE(out, NULL);
    CHNE(ptxhas(out, "shfl.sync.bfly.b32 %r"), 0);
    CHNE(ptxhas(out, "mov.b32 %r"), 0);
    CHEQ(ptxhas(out, "shfl.sync.bfly.b32 %f"), 0);
    PASS();
}
TH_REG("rpi", 159, "a float shuffle passes through a b32 reg", rpi159)

static void rpi160(void)
{
    const char *out = nvptx("rpi160.cu",
        "#include <cuda_bf16.h>\n"
        "__global__ void k(__nv_bfloat16 *d, float s)\n"
        "{ d[threadIdx.x] = __float2bfloat16(s * 2.0f); }\n");

    CHNE(out, NULL);
    CHNE(ptxhas(out, ".reg .b16  %rh"), 0);
    CHEQ(ptxhas(out, ".reg .u16  %rh"), 0);
    CHNE(ptxhas(out, "cvt.rn.bf16.f32 %rh"), 0);
    PASS();
}
TH_REG("rpi", 160, "a bf16 value lives in a .b16 register", rpi160)

static void rpi161(void)
{
    const char *out = nvptx("rpi161.cu",
        "__global__ void k(short *d, short a, short b)\n"
        "{ d[threadIdx.x] = (short)(a * b + 7); }\n");

    CHNE(out, NULL);
    CHNE(ptxhas(out, "cvt.u16.u32 %rh"), 0);
    CHEQ(ptxhas(out, "mul.lo.u32 %rh"), 0);
    CHEQ(ptxhas(out, "add.u32 %rh"), 0);
    PASS();
}
TH_REG("rpi", 161, "short arithmetic widens to 32 bits", rpi161)

static void rpi162(void)
{
    nv_module_t *nv = (nv_module_t *)calloc(1, sizeof *nv);
    nv_minst_t *I;

    CHNE(nv, NULL);
    nv->num_mfunc = 1;
    nv->num_mblk = 1;
    nv->num_minst = 1;
    nv->mfuncs[0].first_blk = 0;
    nv->mfuncs[0].num_blks = 1;
    nv->mblks[0].first_inst = 0;
    nv->mblks[0].num_insts = 1;

    I = &nv->minsts[0];
    I->op = NV_SETP_GE_S32;
    I->num_defs = 1;
    I->num_uses = 2;
    I->ops[0].kind = NV_MOP_REG; I->ops[0].rfile = NV_RF_PRED;
    I->ops[1].kind = NV_MOP_REG; I->ops[1].rfile = NV_RF_U64;
    I->ops[2].kind = NV_MOP_REG; I->ops[2].rfile = NV_RF_U64;
    CHNE(nv_vchk(nv), BC_OK);

    I->op = NV_SETP_GE_S64;
    CHEQ(nv_vchk(nv), BC_OK);

    I->op = NV_CVT_F32_S32;
    I->num_uses = 1;
    I->ops[0].rfile = NV_RF_F32;
    I->ops[1].rfile = NV_RF_U32;
    I->ops[2].kind = NV_MOP_NONE;
    CHEQ(nv_vchk(nv), BC_OK);

    I->ops[1].kind = NV_MOP_IMM;
    CHNE(nv_vchk(nv), BC_OK);

    free(nv);
    PASS();
}
TH_REG("rpi", 162, "the PTX check refuses a type suffix clash", rpi162)

static void rpi163(void)
{
    const char *out = nvptx("rpi163.cu",
        "__global__ void k(float *d, __half h)\n"
        "{ d[threadIdx.x] = __half2float(h); }\n");

    CHNE(out, NULL);
    CHNE(ptxhas(out, ".param .b16 param1"), 0);
    CHNE(ptxhas(out, "ld.param.b16 %"), 0);
    PASS();
}
TH_REG("rpi", 163, "a half kernel parameter is .param .b16", rpi163)

static bir_module_t *vmbuf(void)
{
    static bir_module_t *M;
    if (M == NULL) M = malloc(sizeof *M);
    if (M != NULL) bir_module_init(M);
    return M;
}

static bir_module_t *vmod2(void)
{
    bir_module_t *M = vmbuf();
    uint32_t i32, vt, f;

    if (M == NULL) return NULL;
    i32 = bir_type_int(M, 32);
    vt  = bir_type_void(M);
    M->num_insts = 10;
    for (f = 0; f < 2u; f++) {
        uint32_t o = f * 5u;
        M->insts[o] = (bir_inst_t){ .op = BIR_THREAD_ID, .type = i32 };
        M->insts[o + 1u] = (bir_inst_t){ .op = BIR_ADD, .num_operands = 2,
                                         .type = i32 };
        M->insts[o + 1u].operands[0] = BIR_MAKE_VAL(o);
        M->insts[o + 1u].operands[1] = BIR_MAKE_VAL(o);
        M->insts[o + 2u] = (bir_inst_t){ .op = BIR_BR, .num_operands = 1,
                                         .type = vt };
        M->insts[o + 2u].operands[0] = f * 2u + 1u;
        M->insts[o + 3u] = (bir_inst_t){ .op = BIR_ADD, .num_operands = 2,
                                         .type = i32 };
        M->insts[o + 3u].operands[0] = BIR_MAKE_VAL(o + 1u);
        M->insts[o + 3u].operands[1] = BIR_MAKE_VAL(o);
        M->insts[o + 4u] = (bir_inst_t){ .op = BIR_RET, .num_operands = 1,
                                         .type = vt };
        M->insts[o + 4u].operands[0] = BIR_MAKE_VAL(o + 3u);
        M->blocks[f * 2u] = (bir_block_t){ .name = bir_add_string(M, "entry", 5),
                                           .first_inst = o, .num_insts = 3 };
        M->blocks[f * 2u + 1u] = (bir_block_t){
            .name = bir_add_string(M, "tail", 4),
            .first_inst = o + 3u, .num_insts = 2 };
        M->funcs[f].name = bir_add_string(M, f ? "fone" : "fnil", 4);
        M->funcs[f].first_block = f * 2u;
        M->funcs[f].num_blocks  = 2;
        M->funcs[f].total_insts = 5;
    }
    M->num_blocks = 4;
    M->num_funcs  = 2;
    return M;
}

static bir_module_t *vmod3(void)
{
    bir_module_t *M = vmbuf();
    uint32_t i32, i1, vt;

    if (M == NULL) return NULL;
    i32 = bir_type_int(M, 32);
    i1  = bir_type_int(M, 1);
    vt  = bir_type_void(M);
    M->num_insts = 6;
    M->insts[0] = (bir_inst_t){ .op = BIR_THREAD_ID, .type = i32 };
    M->insts[1] = (bir_inst_t){ .op = BIR_ICMP, .num_operands = 2,
                                .subop = BIR_ICMP_EQ, .type = i1 };
    M->insts[1].operands[0] = BIR_MAKE_VAL(0);
    M->insts[1].operands[1] = BIR_MAKE_VAL(0);
    M->insts[2] = (bir_inst_t){ .op = BIR_BR_COND, .num_operands = 3,
                                .type = vt };
    M->insts[2].operands[0] = BIR_MAKE_VAL(1);
    M->insts[2].operands[1] = 1;
    M->insts[2].operands[2] = 2;
    M->insts[3] = (bir_inst_t){ .op = BIR_BR, .num_operands = 1, .type = vt };
    M->insts[3].operands[0] = 2;
    M->insts[4] = (bir_inst_t){ .op = BIR_PHI, .num_operands = 4, .type = i32 };
    M->insts[4].operands[0] = 0;
    M->insts[4].operands[1] = BIR_MAKE_VAL(0);
    M->insts[4].operands[2] = 1;
    M->insts[4].operands[3] = BIR_MAKE_VAL(0);
    M->insts[5] = (bir_inst_t){ .op = BIR_RET, .num_operands = 1, .type = vt };
    M->insts[5].operands[0] = BIR_MAKE_VAL(4);

    M->blocks[0] = (bir_block_t){ .name = bir_add_string(M, "entry", 5),
                                  .first_inst = 0, .num_insts = 3 };
    M->blocks[1] = (bir_block_t){ .name = bir_add_string(M, "arm", 3),
                                  .first_inst = 3, .num_insts = 1 };
    M->blocks[2] = (bir_block_t){ .name = bir_add_string(M, "join", 4),
                                  .first_inst = 4, .num_insts = 2 };
    M->funcs[0].name = bir_add_string(M, "fdia", 4);
    M->funcs[0].first_block = 0;
    M->funcs[0].num_blocks  = 3;
    M->funcs[0].total_insts = 6;
    M->num_blocks = 3;
    M->num_funcs  = 1;
    return M;
}

static void rpi165(void)
{
    bir_module_t *M = vmod2();
    CHNE(M, NULL);
    CHEQ(bir_vchk(M), BC_OK);
    M = vmod3();
    CHNE(M, NULL);
    CHEQ(bir_vchk(M), BC_OK);
    PASS();
}
TH_REG("rpi", 165, "a well-formed module passes containment", rpi165)

static void rpi166(void)
{
    bir_module_t *M = vmod2();
    CHNE(M, NULL);
    M->insts[2].operands[0] = 3;
    CHEQ(bir_vchk(M), BC_ERR_VERIFY);
    PASS();
}
TH_REG("rpi", 166, "a branch into another function refuses", rpi166)

static void rpi167(void)
{
    bir_module_t *M = vmod2();
    CHNE(M, NULL);
    M->insts[3].operands[0] = BIR_MAKE_VAL(6);
    CHEQ(bir_vchk(M), BC_ERR_VERIFY);
    PASS();
}
TH_REG("rpi", 167, "a value from another function refuses", rpi167)

static void rpi168(void)
{
    bir_module_t *M = vmod2();
    CHNE(M, NULL);
    M->funcs[1].first_block = 1;
    CHEQ(bir_vchk(M), BC_ERR_VERIFY);
    PASS();
}
TH_REG("rpi", 168, "two functions over one block refuse", rpi168)

static void rpi169(void)
{
    bir_module_t *M = vmod3();
    CHNE(M, NULL);
    M->insts[4].operands[2] = 2;
    CHEQ(bir_vchk(M), BC_ERR_VERIFY);
    PASS();
}
TH_REG("rpi", 169, "a phi naming a non-predecessor refuses", rpi169)

static void rpi170(void)
{
    bir_module_t *M = vmod3();
    CHNE(M, NULL);
    M->insts[4].num_operands = 2;
    CHEQ(bir_vchk(M), BC_ERR_VERIFY);
    PASS();
}
TH_REG("rpi", 170, "a phi short of a predecessor refuses", rpi170)

static void rpi171(void)
{
    CHNE(scratch("rpi171.cu",
        "__global__ void k(float *o)\n"
        "{\n"
        "    o[0] = o[1] + 1.0f;\n"
        "    __threadfence();\n"
        "    o[2] = o[3];\n"
        "}\n"), NULL);
    remove("build/rpi171.cpp");
    CHNE(th_run(BC_BIN " --tensix build/rpi171.cu -o build/rpi171.cpp",
                obuf, (int)sizeof obuf), 0);
    CHNE(strstr(obuf, "E866"), NULL);
    CHNE(strstr(obuf, "__threadfence"), NULL);
    CHEQ(th_exist("build/rpi171.cpp"), 0);
    PASS();
}
TH_REG("rpi", 171, "__threadfence refuses on the tile path", rpi171)

static void rpi172(void)
{
    CHNE(scratch("rpi172.cu",
        "__global__ void k(float *o) { o[0] = sqrtf(o[1]); }\n"), NULL);
    remove("build/rpi172.cpp");
    CHNE(th_run(BC_BIN " --tensix build/rpi172.cu -o build/rpi172.cpp",
                obuf, (int)sizeof obuf), 0);
    CHNE(strstr(obuf, "E867"), NULL);
    CHNE(strstr(obuf, "sqrt"), NULL);
    CHEQ(th_exist("build/rpi172.cpp"), 0);
    PASS();
}
TH_REG("rpi", 172, "the tile path refuses an op by name", rpi172)

static void rpi173(void)
{
    const char *p = rpi_ptx(
        "struct A { int pad; struct B *b; };\n"
        "struct B { int u; int v; };\n"
        "__global__ void k(struct A *a, int *o){ o[0] = a->b->v; }\n",
        "rpi173");

    CHNE(p, NULL);
    CHEQ(strstr(obuf, "E110"), NULL);
    CHNE(strstr(p, ".entry k"), NULL);
    CHNE(strstr(p, "add.u64 %rd4, %rd1, 8;"), NULL);
    CHNE(strstr(p, "add.u64 %rd6, %rd5, 4;"), NULL);
    PASS();
}
TH_REG("rpi", 173, "a tag used before its definition resolves", rpi173)

static void rpi174(void)
{
    const char *p = rpi_ptx(
        "struct N { int v; struct N *nx; };\n"
        "__global__ void k(struct N *n, int *o){ o[0] = n->nx->nx->v; }\n",
        "rpi174");

    CHNE(p, NULL);
    CHEQ(strstr(obuf, "E110"), NULL);
    CHNE(strstr(p, ".entry k"), NULL);
    CHNE(strstr(p, "add.u64 %rd4, %rd1, 8;"), NULL);
    CHNE(strstr(p, "add.u64 %rd6, %rd5, 8;"), NULL);
    PASS();
}
TH_REG("rpi", 174, "a struct that points at itself lowers", rpi174)

static void rpi175(void)
{
    const char *p = rpi_ptx(
        "struct P { struct Q *q; int a; };\n"
        "struct Q { struct P *p; int b; };\n"
        "__global__ void k(struct P *p, int *o){ o[0] = p->q->p->a; }\n",
        "rpi175");

    CHNE(p, NULL);
    CHEQ(strstr(obuf, "E110"), NULL);
    CHNE(strstr(p, ".entry k"), NULL);
    CHNE(strstr(p, "add.u64 %rd8, %rd7, 8;"), NULL);
    PASS();
}
TH_REG("rpi", 175, "two structs that point at each other lower", rpi175)

static void rpi176(void)
{
    const char *p = rpi_ptx(
        "struct L1 { struct L2 *n; };\n"
        "struct L2 { struct L3 *n; };\n"
        "struct L3 { int pad; int deep; };\n"
        "__global__ void k(struct L1 *l, int *o){ o[0] = l->n->n->deep; }\n",
        "rpi176");

    CHNE(p, NULL);
    CHEQ(strstr(obuf, "E110"), NULL);
    CHNE(strstr(p, ".entry k"), NULL);
    CHNE(strstr(p, "add.u64 %rd8, %rd7, 4;"), NULL);
    PASS();
}
TH_REG("rpi", 176, "a three-deep pointer chain keeps offsets", rpi176)

static void rpi177(void)
{
    const char *p = rpi_ptx(
        "typedef struct T *Tp;\n"
        "struct T { int pad0; int pad1; int t; };\n"
        "__global__ void k(Tp t, int *o){ o[0] = t->t; }\n",
        "rpi177");

    CHNE(p, NULL);
    CHEQ(strstr(obuf, "E110"), NULL);
    CHNE(strstr(p, ".entry k"), NULL);
    CHNE(strstr(p, "add.u64 %rd4, %rd1, 8;"), NULL);
    PASS();
}
TH_REG("rpi", 177, "a tag reached through a typedef resolves", rpi177)

static void rpi178(void)
{
    int rc = rpi_err(
        "struct C;\n"
        "__global__ void k(struct C *c, int *o){ o[0] = c->v; }\n",
        "rpi178");

    CHNE(rc, 0);
    CHEQ(ecnt(obuf, "E802"), 1);
    CHNE(strstr(obuf, "struct C"), NULL);
    CHEQ(strstr(obuf, "E110"), NULL);
    PASS();
}
TH_REG("rpi", 178, "a tag that is never defined refuses by name", rpi178)

static void rpi179(void)
{
    const char *p = rpi_ptx(
        "struct ctx { int dev; int nb; };\n"
        "__global__ void k(int *o){\n"
        "  int2 v = {0, 0};\n"
        "  v.x = 11;\n"
        "  v.y = 22;\n"
        "  o[0] = v.x + v.y; }\n",
        "rpi179");

    CHNE(p, NULL);
    CHEQ(strstr(obuf, "E110"), NULL);
    CHNE(strstr(p, ".entry k"), NULL);
    CHNE(strstr(p, "mov.u32 %r1, 33;"), NULL);
    PASS();
}
TH_REG("rpi", 179, "int2 keeps its own type beside a struct", rpi179)

static void rpi180(void)
{
    int rc = rpi_err(
        "__global__ void k(__half2 *o, half2 *i){ half2 t = i[0]; o[0] = t; }\n",
        "rpi180");

    CHEQ(rc, 0);
    CHNE(strstr(obuf, "ptr<global, type_2> %0, ptr<global, type_2> %1"),
         NULL);
    PASS();
}
TH_REG("rpi", 180, "half2 and __half2 stay the same BIR type", rpi180)

static void rpi201(void)
{
    const char *p = rpi_ptx(
        "namespace cooperative_groups { }\n"
        "namespace cg = cooperative_groups;\n"
        "__global__ void k(int *d){\n"
        "  cg::grid_group g = cg::this_grid();\n"
        "  int r = g.thread_rank();\n"
        "  d[r] = r;\n"
        "  g.sync();\n"
        "  d[r] += d[0]; }\n", "rpi201");

    CHNE(p, NULL);
    CHNE(strstr(p, "atom.acq_rel.gpu.global.add.u32"), NULL);
    CHNE(strstr(p, "$bgbar"), NULL);
    CHEQ(ecnt(p, ".global .align 4 .u32 $bgbar[2] = {0, 0};"), 1);
    PASS();
}
TH_REG("rpi", 201, "a grid sync arrives on a global counter", rpi201)

static void rpi202(void)
{
    const char *p = rpi_ptx(
        "namespace cooperative_groups { }\n"
        "namespace cg = cooperative_groups;\n"
        "__global__ void k(int *d){\n"
        "  cg::grid_group g = cg::this_grid();\n"
        "  int r = g.thread_rank();\n"
        "  d[r] = r;\n"
        "  g.sync();\n"
        "  d[r] += d[0]; }\n", "rpi202");
    const char *rst, *rel;

    CHNE(p, NULL);
    rst = strstr(p, "st.relaxed.gpu.global.u32 [%rd_gb], 0;");
    rel = strstr(p, "atom.release.gpu.global.add.u32 %r_gb1, [%rd_gb+4]");
    CHNE(rst, NULL);
    CHNE(rel, NULL);
    CHEQ(rst < rel, 1);
    PASS();
}
TH_REG("rpi", 202, "the arrival counter is reset before release", rpi202)

static void rpi203(void)
{
    const char *p = rpi_ptx(
        "namespace cooperative_groups { }\n"
        "namespace cg = cooperative_groups;\n"
        "__global__ void k(int *d){\n"
        "  cg::grid_group g = cg::this_grid();\n"
        "  int r = g.thread_rank();\n"
        "  d[r] = r;\n"
        "  g.sync();\n"
        "  d[r] += d[0]; }\n", "rpi203");

    CHNE(p, NULL);
    CHEQ(ecnt(p, "ld.acquire.gpu.global.u32 %r_gb2, [%rd_gb+4];"), 1);
    CHEQ(ecnt(p, "ld.acquire.gpu.global.u32 %r_gb1, [%rd_gb+4];"), 1);
    CHEQ(strstr(p, "ld.global.u32 %r_gb1"), NULL);
    PASS();
}
TH_REG("rpi", 203, "the phase is read and waited on at gpu scope", rpi203)

static void rpi204(void)
{
    const char *p = rpi_ptx(
        "namespace cooperative_groups { }\n"
        "namespace cg = cooperative_groups;\n"
        "__global__ void k(int *d){\n"
        "  cg::grid_group g = cg::this_grid();\n"
        "  int r = g.thread_rank();\n"
        "  d[r] = r;\n"
        "  g.sync();\n"
        "  d[r] += d[0];\n"
        "  g.sync();\n"
        "  d[r] += 7; }\n", "rpi204");

    CHNE(p, NULL);
    CHEQ(ecnt(p, ".global .align 4 .u32 $bgbar[2] = {0, 0};"), 1);
    CHEQ(ecnt(p, "$GBS0:"), 1);
    CHEQ(ecnt(p, "$GBS1:"), 1);
    CHEQ(ecnt(p, "$GBW0:"), 1);
    CHEQ(ecnt(p, "$GBW1:"), 1);
    CHEQ(ecnt(p, "$GBS2:"), 0);
    PASS();
}
TH_REG("rpi", 204, "two grid syncs share one counter, not labels", rpi204)

static void rpi205(void)
{
    const char *p = rpi_ptx(
        "namespace cooperative_groups { }\n"
        "namespace cg = cooperative_groups;\n"
        "__global__ void k(int *d){\n"
        "  cg::grid_group g = cg::this_grid();\n"
        "  int r = g.thread_rank();\n"
        "  d[r] = r;\n"
        "  g.sync();\n"
        "  d[r] += d[0]; }\n", "rpi205");

    CHNE(p, NULL);
    CHEQ(ecnt(p, "bar.sync 0;"), 2);
    CHEQ(ecnt(p, "setp.ne.u32 %p_gb0, %r_gb0, 0;"), 1);
    PASS();
}
TH_REG("rpi", 205, "a grid sync is bracketed by block barriers", rpi205)

static void rpi206(void)
{
    const char *p = rpi_ptx(
        "namespace cooperative_groups { }\n"
        "namespace cg = cooperative_groups;\n"
        "__global__ void k(int *d){\n"
        "  cg::grid_group g = cg::this_grid();\n"
        "  int r = g.thread_rank();\n"
        "  d[r] = r;\n"
        "  g.sync();\n"
        "  d[r] += d[0]; }\n", "rpi206");

    CHNE(p, NULL);
    CHNE(strstr(p, "mov.u32 %r_gb0, %nctaid.x;"), NULL);
    CHNE(strstr(p, "mov.u32 %r_gb1, %nctaid.y;"), NULL);
    CHNE(strstr(p, "mov.u32 %r_gb1, %nctaid.z;"), NULL);
    CHNE(strstr(p, "sub.u32 %r_gb0, %r_gb0, 1;"), NULL);
    PASS();
}
TH_REG("rpi", 206, "the block count spans all three dimensions", rpi206)

static void rpi207(void)
{
    const char *p = rpi_ptx(
        "namespace cooperative_groups { }\n"
        "namespace cg = cooperative_groups;\n"
        "__global__ void k(int *d){\n"
        "  cg::grid_group g = cg::this_grid();\n"
        "  int r = g.thread_rank();\n"
        "  d[r] = r;\n"
        "  g.sync();\n"
        "  d[r] += d[0];\n"
        "  g.sync();\n"
        "  d[r] += 7; }\n", "rpi207");

    CHNE(p, NULL);
    CHEQ(ecnt(p, ".reg .u32  %r_gb0;"), 1);
    CHEQ(ecnt(p, ".reg .u32  %r_gb1;"), 1);
    CHEQ(ecnt(p, ".reg .u32  %r_gb2;"), 1);
    CHEQ(ecnt(p, ".reg .u64  %rd_gb;"), 1);
    CHEQ(ecnt(p, ".reg .pred %p_gb0;"), 1);
    PASS();
}
TH_REG("rpi", 207, "the barrier scratch registers are declared", rpi207)

static void rpi208(void)
{
    const char *p = rpi_ptx(
        "__global__ void k(int *d){ d[0] = 1; }\n", "rpi208");

    CHNE(p, NULL);
    CHEQ(strstr(p, "$bgbar"), NULL);
    CHEQ(strstr(p, "%r_gb0"), NULL);
    CHEQ(strstr(p, "bar.sync"), NULL);
    PASS();
}
TH_REG("rpi", 208, "a kernel with no grid sync has no counter", rpi208)

static void rpi209(void)
{
    const char *p = rpi_ptx(
        "namespace cooperative_groups { }\n"
        "namespace cg = cooperative_groups;\n"
        "__global__ void k(int *d){\n"
        "  cg::grid_group g = cg::this_grid();\n"
        "  int r = g.thread_rank();\n"
        "  d[r] = r;\n"
        "  g.sync();\n"
        "  d[r] += d[0]; }\n", "rpi209");

    CHNE(p, NULL);
    CHNE(strstr(p, "cudaLaunchCooperativeKernel"), NULL);
    CHNE(strstr(p, "every block is resident"), NULL);
    PASS();
}
TH_REG("rpi", 209, "the PTX names the cooperative launch need", rpi209)

static void rpi210(void)
{
    const char *p = rpi_ptx(
        "namespace cooperative_groups { }\n"
        "namespace cg = cooperative_groups;\n"
        "__global__ void k(int *d){\n"
        "  cg::thread_block b = cg::this_thread_block();\n"
        "  b.sync();\n"
        "  d[b.thread_rank()] = 1; }\n", "rpi210");

    CHNE(p, NULL);
    CHEQ(ecnt(p, "bar.sync 0;"), 1);
    CHEQ(strstr(p, "$bgbar"), NULL);
    PASS();
}
TH_REG("rpi", 210, "a block sync stays one bar.sync, not a grid", rpi210)

/* ---- WMMA ----
 * Register counts and legal combinations come from PTX ISA 9.2 9.7.14.4.1;
 * the numbers checked here are the ones the document tabulates. */

static const char *WM_HDR =
    "__global__ void k(const half *a, const half *b, float *d,\n"
    "                  int la, int lb, int ld){\n"
    "  wmma::fragment<wmma::matrix_a, 16, 16, 16, half,\n"
    "                 wmma::row_major> af;\n"
    "  wmma::fragment<wmma::matrix_b, 16, 16, 16, half,\n"
    "                 wmma::col_major> bf;\n"
    "  wmma::fragment<wmma::accumulator, 16, 16, 16, float> cf;\n";

static const char *wm_ptx(const char *tail, const char *stem)
{
    static char src[2048];

    if (snprintf(src, sizeof src, "%s%s", WM_HDR, tail) < 0) return NULL;
    return rpi_ptx(src, stem);
}

static int wm_err(const char *tail, const char *stem)
{
    static char src[2048];

    if (snprintf(src, sizeof src, "%s%s", WM_HDR, tail) < 0) return -1;
    return rpi_err(src, stem);
}

static void rpi181(void)
{
    const char *p = wm_ptx(
        "  wmma::load_matrix_sync(af, a, la);\n"
        "  wmma::load_matrix_sync(bf, b, lb);\n"
        "  wmma::fill_fragment(cf, 0.0f);\n"
        "  wmma::mma_sync(cf, af, bf, cf);\n"
        "  wmma::store_matrix_sync(d, cf, ld, wmma::mem_row_major); }\n",
        "rpi181");

    CHNE(p, NULL);
    CHNE(strstr(p, "wmma.load.a.sync.aligned.row.m16n16k16.f16 "
                   "{%rb1, %rb2, %rb3, %rb4, %rb5, %rb6, %rb7, %rb8}"), NULL);
    CHNE(strstr(p, "wmma.load.b.sync.aligned.col.m16n16k16.f16"), NULL);
    CHNE(strstr(p, "wmma.mma.sync.aligned.row.col.m16n16k16.f32.f32"), NULL);
    CHNE(strstr(p, "wmma.store.d.sync.aligned.row.m16n16k16.f32"), NULL);
    CHEQ(strstr(p, "mma.sync.aligned.m16n8k16"), NULL);
    PASS();
}
TH_REG("rpi", 181, "a wmma f16 multiply reaches PTX intact", rpi181)

static void rpi182(void)
{
    const char *p = wm_ptx(
        "  wmma::load_matrix_sync(cf, d, ld, wmma::mem_col_major);\n"
        "  wmma::store_matrix_sync(d, cf, ld, wmma::mem_col_major); }\n",
        "rpi182");

    CHNE(p, NULL);
    CHNE(strstr(p, "wmma.load.c.sync.aligned.col.m16n16k16.f32"), NULL);
    CHNE(strstr(p, "wmma.store.d.sync.aligned.col.m16n16k16.f32"), NULL);
    PASS();
}
TH_REG("rpi", 182, "an accumulator load takes its own layout", rpi182)

static void rpi183(void)
{
    int rc = rpi_err(
        "__global__ void k(float *d){\n"
        "  wmma::fragment<wmma::matrix_a, 16, 16, 32, half,\n"
        "                 wmma::row_major> a;\n"
        "  d[0] = 1.0f; }\n", "rpi183");

    CHNE(rc, 0);
    CHNE(strstr(obuf, "E904"), NULL);
    CHNE(strstr(obuf, "m16n16k32"), NULL);
    PASS();
}
TH_REG("rpi", 183, "a wmma shape the ISA omits refuses", rpi183)

static void rpi184(void)
{
    int rc = rpi_err(
        "__global__ void k(float *d){\n"
        "  wmma::fragment<wmma::matrix_a, 16, 16, 16, long,\n"
        "                 wmma::row_major> a;\n"
        "  d[0] = 1.0f; }\n", "rpi184");

    CHNE(rc, 0);
    CHNE(strstr(obuf, "E901"), NULL);
    PASS();
}
TH_REG("rpi", 184, "a wmma element type the ISA omits refuses", rpi184)

static void rpi185(void)
{
    int rc = rpi_err(
        "__global__ void k(float *d){\n"
        "  wmma::fragment<wmma::matrix_c, 16, 16, 16, half,\n"
        "                 wmma::row_major> a;\n"
        "  d[0] = 1.0f; }\n", "rpi185");

    CHNE(rc, 0);
    CHNE(strstr(obuf, "E902"), NULL);
    CHNE(strstr(obuf, "matrix_c"), NULL);
    PASS();
}
TH_REG("rpi", 185, "a wmma use that is not a matrix refuses", rpi185)

static void rpi186(void)
{
    int rc = rpi_err(
        "__global__ void k(float *d){\n"
        "  wmma::fragment<wmma::matrix_a, 16, 16, 16, half,\n"
        "                 wmma::diag_major> a;\n"
        "  d[0] = 1.0f; }\n", "rpi186");

    CHNE(rc, 0);
    CHNE(strstr(obuf, "E903"), NULL);
    CHNE(strstr(obuf, "diag_major"), NULL);
    PASS();
}
TH_REG("rpi", 186, "a wmma layout Booth cannot name refuses", rpi186)

static void rpi187(void)
{
    int rc = wm_err(
        "  wmma::fragment<wmma::matrix_a, 8, 32, 16, half,\n"
        "                 wmma::row_major> qf;\n"
        "  wmma::mma_sync(cf, qf, bf, cf); }\n", "rpi187");

    CHNE(rc, 0);
    CHEQ(ecnt(obuf, "E907"), 1);
    PASS();
}
TH_REG("rpi", 187, "mma_sync across two shapes refuses", rpi187)

static void rpi188(void)
{
    int rc = wm_err(
        "  wmma::store_matrix_sync(d, cf, ld, wmma::mem_diag_major); }\n",
        "rpi188");

    CHNE(rc, 0);
    CHEQ(ecnt(obuf, "E908"), 1);
    CHNE(strstr(obuf, "mem_diag_major"), NULL);
    PASS();
}
TH_REG("rpi", 188, "a store layout that is not mem_ refuses", rpi188)

static void rpi189(void)
{
    int rc = wm_err("  wmma::load_matrix_sync(af, a); }\n", "rpi189");

    CHNE(rc, 0);
    CHEQ(ecnt(obuf, "E906"), 1);
    PASS();
}
TH_REG("rpi", 189, "load_matrix_sync short of arguments refuses", rpi189)

static void rpi190(void)
{
    const char *p = rpi_ptx(
        "__global__ void k(const __nv_bfloat16 *a, float *d, int l){\n"
        "  wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16,\n"
        "                 wmma::row_major> af;\n"
        "  wmma::load_matrix_sync(af, a, l); d[0] = 1.0f; }\n", "rpi190");

    CHNE(p, NULL);
    CHNE(strstr(p, "wmma.load.a.sync.aligned.row.m16n16k16.bf16 "
                   "{%rb1, %rb2, %rb3, %rb4}"), NULL);
    PASS();
}
TH_REG("rpi", 190, "bf16 fragments hold four b32 registers", rpi190)

static void rpi191(void)
{
    const char *p = rpi_ptx(
        "__global__ void k(const signed char *a, const signed char *b,\n"
        "                  int *d, int la, int lb, int ld){\n"
        "  wmma::fragment<wmma::matrix_a, 16, 16, 16, signed char,\n"
        "                 wmma::row_major> af;\n"
        "  wmma::fragment<wmma::matrix_b, 16, 16, 16, signed char,\n"
        "                 wmma::col_major> bf;\n"
        "  wmma::fragment<wmma::accumulator, 16, 16, 16, int> cf;\n"
        "  wmma::load_matrix_sync(af, a, la);\n"
        "  wmma::load_matrix_sync(bf, b, lb);\n"
        "  wmma::fill_fragment(cf, 0);\n"
        "  wmma::mma_sync(cf, af, bf, cf);\n"
        "  wmma::store_matrix_sync(d, cf, ld, wmma::mem_row_major); }\n",
        "rpi191");

    CHNE(p, NULL);
    CHNE(strstr(p, "wmma.load.a.sync.aligned.row.m16n16k16.s8 "
                   "{%rb1, %rb2}"), NULL);
    CHNE(strstr(p, "wmma.mma.sync.aligned.row.col.m16n16k16.s32.s8.s8.s32"),
         NULL);
    PASS();
}
TH_REG("rpi", 191, "s8 multiplicands reach an s32 accumulator", rpi191)

static void rpi192(void)
{
    const char *p = rpi_ptx(
        "__global__ void k(const float *a, float *d, int l){\n"
        "  wmma::fragment<wmma::matrix_a, 16, 16, 8, wmma::precision::tf32,\n"
        "                 wmma::row_major> af;\n"
        "  wmma::load_matrix_sync(af, a, l); d[0] = 1.0f; }\n", "rpi192");

    CHNE(p, NULL);
    CHNE(strstr(p, "wmma.load.a.sync.aligned.row.m16n16k8.tf32 "
                   "{%rb1, %rb2, %rb3, %rb4}"), NULL);
    PASS();
}
TH_REG("rpi", 192, "tf32 lowers at m16n16k8 with four registers", rpi192)

static void rpi193(void)
{
    const char *p = rpi_ptx(
        "__global__ void k(const double *a, double *d, int l){\n"
        "  wmma::fragment<wmma::matrix_a, 8, 8, 4, double,\n"
        "                 wmma::row_major> af;\n"
        "  wmma::fragment<wmma::accumulator, 8, 8, 4, double> cf;\n"
        "  wmma::load_matrix_sync(af, a, l);\n"
        "  wmma::store_matrix_sync(d, cf, l, wmma::mem_row_major); }\n",
        "rpi193");

    CHNE(p, NULL);
    CHNE(strstr(p, "wmma.load.a.sync.aligned.row.m8n8k4.f64 {%fd1}"), NULL);
    CHNE(strstr(p, "wmma.store.d.sync.aligned.row.m8n8k4.f64"), NULL);
    PASS();
}
TH_REG("rpi", 193, "f64 wmma lands in the f64 register file", rpi193)

static void rpi194(void)
{
    int rc = rpi_err(
        "__global__ void k(float *d){\n"
        "  wmma::fragment<wmma::matrix_b, 8, 8, 32,\n"
        "                 wmma::experimental::precision::s4,\n"
        "                 wmma::row_major> b;\n"
        "  d[0] = 1.0f; }\n", "rpi194");

    CHNE(rc, 0);
    CHNE(strstr(obuf, "E909"), NULL);
    PASS();
}
TH_REG("rpi", 194, "a sub-byte matrix_b must be column-major", rpi194)

static void rpi195(void)
{
    const char *p = rpi_ptx(
        "__global__ void k(const unsigned *a, const unsigned *b, int *d,\n"
        "                  int la, int lb, int ld){\n"
        "  wmma::fragment<wmma::matrix_a, 8, 8, 128,\n"
        "                 wmma::experimental::precision::b1,\n"
        "                 wmma::row_major> af;\n"
        "  wmma::fragment<wmma::matrix_b, 8, 8, 128,\n"
        "                 wmma::experimental::precision::b1,\n"
        "                 wmma::col_major> bf;\n"
        "  wmma::fragment<wmma::accumulator, 8, 8, 128, int> cf;\n"
        "  wmma::load_matrix_sync(af, a, la);\n"
        "  wmma::load_matrix_sync(bf, b, lb);\n"
        "  wmma::fill_fragment(cf, 0);\n"
        "  wmma::bmma_sync(cf, af, bf, cf,\n"
        "                  wmma::experimental::bmmaBitOpXOR);\n"
        "  wmma::store_matrix_sync(d, cf, ld, wmma::mem_row_major); }\n",
        "rpi195");

    CHNE(p, NULL);
    CHNE(strstr(p, "wmma.mma.xor.popc.sync.aligned.row.col.m8n8k128"
                   ".s32.b1.b1.s32"), NULL);
    CHNE(strstr(p, "wmma.load.a.sync.aligned.row.m8n8k128.b1 {%rb1}"), NULL);
    PASS();
}
TH_REG("rpi", 195, "bmma_sync picks the xor popc operation", rpi195)

static void rpi196(void)
{
    int rc = rpi_err(
        "__global__ void k(float *d){\n"
        "  wmma::fragment<wmma::accumulator, 16, 16, 16, float,\n"
        "                 wmma::row_major> c;\n"
        "  d[0] = 1.0f; }\n", "rpi196");

    CHNE(rc, 0);
    CHNE(strstr(obuf, "E912"), NULL);
    PASS();
}
TH_REG("rpi", 196, "an accumulator given a layout refuses", rpi196)

static void rpi197(void)
{
    int rc = wm_err("  wmma::fill_fragment(cf, 1.0f);\n"
                    "  wmma::store_matrix_sync(d, cf, ld,\n"
                    "                          wmma::mem_row_major); }\n",
                    "rpi197");

    CHEQ(rc, 0);
    CHEQ(ecnt(obuf, "store f32 1,"), 8);
    PASS();
}
TH_REG("rpi", 197, "fill_fragment writes every accumulator slot", rpi197)

static void rpi198(void)
{
    const char *p = rpi_ptx(
        "__global__ void k(const half *a, float *d, int l){\n"
        "  wmma::fragment<wmma::matrix_a, 8, 32, 16, half,\n"
        "                 wmma::row_major> af;\n"
        "  wmma::load_matrix_sync(af, a, l); d[0] = 1.0f; }\n", "rpi198");

    CHNE(p, NULL);
    CHNE(strstr(p, "wmma.load.a.sync.aligned.row.m8n32k16.f16 "
                   "{%rb1, %rb2, %rb3, %rb4, %rb5, %rb6, %rb7, %rb8}"), NULL);
    PASS();
}
TH_REG("rpi", 198, "m8n32k16 keeps eight f16 registers each", rpi198)

static void rpi199(void)
{
    int rc = wm_err(
        "  wmma::fill_fragment(cf, 0.0f);\n"
        "  cf.x[0] = 3.0f;\n"
        "  wmma::store_matrix_sync(d, cf, ld,\n"
        "                          wmma::mem_row_major); }\n", "rpi199");

    CHNE(rc, 0);
    CHNE(strstr(obuf, "E910"), NULL);
    CHNE(strstr(obuf, "cf"), NULL);
    PASS();
}
TH_REG("rpi", 199, "a fragment member refuses rather than guess", rpi199)

static const char *tid_pre =
    "template <int I_, typename T> struct tile {\n"
    "  static constexpr int I = I_; static constexpr int J = I_ + 1; };\n"
    "template <int I_> struct tile<I_, double> {\n"
    "  static constexpr int I = I_ * 100; };\n"
    "template <class T> using tb = tile<3, T>;\n";

static int tid_ir(const char *body, const char *stem)
{
    char src[2048];

    if (snprintf(src, sizeof src, "%s%s", tid_pre, body) < 0) return -1;
    return rpi_err(src, stem);
}

static int inseg(const char *beg, const char *end, const char *ndl)
{
    const char *p = beg ? strstr(beg, ndl) : NULL;

    return p != NULL && (end == NULL || p < end);
}

static void rpi231(void)
{
    int rc = rpi_err(
        "struct pl { static constexpr int I = 8; };\n"
        "__global__ void k(int *o){\n"
        "  typedef pl tq;\n"
        "  auto g = [&](int *v){ for (int i = 0; i < tq::I; ++i) v[i] = i; };\n"
        "  g(o); }\n", "rpi231");

    CHEQ(rc, 0);
    CHEQ(strstr(obuf, "E106"), NULL);
    CHNE(strstr(obuf, "icmp slt i32 %3, 8"), NULL);
    PASS();
}
TH_REG("rpi", 231, "a local typedef reaches into a lambda body", rpi231)

static void rpi232(void)
{
    int rc = tid_ir(
        "template <typename T> __global__ void k(int *o){\n"
        "  typedef tile<3, T> tq;\n"
        "  auto g = [&](int *v){ for (int i = 0; i < tq::I; ++i) v[i] = i; };\n"
        "  g(o); }\n"
        "void h(int *o){ k<float><<<1,1>>>(o); }\n", "rpi232");

    CHEQ(rc, 0);
    CHEQ(strstr(obuf, "E106"), NULL);
    CHNE(strstr(obuf, "icmp slt i32 %3, 3"), NULL);
    PASS();
}
TH_REG("rpi", 232, "a tile typedef bounds a loop inside a lambda", rpi232)

static void rpi233(void)
{
    int rc = tid_ir(
        "__global__ void k(int *o){\n"
        "  typedef tile<3, float> tq;\n"
        "  auto g = [&](int *v){ int a[tq::I]; a[2] = 5; v[0] = a[2]; };\n"
        "  g(o); }\n", "rpi233");

    CHEQ(rc, 0);
    CHEQ(strstr(obuf, "E131"), NULL);
    CHNE(strstr(obuf, "alloca ptr<private, [3 x i32]>"), NULL);
    PASS();
}
TH_REG("rpi", 233, "a static member sizes an array in a lambda", rpi233)

static void rpi234(void)
{
    int rc = tid_ir(
        "__global__ void k(int *o){\n"
        "  typedef tile<3, double> tq;\n"
        "  auto g = [&](int *v){\n"
        "    if constexpr (tq::I == 300) v[0] = 7; else v[0] = 9; };\n"
        "  g(o); }\n", "rpi234");

    CHEQ(rc, 0);
    CHEQ(strstr(obuf, "E158"), NULL);
    CHNE(strstr(obuf, "store i32 7,"), NULL);
    CHEQ(strstr(obuf, "store i32 9,"), NULL);
    PASS();
}
TH_REG("rpi", 234, "if constexpr in a lambda reads a static member", rpi234)

static void rpi235(void)
{
    int rc = tid_ir(
        "__global__ void k(int *o){\n"
        "  typedef tile<3, float> tq;\n"
        "  auto g = [&](int *v){ v[0] = tile<tq::I, double>::I; };\n"
        "  g(o); }\n", "rpi235");

    CHEQ(rc, 0);
    CHEQ(strstr(obuf, "E106"), NULL);
    CHNE(strstr(obuf, "store i32 300,"), NULL);
    PASS();
}
TH_REG("rpi", 235, "a static member passes as a template argument", rpi235)

static void rpi236(void)
{
    int rc = tid_ir(
        "__global__ void k(int *o){\n"
        "  auto g = [&](int *v){ v[0] = tb<double>::I; v[1] = tb<float>::I; };\n"
        "  g(o); }\n", "rpi236");

    CHEQ(rc, 0);
    CHEQ(strstr(obuf, "E106"), NULL);
    CHNE(strstr(obuf, "store i32 300,"), NULL);
    CHNE(strstr(obuf, "store i32 3,"), NULL);
    PASS();
}
TH_REG("rpi", 236, "an alias template names a static member", rpi236)

static void rpi237(void)
{
    int rc = tid_ir(
        "__global__ void k(int *o){\n"
        "  auto g = [&](int *v){ v[0] = tile<3, double>::I; };\n"
        "  g(o); }\n", "rpi237");

    CHEQ(rc, 0);
    CHEQ(strstr(obuf, "E106"), NULL);
    CHNE(strstr(obuf, "store i32 300,"), NULL);
    PASS();
}
TH_REG("rpi", 237, "a specialisation named in full inside a lambda", rpi237)

static void rpi238(void)
{
    int rc = tid_ir(
        "template <typename T> __global__ void k(int *o){\n"
        "  typedef tile<3, T> tq;\n"
        "  auto g = [&](int *v){ v[0] = tq::I; };\n"
        "  g(o); }\n"
        "void h(int *o){ k<float><<<1,1>>>(o); k<double><<<1,1>>>(o); }\n",
        "rpi238");

    CHEQ(rc, 0);
    CHEQ(ecnt(obuf, "store i32 3,"), 1);
    CHEQ(ecnt(obuf, "store i32 300,"), 1);
    PASS();
}
TH_REG("rpi", 238, "two specialisations keep their own constants", rpi238)

static void rpi239(void)
{
    char src[2048];
    const char *p, *a, *b;

    snprintf(src, sizeof src, "%s%s", tid_pre,
             "template <typename T> __global__ void ksum(int *o){\n"
             "  typedef tile<3, T> tq;\n"
             "  auto g = [&](int *v){ int s = 0;\n"
             "    for (int i = 0; i < tq::I; ++i) s += i; v[0] = s; };\n"
             "  g(o); }\n"
             "void h(int *o){ ksum<float><<<1,1>>>(o);\n"
             "                ksum<double><<<1,1>>>(o); }\n");
    p = rpi_ptx(src, "rpi239");
    CHNE(p, NULL);
    a = strstr(p, ".entry ksum (");
    b = strstr(p, ".entry ksum$1");
    CHNE(a, NULL);
    CHNE(b, NULL);
    CHEQ(inseg(a, b, "%r3, 3;"), 1);
    CHEQ(inseg(a, b, "%r3, 300;"), 0);
    CHEQ(inseg(b, NULL, "%r3, 300;"), 1);
    CHEQ(inseg(b, NULL, "%r3, 3;"), 0);
    PASS();
}
TH_REG("rpi", 239, "each PTX entry carries its own bound", rpi239)

static void rpi240(void)
{
    int rc = tid_ir(
        "typedef tile<3, float> tq;\n"
        "__global__ void k(int *o){\n"
        "  typedef tile<3, double> tq;\n"
        "  o[0] = tq::I;\n"
        "  auto g = [&](int *v){ v[1] = tq::I; };\n"
        "  g(o); }\n", "rpi240");

    CHEQ(rc, 0);
    CHEQ(ecnt(obuf, "store i32 300,"), 2);
    CHEQ(strstr(obuf, "store i32 3,"), NULL);
    PASS();
}
TH_REG("rpi", 240, "an inner typedef shadows the outer one", rpi240)

static void rpi241(void)
{
    int rc = tid_ir(
        "__global__ void k(int *o){\n"
        "  typedef tile<3, nosuch> tq;\n"
        "  auto g = [&](int *v){ v[0] = tq::I; };\n"
        "  g(o); }\n", "rpi241");

    CHNE(rc, 0);
    CHNE(strstr(obuf, "E960"), NULL);
    CHNE(strstr(obuf, "tq"), NULL);
    CHEQ(strstr(obuf, "store i32 3,"), NULL);
    PASS();
}
TH_REG("rpi", 241, "an unresolved specialisation refuses by name", rpi241)

static void rpi242(void)
{
    const char *p = rpi_ptx(
        "__device__ const char *a(void){ return \"same\"; }\n"
        "__device__ const char *b(void){ return \"same\"; }\n"
        "__global__ void k(const char **o){ o[0] = a(); o[1] = b(); }\n",
        "rpi242");

    CHNE(p, NULL);
    CHEQ(ecnt(p, "= {115, 97, 109, 101, 0}"), 1);
    PASS();
}
TH_REG("rpi", 242, "one global for two identical string literals", rpi242)

static void rpi243(void)
{
    const char *p = rpi_ptx(
        "template <typename T> __device__ const char *w(T){\n"
        "  return __FUNCTION__; }\n"
        "__global__ void k(const char **o){ o[0] = w(1); o[1] = w(1.0f); }\n",
        "rpi243");

    CHNE(p, NULL);
    CHEQ(ecnt(p, "= {119, 0}"), 1);
    PASS();
}
TH_REG("rpi", 243, "__func__ is the source name, not the instance", rpi243)

/* ---- A for-init declaration with more than one declarator ----
 * parse_declaration hangs the extra declarators off the first one's sibling
 * chain, so the for node ended up with five children and the condition slot
 * held the second declarator. */

static void rpi211(void)
{
    int rc = rpi_err(
        "__global__ void k(int *o) {\n"
        "  int s = 0;\n"
        "  for (int i = 0, j = 10; i < 4; ++i) s += i + j;\n"
        "  o[0] = s; }\n", "rpi211");

    CHEQ(rc, 0);
    CHEQ(strstr(obuf, "E106"), NULL);
    CHNE(strstr(obuf, "add i32 %3, 10"), NULL);
    PASS();
}
TH_REG("rpi", 211, "a for init takes two declarators", rpi211)

static void rpi212(void)
{
    int rc = rpi_err(
        "__global__ void k(int *o) {\n"
        "  int s = 0;\n"
        "  for (int a = 1, b = a + 6, c; a < 3; ++a) { c = a * b; s += c; }\n"
        "  o[0] = s; }\n", "rpi212");

    CHEQ(rc, 0);
    CHNE(strstr(obuf, "mul i32 %3, 7"), NULL);
    PASS();
}
TH_REG("rpi", 212, "a for declarator sees the one before it", rpi212)

static void rpi213(void)
{
    int rc = rpi_err(
        "__global__ void k(int *o) {\n"
        "  int s = 0;\n"
        "  for (int m = 0, n = 2, q = 5, r; m < 1; ++m) { r = n * q; s += r; }\n"
        "  o[0] = s; }\n", "rpi213");

    CHEQ(rc, 0);
    CHNE(strstr(obuf, "add i32 %2, 10"), NULL);
    PASS();
}
TH_REG("rpi", 213, "four for declarators, one uninitialised", rpi213)

static void rpi214(void)
{
    int rc = rpi_err(
        "__global__ void k(int *o) {\n"
        "  for (int a = 2, b = 3; a < 6; ++a) o[0] = b;\n"
        "  o[1] = b; }\n", "rpi214");

    CHNE(rc, 0);
    CHNE(strstr(obuf, "E101"), NULL);
    PASS();
}
TH_REG("rpi", 214, "a for init name is gone after the loop", rpi214)

static void rpi215(void)
{
    int rc = rpi_err(
        "__global__ void k(int *o) {\n"
        "  int s = 3;\n"
        "  for (int i = 0, *p = &s; i < 1; ++i) { *p += 100; }\n"
        "  o[0] = s; }\n", "rpi215");

    CHEQ(rc, 0);
    CHNE(strstr(obuf, "add i32 %7, 100"), NULL);
    PASS();
}
TH_REG("rpi", 215, "a for init declares a pointer and an int", rpi215)

/* ---- Packed half2 and bfloat162 conversions ----
 * cvt.rn.f16x2.f32 rounds each lane to nearest even, per PTX ISA 9.2 9.7.9.21,
 * and Booth reaches the same numbers through two per-lane converts. */

static void rpi216(void)
{
    const char *p = rpi_ptx(
        "__global__ void k(const float2 *a, half2 *o) {\n"
        "  o[0] = __float22half2_rn(a[0]); }\n", "rpi216");

    CHNE(p, NULL);
    CHEQ(ecnt(p, "cvt.rn.f16.f32"), 2);
    CHEQ(strstr(p, "cvt.rz."), NULL);
    CHEQ(strstr(p, "cvt.rm."), NULL);
    CHEQ(strstr(p, "cvt.rp."), NULL);
    PASS();
}
TH_REG("rpi", 216, "float22half2_rn rounds each lane to even", rpi216)

static void rpi217(void)
{
    const char *p = rpi_ptx(
        "__global__ void k(const float *f, half2 *o) {\n"
        "  o[0] = __floats2half2_rn(f[0], f[1]); }\n", "rpi217");

    CHNE(p, NULL);
    CHEQ(ecnt(p, "cvt.rn.f16.f32"), 2);
    PASS();
}
TH_REG("rpi", 217, "floats2half2_rn converts both arguments", rpi217)

static void rpi218(void)
{
    const char *p = rpi_ptx(
        "__global__ void k(const __half *h, half2 *o) {\n"
        "  o[0] = __halves2half2(h[0], h[1]); }\n", "rpi218");

    CHNE(p, NULL);
    CHEQ(strstr(p, "cvt."), NULL);
    CHEQ(ecnt(p, "st.local.b16"), 2);
    PASS();
}
TH_REG("rpi", 218, "halves2half2 packs with no conversion", rpi218)

static void rpi219(void)
{
    int rc = rpi_err(
        "__global__ void k(const half2 *h, half2 *o) {\n"
        "  o[0] = __lows2half2(h[0], h[1]); }\n", "rpi219");

    CHEQ(rc, 0);
    CHNE(strstr(obuf, "%5 = gep ptr<global, f16>, %3, 0"), NULL);
    CHNE(strstr(obuf, "%7 = gep ptr<global, f16>, %4, 0"), NULL);
    CHEQ(strstr(obuf, "gep ptr<global, f16>, %3, 1"), NULL);
    PASS();
}
TH_REG("rpi", 219, "lows2half2 takes lane zero of both", rpi219)

static void rpi220(void)
{
    int rc = rpi_err(
        "__global__ void k(const half2 *h, half2 *o) {\n"
        "  o[0] = __highs2half2(h[0], h[1]); }\n", "rpi220");

    CHEQ(rc, 0);
    CHNE(strstr(obuf, "%5 = gep ptr<global, f16>, %3, 1"), NULL);
    CHNE(strstr(obuf, "%7 = gep ptr<global, f16>, %4, 1"), NULL);
    CHEQ(strstr(obuf, "gep ptr<global, f16>, %3, 0"), NULL);
    PASS();
}
TH_REG("rpi", 220, "highs2half2 takes lane one of both", rpi220)

static void rpi221(void)
{
    int rc = rpi_err(
        "__global__ void k(const half2 *h, half2 *o) {\n"
        "  o[0] = __lowhigh2highlow(h[0]); }\n", "rpi221");

    CHEQ(rc, 0);
    CHNE(strstr(obuf, "%4 = gep ptr<global, f16>, %3, 1"), NULL);
    CHNE(strstr(obuf, "%6 = gep ptr<global, f16>, %3, 0"), NULL);
    CHNE(strstr(obuf, "store f16 %5, %9"), NULL);
    CHNE(strstr(obuf, "store f16 %7, %11"), NULL);
    PASS();
}
TH_REG("rpi", 221, "lowhigh2highlow swaps the two lanes", rpi221)

static void rpi222(void)
{
    const char *p = rpi_ptx(
        "__global__ void k(const float2 *a, __nv_bfloat162 *o) {\n"
        "  o[0] = __float22bfloat162_rn(a[0]); }\n", "rpi222");

    CHNE(p, NULL);
    CHEQ(ecnt(p, "cvt.rn.bf16.f32"), 2);
    CHEQ(strstr(p, "cvt.rz."), NULL);
    PASS();
}
TH_REG("rpi", 222, "float22bfloat162_rn rounds to bfloat16", rpi222)

static void rpi223(void)
{
    int rc = rpi_err(
        "__global__ void k(const __nv_bfloat162 *b, float2 *o) {\n"
        "  o[0] = __bfloat1622float2(b[0]); }\n", "rpi223");

    CHEQ(rc, 0);
    CHEQ(ecnt(obuf, "fpext bf16"), 2);
    CHEQ(ecnt(obuf, "store f32"), 4);
    PASS();
}
TH_REG("rpi", 223, "bfloat1622float2 widens both lanes", rpi223)

static void rpi224(void)
{
    int rc = rpi_err(
        "__global__ void k(const __nv_bfloat162 *b, __nv_bfloat16 *o) {\n"
        "  o[0] = __low2bfloat16(b[0]);\n"
        "  o[1] = __high2bfloat16(b[0]); }\n", "rpi224");

    CHEQ(rc, 0);
    CHNE(strstr(obuf, "%4 = gep ptr<global, bf16>, %3, 0"), NULL);
    CHNE(strstr(obuf, "%9 = gep ptr<global, bf16>, %8, 1"), NULL);
    CHEQ(strstr(obuf, "fpext"), NULL);
    PASS();
}
TH_REG("rpi", 224, "low2bfloat16 and high2bfloat16 pick a lane", rpi224)

static void rpi225(void)
{
    int rc = rpi_err(
        "__global__ void k(const float *f, half2 *o) {\n"
        "  o[0] = __float22half2_rn(f[0]); }\n", "rpi225");

    CHNE(rc, 0);
    CHNE(strstr(obuf, "E940"), NULL);
    CHNE(strstr(obuf, "__float22half2_rn"), NULL);
    PASS();
}
TH_REG("rpi", 225, "a packed conversion refuses a scalar", rpi225)

/* ---- A constexpr function template that returns a function pointer ----
 * get_dequantize_V in ggml is a constexpr template whose if-constexpr arms
 * each return a template-id. The value has to name the instantiation the
 * arguments picked, not the base name, or the call lands in the wrong body. */

static const char *CF_HDR =
    "typedef void (*fn_t)(int *, int);\n"
    "template <typename T, int ne>\n"
    "static __device__ void da(int *d, int i){ d[0] = 111 + i + ne; }\n"
    "template <typename T, int ne>\n"
    "static __device__ void db(int *d, int i){ d[0] = 222 + i + ne; }\n"
    "template <int sel, typename T, int ne>\n"
    "constexpr __device__ fn_t getf(){\n"
    "  if constexpr (sel == 0) { return da<T, ne>; }\n"
    "  else { return db<T, ne>; } }\n";

static const char *cf_ptx(const char *tail, const char *stem)
{
    char src[1024];

    if (snprintf(src, sizeof src, "%s%s", CF_HDR, tail) < 0) return NULL;
    return rpi_ptx(src, stem);
}

static void rpi226(void)
{
    const char *p = cf_ptx(
        "__global__ void k(int *o){\n"
        "  constexpr fn_t f = getf<0, float, 4>(); f(o, 1); }\n", "rpi226");

    CHNE(p, NULL);
    CHNE(strstr(p, "mov.u32 %r1, 116;"), NULL);
    CHEQ(strstr(p, "227"), NULL);
    PASS();
}
TH_REG("rpi", 226, "a constexpr template yields its function", rpi226)

static void rpi227(void)
{
    const char *p = cf_ptx(
        "__global__ void k(int *o){\n"
        "  constexpr fn_t f = getf<1, float, 4>(); f(o, 1); }\n", "rpi227");

    CHNE(p, NULL);
    CHNE(strstr(p, "mov.u32 %r1, 227;"), NULL);
    CHEQ(strstr(p, "116"), NULL);
    PASS();
}
TH_REG("rpi", 227, "the arm not taken leaves no call", rpi227)

static void rpi228(void)
{
    const char *p = cf_ptx(
        "__global__ void ka(int *o){\n"
        "  constexpr fn_t f = getf<0, float, 4>(); f(o, 1); }\n"
        "__global__ void kb(int *o){\n"
        "  constexpr fn_t f = getf<0, float, 9>(); f(o, 1); }\n", "rpi228");

    CHNE(p, NULL);
    CHNE(strstr(p, "mov.u32 %r1, 116;"), NULL);
    CHNE(strstr(p, "mov.u32 %r1, 121;"), NULL);
    PASS();
}
TH_REG("rpi", 228, "template arguments key the instantiation", rpi228)


static const char *CX_HDR =
    "typedef void (*fnp_t)(float *);\n"
    "template <int N> static __device__ void bodya(float *p)"
    " { p[0] = 1111.0f; }\n"
    "template <int N> static __device__ void bodyb(float *p)"
    " { p[0] = 2222.0f; }\n";

static const char *cx_ptx(const char *tail, const char *stem)
{
    static char src[2048];

    if (snprintf(src, sizeof src, "%s%s", CX_HDR, tail) < 0) return NULL;
    return rpi_ptx(src, stem);
}

static const char *CX_SW =
    "template <int K, int N>\n"
    "static constexpr __device__ fnp_t getfn() {\n"
    "  switch (K) { case 0: return bodya<N>; default: return bodyb<N>; } }\n"
    "template <int K, int N>\n"
    "static __global__ void run(float *p) {\n"
    "  constexpr fnp_t f = getfn<K, N>();\n"
    "  f(p); }\n";

static void rpi251(void)
{
    static char tail[1024];
    const char *p;

    snprintf(tail, sizeof tail, "%s%s", CX_SW,
             "void h(float *p){ run<0, 3><<<1,1>>>(p); }\n");
    p = cx_ptx(tail, "rpi251");
    CHNE(p, NULL);
    CHNE(strstr(p, "0f448AE000"), NULL);
    CHEQ(strstr(p, "0f450AE000"), NULL);
    PASS();
}
TH_REG("rpi", 251, "a constexpr template call picks a body", rpi251)

static void rpi252(void)
{
    static char tail[1024];
    const char *p;

    snprintf(tail, sizeof tail, "%s%s", CX_SW,
             "void h(float *p){ run<1, 3><<<1,1>>>(p); }\n");
    p = cx_ptx(tail, "rpi252");
    CHNE(p, NULL);
    CHNE(strstr(p, "0f450AE000"), NULL);
    CHEQ(strstr(p, "0f448AE000"), NULL);
    PASS();
}
TH_REG("rpi", 252, "the other argument picks the other body", rpi252)

static const char *CX_AGG =
    "struct pk_t { int vdr; fnp_t fn;\n"
    "  constexpr __device__ pk_t(int v, fnp_t f) : vdr(v), fn(f) {} };\n"
    "template <int K, int N> static constexpr __device__ pk_t pick() {\n"
    "  switch (K) { case 0: return pk_t(1, bodya<N>);\n"
    "               default: return pk_t(2, bodyb<N>); } }\n"
    "template <int K, int N> static constexpr __device__ fnp_t getfn()"
    " { return pick<K, N>().fn; }\n"
    "template <int K, int N>\n"
    "static __global__ void run(float *p) {\n"
    "  constexpr fnp_t f = getfn<K, N>();\n"
    "  f(p); }\n";

static void rpi253(void)
{
    static char tail[1024];
    const char *p;

    snprintf(tail, sizeof tail, "%s%s", CX_AGG,
             "void h(float *p){ run<0, 3><<<1,1>>>(p); }\n");
    p = cx_ptx(tail, "rpi253");
    CHNE(p, NULL);
    CHNE(strstr(p, "0f448AE000"), NULL);
    CHEQ(strstr(p, "0f450AE000"), NULL);
    PASS();
}
TH_REG("rpi", 253, "a constexpr aggregate carries a function", rpi253)

static void rpi254(void)
{
    static char tail[1024];
    const char *p;

    snprintf(tail, sizeof tail, "%s%s", CX_AGG,
             "void h(float *p){ run<1, 3><<<1,1>>>(p); }\n");
    p = cx_ptx(tail, "rpi254");
    CHNE(p, NULL);
    CHNE(strstr(p, "0f450AE000"), NULL);
    CHEQ(strstr(p, "0f448AE000"), NULL);
    PASS();
}
TH_REG("rpi", 254, "the aggregate arm gives the other body", rpi254)

static void rpi255(void)
{
    const char *p = rpi_ptx(
        "__global__ void k(int *d){\n"
        "  constexpr int sel = 2;\n"
        "  switch (sel) {\n"
        "    case 1: d[0] = 11; break;\n"
        "    case 2: d[0] = 22; break;\n"
        "    default: d[0] = 33; break; } }\n", "rpi255");

    CHNE(p, NULL);
    CHNE(strstr(p, "22;"), NULL);
    CHEQ(strstr(p, "11;"), NULL);
    CHEQ(strstr(p, "33;"), NULL);
    CHEQ(strstr(p, "bra"), NULL);
    PASS();
}
TH_REG("rpi", 255, "a switch on a constant lowers one arm", rpi255)

static const char *gvsrc(void)
{
    static char src[1 << 14];
    int n = 0;

    for (int i = 0; i < 300; i++) {
        int w = snprintf(src + n, sizeof src - (size_t)n,
                         "__device__ int gv%d = %d;\n", i, i);
        if (w < 0 || (size_t)(n + w) >= sizeof src) return NULL;
        n += w;
    }
    if (snprintf(src + n, sizeof src - (size_t)n,
                 "__global__ void k(int *d){ d[0] = gv299 + gv0; }\n") < 0)
        return NULL;
    return src;
}

static void rpi256(void)
{
    const char *s = gvsrc();
    const char *p;

    CHNE(s, NULL);
    p = rpi_ptx(s, "rpi256");
    CHNE(p, NULL);
    CHEQ(strstr(obuf, "E127"), NULL);
    CHNE(strstr(p, "mov.u64 %rd3, gv299;"), NULL);
    CHNE(strstr(p, "mov.u64 %rd4, gv0;"), NULL);
    PASS();
}
TH_REG("rpi", 256, "a global past 255 is named in the PTX", rpi256)

static void rpi257(void)
{
    const char *s = gvsrc();
    char cmd[512];
    const char *path;

    CHNE(s, NULL);
    path = scratch("rpi257.cu", s);
    CHNE(path, NULL);
    snprintf(cmd, sizeof cmd, "%s --ir %s", BC_BIN, path);
    CHEQ(th_run(cmd, obuf, (int)sizeof obuf), 0);
    CHNE(strstr(obuf, "@gv299"), NULL);
    CHEQ(strstr(obuf, "@?"), NULL);
    PASS();
}
TH_REG("rpi", 257, "the IR printer names a global past 255", rpi257)

static int ehas(const char *ptx, const char *ent, const char *ndl)
{
    const char *p = strstr(ptx, ent);
    const char *e, *h;

    if (!p) return -1;
    e = strstr(p, "\n}");
    if (!e) return -1;
    h = strstr(p, ndl);
    return h != NULL && h < e;
}

static void rpi258(void)
{
    const char *p = rpi_ptx(
        "template <int K> __device__ int lmv(int *o) {\n"
        "    int acc = 0;\n"
        "    auto f = [&](int i) { acc += i * K + 40960 + K; };\n"
        "    f(1);\n"
        "    o[0] = acc;\n"
        "    return acc;\n"
        "}\n"
        "__global__ void kx(int *o) { lmv<11>(o); }\n"
        "__global__ void ky(int *o) { lmv<29>(o); }\n", "rpi258");

    CHNE(p, NULL);
    CHEQ(strstr(obuf, "E030"), NULL);
    CHEQ(ehas(p, ".entry kx", "40982"), 1);
    CHEQ(ehas(p, ".entry kx", "41018"), 0);
    CHEQ(ehas(p, ".entry ky", "41018"), 1);
    CHEQ(ehas(p, ".entry ky", "40982"), 0);
    PASS();
}
TH_REG("rpi", 258, "each instance keeps its own lambda body", rpi258)

static void rpi259(void)
{
    const char *p = rpi_ptx(
        "template <bool W> __device__ int gat(int *o, int x) {\n"
        "    int acc = 0;\n"
        "    if constexpr (W) {\n"
        "        auto g = [&](int i) { acc = i * 100 + 51001; };\n"
        "        g(x);\n"
        "    } else {\n"
        "        auto g = [&](int i) { acc = i * 100 + 52002; };\n"
        "        g(x);\n"
        "    }\n"
        "    o[0] = acc;\n"
        "    return acc;\n"
        "}\n"
        "__global__ void kt(int *o, int x) { gat<true>(o, x); }\n"
        "__global__ void kf(int *o, int x) { gat<false>(o, x); }\n",
        "rpi259");

    CHNE(p, NULL);
    CHEQ(ehas(p, ".entry kt", "51001"), 1);
    CHEQ(ehas(p, ".entry kt", "52002"), 0);
    CHEQ(ehas(p, ".entry kf", "52002"), 1);
    CHEQ(ehas(p, ".entry kf", "51001"), 0);
    PASS();
}
TH_REG("rpi", 259, "a discarded arm lowers no lambda", rpi259)

static void rpi260(void)
{
    const char *p = rpi_ptx(
        "struct pair2 { int a; int b; };\n"
        "__device__ pair2 mkp(int x, int y) { return {x + 1, y + 2}; }\n"
        "__device__ int scl(int x) { return {x * 3}; }\n"
        "__global__ void kp(int *o, int x) {\n"
        "    pair2 p = mkp(x, x);\n"
        "    o[0] = p.a * 10 + p.b + scl(x);\n"
        "}\n", "rpi260");

    CHNE(p, NULL);
    CHEQ(strstr(obuf, "E106"), NULL);
    CHEQ(ehas(p, ".entry kp", "add.u32 %r2, %r1, 1;"), 1);
    CHEQ(ehas(p, ".entry kp", "add.u32 %r3, %r1, 2;"), 1);
    CHEQ(ehas(p, ".entry kp", "mul.lo.u32 %r4, %r2, 10;"), 1);
    CHEQ(ehas(p, ".entry kp", "mul.lo.u32 %r6, %r1, 3;"), 1);
    PASS();
}
TH_REG("rpi", 260, "a braced return fills an aggregate", rpi260)

static void rpi261(void)
{
    static const char *const src =
        "struct p2 { int a; int b; };\n"
        "struct q2 { p2 u; p2 v; };\n"
        "__device__ p2 zp(void) { return {}; }\n"
        "__device__ q2 nq(int x) { return {{x, x + 1}, {x + 2, x + 3}}; }\n"
        "__global__ void kz(int *o, int x) {\n"
        "    p2 z = zp();\n"
        "    q2 n = nq(x);\n"
        "    o[0] = z.a + z.b + n.u.a + n.u.b * 2 + n.v.a * 4 + n.v.b * 8;\n"
        "}\n";
    const char *p = rpi_ptx(src, "rpi261");

    CHNE(p, NULL);
    CHEQ(strstr(obuf, "E106"), NULL);
    CHEQ(ehas(p, ".entry kz", "add.u32 %r6, 0, %r5;"), 1);
    CHEQ(ehas(p, ".entry kz", "mul.lo.u32 %r14, %r13, 8;"), 1);
    PASS();
}
TH_REG("rpi", 261, "a braced return nests and zero-fills", rpi261)

static void rpi262(void)
{
    static const char *const src =
        "struct p2 { int a; int b; };\n"
        "__device__ p2 bad(int x) { return {x, x, x}; }\n"
        "__global__ void kb(int *o, int x) { p2 p = bad(x); o[0] = p.a; }\n";
    const char *path = scratch("rpi262.cu", src);
    char cmd[512];

    CHNE(path, NULL);
    snprintf(cmd, sizeof cmd, "%s --ir %s", BC_BIN, path);
    CHNE(th_run(cmd, obuf, (int)sizeof obuf), 0);
    CHNE(strstr(obuf, "E660"), NULL);
    CHNE(strstr(obuf, "a return value"), NULL);
    PASS();
}
TH_REG("rpi", 262, "a braced return past the fields refuses", rpi262)

static const char *lamsrc(int n)
{
    static char src[1 << 17];
    int at = snprintf(src, sizeof src,
                      "__global__ void k(int *d){ int s = 0;\n");

    if (at < 0) return NULL;
    for (int i = 0; i < n; i++) {
        int w = snprintf(src + at, sizeof src - (size_t)at,
                         "{auto f=[&](int i){s+=i+%d;};f(1);}\n", i);
        if (w < 0 || (size_t)(at + w) >= sizeof src) return NULL;
        at += w;
    }
    if (snprintf(src + at, sizeof src - (size_t)at, "d[0] = s;\n}\n") < 0)
        return NULL;
    return src;
}

static void rpi263(void)
{
    const char *s = lamsrc(1020);
    char cmd[512];
    const char *path;

    CHNE(s, NULL);
    path = scratch("rpi263a.cu", s);
    CHNE(path, NULL);
    snprintf(cmd, sizeof cmd, "%s --ir %s", BC_BIN, path);
    CHEQ(th_run(cmd, obuf, (int)sizeof obuf), 0);

    s = lamsrc(1030);
    CHNE(s, NULL);
    path = scratch("rpi263b.cu", s);
    CHNE(path, NULL);
    snprintf(cmd, sizeof cmd, "%s --ir %s", BC_BIN, path);
    CHNE(th_run(cmd, obuf, (int)sizeof obuf), 0);
    CHNE(strstr(obuf, "more lambdas than Booth lowers"), NULL);
    PASS();
}
TH_REG("rpi", 263, "the lambda table refuses by name when full", rpi263)
