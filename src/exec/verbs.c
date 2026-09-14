/* verbs.c -- kath's front door: run, build and doctor. Each dispatches to the
 * flag machinery kath_compile already drives, and run through the rt_* launcher. */

#include "front.h"
#include "exec.h"
#include "booth_run.h"
#include "backend.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
#define DEVNULL "nul"
#else
#include <sys/wait.h>
#define DEVNULL "/dev/null"
#endif

#define MAXPASS 24
#define PATHN   512

static const char VADD_SRC[] =
    "__global__ void vadd(int *out, int *a, int *b)\n"
    "{\n"
    "    int i = threadIdx.x;\n"
    "    out[i] = a[i] + b[i];\n"
    "}\n";

/* ---- small helpers ---- */

static int streq(const char *a, const char *b) { return a && b && strcmp(a, b) == 0; }

static const char *ext_of(const char *path)
{
    const char *dot = strrchr(path, '.');
    const char *slash = strrchr(path, '/');
    const char *bslash = strrchr(path, '\\');
    const char *sep = slash > bslash ? slash : bslash;
    if (dot == NULL || (sep != NULL && dot < sep)) return "";
    return dot + 1;
}

static int ok_status(int rc)
{
#ifdef _WIN32
    return rc == 0;
#else
    return rc != -1 && WIFEXITED(rc) && WEXITSTATUS(rc) == 0;
#endif
}

static int have_cmd(const char *tool)
{
    char cmd[PATHN];
    snprintf(cmd, sizeof cmd, "%s --version >%s 2>&1", tool, DEVNULL);
    return ok_status(system(cmd));
}

static const char *tmpdir(void)
{
    const char *t = getenv("TEMP");
    if (t == NULL) t = getenv("TMP");
    if (t == NULL) t = getenv("TMPDIR");
#ifndef _WIN32
    if (t == NULL) t = "/tmp";
#endif
    if (t == NULL) t = ".";
    return t;
}

static unsigned g_seq;

static void tmppath(char *buf, size_t n, const char *base, const char *ext)
{
    snprintf(buf, n, "%s/kath_%s_%u.%s", tmpdir(), base, ++g_seq, ext);
}

/* target name -> the backend flag that selects it and the artefact extension. */
static const char *t_flag(const char *target)
{
    if (streq(target, "cpu"))   return "--cpu";
    if (streq(target, "nvptx")) return "--nvidia-ptx";
    return NULL;
}
static const char *t_ext(const char *target)
{
    if (streq(target, "cpu"))   return "o";
    if (streq(target, "nvptx")) return "ptx";
    return "out";
}
static const char *infer_target(const char *file)
{
    const char *e = ext_of(file);
    if (streq(e, "ptx")) return "nvptx";
    if (streq(e, "o"))   return "cpu";
    return NULL;
}

static int is_source(const char *file)
{
    const char *e = ext_of(file);
    return streq(e, "cu") || streq(e, "hip") || streq(e, "py")
        || streq(e, "mlir") || streq(e, "bir") || streq(e, "ml")
        || streq(e, "f90");
}

static int is_spec(const char *a)
{
    return strncmp(a, "in:", 3) == 0 || strncmp(a, "out:", 4) == 0
        || strncmp(a, "io:", 3) == 0 || strncmp(a, "u32:", 4) == 0
        || strncmp(a, "i32:", 4) == 0 || strncmp(a, "f32:", 4) == 0;
}

static const char *pick_target(void)
{
    for (uint32_t i = 0; i < RT_MAX && rt_list[i] != NULL; i++) {
        if (streq(rt_list[i]->name, "cpu")) continue;
        rt_dev_t d;
        if (rt_open(&d, rt_list[i]->name) == RT_OK) {
            rt_shut(&d);
            return rt_list[i]->name;
        }
    }
    return "cpu";
}

/* ---- frontend chains ---- */

static int cvt_f90(const char *file, char *out, size_t n)
{
    if (!have_cmd("lfortran")) {
        fprintf(stderr, "kath: Fortran needs LFortran, which is not on PATH\n");
        return 1;
    }
    char obj[PATHN], side[PATHN], cmd[PATHN * 2];
    tmppath(obj, sizeof obj, "lf", "o");
    snprintf(cmd, sizeof cmd, "lfortran --gpu=cuda -c \"%s\" -o \"%s\"", file, obj);
    if (!ok_status(system(cmd))) {
        fprintf(stderr, "kath: LFortran failed on %s\n", file);
        return 1;
    }
    snprintf(side, sizeof side, "%s.cuda.cu", obj);
    FILE *f = fopen(side, "rb");
    if (f == NULL) { fprintf(stderr, "kath: LFortran wrote no CUDA for %s\n", file); return 1; }
    if (fseek(f, 0, SEEK_END) != 0) { fclose(f); return 1; }
    long sz = ftell(f); rewind(f);
    if (sz <= 0) { fclose(f); return 1; }
    char *buf = (char *)malloc((size_t)sz + 1);
    if (buf == NULL) { fclose(f); return 1; }
    size_t rd = fread(buf, 1, (size_t)sz, f); buf[rd] = '\0'; fclose(f);
    char *cut = strstr(buf, "Auto-generated kernel registration");
    size_t keep = cut ? (size_t)(cut - buf) : rd;
    while (keep > 0 && buf[keep - 1] != '\n') keep--;

    tmppath(out, n, "lf", "cu");
    FILE *o = fopen(out, "wb");
    if (o == NULL) { free(buf); return 1; }
    fwrite(buf, 1, keep, o);
    int good = fclose(o) == 0;
    free(buf);
    remove(obj); remove(side);
    return good ? 0 : 1;
}

static int cvt_ml(const char *file, char *out, size_t n)
{
    if (!have_cmd("dune") && !ok_status(system("opam exec -- dune --version >" DEVNULL " 2>&1"))) {
        fprintf(stderr, "kath: OCaml needs dune, which is not available\n");
        return 1;
    }
    const char *e = ext_of(file);
    const char *b = strrchr(file, '/'); const char *b2 = strrchr(file, '\\');
    const char *base = b > b2 ? b : b2; base = base ? base + 1 : file;
    char stem[PATHN]; size_t bl = strlen(base) - (streq(e, "ml") ? 3u : 0u);
    if (bl >= sizeof stem) bl = sizeof stem - 1;
    memcpy(stem, base, bl); stem[bl] = '\0';

    char cmd[PATHN * 2];
    snprintf(cmd, sizeof cmd, "opam exec -- dune build --root src/ocaml >" DEVNULL " 2>&1");
    if (!ok_status(system(cmd))) { fprintf(stderr, "kath: dune build failed\n"); return 1; }

    tmppath(out, n, "ml", "bir");
    char kc[PATHN], cmt[PATHN];
    snprintf(kc, sizeof kc, "src/ocaml/_build/default/kcomp.exe");
    snprintf(cmt, sizeof cmt,
             "src/ocaml/_build/default/.kernels.objs/byte/%s.cmt", stem);
#ifdef _WIN32
    for (char *p = kc; *p; p++) if (*p == '/') *p = '\\';
    for (char *p = cmt; *p; p++) if (*p == '/') *p = '\\';
#endif
    snprintf(cmd, sizeof cmd, "%s %s -o \"%s\" >" DEVNULL, kc, cmt, out);
    if (!ok_status(system(cmd))) { fprintf(stderr, "kath: kcomp failed on %s\n", file); return 1; }
    return 0;
}

/* ---- compile through kath_compile ---- */

static int compile3(const char *ffe, const char *bflag, const char *file,
                    const char *out, kath_out_t *ko)
{
    char *av[8]; int n = 0;
    av[n++] = (char *)"kath";
    if (ffe)   av[n++] = (char *)ffe;
    if (bflag) av[n++] = (char *)bflag;
    av[n++] = (char *)file;
    av[n++] = (char *)"-o";
    av[n++] = (char *)out;
    av[n] = NULL;
    return kath_compile(n, av, ko);
}

static int build_source(const char *file, const char *bflag,
                        const char *out, kath_out_t *ko)
{
    const char *e = ext_of(file);
    if (streq(e, "f90")) {
        char cu[PATHN]; if (cvt_f90(file, cu, sizeof cu) != 0) return 1;
        int rc = compile3(NULL, bflag, cu, out, ko); remove(cu); return rc;
    }
    if (streq(e, "ml")) {
        char bir[PATHN]; if (cvt_ml(file, bir, sizeof bir) != 0) return 1;
        int rc = compile3("--bir-in", bflag, bir, out, ko); remove(bir); return rc;
    }
    const char *ffe = NULL;
    if (streq(e, "py"))   ffe = "--triton";
    else if (streq(e, "mlir")) ffe = "--mlir";
    else if (streq(e, "bir"))  ffe = "--bir-in";
    else if (streq(e, "hip"))  ffe = "--hip";
    return compile3(ffe, bflag, file, out, ko);
}

/* ---- run ---- */

static int verb_run(int argc, char **argv)
{
    const char *file = NULL, *kernel = NULL, *target = NULL;
    const char *specs[32]; int nspec = 0;
    rt_dim_t dim; memset(&dim, 0, sizeof dim);
    dim.grid[0] = dim.grid[1] = dim.grid[2] = 1u;
    dim.block[0] = dim.block[1] = dim.block[2] = 1u;

    for (int i = 2; i < argc; i++) {
        const char *a = argv[i];
        if (streq(a, "--cpu"))              target = "cpu";
        else if (streq(a, "--nvidia-ptx"))  target = "nvptx";
        else if (streq(a, "--target") && i + 1 < argc) target = argv[++i];
        else if (streq(a, "--grid") && i + 1 < argc) {
            if (!brun_dims(argv[++i], dim.grid)) { fprintf(stderr, "kath run: bad --grid\n"); return 2; }
        } else if (streq(a, "--block") && i + 1 < argc) {
            if (!brun_dims(argv[++i], dim.block)) { fprintf(stderr, "kath run: bad --block\n"); return 2; }
        } else if (streq(a, "--shmem") && i + 1 < argc) {
            dim.shmem = (uint32_t)strtoul(argv[++i], NULL, 0);
        } else if (is_spec(a)) {
            if (nspec < 32) specs[nspec++] = a;
        } else if (a[0] != '-') {
            if (file == NULL) file = a; else if (kernel == NULL) kernel = a;
        } else {
            fprintf(stderr, "kath run: unknown option %s\n", a); return 2;
        }
    }
    if (file == NULL) {
        fprintf(stderr, "usage: kath run <file> [--cpu|--nvidia-ptx] "
                        "[--grid G] [--block B] in:.. out:..:N ...\n");
        return 2;
    }

    char art[PATHN]; char tmp[PATHN]; tmp[0] = '\0';
    const char *use_target;

    if (is_source(file)) {
        if (target == NULL) target = pick_target();
        const char *bflag = t_flag(target);
        if (bflag == NULL) { fprintf(stderr, "kath run: no run target '%s'\n", target); return 2; }
        tmppath(tmp, sizeof tmp, "run", t_ext(target));
        kath_out_t ko; memset(&ko, 0, sizeof ko);
        if (build_source(file, bflag, tmp, &ko) != 0) {
            fprintf(stderr, "kath run: build failed\n"); remove(tmp); return 1;
        }
        if (kernel == NULL) {
            if (!ko.have) { fprintf(stderr, "kath run: no __global__ kernel in %s\n", file); remove(tmp); return 1; }
            kernel = ko.kernel;
        }
        snprintf(art, sizeof art, "%s", tmp);
        use_target = target;
    } else {
        if (target == NULL) target = infer_target(file);
        if (target == NULL) { fprintf(stderr, "kath run: cannot tell target for %s; pass --target\n", file); return 2; }
        if (kernel == NULL) { fprintf(stderr, "kath run: name the kernel: kath run %s <kernel> ...\n", file); return 2; }
        snprintf(art, sizeof art, "%s", file);
        use_target = target;
    }

    brun_reset();
    for (int i = 0; i < nspec; i++) {
        if (!brun_addarg(specs[i])) {
            fprintf(stderr, "kath run: bad argument '%s'\n", specs[i]);
            if (tmp[0]) remove(tmp);
            return 2;
        }
    }
    int rc = brun_launch(art, kernel, use_target, &dim);
    if (tmp[0]) remove(tmp);
    return rc;
}

/* ---- build ---- */

static int is_backend_flag(const char *a, const char **ext)
{
    struct { const char *f, *e; } t[] = {
        { "--cpu", "o" }, { "--rv64", "o" },
        { "--nvidia-ptx", "ptx" }, { "--nvidia-sass", "sass" },
        { "--nvidia-cubin", "cubin" },
        { "--amdgpu", "s" }, { "--amdgpu-bin", "hsaco" },
        { "--tensix", "cpp" }, { "--rv-elf", "elf" },
        { "--metal", "metal" }, { "--intel-spirv", "spv" }
    };
    for (unsigned i = 0; i < sizeof t / sizeof t[0]; i++)
        if (streq(a, t[i].f)) { if (ext) *ext = t[i].e; return 1; }
    return 0;
}

static void repl_ext(char *buf, size_t n, const char *file, const char *ext)
{
    const char *dot = strrchr(file, '.');
    const char *slash = strrchr(file, '/'); const char *bslash = strrchr(file, '\\');
    const char *sep = slash > bslash ? slash : bslash;
    size_t keep = (dot && (sep == NULL || dot > sep)) ? (size_t)(dot - file) : strlen(file);
    if (keep >= n) keep = n - 1;
    memcpy(buf, file, keep); buf[keep] = '\0';
    size_t l = strlen(buf);
    snprintf(buf + l, n - l, ".%s", ext);
}

static int verb_build(int argc, char **argv)
{
    const char *file = NULL, *out = NULL, *bflag = NULL, *ext = "out";
    const char *pass[MAXPASS]; int npass = 0;

    for (int i = 2; i < argc; i++) {
        const char *a = argv[i];
        if (streq(a, "-o") && i + 1 < argc) { out = argv[++i]; continue; }
        if (a[0] != '-') { if (file == NULL) file = a; continue; }
        const char *e = NULL;
        if (is_backend_flag(a, &e)) { bflag = a; ext = e; }
        if (npass < MAXPASS) pass[npass++] = a;
    }
    if (file == NULL) {
        fprintf(stderr, "usage: kath build <file> [-o out] [--cpu|--nvidia-ptx|...]\n");
        return 2;
    }

    if (bflag == NULL) {
        const char *target = pick_target();
        bflag = t_flag(target);
        ext = t_ext(target);
        if (npass < MAXPASS) pass[npass++] = bflag;
    }

    char outbuf[PATHN];
    if (out == NULL) { repl_ext(outbuf, sizeof outbuf, file, ext); out = outbuf; }

    const char *e = ext_of(file);
    char conv[PATHN]; const char *src = file; const char *ffe = NULL;
    if (streq(e, "f90")) { if (cvt_f90(file, conv, sizeof conv) != 0) return 1; src = conv; }
    else if (streq(e, "ml")) { if (cvt_ml(file, conv, sizeof conv) != 0) return 1; src = conv; ffe = "--bir-in"; }
    else if (streq(e, "py"))   ffe = "--triton";
    else if (streq(e, "mlir")) ffe = "--mlir";
    else if (streq(e, "bir"))  ffe = "--bir-in";

    char *av[MAXPASS + 8]; int n = 0;
    av[n++] = (char *)"kath";
    if (ffe) av[n++] = (char *)ffe;
    for (int i = 0; i < npass; i++) av[n++] = (char *)pass[i];
    av[n++] = (char *)src;
    av[n++] = (char *)"-o";
    av[n++] = (char *)out;
    av[n] = NULL;

    int rc = kath_compile(n, av, NULL);
    if (src == conv) remove(conv);
    if (rc == 0) printf("kath build: wrote %s\n", out);
    return rc;
}

/* ---- doctor ---- */

static void report_cmd(const char *label, const char *tool)
{
    printf("  %-22s %s\n", label, have_cmd(tool) ? "found" : "absent");
}

static int wr_i32(const char *path, const int32_t *v, int count)
{
    FILE *f = fopen(path, "wb");
    if (f == NULL) return 0;
    size_t w = fwrite(v, sizeof(int32_t), (size_t)count, f);
    return fclose(f) == 0 && w == (size_t)count;
}

static int rd_i32(const char *path, int32_t *v, int count)
{
    FILE *f = fopen(path, "rb");
    if (f == NULL) return 0;
    size_t r = fread(v, sizeof(int32_t), (size_t)count, f);
    fclose(f);
    return r == (size_t)count;
}

static int selftest(const char *target, const char *bflag, const char *ext)
{
    enum { N = 8 };
    char csrc[PATHN], art[PATHN], ap[PATHN], bp[PATHN], cp[PATHN];
    tmppath(csrc, sizeof csrc, "vadd", "cu");
    tmppath(art, sizeof art, "vadd", ext);
    tmppath(ap, sizeof ap, "va", "bin");
    tmppath(bp, sizeof bp, "vb", "bin");
    tmppath(cp, sizeof cp, "vc", "bin");

    int32_t a[N], b[N], c[N], want[N];
    for (int i = 0; i < N; i++) { a[i] = i + 1; b[i] = 10 * (i + 1); want[i] = a[i] + b[i]; }

    FILE *sf = fopen(csrc, "wb");
    if (sf == NULL) { printf("  %-8s FAIL (temp)\n", target); return 1; }
    fwrite(VADD_SRC, 1, sizeof VADD_SRC - 1, sf); fclose(sf);

    int rc = 1;
    kath_out_t ko; memset(&ko, 0, sizeof ko);
    if (wr_i32(ap, a, N) && wr_i32(bp, b, N)
        && build_source(csrc, bflag, art, &ko) == 0 && ko.have) {
        rt_dim_t dim; memset(&dim, 0, sizeof dim);
        dim.grid[0] = dim.grid[1] = dim.grid[2] = 1u;
        dim.block[0] = (uint32_t)N; dim.block[1] = dim.block[2] = 1u;
        char outspec[PATHN];
        snprintf(outspec, sizeof outspec, "out:%s:%u", cp, (unsigned)(N * sizeof(int32_t)));
        char inA[PATHN], inB[PATHN];
        snprintf(inA, sizeof inA, "in:%s", ap);
        snprintf(inB, sizeof inB, "in:%s", bp);
        brun_reset();
        if (brun_addarg(outspec) && brun_addarg(inA) && brun_addarg(inB)
            && brun_launch(art, ko.kernel, target, &dim) == 0
            && rd_i32(cp, c, N)) {
            int good = 1;
            for (int i = 0; i < N; i++) if (c[i] != want[i]) good = 0;
            rc = good ? 0 : 1;
        }
    }
    printf("  %-8s %s\n", target, rc == 0 ? "PASS" : "FAIL");
    remove(csrc); remove(art); remove(ap); remove(bp); remove(cp);
    return rc;
}

static int verb_doctor(void)
{
    printf("kath doctor\n\ntoolchain\n");
    report_cmd("C compiler (gcc)", "gcc");
    report_cmd("C compiler (cc)", "cc");
    report_cmd("clang", "clang");
    report_cmd("opam", "opam");
    report_cmd("dune", "dune");
    report_cmd("ocaml", "ocaml");
    report_cmd("lfortran", "lfortran");

    printf("\ndevices\n");
    for (uint32_t i = 0; i < RT_MAX && rt_list[i] != NULL; i++) {
        rt_dev_t d;
        int rc = rt_open(&d, rt_list[i]->name);
        if (rc == RT_OK) { printf("  %-8s ready\n", rt_list[i]->name); rt_shut(&d); }
        else printf("  %-8s unavailable (%s)\n", rt_list[i]->name, rt_errs(&d, rc));
    }

    printf("\nself-test (vadd)\n");
    selftest("cpu", "--cpu", "o");
    for (uint32_t i = 0; i < RT_MAX && rt_list[i] != NULL; i++) {
        if (streq(rt_list[i]->name, "cpu")) continue;
        rt_dev_t d;
        if (rt_open(&d, rt_list[i]->name) != RT_OK) continue;
        rt_shut(&d);
        selftest(rt_list[i]->name, t_flag(rt_list[i]->name), t_ext(rt_list[i]->name));
    }
    return 0;
}

/* ---- dispatch ---- */

int booth_is_verb(const char *s)
{
    return streq(s, "run") || streq(s, "build") || streq(s, "doctor");
}

int booth_verb(int argc, char *argv[])
{
    if (streq(argv[1], "run"))    return verb_run(argc, argv);
    if (streq(argv[1], "build"))  return verb_build(argc, argv);
    if (streq(argv[1], "doctor")) return verb_doctor();
    return 2;
}
