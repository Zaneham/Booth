#include "tharns.h"

static char obuf[1 << 14];
static char ptx[1 << 16];

static const char *mfrun(const char *src)
{
    char cmd[512];
    FILE *f = fopen("build/tmof.cu", "w");
    FILE *g;
    size_t n;

    ptx[0] = '\0';
    if (!f) return NULL;
    fputs(src, f);
    fclose(f);
    remove("build/tmof.ptx");
    snprintf(cmd, sizeof cmd, "%s --nvidia-ptx build/tmof.cu -o build/tmof.ptx",
             BC_BIN);
    if (th_run(cmd, obuf, (int)sizeof obuf) != 0) return NULL;
    g = fopen("build/tmof.ptx", "rb");
    if (!g) return NULL;
    n = fread(ptx, 1, sizeof ptx - 1, g);
    fclose(g);
    ptx[n] = '\0';
    return ptx;
}

static const char *kbody(const char *name)
{
    char pat[64];

    snprintf(pat, sizeof pat, ".entry %s ", name);
    return ptx[0] ? strstr(ptx, pat) : NULL;
}

static int hasadd(const char *k, const char *base, int off)
{
    char pat[96];
    const char *end;

    if (!k) return 0;
    end = strstr(k, "\n}");
    snprintf(pat, sizeof pat, ", %s, %d;", base, off);
    k = strstr(k, pat);
    return k != NULL && (end == NULL || k < end);
}

static void mof01(void)
{
    const char *k = mfrun(
        "struct v { char c; double d; int n; };\n"
        "__global__ void k(struct v *s, int *o){\n"
        "  o[0] = (int)s->c; o[1] = (int)s->d; o[2] = s->n; }\n")
        ? kbody("k") : NULL;

    CHECK(k != NULL);
    CHECK(hasadd(k, "%rd1", 0));
    CHECK(hasadd(k, "%rd1", 8));
    CHECK(hasadd(k, "%rd1", 16));
    PASS();
}
TH_REG("mof", 1, "a mixed-size struct pads its fields apart", mof01)

static void mof02(void)
{
    const char *k = mfrun(
        "struct t { float *q; int n; };\n"
        "__global__ void k(struct t *s, float *o){\n"
        "  o[0] = s->q[1]; o[1] = (float)s->n; }\n")
        ? kbody("k") : NULL;

    CHECK(k != NULL);
    CHECK(hasadd(k, "%rd1", 8));
    CHECK(!hasadd(k, "%rd1", 4));
    PASS();
}
TH_REG("mof", 2, "an int after a pointer sits at eight, not four", mof02)

static void mof03(void)
{
    const char *k = mfrun(
        "struct s { float *p[4]; int n; };\n"
        "__global__ void k(struct s *s, float *o){\n"
        "  o[0] = s->p[2][1]; o[1] = (float)s->n; }\n")
        ? kbody("k") : NULL;

    CHECK(k != NULL);
    CHECK(hasadd(k, "%rd4", 16));
    CHECK(hasadd(k, "%rd1", 32));
    CHECK(strstr(k, "ld.global.f32") != NULL);
    PASS();
}
TH_REG("mof", 3, "a member array of pointers strides by eight", mof03)

static void mof04(void)
{
    const char *k = mfrun(
        "struct u { float a[4]; int n; };\n"
        "__global__ void k(struct u *s, float *o){\n"
        "  o[0] = s->a[2]; o[1] = (float)s->n; }\n")
        ? kbody("k") : NULL;

    CHECK(k != NULL);
    CHECK(hasadd(k, "%rd4", 8));
    CHECK(hasadd(k, "%rd1", 16));
    PASS();
}
TH_REG("mof", 4, "a member array strides by its element", mof04)

static void mof05(void)
{
    const char *k = mfrun(
        "struct c { int a; double b; };\n"
        "__global__ void k(struct c *c, double *o){ o[0] = c[2].b; }\n")
        ? kbody("k") : NULL;

    CHECK(k != NULL);
    CHECK(hasadd(k, "%rd1", 32));
    PASS();
}
TH_REG("mof", 5, "an array of structs strides whole", mof05)

static void mof06(void)
{
    const char *k = mfrun(
        "struct n { int v; struct n *next; };\n"
        "__global__ void k(struct n *p, int *o){ o[0] = p->next->v; }\n")
        ? kbody("k") : NULL;

    CHECK(k != NULL);
    CHECK(hasadd(k, "%rd1", 8));
    CHECK(strstr(obuf, "E110") == NULL);
    PASS();
}
TH_REG("mof", 6, "a struct can point at itself", mof06)

static void mof07(void)
{
    const char *k = mfrun(
        "__global__ void k(float *o, int i){\n"
        "  float p, q[8]; q[i] = 1.0f; p = q[3]; o[0] = p; }\n")
        ? kbody("k") : NULL;

    CHECK(k != NULL);
    CHECK(strstr(k, "__local[32]") != NULL);
    CHECK(strstr(k, "st.local.f32") != NULL);
    PASS();
}
TH_REG("mof", 7, "the second declarator keeps its array bound", mof07)

static void mof08(void)
{
    const char *k = mfrun(
        "enum { NN = 3 };\n"
        "struct d { char c[NN]; int n; };\n"
        "__global__ void k(struct d *s, int *o){ o[0] = s->n; }\n")
        ? kbody("k") : NULL;

    CHECK(k != NULL);
    CHECK(hasadd(k, "%rd1", 4));
    CHECK(strstr(obuf, "E071") == NULL);
    PASS();
}
TH_REG("mof", 8, "an array bound is not a second field", mof08)
