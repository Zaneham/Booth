#define _POSIX_C_SOURCE 200809L   /* readlink: POSIX, not ISO C99 */
#include "hgen.h"
#include "bir_lower.h"
#include "bc_err.h"
#include "backend.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#undef OUT
#undef IN
#else
#include <unistd.h>
#endif

#define HG_MAXK   4096u
#define HG_MAXL   4096u
#define HG_MAXD   1024u
#define HG_MAXP   128u
#define HG_MAXT   8u
#define HG_MAXV   16u
#define HG_WBUF   65536u
#define HG_RBUF   65536u
#define HG_COLS   72u
#define HG_NAME   128u
#define HG_TARG   256u

#define HG_FWMAG  0x466243b1
#define HG_FMAG   0xba55ed50u
#define HG_KPTX   1u
#define HG_KCUB   2u

typedef struct {
    uint32_t s0, s1;
    uint32_t nmt;
    uint32_t p0, p1;
    uint32_t t0, t1;
    int      tmpl;
    int      stat;
    int      decl;
} hg_k_t;

typedef struct {
    uint32_t s0, s1;
    uint32_t k0, k1;
    uint32_t c0, c1;
    uint32_t a0, a1;
} hg_l_t;

typedef struct {
    uint32_t s0, s1;
    uint32_t sig;
} hg_d_t;

static hg_k_t   KS[HG_MAXK];
static uint32_t NK;
static hg_l_t   LS[HG_MAXL];
static uint32_t NL;
static hg_d_t   DS[HG_MAXD];
static uint32_t ND;
static FILE    *OF;
static char     WB[HG_WBUF];
static uint32_t WN;
static int      OOVF;
static int      OERR;
static uint8_t  RB[HG_RBUF];


static void wflu(void)
{
    if (WN == 0u) return;
    if (OF == NULL || fwrite(WB, 1u, WN, OF) != WN) OERR = 1;
    WN = 0u;
}

static void ow(const char *s, uint32_t n)
{
    while (n != 0u && !OERR) {
        uint32_t k = HG_WBUF - WN;

        if (k == 0u) { wflu(); continue; }
        if (k > n) k = n;
        memcpy(WB + WN, s, k);
        WN += k;
        s += k;
        n -= k;
    }
}

static void os_(const char *s)
{
    ow(s, (uint32_t)strlen(s));
}

static void onum(uint32_t v)
{
    char b[16];
    int  n = snprintf(b, sizeof b, "%u", (unsigned)v);

    if (n > 0) ow(b, (uint32_t)n);
}


static const char *ttx(const hg_in_t *in, uint32_t i)
{
    return in->src + in->toks[i].offset;
}

static int teq(const hg_in_t *in, uint32_t i, const char *s)
{
    size_t n = strlen(s);

    if (i >= in->ntok) return 0;
    return (size_t)in->toks[i].len == n
        && memcmp(ttx(in, i), s, n) == 0;
}

static void otok(const hg_in_t *in, uint32_t a, uint32_t b)
{
    uint32_t s, e;

    if (a > b || b >= in->ntok) return;
    s = in->toks[a].offset;
    e = in->toks[b].offset + in->toks[b].len;
    ow(in->src + s, e - s);
}

static int isdq(int t)
{
    return t == TOK_CU_GLOBAL || t == TOK_CU_DEVICE
        || t == TOK_CU_CONSTANT || t == TOK_CU_SHARED
        || t == TOK_CU_MANAGED;
}

static int isfq(int t)
{
    return t == TOK_CU_GLOBAL || t == TOK_CU_DEVICE
        || t == TOK_CU_HOST || t == TOK_CU_FORCEINLINE;
}


static uint32_t mbrace(const hg_in_t *in, uint32_t i)
{
    int      d = 0;
    uint32_t g = 0;

    for (; i < in->ntok && g < (uint32_t)in->ntok + 1u; i++, g++) {
        if (in->toks[i].type == TOK_LBRACE) d++;
        else if (in->toks[i].type == TOK_RBRACE) {
            d--;
            if (d == 0) return i;
        }
    }
    return in->ntok;
}

static uint32_t mparen(const hg_in_t *in, uint32_t i)
{
    int      d = 0;
    uint32_t g = 0;

    for (; i < in->ntok && g < (uint32_t)in->ntok + 1u; i++, g++) {
        if (in->toks[i].type == TOK_LPAREN) d++;
        else if (in->toks[i].type == TOK_RPAREN) {
            d--;
            if (d == 0) return i;
        }
    }
    return in->ntok;
}

static uint32_t mpopen(const hg_in_t *in, uint32_t i)
{
    int      d = 0;
    uint32_t g = 0;

    for (; g < (uint32_t)in->ntok + 1u; g++) {
        if (in->toks[i].type == TOK_RPAREN) d++;
        else if (in->toks[i].type == TOK_LPAREN) {
            d--;
            if (d == 0) return i;
        }
        if (i == 0u) break;
        i--;
    }
    return in->ntok;
}

static uint32_t thend(const hg_in_t *in, uint32_t s)
{
    int      d = 0;
    uint32_t g = 0;

    if (!teq(in, s, "template") || in->toks[s + 1u].type != TOK_LT) return s;
    for (uint32_t i = s + 1u; i < in->ntok && g < (uint32_t)in->ntok;
         i++, g++) {
        int t = in->toks[i].type;

        if (t == TOK_LT) d++;
        else if (t == TOK_SHR) d -= 2;
        else if (t == TOK_GT) d--;
        if (d <= 0) return i + 1u;
    }
    return s;
}

static uint32_t dsig(const hg_in_t *in, uint32_t s, uint32_t lb)
{
    uint32_t lp, ts;

    if (lb == 0u || lb <= s || lb >= in->ntok) return 0u;
    if (in->toks[lb - 1u].type != TOK_RPAREN) return 0u;
    lp = mpopen(in, lb - 1u);
    if (lp <= s || lp >= in->ntok) return 0u;
    if (in->toks[lp - 1u].type != TOK_IDENT
        && in->toks[lp - 1u].type != TOK_GT) return 0u;
    ts = thend(in, s);
    for (uint32_t i = ts; i < lp; i++)
        if (in->toks[i].type == TOK_ASSIGN) return 0u;
    return lb - 1u;
}

static uint8_t DIRL[BC_MAX_TOKENS];

static void dmark(const hg_in_t *in)
{
    uint32_t i = 0;

    memset(DIRL, 0, (size_t)in->ntok);
    while (i < in->ntok) {
        uint32_t ln = in->toks[i].line, j = i;
        int      hash = in->toks[i].type == TOK_HASH
                     || in->toks[i].type == TOK_PP_LINE;

        while (j < in->ntok && in->toks[j].line == ln) {
            if (hash) DIRL[j] = 1u;
            j++;
        }
        i = j;
    }
}

static uint32_t dstart(const hg_in_t *in, uint32_t i)
{
    uint32_t j = i;

    while (j > 0u) {
        int t = in->toks[j - 1u].type;

        if (t == TOK_SEMI || t == TOK_RBRACE || t == TOK_LBRACE) break;
        if (t == TOK_PP_LINE || t == TOK_HASH) break;
        if (DIRL[j - 1u]) break;
        j--;
    }
    return j;
}

static uint32_t tmplat(const hg_in_t *in, uint32_t s, uint32_t *t0)
{
    uint32_t j = s;
    int      d;
    uint32_t g = 0;

    *t0 = s;
    if (j == 0u || in->toks[j - 1u].type != TOK_GT) return s;
    d = 0;
    j--;
    for (; j > 0u && g < (uint32_t)in->ntok + 1u; j--, g++) {
        int t = in->toks[j].type;

        if (t == TOK_GT) d++;
        else if (t == TOK_LT) {
            d--;
            if (d == 0) break;
        }
    }
    if (j == 0u || !teq(in, j - 1u, "template")) return s;
    *t0 = j - 1u;
    return j - 1u;
}

static uint32_t klb(const hg_in_t *in, uint32_t i)
{
    if (in->toks[i].type != TOK_CU_LAUNCH_BOUNDS) return i;
    if (i + 1u >= in->ntok || in->toks[i + 1u].type != TOK_LPAREN) return i;
    return mparen(in, i + 1u);
}

static int kstat(const hg_in_t *in, uint32_t s, uint32_t q)
{
    for (uint32_t i = s; i < q; i++)
        if (in->toks[i].type == TOK_STATIC) return 1;
    return 0;
}

static int kadd(const hg_in_t *in, uint32_t q)
{
    uint32_t s = dstart(in, q), nm = 0, lp, rp, e, t0;
    hg_k_t  *k;

    s = tmplat(in, s, &t0);
    for (lp = q; lp < in->ntok; lp++) {
        uint32_t b = klb(in, lp);

        if (b != lp) { lp = b; continue; }
        if (in->toks[lp].type == TOK_LPAREN) break;
        if (in->toks[lp].type == TOK_IDENT) nm = lp;
    }
    if (lp >= in->ntok || nm == 0u) return 0;
    rp = mparen(in, lp);
    if (rp >= in->ntok) return 0;
    for (e = rp + 1u; e < in->ntok && in->toks[e].type != TOK_LBRACE
                      && in->toks[e].type != TOK_SEMI; e++) { }
    if (e >= in->ntok) return 0;
    if (NK >= HG_MAXK) return 0;
    k = &KS[NK++];
    k->decl = in->toks[e].type == TOK_SEMI;
    if (!k->decl) e = mbrace(in, e);
    if (e >= in->ntok) { NK--; return 0; }
    k->s0 = s; k->s1 = e; k->nmt = nm;
    k->p0 = lp + 1u; k->p1 = rp;
    k->t0 = t0; k->t1 = (t0 == s && !teq(in, t0, "template")) ? t0 : lp;
    k->tmpl = teq(in, t0, "template");
    k->stat = kstat(in, s, q);
    return 1;
}

static uint32_t lcallee(const hg_in_t *in, uint32_t open)
{
    uint32_t nm, g = 0;
    int      d = 0;

    if (open == 0u) return in->ntok;
    nm = open - 1u;
    if (in->toks[nm].type != TOK_GT && in->toks[nm].type != TOK_SHR) return nm;
    while (nm > 0u && g < (uint32_t)in->ntok) {
        int t = in->toks[nm].type;

        if (t == TOK_GT) d++;
        else if (t == TOK_SHR) d += 2;
        else if (t == TOK_LT) { d--; if (d == 0) break; }
        nm--;
        g++;
    }
    if (nm == 0u || in->toks[nm].type != TOK_LT) return in->ntok;
    return nm - 1u;
}

static void dadd(uint32_t s, uint32_t e, uint32_t sig)
{
    if (ND >= HG_MAXD) return;
    DS[ND].s0 = s;
    DS[ND].s1 = e;
    DS[ND].sig = sig;
    ND++;
}

static void scan(const hg_in_t *in)
{
    int      d = 0;
    uint32_t g = 0;

    NK = 0u; NL = 0u; ND = 0u;
    dmark(in);
    for (uint32_t i = 0; i < in->ntok && g < (uint32_t)in->ntok + 1u; i++, g++) {
        int t = in->toks[i].type;

        if (t == TOK_LBRACE) { d++; continue; }
        if (t == TOK_RBRACE) { if (d > 0) d--; continue; }
        if (t == TOK_LAUNCH_OPEN) {
            uint32_t c1 = i, lp, rp, gg = 0, nm;
            hg_l_t  *l;

            for (; c1 < in->ntok && in->toks[c1].type != TOK_LAUNCH_CLOSE
                   && gg < (uint32_t)in->ntok; c1++, gg++) { }
            if (c1 >= in->ntok || i == 0u) continue;
            nm = lcallee(in, i);
            if (nm >= in->ntok || in->toks[nm].type != TOK_IDENT) continue;
            lp = c1 + 1u;
            if (lp >= in->ntok || in->toks[lp].type != TOK_LPAREN) continue;
            rp = mparen(in, lp);
            if (rp >= in->ntok || NL >= HG_MAXL) continue;
            l = &LS[NL++];
            l->s0 = nm; l->s1 = rp;
            l->k0 = nm; l->k1 = i - 1u;
            l->c0 = i + 1u; l->c1 = c1;
            l->a0 = lp + 1u; l->a1 = rp;
            continue;
        }
        if (d != 0 || !isdq(t)) continue;
        if (t == TOK_CU_GLOBAL) {
            uint32_t n0 = NK;

            if (kadd(in, i) && NK > n0) { i = KS[NK - 1u].s1; continue; }
        }
        {
            uint32_t s = dstart(in, i), e = s, gg = 0, lb = 0;
            int      hh = 0;

            for (; e < in->ntok && gg < (uint32_t)in->ntok; e++, gg++) {
                if (in->toks[e].type == TOK_CU_HOST) hh = 1;
                if (in->toks[e].type == TOK_CONSTEXPR) hh = 1;
                if (in->toks[e].type == TOK_LBRACE) {
                    lb = e;
                    e = mbrace(in, e);
                    break;
                }
                if (in->toks[e].type == TOK_SEMI) { lb = e; break; }
            }
            if (e >= in->ntok) continue;
            if (!hh) dadd(s, e, dsig(in, s, lb));
            i = e;
        }
    }
}


static uint32_t psplit(const hg_in_t *in, const hg_k_t *k, uint32_t *b,
                       uint32_t *e, uint32_t max)
{
    uint32_t n = 0, s = k->p0;
    int      d = 0;

    if (k->p0 >= k->p1) return 0;
    if (k->p1 - k->p0 == 1u && teq(in, k->p0, "void")) return 0;
    for (uint32_t i = k->p0; i < k->p1 && n < max; i++) {
        int t = in->toks[i].type;

        if (t == TOK_LPAREN || t == TOK_LT || t == TOK_LBRACKET) d++;
        else if (t == TOK_RPAREN || t == TOK_GT || t == TOK_RBRACKET) d--;
        else if (t == TOK_COMMA && d == 0) {
            b[n] = s; e[n] = i; n++; s = i + 1u;
        }
    }
    if (n < max && s < k->p1) { b[n] = s; e[n] = k->p1; n++; }
    return n;
}

static uint32_t pend(const hg_in_t *in, uint32_t b, uint32_t e)
{
    int d = 0;

    for (uint32_t i = b; i < e; i++) {
        int t = in->toks[i].type;

        if (t == TOK_LPAREN || t == TOK_LT || t == TOK_LBRACKET) d++;
        else if (t == TOK_RPAREN || t == TOK_GT || t == TOK_RBRACKET) d--;
        else if (t == TOK_ASSIGN && d == 0) return i;
    }
    return e;
}

static int ptyp(int t)
{
    return t == TOK_CONST || t == TOK_VOLATILE || t == TOK_STRUCT
        || t == TOK_CLASS || t == TOK_ENUM || t == TOK_TYPENAME;
}

static int pnamed(const hg_in_t *in, uint32_t b, uint32_t e)
{
    if (e < b + 2u) return 0;
    if (in->toks[e - 1u].type != TOK_IDENT) return 0;
    return !ptyp(in->toks[e - 2u].type);
}

static int ppack(const hg_in_t *in, uint32_t b, uint32_t e)
{
    for (uint32_t i = b; i < e; i++)
        if (in->toks[i].type == TOK_ELLIPSIS) return 1;
    return 0;
}


typedef struct {
    char nm[64];
    char dc[64];
    int  pk;
    int  ty;
} hg_tp_t;

static uint32_t tpgt(const hg_in_t *in, const hg_k_t *k)
{
    int d = 0;

    for (uint32_t i = k->t0 + 1u; i < in->ntok; i++) {
        int t = in->toks[i].type;

        if (t == TOK_LT) d++;
        else if (t == TOK_GT) { d--; if (d == 0) return i; }
    }
    return in->ntok;
}

static void tcpy(const hg_in_t *in, uint32_t i, char *d, size_t n)
{
    size_t l = (size_t)in->toks[i].len;

    if (l > n - 1u) l = n - 1u;
    memcpy(d, ttx(in, i), l);
    d[l] = 0;
}

static void tp1(const hg_in_t *in, uint32_t a, uint32_t b, hg_tp_t *p)
{
    uint32_t e = b, nm;

    memset(p, 0, sizeof *p);
    for (uint32_t i = a; i < b; i++)
        if (in->toks[i].type == TOK_ASSIGN) { e = i; break; }
    if (e <= a) return;
    nm = e;
    p->ty = in->toks[a].type == TOK_TYPENAME
         || in->toks[a].type == TOK_CLASS;
    for (uint32_t i = a; i < e; i++) {
        if (in->toks[i].type == TOK_ELLIPSIS) p->pk = 1;
        if (in->toks[i].type == TOK_STAR && nm == e && i + 1u < e
            && in->toks[i + 1u].type == TOK_IDENT) nm = i + 1u;
    }
    if (nm == e && in->toks[e - 1u].type == TOK_IDENT) nm = e - 1u;
    if (nm == e) return;
    tcpy(in, nm, p->nm, sizeof p->nm);
    if (!p->ty && nm == a + 1u) tcpy(in, a, p->dc, sizeof p->dc);
}

static uint32_t tpall(const hg_in_t *in, const hg_k_t *k, hg_tp_t *tp,
                      uint32_t max)
{
    uint32_t n = 0, gt, s;
    int      d = 0;

    if (!k->tmpl || k->t0 + 2u >= in->ntok) return 0;
    gt = tpgt(in, k);
    if (gt >= in->ntok) return 0;
    s = k->t0 + 2u;
    for (uint32_t i = s; i <= gt && n < max; i++) {
        int t = in->toks[i].type;

        if (t == TOK_LT || t == TOK_LPAREN) d++;
        else if (t == TOK_RPAREN || (t == TOK_GT && i != gt)) d--;
        else if ((t == TOK_COMMA && d == 0) || i == gt) {
            if (i > s) { tp1(in, s, i, &tp[n]); n++; }
            s = i + 1u;
        }
    }
    return n;
}

static int oksym(const hg_in_t *in, const hg_k_t *k)
{
    hg_tp_t  tp[HG_MAXT];
    uint32_t nt = tpall(in, k, tp, HG_MAXT);

    otok(in, k->nmt, k->nmt);
    if (!k->tmpl) return 1;
    if (nt == 0u) return 0;
    os_("<");
    for (uint32_t i = 0; i < nt; i++) {
        if (tp[i].nm[0] == 0) return 0;
        if (i != 0u) os_(", ");
        os_(tp[i].nm);
        if (tp[i].pk) os_("...");
    }
    os_(">");
    return 1;
}

static void ostub(const hg_in_t *in, const hg_k_t *k)
{
    uint32_t b[HG_MAXP], e[HG_MAXP], n;
    char     nm[HG_NAME];

    n = psplit(in, k, b, e, HG_MAXP);
    os_("\n");
    if (k->tmpl) { otok(in, k->t0, tpgt(in, k)); os_("\n"); }
    if (k->stat) os_("static ");
    os_("void ");
    otok(in, k->nmt, k->nmt);
    os_("(");
    for (uint32_t i = 0; i < n; i++) {
        uint32_t pe = pend(in, b[i], e[i]);

        if (i != 0u) os_(", ");
        if (pe > b[i]) otok(in, b[i], pe - 1u);
        if (!pnamed(in, b[i], pe)) { os_(" __p"); onum(i); }
        if (pe < e[i]) { os_(" "); otok(in, pe, e[i] - 1u); }
    }
    os_(")");
    if (k->decl) { os_(";\n"); return; }
    os_("\n{\n    void *__a[] = { ");
    if (n == 0u) os_("0");
    for (uint32_t i = 0; i < n; i++) {
        uint32_t pe = pend(in, b[i], e[i]);

        if (i != 0u) os_(", ");
        os_("(void *)&");
        if (pnamed(in, b[i], pe)) otok(in, pe - 1u, pe - 1u);
        else { os_("__p"); onum(i); }
        if (ppack(in, b[i], pe)) os_("...");
    }
    os_(" };\n    __bl((const void *)");
    if (!oksym(in, k)) {
        tcpy(in, k->nmt, nm, sizeof nm);
        (void)be_fail(BC_E621, in->file, nm);
        OOVF = 1;
    }
    os_(", __a);\n}\n");
}


static void p16(uint8_t *p, uint32_t o, uint16_t v)
{
    p[o] = (uint8_t)(v & 0xFFu);
    p[o + 1u] = (uint8_t)(v >> 8);
}

static void p32(uint8_t *p, uint32_t o, uint32_t v)
{
    for (uint32_t i = 0; i < 4u; i++) p[o + i] = (uint8_t)(v >> (8u * i));
}

static void p64(uint8_t *p, uint32_t o, uint64_t v)
{
    for (uint32_t i = 0; i < 8u; i++) p[o + i] = (uint8_t)(v >> (8u * i));
}

static uint32_t fsize(const char *path, FILE **fp)
{
    FILE *f = (path != NULL) ? fopen(path, "rb") : NULL;
    long  sz;

    *fp = NULL;
    if (f == NULL) return 0u;
    if (fseek(f, 0L, SEEK_END) != 0) { fclose(f); return 0u; }
    sz = ftell(f);
    rewind(f);
    if (sz <= 0L) { fclose(f); return 0u; }
    *fp = f;
    return (uint32_t)sz;
}

static int fatno(const char *path)
{
    fprintf(stderr, "kath: cannot read %s\n", path != NULL ? path : "?");
    return 1;
}

static int fcpy(FILE *o, FILE *i, uint32_t n)
{
    while (n != 0u) {
        uint32_t k = (n > HG_RBUF) ? HG_RBUF : n;

        if (fread(RB, 1u, k, i) != k) return 1;
        if (fwrite(RB, 1u, k, o) != k) return 1;
        n -= k;
    }
    return 0;
}

static int fent(FILE *o, FILE *i, uint32_t kind, uint32_t sm, uint32_t len)
{
    uint8_t  h[64];
    uint32_t pad = ((len + 7u) & ~7u) - len;

    memset(h, 0, sizeof h);
    p16(h, 0u, (uint16_t)kind);
    p32(h, 4u, 64u);
    p64(h, 8u, (len + 7u) & ~7u);
    p32(h, 16u, len);
    p16(h, 26u, 1u);
    p32(h, 28u, sm);
    if (fwrite(h, 1u, 64u, o) != 64u) return 1;
    if (fcpy(o, i, len) != 0) return 1;
    return pad != 0u && fwrite(h + 8u, 1u, pad, o) != pad;
}

static uint32_t g32(const uint8_t *p)
{
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8)
         | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}

static uint32_t fcar(const char *path)
{
    FILE    *f;
    uint8_t  h[64];
    uint32_t at = 16u, m = 0u, n = fsize(path, &f);

    KA_GUARD(g, 64);
    if (f == NULL) return 0u;
    if (n < 16u || fread(h, 1u, 16u, f) != 16u || g32(h) != HG_FMAG) {
        fclose(f);
        return 0u;
    }
    while (at + 64u <= n && g--) {
        uint32_t k, hs, z;

        if (fseek(f, (long)at, SEEK_SET) != 0)       break;
        if (fread(h, 1u, 64u, f) != 64u)             break;
        k = (uint32_t)h[0] | ((uint32_t)h[1] << 8);
        hs = g32(h + 4u);
        z = g32(h + 8u);
        if (hs < 64u || hs > n - at)                 break;
        if (g32(h + 12u) != 0u)                      break;
        if (z > n - at - hs)                         break;
        if (k == HG_KCUB) m |= 2u;
        if (k == HG_KPTX) m |= 1u;
        at += hs + z;
    }
    fclose(f);
    return m;
}

int hg_fat(const char *cubin, const char *ptx, uint32_t sm, const char *out)
{
    FILE    *cf = NULL, *pf = NULL, *o;
    uint32_t cn = 0u, pn, tot = 0u;
    uint8_t  h[16];
    int      rc = 0;

    if (cubin != NULL) {
        cn = fsize(cubin, &cf);
        if (cn == 0u) return fatno(cubin);
    }
    pn = fsize(ptx, &pf);
    if (cn == 0u && pn == 0u) return fatno(ptx);
    if (cn != 0u) tot += 64u + ((cn + 7u) & ~7u);
    if (pn != 0u) tot += 64u + ((pn + 7u) & ~7u);
    memset(h, 0, sizeof h);
    p32(h, 0u, HG_FMAG);
    p16(h, 4u, 1u);
    p16(h, 6u, 16u);
    p64(h, 8u, tot);
    o = fopen(out, "wb");
    if (o == NULL) rc = 1;
    if (rc == 0 && fwrite(h, 1u, 16u, o) != 16u) rc = 1;
    if (rc == 0 && cn != 0u) rc = fent(o, cf, HG_KCUB, sm, cn);
    if (rc == 0 && pn != 0u) rc = fent(o, pf, HG_KPTX, sm, pn);
    if (cf != NULL) fclose(cf);
    if (pf != NULL) fclose(pf);
    if (o == NULL) return 1;
    if (fclose(o) != 0) rc = 1;
    if (rc != 0) remove(out);
    return rc;
}


int hg_binc(char *buf, size_t n)
{
    char  exe[512];
    char *sep, *bs;

#ifdef _WIN32
    DWORD k = GetModuleFileNameA(NULL, exe, (DWORD)sizeof exe);

    if (k == 0u || (size_t)k >= sizeof exe) return 0;
#else
    ssize_t k = readlink("/proc/self/exe", exe, sizeof exe - 1u);

    if (k <= 0 || (size_t)k >= sizeof exe) return 0;
    exe[k] = '\0';
#endif
    sep = strrchr(exe, '/');
    bs = strrchr(exe, '\\');
    if (bs > sep) sep = bs;
    if (sep == NULL) return 0;
    *sep = '\0';
    return snprintf(buf, n, "%s/runtime/include", exe) > 0;
}

static int capp(char *b, size_t n, size_t *u, const char *s)
{
    size_t l = strlen(s);

    if (*u + l >= n) return 0;
    memcpy(b + *u, s, l);
    *u += l;
    b[*u] = '\0';
    return 1;
}

static int cinc(char *b, size_t n, size_t *u, const char *d)
{
    return capp(b, n, u, " -I\"") && capp(b, n, u, d) && capp(b, n, u, "\"");
}

int hg_cc(const char *src, const char *out, const char *const *inc, int ninc)
{
    char   cmd[4096], own[512];
    const char *cc = getenv("BOOTH_CXX");
    const char *rt = getenv("BOOTH_CUDART");
    size_t u = 0u;
    int    ok, rc;

    if (cc == NULL || cc[0] == '\0') cc = "g++";
    if (rt == NULL || rt[0] == '\0') rt = "-lcudart";
    cmd[0] = '\0';
    ok = capp(cmd, sizeof cmd, &u, cc)
      && capp(cmd, sizeof cmd, &u, " -std=c++17 -O2");
    for (int i = 0; ok && i < ninc; i++)
        ok = cinc(cmd, sizeof cmd, &u, inc[i]);
    if (ok && hg_binc(own, sizeof own))
        ok = cinc(cmd, sizeof cmd, &u, own);
    ok = ok && capp(cmd, sizeof cmd, &u, " \"")
            && capp(cmd, sizeof cmd, &u, src)
            && capp(cmd, sizeof cmd, &u, "\" -o \"")
            && capp(cmd, sizeof cmd, &u, out)
            && capp(cmd, sizeof cmd, &u, "\" ")
            && capp(cmd, sizeof cmd, &u, rt);
    if (!ok) return 1;
    rc = system(cmd);
#ifdef _WIN32
    return rc == 0 ? 0 : rc;
#else
    return rc == 0 ? 0 : 1;
#endif
}

typedef struct { uint32_t s, e; int k; uint32_t i; } hg_e_t;

static hg_e_t   ES[HG_MAXK + HG_MAXL + HG_MAXD];
static uint32_t NE;

static struct {
    char sym[HG_NAME];
    char base[HG_NAME];
    char targ[HG_TARG];
} SM[HG_MAXK];
static uint32_t NSM;

static void eadd(uint32_t s, uint32_t e, int k, uint32_t i)
{
    if (NE >= (uint32_t)(sizeof ES / sizeof ES[0])) return;
    ES[NE].s = s; ES[NE].e = e; ES[NE].k = k; ES[NE].i = i;
    NE++;
}

static void esort(void)
{
    for (uint32_t i = 1; i < NE; i++) {
        hg_e_t   t = ES[i];
        uint32_t j = i;

        while (j > 0u && ES[j - 1u].s > t.s) { ES[j] = ES[j - 1u]; j--; }
        ES[j] = t;
    }
}

static void cpn(char *d, size_t n, const char *s)
{
    size_t k = (s != NULL) ? strlen(s) : 0u;

    if (k > n - 1u) k = n - 1u;
    if (k != 0u) memcpy(d, s, k);
    d[k] = 0;
}

static uint32_t smget(const char *sym, const char *base, const char *targ)
{
    for (uint32_t i = 0; i < NSM; i++)
        if (strcmp(SM[i].sym, sym) == 0) return i;
    if (NSM >= HG_MAXK) return HG_MAXK;
    cpn(SM[NSM].sym, HG_NAME, sym);
    cpn(SM[NSM].base, HG_NAME, base);
    cpn(SM[NSM].targ, HG_TARG, targ);
    NSM++;
    return NSM - 1u;
}

static uint32_t tvsp(const char *targ, char v[][64], uint32_t max)
{
    uint32_t n = 0, j = 0;
    int      d = 0;

    if (targ == NULL || targ[0] == 0) return 0;
    for (const char *p = targ; n < max; p++) {
        if (*p == '<') d++;
        else if (*p == '>') d--;
        if ((*p == ',' && d == 0) || *p == 0) {
            while (j > 0u && v[n][j - 1u] == 32) j--;
            v[n][j] = 0;
            n++;
            j = 0;
            if (*p == 0) break;
            while (p[1] == 32) p++;
        } else if (j < 63u) {
            v[n][j++] = *p;
        }
    }
    return n;
}

static const hg_k_t *kfind(const hg_in_t *in, const char *nm)
{
    for (uint32_t i = 0; i < NK; i++) {
        uint32_t l = in->toks[KS[i].nmt].len;

        if ((size_t)l == strlen(nm)
            && memcmp(ttx(in, KS[i].nmt), nm, l) == 0) return &KS[i];
    }
    return NULL;
}

static void olnch(const hg_in_t *in, uint32_t li)
{
    const hg_l_t *l = &LS[li];
    uint32_t      cb[8], ce[8], n = 0, s = l->c0;
    int           d = 0;

    for (uint32_t i = l->c0; i < l->c1 && n < 4u; i++) {
        int t = in->toks[i].type;

        if (t == TOK_LPAREN || t == TOK_LBRACKET) d++;
        else if (t == TOK_RPAREN || t == TOK_RBRACKET) d--;
        else if (t == TOK_COMMA && d == 0) {
            cb[n] = s; ce[n] = i; n++; s = i + 1u;
        }
    }
    if (n < 4u && s < l->c1) { cb[n] = s; ce[n] = l->c1; n++; }
    os_("(__cudaPushCallConfiguration(");
    for (uint32_t i = 0; i < 4u; i++) {
        if (i != 0u) os_(", ");
        if (i < n && ce[i] > cb[i]) otok(in, cb[i], ce[i] - 1u);
        else if (i == 3u) os_("(void *)0");
        else os_("0");
    }
    os_(") ? (void)0 : ");
    otok(in, l->k0, l->k1);
    os_("(");
    if (l->a1 > l->a0) otok(in, l->a0, l->a1 - 1u);
    os_("))");
}

static void obase(const char *p)
{
    const char *s = p;
    char        c[2];

    if (p == NULL) { os_("?"); return; }
    for (const char *q = p; *q != '\0'; q++)
        if (*q == '/' || *q == '\\') s = q + 1;
    c[1] = '\0';
    for (uint32_t i = 0; i < HG_NAME && s[i] != '\0'; i++) {
        int ok = (s[i] >= '0' && s[i] <= '9') || (s[i] >= 'A' && s[i] <= 'Z')
              || (s[i] >= 'a' && s[i] <= 'z') || s[i] == '.' || s[i] == '_'
              || s[i] == '-';

        c[0] = ok ? s[i] : '_';
        ow(c, 1u);
    }
}

static uint32_t besc(uint8_t b, char *d)
{
    static const char OCT[] = "01234567";

    d[0] = '\\';
    if (b == 0x22u || b == 0x5Cu || b == 0x3Fu) { d[1] = (char)b; return 2u; }
    if (b == 0x0Au) { d[1] = 'n'; return 2u; }
    if (b == 0x09u) { d[1] = 't'; return 2u; }
    if (b == 0x0Du) { d[1] = 'r'; return 2u; }
    if (b >= 0x20u && b <= 0x7Eu) { d[0] = (char)b; return 1u; }
    d[1] = OCT[(b >> 6) & 7u];
    d[2] = OCT[(b >> 3) & 7u];
    d[3] = OCT[b & 7u];
    return 4u;
}

static void ofat(const char *path)
{
    FILE    *f = (path != NULL) ? fopen(path, "rb") : NULL;
    char     e[4];
    uint32_t col = 0u;
    size_t   n;

    os_("\"");
    if (f != NULL) {
        while (!OERR && (n = fread(RB, 1u, HG_RBUF, f)) != 0u) {
            for (size_t i = 0; i < n; i++) {
                uint32_t k = besc(RB[i], e);

                if (col + k > HG_COLS) { os_("\"\n\""); col = 0u; }
                ow(e, k);
                col += k;
            }
        }
        fclose(f);
    }
    os_("\";\n");
}

static void opro(const hg_in_t *in)
{
    uint32_t c = fcar(in->fat);

    if (c == 1u)
        fprintf(stderr, "kath: %s: the fatbin carries PTX only, no cubin\n",
                in->file != NULL ? in->file : "?");
    os_("#include <cuda_runtime.h>\n");
    os_("static void __bl(const void *f, void **a)\n{\n");
    os_("    dim3 g, b;\n    size_t s;\n    void *t;\n\n");
    os_("    if (__cudaPopCallConfiguration(&g, &b, &s, &t) != cudaSuccess)");
    os_(" return;\n");
    os_("    (void)cudaLaunchKernel(f, g, b, a, s, (cudaStream_t)t);\n}\n");
    os_("static const unsigned char __bfat[] =\n");
    ofat(in->fat);
    if (c != 0u) {
        os_("static const char __bcar[] __attribute__((used)) = \"booth-fat: ");
        os_(c == 1u ? "ptx-only " : (c == 3u ? "cubin+ptx " : "cubin "));
        obase(in->file);
        os_("\";\n");
    }
    os_("struct __bfw { int m; int v; const void *d; void *f; };\n");
    os_("static struct __bfw __bw = { ");
    onum((uint32_t)HG_FWMAG);
    os_(", 1, __bfat, 0 };\n");
    os_("static void **__bh;\n");
}

static int isnum(const char *s)
{
    return s[0] == '-' || (s[0] >= '0' && s[0] <= '9');
}

static int oreg(const hg_in_t *in, uint32_t i)
{
    const hg_k_t *k = kfind(in, SM[i].base);
    hg_tp_t       tp[HG_MAXT];
    char          vv[HG_MAXV][64];
    uint32_t      nt, nv, fx = 0u, np = 0u, c = 0u;

    if (k == NULL) { (void)be_fail(BC_E620, in->file, SM[i].base); return 0; }
    os_(SM[i].base);
    if (!k->tmpl) return 1;
    nt = tpall(in, k, tp, HG_MAXT);
    nv = tvsp(SM[i].targ, vv, HG_MAXV);
    for (uint32_t j = 0; j < nt; j++) {
        if (tp[j].pk) np++;
        else fx++;
    }
    if (nt == 0u || np > 1u || nv < fx || (np == 0u && nv != nt)) {
        (void)be_fail(BC_E621, in->file, SM[i].base);
        return 0;
    }
    os_("<");
    for (uint32_t j = 0; j < nt; j++) {
        uint32_t cnt = tp[j].pk ? nv - fx : 1u;

        for (uint32_t m = 0; m < cnt && c < nv; m++, c++) {
            if (c != 0u) os_(", ");
            if (!tp[j].ty && tp[j].dc[0] != 0 && isnum(vv[c])) {
                os_("(");
                os_(tp[j].dc);
                os_(")");
            }
            os_(vv[c]);
        }
    }
    os_(">");
    return 1;
}

static void octor(const hg_in_t *in)
{
    os_("\nstatic void __bctor(void) __attribute__((constructor));\n");
    os_("static void __bctor(void)\n{\n");
    os_("    __bh = __cudaRegisterFatBinary((void *)&__bw);\n");
    for (uint32_t i = 0; i < NSM; i++) {
        os_("    __cudaRegisterFunction(__bh, (const char *)");
        if (!oreg(in, i)) { OOVF = 1; return; }
        os_(", (char *)\"");
        os_(SM[i].sym);
        os_("\", \"");
        os_(SM[i].sym);
        os_("\", -1, 0, 0, 0, 0, 0);\n");
    }
    os_("    __cudaRegisterFatBinaryEnd(__bh);\n}\n");
}

static uint32_t TQ;

static void ogap(const hg_in_t *in, uint32_t a, uint32_t b)
{
    uint32_t c = a;

    while (TQ < in->ntok && in->toks[TQ].offset < a) TQ++;
    for (; TQ < in->ntok && in->toks[TQ].offset < b; TQ++) {
        uint32_t s = in->toks[TQ].offset;

        if (!isfq(in->toks[TQ].type)) continue;
        if (s > c) ow(in->src + c, s - c);
        c = s + in->toks[TQ].len;
    }
    if (b > c) ow(in->src + c, b - c);
}

static void odecl(const hg_in_t *in, uint32_t di)
{
    uint32_t sig = DS[di].sig;

    if (sig == 0u || sig >= in->ntok) return;
    ogap(in, in->toks[DS[di].s0].offset,
         in->toks[sig].offset + in->toks[sig].len);
    os_(";\n");
}

static void obody(const hg_in_t *in)
{
    uint32_t cur = 0u;

    for (uint32_t i = 0; i < NE; i++) {
        uint32_t s = in->toks[ES[i].s].offset;
        uint32_t e = in->toks[ES[i].e].offset + in->toks[ES[i].e].len;

        if (s < cur) continue;
        ogap(in, cur, s);
        if (ES[i].k == 2) olnch(in, ES[i].i);
        else if (ES[i].k == 1) ostub(in, &KS[ES[i].i]);
        else odecl(in, ES[i].i);
        cur = e;
    }
    if (cur < in->slen) ogap(in, cur, in->slen);
}

static int hgmap(const hg_in_t *in)
{
    uint32_t n = bir_nlnch();

    for (uint32_t i = 0; i < n; i++) {
        const char *base = NULL, *sym = NULL, *targ = NULL;

        if (!bir_lnch(i, &base, &sym, &targ)) continue;
        if (smget(sym, base, targ) >= HG_MAXK) {
            (void)be_fail(BC_E622, in->file, (unsigned)HG_MAXK);
            return 1;
        }
    }
    return 0;
}

int hg_emit(const hg_in_t *in, const char *out)
{
    WN = 0u; OOVF = 0; OERR = 0; NE = 0u; NSM = 0u; TQ = 0u;
    scan(in);
    if (hgmap(in) != 0) return 1;
    for (uint32_t i = 0; i < NL; i++) eadd(LS[i].s0, LS[i].s1, 2, i);
    for (uint32_t i = 0; i < NK; i++) eadd(KS[i].s0, KS[i].s1, 1, i);
    for (uint32_t i = 0; i < ND; i++) eadd(DS[i].s0, DS[i].s1, 0, i);
    esort();
    OF = fopen(out, "wb");
    if (OF == NULL) { fprintf(stderr, "kath: cannot write %s\n", out); return 1; }
    opro(in);
    obody(in);
    if (!OOVF) octor(in);
    wflu();
    if (fclose(OF) != 0) OERR = 1;
    OF = NULL;
    if (!OOVF && !OERR) return 0;
    if (OERR) fprintf(stderr, "kath: cannot write %s\n", out);
    remove(out);
    return 1;
}
