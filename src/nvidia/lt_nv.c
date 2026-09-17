/* lt_nv.c -- the cubin a kernel arrives in. (c) 2026 Zane Hambly. */

#include "lt_nv.h"
#include <string.h>
#include <assert.h>

/* ---- Container constants ---- */

#define NV_MACH     190u
#define NV_OSABI    65u
#define NV_ABIVER   8u
#define NV_FLAGS    0x06005904u

#define NV_NSYM     3u

#define SH_PROG     1u
#define SH_SYMTAB   2u
#define SH_STRTAB   3u
#define SH_RELA     4u
#define SH_REL      9u
#define SH_NOBITS   8u
#define SH_NVINFO   0x70000000u
#define SH_NVCG     0x70000001u
#define SH_NVPRO    0x70000002u

#define NV_FRAME    0x11u
#define NV_STACK    0x12u

#define EHDR_SZ     64u
#define SHDR_SZ     64u
#define SYM_SZ      24u
#define RELA_SZ     24u
#define REL_SZ      16u
#define CG_SZ       8u

static const uint8_t FNINF[20] = {
    0x03u, 0x5Fu, 0x00u, 0x00u,
    0x04u, 0x37u, 0x04u, 0x00u, 0x86u, 0x00u, 0x00u, 0x00u,
    0x04u, 0x66u, 0x04u, 0x00u, 0x03u, 0x00u, 0x00u, 0x00u
};

/* ---- Little-endian writes ---- */

static void w16(uint8_t *p, uint32_t v)
{
    p[0] = (uint8_t)(v & 0xFFu);
    p[1] = (uint8_t)((v >> 8) & 0xFFu);
}

static void w32(uint8_t *p, uint32_t v)
{
    w16(p, v & 0xFFFFu);
    w16(p + 2, (v >> 16) & 0xFFFFu);
}

static void w64(uint8_t *p, uint64_t v)
{
    w32(p, (uint32_t)(v & 0xFFFFFFFFu));
    w32(p + 4, (uint32_t)(v >> 32));
}

static uint32_t align(uint32_t v, uint32_t a)
{
    assert(a != 0u);

    return (v + a - 1u) & ~(a - 1u);
}

/* ---- String tables ---- */

typedef struct {
    char     s[LT_MAX_NVSTR];
    uint32_t n;
} str_t;

static uint32_t sadd(str_t *t, const char *a, const char *b)
{
    uint32_t at = t->n;
    size_t   la = strlen(a);
    size_t   lb = (b != NULL) ? strlen(b) : 0u;

    assert(t != NULL && a != NULL);

    if (at + la + lb + 2u > sizeof t->s) return 0u;
    memcpy(t->s + at, a, la);
    if (b != NULL) memcpy(t->s + at + la, b, lb);
    t->s[at + la + lb] = 0;
    t->n = at + (uint32_t)(la + lb) + 1u;
    return at;
}

/* ---- Section headers ---- */

static void shdr(uint8_t *p, uint32_t name, uint32_t type, uint64_t flags,
                 uint64_t off, uint64_t size, uint32_t link, uint32_t info,
                 uint64_t al, uint64_t ent)
{
    memset(p, 0, SHDR_SZ);
    w32(p +  0, name);
    w32(p +  4, type);
    w64(p +  8, flags);
    w64(p + 24, off);
    w64(p + 32, size);
    w32(p + 40, link);
    w32(p + 44, info);
    w64(p + 48, al);
    w64(p + 56, ent);
}

static void sym(uint8_t *p, uint32_t name, uint8_t info, uint8_t other,
                uint16_t sect, uint64_t val, uint64_t size)
{
    memset(p, 0, SYM_SZ);
    w32(p + 0, name);
    p[4] = info;
    p[5] = other;
    w16(p + 6, sect);
    w64(p + 8, val);
    w64(p + 16, size);
}

/* ---- The kernel's own metadata ---- */

static uint32_t pblock(const lt_nvk_t *k)
{
    uint32_t o = 0;

    assert(k != NULL);
    assert(k->np <= LT_MAX_NVPRM);

    for (uint32_t i = 0; i < k->np; i++) {
        uint32_t w = k->psz[i];

        o = align(o, w != 0u ? w : 1u);
        o += w;
    }
    return o;
}

static uint32_t nvinfo(uint8_t *p, const lt_nvk_t *k, uint32_t symi)
{
    uint32_t n = 0, o = 0;

    assert(p != NULL && k != NULL);

    if (k->isfn != 0u) {
        memcpy(p, FNINF, sizeof FNINF);
        return (uint32_t)sizeof FNINF;
    }

    for (uint32_t i = 0; i < k->np; i++) {
        uint32_t w = k->psz[i];

        o = align(o, w != 0u ? w : 1u);
        p[n++] = 0x04u; p[n++] = 0x17u; w16(p + n, 12u); n += 2u;
        w32(p + n, 0u); n += 4u;
        w16(p + n, i); n += 2u;
        w16(p + n, o); n += 2u;
        w16(p + n, 0xF000u); n += 2u;
        w16(p + n, (w << 2) | 1u); n += 2u;
        o += w;
    }

    p[n++] = 0x03u; p[n++] = 0x19u; w16(p + n, o); n += 2u;

    p[n++] = 0x04u; p[n++] = 0x0Au; w16(p + n, 8u); n += 2u;
    w32(p + n, symi); n += 4u;
    w16(p + n, LT_NV_PBASE); n += 2u;
    w16(p + n, o); n += 2u;

    return n;
}

/* ---- Layout ---- */

typedef struct {
    uint32_t text, con, ikr, kern, shr;
    uint32_t otext, ocon, oikr, zcon, zikr, ishr;
    uint32_t itext, icon, iikr, fsym;
    uint32_t nrl, nrla, irel, irla, orel, orla, rnm, anm;
    uint8_t  isfn;
} lkrn_t;

typedef struct {
    uint32_t nm[8];
    uint32_t off[8];
    lkrn_t   k[LT_MAX_NVKRN];
    uint32_t inf, sym, str, ssh, cgnm, prnm;
    uint32_t oinf, osym, ostr, ossh, osh, nsym;
    uint32_t zinf, total, iinf, isym, istr, issh, nsect;
    uint32_t ng, ig, gsyi, og, gtot, gnm;
    uint32_t icg, ocg, zcg, ipro, opro, zpro, nedge, nfn;
    uint32_t gsym[LT_MAX_NVGLB];
} lay_t;

static uint32_t ginfo(uint8_t *p, const lay_t *L, const lt_nvk_t *k,
                      uint32_t nk)
{
    uint32_t n = 0;

    assert(p != NULL && (k != NULL || nk == 0u));

    for (uint32_t i = 0; i < nk; i++) {
        p[n++] = 0x04u; p[n++] = NV_FRAME; w16(p + n, 8u); n += 2u;
        w32(p + n, L->k[i].fsym); n += 4u;
        w32(p + n, k[i].lmem); n += 4u;
    }
    for (uint32_t i = 0; i < nk; i++) {
        if (k[i].isfn != 0u) continue;
        p[n++] = 0x04u; p[n++] = NV_STACK; w16(p + n, 8u); n += 2u;
        w32(p + n, L->k[i].fsym); n += 4u;
        w32(p + n, k[i].lmem); n += 4u;
    }
    return n;
}

static void plrel(lay_t *L, const lt_nvk_t *k, uint32_t nk)
{
    for (uint32_t i = 0; i < nk; i++) {
        lkrn_t *K = &L->k[i];

        K->nrl = 0u;
        K->nrla = 0u;
        if (K->isfn != 0u) L->nfn++;
        for (uint32_t j = 0; j < k[i].nrel; j++) {
            if (k[i].rel[j].kind == (uint8_t)LT_NV_RRET) K->nrla++;
            else                                         K->nrl++;
            if (k[i].rel[j].kind == (uint8_t)LT_NV_RTGT) L->nedge++;
        }
    }
}

static lt_res_t plkrn(lay_t *L, str_t *sst, str_t *st, const lt_nvk_t *k,
                      uint32_t nk, uint32_t *op)
{
    uint32_t o = *op, ns = 0u, sx = 1u, yx = 1u;

    assert(L != NULL && (k != NULL || nk == 0u));

    for (uint32_t i = 0; i < nk; i++) {
        lkrn_t *K = &L->k[i];

        K->isfn = k[i].isfn;
        K->text = sadd(sst, ".text.", k[i].name);
        K->ikr  = sadd(sst, ".nv.info.", k[i].name);
        K->kern = sadd(st, k[i].name, NULL);
        K->itext = sx++;
        if (K->isfn == 0u) {
            K->con  = sadd(sst, ".nv.constant0.", k[i].name);
            K->icon = sx++;
            if (K->con == 0u) return LT_EFULL;
        }
        K->iikr = sx++;
        if (K->text == 0u || K->ikr == 0u || K->kern == 0u) return LT_EFULL;
        K->zcon  = LT_NV_PBASE + pblock(&k[i]);
        K->zikr  = (K->isfn != 0u) ? (uint32_t)sizeof FNINF
                                   : k[i].np * 16u + 16u;
        K->otext = align(o, 128u);
        K->ocon  = align(K->otext + k[i].clen, 4u);
        K->oikr  = align(K->ocon + (K->isfn != 0u ? 0u : K->zcon), 4u);
        o        = K->oikr + K->zikr;
        yx++;
        if (K->isfn == 0u) yx++;
        K->fsym = yx++;
    }
    L->iinf = sx;
    L->nsym = yx;
    for (uint32_t i = 0; i < nk; i++)
        if (k[i].smem != 0u) {
            L->k[i].shr = sadd(sst, ".nv.shared.", k[i].name);
            if (L->k[i].shr == 0u) return LT_EFULL;
            L->k[i].ishr = L->iinf + 1u + lt_nv_noten + ns++;
        }
    L->isym  = L->iinf + 1u + lt_nv_noten + ns;
    L->istr  = L->isym + 1u;
    L->issh  = L->istr + 1u;
    *op      = o;
    return LT_OK;
}

static lt_res_t plglb(lay_t *L, str_t *sst, str_t *st, const lt_nvgd_t *gd,
                      uint32_t *next)
{
    if (L->ng == 0u) return LT_OK;
    L->ig   = (*next)++;
    L->gnm  = sadd(sst, ".nv.global.init", NULL);
    L->gsyi = L->nsym;
    L->nsym += L->ng;
    L->gtot = gd->gtot;
    for (uint32_t i = 0; i < L->ng; i++) {
        L->gsym[i] = sadd(st, gd->g[i].name, NULL);
        if (L->gsym[i] == 0u) return LT_EFULL;
    }
    return LT_OK;
}

static lt_res_t plsec(lay_t *L, str_t *sst, str_t *st, const lt_nvk_t *k,
                      uint32_t nk, const lt_nvgd_t *gd)
{
    uint32_t next = L->issh + 1u;
    lt_res_t rc;

    assert(L != NULL && (k != NULL || nk == 0u));

    rc = plglb(L, sst, st, gd, &next);
    if (rc != LT_OK) return rc;
    for (uint32_t i = 0; i < nk; i++) {
        if (L->k[i].nrl != 0u) {
            L->k[i].irel = next++;
            L->k[i].rnm  = sadd(sst, ".rel.text.", k[i].name);
            if (L->k[i].rnm == 0u) return LT_EFULL;
        }
        if (L->k[i].nrla != 0u) {
            L->k[i].irla = next++;
            L->k[i].anm  = sadd(sst, ".rela.text.", k[i].name);
            if (L->k[i].anm == 0u) return LT_EFULL;
        }
    }
    if (L->nedge != 0u) {
        L->icg  = next++;
        L->cgnm = sadd(sst, ".nv.callgraph", NULL);
        L->zcg  = CG_SZ * (L->nedge + 4u);
        if (L->cgnm == 0u) return LT_EFULL;
    }
    if (L->nfn != 0u) {
        L->ipro = next++;
        L->prnm = sadd(sst, ".nv.prototype", NULL);
        L->zpro = CG_SZ * L->nfn;
        if (L->prnm == 0u) return LT_EFULL;
    }
    L->nsect = next;
    return LT_OK;
}

static void plofs(lay_t *L, const str_t *sst, const str_t *st,
                  const lt_nvk_t *k, uint32_t nk, uint32_t o)
{
    uint32_t p, nen = 0u;

    assert(L != NULL && (k != NULL || nk == 0u));

    o = align(o, 4u);
    for (uint32_t i = 0; i < lt_nv_noten; i++) {
        L->off[i] = align(o, lt_nv_note[i].align);
        o         = L->off[i] + lt_nv_note[i].len;
    }
    for (uint32_t i = 0; i < nk; i++) if (k[i].isfn == 0u) nen++;
    L->zinf  = 12u * nk + 12u * nen;
    L->oinf  = o;
    L->osym  = align(L->oinf + L->zinf, 8u);
    L->ostr  = L->osym + L->nsym * SYM_SZ;
    L->ossh  = L->ostr + st->n;
    p        = align(L->ossh + sst->n, 8u);
    if (L->ng > 0u) { L->og = p; p = align(L->og + L->gtot, 8u); }
    for (uint32_t i = 0; i < nk; i++) {
        if (L->k[i].nrl != 0u) {
            L->k[i].orel = p;
            p = align(p + L->k[i].nrl * REL_SZ, 8u);
        }
        if (L->k[i].nrla != 0u) {
            L->k[i].orla = p;
            p = align(p + L->k[i].nrla * RELA_SZ, 8u);
        }
    }
    if (L->nedge != 0u) { L->ocg = p; p = align(p + L->zcg, 8u); }
    if (L->nfn != 0u)   { L->opro = p; p = align(p + L->zpro, 8u); }
    L->osh   = p;
    L->total = L->osh + L->nsect * SHDR_SZ;
}

static lt_res_t plan(lay_t *L, str_t *sst, str_t *st, const lt_nvk_t *k,
                     uint32_t nk, const lt_nvgd_t *gd)
{
    uint32_t o = EHDR_SZ;
    lt_res_t rc;

    assert(L != NULL && (k != NULL || nk == 0u));
    assert(sst != NULL && st != NULL);
    if (lt_nv_noten > 8u || nk > LT_MAX_NVKRN) return LT_EFULL;

    memset(L, 0, sizeof *L);
    L->ng = (gd != NULL) ? gd->ng : 0u;
    if (L->ng > LT_MAX_NVGLB) return LT_EFULL;
    sst->n = 1u;
    st->n  = 1u;
    L->inf  = sadd(sst, ".nv.info", NULL);
    L->sym  = sadd(sst, ".symtab", NULL);
    L->str  = sadd(sst, ".strtab", NULL);
    L->ssh  = sadd(sst, ".shstrtab", NULL);
    for (uint32_t i = 0; i < lt_nv_noten; i++)
        L->nm[i] = sadd(sst, lt_nv_note[i].name, NULL);

    rc = plkrn(L, sst, st, k, nk, &o);
    if (rc != LT_OK) return rc;
    plrel(L, k, nk);
    rc = plsec(L, sst, st, k, nk, gd);
    if (rc != LT_OK) return rc;
    plofs(L, sst, st, k, nk, o);
    return LT_OK;
}

/* ---- Headers ---- */

static void nvehdr(uint8_t *out, const lay_t *L)
{
    assert(out != NULL && L != NULL);

    out[0] = 0x7Fu; out[1] = 0x45u; out[2] = 0x4Cu; out[3] = 0x46u;
    out[4] = 2u; out[5] = 1u; out[6] = 1u;
    out[7] = NV_OSABI;
    out[8] = NV_ABIVER;
    w16(out + 16, 2u);
    w16(out + 18, NV_MACH);
    w32(out + 20, 1u);
    w64(out + 40, L->osh);
    w32(out + 48, NV_FLAGS);
    w16(out + 52, EHDR_SZ);
    w16(out + 58, SHDR_SZ);
    w16(out + 60, L->nsect);
    w16(out + 62, L->issh);
}

static void hxtra(uint8_t *sh, const lay_t *L, uint32_t nk)
{
    for (uint32_t i = 0; i < nk; i++) {
        const lkrn_t *K = &L->k[i];

        if (K->nrl != 0u)
            shdr(sh + K->irel * SHDR_SZ, K->rnm, SH_REL, 0x40u,
                 K->orel, K->nrl * REL_SZ, L->isym, K->itext, 8u, REL_SZ);
        if (K->nrla != 0u)
            shdr(sh + K->irla * SHDR_SZ, K->anm, SH_RELA, 0x40u,
                 K->orla, K->nrla * RELA_SZ, L->isym, K->itext, 8u, RELA_SZ);
    }
    if (L->nedge != 0u)
        shdr(sh + L->icg * SHDR_SZ, L->cgnm, SH_NVCG, 0u, L->ocg, L->zcg,
             L->isym, 0u, 4u, CG_SZ);
    if (L->nfn != 0u)
        shdr(sh + L->ipro * SHDR_SZ, L->prnm, SH_NVPRO, 0u, L->opro, L->zpro,
             L->isym, 0u, 4u, CG_SZ);
    if (L->ng > 0u)
        shdr(sh + L->ig * SHDR_SZ, L->gnm, SH_PROG, 0x3u, L->og, L->gtot,
             0u, 0u, 8u, 0u);
}

static void heads(uint8_t *out, const lay_t *L, const lt_nvk_t *k,
                  uint32_t nk, uint32_t zst, uint32_t zss)
{
    uint8_t *sh = out + L->osh;

    assert(out != NULL && L != NULL);
    assert(k != NULL || nk == 0u);

    nvehdr(out, L);
    shdr(sh, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u);
    for (uint32_t i = 0; i < nk; i++) {
        const lkrn_t *K = &L->k[i];

        shdr(sh + K->itext * SHDR_SZ, K->text, SH_PROG, 6u, K->otext,
             k[i].clen, L->isym, (k[i].nreg << 24) | K->fsym, 128u, 0u);
        if (K->isfn == 0u)
            shdr(sh + K->icon * SHDR_SZ, K->con, SH_PROG, 0x42u, K->ocon,
                 K->zcon, 0u, K->itext, 4u, 0u);
        shdr(sh + K->iikr * SHDR_SZ, K->ikr, SH_NVINFO, 0x40u, K->oikr,
             K->zikr, L->isym, K->itext, 4u, 0u);
    }
    shdr(sh + L->iinf * SHDR_SZ, L->inf, SH_NVINFO, 0u, L->oinf, L->zinf,
         L->isym, 0u, 4u, 0u);

    for (uint32_t i = 0; i < lt_nv_noten; i++) {
        const lt_nvn_t *v = &lt_nv_note[i];

        shdr(sh + (L->iinf + 1u + i) * SHDR_SZ, L->nm[i], v->type, v->flags,
             L->off[i], v->len, i == 1u ? L->iinf + 1u : 0u, v->info,
             v->align, 0u);
    }

    /* per probe obj/corpus/test_shared2d.cubin */
    for (uint32_t i = 0; i < nk; i++)
        if (L->k[i].ishr != 0u)
            shdr(sh + L->k[i].ishr * SHDR_SZ, L->k[i].shr, SH_NOBITS, 0x43u,
                 L->osym, k[i].smem, 0u, L->k[i].itext, 4u, 0u);

    shdr(sh + L->isym * SHDR_SZ, L->sym, SH_SYMTAB, 0u, L->osym,
         L->nsym * SYM_SZ, L->istr, L->nsym, 8u, SYM_SZ);
    shdr(sh + L->istr * SHDR_SZ, L->str, SH_STRTAB, 0u, L->ostr, zst, 0u, 0u,
         1u, 0u);
    shdr(sh + L->issh * SHDR_SZ, L->ssh, SH_STRTAB, 0u, L->ossh, zss, 0u, 0u,
         1u, 0u);
    hxtra(sh, L, nk);
}

/* ---- Writing ---- */

static uint32_t rsymof(const lay_t *L, const lt_nvk_t *k, uint32_t i,
                       uint32_t j)
{
    const lt_nvrl_t *r = &k[i].rel[j];

    if (r->kind == (uint8_t)LT_NV_RTGT)
        return (r->gi < LT_MAX_NVKRN) ? L->k[r->gi].fsym : 0u;
    if (r->kind == (uint8_t)LT_NV_RRET) return L->k[i].fsym;
    return L->gsyi + r->gi;
}

static void grel(uint8_t *out, const lay_t *L, const lt_nvk_t *k, uint32_t nk)
{
    assert(out != NULL && L != NULL);
    assert(k != NULL || nk == 0u);

    for (uint32_t i = 0; i < nk; i++) {
        uint8_t *r = out + L->k[i].orel;
        uint8_t *a = out + L->k[i].orla;
        uint32_t nr = 0u, na = 0u;

        for (uint32_t j = 0; j < k[i].nrel; j++) {
            const lt_nvrl_t *e = &k[i].rel[j];
            uint32_t s = rsymof(L, k, i, j);
            uint32_t ty;

            if (e->kind == (uint8_t)LT_NV_RRET) {
                ty = (e->hi != 0u) ? 0x39u : 0x38u;
                w64(a + na * RELA_SZ, e->at);
                w64(a + na * RELA_SZ + 8u, ((uint64_t)s << 32) | ty);
                w64(a + na * RELA_SZ + 16u, e->add);
                na++;
                continue;
            }
            ty = (e->kind == (uint8_t)LT_NV_RTGT)
               ? 0x3Au : ((e->hi != 0u) ? 0x39u : 0x38u);
            w64(r + nr * REL_SZ, e->at);
            w64(r + nr * REL_SZ + 8u, ((uint64_t)s << 32) | ty);
            nr++;
        }
    }
}

static void gcgph(uint8_t *out, const lay_t *L, const lt_nvk_t *k, uint32_t nk)
{
    uint8_t *p = out + L->ocg;
    uint32_t n = 1u;

    if (L->nedge == 0u) return;
    w32(p + 0, 0u);
    w32(p + 4, 0xFFFFFFFFu);
    for (uint32_t i = 0; i < nk; i++)
        for (uint32_t j = 0; j < k[i].nrel; j++) {
            if (k[i].rel[j].kind != (uint8_t)LT_NV_RTGT) continue;
            w32(p + n * CG_SZ, L->k[i].fsym);
            w32(p + n * CG_SZ + 4u, rsymof(L, k, i, j));
            n++;
        }
    w32(p + n * CG_SZ, 0u); w32(p + n * CG_SZ + 4u, 0xFFFFFFFEu); n++;
    w32(p + n * CG_SZ, 0u); w32(p + n * CG_SZ + 4u, 0xFFFFFFFDu); n++;
    w32(p + n * CG_SZ, 0u); w32(p + n * CG_SZ + 4u, 0xFFFFFFFCu);
}

static void gproto(uint8_t *out, const lay_t *L, const lt_nvk_t *k,
                   uint32_t nk)
{
    uint8_t *p = out + L->opro;
    uint32_t n = 0u;

    if (L->nfn == 0u) return;
    for (uint32_t i = 0; i < nk; i++) {
        if (k[i].isfn == 0u) continue;
        w32(p + n * CG_SZ, L->k[i].fsym);
        w32(p + n * CG_SZ + 4u, 1u);
        n++;
    }
}

/* per probe obj/corpus */
static void gsyms(uint8_t *out, const lay_t *L, const lt_nvk_t *k,
                  uint32_t nk)
{
    for (uint32_t i = 0; i < nk; i++) {
        const lkrn_t *K = &L->k[i];
        uint32_t      t = K->fsym - (K->isfn != 0u ? 1u : 2u);

        sym(out + L->osym + t * SYM_SZ, 0u, 0x03u, 0u,
            (uint16_t)K->itext, 0u, 0u);
        if (K->isfn == 0u)
            sym(out + L->osym + (t + 1u) * SYM_SZ, 0u, 0x03u, 0u,
                (uint16_t)K->icon, 0u, 0u);
        sym(out + L->osym + K->fsym * SYM_SZ, K->kern, 0x02u,
            (uint8_t)(K->isfn != 0u ? 0x00u : 0x10u),
            (uint16_t)K->itext, 0u, k[i].clen);
    }
}

lt_res_t
lt_nvcub(uint8_t *out, uint32_t max, uint32_t *len, const lt_nvk_t *k,
         uint32_t nk, const lt_nvgd_t *gd)
{
    static str_t sst, st;
    static lay_t L;

    assert(out != NULL && len != NULL);
    assert(k != NULL || nk == 0u);

    if (nk > LT_MAX_NVKRN) return LT_EFULL;
    for (uint32_t i = 0; i < nk; i++) {
        if (k[i].name == NULL || k[i].code == NULL) return LT_EBAD;
        if (k[i].clen == 0u || (k[i].clen & 15u) != 0u) return LT_EBAD;
        if (strlen(k[i].name) + 1u > LT_MAX_NVNAME) return LT_EBAD;
        if (k[i].nreg > 255u || k[i].np > LT_MAX_NVPRM) return LT_EBAD;
    }

    if (plan(&L, &sst, &st, k, nk, gd) != LT_OK) return LT_EFULL;
    if (L.total > max || L.total > LT_MAX_NVIMG) return LT_EFULL;

    memset(out, 0, L.total);
    for (uint32_t i = 0; i < nk; i++) {
        memcpy(out + L.k[i].otext, k[i].code, k[i].clen);
        (void)nvinfo(out + L.k[i].oikr, &k[i], L.k[i].fsym - 1u);
    }
    (void)ginfo(out + L.oinf, &L, k, nk);

    for (uint32_t i = 0; i < lt_nv_noten; i++)
        memcpy(out + L.off[i], lt_nv_note[i].data, lt_nv_note[i].len);

    gsyms(out, &L, k, nk);

    if (L.ng > 0u) {
        memcpy(out + L.og, gd->bytes, L.gtot);
        for (uint32_t i = 0; i < L.ng; i++)
            sym(out + L.osym + (L.gsyi + i) * SYM_SZ, L.gsym[i], 0x01u, 0u,
                (uint16_t)L.ig, gd->g[i].off, gd->g[i].size);
    }
    grel(out, &L, k, nk);
    gcgph(out, &L, k, nk);
    gproto(out, &L, k, nk);

    memcpy(out + L.ostr, st.s, st.n);
    memcpy(out + L.ossh, sst.s, sst.n);

    heads(out, &L, k, nk, st.n, sst.n);
    *len = L.total;
    return LT_OK;
}
