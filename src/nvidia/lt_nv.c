/* lt_nv.c -- the cubin a kernel arrives in. (c) 2026 Zane Hambly. */

#include "lt_nv.h"
#include <string.h>
#include <assert.h>

/* ---- Container constants ---- */

#define NV_MACH     190u
#define NV_OSABI    65u
#define NV_ABIVER   8u
#define NV_FLAGS    0x06005904u

#define NV_NSYM     4u

#define SH_PROG     1u
#define SH_SYMTAB   2u
#define SH_STRTAB   3u
#define SH_NOBITS   8u
#define SH_NVINFO   0x70000000u

#define NV_FRAME    0x11u
#define NV_STACK    0x12u

#define EHDR_SZ     64u
#define SHDR_SZ     64u
#define SYM_SZ      24u

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
    char     s[1024];
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
    t->s[at + la + lb] = '\0';
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

static uint32_t ginfo(uint8_t *p, const lt_nvk_t *k, uint32_t symi)
{
    uint32_t n = 0;

    assert(p != NULL && k != NULL);

    p[n++] = 0x04u; p[n++] = NV_FRAME; w16(p + n, 8u); n += 2u;
    w32(p + n, symi); n += 4u;
    w32(p + n, k->lmem); n += 4u;

    p[n++] = 0x04u; p[n++] = NV_STACK; w16(p + n, 8u); n += 2u;
    w32(p + n, symi); n += 4u;
    w32(p + n, k->lmem); n += 4u;
    return n;
}

/* ---- Layout ---- */

typedef struct {
    uint32_t nm[8];
    uint32_t off[8];
    uint32_t text, con, ikr, inf, sym, str, ssh, kern, shr;
    uint32_t otext, ocon, oikr, oinf, osym, ostr, ossh, osh;
    uint32_t zcon, zikr, zinf, total, isym, istr, issh, ishr, nsect;
} lay_t;

static lt_res_t plan(lay_t *L, str_t *sst, str_t *st, const lt_nvk_t *k,
                     uint32_t clen)
{
    assert(L != NULL && k != NULL);
    assert(sst != NULL && st != NULL);

    if (lt_nv_noten > 8u) return LT_EFULL;

    memset(L, 0, sizeof *L);
    sst->n = 1u;
    st->n  = 1u;

    L->text = sadd(sst, ".text.", k->name);
    L->con  = sadd(sst, ".nv.constant0.", k->name);
    L->ikr  = sadd(sst, ".nv.info.", k->name);
    L->inf  = sadd(sst, ".nv.info", NULL);
    L->sym  = sadd(sst, ".symtab", NULL);
    L->str  = sadd(sst, ".strtab", NULL);
    L->ssh  = sadd(sst, ".shstrtab", NULL);
    L->kern = sadd(st, k->name, NULL);
    for (uint32_t i = 0; i < lt_nv_noten; i++)
        L->nm[i] = sadd(sst, lt_nv_note[i].name, NULL);

    if (k->smem != 0u) L->shr = sadd(sst, ".nv.shared.", k->name);
    L->ishr  = (k->smem != 0u) ? 5u + lt_nv_noten : 0u;
    L->nsect = 8u + lt_nv_noten + ((k->smem != 0u) ? 1u : 0u);
    L->isym  = 5u + lt_nv_noten + ((k->smem != 0u) ? 1u : 0u);
    L->istr  = L->isym + 1u;
    L->issh  = L->istr + 1u;

    L->zcon = LT_NV_PBASE + pblock(k);
    L->zikr = k->np * 16u + 16u;
    L->zinf = 24u;

    L->otext = align(EHDR_SZ, 128u);
    L->ocon  = align(L->otext + clen, 4u);
    L->oikr  = align(L->ocon + L->zcon, 4u);
    L->osym  = align(L->oikr + L->zikr, 4u);

    for (uint32_t i = 0; i < lt_nv_noten; i++) {
        L->off[i] = align(L->osym, lt_nv_note[i].align);
        L->osym   = L->off[i] + lt_nv_note[i].len;
    }
    L->oinf  = L->osym;
    L->osym  = align(L->oinf + L->zinf, 8u);
    L->ostr  = L->osym + NV_NSYM * SYM_SZ;
    L->ossh  = L->ostr + st->n;
    L->osh   = align(L->ossh + sst->n, 8u);
    L->total = L->osh + L->nsect * SHDR_SZ;
    return LT_OK;
}

/* ---- Headers ---- */

static void heads(uint8_t *out, const lay_t *L, const lt_nvk_t *k,
                  uint32_t clen, uint32_t zst, uint32_t zss)
{
    uint8_t *sh = out + L->osh;

    assert(out != NULL && L != NULL);
    assert(k != NULL);

    out[0] = 0x7Fu; out[1] = 'E'; out[2] = 'L'; out[3] = 'F';
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

    shdr(sh, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u);
    shdr(sh + SHDR_SZ, L->text, SH_PROG, 6u, L->otext, clen, L->isym,
         (k->nreg << 24) | 3u, 128u, 0u);
    shdr(sh + 2u * SHDR_SZ, L->con, SH_PROG, 0x42u, L->ocon, L->zcon, 0u,
         1u, 4u, 0u);
    shdr(sh + 3u * SHDR_SZ, L->ikr, SH_NVINFO, 0x40u, L->oikr, L->zikr,
         L->isym, 1u, 4u, 0u);
    shdr(sh + 4u * SHDR_SZ, L->inf, SH_NVINFO, 0u, L->oinf, L->zinf, L->isym,
         0u, 4u, 0u);

    for (uint32_t i = 0; i < lt_nv_noten; i++) {
        const lt_nvn_t *v = &lt_nv_note[i];

        shdr(sh + (5u + i) * SHDR_SZ, L->nm[i], v->type, v->flags, L->off[i],
             v->len, i == 1u ? 5u : 0u, v->info, v->align, 0u);
    }

    /* per probe obj/corpus/test_shared2d.cubin */
    if (L->ishr != 0u)
        shdr(sh + L->ishr * SHDR_SZ, L->shr, SH_NOBITS, 0x43u, L->osym,
             k->smem, 0u, 1u, 4u, 0u);

    shdr(sh + L->isym * SHDR_SZ, L->sym, SH_SYMTAB, 0u, L->osym,
         NV_NSYM * SYM_SZ, L->istr, 3u, 8u, SYM_SZ);
    shdr(sh + L->istr * SHDR_SZ, L->str, SH_STRTAB, 0u, L->ostr, zst, 0u, 0u,
         1u, 0u);
    shdr(sh + L->issh * SHDR_SZ, L->ssh, SH_STRTAB, 0u, L->ossh, zss, 0u, 0u,
         1u, 0u);
}

/* ---- Writing ---- */

lt_res_t
lt_nvcub(uint8_t *out, uint32_t max, uint32_t *len, const lt_nvk_t *k,
         const uint8_t *code, uint32_t clen)
{
    static str_t sst, st;
    static lay_t L;

    assert(out != NULL && len != NULL);
    assert(k != NULL && k->name != NULL);

    if (clen == 0u || (clen & 15u) != 0u) return LT_EBAD;
    if (strlen(k->name) + 1u > LT_MAX_NVNAME) return LT_EBAD;
    if (k->nreg > 255u || k->np > LT_MAX_NVPRM) return LT_EBAD;

    memset(&sst, 0, sizeof sst);
    memset(&st, 0, sizeof st);
    if (plan(&L, &sst, &st, k, clen) != LT_OK) return LT_EFULL;
    if (L.total > max || L.total > LT_MAX_NVIMG) return LT_EFULL;

    memset(out, 0, L.total);
    memcpy(out + L.otext, code, clen);
    (void)nvinfo(out + L.oikr, k, 3u);
    (void)ginfo(out + L.oinf, k, 3u);

    for (uint32_t i = 0; i < lt_nv_noten; i++)
        memcpy(out + L.off[i], lt_nv_note[i].data, lt_nv_note[i].len);

    sym(out + L.osym + 1u * SYM_SZ, 0u, 0x03u, 0u, 1u, 0u, 0u);
    sym(out + L.osym + 2u * SYM_SZ, 0u, 0x03u, 0u, 2u, 0u, 0u);
    /* per probe obj/corpus */
    sym(out + L.osym + 3u * SYM_SZ, L.kern, 0x02u, 0x10u, 1u, 0u, clen);

    memcpy(out + L.ostr, st.s, st.n);
    memcpy(out + L.ossh, sst.s, sst.n);

    heads(out, &L, k, clen, st.n, sst.n);
    *len = L.total;
    return LT_OK;
}
