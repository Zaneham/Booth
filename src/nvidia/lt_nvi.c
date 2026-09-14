/* lt_nvi.c -- placing bits into a Volta instruction word. (c) 2026 Zane Hambly. */

#include "lt_nv.h"
#include <assert.h>

/* ---- Bits ---- */

void
lt_nvput(uint64_t *w, uint32_t hb, uint32_t lb, uint64_t v)
{
    assert(w != NULL);
    assert(hb < 128u && lb <= hb);
    assert(hb - lb < 64u);

    for (uint32_t b = lb; b <= hb; b++) {
        uint64_t bit = (v >> (b - lb)) & 1ull;
        uint32_t i = b >> 6, s = b & 63u;

        w[i] = (w[i] & ~(1ull << s)) | (bit << s);
    }
}

/* ---- The shape every instruction shares ---- */

static void head(uint64_t *w, uint32_t op, uint32_t form)
{
    assert(w != NULL);

    w[0] = 0;
    w[1] = 0;
    lt_nvput(w, 11u, 0u, (uint64_t)op | ((uint64_t)form << 9));
    lt_nvput(w, 14u, 12u, LT_NV_PT);
    lt_nvput(w, 108u, 105u, 15u);
    lt_nvput(w, 112u, 110u, LT_NV_PT);
    lt_nvput(w, 115u, 113u, LT_NV_PT);
}

static void alu1(uint64_t *w, uint32_t op, uint32_t d, uint32_t a, uint32_t b,
                 uint32_t c)
{
    assert(d < 256u && a < 256u);
    assert(b < 256u && c < 256u);

    head(w, op, 1u);
    lt_nvput(w, 23u, 16u, d);
    lt_nvput(w, 31u, 24u, a);
    lt_nvput(w, 39u, 32u, b);
    lt_nvput(w, 71u, 64u, c);
}

/* ---- Moves ---- */

void
lt_nvmovc(uint64_t *w, uint32_t dst, uint32_t bank, uint32_t off)
{
    assert(dst < 256u && bank < 32u);

    head(w, 0x002u, 5u);
    lt_nvput(w, 23u, 16u, dst);
    lt_nvput(w, 53u, 38u, off);
    lt_nvput(w, 58u, 54u, bank);
    lt_nvput(w, 75u, 72u, 15u);
}

void
lt_nvmovi(uint64_t *w, uint32_t dst, uint32_t imm)
{
    assert(dst < 256u);

    head(w, 0x002u, 4u);
    lt_nvput(w, 23u, 16u, dst);
    lt_nvput(w, 63u, 32u, imm);
    lt_nvput(w, 75u, 72u, 15u);
}

void
lt_nvmov(uint64_t *w, uint32_t dst, uint32_t src)
{
    assert(dst < 256u && src < 256u);

    head(w, 0x002u, 1u);
    lt_nvput(w, 23u, 16u, dst);
    lt_nvput(w, 39u, 32u, src);
    lt_nvput(w, 75u, 72u, 15u);
}

void
lt_nvmovu(uint64_t *w, uint32_t dst, uint32_t ur)
{
    assert(dst < 256u && ur < 64u);

    head(w, 0x002u, 6u);
    lt_nvput(w, 23u, 16u, dst);
    lt_nvput(w, 37u, 32u, ur);
    lt_nvput(w, 75u, 72u, 15u);
    lt_nvput(w, 91u, 91u, 1u);
}

void
lt_nvs2r(uint64_t *w, uint32_t dst, uint32_t idx)
{
    assert(dst < 256u && idx < 256u);

    head(w, 0x919u, 0u);
    lt_nvput(w, 23u, 16u, dst);
    lt_nvput(w, 79u, 72u, idx);
}

/* ---- Integer ---- */

static void iadd(uint64_t *w, uint32_t d, uint32_t a, uint32_t b, uint32_t neg)
{
    alu1(w, 0x010u, d, a, b, LT_NV_RZ);
    lt_nvput(w, 63u, 63u, neg);
    lt_nvput(w, 79u, 77u, LT_NV_PT);
    lt_nvput(w, 80u, 80u, 1u);
    lt_nvput(w, 83u, 81u, LT_NV_PT);
    lt_nvput(w, 86u, 84u, LT_NV_PT);
    lt_nvput(w, 89u, 87u, LT_NV_PT);
    lt_nvput(w, 90u, 90u, 1u);
}

void
lt_nvadd(uint64_t *w, uint32_t d, uint32_t a, uint32_t b, uint32_t neg)
{
    assert(w != NULL);
    assert(neg < 2u);

    iadd(w, d, a, b, neg);
}

void
lt_nvadc(uint64_t *w, uint32_t d, uint32_t a, uint32_t b, uint32_t neg)
{
    assert(w != NULL);
    assert(neg < 2u);

    iadd(w, d, a, b, neg);
    lt_nvput(w, 83u, 81u, 0u);
}

void
lt_nvadx(uint64_t *w, uint32_t d, uint32_t a, uint32_t b, uint32_t neg)
{
    assert(w != NULL);
    assert(neg < 2u);

    iadd(w, d, a, b, neg);
    lt_nvput(w, 74u, 74u, 1u);
    lt_nvput(w, 89u, 87u, 0u);
    lt_nvput(w, 90u, 90u, 0u);
}

void
lt_nvadi(uint64_t *w, uint32_t d, uint32_t a, uint32_t imm, uint32_t neg)
{
    assert(d < 256u && a < 256u);
    assert(neg < 2u);

    head(w, 0x010u, 4u);
    lt_nvput(w, 23u, 16u, d);
    lt_nvput(w, 31u, 24u, a);
    lt_nvput(w, 63u, 32u, imm);
    lt_nvput(w, 71u, 64u, LT_NV_RZ);
    lt_nvput(w, 72u, 72u, neg);
    lt_nvput(w, 79u, 77u, LT_NV_PT);
    lt_nvput(w, 80u, 80u, 1u);
    lt_nvput(w, 83u, 81u, LT_NV_PT);
    lt_nvput(w, 86u, 84u, LT_NV_PT);
    lt_nvput(w, 89u, 87u, LT_NV_PT);
    lt_nvput(w, 90u, 90u, 1u);
}

void
lt_nvmad(uint64_t *w, uint32_t d, uint32_t a, uint32_t b, uint32_t c,
         uint32_t hi, uint32_t sgn)
{
    assert(w != NULL);
    assert(hi < 2u && sgn < 2u);

    alu1(w, hi != 0u ? 0x027u : 0x024u, d, a, b, c);
    lt_nvput(w, 73u, 73u, sgn);
    lt_nvput(w, 83u, 81u, LT_NV_PT);
}

/* ---- Bitwise ---- */

static void lop(uint64_t *w, uint32_t lut)
{
    lt_nvput(w, 79u, 72u, lut);
    lt_nvput(w, 80u, 80u, 0u);
    lt_nvput(w, 83u, 81u, LT_NV_PT);
    lt_nvput(w, 89u, 87u, LT_NV_PT);
    lt_nvput(w, 90u, 90u, 1u);
}

void
lt_nvlop(uint64_t *w, uint32_t d, uint32_t a, uint32_t b, uint32_t c,
         uint32_t lut)
{
    assert(w != NULL);
    assert(lut < 256u);

    alu1(w, 0x012u, d, a, b, c);
    lop(w, lut);
}

void
lt_nvlpi(uint64_t *w, uint32_t d, uint32_t a, uint32_t imm, uint32_t lut)
{
    assert(d < 256u && a < 256u);
    assert(lut < 256u);

    head(w, 0x012u, 4u);
    lt_nvput(w, 23u, 16u, d);
    lt_nvput(w, 31u, 24u, a);
    lt_nvput(w, 63u, 32u, imm);
    lt_nvput(w, 71u, 64u, LT_NV_RZ);
    lop(w, lut);
}

void
lt_nvshf(uint64_t *w, uint32_t d, uint32_t lo, uint32_t sh, uint32_t hi,
         uint32_t ty, uint32_t rt, uint32_t dh)
{
    assert(w != NULL);
    assert(ty < 4u && rt < 2u && dh < 2u);

    alu1(w, 0x019u, d, lo, sh, hi);
    lt_nvput(w, 74u, 73u, ty);
    lt_nvput(w, 75u, 75u, 1u);
    lt_nvput(w, 76u, 76u, rt);
    lt_nvput(w, 80u, 80u, dh);
}

void
lt_nvprm(uint64_t *w, uint32_t d, uint32_t a, uint32_t sel, uint32_t b)
{
    assert(w != NULL);
    assert(sel < 256u);

    alu1(w, 0x016u, d, a, sel, b);
}

/* ---- One source ---- */

static void una(uint64_t *w, uint32_t op, uint32_t d, uint32_t s)
{
    assert(d < 256u && s < 256u);

    /* per nak sm70_encode.rs:2448 */
    head(w, op, 1u);
    lt_nvput(w, 23u, 16u, d);
    lt_nvput(w, 39u, 32u, s);
}

void
lt_nvshi(uint64_t *w, uint32_t d, uint32_t lo, uint32_t imm, uint32_t hi,
         uint32_t ty, uint32_t rt, uint32_t dh)
{
    assert(d < 256u && lo < 256u && hi < 256u);
    assert(ty < 4u && rt < 2u && dh < 2u);

    head(w, 0x019u, 4u);
    lt_nvput(w, 23u, 16u, d);
    lt_nvput(w, 31u, 24u, lo);
    lt_nvput(w, 63u, 32u, imm);
    lt_nvput(w, 71u, 64u, hi);
    lt_nvput(w, 74u, 73u, ty);
    lt_nvput(w, 75u, 75u, 1u);
    lt_nvput(w, 76u, 76u, rt);
    lt_nvput(w, 80u, 80u, dh);
}

void
lt_nvpop(uint64_t *w, uint32_t d, uint32_t s)
{
    assert(w != NULL);

    una(w, 0x109u, d, s);
}

void
lt_nvbrv(uint64_t *w, uint32_t d, uint32_t s)
{
    assert(w != NULL);

    una(w, 0x101u, d, s);
}

void
lt_nvflo(uint64_t *w, uint32_t d, uint32_t s)
{
    assert(w != NULL);

    una(w, 0x100u, d, s);
    lt_nvput(w, 83u, 81u, LT_NV_PT);
}

/* ---- Compare and select ---- */

static void stp(uint64_t *w, uint32_t pd, uint32_t cmp, uint32_t sgn,
                uint32_t lc, uint32_t ln)
{
    assert(pd < 8u && cmp < 8u);
    assert(sgn < 2u && lc < 8u && ln < 2u);

    lt_nvput(w, 70u, 68u, lc);
    lt_nvput(w, 71u, 71u, ln);
    lt_nvput(w, 73u, 73u, sgn);
    lt_nvput(w, 78u, 76u, cmp);
    lt_nvput(w, 83u, 81u, pd);
    lt_nvput(w, 86u, 84u, LT_NV_PT);
    lt_nvput(w, 89u, 87u, LT_NV_PT);
}

void
lt_nvstp(uint64_t *w, uint32_t pd, uint32_t a, uint32_t b, uint32_t cmp,
         uint32_t sgn)
{
    assert(a < 256u && b < 256u);

    head(w, 0x00Cu, 1u);
    lt_nvput(w, 31u, 24u, a);
    lt_nvput(w, 39u, 32u, b);
    stp(w, pd, cmp, sgn, LT_NV_PT, 1u);
}

void
lt_nvstx(uint64_t *w, uint32_t pd, uint32_t a, uint32_t b, uint32_t cmp,
         uint32_t sgn, uint32_t lc)
{
    assert(a < 256u && b < 256u);

    head(w, 0x00Cu, 1u);
    lt_nvput(w, 31u, 24u, a);
    lt_nvput(w, 39u, 32u, b);
    lt_nvput(w, 72u, 72u, 1u);
    stp(w, pd, cmp, sgn, lc, 0u);
}

void
lt_nvsti(uint64_t *w, uint32_t pd, uint32_t a, uint32_t imm, uint32_t cmp,
         uint32_t sgn)
{
    assert(a < 256u);

    head(w, 0x00Cu, 4u);
    lt_nvput(w, 31u, 24u, a);
    lt_nvput(w, 63u, 32u, imm);
    stp(w, pd, cmp, sgn, LT_NV_PT, 1u);
}

void
lt_nvsel(uint64_t *w, uint32_t d, uint32_t a, uint32_t b, uint32_t p,
         uint32_t nt)
{
    assert(p < 8u && nt < 2u);

    alu1(w, 0x007u, d, a, b, 0u);
    lt_nvput(w, 89u, 87u, p);
    lt_nvput(w, 90u, 90u, nt);
}

void
lt_nvsei(uint64_t *w, uint32_t d, uint32_t a, uint32_t imm, uint32_t p,
         uint32_t nt)
{
    assert(d < 256u && a < 256u);
    assert(p < 8u && nt < 2u);

    head(w, 0x007u, 4u);
    lt_nvput(w, 23u, 16u, d);
    lt_nvput(w, 31u, 24u, a);
    lt_nvput(w, 63u, 32u, imm);
    lt_nvput(w, 89u, 87u, p);
    lt_nvput(w, 90u, 90u, nt);
}

void
lt_nvstc(uint64_t *w, uint32_t pd, uint32_t a, uint32_t b, uint32_t cmp,
         uint32_t acc, uint32_t an)
{
    assert(a < 256u && b < 256u);
    assert(acc < 8u && an < 2u);

    head(w, 0x00Cu, 1u);
    lt_nvput(w, 31u, 24u, a);
    lt_nvput(w, 39u, 32u, b);
    stp(w, pd, cmp, 0u, LT_NV_PT, 1u);
    lt_nvput(w, 89u, 87u, acc);
    lt_nvput(w, 90u, 90u, an);
}

/* ---- Memory ---- */

void
lt_nvstg(uint64_t *w, uint32_t ra, uint32_t rb, uint32_t off, uint32_t ty,
         uint32_t ord)
{
    assert(ra < 256u && rb < 256u && ty < 8u);
    assert(ord < 16u);

    head(w, 0x986u, 0u);
    lt_nvput(w, 31u, 24u, ra);
    lt_nvput(w, 39u, 32u, rb);
    lt_nvput(w, 63u, 40u, off);
    lt_nvput(w, 69u, 64u, LT_NV_URZ);
    lt_nvput(w, 72u, 72u, 1u);
    lt_nvput(w, 75u, 73u, ty);
    lt_nvput(w, 80u, 77u, ord);   /* per nak sm70_encode.rs:3272 */
    lt_nvput(w, 86u, 84u, 1u);
    lt_nvput(w, 90u, 90u, 1u);
    lt_nvput(w, 91u, 91u, 1u);
}

void
lt_nvldg(uint64_t *w, uint32_t dst, uint32_t ra, uint32_t off, uint32_t ty,
         uint32_t ord)
{
    assert(dst < 256u && ra < 256u && ty < 8u);
    assert(ord < 16u);

    head(w, 0x981u, 0u);
    lt_nvput(w, 23u, 16u, dst);
    lt_nvput(w, 31u, 24u, ra);
    lt_nvput(w, 37u, 32u, LT_NV_URZ);
    lt_nvput(w, 63u, 40u, off);
    lt_nvput(w, 72u, 72u, 1u);
    lt_nvput(w, 75u, 73u, ty);
    lt_nvput(w, 80u, 77u, ord);   /* per nak sm70_encode.rs:3272 */
    lt_nvput(w, 83u, 81u, LT_NV_PT);
    lt_nvput(w, 86u, 84u, 1u);
    lt_nvput(w, 90u, 90u, 1u);
    lt_nvput(w, 91u, 91u, 1u);
}

void
lt_nvldl(uint64_t *w, uint32_t dst, uint32_t ra, uint32_t off, uint32_t ty)
{
    assert(dst < 256u && ra < 256u && ty < 8u);

    head(w, 0x983u, 0u);
    lt_nvput(w, 23u, 16u, dst);
    lt_nvput(w, 31u, 24u, ra);
    lt_nvput(w, 63u, 40u, off);
    lt_nvput(w, 75u, 73u, ty);
    lt_nvput(w, 86u, 84u, 1u);
}

void
lt_nvstl(uint64_t *w, uint32_t ra, uint32_t rb, uint32_t off, uint32_t ty)
{
    assert(ra < 256u && rb < 256u && ty < 8u);

    head(w, 0x387u, 0u);
    lt_nvput(w, 31u, 24u, ra);
    lt_nvput(w, 39u, 32u, rb);
    lt_nvput(w, 63u, 40u, off);
    lt_nvput(w, 75u, 73u, ty);
    lt_nvput(w, 86u, 84u, 1u);
}

void
lt_nvlds(uint64_t *w, uint32_t dst, uint32_t ra, uint32_t off, uint32_t ty)
{
    assert(dst < 256u && ra < 256u && ty < 8u);

    head(w, 0x984u, 0u);
    lt_nvput(w, 23u, 16u, dst);
    lt_nvput(w, 31u, 24u, ra);
    lt_nvput(w, 63u, 40u, off);
    lt_nvput(w, 75u, 73u, ty);
}

void
lt_nvsts(uint64_t *w, uint32_t ra, uint32_t rb, uint32_t off, uint32_t ty)
{
    assert(ra < 256u && rb < 256u && ty < 8u);

    head(w, 0x388u, 0u);
    lt_nvput(w, 31u, 24u, ra);
    lt_nvput(w, 39u, 32u, rb);
    lt_nvput(w, 63u, 40u, off);
    lt_nvput(w, 75u, 73u, ty);
}

/* ---- Lanes ---- */

void
lt_nvvot(uint64_t *w, uint32_t d, uint32_t pd, uint32_t p, uint32_t nt,
         uint32_t vop)
{
    assert(d < 256u && pd < 8u && p < 8u);
    assert(nt < 2u && vop < 4u);

    /* per nak sm70_encode.rs:4465 */
    head(w, 0x806u, 0u);
    lt_nvput(w, 23u, 16u, d);
    lt_nvput(w, 73u, 72u, vop);
    lt_nvput(w, 83u, 81u, pd);
    lt_nvput(w, 89u, 87u, p);
    lt_nvput(w, 90u, 90u, nt);
}

void
lt_nvsfl(uint64_t *w, uint32_t d, uint32_t pd, uint32_t ra, uint32_t rb,
         uint32_t ic, uint32_t smod)
{
    assert(d < 256u && pd < 8u);
    assert(ra < 256u && rb < 256u);
    assert(ic < 0x2000u && smod < 4u);

    head(w, 0x189u, 2u);
    lt_nvput(w, 23u, 16u, d);
    lt_nvput(w, 31u, 24u, ra);
    lt_nvput(w, 39u, 32u, rb);
    lt_nvput(w, 52u, 40u, ic);
    lt_nvput(w, 59u, 58u, smod);
    lt_nvput(w, 83u, 81u, pd);
}

void
lt_nvatg(uint64_t *w, uint32_t dst, uint32_t ra, uint32_t rb, uint32_t off,
         uint32_t aop, uint32_t aty, uint32_t ord)
{
    assert(dst < 256u && ra < 256u && ord < 16u);
    assert(rb < 256u && aop < 16u && aty < 8u);

    head(w, 0x9A8u, 0u);
    lt_nvput(w, 23u, 16u, dst);
    lt_nvput(w, 31u, 24u, ra);
    lt_nvput(w, 39u, 32u, rb);
    lt_nvput(w, 63u, 40u, off);
    lt_nvput(w, 69u, 64u, LT_NV_URZ);
    lt_nvput(w, 70u, 70u, 1u);
    lt_nvput(w, 72u, 72u, 1u);
    lt_nvput(w, 75u, 73u, aty);
    lt_nvput(w, 80u, 77u, ord);   /* per nak sm70_encode.rs:3272 */
    lt_nvput(w, 83u, 81u, LT_NV_PT);
    lt_nvput(w, 86u, 84u, 1u);
    lt_nvput(w, 90u, 87u, aop);
    lt_nvput(w, 91u, 91u, 1u);
}

void
lt_nvatc(uint64_t *w, uint32_t dst, uint32_t ra, uint32_t rb, uint32_t rc,
         uint32_t off, uint32_t aty, uint32_t ord)
{
    assert(dst < 256u && ra < 256u && ord < 16u);
    assert(rb < 256u && rc < 256u && aty < 8u);

    head(w, 0x3A9u, 0u);
    lt_nvput(w, 23u, 16u, dst);
    lt_nvput(w, 31u, 24u, ra);
    lt_nvput(w, 39u, 32u, rb);
    lt_nvput(w, 63u, 40u, off);
    lt_nvput(w, 71u, 64u, rc);
    lt_nvput(w, 72u, 72u, 1u);
    lt_nvput(w, 75u, 73u, aty);
    lt_nvput(w, 80u, 77u, ord);   /* per nak sm70_encode.rs:3272 */
    lt_nvput(w, 83u, 81u, LT_NV_PT);
    lt_nvput(w, 86u, 84u, 1u);
}

void
lt_nvmbr(uint64_t *w, uint32_t scp)
{
    assert(w != NULL);
    assert(scp < 8u);

    head(w, 0x992u, 0u);
    lt_nvput(w, 78u, 76u, scp);
}

void
lt_nvrdx(uint64_t *w, uint32_t ud, uint32_t ra, uint32_t rop, uint32_t sgn)
{
    assert(ud < 64u && ra < 256u);
    assert(rop < 8u && sgn < 2u);

    head(w, 0x3C4u, 0u);
    lt_nvput(w, 21u, 16u, ud);
    lt_nvput(w, 31u, 24u, ra);
    lt_nvput(w, 73u, 73u, sgn);
    lt_nvput(w, 80u, 78u, rop);
}

/* ---- Scoreboards ---- */

void
lt_nvbar(uint64_t *w, uint32_t idx)
{
    assert(w != NULL);
    assert(idx < LT_NV_NBAR);

    lt_nvput(w, 112u, 110u, idx);
}

void
lt_nvrb(uint64_t *w, uint32_t idx)
{
    assert(w != NULL);
    assert(idx < LT_NV_NBAR);

    lt_nvput(w, 115u, 113u, idx);
}

void
lt_nvwait(uint64_t *w, uint32_t mask)
{
    assert(w != NULL);
    assert(mask < 64u);

    lt_nvput(w, 121u, 116u, mask);
}

/* ---- Control ---- */

void
lt_nvsyn(uint64_t *w)
{
    assert(w != NULL);

    head(w, 0xB1Du, 0u);
    lt_nvput(w, 109u, 109u, 1u);
}

void
lt_nvbrs(uint64_t *w, uint32_t rb, uint32_t cnt)
{
    assert(w != NULL);
    assert(rb < 256u && cnt < 4096u);

    head(w, 0x51Du, 0u);
    /* per probe obj/nvx/barsr.cubin */
    lt_nvput(w, 39u, 32u, rb);
    lt_nvput(w, 53u, 42u, cnt);
    lt_nvput(w, 80u, 80u, 1u);
    lt_nvput(w, 109u, 109u, 1u);
}

void
lt_nvexit(uint64_t *w)
{
    assert(w != NULL);

    head(w, 0x94Du, 0u);
    lt_nvput(w, 89u, 87u, LT_NV_PT);
}

void
lt_nvbpt(uint64_t *w)
{
    assert(w != NULL);

    /* per probe obj/nvx/trapp.cubin */
    head(w, 0x15Cu, 4u);
    lt_nvput(w, 39u, 32u, 4u);
    lt_nvput(w, 85u, 84u, 3u);
    lt_nvput(w, 89u, 87u, LT_NV_PT);
}

void
lt_nvbrp(uint64_t *w, int32_t rel, uint32_t p, uint32_t nt)
{
    assert(w != NULL);
    assert(p < 8u && nt < 2u);

    head(w, 0x947u, 0u);
    lt_nvput(w, 81u, 34u, (uint64_t)(int64_t)rel & (uint64_t)0xFFFFFFFFFFFF);
    lt_nvput(w, 89u, 87u, p);
    lt_nvput(w, 90u, 90u, nt);
}

void
lt_nvbra(uint64_t *w, int32_t rel)
{
    assert(w != NULL);

    lt_nvbrp(w, rel, LT_NV_PT, 0u);
}

void
lt_nvnop(uint64_t *w)
{
    assert(w != NULL);

    head(w, 0x918u, 0u);
}

/* ---- Convergence ---- */

void
lt_nvbsy(uint64_t *w, uint32_t bar, int32_t rel)
{
    assert(w != NULL);
    assert(bar < LT_NV_NJB);

    /* per nak sm70_encode.rs:4165 */
    head(w, 0x945u, 0u);
    lt_nvput(w, 19u, 16u, bar);
    lt_nvput(w, 63u, 34u, (uint64_t)(int64_t)rel & (uint64_t)0x3FFFFFFF);
    lt_nvput(w, 89u, 87u, LT_NV_PT);
}

void
lt_nvbsn(uint64_t *w, uint32_t bar)
{
    assert(w != NULL);
    assert(bar < LT_NV_NJB);

    /* per nak sm70_encode.rs:4179 */
    head(w, 0x941u, 0u);
    lt_nvput(w, 19u, 16u, bar);
    lt_nvput(w, 89u, 87u, LT_NV_PT);
}

void
lt_nvbrk(uint64_t *w, uint32_t bar, uint32_t p, uint32_t nt)
{
    assert(bar < LT_NV_NJB);
    assert(p < 8u && nt < 2u);

    /* per nak sm70_encode.rs:4152 */
    head(w, 0x942u, 0u);
    lt_nvput(w, 19u, 16u, bar);
    lt_nvput(w, 89u, 87u, p);
    lt_nvput(w, 90u, 90u, nt);
}

/* ---- Float ---- */

void
lt_nvfad(uint64_t *w, uint32_t d, uint32_t a, uint32_t b, uint32_t neg)
{
    assert(w != NULL);
    assert(d < 256u && neg < 2u);

    alu1(w, 0x021u, d, a, b, 0u);
    lt_nvput(w, 63u, 63u, neg);
}

void
lt_nvfmu(uint64_t *w, uint32_t d, uint32_t a, uint32_t b)
{
    assert(w != NULL);
    assert(d < 256u);

    alu1(w, 0x020u, d, a, b, LT_NV_RZ);
    lt_nvput(w, 86u, 84u, 4u);
}

void
lt_nvffm(uint64_t *w, uint32_t d, uint32_t a, uint32_t b, uint32_t c)
{
    assert(w != NULL);
    assert(d < 256u);

    alu1(w, 0x023u, d, a, b, c);
}

static void ftp(uint64_t *w, uint32_t pd, uint32_t cmp)
{
    assert(pd < 8u && cmp < 16u);

    lt_nvput(w, 79u, 76u, cmp);
    lt_nvput(w, 83u, 81u, pd);
    lt_nvput(w, 86u, 84u, LT_NV_PT);
    lt_nvput(w, 89u, 87u, LT_NV_PT);
}

void
lt_nvftp(uint64_t *w, uint32_t pd, uint32_t a, uint32_t b, uint32_t cmp)
{
    assert(a < 256u && b < 256u);

    head(w, 0x00Bu, 1u);
    lt_nvput(w, 31u, 24u, a);
    lt_nvput(w, 39u, 32u, b);
    ftp(w, pd, cmp);
}

void
lt_nvfti(uint64_t *w, uint32_t pd, uint32_t a, uint32_t imm, uint32_t cmp)
{
    assert(a < 256u);

    head(w, 0x00Bu, 4u);
    lt_nvput(w, 31u, 24u, a);
    lt_nvput(w, 63u, 32u, imm);
    ftp(w, pd, cmp);
}

/* ---- Double ---- */

void
lt_nvdad(uint64_t *w, uint32_t d, uint32_t a, uint32_t b, uint32_t neg)
{
    assert(d < 256u && a < 256u);
    assert(b < 256u && neg < 2u);

    head(w, 0x029u, 1u);
    lt_nvput(w, 23u, 16u, d);
    lt_nvput(w, 31u, 24u, a);
    lt_nvput(w, 71u, 64u, b);
    lt_nvput(w, 75u, 75u, neg);
}

void
lt_nvdmu(uint64_t *w, uint32_t d, uint32_t a, uint32_t b)
{
    assert(d < 256u && a < 256u && b < 256u);

    head(w, 0x028u, 1u);
    lt_nvput(w, 23u, 16u, d);
    lt_nvput(w, 31u, 24u, a);
    lt_nvput(w, 39u, 32u, b);
}

void
lt_nvdfm(uint64_t *w, uint32_t d, uint32_t a, uint32_t b, uint32_t c,
         uint32_t nga)
{
    assert(d < 256u && a < 256u);
    assert(b < 256u && c < 256u && nga < 2u);

    head(w, 0x02Bu, 1u);
    lt_nvput(w, 23u, 16u, d);
    lt_nvput(w, 31u, 24u, a);
    lt_nvput(w, 39u, 32u, b);
    lt_nvput(w, 71u, 64u, c);
    lt_nvput(w, 72u, 72u, nga);
}

void
lt_nvdtp(uint64_t *w, uint32_t pd, uint32_t a, uint32_t b, uint32_t cmp)
{
    assert(a < 256u && b < 256u);

    head(w, 0x02Au, 1u);
    lt_nvput(w, 31u, 24u, a);
    lt_nvput(w, 39u, 32u, b);
    ftp(w, pd, cmp);
}

void
lt_nvmuf(uint64_t *w, uint32_t d, uint32_t s, uint32_t op)
{
    assert(d < 256u && s < 256u);
    assert(op < 64u);

    /* per nak sm70_encode.rs:1205 */
    head(w, 0x108u, 1u);
    lt_nvput(w, 23u, 16u, d);
    lt_nvput(w, 39u, 32u, s);
    lt_nvput(w, 79u, 74u, op);
}

void
lt_nvsrm(uint64_t *w, uint32_t rnd)
{
    assert(w != NULL);
    assert(rnd < 4u);

    /* per nak sm70_encode.rs:765 */
    lt_nvput(w, 79u, 78u, rnd);
}

void
lt_nvfrn(uint64_t *w, uint32_t d, uint32_t s, uint32_t rnd)
{
    assert(d < 256u && s < 256u);
    assert(rnd < 4u);

    head(w, 0x107u, 1u);
    lt_nvput(w, 23u, 16u, d);
    lt_nvput(w, 39u, 32u, s);
    lt_nvput(w, 76u, 75u, 2u);
    lt_nvput(w, 79u, 78u, rnd);
    lt_nvput(w, 85u, 84u, 2u);
}

void
lt_nvf2f(uint64_t *w, uint32_t d, uint32_t s, uint32_t dsz, uint32_t ssz,
         uint32_t rnd)
{
    assert(d < 256u && s < 256u);
    assert(dsz < 4u && ssz < 4u && rnd < 4u);

    /* per nak sm70_encode.rs:2189 */
    head(w, (dsz < 3u && ssz < 3u) ? 0x104u : 0x110u, 1u);
    lt_nvput(w, 23u, 16u, d);
    lt_nvput(w, 39u, 32u, s);
    lt_nvput(w, 76u, 75u, dsz);
    lt_nvput(w, 79u, 78u, rnd);
    lt_nvput(w, 85u, 84u, ssz);
}

void
lt_nvhfm(uint64_t *w, uint32_t d, uint32_t a, uint32_t b, uint32_t c)
{
    assert(w != NULL);
    assert(d < 256u);

    alu1(w, 0x031u, d, a, b, c);
}

void
lt_nvi2f(uint64_t *w, uint32_t d, uint32_t s, uint32_t sgn, uint32_t dsz,
         uint32_t ssz)
{
    assert(d < 256u && s < 256u);
    assert(sgn < 2u && dsz < 4u && ssz < 4u);

    /* per nak sm70_encode.rs:2282 */
    head(w, (dsz < 3u && ssz < 3u) ? 0x106u : 0x112u, 1u);
    lt_nvput(w, 23u, 16u, d);
    lt_nvput(w, 39u, 32u, s);
    lt_nvput(w, 74u, 74u, sgn);
    lt_nvput(w, 76u, 75u, dsz);
    lt_nvput(w, 85u, 84u, ssz);
}

void
lt_nvf2i(uint64_t *w, uint32_t d, uint32_t s, uint32_t sgn, uint32_t dsz,
         uint32_t ssz)
{
    assert(d < 256u && s < 256u);
    assert(sgn < 2u && dsz < 4u && ssz < 4u);

    /* per nak sm70_encode.rs:2282 */
    head(w, (dsz < 3u && ssz < 3u) ? 0x105u : 0x111u, 1u);
    lt_nvput(w, 23u, 16u, d);
    lt_nvput(w, 39u, 32u, s);
    lt_nvput(w, 72u, 72u, sgn);
    lt_nvput(w, 76u, 75u, dsz);
    lt_nvput(w, 79u, 78u, 3u);
    lt_nvput(w, 85u, 84u, ssz);
}
