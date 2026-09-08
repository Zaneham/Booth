/*
 * lt_nvcf.c — where a warp's lanes part and where they meet again.
 *
 * (c) 2026 Zane Hambly.
 */

#include "lt_nv.h"
#include <string.h>
#include <assert.h>

/* ---- Block sets ---- */

static void bsclr(lt_nbs_t *s)
{
    assert(s != NULL);
    assert(sizeof s->w == LT_NV_NBW * 8u);

    memset(s, 0, sizeof *s);
}

static void bsall(lt_nbs_t *s)
{
    assert(s != NULL);
    assert(sizeof s->w == LT_NV_NBW * 8u);

    memset(s, 0xFF, sizeof *s);
}

static void bsadd(lt_nbs_t *s, uint32_t k)
{
    assert(s != NULL);
    assert(k < LT_MAX_NVBB);

    s->w[k >> 6] |= 1ull << (k & 63u);
}

static int bshas(const lt_nbs_t *s, uint32_t k)
{
    assert(s != NULL);
    assert(k < LT_MAX_NVBB);

    return (int)((s->w[k >> 6] >> (k & 63u)) & 1ull);
}

static void bsand(lt_nbs_t *d, const lt_nbs_t *s)
{
    assert(d != NULL && s != NULL);

    for (uint32_t i = 0; i < LT_NV_NBW; i++) d->w[i] &= s->w[i];
}

static int bseq(const lt_nbs_t *a, const lt_nbs_t *b)
{
    assert(a != NULL && b != NULL);

    return memcmp(a, b, sizeof *a) == 0;
}

static int bsfull(const lt_nbs_t *s)
{
    assert(s != NULL);

    for (uint32_t i = 0; i < LT_NV_NBW; i++)
        if (s->w[i] != ~0ull) return 0;
    return 1;
}

static uint32_t bspop(const lt_nbs_t *s)
{
    uint32_t n = 0u;

    assert(s != NULL);

    for (uint32_t i = 0; i < LT_NV_NBW; i++) {
        uint64_t x = s->w[i];

        for (uint32_t g = 0; g < 64u && x != 0ull; g++) {
            x &= x - 1ull;
            n++;
        }
    }
    return n;
}

/* ---- Edges ---- */

static void preds(lt_ncf_t *c, int fin)
{
    assert(c != NULL);
    assert(c->nbb <= LT_MAX_NVBB);

    memset(c->po, 0, sizeof c->po);
    for (uint32_t k = 0; k < c->nbb; k++) {
        if (fin && c->fw[k] != 0u) continue;
        if (c->s0[k] < c->nbb) c->po[c->s0[k] + 1u]++;
        if (c->s1[k] < c->nbb) c->po[c->s1[k] + 1u]++;
    }
    for (uint32_t k = 0; k < c->nbb; k++) {
        c->po[k + 1u] = (uint16_t)(c->po[k + 1u] + c->po[k]);
        c->pc[k]      = c->po[k];
    }
    for (uint32_t k = 0; k < c->nbb; k++) {
        if (fin && c->fw[k] != 0u) continue;
        if (c->s0[k] < c->nbb) c->pe[c->pc[c->s0[k]]++] = (uint16_t)k;
        if (c->s1[k] < c->nbb) c->pe[c->pc[c->s1[k]]++] = (uint16_t)k;
    }
}

static uint32_t npred(const lt_ncf_t *c, uint32_t k)
{
    uint32_t n = 0u, last = LT_NV_NOIX;

    assert(c != NULL);
    assert(k < c->nbb);

    for (uint32_t i = c->po[k]; i < c->po[k + 1u]; i++) {
        if (c->pe[i] == last) continue;
        last = c->pe[i];
        n++;
    }
    return n;
}

static int isfwd(const lt_ncf_t *c, uint32_t k)
{
    assert(c != NULL);
    assert(k < c->nbb);

    return c->bs[k + 1u] - c->bs[k] == 1u && c->tk[k] == (uint8_t)TBR
        && c->s0[k] < c->nbb && c->s0[k] != k && npred(c, k) == 1u;
}

static void resolv(lt_ncf_t *c)
{
    assert(c != NULL);
    assert(c->nbb <= LT_MAX_NVBB);

    for (uint32_t k = 0; k < c->nbb; k++) c->fw[k] = (uint8_t)isfwd(c, k);
    for (uint32_t k = 0; k < c->nbb; k++) {
        for (uint32_t e = 0; e < 2u; e++) {
            uint32_t t   = (e == 0u) ? c->s0[k] : c->s1[k];
            uint32_t via = k;

            for (uint32_t g = 0; g < c->nbb; g++) {
                if (t >= c->nbb || c->fw[t] == 0u) break;
                via = t;
                t   = c->s0[t];
            }
            if (e == 0u) { c->s0[k] = (uint16_t)t; c->v0[k] = (uint16_t)via; }
            else         { c->s1[k] = (uint16_t)t; c->v1[k] = (uint16_t)via; }
        }
    }
}

static uint32_t sof(const lt_ncf_t *c, uint32_t k, uint32_t e)
{
    assert(c != NULL);
    assert(k < c->nbb && e < 2u);

    return (e == 0u) ? c->s0[k] : c->s1[k];
}

static uint32_t vof(const lt_ncf_t *c, uint32_t k, uint32_t e)
{
    assert(c != NULL);
    assert(k < c->nbb && e < 2u);

    return (e == 0u) ? c->v0[k] : c->v1[k];
}

static int jumps(const lt_ncf_t *c, uint32_t k, uint32_t e)
{
    assert(c != NULL);
    assert(k < c->nbb && e < 2u);

    if (vof(c, k, e) != k) return 1;
    return e == 0u && (c->tk[k] == (uint8_t)TBR || c->tk[k] == (uint8_t)TBRC);
}

/* ---- Dominators ---- */

static int reach(const lt_ncf_t *c, uint32_t k)
{
    assert(c != NULL);
    assert(k < c->nbb);

    return !bsfull(&c->dom[k]);
}

static void doms(lt_ncf_t *c)
{
    assert(c != NULL);
    assert(c->nbb <= LT_MAX_NVBB);

    for (uint32_t k = 0; k < c->nbb; k++) bsall(&c->dom[k]);
    bsclr(&c->dom[0]);
    bsadd(&c->dom[0], 0u);

    for (uint32_t r = 0; r <= c->nbb; r++) {
        uint32_t ch = 0u;

        for (uint32_t k = 1; k < c->nbb; k++) {
            bsall(&c->tmp);
            for (uint32_t i = c->po[k]; i < c->po[k + 1u]; i++)
                bsand(&c->tmp, &c->dom[c->pe[i]]);
            bsadd(&c->tmp, k);
            if (bseq(&c->tmp, &c->dom[k])) continue;
            c->dom[k] = c->tmp;
            ch = 1u;
        }
        if (ch == 0u) break;
    }
}

static void idoms(lt_ncf_t *c)
{
    assert(c != NULL);
    assert(c->nbb <= LT_MAX_NVBB);

    for (uint32_t k = 0; k < c->nbb; k++) {
        uint32_t best = LT_NV_NOIX, bn = 0u;

        c->idom[k] = (uint16_t)LT_NV_NOIX;
        if (k == 0u || !reach(c, k)) continue;
        for (uint32_t j = 0; j < c->nbb; j++) {
            uint32_t n;

            if (j == k || !bshas(&c->dom[k], j) || !reach(c, j)) continue;
            n = bspop(&c->dom[j]);
            if (n > bn) { bn = n; best = j; }
        }
        c->idom[k] = (uint16_t)best;
    }
}

static uint32_t nfwd(const lt_ncf_t *c, uint32_t k)
{
    uint32_t n = 0u, last = LT_NV_NOIX;

    assert(c != NULL);
    assert(k < c->nbb);

    for (uint32_t i = c->po[k]; i < c->po[k + 1u]; i++) {
        uint32_t p = c->pe[i];

        if (p == last || !reach(c, p) || bshas(&c->dom[p], k)) continue;
        last = p;
        n++;
    }
    return n;
}

static void merges(lt_ncf_t *c)
{
    assert(c != NULL);
    assert(c->nbb <= LT_MAX_NVBB);

    memset(c->ism, 0, sizeof c->ism);
    for (uint32_t k = 0; k < c->nbb; k++) {
        uint32_t best = LT_NV_NOIX;

        c->mof[k] = (uint16_t)LT_NV_NOIX;
        if (c->tk[k] != (uint8_t)TBRC || !reach(c, k)) continue;
        for (uint32_t j = 0; j < c->nbb; j++) {
            if (!reach(c, j) || c->idom[j] != k || nfwd(c, j) < 2u) continue;
            if (j < best) best = j;
        }
        c->mof[k] = (uint16_t)best;
        if (best < c->nbb) c->ism[best] = 1u;
    }
}

/* ---- Loops ---- */

static int isback(const lt_ncf_t *c, uint32_t x, uint32_t h)
{
    assert(c != NULL);
    assert(x < c->nbb && h < c->nbb);

    if (!reach(c, x) || !bshas(&c->dom[x], h)) return 0;
    return c->s0[x] == h || c->s1[x] == h;
}

static void loop1(lt_ncf_t *c, uint32_t h)
{
    lt_nbs_t *L = &c->lb[h];
    uint32_t  any = 0u;

    assert(c != NULL);
    assert(h < c->nbb);

    bsclr(L);
    c->lsz[h] = 0u;
    for (uint32_t x = 0; x < c->nbb; x++)
        if (isback(c, x, h)) { bsadd(L, x); any = 1u; }
    if (any == 0u || !reach(c, h)) return;
    bsadd(L, h);

    for (uint32_t r = 0; r <= c->nbb; r++) {
        uint32_t ch = 0u;

        for (uint32_t y = 0; y < c->nbb; y++) {
            if (y == h || !bshas(L, y)) continue;
            for (uint32_t i = c->po[y]; i < c->po[y + 1u]; i++) {
                uint32_t p = c->pe[i];

                if (bshas(L, p) || !reach(c, p)) continue;
                bsadd(L, p);
                ch = 1u;
            }
        }
        if (ch == 0u) break;
    }
    c->lsz[h] = (uint16_t)bspop(L);
}

static void nest(lt_ncf_t *c, uint32_t h)
{
    assert(c != NULL);
    assert(h < c->nbb && c->lsz[h] != 0u);

    for (uint32_t k = 0; k < c->nbb; k++) {
        if (k == h || !bshas(&c->lb[h], k)) continue;
        if (c->lp[k] == LT_NV_NOIX || c->lsz[c->lp[k]] > c->lsz[h])
            c->lp[k] = (uint16_t)h;
        if (c->lsz[k] == 0u) continue;
        if (c->lpar[k] == LT_NV_NOIX || c->lsz[c->lpar[k]] > c->lsz[h])
            c->lpar[k] = (uint16_t)h;
    }
}

static void loops(lt_ncf_t *c)
{
    uint32_t n = 0u;

    assert(c != NULL);
    assert(c->nbb <= LT_MAX_NVBB);

    for (uint32_t h = 0; h < c->nbb; h++) loop1(c, h);
    memset(c->lp, 0xFF, sizeof c->lp);
    memset(c->lpar, 0xFF, sizeof c->lpar);
    for (uint32_t h = 0; h < c->nbb; h++)
        if (c->lsz[h] != 0u) nest(c, h);
    for (uint32_t h = 0; h < c->nbb; h++) {
        uint32_t i;

        if (c->lsz[h] == 0u) continue;
        c->lp[h] = (uint16_t)h;
        for (i = n; i > 0u && c->lsz[c->ord[i - 1u]] > c->lsz[h]; i--)
            c->ord[i] = c->ord[i - 1u];
        c->ord[i] = (uint16_t)h;
        n++;
    }
    c->ord[n] = (uint16_t)LT_NV_NOIX;
}

static lt_res_t backs(lt_ncf_t *c)
{
    assert(c != NULL);
    assert(c->nbb <= LT_MAX_NVBB);

    memset(c->back, 0, sizeof c->back);
    for (uint32_t x = 0; x < c->nbb; x++) {
        c->tgt[x] = c->s0[x];
        for (uint32_t e = 0; e < 2u; e++) {
            uint32_t h = sof(c, x, e), via = vof(c, x, e);

            if (h >= c->nbb || !isback(c, x, h)) continue;
            if (!jumps(c, x, e))
                return LT_WHY(LT_W_NOJOIN, LT_EHOLE, NULL, c->bs[x], x, 5u);
            c->back[via] = 1u;
            c->tgt[via]  = (uint16_t)h;
        }
    }
    return LT_OK;
}

static int inloop(const lt_ncf_t *c, uint32_t k, uint32_t h)
{
    uint32_t x;

    assert(c != NULL);
    assert(k < c->nbb && h < c->nbb);

    x = c->lp[k];
    for (uint32_t g = 0; g <= c->nbb && x != LT_NV_NOIX; g++) {
        if (x == h) return 1;
        x = c->lpar[x];
    }
    return 0;
}

/* ---- Regions ---- */

static uint32_t addrg(lt_ncf_t *c, uint32_t j, uint32_t at, uint32_t kind)
{
    lt_nrg_t *r;

    assert(c != NULL);
    assert(at < c->nbb && kind <= LT_NV_KCONT);

    if (c->nrg >= LT_MAX_NVRG) return LT_NV_NOIX;
    r = &c->rg[c->nrg];
    memset(r, 0, sizeof *r);
    r->j    = (uint16_t)j;
    r->at   = (uint16_t)at;
    r->kind = (uint8_t)kind;
    r->stub = (uint16_t)LT_NV_NOIX;
    r->lo   = (uint16_t)at;
    r->hi   = (uint16_t)at;
    if (j < c->nbb) {
        c->isj[j] = 1u;
        if (j < r->lo) r->lo = (uint16_t)j;
        if (j > r->hi) r->hi = (uint16_t)j;
    }
    return c->nrg++;
}

static lt_res_t leaves(lt_ncf_t *c, uint32_t r)
{
    const lt_nrg_t *g = &c->rg[r];

    assert(c != NULL);
    assert(r < c->nrg);

    for (uint32_t x = 0; x < c->nbb; x++) {
        if (!bshas(&c->body, x)) continue;
        if (x < c->rg[r].lo) c->rg[r].lo = (uint16_t)x;
        if (x > c->rg[r].hi) c->rg[r].hi = (uint16_t)x;
        for (uint32_t e = 0; e < 2u; e++) {
            uint32_t t = sof(c, x, e);

            if (t >= c->nbb || t == g->j) continue;
            if (bshas(&c->body, t) && (t != g->at || g->kind != LT_NV_KIF))
                continue;
            if (c->nbk >= LT_MAX_NVBK)
                return LT_WHY(LT_W_NOJOIN, LT_EHOLE, NULL, c->bs[x], c->nbk,
                              LT_MAX_NVBK);
            c->bk[c->nbk].via   = (uint16_t)vof(c, x, e);
            c->bk[c->nbk].rg    = (uint16_t)r;
            c->bk[c->nbk].taken = (uint8_t)jumps(c, x, e);
            c->nbk++;
        }
    }
    return LT_OK;
}

static int gather(const lt_ncf_t *c, uint32_t p, uint32_t q)
{
    assert(c != NULL);
    assert(p < c->nbb && q < c->nbb);

    for (uint32_t r = 0; r < c->nrg; r++) {
        uint32_t e = c->rg[r].at, j = c->rg[r].j;

        if (j >= c->nbb || !bshas(&c->dom[p], e) || bshas(&c->dom[p], j))
            continue;
        if (bshas(&c->dom[q], j)) return 1;
    }
    return 0;
}

static int serial(const lt_ncf_t *c, uint32_t p, uint32_t q, uint32_t m)
{
    assert(c != NULL);
    assert(p < c->nbb && q < c->nbb);

    if (p == q) return 1;
    if (m < c->nbb && bshas(&c->dom[q], p) && bshas(&c->dom[q], m)) return 1;
    if (m < c->nbb && bshas(&c->dom[p], q) && bshas(&c->dom[p], m)) return 1;
    return gather(c, p, q) || gather(c, q, p);
}

static int conv(const lt_ncf_t *c, uint32_t e, uint32_t m)
{
    assert(c != NULL);
    assert(e < c->nbb);

    if (e == 0u || c->ism[e] != 0u || c->isj[e] != 0u) return 1;
    for (uint32_t i = c->po[e]; i < c->po[e + 1u]; i++) {
        uint32_t p = c->pe[i];

        if (!reach(c, p) || bshas(&c->dom[p], e)) continue;
        for (uint32_t k = c->po[e]; k < c->po[e + 1u]; k++) {
            uint32_t q = c->pe[k];

            if (!reach(c, q) || bshas(&c->dom[q], e)) continue;
            if (!serial(c, p, q, m)) return 0;
        }
    }
    return 1;
}

static int calm(const lt_ncf_t *c, uint32_t x, uint32_t h)
{
    uint32_t cur = x;

    assert(c != NULL);
    assert(x < c->nbb && h < c->nbb);

    for (uint32_t g = 0; g <= c->nbb; g++) {
        if (cur == h || c->ism[cur] != 0u || c->isj[cur] != 0u) return 1;
        if (npred(c, cur) != 1u) return 0;
        cur = c->pe[c->po[cur]];
    }
    return 0;
}

static uint32_t lexit(const lt_ncf_t *c, uint32_t h)
{
    uint32_t j = LT_NV_NOIX;

    assert(c != NULL);
    assert(h < c->nbb);

    for (uint32_t x = 0; x < c->nbb; x++) {
        if (!bshas(&c->lb[h], x)) continue;
        for (uint32_t e = 0; e < 2u; e++) {
            uint32_t t = sof(c, x, e);

            if (t >= c->nbb || bshas(&c->lb[h], t)) continue;
            if (c->tk[t] == (uint8_t)TEXIT && c->bs[t + 1u] - c->bs[t] == 1u)
                continue;
            if (j == LT_NV_NOIX) j = t;
            else if (j != t) return LT_NV_BEXIT;
        }
    }
    return j;
}

static int guard(const lt_ncf_t *c, uint32_t h, uint32_t j)
{
    uint32_t p = LT_NV_NOIX, n = 0u;

    assert(c != NULL);
    assert(h < c->nbb);

    for (uint32_t i = c->po[h]; i < c->po[h + 1u]; i++) {
        if (bshas(&c->lb[h], c->pe[i])) continue;
        p = c->pe[i];
        n++;
    }
    return n == 1u && c->tk[p] == (uint8_t)TBRC && c->mof[p] == j;
}

static lt_res_t lregn(lt_ncf_t *c, uint32_t h)
{
    uint32_t j = lexit(c, h), r;

    assert(c != NULL);
    assert(h < c->nbb);

    if (j == LT_NV_BEXIT)
        return LT_WHY(LT_W_NOJOIN, LT_EHOLE, NULL, c->bs[h], h, 0u);
    if (j == LT_NV_NOIX || guard(c, h, j)) return LT_OK;
    if (!conv(c, h, j))
        return LT_WHY(LT_W_NOJOIN, LT_EHOLE, NULL, c->bs[h], h, 1u);
    r = addrg(c, j, h, LT_NV_KLOOP);
    if (r == LT_NV_NOIX)
        return LT_WHY(LT_W_NOJOIN, LT_EHOLE, NULL, c->bs[h], c->nrg,
                      LT_MAX_NVRG);
    c->body = c->lb[h];
    return leaves(c, r);
}

static lt_res_t cregn(lt_ncf_t *c, uint32_t h)
{
    uint32_t nb = 0u, latch = LT_NV_NOIX, r, s;
    lt_res_t rc;

    assert(c != NULL);
    assert(h < c->nbb);

    for (uint32_t x = 0; x < c->nbb; x++) {
        if (!isback(c, x, h)) continue;
        if (c->s0[x] == h) { nb++; latch = x; }
        if (c->s1[x] == h) { nb++; latch = x; }
    }
    if (nb < 2u && calm(c, latch, h)) return LT_OK;
    if (!conv(c, h, LT_NV_NOIX) || c->nst >= LT_MAX_NVST)
        return LT_WHY(LT_W_NOJOIN, LT_EHOLE, NULL, c->bs[h], h, 2u);
    for (uint32_t x = 0; x < c->nbb; x++) {
        for (uint32_t e = 0; e < 2u; e++) {
            uint32_t via = vof(c, x, e);

            if (!isback(c, x, h) || sof(c, x, e) != h) continue;
            if (!jumps(c, x, e) || (c->rd[via] != LT_NV_NOIX
                                    && c->rd[via] != c->nst))
                return LT_WHY(LT_W_NOJOIN, LT_EHOLE, NULL, c->bs[x], x, 3u);
            c->rd[via] = (uint16_t)c->nst;
        }
    }
    s = c->nst++;
    r = addrg(c, LT_NV_NOIX, h, LT_NV_KCONT);
    if (r == LT_NV_NOIX)
        return LT_WHY(LT_W_NOJOIN, LT_EHOLE, NULL, c->bs[h], c->nrg,
                      LT_MAX_NVRG);
    c->st[s].rg = (uint16_t)r;
    c->st[s].to = (uint16_t)h;
    c->rg[r].stub = (uint16_t)s;
    c->body = c->lb[h];
    rc = leaves(c, r);
    c->rg[r].hi++;
    return rc;
}

static void upto(lt_ncf_t *c, uint32_t m)
{
    assert(c != NULL);
    assert(m < c->nbb);

    bsclr(&c->tmp);
    for (uint32_t i = c->po[m]; i < c->po[m + 1u]; i++)
        if (c->pe[i] != m) bsadd(&c->tmp, c->pe[i]);
    for (uint32_t r = 0; r <= c->nbb; r++) {
        uint32_t ch = 0u;

        for (uint32_t y = 0; y < c->nbb; y++) {
            if (!bshas(&c->tmp, y)) continue;
            for (uint32_t i = c->po[y]; i < c->po[y + 1u]; i++) {
                uint32_t p = c->pe[i];

                if (p == m || bshas(&c->tmp, p)) continue;
                bsadd(&c->tmp, p);
                ch = 1u;
            }
        }
        if (ch == 0u) break;
    }
}

static lt_res_t iregn(lt_ncf_t *c, uint32_t bb)
{
    uint32_t mm = c->mof[bb], r;

    assert(c != NULL);
    assert(bb < c->nbb);

    if (mm == LT_NV_NOIX) return LT_OK;
    if (c->lp[bb] != LT_NV_NOIX && !inloop(c, mm, c->lp[bb])) return LT_OK;
    if (!conv(c, bb, mm))
        return LT_WHY(LT_W_NOJOIN, LT_EHOLE, NULL, c->bs[bb], bb, 4u);
    upto(c, mm);
    bsclr(&c->body);
    for (uint32_t k = 0; k < c->nbb; k++) {
        if (!reach(c, k) || !bshas(&c->dom[k], bb)) continue;
        if (k == mm || bshas(&c->dom[k], mm) || !bshas(&c->tmp, k)) continue;
        bsadd(&c->body, k);
    }
    r = addrg(c, mm, bb, LT_NV_KIF);
    if (r == LT_NV_NOIX)
        return LT_WHY(LT_W_NOJOIN, LT_EHOLE, NULL, c->bs[bb], c->nrg,
                      LT_MAX_NVRG);
    return leaves(c, r);
}

/* ---- Barrier slots ---- */

static lt_res_t slots(lt_ncf_t *c)
{
    uint32_t end[LT_NV_NJB];
    uint32_t used[LT_NV_NJB];

    assert(c != NULL);
    assert(c->nrg <= LT_MAX_NVRG);

    memset(used, 0, sizeof used);
    for (uint32_t r = 0; r < c->nrg; r++) {
        uint32_t i;

        for (i = r; i > 0u && c->rg[c->rord[i - 1u]].lo > c->rg[r].lo; i--)
            c->rord[i] = c->rord[i - 1u];
        c->rord[i] = (uint16_t)r;
    }
    for (uint32_t i = 0; i < c->nrg; i++) {
        lt_nrg_t *g = &c->rg[c->rord[i]];
        uint32_t  s;

        for (s = 0; s < LT_NV_NJB; s++)
            if (used[s] == 0u || end[s] <= g->lo) break;
        if (s >= LT_NV_NJB)
            return LT_WHY(LT_W_TOOJOIN, LT_EHOLE, NULL, c->bs[g->at],
                          LT_NV_NJB + 1u, LT_NV_NJB);
        g->slot = (uint8_t)s;
        used[s] = 1u;
        end[s]  = g->hi;
    }
    return LT_OK;
}

/* ---- Entry ---- */

lt_res_t
lt_nvcf(lt_ncf_t *c)
{
    lt_res_t rc;

    assert(c != NULL);
    assert(c->nbb <= LT_MAX_NVBB);

    c->nrg = 0u;
    c->nbk = 0u;
    c->nst = 0u;
    c->cur = LT_NV_NOIX;
    memset(c->rd, 0xFF, sizeof c->rd);
    memset(c->isj, 0, sizeof c->isj);
    if (c->nbb == 0u) return LT_OK;

    preds(c, 0);
    resolv(c);
    preds(c, 1);
    doms(c);
    idoms(c);
    merges(c);
    loops(c);
    rc = backs(c);
    if (rc != LT_OK) return rc;

    for (uint32_t i = 0; i < c->nbb && c->ord[i] != LT_NV_NOIX; i++) {
        rc = lregn(c, c->ord[i]);
        if (rc == LT_OK) rc = cregn(c, c->ord[i]);
        if (rc != LT_OK) return rc;
    }
    for (uint32_t k = 0; k < c->nbb; k++) {
        rc = iregn(c, k);
        if (rc != LT_OK) return rc;
    }
    return slots(c);
}
