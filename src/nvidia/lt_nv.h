/* lt_nv.h -- NVIDIA host back end. (c) 2026 Zane Hambly. */

#ifndef LT_NV_H
#define LT_NV_H

#include <stdint.h>

/* ---- Result codes ---- */

typedef enum {
    LT_OK      =  0,
    LT_EBAD    = -1,
    LT_ETRUNC  = -2,
    LT_EFULL   = -3,
    LT_EISA    = -4,
    LT_EHOLE   = -5
} lt_res_t;

/* ---- Ceilings ---- */

#define LT_MAX_NVIMG  (1024u * 1024u)
#define LT_MAX_NVNAME 96u
#define LT_NV_PBASE   0x160u
#define LT_MAX_NVPRM  32u
#define LT_MAX_NVCODE (256u * 1024u)
#define LT_MAX_NVREG  253u
#define LT_MAX_NVFIX  4096u
#define LT_MAX_NVLP   256u
#define LT_NV_NOIX    0xFFFFu

#define LT_NV_T0      0u
#define LT_NV_T1      1u
#define LT_NV_T2      2u
#define LT_NV_T3      3u
#define LT_NV_D0      4u
#define LT_NV_D1      6u
#define LT_NV_D2      8u
#define LT_NV_D3      10u
#define LT_NV_D4      12u
#define LT_NV_D5      14u
#define LT_NV_NSCR    16u

#define LT_NV_W0      16u
#define LT_NV_W1      17u
#define LT_NV_W2      18u
#define LT_NV_W3      19u
#define LT_NV_W4      20u
#define LT_NV_W6      22u
#define LT_NV_NW      8u
#define LT_NV_WSLOT   512u
#define LT_NV_MAXWV   16u
#define LT_NV_NBARH   16u

#define LT_NV_U0      0u

#define LT_NV_P0      0u
#define LT_NV_P1      1u
#define LT_NV_PE      6u
#define LT_MAX_NVXD   4u

#define LT_NV_PT      7u
#define LT_NV_URZ     63u
#define LT_NV_RZ      255u
#define LT_NV_NBAR    6u

#define LT_MAX_NVBB   1024u
#define LT_NV_NBW     16u
#define LT_NV_NJB     16u
#define LT_MAX_NVRG   2048u
#define LT_MAX_NVBK   4096u
#define LT_MAX_NVST   2048u
#define LT_NV_BEXIT   0xFFFEu

#define LT_NV_FLBL    0u
#define LT_NV_FRG     1u
#define LT_NV_FST     2u
#define LT_NV_FBLA    3u

#define LT_NV_KLOOP   0u
#define LT_NV_KIF     1u
#define LT_NV_KCONT   2u

/* ---- Generated notes ---- */

typedef struct {
    const char    *name;
    uint32_t       type;
    uint32_t       flags;
    uint32_t       info;
    uint32_t       align;
    const uint8_t *data;
    uint32_t       len;
} lt_nvn_t;

extern const lt_nvn_t    lt_nv_note[];
extern const uint32_t    lt_nv_noten;
extern const char *const lt_nv_nsrc;

/* ---- What a kernel needs around it ---- */

typedef struct {
    const char *name;
    uint32_t    nreg;
    uint32_t    smem;
    uint32_t    lmem;
    uint8_t     psz[LT_MAX_NVPRM];
    uint32_t    np;
    uint32_t    wave;
    uint32_t    mthr;
} lt_nvk_t;

/* ---- Scoreboards and branch targets ---- */

typedef struct {
    uint64_t w[4];
} lt_nrm_t;

typedef struct {
    lt_nrm_t rd;
    lt_nrm_t wr;
} lt_nrw_t;

typedef struct {
    uint32_t at;
    int32_t  lb;
    uint8_t  kind;
} lt_nfx_t;

/* ---- Where lanes part and meet ---- */

typedef struct {
    uint64_t w[LT_NV_NBW];
} lt_nbs_t;

typedef struct {
    uint16_t j;
    uint16_t at;
    uint8_t  kind;
    uint8_t  slot;
    uint16_t lo;
    uint16_t hi;
    uint16_t stub;
} lt_nrg_t;

typedef struct {
    uint16_t via;
    uint16_t rg;
    uint8_t  taken;
} lt_nbk_t;

typedef struct {
    uint16_t rg;
    uint16_t to;
} lt_nst_t;

typedef struct {
    uint32_t nbb;
    uint32_t cur;
    uint16_t bs[LT_MAX_NVBB + 1u];
    uint16_t s0[LT_MAX_NVBB];
    uint16_t s1[LT_MAX_NVBB];
    uint16_t v0[LT_MAX_NVBB];
    uint16_t v1[LT_MAX_NVBB];
    uint8_t  tk[LT_MAX_NVBB];
    uint8_t  fw[LT_MAX_NVBB];
    uint16_t po[LT_MAX_NVBB + 1u];
    uint16_t pe[2u * LT_MAX_NVBB];
    lt_nbs_t dom[LT_MAX_NVBB];
    lt_nbs_t lb[LT_MAX_NVBB];
    uint16_t idom[LT_MAX_NVBB];
    uint16_t mof[LT_MAX_NVBB];
    uint16_t lp[LT_MAX_NVBB];
    uint16_t lpar[LT_MAX_NVBB];
    uint16_t lsz[LT_MAX_NVBB];
    uint16_t ord[LT_MAX_NVBB + 1u];
    uint16_t tgt[LT_MAX_NVBB];
    uint8_t  back[LT_MAX_NVBB];
    uint16_t pc[LT_MAX_NVBB];
    uint8_t  ism[LT_MAX_NVBB];
    uint8_t  isj[LT_MAX_NVBB];
    uint16_t rd[LT_MAX_NVBB];
    uint32_t bat[LT_MAX_NVBB];
    uint32_t bla[LT_MAX_NVBB];
    uint16_t rord[LT_MAX_NVRG];
    uint32_t jat[LT_MAX_NVRG];
    uint32_t sat[LT_MAX_NVST];
    lt_nrg_t rg[LT_MAX_NVRG];
    lt_nbk_t bk[LT_MAX_NVBK];
    lt_nst_t st[LT_MAX_NVST];
    uint32_t nrg;
    uint32_t nbk;
    uint32_t nst;
    lt_nbs_t body;
    lt_nbs_t tmp;
} lt_ncf_t;

/* ---- Terminator kinds ---- */

#define TFALL 0u
#define TBR   1u
#define TBRC  2u
#define TEXIT 3u

/* ---- What a refusal is about ---- */

#define LT_W_TOOBB   1u
#define LT_W_NOJOIN  2u
#define LT_W_TOOJOIN 3u

#define LT_WHY(w, r, s, l, a, b)     lt_nvwhy((uint32_t)(w), (r), (s), (uint32_t)(l), (uint32_t)(a),              (uint32_t)(b))

lt_res_t lt_nvwhy(uint32_t w, lt_res_t r, const char *s, uint32_t l,
                  uint32_t a, uint32_t b);

/* ---- src/be/nv ---- */

lt_res_t lt_nvcf(lt_ncf_t *c);

void lt_nvput(uint64_t *w, uint32_t hb, uint32_t lb, uint64_t v);
void lt_nvmovc(uint64_t *w, uint32_t dst, uint32_t bank, uint32_t off);
void lt_nvmovi(uint64_t *w, uint32_t dst, uint32_t imm);
void lt_nvmov(uint64_t *w, uint32_t dst, uint32_t src);
void lt_nvmovu(uint64_t *w, uint32_t dst, uint32_t ur);
void lt_nvs2r(uint64_t *w, uint32_t dst, uint32_t idx);
void lt_nvadd(uint64_t *w, uint32_t d, uint32_t a, uint32_t b, uint32_t neg);
void lt_nvadi(uint64_t *w, uint32_t d, uint32_t a, uint32_t imm, uint32_t neg);
void lt_nvadc(uint64_t *w, uint32_t d, uint32_t a, uint32_t b, uint32_t neg);
void lt_nvadx(uint64_t *w, uint32_t d, uint32_t a, uint32_t b, uint32_t neg);
void lt_nvmad(uint64_t *w, uint32_t d, uint32_t a, uint32_t b, uint32_t c,
              uint32_t hi, uint32_t sgn);
void lt_nvlop(uint64_t *w, uint32_t d, uint32_t a, uint32_t b, uint32_t c,
              uint32_t lut);
void lt_nvlpi(uint64_t *w, uint32_t d, uint32_t a, uint32_t imm, uint32_t lut);
void lt_nvshf(uint64_t *w, uint32_t d, uint32_t lo, uint32_t sh, uint32_t hi,
              uint32_t ty, uint32_t rt, uint32_t dh);
void lt_nvshi(uint64_t *w, uint32_t d, uint32_t lo, uint32_t imm, uint32_t hi,
              uint32_t ty, uint32_t rt, uint32_t dh);
void lt_nvprm(uint64_t *w, uint32_t d, uint32_t a, uint32_t sel, uint32_t b);
void lt_nvpop(uint64_t *w, uint32_t d, uint32_t s);
void lt_nvbrv(uint64_t *w, uint32_t d, uint32_t s);
void lt_nvflo(uint64_t *w, uint32_t d, uint32_t s);
void lt_nvstp(uint64_t *w, uint32_t pd, uint32_t a, uint32_t b, uint32_t cmp,
              uint32_t sgn);
void lt_nvsti(uint64_t *w, uint32_t pd, uint32_t a, uint32_t imm, uint32_t cmp,
              uint32_t sgn);
void lt_nvstx(uint64_t *w, uint32_t pd, uint32_t a, uint32_t b, uint32_t cmp,
              uint32_t sgn, uint32_t lc);
void lt_nvstc(uint64_t *w, uint32_t pd, uint32_t a, uint32_t b, uint32_t cmp,
              uint32_t acc, uint32_t an);
void lt_nvsel(uint64_t *w, uint32_t d, uint32_t a, uint32_t b, uint32_t p,
              uint32_t nt);
void lt_nvsei(uint64_t *w, uint32_t d, uint32_t a, uint32_t imm, uint32_t p,
              uint32_t nt);
void lt_nvfad(uint64_t *w, uint32_t d, uint32_t a, uint32_t b, uint32_t neg);
void lt_nvfmu(uint64_t *w, uint32_t d, uint32_t a, uint32_t b);
void lt_nvffm(uint64_t *w, uint32_t d, uint32_t a, uint32_t b, uint32_t c);
void lt_nvftp(uint64_t *w, uint32_t pd, uint32_t a, uint32_t b, uint32_t cmp);
void lt_nvfti(uint64_t *w, uint32_t pd, uint32_t a, uint32_t imm, uint32_t cmp);
void lt_nvdad(uint64_t *w, uint32_t d, uint32_t a, uint32_t b, uint32_t neg);
void lt_nvdmu(uint64_t *w, uint32_t d, uint32_t a, uint32_t b);
void lt_nvdfm(uint64_t *w, uint32_t d, uint32_t a, uint32_t b, uint32_t c,
              uint32_t nga);
void lt_nvdtp(uint64_t *w, uint32_t pd, uint32_t a, uint32_t b, uint32_t cmp);
void lt_nvmuf(uint64_t *w, uint32_t d, uint32_t s, uint32_t op);
void lt_nvfrn(uint64_t *w, uint32_t d, uint32_t s, uint32_t rnd);
void lt_nvsrm(uint64_t *w, uint32_t rnd);
void lt_nvf2f(uint64_t *w, uint32_t d, uint32_t s, uint32_t dsz, uint32_t ssz,
              uint32_t rnd);
void lt_nvhfm(uint64_t *w, uint32_t d, uint32_t a, uint32_t b, uint32_t c);
void lt_nvi2f(uint64_t *w, uint32_t d, uint32_t s, uint32_t sgn, uint32_t dsz,
              uint32_t ssz);
void lt_nvf2i(uint64_t *w, uint32_t d, uint32_t s, uint32_t sgn, uint32_t dsz,
              uint32_t ssz);
void lt_nvstg(uint64_t *w, uint32_t ra, uint32_t rb, uint32_t off, uint32_t ty,
              uint32_t ord);
void lt_nvldg(uint64_t *w, uint32_t dst, uint32_t ra, uint32_t off, uint32_t ty,
              uint32_t ord);
void lt_nvldl(uint64_t *w, uint32_t dst, uint32_t ra, uint32_t off, uint32_t ty);
void lt_nvstl(uint64_t *w, uint32_t ra, uint32_t rb, uint32_t off, uint32_t ty);
void lt_nvlds(uint64_t *w, uint32_t dst, uint32_t ra, uint32_t off, uint32_t ty);
void lt_nvsts(uint64_t *w, uint32_t ra, uint32_t rb, uint32_t off, uint32_t ty);
void lt_nvvot(uint64_t *w, uint32_t d, uint32_t pd, uint32_t p, uint32_t nt,
              uint32_t vop);
void lt_nvsfl(uint64_t *w, uint32_t d, uint32_t pd, uint32_t ra, uint32_t rb,
              uint32_t ic, uint32_t smod);
void lt_nvatg(uint64_t *w, uint32_t dst, uint32_t ra, uint32_t rb,
              uint32_t off, uint32_t aop, uint32_t aty, uint32_t ord);
void lt_nvatc(uint64_t *w, uint32_t dst, uint32_t ra, uint32_t rb,
              uint32_t rc, uint32_t off, uint32_t aty, uint32_t ord);
void lt_nvmbr(uint64_t *w, uint32_t scp);
void lt_nvrdx(uint64_t *w, uint32_t ud, uint32_t ra, uint32_t rop,
              uint32_t sgn);
void lt_nvbar(uint64_t *w, uint32_t idx);
void lt_nvrb(uint64_t *w, uint32_t idx);
void lt_nvwait(uint64_t *w, uint32_t mask);
void lt_nvsyn(uint64_t *w);
void lt_nvexit(uint64_t *w);
void lt_nvbra(uint64_t *w, int32_t rel);
void lt_nvbrp(uint64_t *w, int32_t rel, uint32_t p, uint32_t nt);
void lt_nvnop(uint64_t *w);
void lt_nvbpt(uint64_t *w);
void lt_nvbsy(uint64_t *w, uint32_t bar, int32_t rel);
void lt_nvbsn(uint64_t *w, uint32_t bar);
void lt_nvbrk(uint64_t *w, uint32_t bar, uint32_t p, uint32_t nt);
void lt_nvbrs(uint64_t *w, uint32_t rb, uint32_t cnt);

lt_res_t lt_nvcub(uint8_t *out, uint32_t max, uint32_t *len,
                  const lt_nvk_t *k, const uint8_t *code, uint32_t clen);

#endif /* LT_NV_H */
