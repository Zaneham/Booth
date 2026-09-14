#include "nvidia.h"
#include "backend.h"
#include "bc_err.h"
#include "lt_nv.h"
#include <stdio.h>
#include <string.h>

#define SA_NBB   LT_MAX_NVBB
#define SA_NV    16384u
#define SA_RBAS  LT_NV_NSCR
#define SA_IMM0  0u
#define SA_IMM1  2u
#define SA_IMM2  4u
#define SA_TMP0  6u
#define SA_TMP1  8u
#define SA_TMP2  10u
#define SA_NPR   6u
#define SA_NOIX  0xFFFFu

typedef struct {
    lt_ncf_t cf;
    lt_nvk_t k;
    uint8_t  code[LT_MAX_NVCODE];
    uint32_t clen;
    lt_nrm_t brs[LT_NV_NBAR];
    uint8_t  bkd[LT_NV_NBAR];
    uint32_t pend;
    uint8_t  gate;
    lt_nfx_t fx[LT_MAX_NVFIX];
    uint32_t nfx;
    uint32_t lat[SA_NBB + 1u];
    uint16_t bmap[SA_NBB + 1u];
    uint32_t ist[SA_NBB + 1u];
    uint16_t rmap[SA_NV];
    uint16_t vlo[SA_NV];
    uint16_t vhi[SA_NV];
    uint16_t vbk[SA_NV];
    uint8_t  vlc[SA_NV];
    uint8_t  vwd[SA_NV];
    uint8_t  vpr[SA_NV];
    uint16_t ru[LT_MAX_NVREG];
    uint16_t pu[SA_NPR];
    uint32_t rbas[NV_RF_COUNT];
    uint32_t nvr;
    uint32_t nreg;
    uint32_t npr;
    uint32_t f0;
    uint32_t fn;
    uint32_t nerr;
    uint32_t nbad;
    uint16_t lpt[LT_MAX_NVLP];
    uint16_t lpi[LT_MAX_NVLP];
    uint32_t nlp;
    uint8_t  lpof;
    char     name[LT_MAX_NVNAME];
    const nv_module_t *M;
    const nv_mfunc_t  *F;
    uint32_t pofs[NV_MAX_PARAMS];
    uint8_t  img[LT_MAX_NVIMG];
    uint32_t ilen;
    uint16_t cop;
    uint16_t src[LT_MAX_NVCODE / 16u];
} sass_t;

static sass_t sast;

static uint32_t sawhy;
static uint32_t sawa;
static uint32_t sawb;

lt_res_t
lt_nvwhy(uint32_t w, lt_res_t r, const char *s, uint32_t l,
         uint32_t a, uint32_t b)
{
    (void)s;
    (void)l;
    sawhy = w;
    sawa  = a;
    sawb  = b;
    return r;
}

static const char *const saopn[NV_OP_COUNT] = {
    "NV_ADD_U32", "NV_ADD_U64", "NV_ADD_S32", "NV_SUB_U32", "NV_SUB_S32",
    "NV_SUB_S64", "NV_MUL_LO_U32", "NV_MUL_LO_S32", "NV_MUL_LO_U64", "NV_MUL_HI_U32",
    "NV_MUL_HI_S32", "NV_MUL_HI_U64", "NV_MAD_LO_U64", "NV_DIV_U32", "NV_DIV_S32",
    "NV_REM_U32", "NV_REM_S32", "NV_NEG_S32", "NV_ADD_F32", "NV_ADD_F64",
    "NV_SUB_F32", "NV_SUB_F64", "NV_MUL_F32", "NV_MUL_F64", "NV_DIV_F32",
    "NV_DIV_F64", "NV_FMA_F32", "NV_FMA_F64", "NV_NEG_F32", "NV_NEG_F64",
    "NV_ABS_F32", "NV_ABS_F64", "NV_AND_B32", "NV_AND_B64", "NV_OR_B32",
    "NV_OR_B64", "NV_XOR_B32", "NV_XOR_B64", "NV_NOT_B32", "NV_NOT_B64",
    "NV_SHL_B32", "NV_SHL_B64", "NV_SHR_U32", "NV_SHR_S32", "NV_SHR_U64",
    "NV_POPC_B32", "NV_POPC_B64", "NV_CLZ_B32", "NV_CLZ_B64", "NV_BREV_B32",
    "NV_BREV_B64", "NV_SETP_EQ_U32", "NV_SETP_NE_U32", "NV_SETP_LT_U32", "NV_SETP_LE_U32",
    "NV_SETP_GT_U32", "NV_SETP_GE_U32", "NV_SETP_LT_S32", "NV_SETP_LE_S32", "NV_SETP_GT_S32",
    "NV_SETP_GE_S32", "NV_SETP_EQ_F32", "NV_SETP_NE_F32", "NV_SETP_LT_F32", "NV_SETP_LE_F32",
    "NV_SETP_GT_F32", "NV_SETP_GE_F32", "NV_SETP_EQ_F64", "NV_SETP_NE_F64", "NV_SETP_LT_F64",
    "NV_SETP_LE_F64", "NV_SETP_GT_F64", "NV_SETP_GE_F64", "NV_SETP_EQ_U64", "NV_SETP_NE_U64",
    "NV_SELP_U32", "NV_SELP_U64", "NV_SELP_F32", "NV_SELP_F64", "NV_MOV_U32",
    "NV_MOV_U64", "NV_MOV_F32", "NV_MOV_F64", "NV_MOV_PRED", "NV_CVT_U32_F32",
    "NV_CVT_S32_F32", "NV_CVT_U32_F64", "NV_CVT_S32_F64", "NV_CVT_F32_U32", "NV_CVT_F32_S32",
    "NV_CVT_F32_F64", "NV_CVT_F64_F32", "NV_CVT_U64_U32", "NV_CVT_S64_S32", "NV_CVT_U32_U64",
    "NV_CVT_U64_F64", "NV_CVT_S64_F64", "NV_CVT_F64_U64", "NV_CVT_F64_S64", "NV_CVT_F64_U32",
    "NV_CVT_F64_S32", "NV_CVT_F32_F16", "NV_CVT_F16_F32", "NV_LD_GLB_U32", "NV_LD_GLB_U64",
    "NV_LD_GLB_F32", "NV_LD_GLB_F64", "NV_LD_GLB_U8", "NV_LD_GLB_U16", "NV_LD_GLB_B16",
    "NV_ST_GLB_U32", "NV_ST_GLB_U64", "NV_ST_GLB_F32", "NV_ST_GLB_F64", "NV_ST_GLB_U8",
    "NV_ST_GLB_U16", "NV_ST_GLB_B16", "NV_LD_SHR_U32", "NV_LD_SHR_F32", "NV_LD_SHR_U8",
    "NV_LD_SHR_U16", "NV_LD_SHR_B16", "NV_ST_SHR_U32", "NV_ST_SHR_F32", "NV_ST_SHR_U8",
    "NV_ST_SHR_U16", "NV_ST_SHR_B16", "NV_LD_LOC_U32", "NV_LD_LOC_U64", "NV_LD_LOC_F32",
    "NV_LD_LOC_F64", "NV_LD_LOC_U8", "NV_LD_LOC_U16", "NV_LD_LOC_B16", "NV_ST_LOC_U32",
    "NV_ST_LOC_U64", "NV_ST_LOC_F32", "NV_ST_LOC_F64", "NV_ST_LOC_U8", "NV_ST_LOC_U16",
    "NV_ST_LOC_B16", "NV_LD_PARAM_U32", "NV_LD_PARAM_U64", "NV_LD_PARAM_F32", "NV_LD_PARAM_F64",
    "NV_ATOM_ADD_U32", "NV_ATOM_ADD_F32", "NV_ATOM_MIN_U32", "NV_ATOM_MAX_U32", "NV_ATOM_AND_B32",
    "NV_ATOM_OR_B32", "NV_ATOM_XOR_B32", "NV_ATOM_XCHG_B32", "NV_ATOM_CAS_B32", "NV_ATOM_ADD_U64",
    "NV_ATOM_ADD_F64", "NV_ATOM_AND_B64", "NV_ATOM_OR_B64", "NV_ATOM_XOR_B64", "NV_ATOM_XCHG_B64",
    "NV_ATOM_CAS_B64", "NV_BRA", "NV_BRA_PRED", "NV_BAR_SYNC", "NV_MEMBAR",
    "NV_NANOSLP", "NV_BARRED_OR", "NV_BARRED_AND", "NV_BARRED_POPC", "NV_SHFL_IDX",
    "NV_SHFL_UP", "NV_SHFL_DOWN", "NV_SHFL_XOR", "NV_VOTE_BALLOT", "NV_VOTE_ANY",
    "NV_VOTE_ALL", "NV_SQRT_F32", "NV_SQRT_F64", "NV_RSQ_F32", "NV_RCP_F32",
    "NV_SIN_F32", "NV_COS_F32", "NV_EX2_F32", "NV_LG2_F32", "NV_FLOOR_F32",
    "NV_CEIL_F32", "NV_TRUNC_F32", "NV_ROUND_F32", "NV_MIN_F32", "NV_MAX_F32",
    "NV_MIN_U32", "NV_MAX_U32", "NV_MIN_S32", "NV_MAX_S32", "NV_RET",
    "NV_EXIT", "NV_MOV_F64_LIT", "NV_LEA_LOCAL", "NV_LEA_GLB", "NV_LEA_DSH",
    "NV_MOV_PK16", "NV_MMA", "NV_WLD", "NV_WST", "NV_WMMA",
    "NV_ASM", "NV_BARWARP", "NV_TRAP", "NV_CVTA_GLB", "NV_CVTA_LOC",
    "NV_CALL", "NV_ST_RETP", "NV_CVT_U32_U16", "NV_SETP_LT_S64", "NV_SETP_LE_S64",
    "NV_SETP_GT_S64", "NV_SETP_GE_S64", "NV_SETP_LT_U64", "NV_SETP_LE_U64", "NV_SETP_GT_U64",
    "NV_SETP_GE_U64", "NV_DIV_U64", "NV_DIV_S64", "NV_REM_U64", "NV_REM_S64",
    "NV_NEG_S64", "NV_SHR_S64", "NV_MOV_B16", "NV_MOV_B32", "NV_MOV_B64",
    "NV_CVT_S32_S16", "NV_CVT_U16_U32", "NV_CVT_F32_U64", "NV_CVT_F32_S64", "NV_CVT_U64_F32",
    "NV_CVT_S64_F32", "NV_CVT_F16_F64", "NV_CVT_F64_F16", "NV_CVT_F32_BF16", "NV_CVT_BF16_F32",
    "NV_LD_PARAM_B16", "NV_LD_SHR_U64", "NV_LD_SHR_F64", "NV_ST_SHR_U64", "NV_ST_SHR_F64",
    "NV_MIN_U64", "NV_MAX_U64", "NV_MIN_S64", "NV_MAX_S64", "NV_GBAR",
};

static const char *saonm(uint16_t op)
{
    if (op >= (uint16_t)NV_OP_COUNT || saopn[op] == NULL)
        return "an unknown machine op";
    return saopn[op];
}

static void samsr(lt_nrm_t *s, uint32_t r, uint32_t n)
{
    for (uint32_t k = 0; k < n; k++) {
        uint32_t x = r + k;

        if (x >= LT_MAX_NVREG) continue;
        s->w[x >> 6] |= 1ull << (x & 63u);
    }
}

static int sahit(const lt_nrm_t *a, const lt_nrm_t *b)
{
    for (uint32_t k = 0; k < 4u; k++)
        if ((a->w[k] & b->w[k]) != 0ull) return 1;
    return 0;
}

static void saput(const uint64_t *w)
{
    if (sast.clen + 16u > LT_MAX_NVCODE) {
        sast.nbad++;
        return;
    }
    sast.src[sast.clen >> 4] = sast.cop;
    memcpy(sast.code + sast.clen, w, 16u);
    sast.clen += 16u;
}

static uint32_t salive(void)
{
    uint32_t msk = 0u;

    for (uint32_t k = 0; k < LT_NV_NBAR; k++)
        if (sast.bkd[k] != 0u) msk |= 1u << k;
    return msk;
}

static void sadrn(void)
{
    sast.pend |= salive();
    memset(sast.bkd, 0, sizeof sast.bkd);
}

static uint32_t sahaz(const lt_nrw_t *x)
{
    uint32_t msk = 0u;

    for (uint32_t k = 0; k < LT_NV_NBAR; k++) {
        if (sast.bkd[k] == 1u
            && (sahit(&sast.brs[k], &x->rd) || sahit(&sast.brs[k], &x->wr)))
            msk |= 1u << k;
        else if (sast.bkd[k] == 2u && sahit(&sast.brs[k], &x->wr))
            msk |= 1u << k;
    }
    return msk;
}

static uint32_t safre(uint32_t msk)
{
    uint32_t n = 0u;

    for (uint32_t k = 0; k < LT_NV_NBAR; k++)
        if (sast.bkd[k] == 0u || (msk & (1u << k)) != 0u) n++;
    return n;
}

static uint32_t satake(const lt_nrm_t *s, uint32_t kind)
{
    uint32_t k = 0u;

    while (k < LT_NV_NBAR && sast.bkd[k] != 0u) k++;
    if (k >= LT_NV_NBAR) k = 0u;
    sast.bkd[k] = (uint8_t)kind;
    sast.brs[k] = *s;
    return k;
}

static void safire(uint64_t *w, const lt_nrw_t *x, uint32_t dec)
{
    uint32_t msk = sast.pend | sahaz(x);
    uint32_t need = ((dec & 1u) != 0u ? 1u : 0u) + ((dec & 2u) != 0u ? 1u : 0u);

    sast.pend = 0u;
    if (safre(msk) < need) msk |= salive();
    for (uint32_t k = 0; k < LT_NV_NBAR; k++)
        if ((msk & (1u << k)) != 0u) sast.bkd[k] = 0u;

    if ((dec & 2u) != 0u) lt_nvrb(w, satake(&x->rd, 2u));
    if ((dec & 1u) != 0u) lt_nvbar(w, satake(&x->wr, 1u));
    if (msk != 0u) lt_nvwait(w, msk);
    lt_nvput(w, 14u, 12u, sast.gate);
    saput(w);
}

static void safix(uint32_t kind, int32_t ref)
{
    if (sast.nfx >= LT_MAX_NVFIX) {
        sast.nbad++;
        return;
    }
    sast.fx[sast.nfx].at   = sast.clen;
    sast.fx[sast.nfx].lb   = ref;
    sast.fx[sast.nfx].kind = (uint8_t)kind;
    sast.nfx++;
}

static uint32_t satgt(const nv_opnd_t *o)
{
    uint32_t g;

    if (o->kind != (uint8_t)NV_MOP_LABEL || o->imm < 0) return SA_NOIX;
    g = (uint32_t)o->imm;
    if (g < sast.F->first_blk) return SA_NOIX;
    g -= sast.F->first_blk;
    if (g >= sast.F->num_blks) return SA_NOIX;
    return sast.bmap[g];
}

static int sapair(const nv_minst_t *a, const nv_minst_t *b)
{
    return a->op == (uint16_t)NV_BRA_PRED && b->op == (uint16_t)NV_BRA;
}

static uint32_t saterm(const nv_minst_t *I)
{
    if (I->op == (uint16_t)NV_BRA) return TBR;
    if (I->op == (uint16_t)NV_BRA_PRED) return TBRC;
    if (I->op == (uint16_t)NV_RET || I->op == (uint16_t)NV_EXIT) return TEXIT;
    return TFALL;
}

static int sasplit(void)
{
    const nv_module_t *M = sast.M;
    uint32_t n = 0u;

    for (uint32_t bi = 0; bi < sast.F->num_blks; bi++) {
        const nv_mblk_t *B = &M->mblks[sast.F->first_blk + bi];
        uint32_t s = B->first_inst - sast.f0;
        uint32_t e = s + B->num_insts;

        if (n + 2u >= SA_NBB) return 1;
        sast.bmap[bi] = (uint16_t)n;
        sast.ist[n] = s;
        n++;
        if (B->num_insts >= 2u
            && sapair(&M->minsts[B->first_inst + B->num_insts - 2u],
                      &M->minsts[B->first_inst + B->num_insts - 1u])) {
            sast.ist[n] = e - 1u;
            n++;
        }
    }
    sast.ist[n] = sast.fn;
    sast.cf.nbb = n;
    for (uint32_t k = 0; k <= n; k++) sast.cf.bs[k] = (uint16_t)sast.ist[k];
    return 0;
}

static int sasucc(void)
{
    const nv_module_t *M = sast.M;
    lt_ncf_t *c = &sast.cf;

    for (uint32_t k = 0; k < c->nbb; k++) {
        uint32_t nx = (k + 1u < c->nbb) ? k + 1u : LT_NV_BEXIT;
        uint32_t z  = sast.ist[k + 1u];
        const nv_minst_t *T;
        uint32_t kind;

        c->s0[k] = (uint16_t)LT_NV_NOIX;
        c->s1[k] = (uint16_t)LT_NV_NOIX;
        c->v0[k] = (uint16_t)k;
        c->v1[k] = (uint16_t)k;
        c->tk[k] = (uint8_t)TFALL;
        if (z == sast.ist[k]) { c->s0[k] = (uint16_t)nx; continue; }
        T = &M->minsts[sast.f0 + z - 1u];
        kind = saterm(T);
        c->tk[k] = (uint8_t)kind;
        if (kind == TBR) {
            if (satgt(&T->ops[0]) == SA_NOIX) return 1;
            c->s0[k] = (uint16_t)satgt(&T->ops[0]);
        } else if (kind == TBRC) {
            if (satgt(&T->ops[1]) == SA_NOIX) return 1;
            c->s0[k] = (uint16_t)satgt(&T->ops[1]);
            c->s1[k] = (uint16_t)nx;
        } else if (kind == TFALL) {
            c->s0[k] = (uint16_t)nx;
        }
    }
    return 0;
}

static uint32_t savw(uint8_t rf)
{
    return (rf == (uint8_t)NV_RF_U64 || rf == (uint8_t)NV_RF_F64) ? 2u : 1u;
}

static uint32_t savx(const nv_opnd_t *o)
{
    uint32_t v;

    if (o->rfile >= (uint8_t)NV_RF_COUNT) return SA_NOIX;
    v = sast.rbas[o->rfile] + o->reg_num;
    return (v < SA_NV) ? v : SA_NOIX;
}

static int sabase(void)
{
    uint32_t t = 0u;

    for (uint32_t f = 0; f < (uint32_t)NV_RF_COUNT; f++) {
        sast.rbas[f] = t;
        t += (uint32_t)sast.M->rc[f] + 1u;
        if (t > SA_NV) return 1;
    }
    sast.nvr = t;
    return 0;
}

static void saedge(uint32_t i, const nv_minst_t *I)
{
    uint32_t n = (uint32_t)I->num_defs + (uint32_t)I->num_uses;
    uint32_t t;

    for (uint32_t k = 0; k < n && k < NV_MAX_OPS; k++) {
        if (I->ops[k].kind != (uint8_t)NV_MOP_LABEL) continue;
        t = satgt(&I->ops[k]);
        if (t == SA_NOIX || sast.ist[t] > i) continue;
        if (sast.nlp >= LT_MAX_NVLP) { sast.lpof = 1u; return; }
        sast.lpt[sast.nlp] = (uint16_t)sast.ist[t];
        sast.lpi[sast.nlp] = (uint16_t)i;
        sast.nlp++;
    }
}

static int sardt(const nv_minst_t *I, const nv_opnd_t *d, uint32_t n)
{
    for (uint32_t k = 1; k < n && k < NV_MAX_OPS; k++)
        if (I->ops[k].kind == (uint8_t)NV_MOP_REG
            && I->ops[k].rfile == d->rfile
            && I->ops[k].reg_num == d->reg_num) return 1;
    return 0;
}

static int sascan(void)
{
    uint32_t cb = 0u;

    for (uint32_t i = 0; i < sast.fn; i++) {
        const nv_minst_t *I = &sast.M->minsts[sast.f0 + i];
        uint32_t n = (uint32_t)I->num_defs + (uint32_t)I->num_uses;

        while (cb + 1u < sast.cf.nbb && sast.ist[cb + 1u] <= i) cb++;
        saedge(i, I);
        for (uint32_t k = 0; k < n && k < NV_MAX_OPS; k++) {
            const nv_opnd_t *o = &I->ops[k];
            uint32_t v;

            if (o->kind != (uint8_t)NV_MOP_REG) continue;
            v = savx(o);
            if (v == SA_NOIX) return 1;
            sast.vwd[v] = (uint8_t)savw(o->rfile);
            sast.vpr[v] = (o->rfile == (uint8_t)NV_RF_PRED) ? 1u : 0u;
            if (sast.vlo[v] != (uint16_t)SA_NOIX) {
                if (sast.vbk[v] != (uint16_t)cb) sast.vlc[v] = 0u;
            } else {
                sast.vlo[v] = (uint16_t)i;
                sast.vbk[v] = (uint16_t)cb;
                sast.vlc[v] = (k == 0u && I->num_defs == 1u
                               && sardt(I, o, n) == 0) ? 1u : 0u;
            }
            sast.vhi[v] = (uint16_t)i;
        }
    }
    return 0;
}

static void samrg(void)
{
    for (uint32_t r = 0; r <= sast.nlp; r++) {
        uint32_t ch = 0u;

        for (uint32_t a = 0; a < sast.nlp; a++)
            for (uint32_t c = 0; c < sast.nlp; c++) {
                uint16_t t = sast.lpt[c], e = sast.lpi[c];

                if (a == c || sast.lpt[a] > e || sast.lpi[a] < t) continue;
                if (t < sast.lpt[a]) { sast.lpt[a] = t; ch = 1u; }
                if (e > sast.lpi[a]) { sast.lpi[a] = e; ch = 1u; }
            }
        if (ch == 0u) break;
    }
}

static void sastr(void)
{
    if (sast.lpof != 0u) {
        sast.lpt[0] = 0u;
        sast.lpi[0] = (uint16_t)(sast.fn > 0u ? sast.fn - 1u : 0u);
        sast.nlp    = 1u;
    } else {
        samrg();
    }
    for (uint32_t v = 0; v < sast.nvr; v++) {
        if (sast.vlo[v] == (uint16_t)SA_NOIX || sast.vlc[v] != 0u) continue;
        for (uint32_t a = 0; a < sast.nlp; a++) {
            if (sast.vhi[v] < sast.lpt[a] || sast.vlo[v] > sast.lpi[a])
                continue;
            if (sast.lpt[a] < sast.vlo[v]) sast.vlo[v] = sast.lpt[a];
            if (sast.lpi[a] > sast.vhi[v]) sast.vhi[v] = sast.lpi[a];
        }
    }
}

static uint32_t sapick(uint32_t w, uint32_t lo)
{
    for (uint32_t r = SA_RBAS; r + w <= LT_MAX_NVREG; r += w) {
        if (sast.ru[r] != (uint16_t)SA_NOIX && sast.ru[r] >= lo) continue;
        if (w == 2u && sast.ru[r + 1u] != (uint16_t)SA_NOIX
            && sast.ru[r + 1u] >= lo) continue;
        return r;
    }
    return LT_MAX_NVREG;
}

static uint32_t sappik(uint32_t lo)
{
    for (uint32_t r = 1u; r < SA_NPR; r++) {
        if (sast.pu[r] != (uint16_t)SA_NOIX && sast.pu[r] >= lo) continue;
        return r;
    }
    return SA_NPR;
}

static int saall(void)
{
    uint32_t top = SA_RBAS;

    for (uint32_t i = 0; i < sast.fn; i++) {
        const nv_minst_t *I = &sast.M->minsts[sast.f0 + i];
        uint32_t n = (uint32_t)I->num_defs + (uint32_t)I->num_uses;

        for (uint32_t k = 0; k < n && k < NV_MAX_OPS; k++) {
            uint32_t v, w, r;

            if (I->ops[k].kind != (uint8_t)NV_MOP_REG) continue;
            v = savx(&I->ops[k]);
            if (v == SA_NOIX) return 1;
            if (sast.rmap[v] != (uint16_t)SA_NOIX) continue;
            if (sast.vpr[v] != 0u) {
                r = sappik(sast.vlo[v]);
                if (r >= SA_NPR) { sast.npr = 1u; return 1; }
                sast.rmap[v] = (uint16_t)r;
                sast.pu[r] = sast.vhi[v];
                continue;
            }
            w = sast.vwd[v];
            r = sapick(w, sast.vlo[v]);
            if (r >= LT_MAX_NVREG) return 1;
            sast.rmap[v] = (uint16_t)r;
            sast.ru[r] = sast.vhi[v];
            if (w == 2u) sast.ru[r + 1u] = sast.vhi[v];
            if (r + w > top) top = r + w;
        }
    }
    sast.nreg = top;
    return 0;
}

static void sacfp(uint64_t *w)
{
    lt_nrw_t x;
    uint8_t  g = sast.gate;
    uint16_t c = sast.cop;

    memset(&x, 0, sizeof x);
    sast.gate = (uint8_t)LT_NV_PT;
    sast.cop  = 0xFFFEu;
    safire(w, &x, 0u);
    sast.cop  = c;
    sast.gate = g;
}

static void sabsy(uint32_t r)
{
    uint64_t w[2];

    safix(LT_NV_FRG, (int32_t)r);
    lt_nvbsy(w, sast.cf.rg[r].slot, 0);
    sacfp(w);
}

static void sabsn(uint32_t r)
{
    uint64_t w[2];

    sast.cf.jat[r] = sast.clen;
    lt_nvbsn(w, sast.cf.rg[r].slot);
    sacfp(w);
}

static void sabrk(uint32_t via, uint32_t taken, uint32_t p)
{
    uint64_t w[2];

    for (uint32_t i = 0; i < sast.cf.nbk; i++) {
        if (sast.cf.bk[i].via != via) continue;
        if (sast.cf.bk[i].taken != (uint8_t)taken) continue;
        lt_nvbrk(w, sast.cf.rg[sast.cf.bk[i].rg].slot, p, 0u);
        sacfp(w);
    }
}

static void sahead(uint32_t k)
{
    lt_ncf_t *c = &sast.cf;

    c->bat[k] = sast.clen;
    for (uint32_t r = 0; r < c->nrg; r++)
        if (c->rg[r].j == k) sabsn(r);
    c->bla[k] = sast.clen;
    for (uint32_t kind = 0; kind <= LT_NV_KCONT; kind++) {
        for (uint32_t r = 0; r < c->nrg; r++) {
            if (c->rg[r].at != k || c->rg[r].kind != kind) continue;
            sabsy(r);
        }
        if (kind == LT_NV_KLOOP) c->bla[k] = sast.clen;
    }
}

static void sastub(void)
{
    lt_ncf_t *c = &sast.cf;
    uint64_t  w[2];

    for (uint32_t s = 0; s < c->nst; s++) {
        c->sat[s] = sast.clen;
        sabsn(c->st[s].rg);
        safix(LT_NV_FBLA, (int32_t)c->st[s].to);
        lt_nvbrp(w, 0, LT_NV_PT, 0u);
        sacfp(w);
    }
}

static void sajmp(int32_t lb)
{
    lt_ncf_t *c = &sast.cf;

    if (c->rd[c->cur] != (uint16_t)LT_NV_NOIX)
        safix(LT_NV_FST, (int32_t)c->rd[c->cur]);
    else if (c->back[c->cur] != 0u)
        safix(LT_NV_FBLA, (int32_t)c->tgt[c->cur]);
    else
        safix(LT_NV_FLBL, lb);
}

static int64_t sadest(const lt_nfx_t *f)
{
    uint32_t r = (f->lb < 0) ? SA_NBB : (uint32_t)f->lb;

    switch (f->kind) {
    case LT_NV_FLBL: return r < sast.cf.nbb ? (int64_t)sast.lat[r] : -1;
    case LT_NV_FRG:  return r < sast.cf.nrg ? (int64_t)sast.cf.jat[r] + 16 : -1;
    case LT_NV_FST:  return r < sast.cf.nst ? (int64_t)sast.cf.sat[r] : -1;
    case LT_NV_FBLA: return r < sast.cf.nbb ? (int64_t)sast.cf.bla[r] : -1;
    default:         return -1;
    }
}

static void sapat(void)
{
    uint64_t w[2];

    for (uint32_t k = 0; k < sast.nfx; k++) {
        uint32_t at = sast.fx[k].at;
        int64_t  to = sadest(&sast.fx[k]);
        int64_t  rl;

        if (at + 16u > sast.clen || to < 0) continue;
        rl = (to - (int64_t)at - 16) / 4;
        memcpy(w, sast.code + at, 16u);
        if (sast.fx[k].kind == (uint8_t)LT_NV_FRG)
            lt_nvput(w, 63u, 34u, (uint64_t)rl & (uint64_t)0x3FFFFFFF);
        else
            lt_nvput(w, 81u, 34u, (uint64_t)rl & (uint64_t)0xFFFFFFFFFFFF);
        memcpy(sast.code + at, w, 16u);
    }
}

static int sawchk(const nv_minst_t *I, uint32_t want)
{
    uint32_t n = (uint32_t)I->num_defs + (uint32_t)I->num_uses;

    for (uint32_t k = 0; k < n && k < NV_MAX_OPS; k++) {
        if (I->ops[k].kind != (uint8_t)NV_MOP_REG) continue;
        if (I->ops[k].rfile == (uint8_t)NV_RF_PRED) continue;
        if (savw(I->ops[k].rfile) != want) return 1;
    }
    return 0;
}

static uint32_t sadw(const nv_minst_t *I)
{
    if (I->num_defs == 0u || I->ops[0].kind != (uint8_t)NV_MOP_REG) return 1u;
    return savw(I->ops[0].rfile);
}

static void sabad(uint16_t op, int eid)
{
    (void)be_fail(eid, "nvptx", saonm(op));
    sast.nbad++;
}

static uint32_t sarg(const nv_opnd_t *o)
{
    uint32_t v = savx(o);

    if (v == SA_NOIX || sast.rmap[v] == (uint16_t)SA_NOIX) return LT_NV_RZ;
    return sast.rmap[v];
}

static void samat(uint32_t r, int32_t v, uint32_t w)
{
    lt_nrw_t y;
    uint64_t q[2];

    memset(&y, 0, sizeof y);
    samsr(&y.wr, r, w);
    lt_nvmovi(q, r, (uint32_t)v);
    safire(q, &y, 0u);
    if (w != 2u) return;
    lt_nvmovi(q, r + 1u, (v < 0) ? 0xFFFFFFFFu : 0u);
    safire(q, &y, 0u);
}

static uint32_t sasrc(uint16_t op, const nv_opnd_t *o, uint32_t scr,
                      uint32_t w, lt_nrw_t *x)
{
    if (o->kind == (uint8_t)NV_MOP_REG) {
        uint32_t r = sarg(o);

        samsr(&x->rd, r, savw(o->rfile));
        return r;
    }
    if (o->kind == (uint8_t)NV_MOP_IMM) {
        samat(scr, o->imm, w);
        samsr(&x->rd, scr, w);
        return scr;
    }
    sabad(op, BC_E609);
    return LT_NV_RZ;
}

static uint32_t sadef(const nv_minst_t *I, lt_nrw_t *x)
{
    uint32_t r;

    if (I->num_defs == 0u || I->ops[0].kind != (uint8_t)NV_MOP_REG)
        return LT_NV_RZ;
    r = sarg(&I->ops[0]);
    samsr(&x->wr, r, savw(I->ops[0].rfile));
    return r;
}

static uint32_t sapd(const nv_minst_t *I)
{
    uint32_t v;

    if (I->num_defs == 0u || I->ops[0].kind != (uint8_t)NV_MOP_REG)
        return LT_NV_PT;
    v = savx(&I->ops[0]);
    if (v == SA_NOIX || sast.rmap[v] == (uint16_t)SA_NOIX) return LT_NV_PT;
    return sast.rmap[v];
}

static uint32_t saprd(const nv_opnd_t *o)
{
    uint32_t v;

    if (o->kind != (uint8_t)NV_MOP_REG) return LT_NV_PT;
    v = savx(o);
    if (v == SA_NOIX || sast.rmap[v] == (uint16_t)SA_NOIX) return LT_NV_PT;
    return sast.rmap[v];
}

/* per nak-mesa-90320e129e94e2dd68868db068c7625f4054e547 tables/sass_sreg.c */
static const uint8_t sasr[13] = {
    0x21u, 0x22u, 0x23u, 0x25u, 0x26u, 0x27u,
    0x00u, 0x00u, 0x00u, 0x00u, 0x00u, 0x00u, 0x00u
};

/* per probed-driver-sm89 tables/sass_cbank.c */
static const uint16_t sacb[13] = {
    0u, 0u, 0u, 0u, 0u, 0u,
    0x00u, 0x04u, 0x08u, 0x0Cu, 0x10u, 0x14u, 0u
};

static int saspec(const nv_minst_t *I, int32_t id, uint32_t d)
{
    lt_nrw_t x;
    uint64_t w[2];

    memset(&x, 0, sizeof x);
    samsr(&x.wr, d, 1u);
    if (id < 0 || id > NV_SPEC_LANEID) {
        (void)be_fail(BC_E610, "nvptx", (unsigned)id);
        sast.nbad++;
        return 1;
    }
    if (id >= NV_SPEC_NTID_X && id <= NV_SPEC_NCTAID_Z) {
        lt_nvmovc(w, d, 0u, sacb[id]);
        safire(w, &x, 0u);
        return 0;
    }
    (void)I;
    lt_nvs2r(w, d, sasr[id]);
    safire(w, &x, 1u);
    return 0;
}

static void samov(const nv_minst_t *I)
{
    lt_nrw_t x;
    uint64_t w[2];
    uint32_t d, n;

    memset(&x, 0, sizeof x);
    d = sadef(I, &x);
    n = sadw(I);
    if (I->ops[1].kind == (uint8_t)NV_MOP_SPEC) {
        if (n != 1u) { sabad(I->op, BC_E611); return; }
        (void)saspec(I, I->ops[1].imm, d);
        return;
    }
    if (I->ops[1].kind == (uint8_t)NV_MOP_REG
        && savw(I->ops[1].rfile) != n) {
        sabad(I->op, BC_E611);
        return;
    }
    if (I->ops[1].kind == (uint8_t)NV_MOP_IMM) {
        lt_nvmovi(w, d, (uint32_t)I->ops[1].imm);
        safire(w, &x, 0u);
        if (n != 2u) return;
        lt_nvmovi(w, d + 1u, (I->ops[1].imm < 0) ? 0xFFFFFFFFu : 0u);
        safire(w, &x, 0u);
        return;
    }
    if (I->ops[1].kind != (uint8_t)NV_MOP_REG) {
        sabad(I->op, BC_E609);
        return;
    }
    samsr(&x.rd, sarg(&I->ops[1]), savw(I->ops[1].rfile));
    lt_nvmov(w, d, sarg(&I->ops[1]));
    safire(w, &x, 0u);
    if (n != 2u) return;
    lt_nvmov(w, d + 1u, sarg(&I->ops[1]) + 1u);
    safire(w, &x, 0u);
}

static void sadlit(const nv_minst_t *I)
{
    lt_nrw_t x;
    uint64_t w[2];
    uint32_t d;

    memset(&x, 0, sizeof x);
    if (sadw(I) != 2u) { sabad(I->op, BC_E611); return; }
    d = sadef(I, &x);
    lt_nvmovi(w, d, (uint32_t)I->ops[2].imm);
    safire(w, &x, 0u);
    lt_nvmovi(w, d + 1u, (uint32_t)I->ops[1].imm);
    safire(w, &x, 0u);
}

static uint32_t salut(uint16_t op)
{
    if (op == (uint16_t)NV_AND_B32 || op == (uint16_t)NV_AND_B64) return 0xC0u;
    if (op == (uint16_t)NV_OR_B32 || op == (uint16_t)NV_OR_B64) return 0xFCu;
    if (op == (uint16_t)NV_XOR_B32 || op == (uint16_t)NV_XOR_B64) return 0x3Cu;
    return 0x0Fu;
}

static void sabit(const nv_minst_t *I, int un, int wide)
{
    lt_nrw_t x;
    uint64_t w[2];
    uint32_t d, a, b, l = salut(I->op);

    memset(&x, 0, sizeof x);
    if (sawchk(I, wide ? 2u : 1u) != 0) { sabad(I->op, BC_E611); return; }
    d = sadef(I, &x);
    a = sasrc(I->op, &I->ops[1], SA_IMM0, wide ? 2u : 1u, &x);
    b = un ? LT_NV_RZ
           : sasrc(I->op, &I->ops[2], SA_IMM1, wide ? 2u : 1u, &x);
    lt_nvlop(w, d, a, b, LT_NV_RZ, l);
    safire(w, &x, 0u);
    if (!wide) return;
    lt_nvlop(w, d + 1u, a + 1u, un ? LT_NV_RZ : b + 1u, LT_NV_RZ, l);
    safire(w, &x, 0u);
}

static void saal32(const nv_minst_t *I)
{
    lt_nrw_t x;
    uint64_t w[2];
    uint32_t d, a, b;

    memset(&x, 0, sizeof x);
    if (sawchk(I, 1u) != 0) { sabad(I->op, BC_E611); return; }
    d = sadef(I, &x);
    a = sasrc(I->op, &I->ops[1], SA_IMM0, 1u, &x);
    b = (I->num_uses > 1u) ? sasrc(I->op, &I->ops[2], SA_IMM1, 1u, &x)
                           : LT_NV_RZ;
    switch (I->op) {
    case NV_ADD_U32: case NV_ADD_S32:
        lt_nvadd(w, d, a, b, 0u); break;
    case NV_SUB_U32: case NV_SUB_S32:
        lt_nvadd(w, d, a, b, 1u); break;
    case NV_NEG_S32:
        lt_nvadd(w, d, LT_NV_RZ, a, 1u); break;
    case NV_MUL_LO_U32:
        lt_nvmad(w, d, a, b, LT_NV_RZ, 0u, 0u); break;
    case NV_MUL_LO_S32:
        lt_nvmad(w, d, a, b, LT_NV_RZ, 0u, 1u); break;
    case NV_MUL_HI_U32:
        lt_nvmad(w, d, a, b, LT_NV_RZ, 1u, 0u); break;
    case NV_MUL_HI_S32:
        lt_nvmad(w, d, a, b, LT_NV_RZ, 1u, 1u); break;
    case NV_SHL_B32:
        lt_nvshf(w, d, a, b, LT_NV_RZ, 3u, 0u, 0u); break;
    case NV_SHR_U32:
        lt_nvshf(w, d, LT_NV_RZ, b, a, 3u, 1u, 1u); break;
    case NV_SHR_S32:
        lt_nvshf(w, d, LT_NV_RZ, b, a, 2u, 1u, 1u); break;
    case NV_ADD_F32:
        lt_nvfad(w, d, a, b, 0u); break;
    case NV_SUB_F32:
        lt_nvfad(w, d, a, b, 1u); break;
    case NV_MUL_F32:
        lt_nvfmu(w, d, a, b); break;
    case NV_NEG_F32:
        lt_nvlpi(w, d, a, 0x80000000u, 0x3Cu); break;
    case NV_ABS_F32:
        lt_nvlpi(w, d, a, 0x7FFFFFFFu, 0xC0u); break;
    case NV_POPC_B32:
        lt_nvpop(w, d, a); safire(w, &x, 1u); return;
    case NV_BREV_B32:
        lt_nvbrv(w, d, a); safire(w, &x, 1u); return;
    default:
        sabad(I->op, BC_E601); return;
    }
    safire(w, &x, 0u);
}

static void sash64(const nv_minst_t *I)
{
    lt_nrw_t x;
    uint64_t w[2];
    uint32_t d, a, b, rt, sg, ty;

    memset(&x, 0, sizeof x);
    if (I->ops[1].kind != (uint8_t)NV_MOP_REG
        || savw(I->ops[1].rfile) != 2u) {
        sabad(I->op, BC_E611);
        return;
    }
    d  = sadef(I, &x);
    a  = sasrc(I->op, &I->ops[1], SA_IMM0, 2u, &x);
    b  = sasrc(I->op, &I->ops[2], SA_IMM1, 1u, &x);
    rt = (I->op == (uint16_t)NV_SHL_B64) ? 0u : 1u;
    sg = (I->op == (uint16_t)NV_SHR_S32
       || I->op == (uint16_t)NV_SHR_S64) ? 1u : 0u;
    ty = 1u - sg;
    if (rt != 0u) lt_nvshf(w, d, a, b, a + 1u, ty, 1u, 0u);
    else          lt_nvshf(w, d, a, b, LT_NV_RZ, 1u, 0u, 0u);
    safire(w, &x, 0u);
    if (rt != 0u) lt_nvshf(w, d + 1u, LT_NV_RZ, b, a + 1u, ty, 1u, 1u);
    else          lt_nvshf(w, d + 1u, a, b, a + 1u, 1u, 0u, 1u);
    safire(w, &x, 0u);
}

static void saclz(const nv_minst_t *I)
{
    lt_nrw_t x;
    uint64_t w[2];
    uint32_t d, a;

    memset(&x, 0, sizeof x);
    if (sawchk(I, 1u) != 0) { sabad(I->op, BC_E611); return; }
    d = sadef(I, &x);
    a = sasrc(I->op, &I->ops[1], SA_IMM0, 1u, &x);
    lt_nvflo(w, d, a);
    safire(w, &x, 1u);
    lt_nvadi(w, d, d, 31u, 1u);
    safire(w, &x, 0u);
}

static void safma(const nv_minst_t *I)
{
    lt_nrw_t x;
    uint64_t w[2];
    uint32_t d, a, b, c;

    memset(&x, 0, sizeof x);
    if (sawchk(I, 1u) != 0) { sabad(I->op, BC_E611); return; }
    d = sadef(I, &x);
    a = sasrc(I->op, &I->ops[1], SA_IMM0, 1u, &x);
    b = sasrc(I->op, &I->ops[2], SA_IMM1, 1u, &x);
    c = sasrc(I->op, &I->ops[3], SA_IMM2, 1u, &x);
    lt_nvffm(w, d, a, b, c);
    safire(w, &x, 0u);
}

static void sasel(const nv_minst_t *I, int wide)
{
    lt_nrw_t x;
    uint64_t w[2];
    uint32_t d, a, b, p;

    memset(&x, 0, sizeof x);
    if (sawchk(I, wide ? 2u : 1u) != 0) { sabad(I->op, BC_E611); return; }
    d = sadef(I, &x);
    a = sasrc(I->op, &I->ops[1], SA_IMM0, wide ? 2u : 1u, &x);
    b = sasrc(I->op, &I->ops[2], SA_IMM1, wide ? 2u : 1u, &x);
    p = saprd(&I->ops[3]);
    lt_nvsel(w, d, a, b, p, 0u);
    safire(w, &x, 0u);
    if (!wide) return;
    lt_nvsel(w, d + 1u, a + 1u, b + 1u, p, 0u);
    safire(w, &x, 0u);
}

static void saad64(uint32_t d, uint32_t a, uint32_t b, const lt_nrw_t *x)
{
    uint64_t w[2];

    lt_nvadc(w, d, a, b, 0u);
    safire(w, x, 0u);
    lt_nvadx(w, d + 1u, a + 1u, b + 1u, 0u);
    safire(w, x, 0u);
}

static void sang64(uint32_t d, uint32_t s, const lt_nrw_t *x)
{
    uint64_t w[2];

    lt_nvadd(w, d, LT_NV_RZ, s, 1u);
    safire(w, x, 0u);
    lt_nvstp(w, LT_NV_P0, s, LT_NV_RZ, 2u, 0u);
    safire(w, x, 0u);
    lt_nvlop(w, d + 1u, s + 1u, LT_NV_RZ, LT_NV_RZ, 0x0Fu);
    safire(w, x, 0u);
    lt_nvsei(w, SA_TMP2, LT_NV_RZ, 1u, LT_NV_P0, 1u);
    safire(w, x, 0u);
    lt_nvadd(w, d + 1u, d + 1u, SA_TMP2, 0u);
    safire(w, x, 0u);
}

static void samu64(uint32_t d, uint32_t a, uint32_t b, const lt_nrw_t *x)
{
    uint64_t w[2];

    lt_nvmad(w, d, a, b, LT_NV_RZ, 0u, 0u);
    safire(w, x, 0u);
    lt_nvmad(w, d + 1u, a, b, LT_NV_RZ, 1u, 0u);
    safire(w, x, 0u);
    lt_nvmad(w, d + 1u, a, b + 1u, d + 1u, 0u, 0u);
    safire(w, x, 0u);
    lt_nvmad(w, d + 1u, a + 1u, b, d + 1u, 0u, 0u);
    safire(w, x, 0u);
}

static void saal64(const nv_minst_t *I)
{
    lt_nrw_t x;
    uint32_t d, a, b;

    memset(&x, 0, sizeof x);
    if (sawchk(I, 2u) != 0) { sabad(I->op, BC_E611); return; }
    d = sadef(I, &x);
    a = sasrc(I->op, &I->ops[1], SA_IMM0, 2u, &x);
    b = sasrc(I->op, &I->ops[2], SA_IMM1, 2u, &x);
    samsr(&x.wr, SA_TMP0, 4u);
    samsr(&x.rd, SA_TMP0, 4u);
    samsr(&x.wr, SA_TMP2, 1u);
    samsr(&x.rd, SA_TMP2, 1u);
    if (I->op == (uint16_t)NV_ADD_U64) { saad64(d, a, b, &x); return; }
    if (I->op == (uint16_t)NV_SUB_S64) {
        sang64(SA_TMP0, b, &x);
        saad64(d, a, SA_TMP0, &x);
        return;
    }
    if (I->op == (uint16_t)NV_MUL_LO_U64) { samu64(d, a, b, &x); return; }
    sabad(I->op, BC_E601);
}

static void samad64(const nv_minst_t *I)
{
    lt_nrw_t x;
    uint32_t d, a, b, c;

    memset(&x, 0, sizeof x);
    if (sawchk(I, 2u) != 0) { sabad(I->op, BC_E611); return; }
    d = sadef(I, &x);
    a = sasrc(I->op, &I->ops[1], SA_IMM0, 2u, &x);
    b = sasrc(I->op, &I->ops[2], SA_IMM1, 2u, &x);
    c = sasrc(I->op, &I->ops[3], SA_IMM2, 2u, &x);
    samsr(&x.wr, SA_TMP0, 2u);
    samsr(&x.rd, SA_TMP0, 2u);
    samu64(SA_TMP0, a, b, &x);
    saad64(d, SA_TMP0, c, &x);
}

static void sacvt(const nv_minst_t *I)
{
    lt_nrw_t x;
    uint64_t w[2];
    uint32_t d, a;

    memset(&x, 0, sizeof x);
    if (I->ops[1].kind != (uint8_t)NV_MOP_REG
        || sadw(I) + savw(I->ops[1].rfile) != 3u
        || (I->op == (uint16_t)NV_CVT_U32_U64) != (sadw(I) == 1u)) {
        sabad(I->op, BC_E611);
        return;
    }
    d = sadef(I, &x);
    a = sasrc(I->op, &I->ops[1], SA_IMM0, 1u, &x);
    if (I->op == (uint16_t)NV_CVT_U32_U64) {
        lt_nvmov(w, d, a);
        safire(w, &x, 0u);
        return;
    }
    lt_nvmov(w, d, a);
    safire(w, &x, 0u);
    if (I->op == (uint16_t)NV_CVT_U64_U32) {
        lt_nvmov(w, d + 1u, LT_NV_RZ);
        safire(w, &x, 0u);
        return;
    }
    lt_nvshi(w, d + 1u, LT_NV_RZ, 31u, a, 2u, 1u, 1u);
    safire(w, &x, 0u);
}

static uint32_t sacc(uint16_t op)
{
    switch (op) {
    case NV_SETP_EQ_U32: case NV_SETP_EQ_F32:
    case NV_SETP_EQ_F64: case NV_SETP_EQ_U64: return 2u;
    case NV_SETP_NE_U32: case NV_SETP_NE_F32:
    case NV_SETP_NE_F64: case NV_SETP_NE_U64: return 5u;
    case NV_SETP_LT_U32: case NV_SETP_LT_S32:
    case NV_SETP_LT_F32: case NV_SETP_LT_F64:
    case NV_SETP_LT_S64: case NV_SETP_LT_U64: return 1u;
    case NV_SETP_LE_U32: case NV_SETP_LE_S32:
    case NV_SETP_LE_F32: case NV_SETP_LE_F64:
    case NV_SETP_LE_S64: case NV_SETP_LE_U64: return 3u;
    case NV_SETP_GT_U32: case NV_SETP_GT_S32:
    case NV_SETP_GT_F32: case NV_SETP_GT_F64:
    case NV_SETP_GT_S64: case NV_SETP_GT_U64: return 4u;
    default: return 6u;
    }
}

static uint32_t sasg(uint16_t op)
{
    switch (op) {
    case NV_SETP_LT_S32: case NV_SETP_LE_S32:
    case NV_SETP_GT_S32: case NV_SETP_GE_S32: return 1u;
    default: return 0u;
    }
}

static void satp32(const nv_minst_t *I, int flt)
{
    lt_nrw_t x;
    uint64_t w[2];
    uint32_t p, a, b;

    memset(&x, 0, sizeof x);
    if (sawchk(I, 1u) != 0) { sabad(I->op, BC_E611); return; }
    p = sapd(I);
    a = sasrc(I->op, &I->ops[1], SA_IMM0, 1u, &x);
    b = sasrc(I->op, &I->ops[2], SA_IMM1, 1u, &x);
    if (flt) lt_nvftp(w, p, a, b, sacc(I->op));
    else     lt_nvstp(w, p, a, b, sacc(I->op), sasg(I->op));
    safire(w, &x, 0u);
}

static uint32_t sats64(uint16_t op)
{
    switch (op) {
    case NV_SETP_LT_S64: case NV_SETP_LE_S64:
    case NV_SETP_GT_S64: case NV_SETP_GE_S64: return 1u;
    default: return 0u;
    }
}

static void satp64(const nv_minst_t *I)
{
    lt_nrw_t x;
    uint64_t w[2];
    uint32_t p, a, b, cmp = sacc(I->op), sg = sats64(I->op);

    memset(&x, 0, sizeof x);
    if (sawchk(I, 2u) != 0) { sabad(I->op, BC_E611); return; }
    p = sapd(I);
    a = sasrc(I->op, &I->ops[1], SA_IMM0, 2u, &x);
    b = sasrc(I->op, &I->ops[2], SA_IMM1, 2u, &x);
    lt_nvstp(w, p, a, b, cmp, 0u);
    safire(w, &x, 0u);
    lt_nvstx(w, p, a + 1u, b + 1u, cmp, sg, p);
    safire(w, &x, 0u);
}

static int sacvk(uint16_t op, uint32_t *sgn, uint32_t *dsz, uint32_t *ssz)
{
    switch (op) {
    case NV_CVT_F32_S32: *sgn = 1u; *dsz = 2u; *ssz = 2u; return 0;
    case NV_CVT_F32_U32: *sgn = 0u; *dsz = 2u; *ssz = 2u; return 0;
    case NV_CVT_F64_S32: *sgn = 1u; *dsz = 3u; *ssz = 2u; return 0;
    case NV_CVT_F64_U32: *sgn = 0u; *dsz = 3u; *ssz = 2u; return 0;
    case NV_CVT_F32_S64: *sgn = 1u; *dsz = 2u; *ssz = 3u; return 0;
    case NV_CVT_F32_U64: *sgn = 0u; *dsz = 2u; *ssz = 3u; return 0;
    case NV_CVT_F64_S64: *sgn = 1u; *dsz = 3u; *ssz = 3u; return 0;
    case NV_CVT_F64_U64: *sgn = 0u; *dsz = 3u; *ssz = 3u; return 0;
    case NV_CVT_S32_F32: *sgn = 1u; *dsz = 2u; *ssz = 2u; return 1;
    case NV_CVT_U32_F32: *sgn = 0u; *dsz = 2u; *ssz = 2u; return 1;
    case NV_CVT_S32_F64: *sgn = 1u; *dsz = 2u; *ssz = 3u; return 1;
    case NV_CVT_U32_F64: *sgn = 0u; *dsz = 2u; *ssz = 3u; return 1;
    case NV_CVT_S64_F32: *sgn = 1u; *dsz = 3u; *ssz = 2u; return 1;
    case NV_CVT_U64_F32: *sgn = 0u; *dsz = 3u; *ssz = 2u; return 1;
    case NV_CVT_S64_F64: *sgn = 1u; *dsz = 3u; *ssz = 3u; return 1;
    case NV_CVT_U64_F64: *sgn = 0u; *dsz = 3u; *ssz = 3u; return 1;
    case NV_CVT_F32_F64: *sgn = 0u; *dsz = 2u; *ssz = 3u; return 2;
    case NV_CVT_F64_F32: *sgn = 0u; *dsz = 3u; *ssz = 2u; return 2;
    default:             *sgn = 0u; *dsz = 0u; *ssz = 0u; return -1;
    }
}

static void sacvtf(const nv_minst_t *I)
{
    lt_nrw_t x;
    uint64_t w[2];
    uint32_t d, a, sgn, dsz, ssz, dw, aw;
    int kind = sacvk(I->op, &sgn, &dsz, &ssz);

    memset(&x, 0, sizeof x);
    dw = (dsz >= 3u) ? 2u : 1u;
    aw = (ssz >= 3u) ? 2u : 1u;
    if (kind < 0 || I->ops[1].kind != (uint8_t)NV_MOP_REG
        || sadw(I) != dw || savw(I->ops[1].rfile) != aw) {
        sabad(I->op, BC_E611);
        return;
    }
    d = sadef(I, &x);
    a = sasrc(I->op, &I->ops[1], SA_IMM0, aw, &x);
    if (kind == 0)      lt_nvi2f(w, d, a, sgn, dsz, ssz);
    else if (kind == 1) lt_nvf2i(w, d, a, sgn, dsz, ssz);
    else                lt_nvf2f(w, d, a, dsz, ssz, 0u);
    safire(w, &x, 1u);
}

static uint32_t samfu(uint16_t op)
{
    switch (op) {
    case NV_COS_F32:  return 0u;
    case NV_SIN_F32:  return 1u;
    case NV_EX2_F32:  return 2u;
    case NV_LG2_F32:  return 3u;
    case NV_RCP_F32:  return 4u;
    case NV_RSQ_F32:  return 5u;
    case NV_SQRT_F32: return 8u;
    default:          return 64u;
    }
}

static void samuf(const nv_minst_t *I)
{
    lt_nrw_t x;
    uint64_t w[2];
    uint32_t d, a, f = samfu(I->op);

    memset(&x, 0, sizeof x);
    if (f >= 64u || sawchk(I, 1u) != 0) { sabad(I->op, BC_E611); return; }
    d = sadef(I, &x);
    a = sasrc(I->op, &I->ops[1], SA_IMM0, 1u, &x);
    lt_nvmuf(w, d, a, f);
    safire(w, &x, 1u);
}

#define SA_DW  1u
#define SCT(k) ((uint32_t)(k))
#define SCD(k) ((uint32_t)(LT_NV_D0 + 2u * (k)))

static void sadcon(const lt_nrw_t *x, uint32_t d, uint32_t hi)
{
    uint64_t w[2];

    lt_nvmovi(w, d, 0u);
    safire(w, x, 0u);
    lt_nvmovi(w, d + 1u, hi);
    safire(w, x, 0u);
}

static void sadnewt(const lt_nrw_t *x)
{
    uint64_t w[2];

    sadcon(x, SCD(2u), 0x3FE80000u);
    sadcon(x, SCD(4u), 0x3FF00000u);
    for (uint32_t k = 0; k < 6u; k++) {
        lt_nvdfm(w, SCD(3u), SCD(1u), SCD(2u), SCD(4u), 1u);
        safire(w, x, SA_DW);
        lt_nvdfm(w, SCD(2u), SCD(2u), SCD(3u), SCD(2u), 0u);
        safire(w, x, SA_DW);
    }
}

static void sadquo(const lt_nrw_t *x)
{
    uint64_t w[2];

    lt_nvshi(w, SCT(0u), LT_NV_RZ, 20u, SCD(1u) + 1u, 3u, 1u, 1u);
    safire(w, x, 0u);
    lt_nvlpi(w, SCT(0u), SCT(0u), 0x7FFu, 0xC0u);
    safire(w, x, 0u);
    lt_nvlpi(w, SCD(1u) + 1u, SCD(1u) + 1u, 0x000FFFFFu, 0xC0u);
    safire(w, x, 0u);
    lt_nvlpi(w, SCD(1u) + 1u, SCD(1u) + 1u, 0x3FF00000u, 0xFCu);
    safire(w, x, 0u);
    sadnewt(x);
    lt_nvdmu(w, SCD(0u), SCD(0u), SCD(2u));
    safire(w, x, SA_DW);
    lt_nvadi(w, SCT(0u), SCT(0u), 2046u, 1u);
    safire(w, x, 0u);
    lt_nvshi(w, SCT(0u), SCT(0u), 20u, LT_NV_RZ, 3u, 0u, 0u);
    safire(w, x, 0u);
    lt_nvmovi(w, SCD(3u), 0u);
    safire(w, x, 0u);
    lt_nvmov(w, SCD(3u) + 1u, SCT(0u));
    safire(w, x, 0u);
    lt_nvdmu(w, SCD(0u), SCD(0u), SCD(3u));
    safire(w, x, SA_DW);
}

static void safdivs(const lt_nrw_t *x, uint32_t d, uint32_t aa, uint32_t bb)
{
    uint64_t w[2];

    lt_nvlpi(w, SCT(2u), aa, 0x7FFFFFFFu, 0xC0u);
    safire(w, x, 0u);
    lt_nvlpi(w, SCT(3u), bb, 0x7FFFFFFFu, 0xC0u);
    safire(w, x, 0u);
    lt_nvsti(w, LT_NV_P0, SCT(3u), 0x7F800000u, 2u, 0u);
    safire(w, x, 0u);
    lt_nvlop(w, SCT(0u), aa, bb, LT_NV_RZ, 0x3Cu);
    safire(w, x, 0u);
    lt_nvlpi(w, SCT(0u), SCT(0u), 0x80000000u, 0xC0u);
    safire(w, x, 0u);
    lt_nvsel(w, d, SCT(0u), d, LT_NV_P0, 0u);
    safire(w, x, 0u);
    lt_nvstp(w, LT_NV_P1, SCT(2u), SCT(3u), 2u, 0u);
    safire(w, x, 0u);
    lt_nvstc(w, LT_NV_P0, SCT(2u), LT_NV_RZ, 2u, LT_NV_P1, 0u);
    safire(w, x, 0u);
    lt_nvsei(w, d, d, 0x7FC00000u, LT_NV_P0, 1u);
    safire(w, x, 0u);
    lt_nvmovi(w, SCT(0u), 0x7F800000u);
    safire(w, x, 0u);
    lt_nvstc(w, LT_NV_P0, SCT(2u), SCT(0u), 2u, LT_NV_P1, 0u);
    safire(w, x, 0u);
    lt_nvsei(w, d, d, 0x7FC00000u, LT_NV_P0, 1u);
    safire(w, x, 0u);
    lt_nvftp(w, LT_NV_P0, aa, bb, 8u);
    safire(w, x, 0u);
    lt_nvsei(w, d, d, 0x7FC00000u, LT_NV_P0, 1u);
    safire(w, x, 0u);
}

static void safdiv32(const lt_nrw_t *x, uint32_t d, uint32_t a, uint32_t b)
{
    uint32_t aa = SCD(5u), bb = SCD(5u) + 1u;
    uint64_t w[2];

    lt_nvmov(w, aa, a);
    safire(w, x, 0u);
    lt_nvmov(w, bb, b);
    safire(w, x, 0u);
    lt_nvf2f(w, SCD(0u), aa, 3u, 2u, 0u);
    safire(w, x, SA_DW);
    lt_nvf2f(w, SCD(1u), bb, 3u, 2u, 0u);
    safire(w, x, SA_DW);
    sadquo(x);
    lt_nvlpi(w, SCT(1u), bb, 0x80000000u, 0xC0u);
    safire(w, x, 0u);
    lt_nvlop(w, SCD(0u) + 1u, SCD(0u) + 1u, SCT(1u), LT_NV_RZ, 0x3Cu);
    safire(w, x, 0u);
    lt_nvf2f(w, SCT(1u), SCD(0u), 2u, 3u, 0u);
    safire(w, x, SA_DW);
    safdivs(x, SCT(1u), aa, bb);
    lt_nvmov(w, d, SCT(1u));
    safire(w, x, 0u);
}

static void saudivc(const lt_nrw_t *x, uint32_t xr, uint32_t yr)
{
    uint64_t w[2];

    lt_nvmad(w, SCT(2u), SCT(1u), yr, LT_NV_RZ, 0u, 0u);
    safire(w, x, 0u);
    lt_nvmad(w, SCT(3u), SCT(1u), yr, LT_NV_RZ, 1u, 0u);
    safire(w, x, 0u);
    lt_nvstp(w, LT_NV_P1, SCT(2u), xr, 4u, 0u);
    safire(w, x, 0u);
    lt_nvstx(w, LT_NV_P0, SCT(3u), LT_NV_RZ, 4u, 0u, LT_NV_P1);
    safire(w, x, 0u);
    lt_nvsei(w, SCT(0u), LT_NV_RZ, 1u, LT_NV_P0, 1u);
    safire(w, x, 0u);
    lt_nvadd(w, SCT(1u), SCT(1u), SCT(0u), 1u);
    safire(w, x, 0u);
    lt_nvmad(w, SCT(2u), SCT(1u), yr, LT_NV_RZ, 0u, 0u);
    safire(w, x, 0u);
    lt_nvadd(w, SCT(2u), xr, SCT(2u), 1u);
    safire(w, x, 0u);
    lt_nvstp(w, LT_NV_P0, SCT(2u), yr, 6u, 0u);
    safire(w, x, 0u);
    lt_nvsei(w, SCT(0u), LT_NV_RZ, 1u, LT_NV_P0, 1u);
    safire(w, x, 0u);
    lt_nvadd(w, SCT(1u), SCT(1u), SCT(0u), 0u);
    safire(w, x, 0u);
    lt_nvsel(w, SCT(0u), yr, LT_NV_RZ, LT_NV_P0, 0u);
    safire(w, x, 0u);
    lt_nvadd(w, SCT(2u), SCT(2u), SCT(0u), 1u);
    safire(w, x, 0u);
}

static void saudiv32(const lt_nrw_t *x, uint32_t xr, uint32_t yr)
{
    uint64_t w[2];

    lt_nvi2f(w, SCD(0u), xr, 0u, 3u, 2u);
    safire(w, x, SA_DW);
    lt_nvi2f(w, SCD(1u), yr, 0u, 3u, 2u);
    safire(w, x, SA_DW);
    sadquo(x);
    lt_nvf2i(w, SCT(1u), SCD(0u), 0u, 2u, 3u);
    safire(w, x, SA_DW);
    saudivc(x, xr, yr);
}

static void sasgnf(const lt_nrw_t *x, uint32_t d, uint32_t r, uint32_t sx,
                   uint32_t sy)
{
    uint64_t w[2];

    lt_nvshi(w, SCT(0u), LT_NV_RZ, 31u, sx, 2u, 1u, 1u);
    safire(w, x, 0u);
    if (sy != LT_NV_RZ) {
        lt_nvshi(w, SCD(0u), LT_NV_RZ, 31u, sy, 2u, 1u, 1u);
        safire(w, x, 0u);
        lt_nvlop(w, SCT(0u), SCT(0u), SCD(0u), LT_NV_RZ, 0x3Cu);
        safire(w, x, 0u);
    }
    lt_nvlop(w, d, r, SCT(0u), LT_NV_RZ, 0x3Cu);
    safire(w, x, 0u);
    lt_nvadd(w, d, d, SCT(0u), 1u);
    safire(w, x, 0u);
}

static void sadiv32(const nv_minst_t *I, int sg, int rem)
{
    lt_nrw_t y;
    uint64_t w[2];
    uint32_t d, sx, sy, xr = SCD(5u), yr = SCD(5u) + 1u;

    if (I->ops[1].kind != (uint8_t)NV_MOP_REG
        || I->ops[2].kind != (uint8_t)NV_MOP_REG) {
        sabad(I->op, BC_E609);
        return;
    }
    d  = sarg(&I->ops[0]);
    sx = sarg(&I->ops[1]);
    sy = sarg(&I->ops[2]);
    memset(&y, 0, sizeof y);
    samsr(&y.rd, 0u, LT_NV_NSCR);
    samsr(&y.wr, 0u, LT_NV_NSCR);
    samsr(&y.rd, sx, 1u);
    samsr(&y.rd, sy, 1u);
    samsr(&y.wr, d, 1u);
    if (sg != 0) {
        sasgnf(&y, xr, sx, sx, LT_NV_RZ);
        sasgnf(&y, yr, sy, sy, LT_NV_RZ);
    } else {
        lt_nvmov(w, xr, sx);
        safire(w, &y, 0u);
        lt_nvmov(w, yr, sy);
        safire(w, &y, 0u);
    }
    saudiv32(&y, xr, yr);
    lt_nvmov(w, SCT(3u), rem ? SCT(2u) : SCT(1u));
    safire(w, &y, 0u);
    if (sg != 0)
        sasgnf(&y, SCT(3u), SCT(3u), sx, rem ? LT_NV_RZ : sy);
    lt_nvstp(w, LT_NV_P0, sy, LT_NV_RZ, 2u, 0u);
    safire(w, &y, 0u);
    lt_nvsei(w, d, SCT(3u), 0u, LT_NV_P0, 1u);
    safire(w, &y, 0u);
}

static void safdiv(const nv_minst_t *I)
{
    lt_nrw_t y;
    uint32_t d, a, b;

    if (I->ops[1].kind != (uint8_t)NV_MOP_REG
        || I->ops[2].kind != (uint8_t)NV_MOP_REG) {
        sabad(I->op, BC_E609);
        return;
    }
    d = sarg(&I->ops[0]);
    a = sarg(&I->ops[1]);
    b = sarg(&I->ops[2]);
    memset(&y, 0, sizeof y);
    samsr(&y.rd, 0u, LT_NV_NSCR);
    samsr(&y.wr, 0u, LT_NV_NSCR);
    samsr(&y.rd, a, 1u);
    samsr(&y.rd, b, 1u);
    samsr(&y.wr, d, 1u);
    safdiv32(&y, d, a, b);
}

static void sadpick(const lt_nrw_t *x, uint32_t d, uint32_t a, uint32_t b,
                    uint32_t p, uint32_t nt)
{
    uint32_t ah = (a == LT_NV_RZ) ? LT_NV_RZ : a + 1u;
    uint32_t bh = (b == LT_NV_RZ) ? LT_NV_RZ : b + 1u;
    uint64_t w[2];

    lt_nvsel(w, d, a, b, p, nt);
    safire(w, x, 0u);
    lt_nvsel(w, d + 1u, ah, bh, p, nt);
    safire(w, x, 0u);
}

static void saq64abs(const lt_nrw_t *x, uint32_t d, uint32_t s, uint32_t sh)
{
    uint64_t w[2];

    lt_nvadc(w, SCT(0u), LT_NV_RZ, s, 1u);
    safire(w, x, 0u);
    lt_nvlop(w, SCT(1u), sh, LT_NV_RZ, LT_NV_RZ, 0x0Fu);
    safire(w, x, 0u);
    lt_nvadx(w, SCT(1u), LT_NV_RZ, SCT(1u), 0u);
    safire(w, x, 0u);
    lt_nvstp(w, LT_NV_P0, sh, LT_NV_RZ, 1u, 1u);
    safire(w, x, 0u);
    lt_nvsel(w, d, SCT(0u), s, LT_NV_P0, 0u);
    safire(w, x, 0u);
    lt_nvsel(w, d + 1u, SCT(1u), sh, LT_NV_P0, 0u);
    safire(w, x, 0u);
}

static void saq64stp(const lt_nrw_t *x)
{
    uint64_t w[2];

    lt_nvshi(w, SCT(0u), LT_NV_RZ, 31u, SCD(0u) + 1u, 3u, 1u, 1u);
    safire(w, x, 0u);
    lt_nvshi(w, SCD(0u) + 1u, SCD(0u), 1u, SCD(0u) + 1u, 1u, 0u, 1u);
    safire(w, x, 0u);
    lt_nvshi(w, SCD(0u), SCD(1u) + 1u, 1u, SCD(0u), 1u, 0u, 1u);
    safire(w, x, 0u);
    lt_nvshi(w, SCD(1u) + 1u, SCD(1u), 1u, SCD(1u) + 1u, 1u, 0u, 1u);
    safire(w, x, 0u);
    lt_nvshi(w, SCD(1u), SCD(1u), 1u, LT_NV_RZ, 3u, 0u, 0u);
    safire(w, x, 0u);
    lt_nvstp(w, LT_NV_P1, SCD(0u), SCD(5u), 6u, 0u);
    safire(w, x, 0u);
    lt_nvstx(w, LT_NV_P0, SCD(0u) + 1u, SCD(5u) + 1u, 6u, 0u, LT_NV_P1);
    safire(w, x, 0u);
    lt_nvsei(w, SCT(1u), LT_NV_RZ, 1u, LT_NV_P0, 1u);
    safire(w, x, 0u);
    lt_nvlop(w, SCT(1u), SCT(1u), SCT(0u), LT_NV_RZ, 0xFCu);
    safire(w, x, 0u);
    lt_nvstp(w, LT_NV_P0, SCT(1u), LT_NV_RZ, 5u, 0u);
    safire(w, x, 0u);
    lt_nvlop(w, SCD(1u), SCD(1u), SCT(1u), LT_NV_RZ, 0xFCu);
    safire(w, x, 0u);
    lt_nvsel(w, SCT(0u), SCD(5u), LT_NV_RZ, LT_NV_P0, 0u);
    safire(w, x, 0u);
    lt_nvsel(w, SCT(1u), SCD(5u) + 1u, LT_NV_RZ, LT_NV_P0, 0u);
    safire(w, x, 0u);
    lt_nvlop(w, SCT(1u), SCT(1u), LT_NV_RZ, LT_NV_RZ, 0x0Fu);
    safire(w, x, 0u);
    lt_nvadc(w, SCD(0u), SCD(0u), SCT(0u), 1u);
    safire(w, x, 0u);
    lt_nvadx(w, SCD(0u) + 1u, SCD(0u) + 1u, SCT(1u), 0u);
    safire(w, x, 0u);
}

static void saq64lop(const lt_nrw_t *x)
{
    uint8_t  g = sast.gate;
    uint32_t hd;
    uint64_t w[2];

    lt_nvmovi(w, SCT(3u), 64u);
    safire(w, x, 0u);
    sadrn();
    hd = sast.clen;
    saq64stp(x);
    sast.gate = (uint8_t)LT_NV_PT;
    lt_nvadi(w, SCT(3u), SCT(3u), 0xFFFFFFFFu, 0u);
    safire(w, x, 0u);
    lt_nvstp(w, LT_NV_P1, SCT(3u), LT_NV_RZ, 5u, 0u);
    safire(w, x, 0u);
    sadrn();
    lt_nvbrp(w, ((int32_t)hd - (int32_t)sast.clen - 16) / 4, LT_NV_P1, 0u);
    safire(w, x, 0u);
    sast.gate = g;
}

static void saq64sgn(const lt_nrw_t *x, uint32_t d, uint32_t s, uint32_t sa,
                     uint32_t sb)
{
    uint64_t w[2];

    lt_nvadc(w, SCT(0u), LT_NV_RZ, s, 1u);
    safire(w, x, 0u);
    lt_nvlop(w, SCT(1u), s + 1u, LT_NV_RZ, LT_NV_RZ, 0x0Fu);
    safire(w, x, 0u);
    lt_nvadx(w, SCT(1u), LT_NV_RZ, SCT(1u), 0u);
    safire(w, x, 0u);
    lt_nvlop(w, SCT(2u), sa, sb, LT_NV_RZ, 0x3Cu);
    safire(w, x, 0u);
    lt_nvstp(w, LT_NV_P0, SCT(2u), LT_NV_RZ, 1u, 1u);
    safire(w, x, 0u);
    lt_nvsel(w, d, SCT(0u), s, LT_NV_P0, 0u);
    safire(w, x, 0u);
    lt_nvsel(w, d + 1u, SCT(1u), s + 1u, LT_NV_P0, 0u);
    safire(w, x, 0u);
}

static void sadiv64(const nv_minst_t *I, int sg, int rem)
{
    lt_nrw_t y;
    uint64_t w[2];
    uint32_t d, a0, a1, b0, b1;

    if (I->ops[1].kind != (uint8_t)NV_MOP_REG
        || I->ops[2].kind != (uint8_t)NV_MOP_REG) {
        sabad(I->op, BC_E609);
        return;
    }
    d  = sarg(&I->ops[0]);
    a0 = sarg(&I->ops[1]); a1 = a0 + 1u;
    b0 = sarg(&I->ops[2]); b1 = b0 + 1u;
    memset(&y, 0, sizeof y);
    samsr(&y.rd, 0u, LT_NV_NSCR);
    samsr(&y.wr, 0u, LT_NV_NSCR);
    samsr(&y.rd, a0, 2u);
    samsr(&y.rd, b0, 2u);
    samsr(&y.wr, d, 2u);
    if (sg != 0) {
        saq64abs(&y, SCD(1u), a0, a1);
        saq64abs(&y, SCD(5u), b0, b1);
    } else {
        lt_nvmov(w, SCD(1u), a0);
        safire(w, &y, 0u);
        lt_nvmov(w, SCD(1u) + 1u, a1);
        safire(w, &y, 0u);
        lt_nvmov(w, SCD(5u), b0);
        safire(w, &y, 0u);
        lt_nvmov(w, SCD(5u) + 1u, b1);
        safire(w, &y, 0u);
    }
    lt_nvmovi(w, SCD(0u), 0u);
    safire(w, &y, 0u);
    lt_nvmovi(w, SCD(0u) + 1u, 0u);
    safire(w, &y, 0u);
    saq64lop(&y);
    if (sg != 0)
        saq64sgn(&y, SCD(2u), rem ? SCD(0u) : SCD(1u), a1,
                 rem ? LT_NV_RZ : b1);
    else
        sadpick(&y, SCD(2u), rem ? SCD(0u) : SCD(1u), LT_NV_RZ,
                LT_NV_PT, 0u);
    lt_nvlop(w, SCT(0u), b0, b1, LT_NV_RZ, 0xFCu);
    safire(w, &y, 0u);
    lt_nvstp(w, LT_NV_P0, SCT(0u), LT_NV_RZ, 5u, 0u);
    safire(w, &y, 0u);
    sadpick(&y, d, SCD(2u), LT_NV_RZ, LT_NV_P0, 0u);
}

static void samnx(const nv_minst_t *I, int mx)
{
    lt_nrw_t x;
    uint64_t w[2];
    uint32_t d, a, b;

    memset(&x, 0, sizeof x);
    if (sawchk(I, 1u) != 0) { sabad(I->op, BC_E611); return; }
    d = sadef(I, &x);
    a = sasrc(I->op, &I->ops[1], SA_IMM0, 1u, &x);
    b = sasrc(I->op, &I->ops[2], SA_IMM1, 1u, &x);
    lt_nvftp(w, LT_NV_P0, a, b, mx ? 4u : 1u);
    safire(w, &x, 0u);
    lt_nvsel(w, d, a, b, LT_NV_P0, 0u);
    safire(w, &x, 0u);
    lt_nvftp(w, LT_NV_P0, a, b, 8u);
    safire(w, &x, 0u);
    lt_nvsei(w, d, d, 0x7FC00000u, LT_NV_P0, 1u);
    safire(w, &x, 0u);
}

static uint32_t sats(uint16_t op)
{
    switch (op) {
    case NV_LD_GLB_U8:  case NV_ST_GLB_U8:
    case NV_LD_SHR_U8:  case NV_ST_SHR_U8:  return 0u;
    case NV_LD_GLB_U16: case NV_ST_GLB_U16:
    case NV_LD_GLB_B16: case NV_ST_GLB_B16:
    case NV_LD_SHR_U16: case NV_ST_SHR_U16:
    case NV_LD_SHR_B16: case NV_ST_SHR_B16: return 2u;
    case NV_LD_GLB_U64: case NV_ST_GLB_U64:
    case NV_LD_GLB_F64: case NV_ST_GLB_F64: return 5u;
    default: return 4u;
    }
}

static void samem(const nv_minst_t *I, int shr, int st)
{
    lt_nrw_t x;
    uint64_t w[2];
    uint32_t t = sats(I->op);
    const nv_opnd_t *ao = st ? &I->ops[0] : &I->ops[1];
    const nv_opnd_t *vo = st ? &I->ops[1] : &I->ops[0];
    uint32_t ra, rv;

    memset(&x, 0, sizeof x);
    if (ao->kind != (uint8_t)NV_MOP_REG || vo->kind != (uint8_t)NV_MOP_REG) {
        sabad(I->op, BC_E609);
        return;
    }
    if (!shr && ao->rfile != (uint8_t)NV_RF_U64) {
        sabad(I->op, BC_E611);
        return;
    }
    if (shr && ao->rfile != (uint8_t)NV_RF_U64
        && ao->rfile != (uint8_t)NV_RF_U32) {
        sabad(I->op, BC_E611);
        return;
    }
    if ((t == 5u) != (savw(vo->rfile) == 2u)) {
        sabad(I->op, BC_E611);
        return;
    }
    ra = sarg(ao);
    rv = sarg(vo);
    samsr(&x.rd, ra, savw(ao->rfile));
    if (st) samsr(&x.rd, rv, savw(vo->rfile));
    else    samsr(&x.wr, rv, savw(vo->rfile));
    if (shr && st)       lt_nvsts(w, ra, rv, 0u, t);
    else if (shr)        lt_nvlds(w, rv, ra, 0u, t);
    else if (st)         lt_nvstg(w, ra, rv, 0u, t, 0u);
    else                 lt_nvldg(w, rv, ra, 0u, t, 0u);
    safire(w, &x, st ? 2u : 3u);
}

static void sapld(const nv_minst_t *I)
{
    lt_nrw_t x;
    uint64_t w[2];
    uint32_t d, o, n;

    memset(&x, 0, sizeof x);
    n = sadw(I);
    if ((I->op == (uint16_t)NV_LD_PARAM_U64
         || I->op == (uint16_t)NV_LD_PARAM_F64) != (n == 2u)) {
        sabad(I->op, BC_E611);
        return;
    }
    d = sadef(I, &x);
    if (I->ops[1].kind != (uint8_t)NV_MOP_IMM || I->ops[1].imm < 0
        || (uint32_t)I->ops[1].imm >= sast.k.np) {
        sabad(I->op, BC_E609);
        return;
    }
    o = LT_NV_PBASE + sast.pofs[I->ops[1].imm];
    lt_nvmovc(w, d, 0u, o);
    safire(w, &x, 0u);
    if (n != 2u) return;
    lt_nvmovc(w, d + 1u, 0u, o + 4u);
    safire(w, &x, 0u);
}

static void sactl(const nv_minst_t *I)
{
    lt_nrw_t x;
    uint64_t w[2];
    uint32_t p = LT_NV_PT;
    int32_t  lb;

    memset(&x, 0, sizeof x);
    if (I->op == (uint16_t)NV_BRA_PRED) p = saprd(&I->ops[0]);
    lb = (int32_t)satgt(I->op == (uint16_t)NV_BRA_PRED ? &I->ops[1]
                                                       : &I->ops[0]);
    sast.gate = (uint8_t)LT_NV_PT;
    sabrk(sast.cf.cur, 1u, p);
    sadrn();
    sajmp(lb);
    lt_nvbrp(w, 0, p, 0u);
    safire(w, &x, 0u);
}

static void sabar(void)
{
    lt_nrw_t x;
    uint64_t w[2];

    memset(&x, 0, sizeof x);
    lt_nvsyn(w);
    sadrn();
    safire(w, &x, 0u);
}

static void saquit(void)
{
    lt_nrw_t x;
    uint64_t w[2];

    memset(&x, 0, sizeof x);
    lt_nvexit(w);
    safire(w, &x, 0u);
}

static void samemc(const nv_minst_t *I)
{
    switch (I->op) {
    case NV_LD_GLB_U32: case NV_LD_GLB_U64:
    case NV_LD_GLB_F32: case NV_LD_GLB_F64:
    case NV_LD_GLB_U8:  case NV_LD_GLB_U16:
    case NV_LD_GLB_B16:                 samem(I, 0, 0); return;
    case NV_ST_GLB_U32: case NV_ST_GLB_U64:
    case NV_ST_GLB_F32: case NV_ST_GLB_F64:
    case NV_ST_GLB_U8:  case NV_ST_GLB_U16:
    case NV_ST_GLB_B16:                 samem(I, 0, 1); return;
    case NV_LD_SHR_U32: case NV_LD_SHR_F32:
    case NV_LD_SHR_U8:  case NV_LD_SHR_U16:
    case NV_LD_SHR_B16:                 samem(I, 1, 0); return;
    case NV_ST_SHR_U32: case NV_ST_SHR_F32:
    case NV_ST_SHR_U8:  case NV_ST_SHR_U16:
    case NV_ST_SHR_B16:                 samem(I, 1, 1); return;
    case NV_LD_PARAM_U32: case NV_LD_PARAM_U64:
    case NV_LD_PARAM_F32: case NV_LD_PARAM_F64: sapld(I); return;
    case NV_BRA: case NV_BRA_PRED:      sactl(I); return;
    case NV_BAR_SYNC:                   sabar(); return;
    case NV_RET: case NV_EXIT:          saquit(); return;
    default:                            sabad(I->op, BC_E601); return;
    }
}

static void saext(const nv_minst_t *I)
{
    switch (I->op) {
    case NV_CVT_F32_S32: case NV_CVT_F32_U32:
    case NV_CVT_F64_S32: case NV_CVT_F64_U32:
    case NV_CVT_F32_S64: case NV_CVT_F32_U64:
    case NV_CVT_F64_S64: case NV_CVT_F64_U64:
    case NV_CVT_S32_F32: case NV_CVT_U32_F32:
    case NV_CVT_S32_F64: case NV_CVT_U32_F64:
    case NV_CVT_S64_F32: case NV_CVT_U64_F32:
    case NV_CVT_S64_F64: case NV_CVT_U64_F64:
    case NV_CVT_F32_F64: case NV_CVT_F64_F32: sacvtf(I); return;
    case NV_SQRT_F32: case NV_RSQ_F32: case NV_RCP_F32:
    case NV_SIN_F32: case NV_COS_F32:
    case NV_EX2_F32: case NV_LG2_F32:  samuf(I); return;
    case NV_DIV_U32:                    sadiv32(I, 0, 0); return;
    case NV_DIV_S32:                    sadiv32(I, 1, 0); return;
    case NV_REM_U32:                    sadiv32(I, 0, 1); return;
    case NV_REM_S32:                    sadiv32(I, 1, 1); return;
    case NV_DIV_U64:                    sadiv64(I, 0, 0); return;
    case NV_DIV_S64:                    sadiv64(I, 1, 0); return;
    case NV_REM_U64:                    sadiv64(I, 0, 1); return;
    case NV_REM_S64:                    sadiv64(I, 1, 1); return;
    case NV_DIV_F32:                    safdiv(I); return;
    case NV_MAX_F32:                    samnx(I, 1); return;
    case NV_MIN_F32:                    samnx(I, 0); return;
    default:                            samemc(I); return;
    }
}

static void sastep(const nv_minst_t *I)
{
    sast.cop = I->op;
    switch (I->op) {
    case NV_MOV_U32: case NV_MOV_U64:
    case NV_MOV_F32: case NV_MOV_F64:  samov(I); return;
    case NV_MOV_F64_LIT:               sadlit(I); return;
    case NV_AND_B32: case NV_OR_B32:
    case NV_XOR_B32:                   sabit(I, 0, 0); return;
    case NV_NOT_B32:                   sabit(I, 1, 0); return;
    case NV_AND_B64: case NV_OR_B64:
    case NV_XOR_B64:                   sabit(I, 0, 1); return;
    case NV_NOT_B64:                   sabit(I, 1, 1); return;
    case NV_ADD_U32: case NV_ADD_S32:
    case NV_SUB_U32: case NV_SUB_S32:
    case NV_NEG_S32:
    case NV_MUL_LO_U32: case NV_MUL_LO_S32:
    case NV_MUL_HI_U32: case NV_MUL_HI_S32:
    case NV_ADD_F32: case NV_SUB_F32: case NV_MUL_F32:
    case NV_NEG_F32: case NV_ABS_F32:
    case NV_POPC_B32: case NV_BREV_B32: saal32(I); return;
    case NV_SHL_B32: case NV_SHR_U32: case NV_SHR_S32:
        if (sadw(I) == 2u) sash64(I); else saal32(I);
        return;
    case NV_SHL_B64: case NV_SHR_U64:
    case NV_SHR_S64:                    sash64(I); return;
    case NV_CLZ_B32:                    saclz(I); return;
    case NV_FMA_F32:                    safma(I); return;
    case NV_SELP_U32: case NV_SELP_F32: sasel(I, 0); return;
    case NV_SELP_U64: case NV_SELP_F64: sasel(I, 1); return;
    case NV_ADD_U64: case NV_SUB_S64:
    case NV_MUL_LO_U64:                 saal64(I); return;
    case NV_MAD_LO_U64:                 samad64(I); return;
    case NV_CVT_U64_U32: case NV_CVT_S64_S32:
    case NV_CVT_U32_U64:                sacvt(I); return;
    case NV_SETP_EQ_U32: case NV_SETP_NE_U32:
    case NV_SETP_LT_U32: case NV_SETP_LE_U32:
    case NV_SETP_GT_U32: case NV_SETP_GE_U32:
    case NV_SETP_LT_S32: case NV_SETP_LE_S32:
    case NV_SETP_GT_S32: case NV_SETP_GE_S32: satp32(I, 0); return;
    case NV_SETP_EQ_F32: case NV_SETP_NE_F32:
    case NV_SETP_LT_F32: case NV_SETP_LE_F32:
    case NV_SETP_GT_F32: case NV_SETP_GE_F32: satp32(I, 1); return;
    case NV_SETP_EQ_U64: case NV_SETP_NE_U64:
    case NV_SETP_LT_S64: case NV_SETP_LE_S64:
    case NV_SETP_GT_S64: case NV_SETP_GE_S64:
    case NV_SETP_LT_U64: case NV_SETP_LE_U64:
    case NV_SETP_GT_U64: case NV_SETP_GE_U64: satp64(I); return;
    case NV_MOV_B16: case NV_MOV_B32: case NV_MOV_B64: samov(I); return;
    default:                            saext(I); return;
    }
}

static uint32_t sapsz(uint8_t rf)
{
    if (rf == (uint8_t)NV_RF_U64 || rf == (uint8_t)NV_RF_F64) return 8u;
    if (rf == (uint8_t)NV_RF_U16 || rf == (uint8_t)NV_RF_F16) return 2u;
    return 4u;
}

static void sapar(void)
{
    uint32_t o = 0u;

    sast.k.np = sast.F->num_params;
    for (uint32_t i = 0; i < sast.k.np && i < LT_MAX_NVPRM; i++) {
        uint32_t z = sapsz(sast.F->params[i].rfile);

        o = (o + z - 1u) & ~(z - 1u);
        sast.pofs[i] = o;
        sast.k.psz[i] = (uint8_t)z;
        o += z;
    }
}

static void sazero(void)
{
    memset(&sast.cf, 0, sizeof sast.cf);
    memset(&sast.k, 0, sizeof sast.k);
    memset(sast.bkd, 0, sizeof sast.bkd);
    memset(sast.brs, 0, sizeof sast.brs);
    memset(sast.rmap, 0xFF, sizeof sast.rmap);
    memset(sast.vlo, 0xFF, sizeof sast.vlo);
    memset(sast.vhi, 0, sizeof sast.vhi);
    memset(sast.vbk, 0, sizeof sast.vbk);
    memset(sast.vlc, 0, sizeof sast.vlc);
    memset(sast.vwd, 0, sizeof sast.vwd);
    memset(sast.vpr, 0, sizeof sast.vpr);
    memset(sast.ru, 0xFF, sizeof sast.ru);
    memset(sast.pu, 0xFF, sizeof sast.pu);
    sast.clen = 0u;
    sast.nfx  = 0u;
    sast.pend = 0u;
    sast.nlp  = 0u;
    sast.lpof = 0u;
    sast.nerr = 0u;
    sast.nbad = 0u;
    sast.npr  = 0u;
    sast.gate = (uint8_t)LT_NV_PT;
    sast.cop  = 0xFFFEu;
}

static void sarun(void)
{
    uint64_t w[2];

    for (uint32_t k = 0; k < sast.cf.nbb; k++) {
        sast.cop = 0xFFFEu;
        if (k != 0u) sabrk(k - 1u, 0u, LT_NV_PT);
        sast.cf.cur = k;
        sast.lat[k] = sast.clen;
        sadrn();
        sahead(k);
        for (uint32_t i = sast.ist[k]; i < sast.ist[k + 1u]; i++)
            sastep(&sast.M->minsts[sast.f0 + i]);
    }
    sast.cop = 0xFFFEu;
    sast.lat[sast.cf.nbb] = sast.clen;
    lt_nvexit(w);
    saput(w);
    lt_nvbra(w, -4);
    saput(w);
    sastub();
    while (sast.clen < 128u) { lt_nvnop(w); saput(w); }
    sapat();
}

static int sakern(const nv_module_t *M, uint32_t *fi)
{
    uint32_t n = 0u;

    for (uint32_t i = 0; i < M->num_mfunc; i++) {
        if (M->mfuncs[i].is_kern == 0u) continue;
        *fi = i;
        n++;
    }
    if (n == 1u) return 0;
    (void)be_fail(BC_E600, "nvptx", (unsigned)n);
    return 1;
}

static int saname(void)
{
    const char *s;
    uint32_t    o = sast.F->name;

    if (sast.M->bir == NULL || o >= sast.M->bir->string_len) return 1;
    s = &sast.M->bir->strings[o];
    if (strlen(s) + 1u > LT_MAX_NVNAME) return 1;
    memcpy(sast.name, s, strlen(s) + 1u);
    sast.k.name = sast.name;
    return 0;
}

static int sabld(const nv_module_t *M, uint32_t fi)
{
    const nv_mfunc_t *F = &M->mfuncs[fi];
    uint32_t last;

    sazero();
    sast.M = M;
    sast.F = F;
    if (F->num_blks == 0u || F->num_params > LT_MAX_NVPRM) {
        (void)be_fail(BC_E600, "nvptx", 0u);
        return 1;
    }
    last = F->first_blk + F->num_blks - 1u;
    sast.f0 = M->mblks[F->first_blk].first_inst;
    sast.fn = M->mblks[last].first_inst + M->mblks[last].num_insts - sast.f0;
    if (saname() != 0) {
        (void)be_fail(BC_E600, "nvptx", 1u);
        return 1;
    }
    if (sast.fn >= 0xFFFFu || F->num_blks >= SA_NBB) {
        (void)be_fail(BC_E602, "nvptx", sast.name, (unsigned)F->num_blks,
                      (unsigned)SA_NBB);
        return 1;
    }
    if (sabase() != 0) {
        (void)be_fail(BC_E606, "nvptx", sast.name, (unsigned)SA_NV);
        return 1;
    }
    sapar();
    if (sasplit() != 0) {
        (void)be_fail(BC_E602, "nvptx", sast.name, (unsigned)F->num_blks,
                      (unsigned)SA_NBB);
        return 1;
    }
    if (sasucc() != 0) {
        (void)be_fail(BC_E605, "nvptx", sast.name, 0u, 0u);
        return 1;
    }
    return 0;
}

static int sarega(void)
{
    if (sascan() != 0) {
        (void)be_fail(BC_E606, "nvptx", sast.name, (unsigned)SA_NV);
        return 1;
    }
    sastr();
    if (saall() != 0) {
        if (sast.npr != 0u)
            (void)be_fail(BC_E604, "nvptx", sast.name, (unsigned)SA_NPR - 1u);
        else
            (void)be_fail(BC_E603, "nvptx", sast.name,
                          (unsigned)(LT_MAX_NVREG - SA_RBAS));
        return 1;
    }
    return 0;
}

static int sacomp(nv_module_t *nv)
{
    uint32_t fi = 0u;
    lt_res_t rc;

    if (sakern(nv, &fi) != 0) return 1;
    if (sabld(nv, fi) != 0) return 1;
    if (sarega() != 0) return 1;

    sawhy = 0u;
    rc = lt_nvcf(&sast.cf);
    if (rc != LT_OK) {
        (void)be_fail(BC_E605, "nvptx", sast.name, (unsigned)sawa,
                      (unsigned)sawb);
        return 1;
    }
    sarun();
    if (sast.nbad != 0u) return 1;
    sast.k.nreg = sast.nreg + 2u;
    sast.k.smem = sast.F->lds_bytes;
    sast.k.lmem = sast.F->lcl_bytes;
    if (sast.clen == 0u || sast.clen > LT_MAX_NVCODE) {
        (void)be_fail(BC_E607, "nvptx", sast.name, (unsigned)LT_MAX_NVCODE);
        return 1;
    }
    if (lt_nvcub(sast.img, LT_MAX_NVIMG, &sast.ilen, &sast.k,
                 sast.code, sast.clen) != LT_OK) {
        (void)be_fail(BC_E608, "nvptx", sast.name);
        return 1;
    }
    return 0;
}

static int sawrit(const char *path, const void *p, uint32_t n)
{
    FILE *fp = fopen(path, "wb");
    size_t got;

    if (fp == NULL) {
        fprintf(stderr, "error: cannot open '%s' for writing\n", path);
        return 1;
    }
    got = fwrite(p, 1u, n, fp);
    fclose(fp);
    return (got == (size_t)n) ? 0 : 1;
}

static uint32_t satxt(char *out, uint32_t max)
{
    uint32_t n = 0;
    int      k;

    k = snprintf(out, max, "// Booth SASS listing for %s, sm_89\n"
                 "// %u registers, %u bytes shared, %u bytes local\n\n",
                 sast.name, (unsigned)sast.k.nreg,
                 (unsigned)sast.k.smem, (unsigned)sast.k.lmem);
    if (k < 0 || (uint32_t)k >= max) return 0;
    n = (uint32_t)k;
    for (uint32_t o = 0; o + 16u <= sast.clen; o += 16u) {
        uint64_t w[2];
        uint16_t c = sast.src[o >> 4];

        memcpy(w, sast.code + o, 16u);
        k = snprintf(out + n, max - n, "%04x  %016llx %016llx  %s\n",
                     (unsigned)o, (unsigned long long)w[0],
                     (unsigned long long)w[1],
                     (c == 0xFFFEu) ? "(sass)" : saonm(c));
        if (k < 0 || (uint32_t)k >= max - n) return n;
        n += (uint32_t)k;
    }
    return n;
}

int nv_sass(nv_module_t *nv, const char *path, int text)
{
    if (sacomp(nv) != 0) return BC_ERR_NVIDIA;
    if (text == 0) {
        if (sawrit(path, sast.img, sast.ilen) != 0) return BC_ERR_IO;
        printf("wrote %s (%u bytes cubin, 1 kernel, %u instructions)\n",
               path, (unsigned)sast.ilen, (unsigned)(sast.clen / 16u));
        return BC_OK;
    }
    sast.ilen = satxt((char *)sast.img, LT_MAX_NVIMG);
    if (sawrit(path, sast.img, sast.ilen) != 0) return BC_ERR_IO;
    printf("wrote %s (%u bytes, 1 kernel, %u instructions)\n",
           path, (unsigned)sast.ilen, (unsigned)(sast.clen / 16u));
    return BC_OK;
}
