#include "tharns.h"
#include "amdgpu.h"
#include "encode.h"

enum {
    ASY_NONE = 0,
    ASY_S1, ASY_S1_64, ASY_S1_GET, ASY_S1_SET,
    ASY_S2, ASY_S2_64, ASY_SC,
    ASY_SPP0, ASY_SPPI, ASY_SPPB, ASY_SPPW,
    ASY_SM1, ASY_SM2, ASY_SM4,
    ASY_V1, ASY_V1_RFL, ASY_V1_D64, ASY_V1_S64,
    ASY_V2, ASY_V2_CND, ASY_V3A, ASY_V3B, ASY_VC,
    ASY_DSR, ASY_DSW, ASY_DSRTN, ASY_DSBPM,
    ASY_GLD, ASY_GLD2, ASY_GST, ASY_GST2, ASY_GATM, ASY_GCAS,
    ASY_SLD, ASY_SST, ASY_FLD, ASY_FST
};

static const uint8_t asy_sh[AMD_OP_COUNT] = {
    [AMD_S_ADD_U32] = ASY_S2, [AMD_S_ADDC_U32] = ASY_S2,
    [AMD_S_ADD_I32] = ASY_S2, [AMD_S_SUB_U32] = ASY_S2,
    [AMD_S_MUL_I32] = ASY_S2, [AMD_S_AND_B32] = ASY_S2,
    [AMD_S_OR_B32] = ASY_S2, [AMD_S_XOR_B32] = ASY_S2,
    [AMD_S_LSHL_B32] = ASY_S2, [AMD_S_LSHR_B32] = ASY_S2,
    [AMD_S_ASHR_I32] = ASY_S2, [AMD_S_ANDN2_B32] = ASY_S2,
    [AMD_S_ORN2_B32] = ASY_S2, [AMD_S_BFE_I32] = ASY_S2,
    [AMD_S_CSELECT_B32] = ASY_S2,
    [AMD_S_AND_B64] = ASY_S2_64, [AMD_S_OR_B64] = ASY_S2_64,
    [AMD_S_XOR_B64] = ASY_S2_64, [AMD_S_ANDN2_B64] = ASY_S2_64,
    [AMD_S_ORN2_B64] = ASY_S2_64,
    [AMD_S_MOV_B32] = ASY_S1, [AMD_S_NOT_B32] = ASY_S1,
    [AMD_S_AND_SAVEEXEC_B32] = ASY_S1,
    [AMD_S_MOV_B64] = ASY_S1_64, [AMD_S_NOT_B64] = ASY_S1_64,
    [AMD_S_AND_SAVEEXEC_B64] = ASY_S1_64, [AMD_S_SWAPPC_B64] = ASY_S1_64,
    [AMD_S_GETPC_B64] = ASY_S1_GET, [AMD_S_SETPC_B64] = ASY_S1_SET,
    [AMD_S_CMP_EQ_U32] = ASY_SC, [AMD_S_CMP_NE_U32] = ASY_SC,
    [AMD_S_CMP_LT_U32] = ASY_SC, [AMD_S_CMP_LE_U32] = ASY_SC,
    [AMD_S_CMP_GT_U32] = ASY_SC, [AMD_S_CMP_GE_U32] = ASY_SC,
    [AMD_S_CMP_LT_I32] = ASY_SC, [AMD_S_CMP_LE_I32] = ASY_SC,
    [AMD_S_CMP_GT_I32] = ASY_SC, [AMD_S_CMP_GE_I32] = ASY_SC,
    [AMD_S_CMP_EQ_I32] = ASY_SC, [AMD_S_CMP_NE_I32] = ASY_SC,
    [AMD_S_BRANCH] = ASY_SPPB, [AMD_S_CBRANCH_SCC0] = ASY_SPPB,
    [AMD_S_CBRANCH_SCC1] = ASY_SPPB, [AMD_S_CBRANCH_EXECZ] = ASY_SPPB,
    [AMD_S_CBRANCH_EXECNZ] = ASY_SPPB,
    [AMD_S_ENDPGM] = ASY_SPP0, [AMD_S_BARRIER] = ASY_SPP0,
    [AMD_S_TRAP] = ASY_SPPI, [AMD_S_NOP] = ASY_SPPI,
    [AMD_S_WAIT_LOADCNT] = ASY_SPPI, [AMD_S_WAIT_STORECNT] = ASY_SPPI,
    [AMD_S_WAIT_DSCNT] = ASY_SPPI, [AMD_S_WAIT_KMCNT] = ASY_SPPI,
    [AMD_S_WAITCNT] = ASY_SPPW,
    [AMD_S_LOAD_DWORD] = ASY_SM1, [AMD_S_LOAD_DWORDX2] = ASY_SM2,
    [AMD_S_LOAD_DWORDX4] = ASY_SM4,
    [AMD_V_ADD_U32] = ASY_V2, [AMD_V_SUB_U32] = ASY_V2,
    [AMD_V_AND_B32] = ASY_V2, [AMD_V_OR_B32] = ASY_V2,
    [AMD_V_XOR_B32] = ASY_V2, [AMD_V_LSHLREV_B32] = ASY_V2,
    [AMD_V_LSHRREV_B32] = ASY_V2, [AMD_V_ASHRREV_I32] = ASY_V2,
    [AMD_V_ADD_F32] = ASY_V2, [AMD_V_SUB_F32] = ASY_V2,
    [AMD_V_MUL_F32] = ASY_V2, [AMD_V_MIN_F32] = ASY_V2,
    [AMD_V_MAX_F32] = ASY_V2, [AMD_V_MIN_U32] = ASY_V2,
    [AMD_V_CNDMASK_B32] = ASY_V2_CND,
    [AMD_V_MUL_LO_U32] = ASY_V3A, [AMD_V_MUL_HI_U32] = ASY_V3A,
    [AMD_V_BCNT_U32_B32] = ASY_V3A,
    [AMD_V_MAD_U32_U24] = ASY_V3B, [AMD_V_BFE_I32] = ASY_V3B,
    [AMD_V_BFE_U32] = ASY_V3B, [AMD_V_ADD3_U32] = ASY_V3B,
    [AMD_V_LSHL_ADD_U32] = ASY_V3B,
    [AMD_V_MOV_B32] = ASY_V1, [AMD_V_CVT_F32_I32] = ASY_V1,
    [AMD_V_CVT_F32_U32] = ASY_V1, [AMD_V_CVT_I32_F32] = ASY_V1,
    [AMD_V_CVT_U32_F32] = ASY_V1, [AMD_V_CVT_F32_F16] = ASY_V1,
    [AMD_V_CVT_F16_F32] = ASY_V1, [AMD_V_CVT_F32_BF16] = ASY_V1,
    [AMD_V_CVT_BF16_F32] = ASY_V1, [AMD_V_RCP_F32] = ASY_V1,
    [AMD_V_SQRT_F32] = ASY_V1, [AMD_V_RSQ_F32] = ASY_V1,
    [AMD_V_EXP_F32] = ASY_V1, [AMD_V_LOG_F32] = ASY_V1,
    [AMD_V_SIN_F32] = ASY_V1, [AMD_V_COS_F32] = ASY_V1,
    [AMD_V_FLOOR_F32] = ASY_V1, [AMD_V_CEIL_F32] = ASY_V1,
    [AMD_V_TRUNC_F32] = ASY_V1, [AMD_V_RNDNE_F32] = ASY_V1,
    [AMD_V_FRACT_F32] = ASY_V1, [AMD_V_NOT_B32] = ASY_V1,
    [AMD_V_FFBL_B32] = ASY_V1, [AMD_V_FFBH_U32] = ASY_V1,
    [AMD_V_BFREV_B32] = ASY_V1,
    [AMD_V_READFIRSTLANE_B32] = ASY_V1_RFL,
    [AMD_V_CVT_F64_F32] = ASY_V1_D64, [AMD_V_CVT_F32_F64] = ASY_V1_S64,
    [AMD_V_CMP_EQ_U32] = ASY_VC, [AMD_V_CMP_NE_U32] = ASY_VC,
    [AMD_V_CMP_LT_U32] = ASY_VC, [AMD_V_CMP_LE_U32] = ASY_VC,
    [AMD_V_CMP_GT_U32] = ASY_VC, [AMD_V_CMP_GE_U32] = ASY_VC,
    [AMD_V_CMP_LT_I32] = ASY_VC, [AMD_V_CMP_LE_I32] = ASY_VC,
    [AMD_V_CMP_GT_I32] = ASY_VC, [AMD_V_CMP_GE_I32] = ASY_VC,
    [AMD_V_CMP_EQ_I32] = ASY_VC, [AMD_V_CMP_NE_I32] = ASY_VC,
    [AMD_V_CMP_EQ_F32] = ASY_VC, [AMD_V_CMP_NE_F32] = ASY_VC,
    [AMD_V_CMP_LT_F32] = ASY_VC, [AMD_V_CMP_LE_F32] = ASY_VC,
    [AMD_V_CMP_GT_F32] = ASY_VC, [AMD_V_CMP_GE_F32] = ASY_VC,
    [AMD_V_CMP_O_F32] = ASY_VC, [AMD_V_CMP_U_F32] = ASY_VC,
    [AMD_V_CMP_NLT_F32] = ASY_VC, [AMD_V_CMP_NLE_F32] = ASY_VC,
    [AMD_V_CMP_NGT_F32] = ASY_VC, [AMD_V_CMP_NGE_F32] = ASY_VC,
    [AMD_V_CMP_NEQ_F32] = ASY_VC,
    [AMD_DS_READ_B32] = ASY_DSR, [AMD_DS_SWIZZLE_B32] = ASY_DSR,
    [AMD_DS_WRITE_B32] = ASY_DSW,
    [AMD_DS_ADD_RTN_U32] = ASY_DSRTN, [AMD_DS_SUB_RTN_U32] = ASY_DSRTN,
    [AMD_DS_AND_RTN_B32] = ASY_DSRTN, [AMD_DS_OR_RTN_B32] = ASY_DSRTN,
    [AMD_DS_XOR_RTN_B32] = ASY_DSRTN, [AMD_DS_MIN_RTN_I32] = ASY_DSRTN,
    [AMD_DS_MAX_RTN_I32] = ASY_DSRTN,
    [AMD_DS_BPERMUTE_B32] = ASY_DSBPM,
    [AMD_GLOBAL_LOAD_DWORD] = ASY_GLD, [AMD_GLOBAL_LOAD_DWORDX2] = ASY_GLD2,
    [AMD_GLOBAL_STORE_DWORD] = ASY_GST, [AMD_GLOBAL_STORE_DWORDX2] = ASY_GST2,
    [AMD_GLOBAL_ATOMIC_ADD] = ASY_GATM, [AMD_GLOBAL_ATOMIC_SUB] = ASY_GATM,
    [AMD_GLOBAL_ATOMIC_AND] = ASY_GATM, [AMD_GLOBAL_ATOMIC_OR] = ASY_GATM,
    [AMD_GLOBAL_ATOMIC_XOR] = ASY_GATM, [AMD_GLOBAL_ATOMIC_SMIN] = ASY_GATM,
    [AMD_GLOBAL_ATOMIC_SMAX] = ASY_GATM, [AMD_GLOBAL_ATOMIC_SWAP] = ASY_GATM,
    [AMD_GLOBAL_ATOMIC_CMPSWAP] = ASY_GCAS,
    [AMD_SCRATCH_LOAD_DWORD] = ASY_SLD, [AMD_SCRATCH_STORE_DWORD] = ASY_SST,
    [AMD_FLAT_LOAD_DWORD] = ASY_FLD, [AMD_FLAT_STORE_DWORD] = ASY_FST
};

static moperand_t asy_sg(uint16_t r, uint8_t n)
{
    moperand_t o;
    memset(&o, 0, sizeof(o));
    o.kind = MOP_SGPR;
    o.reg_num = r;
    o.nreg = n;
    return o;
}

static moperand_t asy_vg(uint16_t r, uint8_t n)
{
    moperand_t o;
    memset(&o, 0, sizeof(o));
    o.kind = MOP_VGPR;
    o.reg_num = r;
    o.nreg = n;
    return o;
}

static moperand_t asy_im(int32_t v)
{
    moperand_t o;
    memset(&o, 0, sizeof(o));
    o.kind = MOP_IMM;
    o.imm = v;
    return o;
}

static int asy_bld(minst_t *mi, uint16_t op, uint8_t sh)
{
    memset(mi, 0, sizeof(*mi));
    mi->op = op;
    switch (sh) {
    case ASY_S1:
        mi->num_defs = 1; mi->num_uses = 1;
        mi->operands[0] = asy_sg(4, 1); mi->operands[1] = asy_sg(6, 1);
        return 1;
    case ASY_S1_64:
        mi->num_defs = 1; mi->num_uses = 1;
        mi->operands[0] = asy_sg(4, 2); mi->operands[1] = asy_sg(6, 2);
        return 1;
    case ASY_S1_GET:
        mi->num_defs = 1; mi->num_uses = 0;
        mi->operands[0] = asy_sg(4, 2);
        return 1;
    case ASY_S1_SET:
        mi->num_defs = 0; mi->num_uses = 1;
        mi->operands[0] = asy_sg(6, 2);
        return 1;
    case ASY_S2:
        mi->num_defs = 1; mi->num_uses = 2;
        mi->operands[0] = asy_sg(4, 1); mi->operands[1] = asy_sg(6, 1);
        mi->operands[2] = asy_sg(8, 1);
        return 1;
    case ASY_S2_64:
        mi->num_defs = 1; mi->num_uses = 2;
        mi->operands[0] = asy_sg(4, 2); mi->operands[1] = asy_sg(6, 2);
        mi->operands[2] = asy_sg(8, 2);
        return 1;
    case ASY_SC:
        mi->num_defs = 0; mi->num_uses = 2;
        mi->operands[0] = asy_sg(6, 1); mi->operands[1] = asy_sg(8, 1);
        return 1;
    case ASY_SPP0:
        return 1;
    case ASY_SPPI:
    case ASY_SPPB:
        mi->num_uses = 1; mi->operands[0] = asy_im(0);
        return 1;
    case ASY_SPPW:
        mi->num_uses = 0; mi->flags = 0;
        return 1;
    case ASY_SM1:
        mi->num_defs = 1; mi->num_uses = 2;
        mi->operands[0] = asy_sg(4, 1); mi->operands[1] = asy_sg(6, 2);
        mi->operands[2] = asy_im(0);
        return 1;
    case ASY_SM2:
        mi->num_defs = 1; mi->num_uses = 2;
        mi->operands[0] = asy_sg(4, 2); mi->operands[1] = asy_sg(6, 2);
        mi->operands[2] = asy_im(0);
        return 1;
    case ASY_SM4:
        mi->num_defs = 1; mi->num_uses = 2;
        mi->operands[0] = asy_sg(4, 4); mi->operands[1] = asy_sg(8, 2);
        mi->operands[2] = asy_im(0);
        return 1;
    case ASY_V1:
        mi->num_defs = 1; mi->num_uses = 1;
        mi->operands[0] = asy_vg(1, 1); mi->operands[1] = asy_vg(2, 1);
        return 1;
    case ASY_V1_RFL:
        mi->num_defs = 1; mi->num_uses = 1;
        mi->operands[0] = asy_sg(4, 1); mi->operands[1] = asy_vg(2, 1);
        return 1;
    case ASY_V1_D64:
        mi->num_defs = 1; mi->num_uses = 1;
        mi->operands[0] = asy_vg(4, 2); mi->operands[1] = asy_vg(2, 1);
        return 1;
    case ASY_V1_S64:
        mi->num_defs = 1; mi->num_uses = 1;
        mi->operands[0] = asy_vg(1, 1); mi->operands[1] = asy_vg(2, 2);
        return 1;
    case ASY_V2:
    case ASY_V2_CND:
    case ASY_V3A:
        mi->num_defs = 1; mi->num_uses = 2;
        mi->operands[0] = asy_vg(1, 1); mi->operands[1] = asy_vg(2, 1);
        mi->operands[2] = asy_vg(3, 1);
        return 1;
    case ASY_V3B:
        mi->num_defs = 1; mi->num_uses = 3;
        mi->operands[0] = asy_vg(1, 1); mi->operands[1] = asy_vg(2, 1);
        mi->operands[2] = asy_vg(3, 1); mi->operands[3] = asy_vg(4, 1);
        return 1;
    case ASY_VC:
        mi->num_defs = 0; mi->num_uses = 2;
        mi->operands[0] = asy_vg(2, 1); mi->operands[1] = asy_vg(3, 1);
        return 1;
    case ASY_DSR:
        mi->num_defs = 1; mi->num_uses = 2;
        mi->operands[0] = asy_vg(1, 1); mi->operands[1] = asy_vg(2, 1);
        mi->operands[2] = asy_im(0);
        return 1;
    case ASY_DSW:
        mi->num_defs = 0; mi->num_uses = 3;
        mi->operands[0] = asy_vg(2, 1); mi->operands[1] = asy_vg(3, 1);
        mi->operands[2] = asy_im(0);
        return 1;
    case ASY_DSRTN:
    case ASY_DSBPM:
        mi->num_defs = 1; mi->num_uses = 3;
        mi->operands[0] = asy_vg(1, 1); mi->operands[1] = asy_vg(2, 1);
        mi->operands[2] = asy_vg(3, 1); mi->operands[3] = asy_im(0);
        return 1;
    case ASY_GLD:
    case ASY_SLD:
    case ASY_FLD:
        mi->num_defs = 1; mi->num_uses = 2;
        mi->operands[0] = asy_vg(1, 1); mi->operands[1] = asy_vg(2, 2);
        mi->operands[2] = asy_im(0);
        return 1;
    case ASY_GLD2:
        mi->num_defs = 1; mi->num_uses = 2;
        mi->operands[0] = asy_vg(4, 2); mi->operands[1] = asy_vg(2, 2);
        mi->operands[2] = asy_im(0);
        return 1;
    case ASY_GST:
    case ASY_GST2:
    case ASY_GATM:
    case ASY_GCAS:
    case ASY_FST:
        mi->num_defs = 0; mi->num_uses = 3;
        mi->operands[0] = asy_vg(2, 2); mi->operands[1] = asy_vg(4, 1);
        mi->operands[2] = asy_im(0);
        return 1;
    case ASY_SST:
        mi->num_defs = 0; mi->num_uses = 3;
        mi->operands[0] = asy_vg(2, 1); mi->operands[1] = asy_vg(3, 1);
        mi->operands[2] = asy_im(0);
        return 1;
    default:
        return 0;
    }
}

static int asy_txt(char *buf, int n, uint8_t sh, const char *mn, int w64)
{
    const char *vcc = w64 ? "vcc" : "vcc_lo";
    switch (sh) {
    case ASY_S1:     return snprintf(buf, n, "%s s4, s6", mn);
    case ASY_S1_64:  return snprintf(buf, n, "%s s[4:5], s[6:7]", mn);
    case ASY_S1_GET: return snprintf(buf, n, "%s s[4:5]", mn);
    case ASY_S1_SET: return snprintf(buf, n, "%s s[6:7]", mn);
    case ASY_S2:     return snprintf(buf, n, "%s s4, s6, s8", mn);
    case ASY_S2_64:  return snprintf(buf, n, "%s s[4:5], s[6:7], s[8:9]", mn);
    case ASY_SC:     return snprintf(buf, n, "%s s6, s8", mn);
    case ASY_SPP0:   return snprintf(buf, n, "%s", mn);
    case ASY_SPPI:   return snprintf(buf, n, "%s 0", mn);
    case ASY_SPPB:   return snprintf(buf, n, "%s 0", mn);
    case ASY_SPPW:   return snprintf(buf, n, "%s", mn);
    case ASY_SM1:    return snprintf(buf, n, "%s s4, s[6:7], 0x0", mn);
    case ASY_SM2:    return snprintf(buf, n, "%s s[4:5], s[6:7], 0x0", mn);
    case ASY_SM4:    return snprintf(buf, n, "%s s[4:7], s[8:9], 0x0", mn);
    case ASY_V1:     return snprintf(buf, n, "%s_e32 v1, v2", mn);
    case ASY_V1_RFL: return snprintf(buf, n, "%s_e32 s4, v2", mn);
    case ASY_V1_D64: return snprintf(buf, n, "%s_e32 v[4:5], v2", mn);
    case ASY_V1_S64: return snprintf(buf, n, "%s_e32 v1, v[2:3]", mn);
    case ASY_V2:     return snprintf(buf, n, "%s_e32 v1, v2, v3", mn);
    case ASY_V2_CND: return snprintf(buf, n, "%s_e32 v1, v2, v3, %s", mn, vcc);
    case ASY_V3A:    return snprintf(buf, n, "%s_e64 v1, v2, v3", mn);
    case ASY_V3B:    return snprintf(buf, n, "%s_e64 v1, v2, v3, v4", mn);
    case ASY_VC:     return snprintf(buf, n, "%s_e32 %s, v2, v3", mn, vcc);
    case ASY_DSR:    return snprintf(buf, n, "%s v1, v2", mn);
    case ASY_DSW:    return snprintf(buf, n, "%s v2, v3", mn);
    case ASY_DSRTN:  return snprintf(buf, n, "%s v1, v2, v3", mn);
    case ASY_DSBPM:  return snprintf(buf, n, "%s v1, v2, v3", mn);
    case ASY_GLD:    return snprintf(buf, n, "%s v1, v[2:3], off", mn);
    case ASY_GLD2:   return snprintf(buf, n, "%s v[4:5], v[2:3], off", mn);
    case ASY_GST:    return snprintf(buf, n, "%s v[2:3], v4, off", mn);
    case ASY_GST2:   return snprintf(buf, n, "%s v[2:3], v[4:5], off", mn);
    case ASY_GATM:   return snprintf(buf, n, "%s v[2:3], v4, off", mn);
    case ASY_GCAS:   return snprintf(buf, n, "%s v[2:3], v[4:5], off", mn);
    case ASY_SLD:    return snprintf(buf, n, "%s v1, v2, off", mn);
    case ASY_SST:    return snprintf(buf, n, "%s v2, v3, off", mn);
    case ASY_FLD:    return snprintf(buf, n, "%s v1, v[2:3]", mn);
    case ASY_FST:    return snprintf(buf, n, "%s v[2:3], v4", mn);
    default:         return 0;
    }
}

static const amd_target_t asy_tgt[] = {
    AMD_TARGET_GFX90A, AMD_TARGET_GFX942, AMD_TARGET_GFX1030,
    AMD_TARGET_GFX1100, AMD_TARGET_GFX1200
};

#define ASY_NTGT 5

static amd_module_t *asy_mod;

static int asy_enc(amd_target_t tgt, uint16_t op, uint8_t sh, uint8_t *out)
{
    if (!asy_mod) asy_mod = (amd_module_t *)malloc(sizeof(amd_module_t));
    if (!asy_mod) return 0;
    memset(asy_mod, 0, sizeof(*asy_mod));
    asy_mod->target = tgt;
    asy_mod->mfuncs[0].first_block = 0;
    asy_mod->mfuncs[0].num_blocks = 1;
    asy_mod->num_mfuncs = 1;
    asy_mod->mblocks[0].first_inst = 0;
    asy_mod->mblocks[0].num_insts = 1;
    asy_mod->num_mblocks = 1;
    asy_mod->num_minsts = 1;
    if (!asy_bld(&asy_mod->minsts[0], op, sh)) return 0;
    encode_function(asy_mod, 0);
    if (asy_mod->enc_err) return 0;
    if (asy_mod->code_len > 12) return 0;
    memcpy(out, asy_mod->code, asy_mod->code_len);
    return (int)asy_mod->code_len;
}

typedef struct {
    uint16_t    tgt;
    uint16_t    op;
    const char *hex;
} asy_exp_t;

static const asy_exp_t asy_exp[] = {
    { AMD_TARGET_GFX90A,     AMD_S_ADD_U32,             "06080480" },
    { AMD_TARGET_GFX90A,     AMD_S_ADDC_U32,            "06080482" },
    { AMD_TARGET_GFX90A,     AMD_S_ADD_I32,             "06080481" },
    { AMD_TARGET_GFX90A,     AMD_S_SUB_U32,             "06088480" },
    { AMD_TARGET_GFX90A,     AMD_S_MUL_I32,             "06080492" },
    { AMD_TARGET_GFX90A,     AMD_S_AND_B32,             "06080486" },
    { AMD_TARGET_GFX90A,     AMD_S_OR_B32,              "06080487" },
    { AMD_TARGET_GFX90A,     AMD_S_XOR_B32,             "06080488" },
    { AMD_TARGET_GFX90A,     AMD_S_LSHL_B32,            "0608048e" },
    { AMD_TARGET_GFX90A,     AMD_S_LSHR_B32,            "0608048f" },
    { AMD_TARGET_GFX90A,     AMD_S_ASHR_I32,            "06080490" },
    { AMD_TARGET_GFX90A,     AMD_S_ANDN2_B32,           "06080489" },
    { AMD_TARGET_GFX90A,     AMD_S_ORN2_B32,            "0608048a" },
    { AMD_TARGET_GFX90A,     AMD_S_BFE_I32,             "06080493" },
    { AMD_TARGET_GFX90A,     AMD_S_CSELECT_B32,         "06080485" },
    { AMD_TARGET_GFX90A,     AMD_S_AND_B64,             "06088486" },
    { AMD_TARGET_GFX90A,     AMD_S_OR_B64,              "06088487" },
    { AMD_TARGET_GFX90A,     AMD_S_XOR_B64,             "06088488" },
    { AMD_TARGET_GFX90A,     AMD_S_ANDN2_B64,           "06088489" },
    { AMD_TARGET_GFX90A,     AMD_S_ORN2_B64,            "0608848a" },
    { AMD_TARGET_GFX90A,     AMD_S_MOV_B32,             "060084be" },
    { AMD_TARGET_GFX90A,     AMD_S_NOT_B32,             "060484be" },
    { AMD_TARGET_GFX90A,     AMD_S_MOV_B64,             "060184be" },
    { AMD_TARGET_GFX90A,     AMD_S_NOT_B64,             "060584be" },
    { AMD_TARGET_GFX90A,     AMD_S_AND_SAVEEXEC_B64,    "062084be" },
    { AMD_TARGET_GFX90A,     AMD_S_SETPC_B64,           "061d80be" },
    { AMD_TARGET_GFX90A,     AMD_S_SWAPPC_B64,          "061e84be" },
    { AMD_TARGET_GFX90A,     AMD_S_GETPC_B64,           "001c84be" },
    { AMD_TARGET_GFX90A,     AMD_S_CMP_EQ_U32,          "060806bf" },
    { AMD_TARGET_GFX90A,     AMD_S_CMP_NE_U32,          "060807bf" },
    { AMD_TARGET_GFX90A,     AMD_S_CMP_LT_U32,          "06080abf" },
    { AMD_TARGET_GFX90A,     AMD_S_CMP_LE_U32,          "06080bbf" },
    { AMD_TARGET_GFX90A,     AMD_S_CMP_GT_U32,          "060808bf" },
    { AMD_TARGET_GFX90A,     AMD_S_CMP_GE_U32,          "060809bf" },
    { AMD_TARGET_GFX90A,     AMD_S_CMP_LT_I32,          "060804bf" },
    { AMD_TARGET_GFX90A,     AMD_S_CMP_LE_I32,          "060805bf" },
    { AMD_TARGET_GFX90A,     AMD_S_CMP_GT_I32,          "060802bf" },
    { AMD_TARGET_GFX90A,     AMD_S_CMP_GE_I32,          "060803bf" },
    { AMD_TARGET_GFX90A,     AMD_S_CMP_EQ_I32,          "060800bf" },
    { AMD_TARGET_GFX90A,     AMD_S_CMP_NE_I32,          "060801bf" },
    { AMD_TARGET_GFX90A,     AMD_S_BRANCH,              "000082bf" },
    { AMD_TARGET_GFX90A,     AMD_S_CBRANCH_SCC0,        "000084bf" },
    { AMD_TARGET_GFX90A,     AMD_S_CBRANCH_SCC1,        "000085bf" },
    { AMD_TARGET_GFX90A,     AMD_S_CBRANCH_EXECZ,       "000088bf" },
    { AMD_TARGET_GFX90A,     AMD_S_CBRANCH_EXECNZ,      "000089bf" },
    { AMD_TARGET_GFX90A,     AMD_S_ENDPGM,              "000081bf" },
    { AMD_TARGET_GFX90A,     AMD_S_BARRIER,             "00008abf" },
    { AMD_TARGET_GFX90A,     AMD_S_TRAP,                "000092bf" },
    { AMD_TARGET_GFX90A,     AMD_S_WAITCNT,             "7fcf8cbf" },
    { AMD_TARGET_GFX90A,     AMD_S_NOP,                 "000080bf" },
    { AMD_TARGET_GFX90A,     AMD_S_LOAD_DWORD,          "030102c000000000" },
    { AMD_TARGET_GFX90A,     AMD_S_LOAD_DWORDX2,        "030106c000000000" },
    { AMD_TARGET_GFX90A,     AMD_S_LOAD_DWORDX4,        "04010ac000000000" },
    { AMD_TARGET_GFX90A,     AMD_V_ADD_U32,             "02070268" },
    { AMD_TARGET_GFX90A,     AMD_V_SUB_U32,             "0207026a" },
    { AMD_TARGET_GFX90A,     AMD_V_MUL_LO_U32,          "010085d202070200" },
    { AMD_TARGET_GFX90A,     AMD_V_MUL_HI_U32,          "010086d202070200" },
    { AMD_TARGET_GFX90A,     AMD_V_AND_B32,             "02070226" },
    { AMD_TARGET_GFX90A,     AMD_V_OR_B32,              "02070228" },
    { AMD_TARGET_GFX90A,     AMD_V_XOR_B32,             "0207022a" },
    { AMD_TARGET_GFX90A,     AMD_V_LSHLREV_B32,         "02070224" },
    { AMD_TARGET_GFX90A,     AMD_V_LSHRREV_B32,         "02070220" },
    { AMD_TARGET_GFX90A,     AMD_V_ASHRREV_I32,         "02070222" },
    { AMD_TARGET_GFX90A,     AMD_V_ADD_F32,             "02070202" },
    { AMD_TARGET_GFX90A,     AMD_V_SUB_F32,             "02070204" },
    { AMD_TARGET_GFX90A,     AMD_V_MUL_F32,             "0207020a" },
    { AMD_TARGET_GFX90A,     AMD_V_CNDMASK_B32,         "02070200" },
    { AMD_TARGET_GFX90A,     AMD_V_MIN_F32,             "02070214" },
    { AMD_TARGET_GFX90A,     AMD_V_MAX_F32,             "02070216" },
    { AMD_TARGET_GFX90A,     AMD_V_MIN_U32,             "0207021c" },
    { AMD_TARGET_GFX90A,     AMD_V_MOV_B32,             "0203027e" },
    { AMD_TARGET_GFX90A,     AMD_V_CVT_F32_I32,         "020b027e" },
    { AMD_TARGET_GFX90A,     AMD_V_CVT_F32_U32,         "020d027e" },
    { AMD_TARGET_GFX90A,     AMD_V_CVT_I32_F32,         "0211027e" },
    { AMD_TARGET_GFX90A,     AMD_V_CVT_U32_F32,         "020f027e" },
    { AMD_TARGET_GFX90A,     AMD_V_CVT_F32_F16,         "0217027e" },
    { AMD_TARGET_GFX90A,     AMD_V_CVT_F16_F32,         "0215027e" },
    { AMD_TARGET_GFX90A,     AMD_V_CVT_F64_F32,         "0221087e" },
    { AMD_TARGET_GFX90A,     AMD_V_CVT_F32_F64,         "021f027e" },
    { AMD_TARGET_GFX90A,     AMD_V_RCP_F32,             "0245027e" },
    { AMD_TARGET_GFX90A,     AMD_V_SQRT_F32,            "024f027e" },
    { AMD_TARGET_GFX90A,     AMD_V_RSQ_F32,             "0249027e" },
    { AMD_TARGET_GFX90A,     AMD_V_EXP_F32,             "0241027e" },
    { AMD_TARGET_GFX90A,     AMD_V_LOG_F32,             "0243027e" },
    { AMD_TARGET_GFX90A,     AMD_V_SIN_F32,             "0253027e" },
    { AMD_TARGET_GFX90A,     AMD_V_COS_F32,             "0255027e" },
    { AMD_TARGET_GFX90A,     AMD_V_FLOOR_F32,           "023f027e" },
    { AMD_TARGET_GFX90A,     AMD_V_CEIL_F32,            "023b027e" },
    { AMD_TARGET_GFX90A,     AMD_V_TRUNC_F32,           "0239027e" },
    { AMD_TARGET_GFX90A,     AMD_V_RNDNE_F32,           "023d027e" },
    { AMD_TARGET_GFX90A,     AMD_V_FRACT_F32,           "0237027e" },
    { AMD_TARGET_GFX90A,     AMD_V_NOT_B32,             "0257027e" },
    { AMD_TARGET_GFX90A,     AMD_V_READFIRSTLANE_B32,   "0205087e" },
    { AMD_TARGET_GFX90A,     AMD_V_FFBL_B32,            "025d027e" },
    { AMD_TARGET_GFX90A,     AMD_V_FFBH_U32,            "025b027e" },
    { AMD_TARGET_GFX90A,     AMD_V_BFREV_B32,           "0259027e" },
    { AMD_TARGET_GFX90A,     AMD_V_BCNT_U32_B32,        "01008bd202070200" },
    { AMD_TARGET_GFX90A,     AMD_V_MAD_U32_U24,         "0100c3d102071204" },
    { AMD_TARGET_GFX90A,     AMD_V_BFE_I32,             "0100c9d102071204" },
    { AMD_TARGET_GFX90A,     AMD_V_BFE_U32,             "0100c8d102071204" },
    { AMD_TARGET_GFX90A,     AMD_V_LSHL_ADD_U32,        "0100fdd102071204" },
    { AMD_TARGET_GFX90A,     AMD_V_ADD3_U32,            "0100ffd102071204" },
    { AMD_TARGET_GFX90A,     AMD_V_CMP_EQ_U32,          "0207947d" },
    { AMD_TARGET_GFX90A,     AMD_V_CMP_NE_U32,          "02079a7d" },
    { AMD_TARGET_GFX90A,     AMD_V_CMP_LT_U32,          "0207927d" },
    { AMD_TARGET_GFX90A,     AMD_V_CMP_LE_U32,          "0207967d" },
    { AMD_TARGET_GFX90A,     AMD_V_CMP_GT_U32,          "0207987d" },
    { AMD_TARGET_GFX90A,     AMD_V_CMP_GE_U32,          "02079c7d" },
    { AMD_TARGET_GFX90A,     AMD_V_CMP_LT_I32,          "0207827d" },
    { AMD_TARGET_GFX90A,     AMD_V_CMP_LE_I32,          "0207867d" },
    { AMD_TARGET_GFX90A,     AMD_V_CMP_GT_I32,          "0207887d" },
    { AMD_TARGET_GFX90A,     AMD_V_CMP_GE_I32,          "02078c7d" },
    { AMD_TARGET_GFX90A,     AMD_V_CMP_EQ_I32,          "0207847d" },
    { AMD_TARGET_GFX90A,     AMD_V_CMP_NE_I32,          "02078a7d" },
    { AMD_TARGET_GFX90A,     AMD_V_CMP_EQ_F32,          "0207847c" },
    { AMD_TARGET_GFX90A,     AMD_V_CMP_NE_F32,          "02079a7c" },
    { AMD_TARGET_GFX90A,     AMD_V_CMP_LT_F32,          "0207827c" },
    { AMD_TARGET_GFX90A,     AMD_V_CMP_LE_F32,          "0207867c" },
    { AMD_TARGET_GFX90A,     AMD_V_CMP_GT_F32,          "0207887c" },
    { AMD_TARGET_GFX90A,     AMD_V_CMP_GE_F32,          "02078c7c" },
    { AMD_TARGET_GFX90A,     AMD_V_CMP_O_F32,           "02078e7c" },
    { AMD_TARGET_GFX90A,     AMD_V_CMP_U_F32,           "0207907c" },
    { AMD_TARGET_GFX90A,     AMD_V_CMP_NLT_F32,         "02079c7c" },
    { AMD_TARGET_GFX90A,     AMD_V_CMP_NLE_F32,         "0207987c" },
    { AMD_TARGET_GFX90A,     AMD_V_CMP_NGT_F32,         "0207967c" },
    { AMD_TARGET_GFX90A,     AMD_V_CMP_NGE_F32,         "0207927c" },
    { AMD_TARGET_GFX90A,     AMD_V_CMP_NEQ_F32,         "02079a7c" },
    { AMD_TARGET_GFX90A,     AMD_DS_READ_B32,           "00006cd802000001" },
    { AMD_TARGET_GFX90A,     AMD_DS_WRITE_B32,          "00001ad802030000" },
    { AMD_TARGET_GFX90A,     AMD_DS_ADD_RTN_U32,        "000040d802030001" },
    { AMD_TARGET_GFX90A,     AMD_DS_SUB_RTN_U32,        "000042d802030001" },
    { AMD_TARGET_GFX90A,     AMD_DS_AND_RTN_B32,        "000052d802030001" },
    { AMD_TARGET_GFX90A,     AMD_DS_OR_RTN_B32,         "000054d802030001" },
    { AMD_TARGET_GFX90A,     AMD_DS_XOR_RTN_B32,        "000056d802030001" },
    { AMD_TARGET_GFX90A,     AMD_DS_MIN_RTN_I32,        "00004ad802030001" },
    { AMD_TARGET_GFX90A,     AMD_DS_MAX_RTN_I32,        "00004cd802030001" },
    { AMD_TARGET_GFX90A,     AMD_DS_SWIZZLE_B32,        "00007ad802000001" },
    { AMD_TARGET_GFX90A,     AMD_DS_BPERMUTE_B32,       "00007ed802030001" },
    { AMD_TARGET_GFX90A,     AMD_GLOBAL_LOAD_DWORD,     "008050dc02007f01" },
    { AMD_TARGET_GFX90A,     AMD_GLOBAL_STORE_DWORD,    "008070dc02047f00" },
    { AMD_TARGET_GFX90A,     AMD_GLOBAL_LOAD_DWORDX2,   "008054dc02007f04" },
    { AMD_TARGET_GFX90A,     AMD_GLOBAL_STORE_DWORDX2,  "008074dc02047f00" },
    { AMD_TARGET_GFX90A,     AMD_GLOBAL_ATOMIC_ADD,     "008008dd02047f00" },
    { AMD_TARGET_GFX90A,     AMD_GLOBAL_ATOMIC_SUB,     "00800cdd02047f00" },
    { AMD_TARGET_GFX90A,     AMD_GLOBAL_ATOMIC_AND,     "008020dd02047f00" },
    { AMD_TARGET_GFX90A,     AMD_GLOBAL_ATOMIC_OR,      "008024dd02047f00" },
    { AMD_TARGET_GFX90A,     AMD_GLOBAL_ATOMIC_XOR,     "008028dd02047f00" },
    { AMD_TARGET_GFX90A,     AMD_GLOBAL_ATOMIC_SMIN,    "008010dd02047f00" },
    { AMD_TARGET_GFX90A,     AMD_GLOBAL_ATOMIC_SMAX,    "008018dd02047f00" },
    { AMD_TARGET_GFX90A,     AMD_GLOBAL_ATOMIC_SWAP,    "008000dd02047f00" },
    { AMD_TARGET_GFX90A,     AMD_GLOBAL_ATOMIC_CMPSWAP, "008004dd02047f00" },
    { AMD_TARGET_GFX90A,     AMD_SCRATCH_LOAD_DWORD,    "004050dc02007f01" },
    { AMD_TARGET_GFX90A,     AMD_SCRATCH_STORE_DWORD,   "004070dc02037f00" },
    { AMD_TARGET_GFX90A,     AMD_FLAT_LOAD_DWORD,       "000050dc02000001" },
    { AMD_TARGET_GFX90A,     AMD_FLAT_STORE_DWORD,      "000070dc02040000" },
    { AMD_TARGET_GFX942,     AMD_S_ADD_U32,             "06080480" },
    { AMD_TARGET_GFX942,     AMD_S_ADDC_U32,            "06080482" },
    { AMD_TARGET_GFX942,     AMD_S_ADD_I32,             "06080481" },
    { AMD_TARGET_GFX942,     AMD_S_SUB_U32,             "06088480" },
    { AMD_TARGET_GFX942,     AMD_S_MUL_I32,             "06080492" },
    { AMD_TARGET_GFX942,     AMD_S_AND_B32,             "06080486" },
    { AMD_TARGET_GFX942,     AMD_S_OR_B32,              "06080487" },
    { AMD_TARGET_GFX942,     AMD_S_XOR_B32,             "06080488" },
    { AMD_TARGET_GFX942,     AMD_S_LSHL_B32,            "0608048e" },
    { AMD_TARGET_GFX942,     AMD_S_LSHR_B32,            "0608048f" },
    { AMD_TARGET_GFX942,     AMD_S_ASHR_I32,            "06080490" },
    { AMD_TARGET_GFX942,     AMD_S_ANDN2_B32,           "06080489" },
    { AMD_TARGET_GFX942,     AMD_S_ORN2_B32,            "0608048a" },
    { AMD_TARGET_GFX942,     AMD_S_BFE_I32,             "06080493" },
    { AMD_TARGET_GFX942,     AMD_S_CSELECT_B32,         "06080485" },
    { AMD_TARGET_GFX942,     AMD_S_AND_B64,             "06088486" },
    { AMD_TARGET_GFX942,     AMD_S_OR_B64,              "06088487" },
    { AMD_TARGET_GFX942,     AMD_S_XOR_B64,             "06088488" },
    { AMD_TARGET_GFX942,     AMD_S_ANDN2_B64,           "06088489" },
    { AMD_TARGET_GFX942,     AMD_S_ORN2_B64,            "0608848a" },
    { AMD_TARGET_GFX942,     AMD_S_MOV_B32,             "060084be" },
    { AMD_TARGET_GFX942,     AMD_S_NOT_B32,             "060484be" },
    { AMD_TARGET_GFX942,     AMD_S_MOV_B64,             "060184be" },
    { AMD_TARGET_GFX942,     AMD_S_NOT_B64,             "060584be" },
    { AMD_TARGET_GFX942,     AMD_S_AND_SAVEEXEC_B64,    "062084be" },
    { AMD_TARGET_GFX942,     AMD_S_SETPC_B64,           "061d80be" },
    { AMD_TARGET_GFX942,     AMD_S_SWAPPC_B64,          "061e84be" },
    { AMD_TARGET_GFX942,     AMD_S_GETPC_B64,           "001c84be" },
    { AMD_TARGET_GFX942,     AMD_S_CMP_EQ_U32,          "060806bf" },
    { AMD_TARGET_GFX942,     AMD_S_CMP_NE_U32,          "060807bf" },
    { AMD_TARGET_GFX942,     AMD_S_CMP_LT_U32,          "06080abf" },
    { AMD_TARGET_GFX942,     AMD_S_CMP_LE_U32,          "06080bbf" },
    { AMD_TARGET_GFX942,     AMD_S_CMP_GT_U32,          "060808bf" },
    { AMD_TARGET_GFX942,     AMD_S_CMP_GE_U32,          "060809bf" },
    { AMD_TARGET_GFX942,     AMD_S_CMP_LT_I32,          "060804bf" },
    { AMD_TARGET_GFX942,     AMD_S_CMP_LE_I32,          "060805bf" },
    { AMD_TARGET_GFX942,     AMD_S_CMP_GT_I32,          "060802bf" },
    { AMD_TARGET_GFX942,     AMD_S_CMP_GE_I32,          "060803bf" },
    { AMD_TARGET_GFX942,     AMD_S_CMP_EQ_I32,          "060800bf" },
    { AMD_TARGET_GFX942,     AMD_S_CMP_NE_I32,          "060801bf" },
    { AMD_TARGET_GFX942,     AMD_S_BRANCH,              "000082bf" },
    { AMD_TARGET_GFX942,     AMD_S_CBRANCH_SCC0,        "000084bf" },
    { AMD_TARGET_GFX942,     AMD_S_CBRANCH_SCC1,        "000085bf" },
    { AMD_TARGET_GFX942,     AMD_S_CBRANCH_EXECZ,       "000088bf" },
    { AMD_TARGET_GFX942,     AMD_S_CBRANCH_EXECNZ,      "000089bf" },
    { AMD_TARGET_GFX942,     AMD_S_ENDPGM,              "000081bf" },
    { AMD_TARGET_GFX942,     AMD_S_BARRIER,             "00008abf" },
    { AMD_TARGET_GFX942,     AMD_S_TRAP,                "000092bf" },
    { AMD_TARGET_GFX942,     AMD_S_WAITCNT,             "7fcf8cbf" },
    { AMD_TARGET_GFX942,     AMD_S_NOP,                 "000080bf" },
    { AMD_TARGET_GFX942,     AMD_S_LOAD_DWORD,          "030102c000000000" },
    { AMD_TARGET_GFX942,     AMD_S_LOAD_DWORDX2,        "030106c000000000" },
    { AMD_TARGET_GFX942,     AMD_S_LOAD_DWORDX4,        "04010ac000000000" },
    { AMD_TARGET_GFX942,     AMD_V_ADD_U32,             "02070268" },
    { AMD_TARGET_GFX942,     AMD_V_SUB_U32,             "0207026a" },
    { AMD_TARGET_GFX942,     AMD_V_MUL_LO_U32,          "010085d202070200" },
    { AMD_TARGET_GFX942,     AMD_V_MUL_HI_U32,          "010086d202070200" },
    { AMD_TARGET_GFX942,     AMD_V_AND_B32,             "02070226" },
    { AMD_TARGET_GFX942,     AMD_V_OR_B32,              "02070228" },
    { AMD_TARGET_GFX942,     AMD_V_XOR_B32,             "0207022a" },
    { AMD_TARGET_GFX942,     AMD_V_LSHLREV_B32,         "02070224" },
    { AMD_TARGET_GFX942,     AMD_V_LSHRREV_B32,         "02070220" },
    { AMD_TARGET_GFX942,     AMD_V_ASHRREV_I32,         "02070222" },
    { AMD_TARGET_GFX942,     AMD_V_ADD_F32,             "02070202" },
    { AMD_TARGET_GFX942,     AMD_V_SUB_F32,             "02070204" },
    { AMD_TARGET_GFX942,     AMD_V_MUL_F32,             "0207020a" },
    { AMD_TARGET_GFX942,     AMD_V_CNDMASK_B32,         "02070200" },
    { AMD_TARGET_GFX942,     AMD_V_MIN_F32,             "02070214" },
    { AMD_TARGET_GFX942,     AMD_V_MAX_F32,             "02070216" },
    { AMD_TARGET_GFX942,     AMD_V_MIN_U32,             "0207021c" },
    { AMD_TARGET_GFX942,     AMD_V_MOV_B32,             "0203027e" },
    { AMD_TARGET_GFX942,     AMD_V_CVT_F32_I32,         "020b027e" },
    { AMD_TARGET_GFX942,     AMD_V_CVT_F32_U32,         "020d027e" },
    { AMD_TARGET_GFX942,     AMD_V_CVT_I32_F32,         "0211027e" },
    { AMD_TARGET_GFX942,     AMD_V_CVT_U32_F32,         "020f027e" },
    { AMD_TARGET_GFX942,     AMD_V_CVT_F32_F16,         "0217027e" },
    { AMD_TARGET_GFX942,     AMD_V_CVT_F16_F32,         "0215027e" },
    { AMD_TARGET_GFX942,     AMD_V_CVT_F64_F32,         "0221087e" },
    { AMD_TARGET_GFX942,     AMD_V_CVT_F32_F64,         "021f027e" },
    { AMD_TARGET_GFX942,     AMD_V_RCP_F32,             "0245027e" },
    { AMD_TARGET_GFX942,     AMD_V_SQRT_F32,            "024f027e" },
    { AMD_TARGET_GFX942,     AMD_V_RSQ_F32,             "0249027e" },
    { AMD_TARGET_GFX942,     AMD_V_EXP_F32,             "0241027e" },
    { AMD_TARGET_GFX942,     AMD_V_LOG_F32,             "0243027e" },
    { AMD_TARGET_GFX942,     AMD_V_SIN_F32,             "0253027e" },
    { AMD_TARGET_GFX942,     AMD_V_COS_F32,             "0255027e" },
    { AMD_TARGET_GFX942,     AMD_V_FLOOR_F32,           "023f027e" },
    { AMD_TARGET_GFX942,     AMD_V_CEIL_F32,            "023b027e" },
    { AMD_TARGET_GFX942,     AMD_V_TRUNC_F32,           "0239027e" },
    { AMD_TARGET_GFX942,     AMD_V_RNDNE_F32,           "023d027e" },
    { AMD_TARGET_GFX942,     AMD_V_FRACT_F32,           "0237027e" },
    { AMD_TARGET_GFX942,     AMD_V_NOT_B32,             "0257027e" },
    { AMD_TARGET_GFX942,     AMD_V_READFIRSTLANE_B32,   "0205087e" },
    { AMD_TARGET_GFX942,     AMD_V_FFBL_B32,            "025d027e" },
    { AMD_TARGET_GFX942,     AMD_V_FFBH_U32,            "025b027e" },
    { AMD_TARGET_GFX942,     AMD_V_BFREV_B32,           "0259027e" },
    { AMD_TARGET_GFX942,     AMD_V_BCNT_U32_B32,        "01008bd202070200" },
    { AMD_TARGET_GFX942,     AMD_V_MAD_U32_U24,         "0100c3d102071204" },
    { AMD_TARGET_GFX942,     AMD_V_BFE_I32,             "0100c9d102071204" },
    { AMD_TARGET_GFX942,     AMD_V_BFE_U32,             "0100c8d102071204" },
    { AMD_TARGET_GFX942,     AMD_V_LSHL_ADD_U32,        "0100fdd102071204" },
    { AMD_TARGET_GFX942,     AMD_V_ADD3_U32,            "0100ffd102071204" },
    { AMD_TARGET_GFX942,     AMD_V_CMP_EQ_U32,          "0207947d" },
    { AMD_TARGET_GFX942,     AMD_V_CMP_NE_U32,          "02079a7d" },
    { AMD_TARGET_GFX942,     AMD_V_CMP_LT_U32,          "0207927d" },
    { AMD_TARGET_GFX942,     AMD_V_CMP_LE_U32,          "0207967d" },
    { AMD_TARGET_GFX942,     AMD_V_CMP_GT_U32,          "0207987d" },
    { AMD_TARGET_GFX942,     AMD_V_CMP_GE_U32,          "02079c7d" },
    { AMD_TARGET_GFX942,     AMD_V_CMP_LT_I32,          "0207827d" },
    { AMD_TARGET_GFX942,     AMD_V_CMP_LE_I32,          "0207867d" },
    { AMD_TARGET_GFX942,     AMD_V_CMP_GT_I32,          "0207887d" },
    { AMD_TARGET_GFX942,     AMD_V_CMP_GE_I32,          "02078c7d" },
    { AMD_TARGET_GFX942,     AMD_V_CMP_EQ_I32,          "0207847d" },
    { AMD_TARGET_GFX942,     AMD_V_CMP_NE_I32,          "02078a7d" },
    { AMD_TARGET_GFX942,     AMD_V_CMP_EQ_F32,          "0207847c" },
    { AMD_TARGET_GFX942,     AMD_V_CMP_NE_F32,          "02079a7c" },
    { AMD_TARGET_GFX942,     AMD_V_CMP_LT_F32,          "0207827c" },
    { AMD_TARGET_GFX942,     AMD_V_CMP_LE_F32,          "0207867c" },
    { AMD_TARGET_GFX942,     AMD_V_CMP_GT_F32,          "0207887c" },
    { AMD_TARGET_GFX942,     AMD_V_CMP_GE_F32,          "02078c7c" },
    { AMD_TARGET_GFX942,     AMD_V_CMP_O_F32,           "02078e7c" },
    { AMD_TARGET_GFX942,     AMD_V_CMP_U_F32,           "0207907c" },
    { AMD_TARGET_GFX942,     AMD_V_CMP_NLT_F32,         "02079c7c" },
    { AMD_TARGET_GFX942,     AMD_V_CMP_NLE_F32,         "0207987c" },
    { AMD_TARGET_GFX942,     AMD_V_CMP_NGT_F32,         "0207967c" },
    { AMD_TARGET_GFX942,     AMD_V_CMP_NGE_F32,         "0207927c" },
    { AMD_TARGET_GFX942,     AMD_V_CMP_NEQ_F32,         "02079a7c" },
    { AMD_TARGET_GFX942,     AMD_DS_READ_B32,           "00006cd802000001" },
    { AMD_TARGET_GFX942,     AMD_DS_WRITE_B32,          "00001ad802030000" },
    { AMD_TARGET_GFX942,     AMD_DS_ADD_RTN_U32,        "000040d802030001" },
    { AMD_TARGET_GFX942,     AMD_DS_SUB_RTN_U32,        "000042d802030001" },
    { AMD_TARGET_GFX942,     AMD_DS_AND_RTN_B32,        "000052d802030001" },
    { AMD_TARGET_GFX942,     AMD_DS_OR_RTN_B32,         "000054d802030001" },
    { AMD_TARGET_GFX942,     AMD_DS_XOR_RTN_B32,        "000056d802030001" },
    { AMD_TARGET_GFX942,     AMD_DS_MIN_RTN_I32,        "00004ad802030001" },
    { AMD_TARGET_GFX942,     AMD_DS_MAX_RTN_I32,        "00004cd802030001" },
    { AMD_TARGET_GFX942,     AMD_DS_SWIZZLE_B32,        "00007ad802000001" },
    { AMD_TARGET_GFX942,     AMD_DS_BPERMUTE_B32,       "00007ed802030001" },
    { AMD_TARGET_GFX942,     AMD_GLOBAL_LOAD_DWORD,     "008050dc02007f01" },
    { AMD_TARGET_GFX942,     AMD_GLOBAL_STORE_DWORD,    "008070dc02047f00" },
    { AMD_TARGET_GFX942,     AMD_GLOBAL_LOAD_DWORDX2,   "008054dc02007f04" },
    { AMD_TARGET_GFX942,     AMD_GLOBAL_STORE_DWORDX2,  "008074dc02047f00" },
    { AMD_TARGET_GFX942,     AMD_GLOBAL_ATOMIC_ADD,     "008008dd02047f00" },
    { AMD_TARGET_GFX942,     AMD_GLOBAL_ATOMIC_SUB,     "00800cdd02047f00" },
    { AMD_TARGET_GFX942,     AMD_GLOBAL_ATOMIC_AND,     "008020dd02047f00" },
    { AMD_TARGET_GFX942,     AMD_GLOBAL_ATOMIC_OR,      "008024dd02047f00" },
    { AMD_TARGET_GFX942,     AMD_GLOBAL_ATOMIC_XOR,     "008028dd02047f00" },
    { AMD_TARGET_GFX942,     AMD_GLOBAL_ATOMIC_SMIN,    "008010dd02047f00" },
    { AMD_TARGET_GFX942,     AMD_GLOBAL_ATOMIC_SMAX,    "008018dd02047f00" },
    { AMD_TARGET_GFX942,     AMD_GLOBAL_ATOMIC_SWAP,    "008000dd02047f00" },
    { AMD_TARGET_GFX942,     AMD_GLOBAL_ATOMIC_CMPSWAP, "008004dd02047f00" },
    { AMD_TARGET_GFX942,     AMD_SCRATCH_LOAD_DWORD,    "006050dc02007f01" },
    { AMD_TARGET_GFX942,     AMD_SCRATCH_STORE_DWORD,   "006070dc02037f00" },
    { AMD_TARGET_GFX942,     AMD_FLAT_LOAD_DWORD,       "000050dc02000001" },
    { AMD_TARGET_GFX942,     AMD_FLAT_STORE_DWORD,      "000070dc02040000" },
    { AMD_TARGET_GFX1030,    AMD_S_ADD_U32,             "06080480" },
    { AMD_TARGET_GFX1030,    AMD_S_ADDC_U32,            "06080482" },
    { AMD_TARGET_GFX1030,    AMD_S_ADD_I32,             "06080481" },
    { AMD_TARGET_GFX1030,    AMD_S_SUB_U32,             "06088480" },
    { AMD_TARGET_GFX1030,    AMD_S_MUL_I32,             "06080493" },
    { AMD_TARGET_GFX1030,    AMD_S_AND_B32,             "06080487" },
    { AMD_TARGET_GFX1030,    AMD_S_OR_B32,              "06080488" },
    { AMD_TARGET_GFX1030,    AMD_S_XOR_B32,             "06080489" },
    { AMD_TARGET_GFX1030,    AMD_S_LSHL_B32,            "0608048f" },
    { AMD_TARGET_GFX1030,    AMD_S_LSHR_B32,            "06080490" },
    { AMD_TARGET_GFX1030,    AMD_S_ASHR_I32,            "06080491" },
    { AMD_TARGET_GFX1030,    AMD_S_ANDN2_B32,           "0608048a" },
    { AMD_TARGET_GFX1030,    AMD_S_ORN2_B32,            "0608048b" },
    { AMD_TARGET_GFX1030,    AMD_S_BFE_I32,             "06080494" },
    { AMD_TARGET_GFX1030,    AMD_S_CSELECT_B32,         "06080485" },
    { AMD_TARGET_GFX1030,    AMD_S_MOV_B32,             "060384be" },
    { AMD_TARGET_GFX1030,    AMD_S_NOT_B32,             "060784be" },
    { AMD_TARGET_GFX1030,    AMD_S_AND_SAVEEXEC_B32,    "063c84be" },
    { AMD_TARGET_GFX1030,    AMD_S_SETPC_B64,           "062080be" },
    { AMD_TARGET_GFX1030,    AMD_S_SWAPPC_B64,          "062184be" },
    { AMD_TARGET_GFX1030,    AMD_S_GETPC_B64,           "001f84be" },
    { AMD_TARGET_GFX1030,    AMD_S_CMP_EQ_U32,          "060806bf" },
    { AMD_TARGET_GFX1030,    AMD_S_CMP_NE_U32,          "060807bf" },
    { AMD_TARGET_GFX1030,    AMD_S_CMP_LT_U32,          "06080abf" },
    { AMD_TARGET_GFX1030,    AMD_S_CMP_LE_U32,          "06080bbf" },
    { AMD_TARGET_GFX1030,    AMD_S_CMP_GT_U32,          "060808bf" },
    { AMD_TARGET_GFX1030,    AMD_S_CMP_GE_U32,          "060809bf" },
    { AMD_TARGET_GFX1030,    AMD_S_CMP_LT_I32,          "060804bf" },
    { AMD_TARGET_GFX1030,    AMD_S_CMP_LE_I32,          "060805bf" },
    { AMD_TARGET_GFX1030,    AMD_S_CMP_GT_I32,          "060802bf" },
    { AMD_TARGET_GFX1030,    AMD_S_CMP_GE_I32,          "060803bf" },
    { AMD_TARGET_GFX1030,    AMD_S_CMP_EQ_I32,          "060800bf" },
    { AMD_TARGET_GFX1030,    AMD_S_CMP_NE_I32,          "060801bf" },
    { AMD_TARGET_GFX1030,    AMD_S_BRANCH,              "000082bf" },
    { AMD_TARGET_GFX1030,    AMD_S_CBRANCH_SCC0,        "000084bf" },
    { AMD_TARGET_GFX1030,    AMD_S_CBRANCH_SCC1,        "000085bf" },
    { AMD_TARGET_GFX1030,    AMD_S_CBRANCH_EXECZ,       "000088bf" },
    { AMD_TARGET_GFX1030,    AMD_S_CBRANCH_EXECNZ,      "000089bf" },
    { AMD_TARGET_GFX1030,    AMD_S_ENDPGM,              "000081bf" },
    { AMD_TARGET_GFX1030,    AMD_S_BARRIER,             "00008abf" },
    { AMD_TARGET_GFX1030,    AMD_S_TRAP,                "000092bf" },
    { AMD_TARGET_GFX1030,    AMD_S_WAITCNT,             "7fff8cbf" },
    { AMD_TARGET_GFX1030,    AMD_S_NOP,                 "000080bf" },
    { AMD_TARGET_GFX1030,    AMD_S_LOAD_DWORD,          "030100f4000000fa" },
    { AMD_TARGET_GFX1030,    AMD_S_LOAD_DWORDX2,        "030104f4000000fa" },
    { AMD_TARGET_GFX1030,    AMD_S_LOAD_DWORDX4,        "040108f4000000fa" },
    { AMD_TARGET_GFX1030,    AMD_V_ADD_U32,             "0207024a" },
    { AMD_TARGET_GFX1030,    AMD_V_SUB_U32,             "0207024c" },
    { AMD_TARGET_GFX1030,    AMD_V_MUL_LO_U32,          "010069d502070200" },
    { AMD_TARGET_GFX1030,    AMD_V_MUL_HI_U32,          "01006ad502070200" },
    { AMD_TARGET_GFX1030,    AMD_V_AND_B32,             "02070236" },
    { AMD_TARGET_GFX1030,    AMD_V_OR_B32,              "02070238" },
    { AMD_TARGET_GFX1030,    AMD_V_XOR_B32,             "0207023a" },
    { AMD_TARGET_GFX1030,    AMD_V_LSHLREV_B32,         "02070234" },
    { AMD_TARGET_GFX1030,    AMD_V_LSHRREV_B32,         "0207022c" },
    { AMD_TARGET_GFX1030,    AMD_V_ASHRREV_I32,         "02070230" },
    { AMD_TARGET_GFX1030,    AMD_V_ADD_F32,             "02070206" },
    { AMD_TARGET_GFX1030,    AMD_V_SUB_F32,             "02070208" },
    { AMD_TARGET_GFX1030,    AMD_V_MUL_F32,             "02070210" },
    { AMD_TARGET_GFX1030,    AMD_V_CNDMASK_B32,         "02070202" },
    { AMD_TARGET_GFX1030,    AMD_V_MIN_F32,             "0207021e" },
    { AMD_TARGET_GFX1030,    AMD_V_MAX_F32,             "02070220" },
    { AMD_TARGET_GFX1030,    AMD_V_MIN_U32,             "02070226" },
    { AMD_TARGET_GFX1030,    AMD_V_MOV_B32,             "0203027e" },
    { AMD_TARGET_GFX1030,    AMD_V_CVT_F32_I32,         "020b027e" },
    { AMD_TARGET_GFX1030,    AMD_V_CVT_F32_U32,         "020d027e" },
    { AMD_TARGET_GFX1030,    AMD_V_CVT_I32_F32,         "0211027e" },
    { AMD_TARGET_GFX1030,    AMD_V_CVT_U32_F32,         "020f027e" },
    { AMD_TARGET_GFX1030,    AMD_V_CVT_F32_F16,         "0217027e" },
    { AMD_TARGET_GFX1030,    AMD_V_CVT_F16_F32,         "0215027e" },
    { AMD_TARGET_GFX1030,    AMD_V_CVT_F64_F32,         "0221087e" },
    { AMD_TARGET_GFX1030,    AMD_V_CVT_F32_F64,         "021f027e" },
    { AMD_TARGET_GFX1030,    AMD_V_RCP_F32,             "0255027e" },
    { AMD_TARGET_GFX1030,    AMD_V_SQRT_F32,            "0267027e" },
    { AMD_TARGET_GFX1030,    AMD_V_RSQ_F32,             "025d027e" },
    { AMD_TARGET_GFX1030,    AMD_V_EXP_F32,             "024b027e" },
    { AMD_TARGET_GFX1030,    AMD_V_LOG_F32,             "024f027e" },
    { AMD_TARGET_GFX1030,    AMD_V_SIN_F32,             "026b027e" },
    { AMD_TARGET_GFX1030,    AMD_V_COS_F32,             "026d027e" },
    { AMD_TARGET_GFX1030,    AMD_V_FLOOR_F32,           "0249027e" },
    { AMD_TARGET_GFX1030,    AMD_V_CEIL_F32,            "0245027e" },
    { AMD_TARGET_GFX1030,    AMD_V_TRUNC_F32,           "0243027e" },
    { AMD_TARGET_GFX1030,    AMD_V_RNDNE_F32,           "0247027e" },
    { AMD_TARGET_GFX1030,    AMD_V_FRACT_F32,           "0241027e" },
    { AMD_TARGET_GFX1030,    AMD_V_NOT_B32,             "026f027e" },
    { AMD_TARGET_GFX1030,    AMD_V_READFIRSTLANE_B32,   "0205087e" },
    { AMD_TARGET_GFX1030,    AMD_V_FFBL_B32,            "0275027e" },
    { AMD_TARGET_GFX1030,    AMD_V_FFBH_U32,            "0273027e" },
    { AMD_TARGET_GFX1030,    AMD_V_BFREV_B32,           "0271027e" },
    { AMD_TARGET_GFX1030,    AMD_V_BCNT_U32_B32,        "010064d702070200" },
    { AMD_TARGET_GFX1030,    AMD_V_MAD_U32_U24,         "010043d502071204" },
    { AMD_TARGET_GFX1030,    AMD_V_BFE_I32,             "010049d502071204" },
    { AMD_TARGET_GFX1030,    AMD_V_BFE_U32,             "010048d502071204" },
    { AMD_TARGET_GFX1030,    AMD_V_LSHL_ADD_U32,        "010046d702071204" },
    { AMD_TARGET_GFX1030,    AMD_V_ADD3_U32,            "01006dd702071204" },
    { AMD_TARGET_GFX1030,    AMD_V_CMP_EQ_U32,          "0207847d" },
    { AMD_TARGET_GFX1030,    AMD_V_CMP_NE_U32,          "02078a7d" },
    { AMD_TARGET_GFX1030,    AMD_V_CMP_LT_U32,          "0207827d" },
    { AMD_TARGET_GFX1030,    AMD_V_CMP_LE_U32,          "0207867d" },
    { AMD_TARGET_GFX1030,    AMD_V_CMP_GT_U32,          "0207887d" },
    { AMD_TARGET_GFX1030,    AMD_V_CMP_GE_U32,          "02078c7d" },
    { AMD_TARGET_GFX1030,    AMD_V_CMP_LT_I32,          "0207027d" },
    { AMD_TARGET_GFX1030,    AMD_V_CMP_LE_I32,          "0207067d" },
    { AMD_TARGET_GFX1030,    AMD_V_CMP_GT_I32,          "0207087d" },
    { AMD_TARGET_GFX1030,    AMD_V_CMP_GE_I32,          "02070c7d" },
    { AMD_TARGET_GFX1030,    AMD_V_CMP_EQ_I32,          "0207047d" },
    { AMD_TARGET_GFX1030,    AMD_V_CMP_NE_I32,          "02070a7d" },
    { AMD_TARGET_GFX1030,    AMD_V_CMP_EQ_F32,          "0207047c" },
    { AMD_TARGET_GFX1030,    AMD_V_CMP_NE_F32,          "02071a7c" },
    { AMD_TARGET_GFX1030,    AMD_V_CMP_LT_F32,          "0207027c" },
    { AMD_TARGET_GFX1030,    AMD_V_CMP_LE_F32,          "0207067c" },
    { AMD_TARGET_GFX1030,    AMD_V_CMP_GT_F32,          "0207087c" },
    { AMD_TARGET_GFX1030,    AMD_V_CMP_GE_F32,          "02070c7c" },
    { AMD_TARGET_GFX1030,    AMD_V_CMP_O_F32,           "02070e7c" },
    { AMD_TARGET_GFX1030,    AMD_V_CMP_U_F32,           "0207107c" },
    { AMD_TARGET_GFX1030,    AMD_V_CMP_NLT_F32,         "02071c7c" },
    { AMD_TARGET_GFX1030,    AMD_V_CMP_NLE_F32,         "0207187c" },
    { AMD_TARGET_GFX1030,    AMD_V_CMP_NGT_F32,         "0207167c" },
    { AMD_TARGET_GFX1030,    AMD_V_CMP_NGE_F32,         "0207127c" },
    { AMD_TARGET_GFX1030,    AMD_V_CMP_NEQ_F32,         "02071a7c" },
    { AMD_TARGET_GFX1030,    AMD_DS_READ_B32,           "0000d8d802000001" },
    { AMD_TARGET_GFX1030,    AMD_DS_WRITE_B32,          "000034d802030000" },
    { AMD_TARGET_GFX1030,    AMD_DS_ADD_RTN_U32,        "000080d802030001" },
    { AMD_TARGET_GFX1030,    AMD_DS_SUB_RTN_U32,        "000084d802030001" },
    { AMD_TARGET_GFX1030,    AMD_DS_AND_RTN_B32,        "0000a4d802030001" },
    { AMD_TARGET_GFX1030,    AMD_DS_OR_RTN_B32,         "0000a8d802030001" },
    { AMD_TARGET_GFX1030,    AMD_DS_XOR_RTN_B32,        "0000acd802030001" },
    { AMD_TARGET_GFX1030,    AMD_DS_MIN_RTN_I32,        "000094d802030001" },
    { AMD_TARGET_GFX1030,    AMD_DS_MAX_RTN_I32,        "000098d802030001" },
    { AMD_TARGET_GFX1030,    AMD_DS_SWIZZLE_B32,        "0000d4d802000001" },
    { AMD_TARGET_GFX1030,    AMD_DS_BPERMUTE_B32,       "0000ccda02030001" },
    { AMD_TARGET_GFX1030,    AMD_GLOBAL_LOAD_DWORD,     "008030dc02007d01" },
    { AMD_TARGET_GFX1030,    AMD_GLOBAL_STORE_DWORD,    "008070dc02047d00" },
    { AMD_TARGET_GFX1030,    AMD_GLOBAL_LOAD_DWORDX2,   "008034dc02007d04" },
    { AMD_TARGET_GFX1030,    AMD_GLOBAL_STORE_DWORDX2,  "008074dc02047d00" },
    { AMD_TARGET_GFX1030,    AMD_GLOBAL_ATOMIC_ADD,     "0080c8dc02047d00" },
    { AMD_TARGET_GFX1030,    AMD_GLOBAL_ATOMIC_SUB,     "0080ccdc02047d00" },
    { AMD_TARGET_GFX1030,    AMD_GLOBAL_ATOMIC_AND,     "0080e4dc02047d00" },
    { AMD_TARGET_GFX1030,    AMD_GLOBAL_ATOMIC_OR,      "0080e8dc02047d00" },
    { AMD_TARGET_GFX1030,    AMD_GLOBAL_ATOMIC_XOR,     "0080ecdc02047d00" },
    { AMD_TARGET_GFX1030,    AMD_GLOBAL_ATOMIC_SMIN,    "0080d4dc02047d00" },
    { AMD_TARGET_GFX1030,    AMD_GLOBAL_ATOMIC_SMAX,    "0080dcdc02047d00" },
    { AMD_TARGET_GFX1030,    AMD_GLOBAL_ATOMIC_SWAP,    "0080c0dc02047d00" },
    { AMD_TARGET_GFX1030,    AMD_GLOBAL_ATOMIC_CMPSWAP, "0080c4dc02047d00" },
    { AMD_TARGET_GFX1030,    AMD_SCRATCH_LOAD_DWORD,    "004030dc02007d01" },
    { AMD_TARGET_GFX1030,    AMD_SCRATCH_STORE_DWORD,   "004070dc02037d00" },
    { AMD_TARGET_GFX1100,    AMD_S_ADD_U32,             "06080480" },
    { AMD_TARGET_GFX1100,    AMD_S_ADDC_U32,            "06080482" },
    { AMD_TARGET_GFX1100,    AMD_S_ADD_I32,             "06080481" },
    { AMD_TARGET_GFX1100,    AMD_S_SUB_U32,             "06088480" },
    { AMD_TARGET_GFX1100,    AMD_S_MUL_I32,             "06080496" },
    { AMD_TARGET_GFX1100,    AMD_S_AND_B32,             "0608048b" },
    { AMD_TARGET_GFX1100,    AMD_S_OR_B32,              "0608048c" },
    { AMD_TARGET_GFX1100,    AMD_S_XOR_B32,             "0608048d" },
    { AMD_TARGET_GFX1100,    AMD_S_LSHL_B32,            "06080484" },
    { AMD_TARGET_GFX1100,    AMD_S_LSHR_B32,            "06080485" },
    { AMD_TARGET_GFX1100,    AMD_S_ASHR_I32,            "06080486" },
    { AMD_TARGET_GFX1100,    AMD_S_ANDN2_B32,           "06080491" },
    { AMD_TARGET_GFX1100,    AMD_S_ORN2_B32,            "06080492" },
    { AMD_TARGET_GFX1100,    AMD_S_BFE_I32,             "06088493" },
    { AMD_TARGET_GFX1100,    AMD_S_CSELECT_B32,         "06080498" },
    { AMD_TARGET_GFX1100,    AMD_S_MOV_B32,             "060084be" },
    { AMD_TARGET_GFX1100,    AMD_S_NOT_B32,             "061e84be" },
    { AMD_TARGET_GFX1100,    AMD_S_AND_SAVEEXEC_B32,    "062084be" },
    { AMD_TARGET_GFX1100,    AMD_S_SETPC_B64,           "064880be" },
    { AMD_TARGET_GFX1100,    AMD_S_SWAPPC_B64,          "064984be" },
    { AMD_TARGET_GFX1100,    AMD_S_GETPC_B64,           "004784be" },
    { AMD_TARGET_GFX1100,    AMD_S_CMP_EQ_U32,          "060806bf" },
    { AMD_TARGET_GFX1100,    AMD_S_CMP_NE_U32,          "060807bf" },
    { AMD_TARGET_GFX1100,    AMD_S_CMP_LT_U32,          "06080abf" },
    { AMD_TARGET_GFX1100,    AMD_S_CMP_LE_U32,          "06080bbf" },
    { AMD_TARGET_GFX1100,    AMD_S_CMP_GT_U32,          "060808bf" },
    { AMD_TARGET_GFX1100,    AMD_S_CMP_GE_U32,          "060809bf" },
    { AMD_TARGET_GFX1100,    AMD_S_CMP_LT_I32,          "060804bf" },
    { AMD_TARGET_GFX1100,    AMD_S_CMP_LE_I32,          "060805bf" },
    { AMD_TARGET_GFX1100,    AMD_S_CMP_GT_I32,          "060802bf" },
    { AMD_TARGET_GFX1100,    AMD_S_CMP_GE_I32,          "060803bf" },
    { AMD_TARGET_GFX1100,    AMD_S_CMP_EQ_I32,          "060800bf" },
    { AMD_TARGET_GFX1100,    AMD_S_CMP_NE_I32,          "060801bf" },
    { AMD_TARGET_GFX1100,    AMD_S_BRANCH,              "0000a0bf" },
    { AMD_TARGET_GFX1100,    AMD_S_CBRANCH_SCC0,        "0000a1bf" },
    { AMD_TARGET_GFX1100,    AMD_S_CBRANCH_SCC1,        "0000a2bf" },
    { AMD_TARGET_GFX1100,    AMD_S_CBRANCH_EXECZ,       "0000a5bf" },
    { AMD_TARGET_GFX1100,    AMD_S_CBRANCH_EXECNZ,      "0000a6bf" },
    { AMD_TARGET_GFX1100,    AMD_S_ENDPGM,              "0000b0bf" },
    { AMD_TARGET_GFX1100,    AMD_S_BARRIER,             "0000bdbf" },
    { AMD_TARGET_GFX1100,    AMD_S_TRAP,                "000090bf" },
    { AMD_TARGET_GFX1100,    AMD_S_WAITCNT,             "f7ff89bf" },
    { AMD_TARGET_GFX1100,    AMD_S_NOP,                 "000080bf" },
    { AMD_TARGET_GFX1100,    AMD_S_WAIT_LOADCNT,        "" },
    { AMD_TARGET_GFX1100,    AMD_S_WAIT_STORECNT,       "" },
    { AMD_TARGET_GFX1100,    AMD_S_WAIT_DSCNT,          "" },
    { AMD_TARGET_GFX1100,    AMD_S_WAIT_KMCNT,          "" },
    { AMD_TARGET_GFX1100,    AMD_S_LOAD_DWORD,          "030100f4000000f8" },
    { AMD_TARGET_GFX1100,    AMD_S_LOAD_DWORDX2,        "030104f4000000f8" },
    { AMD_TARGET_GFX1100,    AMD_S_LOAD_DWORDX4,        "040108f4000000f8" },
    { AMD_TARGET_GFX1100,    AMD_V_ADD_U32,             "0207024a" },
    { AMD_TARGET_GFX1100,    AMD_V_SUB_U32,             "0207024c" },
    { AMD_TARGET_GFX1100,    AMD_V_MUL_LO_U32,          "01002cd702070200" },
    { AMD_TARGET_GFX1100,    AMD_V_MUL_HI_U32,          "01002dd702070200" },
    { AMD_TARGET_GFX1100,    AMD_V_AND_B32,             "02070236" },
    { AMD_TARGET_GFX1100,    AMD_V_OR_B32,              "02070238" },
    { AMD_TARGET_GFX1100,    AMD_V_XOR_B32,             "0207023a" },
    { AMD_TARGET_GFX1100,    AMD_V_LSHLREV_B32,         "02070230" },
    { AMD_TARGET_GFX1100,    AMD_V_LSHRREV_B32,         "02070232" },
    { AMD_TARGET_GFX1100,    AMD_V_ASHRREV_I32,         "02070234" },
    { AMD_TARGET_GFX1100,    AMD_V_ADD_F32,             "02070206" },
    { AMD_TARGET_GFX1100,    AMD_V_SUB_F32,             "02070208" },
    { AMD_TARGET_GFX1100,    AMD_V_MUL_F32,             "02070210" },
    { AMD_TARGET_GFX1100,    AMD_V_CNDMASK_B32,         "02070202" },
    { AMD_TARGET_GFX1100,    AMD_V_MIN_F32,             "0207021e" },
    { AMD_TARGET_GFX1100,    AMD_V_MAX_F32,             "02070220" },
    { AMD_TARGET_GFX1100,    AMD_V_MIN_U32,             "02070226" },
    { AMD_TARGET_GFX1100,    AMD_V_MOV_B32,             "0203027e" },
    { AMD_TARGET_GFX1100,    AMD_V_CVT_F32_I32,         "020b027e" },
    { AMD_TARGET_GFX1100,    AMD_V_CVT_F32_U32,         "020d027e" },
    { AMD_TARGET_GFX1100,    AMD_V_CVT_I32_F32,         "0211027e" },
    { AMD_TARGET_GFX1100,    AMD_V_CVT_U32_F32,         "020f027e" },
    { AMD_TARGET_GFX1100,    AMD_V_CVT_F32_F16,         "0217027e" },
    { AMD_TARGET_GFX1100,    AMD_V_CVT_F16_F32,         "0215027e" },
    { AMD_TARGET_GFX1100,    AMD_V_CVT_F64_F32,         "0221087e" },
    { AMD_TARGET_GFX1100,    AMD_V_CVT_F32_F64,         "021f027e" },
    { AMD_TARGET_GFX1100,    AMD_V_RCP_F32,             "0255027e" },
    { AMD_TARGET_GFX1100,    AMD_V_SQRT_F32,            "0267027e" },
    { AMD_TARGET_GFX1100,    AMD_V_RSQ_F32,             "025d027e" },
    { AMD_TARGET_GFX1100,    AMD_V_EXP_F32,             "024b027e" },
    { AMD_TARGET_GFX1100,    AMD_V_LOG_F32,             "024f027e" },
    { AMD_TARGET_GFX1100,    AMD_V_SIN_F32,             "026b027e" },
    { AMD_TARGET_GFX1100,    AMD_V_COS_F32,             "026d027e" },
    { AMD_TARGET_GFX1100,    AMD_V_FLOOR_F32,           "0249027e" },
    { AMD_TARGET_GFX1100,    AMD_V_CEIL_F32,            "0245027e" },
    { AMD_TARGET_GFX1100,    AMD_V_TRUNC_F32,           "0243027e" },
    { AMD_TARGET_GFX1100,    AMD_V_RNDNE_F32,           "0247027e" },
    { AMD_TARGET_GFX1100,    AMD_V_FRACT_F32,           "0241027e" },
    { AMD_TARGET_GFX1100,    AMD_V_NOT_B32,             "026f027e" },
    { AMD_TARGET_GFX1100,    AMD_V_READFIRSTLANE_B32,   "0205087e" },
    { AMD_TARGET_GFX1100,    AMD_V_FFBL_B32,            "0275027e" },
    { AMD_TARGET_GFX1100,    AMD_V_FFBH_U32,            "0273027e" },
    { AMD_TARGET_GFX1100,    AMD_V_BFREV_B32,           "0271027e" },
    { AMD_TARGET_GFX1100,    AMD_V_BCNT_U32_B32,        "01001ed702070200" },
    { AMD_TARGET_GFX1100,    AMD_V_MAD_U32_U24,         "01000bd602071204" },
    { AMD_TARGET_GFX1100,    AMD_V_BFE_I32,             "010011d602071204" },
    { AMD_TARGET_GFX1100,    AMD_V_BFE_U32,             "010010d602071204" },
    { AMD_TARGET_GFX1100,    AMD_V_LSHL_ADD_U32,        "010046d602071204" },
    { AMD_TARGET_GFX1100,    AMD_V_ADD3_U32,            "010055d602071204" },
    { AMD_TARGET_GFX1100,    AMD_V_CMP_EQ_U32,          "0207947c" },
    { AMD_TARGET_GFX1100,    AMD_V_CMP_NE_U32,          "02079a7c" },
    { AMD_TARGET_GFX1100,    AMD_V_CMP_LT_U32,          "0207927c" },
    { AMD_TARGET_GFX1100,    AMD_V_CMP_LE_U32,          "0207967c" },
    { AMD_TARGET_GFX1100,    AMD_V_CMP_GT_U32,          "0207987c" },
    { AMD_TARGET_GFX1100,    AMD_V_CMP_GE_U32,          "02079c7c" },
    { AMD_TARGET_GFX1100,    AMD_V_CMP_LT_I32,          "0207827c" },
    { AMD_TARGET_GFX1100,    AMD_V_CMP_LE_I32,          "0207867c" },
    { AMD_TARGET_GFX1100,    AMD_V_CMP_GT_I32,          "0207887c" },
    { AMD_TARGET_GFX1100,    AMD_V_CMP_GE_I32,          "02078c7c" },
    { AMD_TARGET_GFX1100,    AMD_V_CMP_EQ_I32,          "0207847c" },
    { AMD_TARGET_GFX1100,    AMD_V_CMP_NE_I32,          "02078a7c" },
    { AMD_TARGET_GFX1100,    AMD_V_CMP_EQ_F32,          "0207247c" },
    { AMD_TARGET_GFX1100,    AMD_V_CMP_NE_F32,          "02073a7c" },
    { AMD_TARGET_GFX1100,    AMD_V_CMP_LT_F32,          "0207227c" },
    { AMD_TARGET_GFX1100,    AMD_V_CMP_LE_F32,          "0207267c" },
    { AMD_TARGET_GFX1100,    AMD_V_CMP_GT_F32,          "0207287c" },
    { AMD_TARGET_GFX1100,    AMD_V_CMP_GE_F32,          "02072c7c" },
    { AMD_TARGET_GFX1100,    AMD_V_CMP_O_F32,           "02072e7c" },
    { AMD_TARGET_GFX1100,    AMD_V_CMP_U_F32,           "0207307c" },
    { AMD_TARGET_GFX1100,    AMD_V_CMP_NLT_F32,         "02073c7c" },
    { AMD_TARGET_GFX1100,    AMD_V_CMP_NLE_F32,         "0207387c" },
    { AMD_TARGET_GFX1100,    AMD_V_CMP_NGT_F32,         "0207367c" },
    { AMD_TARGET_GFX1100,    AMD_V_CMP_NGE_F32,         "0207327c" },
    { AMD_TARGET_GFX1100,    AMD_V_CMP_NEQ_F32,         "02073a7c" },
    { AMD_TARGET_GFX1100,    AMD_DS_READ_B32,           "0000d8d802000001" },
    { AMD_TARGET_GFX1100,    AMD_DS_WRITE_B32,          "000034d802030000" },
    { AMD_TARGET_GFX1100,    AMD_DS_ADD_RTN_U32,        "000080d802030001" },
    { AMD_TARGET_GFX1100,    AMD_DS_SUB_RTN_U32,        "000084d802030001" },
    { AMD_TARGET_GFX1100,    AMD_DS_AND_RTN_B32,        "0000a4d802030001" },
    { AMD_TARGET_GFX1100,    AMD_DS_OR_RTN_B32,         "0000a8d802030001" },
    { AMD_TARGET_GFX1100,    AMD_DS_XOR_RTN_B32,        "0000acd802030001" },
    { AMD_TARGET_GFX1100,    AMD_DS_MIN_RTN_I32,        "000094d802030001" },
    { AMD_TARGET_GFX1100,    AMD_DS_MAX_RTN_I32,        "000098d802030001" },
    { AMD_TARGET_GFX1100,    AMD_DS_SWIZZLE_B32,        "0000d4d802000001" },
    { AMD_TARGET_GFX1100,    AMD_DS_BPERMUTE_B32,       "0000ccda02030001" },
    { AMD_TARGET_GFX1100,    AMD_GLOBAL_LOAD_DWORD,     "000052dc02007c01" },
    { AMD_TARGET_GFX1100,    AMD_GLOBAL_STORE_DWORD,    "00006adc02047c00" },
    { AMD_TARGET_GFX1100,    AMD_GLOBAL_LOAD_DWORDX2,   "000056dc02007c04" },
    { AMD_TARGET_GFX1100,    AMD_GLOBAL_STORE_DWORDX2,  "00006edc02047c00" },
    { AMD_TARGET_GFX1100,    AMD_GLOBAL_ATOMIC_ADD,     "0000d6dc02047c00" },
    { AMD_TARGET_GFX1100,    AMD_GLOBAL_ATOMIC_SUB,     "0000dadc02047c00" },
    { AMD_TARGET_GFX1100,    AMD_GLOBAL_ATOMIC_AND,     "0000f2dc02047c00" },
    { AMD_TARGET_GFX1100,    AMD_GLOBAL_ATOMIC_OR,      "0000f6dc02047c00" },
    { AMD_TARGET_GFX1100,    AMD_GLOBAL_ATOMIC_XOR,     "0000fadc02047c00" },
    { AMD_TARGET_GFX1100,    AMD_GLOBAL_ATOMIC_SMIN,    "0000e2dc02047c00" },
    { AMD_TARGET_GFX1100,    AMD_GLOBAL_ATOMIC_SMAX,    "0000eadc02047c00" },
    { AMD_TARGET_GFX1100,    AMD_GLOBAL_ATOMIC_SWAP,    "0000cedc02047c00" },
    { AMD_TARGET_GFX1100,    AMD_GLOBAL_ATOMIC_CMPSWAP, "0000d2dc02047c00" },
    { AMD_TARGET_GFX1100,    AMD_SCRATCH_LOAD_DWORD,    "000051dc0200fc01" },
    { AMD_TARGET_GFX1100,    AMD_SCRATCH_STORE_DWORD,   "000069dc0203fc00" },
    { AMD_TARGET_GFX1200,    AMD_S_ADD_U32,             "06080480" },
    { AMD_TARGET_GFX1200,    AMD_S_ADDC_U32,            "06080482" },
    { AMD_TARGET_GFX1200,    AMD_S_ADD_I32,             "06080481" },
    { AMD_TARGET_GFX1200,    AMD_S_SUB_U32,             "06088480" },
    { AMD_TARGET_GFX1200,    AMD_S_MUL_I32,             "06080496" },
    { AMD_TARGET_GFX1200,    AMD_S_AND_B32,             "0608048b" },
    { AMD_TARGET_GFX1200,    AMD_S_OR_B32,              "0608048c" },
    { AMD_TARGET_GFX1200,    AMD_S_XOR_B32,             "0608048d" },
    { AMD_TARGET_GFX1200,    AMD_S_LSHL_B32,            "06080484" },
    { AMD_TARGET_GFX1200,    AMD_S_LSHR_B32,            "06080485" },
    { AMD_TARGET_GFX1200,    AMD_S_ASHR_I32,            "06080486" },
    { AMD_TARGET_GFX1200,    AMD_S_ANDN2_B32,           "06080491" },
    { AMD_TARGET_GFX1200,    AMD_S_ORN2_B32,            "06080492" },
    { AMD_TARGET_GFX1200,    AMD_S_BFE_I32,             "06088493" },
    { AMD_TARGET_GFX1200,    AMD_S_CSELECT_B32,         "06080498" },
    { AMD_TARGET_GFX1200,    AMD_S_MOV_B32,             "060084be" },
    { AMD_TARGET_GFX1200,    AMD_S_NOT_B32,             "061e84be" },
    { AMD_TARGET_GFX1200,    AMD_S_AND_SAVEEXEC_B32,    "062084be" },
    { AMD_TARGET_GFX1200,    AMD_S_SETPC_B64,           "064880be" },
    { AMD_TARGET_GFX1200,    AMD_S_SWAPPC_B64,          "064984be" },
    { AMD_TARGET_GFX1200,    AMD_S_GETPC_B64,           "004784be" },
    { AMD_TARGET_GFX1200,    AMD_S_CMP_EQ_U32,          "060806bf" },
    { AMD_TARGET_GFX1200,    AMD_S_CMP_NE_U32,          "060807bf" },
    { AMD_TARGET_GFX1200,    AMD_S_CMP_LT_U32,          "06080abf" },
    { AMD_TARGET_GFX1200,    AMD_S_CMP_LE_U32,          "06080bbf" },
    { AMD_TARGET_GFX1200,    AMD_S_CMP_GT_U32,          "060808bf" },
    { AMD_TARGET_GFX1200,    AMD_S_CMP_GE_U32,          "060809bf" },
    { AMD_TARGET_GFX1200,    AMD_S_CMP_LT_I32,          "060804bf" },
    { AMD_TARGET_GFX1200,    AMD_S_CMP_LE_I32,          "060805bf" },
    { AMD_TARGET_GFX1200,    AMD_S_CMP_GT_I32,          "060802bf" },
    { AMD_TARGET_GFX1200,    AMD_S_CMP_GE_I32,          "060803bf" },
    { AMD_TARGET_GFX1200,    AMD_S_CMP_EQ_I32,          "060800bf" },
    { AMD_TARGET_GFX1200,    AMD_S_CMP_NE_I32,          "060801bf" },
    { AMD_TARGET_GFX1200,    AMD_S_BRANCH,              "0000a0bf" },
    { AMD_TARGET_GFX1200,    AMD_S_CBRANCH_SCC0,        "0000a1bf" },
    { AMD_TARGET_GFX1200,    AMD_S_CBRANCH_SCC1,        "0000a2bf" },
    { AMD_TARGET_GFX1200,    AMD_S_CBRANCH_EXECZ,       "0000a5bf" },
    { AMD_TARGET_GFX1200,    AMD_S_CBRANCH_EXECNZ,      "0000a6bf" },
    { AMD_TARGET_GFX1200,    AMD_S_ENDPGM,              "0000b0bf" },
    { AMD_TARGET_GFX1200,    AMD_S_BARRIER,             "0000bdbf" },
    { AMD_TARGET_GFX1200,    AMD_S_TRAP,                "000090bf" },
    { AMD_TARGET_GFX1200,    AMD_S_WAITCNT,             "f7ff89bf" },
    { AMD_TARGET_GFX1200,    AMD_S_NOP,                 "000080bf" },
    { AMD_TARGET_GFX1200,    AMD_S_WAIT_LOADCNT,        "0000c0bf" },
    { AMD_TARGET_GFX1200,    AMD_S_WAIT_STORECNT,       "0000c1bf" },
    { AMD_TARGET_GFX1200,    AMD_S_WAIT_DSCNT,          "0000c6bf" },
    { AMD_TARGET_GFX1200,    AMD_S_WAIT_KMCNT,          "0000c7bf" },
    { AMD_TARGET_GFX1200,    AMD_S_LOAD_DWORD,          "030100f4000000f8" },
    { AMD_TARGET_GFX1200,    AMD_S_LOAD_DWORDX2,        "032100f4000000f8" },
    { AMD_TARGET_GFX1200,    AMD_S_LOAD_DWORDX4,        "044100f4000000f8" },
    { AMD_TARGET_GFX1200,    AMD_V_ADD_U32,             "0207024a" },
    { AMD_TARGET_GFX1200,    AMD_V_SUB_U32,             "0207024c" },
    { AMD_TARGET_GFX1200,    AMD_V_MUL_LO_U32,          "01002cd702070200" },
    { AMD_TARGET_GFX1200,    AMD_V_MUL_HI_U32,          "01002dd702070200" },
    { AMD_TARGET_GFX1200,    AMD_V_AND_B32,             "02070236" },
    { AMD_TARGET_GFX1200,    AMD_V_OR_B32,              "02070238" },
    { AMD_TARGET_GFX1200,    AMD_V_XOR_B32,             "0207023a" },
    { AMD_TARGET_GFX1200,    AMD_V_LSHLREV_B32,         "02070230" },
    { AMD_TARGET_GFX1200,    AMD_V_LSHRREV_B32,         "02070232" },
    { AMD_TARGET_GFX1200,    AMD_V_ASHRREV_I32,         "02070234" },
    { AMD_TARGET_GFX1200,    AMD_V_ADD_F32,             "02070206" },
    { AMD_TARGET_GFX1200,    AMD_V_SUB_F32,             "02070208" },
    { AMD_TARGET_GFX1200,    AMD_V_MUL_F32,             "02070210" },
    { AMD_TARGET_GFX1200,    AMD_V_CNDMASK_B32,         "02070202" },
    { AMD_TARGET_GFX1200,    AMD_V_MIN_F32,             "0207022a" },
    { AMD_TARGET_GFX1200,    AMD_V_MAX_F32,             "0207022c" },
    { AMD_TARGET_GFX1200,    AMD_V_MIN_U32,             "02070226" },
    { AMD_TARGET_GFX1200,    AMD_V_MOV_B32,             "0203027e" },
    { AMD_TARGET_GFX1200,    AMD_V_CVT_F32_I32,         "020b027e" },
    { AMD_TARGET_GFX1200,    AMD_V_CVT_F32_U32,         "020d027e" },
    { AMD_TARGET_GFX1200,    AMD_V_CVT_I32_F32,         "0211027e" },
    { AMD_TARGET_GFX1200,    AMD_V_CVT_U32_F32,         "020f027e" },
    { AMD_TARGET_GFX1200,    AMD_V_CVT_F32_F16,         "0217027e" },
    { AMD_TARGET_GFX1200,    AMD_V_CVT_F16_F32,         "0215027e" },
    { AMD_TARGET_GFX1200,    AMD_V_CVT_F64_F32,         "0221087e" },
    { AMD_TARGET_GFX1200,    AMD_V_CVT_F32_F64,         "021f027e" },
    { AMD_TARGET_GFX1200,    AMD_V_RCP_F32,             "0255027e" },
    { AMD_TARGET_GFX1200,    AMD_V_SQRT_F32,            "0267027e" },
    { AMD_TARGET_GFX1200,    AMD_V_RSQ_F32,             "025d027e" },
    { AMD_TARGET_GFX1200,    AMD_V_EXP_F32,             "024b027e" },
    { AMD_TARGET_GFX1200,    AMD_V_LOG_F32,             "024f027e" },
    { AMD_TARGET_GFX1200,    AMD_V_SIN_F32,             "026b027e" },
    { AMD_TARGET_GFX1200,    AMD_V_COS_F32,             "026d027e" },
    { AMD_TARGET_GFX1200,    AMD_V_FLOOR_F32,           "0249027e" },
    { AMD_TARGET_GFX1200,    AMD_V_CEIL_F32,            "0245027e" },
    { AMD_TARGET_GFX1200,    AMD_V_TRUNC_F32,           "0243027e" },
    { AMD_TARGET_GFX1200,    AMD_V_RNDNE_F32,           "0247027e" },
    { AMD_TARGET_GFX1200,    AMD_V_FRACT_F32,           "0241027e" },
    { AMD_TARGET_GFX1200,    AMD_V_NOT_B32,             "026f027e" },
    { AMD_TARGET_GFX1200,    AMD_V_READFIRSTLANE_B32,   "0205087e" },
    { AMD_TARGET_GFX1200,    AMD_V_FFBL_B32,            "0275027e" },
    { AMD_TARGET_GFX1200,    AMD_V_FFBH_U32,            "0273027e" },
    { AMD_TARGET_GFX1200,    AMD_V_BFREV_B32,           "0271027e" },
    { AMD_TARGET_GFX1200,    AMD_V_BCNT_U32_B32,        "01001ed702070200" },
    { AMD_TARGET_GFX1200,    AMD_V_MAD_U32_U24,         "01000bd602071204" },
    { AMD_TARGET_GFX1200,    AMD_V_BFE_I32,             "010011d602071204" },
    { AMD_TARGET_GFX1200,    AMD_V_BFE_U32,             "010010d602071204" },
    { AMD_TARGET_GFX1200,    AMD_V_LSHL_ADD_U32,        "010046d602071204" },
    { AMD_TARGET_GFX1200,    AMD_V_ADD3_U32,            "010055d602071204" },
    { AMD_TARGET_GFX1200,    AMD_V_CMP_EQ_U32,          "0207947c" },
    { AMD_TARGET_GFX1200,    AMD_V_CMP_NE_U32,          "02079a7c" },
    { AMD_TARGET_GFX1200,    AMD_V_CMP_LT_U32,          "0207927c" },
    { AMD_TARGET_GFX1200,    AMD_V_CMP_LE_U32,          "0207967c" },
    { AMD_TARGET_GFX1200,    AMD_V_CMP_GT_U32,          "0207987c" },
    { AMD_TARGET_GFX1200,    AMD_V_CMP_GE_U32,          "02079c7c" },
    { AMD_TARGET_GFX1200,    AMD_V_CMP_LT_I32,          "0207827c" },
    { AMD_TARGET_GFX1200,    AMD_V_CMP_LE_I32,          "0207867c" },
    { AMD_TARGET_GFX1200,    AMD_V_CMP_GT_I32,          "0207887c" },
    { AMD_TARGET_GFX1200,    AMD_V_CMP_GE_I32,          "02078c7c" },
    { AMD_TARGET_GFX1200,    AMD_V_CMP_EQ_I32,          "0207847c" },
    { AMD_TARGET_GFX1200,    AMD_V_CMP_NE_I32,          "02078a7c" },
    { AMD_TARGET_GFX1200,    AMD_V_CMP_EQ_F32,          "0207247c" },
    { AMD_TARGET_GFX1200,    AMD_V_CMP_NE_F32,          "02073a7c" },
    { AMD_TARGET_GFX1200,    AMD_V_CMP_LT_F32,          "0207227c" },
    { AMD_TARGET_GFX1200,    AMD_V_CMP_LE_F32,          "0207267c" },
    { AMD_TARGET_GFX1200,    AMD_V_CMP_GT_F32,          "0207287c" },
    { AMD_TARGET_GFX1200,    AMD_V_CMP_GE_F32,          "02072c7c" },
    { AMD_TARGET_GFX1200,    AMD_V_CMP_O_F32,           "02072e7c" },
    { AMD_TARGET_GFX1200,    AMD_V_CMP_U_F32,           "0207307c" },
    { AMD_TARGET_GFX1200,    AMD_V_CMP_NLT_F32,         "02073c7c" },
    { AMD_TARGET_GFX1200,    AMD_V_CMP_NLE_F32,         "0207387c" },
    { AMD_TARGET_GFX1200,    AMD_V_CMP_NGT_F32,         "0207367c" },
    { AMD_TARGET_GFX1200,    AMD_V_CMP_NGE_F32,         "0207327c" },
    { AMD_TARGET_GFX1200,    AMD_V_CMP_NEQ_F32,         "02073a7c" },
    { AMD_TARGET_GFX1200,    AMD_DS_READ_B32,           "0000d8d802000001" },
    { AMD_TARGET_GFX1200,    AMD_DS_WRITE_B32,          "000034d802030000" },
    { AMD_TARGET_GFX1200,    AMD_DS_ADD_RTN_U32,        "000080d802030001" },
    { AMD_TARGET_GFX1200,    AMD_DS_SUB_RTN_U32,        "000084d802030001" },
    { AMD_TARGET_GFX1200,    AMD_DS_AND_RTN_B32,        "0000a4d802030001" },
    { AMD_TARGET_GFX1200,    AMD_DS_OR_RTN_B32,         "0000a8d802030001" },
    { AMD_TARGET_GFX1200,    AMD_DS_XOR_RTN_B32,        "0000acd802030001" },
    { AMD_TARGET_GFX1200,    AMD_DS_MIN_RTN_I32,        "000094d802030001" },
    { AMD_TARGET_GFX1200,    AMD_DS_MAX_RTN_I32,        "000098d802030001" },
    { AMD_TARGET_GFX1200,    AMD_DS_SWIZZLE_B32,        "0000d4d802000001" },
    { AMD_TARGET_GFX1200,    AMD_DS_BPERMUTE_B32,       "0000ccda02030001" },
    { AMD_TARGET_GFX1200,    AMD_GLOBAL_LOAD_DWORD,     "7c0005ee0100000002000000" },
    { AMD_TARGET_GFX1200,    AMD_GLOBAL_STORE_DWORD,    "7c8006ee0000000202000000" },
    { AMD_TARGET_GFX1200,    AMD_GLOBAL_LOAD_DWORDX2,   "7c4005ee0400000002000000" },
    { AMD_TARGET_GFX1200,    AMD_GLOBAL_STORE_DWORDX2,  "7cc006ee0000000202000000" },
    { AMD_TARGET_GFX1200,    AMD_GLOBAL_ATOMIC_ADD,     "7c400dee0000000202000000" },
    { AMD_TARGET_GFX1200,    AMD_GLOBAL_ATOMIC_SUB,     "7c800dee0000000202000000" },
    { AMD_TARGET_GFX1200,    AMD_GLOBAL_ATOMIC_AND,     "7c000fee0000000202000000" },
    { AMD_TARGET_GFX1200,    AMD_GLOBAL_ATOMIC_OR,      "7c400fee0000000202000000" },
    { AMD_TARGET_GFX1200,    AMD_GLOBAL_ATOMIC_XOR,     "7c800fee0000000202000000" },
    { AMD_TARGET_GFX1200,    AMD_GLOBAL_ATOMIC_SMIN,    "7c000eee0000000202000000" },
    { AMD_TARGET_GFX1200,    AMD_GLOBAL_ATOMIC_SMAX,    "7c800eee0000000202000000" },
    { AMD_TARGET_GFX1200,    AMD_GLOBAL_ATOMIC_SWAP,    "7cc00cee0000000202000000" },
    { AMD_TARGET_GFX1200,    AMD_GLOBAL_ATOMIC_CMPSWAP, "7c000dee0000000202000000" },
    { AMD_TARGET_GFX1200,    AMD_SCRATCH_LOAD_DWORD,    "7c0005ed0100020002000000" },
    { AMD_TARGET_GFX1200,    AMD_SCRATCH_STORE_DWORD,   "7c8006ed0000820102000000" },
};

#define ASY_NEXP ((int)(sizeof(asy_exp) / sizeof(asy_exp[0])))

static int asy_nib(char c)
{
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    return -1;
}

static int asy_hex(const char *h, uint8_t *out, int max)
{
    int n = 0, i;
    for (i = 0; i < 2 * max && h[i] && h[i + 1]; i += 2) {
        int hi = asy_nib(h[i]), lo = asy_nib(h[i + 1]);
        if (hi < 0 || lo < 0) return -1;
        out[n++] = (uint8_t)((hi << 4) | lo);
    }
    if (h[i] != '\0') return -1;
    return n;
}

static const amd_enc_entry_t *asy_tbl(amd_target_t t)
{
    if (t <= AMD_TARGET_GFX942) return amd_enc_table_gfx9;
    if (t <= AMD_TARGET_GFX1030) return amd_enc_table_gfx10;
    return amd_enc_table;
}

static const amd_enc_entry_t *asy_look(amd_target_t t, uint16_t op)
{
    if (t >= AMD_TARGET_GFX1200 && amd_enc_ovr_gfx12[op].mnemonic)
        return &amd_enc_ovr_gfx12[op];
    return &asy_tbl(t)[op];
}

static void asy01(void)
{
    uint8_t want[12], got[12];
    int i;
    for (i = 0; i < ASY_NEXP; i++) {
        const asy_exp_t *e = &asy_exp[i];
        int nw, ng;
        if (e->op >= AMD_OP_COUNT) { CHECK(0); }
        if (e->hex[0] == '\0') continue;
        nw = asy_hex(e->hex, want, 12);
        CHECK(nw > 0);
        ng = asy_enc((amd_target_t)e->tgt, e->op, asy_sh[e->op], got);
        if (ng != nw || memcmp(got, want, (size_t)nw) != 0) {
            printf("  FAIL tgt=%u op=%u want=%s\n",
                   (unsigned)e->tgt, (unsigned)e->op, e->hex);
            nfail++;
            return;
        }
    }
    PASS();
}
TH_REG("asy", 1, "every opcode matches llvm-mc bytes", asy01)

static void asy02(void)
{
    static uint8_t seen[ASY_NTGT][AMD_OP_COUNT];
    int i, ti, op;
    memset(seen, 0, sizeof(seen));
    for (i = 0; i < ASY_NEXP; i++) {
        const asy_exp_t *e = &asy_exp[i];
        for (ti = 0; ti < ASY_NTGT; ti++)
            if (asy_tgt[ti] == (amd_target_t)e->tgt && e->op < AMD_OP_COUNT)
                seen[ti][e->op] = 1;
    }
    for (ti = 0; ti < ASY_NTGT; ti++) {
        for (op = 0; op < AMD_OP_COUNT; op++) {
            const amd_enc_entry_t *ent = asy_look(asy_tgt[ti], (uint16_t)op);
            if (asy_sh[op] == ASY_NONE) continue;
            if (!ent->mnemonic) continue;
            if (!seen[ti][op]) {
                printf("  FAIL tgt=%d op=%d %s not in the assay\n",
                       (int)asy_tgt[ti], op, ent->mnemonic);
                nfail++;
                return;
            }
        }
    }
    PASS();
}
TH_REG("asy", 2, "every encodable opcode is assayed", asy02)

static void asy03(void)
{
    int ti, a, b;
    for (ti = 0; ti < ASY_NTGT; ti++) {
        for (a = 0; a < AMD_OP_COUNT; a++) {
            const amd_enc_entry_t *ea = asy_look(asy_tgt[ti], (uint16_t)a);
            if (!ea->mnemonic || ea->fmt == AMD_FMT_PSEUDO) continue;
            if (ea->fmt == AMD_FMT_GADDR) continue;
            for (b = a + 1; b < AMD_OP_COUNT; b++) {
                const amd_enc_entry_t *eb = asy_look(asy_tgt[ti], (uint16_t)b);
                if (!eb->mnemonic || eb->fmt != ea->fmt) continue;
                if (eb->hw_opcode != ea->hw_opcode) continue;
                if (strcmp(eb->mnemonic, ea->mnemonic) == 0) continue;
                printf("  FAIL tgt=%d %s and %s share opcode 0x%X\n",
                       (int)asy_tgt[ti], ea->mnemonic, eb->mnemonic,
                       (unsigned)ea->hw_opcode);
                nfail++;
                return;
            }
        }
    }
    PASS();
}
TH_REG("asy", 3, "no two opcodes collide in one table", asy03)

static void asy04(void)
{
    char txt[128];
    int ti, op;
    for (ti = 0; ti < ASY_NTGT; ti++) {
        int w64 = (asy_tgt[ti] <= AMD_TARGET_GFX942);
        for (op = 0; op < AMD_OP_COUNT; op++) {
            const amd_enc_entry_t *ent = asy_look(asy_tgt[ti], (uint16_t)op);
            int n;
            if (asy_sh[op] == ASY_NONE) continue;
            if (!ent->mnemonic) continue;
            txt[0] = '\0';
            n = asy_txt(txt, (int)sizeof(txt), asy_sh[op], ent->mnemonic, w64);
            if (n <= 0 || n >= (int)sizeof(txt) || txt[0] == '\0') {
                printf("  FAIL tgt=%d %s renders no operand form\n",
                       (int)asy_tgt[ti], ent->mnemonic);
                nfail++;
                return;
            }
        }
    }
    PASS();
}
TH_REG("asy", 4, "every assayed opcode renders operands", asy04)
