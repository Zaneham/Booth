#include "nvidia.h"
#include "backend.h"
#include "barracuda.h"
#include <stdio.h>

#define VC_ANY  0
#define VC_R32  1
#define VC_R64  2
#define VC_F32  3
#define VC_F64  4
#define VC_PRD  5
#define VC_R16  6
#define VC_H16  7
#define VC_B32  8
#define VC_X32  9
#define VC_X16 10
#define VC_I8  11
#define VC_I16 12
#define VC_REQ 0x80u

typedef struct {
    const char *nm;
    uint8_t     s[NV_MAX_OPS];
} nv_sig_t;

static const nv_sig_t nv_sig[NV_OP_COUNT] = {
    [NV_ADD_U32]     = { "add.u32",     { VC_R32, VC_R32, VC_R32 } },
    [NV_ADD_U64]     = { "add.u64",     { VC_R64, VC_R64, VC_R64 } },
    [NV_ADD_S32]     = { "add.s32",     { VC_R32, VC_R32, VC_R32 } },
    [NV_SUB_U32]     = { "sub.u32",     { VC_R32, VC_R32, VC_R32 } },
    [NV_SUB_S32]     = { "sub.s32",     { VC_R32, VC_R32, VC_R32 } },
    [NV_SUB_S64]     = { "sub.s64",     { VC_R64, VC_R64, VC_R64 } },
    [NV_MUL_LO_U32]  = { "mul.lo.u32",  { VC_R32, VC_R32, VC_R32 } },
    [NV_MUL_LO_S32]  = { "mul.lo.s32",  { VC_R32, VC_R32, VC_R32 } },
    [NV_MUL_LO_U64]  = { "mul.lo.u64",  { VC_R64, VC_R64, VC_R64 } },
    [NV_MUL_HI_U32]  = { "mul.hi.u32",  { VC_R32, VC_R32, VC_R32 } },
    [NV_MUL_HI_S32]  = { "mul.hi.s32",  { VC_R32, VC_R32, VC_R32 } },
    [NV_MUL_HI_U64]  = { "mul.hi.u64",  { VC_R64, VC_R64, VC_R64 } },
    [NV_MAD_LO_U64]  = { "mad.lo.u64",  { VC_R64, VC_R64, VC_R64, VC_R64 } },
    [NV_DIV_U32]     = { "div.u32",     { VC_R32, VC_R32, VC_R32 } },
    [NV_DIV_S32]     = { "div.s32",     { VC_R32, VC_R32, VC_R32 } },
    [NV_REM_U32]     = { "rem.u32",     { VC_R32, VC_R32, VC_R32 } },
    [NV_REM_S32]     = { "rem.s32",     { VC_R32, VC_R32, VC_R32 } },
    [NV_NEG_S32]     = { "neg.s32",     { VC_R32, VC_R32 } },

    [NV_ADD_F32]     = { "add.rn.f32",  { VC_F32, VC_F32, VC_F32 } },
    [NV_ADD_F64]     = { "add.rn.f64",  { VC_F64, VC_F64, VC_F64 } },
    [NV_SUB_F32]     = { "sub.rn.f32",  { VC_F32, VC_F32, VC_F32 } },
    [NV_SUB_F64]     = { "sub.rn.f64",  { VC_F64, VC_F64, VC_F64 } },
    [NV_MUL_F32]     = { "mul.rn.f32",  { VC_F32, VC_F32, VC_F32 } },
    [NV_MUL_F64]     = { "mul.rn.f64",  { VC_F64, VC_F64, VC_F64 } },
    [NV_DIV_F32]     = { "div.rn.f32",  { VC_F32, VC_F32, VC_F32 } },
    [NV_DIV_F64]     = { "div.rn.f64",  { VC_F64, VC_F64, VC_F64 } },
    [NV_FMA_F32]     = { "fma.rn.f32",  { VC_F32, VC_F32, VC_F32, VC_F32 } },
    [NV_FMA_F64]     = { "fma.rn.f64",  { VC_F64, VC_F64, VC_F64, VC_F64 } },
    [NV_NEG_F32]     = { "neg.f32",     { VC_F32, VC_F32 } },
    [NV_NEG_F64]     = { "neg.f64",     { VC_F64, VC_F64 } },
    [NV_ABS_F32]     = { "abs.f32",     { VC_F32, VC_F32 } },
    [NV_ABS_F64]     = { "abs.f64",     { VC_F64, VC_F64 } },

    [NV_AND_B32]     = { "and.b32",     { VC_X32, VC_X32, VC_X32 } },
    [NV_AND_B64]     = { "and.b64",     { VC_R64, VC_R64, VC_R64 } },
    [NV_OR_B32]      = { "or.b32",      { VC_X32, VC_X32, VC_X32 } },
    [NV_OR_B64]      = { "or.b64",      { VC_R64, VC_R64, VC_R64 } },
    [NV_XOR_B32]     = { "xor.b32",     { VC_X32, VC_X32, VC_X32 } },
    [NV_XOR_B64]     = { "xor.b64",     { VC_R64, VC_R64, VC_R64 } },
    [NV_NOT_B32]     = { "not.b32",     { VC_X32, VC_X32 } },
    [NV_NOT_B64]     = { "not.b64",     { VC_R64, VC_R64 } },
    [NV_SHL_B32]     = { "shl.b32",     { VC_X32, VC_X32, VC_R32 } },
    [NV_SHL_B64]     = { "shl.b64",     { VC_R64, VC_R64, VC_R32 } },
    [NV_SHR_U32]     = { "shr.u32",     { VC_R32, VC_R32, VC_R32 } },
    [NV_SHR_S32]     = { "shr.s32",     { VC_R32, VC_R32, VC_R32 } },
    [NV_SHR_U64]     = { "shr.u64",     { VC_R64, VC_R64, VC_R32 } },

    [NV_POPC_B32]    = { "popc.b32",    { VC_R32, VC_X32 } },
    [NV_POPC_B64]    = { "popc.b64",    { VC_R32, VC_R64 } },
    [NV_CLZ_B32]     = { "clz.b32",     { VC_R32, VC_X32 } },
    [NV_CLZ_B64]     = { "clz.b64",     { VC_R32, VC_R64 } },
    [NV_BREV_B32]    = { "brev.b32",    { VC_X32, VC_X32 } },
    [NV_BREV_B64]    = { "brev.b64",    { VC_R64, VC_R64 } },

    [NV_SETP_EQ_U32] = { "setp.eq.u32", { VC_PRD, VC_R32, VC_R32 } },
    [NV_SETP_NE_U32] = { "setp.ne.u32", { VC_PRD, VC_R32, VC_R32 } },
    [NV_SETP_LT_U32] = { "setp.lt.u32", { VC_PRD, VC_R32, VC_R32 } },
    [NV_SETP_LE_U32] = { "setp.le.u32", { VC_PRD, VC_R32, VC_R32 } },
    [NV_SETP_GT_U32] = { "setp.gt.u32", { VC_PRD, VC_R32, VC_R32 } },
    [NV_SETP_GE_U32] = { "setp.ge.u32", { VC_PRD, VC_R32, VC_R32 } },
    [NV_SETP_LT_S32] = { "setp.lt.s32", { VC_PRD, VC_R32, VC_R32 } },
    [NV_SETP_LE_S32] = { "setp.le.s32", { VC_PRD, VC_R32, VC_R32 } },
    [NV_SETP_GT_S32] = { "setp.gt.s32", { VC_PRD, VC_R32, VC_R32 } },
    [NV_SETP_GE_S32] = { "setp.ge.s32", { VC_PRD, VC_R32, VC_R32 } },
    [NV_SETP_EQ_F32] = { "setp.eq.f32", { VC_PRD, VC_F32, VC_F32 } },
    [NV_SETP_NE_F32] = { "setp.ne.f32", { VC_PRD, VC_F32, VC_F32 } },
    [NV_SETP_LT_F32] = { "setp.lt.f32", { VC_PRD, VC_F32, VC_F32 } },
    [NV_SETP_LE_F32] = { "setp.le.f32", { VC_PRD, VC_F32, VC_F32 } },
    [NV_SETP_GT_F32] = { "setp.gt.f32", { VC_PRD, VC_F32, VC_F32 } },
    [NV_SETP_GE_F32] = { "setp.ge.f32", { VC_PRD, VC_F32, VC_F32 } },
    [NV_SETP_EQ_F64] = { "setp.eq.f64", { VC_PRD, VC_F64, VC_F64 } },
    [NV_SETP_NE_F64] = { "setp.ne.f64", { VC_PRD, VC_F64, VC_F64 } },
    [NV_SETP_LT_F64] = { "setp.lt.f64", { VC_PRD, VC_F64, VC_F64 } },
    [NV_SETP_LE_F64] = { "setp.le.f64", { VC_PRD, VC_F64, VC_F64 } },
    [NV_SETP_GT_F64] = { "setp.gt.f64", { VC_PRD, VC_F64, VC_F64 } },
    [NV_SETP_GE_F64] = { "setp.ge.f64", { VC_PRD, VC_F64, VC_F64 } },
    [NV_SETP_EQ_U64] = { "setp.eq.u64", { VC_PRD, VC_R64, VC_R64 } },
    [NV_SETP_NE_U64] = { "setp.ne.u64", { VC_PRD, VC_R64, VC_R64 } },

    [NV_SELP_U32]    = { "selp.u32",    { VC_R32, VC_R32, VC_R32, VC_PRD } },
    [NV_SELP_U64]    = { "selp.u64",    { VC_R64, VC_R64, VC_R64, VC_PRD } },
    [NV_SELP_F32]    = { "selp.f32",    { VC_F32, VC_F32, VC_F32, VC_PRD } },
    [NV_SELP_F64]    = { "selp.f64",    { VC_F64, VC_F64, VC_F64, VC_PRD } },

    [NV_MOV_U32]     = { "mov.u32",     { VC_R32, VC_R32 } },
    [NV_MOV_U64]     = { "mov.u64",     { VC_R64, VC_R64 } },
    [NV_MOV_F32]     = { "mov.f32",     { VC_F32, VC_F32 } },
    [NV_MOV_F64]     = { "mov.f64",     { VC_F64, VC_F64 } },
    [NV_MOV_PRED]    = { "mov.pred",    { VC_PRD, VC_PRD } },

    [NV_CVT_U32_F32] = { "cvt.rzi.u32.f32", { VC_R32, (uint8_t)(VC_F32 | VC_REQ) } },
    [NV_CVT_S32_F32] = { "cvt.rzi.s32.f32", { VC_R32, (uint8_t)(VC_F32 | VC_REQ) } },
    [NV_CVT_U32_F64] = { "cvt.rzi.u32.f64", { VC_R32, (uint8_t)(VC_F64 | VC_REQ) } },
    [NV_CVT_S32_F64] = { "cvt.rzi.s32.f64", { VC_R32, (uint8_t)(VC_F64 | VC_REQ) } },
    [NV_CVT_F32_U32] = { "cvt.rn.f32.u32",  { VC_F32, (uint8_t)(VC_R32 | VC_REQ) } },
    [NV_CVT_F32_S32] = { "cvt.rn.f32.s32",  { VC_F32, (uint8_t)(VC_R32 | VC_REQ) } },
    [NV_CVT_F32_F64] = { "cvt.rn.f32.f64",  { VC_F32, (uint8_t)(VC_F64 | VC_REQ) } },
    [NV_CVT_F64_F32] = { "cvt.f64.f32",     { VC_F64, (uint8_t)(VC_F32 | VC_REQ) } },
    [NV_CVT_U64_U32] = { "cvt.u64.u32",     { VC_R64, (uint8_t)(VC_R32 | VC_REQ) } },
    [NV_CVT_S64_S32] = { "cvt.s64.s32",     { VC_R64, (uint8_t)(VC_R32 | VC_REQ) } },
    [NV_CVT_U32_U64] = { "cvt.u32.u64",     { VC_R32, (uint8_t)(VC_R64 | VC_REQ) } },
    [NV_CVT_U64_F64] = { "cvt.rzi.u64.f64", { VC_R64, (uint8_t)(VC_F64 | VC_REQ) } },
    [NV_CVT_S64_F64] = { "cvt.rzi.s64.f64", { VC_R64, (uint8_t)(VC_F64 | VC_REQ) } },
    [NV_CVT_F64_U64] = { "cvt.rn.f64.u64",  { VC_F64, (uint8_t)(VC_R64 | VC_REQ) } },
    [NV_CVT_F64_S64] = { "cvt.rn.f64.s64",  { VC_F64, (uint8_t)(VC_R64 | VC_REQ) } },
    [NV_CVT_F64_U32] = { "cvt.rn.f64.u32",  { VC_F64, (uint8_t)(VC_R32 | VC_REQ) } },
    [NV_CVT_F64_S32] = { "cvt.rn.f64.s32",  { VC_F64, (uint8_t)(VC_R32 | VC_REQ) } },
    [NV_CVT_F32_F16] = { "cvt.f32.f16",     { VC_F32, (uint8_t)(VC_H16 | VC_REQ) } },
    [NV_CVT_F16_F32] = { "cvt.rn.f16.f32",  { VC_H16, (uint8_t)(VC_F32 | VC_REQ) } },
    [NV_CVT_U32_U16] = { "cvt.u32.u16",     { VC_R32, (uint8_t)(VC_R16 | VC_REQ) } },

    [NV_LD_GLB_U32]  = { "ld.global.u32", { VC_R32, VC_R64 } },
    [NV_LD_GLB_U64]  = { "ld.global.u64", { VC_R64, VC_R64 } },
    [NV_LD_GLB_F32]  = { "ld.global.f32", { VC_F32, VC_R64 } },
    [NV_LD_GLB_F64]  = { "ld.global.f64", { VC_F64, VC_R64 } },
    [NV_LD_GLB_U8]   = { "ld.global.u8",  { VC_I8,  VC_R64 } },
    [NV_LD_GLB_U16]  = { "ld.global.u16", { VC_I16, VC_R64 } },
    [NV_LD_GLB_B16]  = { "ld.global.b16", { VC_X16, VC_R64 } },
    [NV_ST_GLB_U32]  = { "st.global.u32", { VC_R64, VC_R32 } },
    [NV_ST_GLB_U64]  = { "st.global.u64", { VC_R64, VC_R64 } },
    [NV_ST_GLB_F32]  = { "st.global.f32", { VC_R64, VC_F32 } },
    [NV_ST_GLB_F64]  = { "st.global.f64", { VC_R64, VC_F64 } },
    [NV_ST_GLB_U8]   = { "st.global.u8",  { VC_R64, VC_I8  } },
    [NV_ST_GLB_U16]  = { "st.global.u16", { VC_R64, VC_I16 } },
    [NV_ST_GLB_B16]  = { "st.global.b16", { VC_R64, VC_X16 } },

    [NV_LD_SHR_U32]  = { "ld.shared.u32", { VC_R32, VC_R64 } },
    [NV_LD_SHR_F32]  = { "ld.shared.f32", { VC_F32, VC_R64 } },
    [NV_LD_SHR_U8]   = { "ld.shared.u8",  { VC_I8,  VC_R64 } },
    [NV_LD_SHR_U16]  = { "ld.shared.u16", { VC_I16, VC_R64 } },
    [NV_LD_SHR_B16]  = { "ld.shared.b16", { VC_X16, VC_R64 } },
    [NV_ST_SHR_U32]  = { "st.shared.u32", { VC_R64, VC_R32 } },
    [NV_ST_SHR_F32]  = { "st.shared.f32", { VC_R64, VC_F32 } },
    [NV_ST_SHR_U8]   = { "st.shared.u8",  { VC_R64, VC_I8  } },
    [NV_ST_SHR_U16]  = { "st.shared.u16", { VC_R64, VC_I16 } },
    [NV_ST_SHR_B16]  = { "st.shared.b16", { VC_R64, VC_X16 } },
    [NV_LD_SHR_U64]  = { "ld.shared.u64", { VC_R64, VC_R64 } },
    [NV_LD_SHR_F64]  = { "ld.shared.f64", { VC_F64, VC_R64 } },
    [NV_ST_SHR_U64]  = { "st.shared.u64", { VC_R64, VC_R64 } },
    [NV_ST_SHR_F64]  = { "st.shared.f64", { VC_R64, VC_F64 } },

    [NV_LD_LOC_U32]  = { "ld.local.u32",  { VC_X32, VC_R64 } },
    [NV_LD_LOC_U64]  = { "ld.local.u64",  { VC_R64, VC_R64 } },
    [NV_LD_LOC_F32]  = { "ld.local.f32",  { VC_F32, VC_R64 } },
    [NV_LD_LOC_F64]  = { "ld.local.f64",  { VC_F64, VC_R64 } },
    [NV_LD_LOC_U8]   = { "ld.local.u8",   { VC_I8,  VC_R64 } },
    [NV_LD_LOC_U16]  = { "ld.local.u16",  { VC_I16, VC_R64 } },
    [NV_LD_LOC_B16]  = { "ld.local.b16",  { VC_X16, VC_R64 } },
    [NV_ST_LOC_U32]  = { "st.local.u32",  { VC_R64, VC_X32 } },
    [NV_ST_LOC_U64]  = { "st.local.u64",  { VC_R64, VC_R64 } },
    [NV_ST_LOC_F32]  = { "st.local.f32",  { VC_R64, VC_F32 } },
    [NV_ST_LOC_F64]  = { "st.local.f64",  { VC_R64, VC_F64 } },
    [NV_ST_LOC_U8]   = { "st.local.u8",   { VC_R64, VC_I8  } },
    [NV_ST_LOC_U16]  = { "st.local.u16",  { VC_R64, VC_I16 } },
    [NV_ST_LOC_B16]  = { "st.local.b16",  { VC_R64, VC_X16 } },

    [NV_LD_PARAM_U32] = { "ld.param.u32", { VC_R32, VC_ANY } },
    [NV_LD_PARAM_U64] = { "ld.param.u64", { VC_R64, VC_ANY } },
    [NV_LD_PARAM_F32] = { "ld.param.f32", { VC_F32, VC_ANY } },
    [NV_LD_PARAM_F64] = { "ld.param.f64", { VC_F64, VC_ANY } },
    [NV_LD_PARAM_B16] = { "ld.param.b16", { VC_X16, VC_ANY } },

    [NV_ATOM_ADD_U32]  = { "atom.add.u32",  { VC_R32, VC_R64, VC_R32 } },
    [NV_ATOM_ADD_F32]  = { "atom.add.f32",  { VC_F32, VC_R64, VC_F32 } },
    [NV_ATOM_ADD_U64]  = { "atom.add.u64",  { VC_R64, VC_R64, VC_R64 } },
    [NV_ATOM_ADD_F64]  = { "atom.add.f64",  { VC_F64, VC_R64, VC_F64 } },
    [NV_ATOM_MIN_U32]  = { "atom.min.u32",  { VC_R32, VC_R64, VC_R32 } },
    [NV_ATOM_MAX_U32]  = { "atom.max.u32",  { VC_R32, VC_R64, VC_R32 } },
    [NV_ATOM_AND_B32]  = { "atom.and.b32",  { VC_X32, VC_R64, VC_X32 } },
    [NV_ATOM_OR_B32]   = { "atom.or.b32",   { VC_X32, VC_R64, VC_X32 } },
    [NV_ATOM_XOR_B32]  = { "atom.xor.b32",  { VC_X32, VC_R64, VC_X32 } },
    [NV_ATOM_AND_B64]  = { "atom.and.b64",  { VC_R64, VC_R64, VC_R64 } },
    [NV_ATOM_OR_B64]   = { "atom.or.b64",   { VC_R64, VC_R64, VC_R64 } },
    [NV_ATOM_XOR_B64]  = { "atom.xor.b64",  { VC_R64, VC_R64, VC_R64 } },
    [NV_ATOM_XCHG_B32] = { "atom.exch.b32", { VC_X32, VC_R64, VC_X32 } },
    [NV_ATOM_XCHG_B64] = { "atom.exch.b64", { VC_R64, VC_R64, VC_R64 } },
    [NV_ATOM_CAS_B32]  = { "atom.cas.b32",
                           { VC_X32, VC_R64, VC_X32, VC_X32 } },
    [NV_ATOM_CAS_B64]  = { "atom.cas.b64",
                           { VC_R64, VC_R64, VC_R64, VC_R64 } },

    [NV_BRA]         = { "bra",           { VC_ANY } },
    [NV_BRA_PRED]    = { "bra.pred",      { VC_PRD, VC_ANY } },
    [NV_BAR_SYNC]    = { "bar.sync",      { VC_ANY } },
    [NV_MEMBAR]      = { "membar",        { VC_ANY } },
    [NV_NANOSLP]     = { "nanosleep.u32", { VC_R32 } },
    [NV_BARRED_OR]   = { "bar.red.or.pred",  { VC_PRD, VC_PRD } },
    [NV_BARRED_AND]  = { "bar.red.and.pred", { VC_PRD, VC_PRD } },
    [NV_BARRED_POPC] = { "bar.red.popc.u32", { VC_R32, VC_PRD } },
    [NV_GBAR]        = { "gridbar",       { VC_ANY } },

    [NV_SHFL_IDX]    = { "shfl.sync.idx.b32",  { VC_X32, VC_X32, VC_R32 } },
    [NV_SHFL_UP]     = { "shfl.sync.up.b32",   { VC_X32, VC_X32, VC_R32 } },
    [NV_SHFL_DOWN]   = { "shfl.sync.down.b32", { VC_X32, VC_X32, VC_R32 } },
    [NV_SHFL_XOR]    = { "shfl.sync.bfly.b32", { VC_X32, VC_X32, VC_R32 } },
    [NV_VOTE_BALLOT] = { "vote.sync.ballot.b32", { VC_X32, VC_PRD } },
    [NV_VOTE_ANY]    = { "vote.sync.any.pred",   { VC_PRD, VC_PRD } },
    [NV_VOTE_ALL]    = { "vote.sync.all.pred",   { VC_PRD, VC_PRD } },

    [NV_SQRT_F32]    = { "sqrt.approx.f32",  { VC_F32, VC_F32 } },
    [NV_SQRT_F64]    = { "sqrt.rn.f64",      { VC_F64, VC_F64 } },
    [NV_RSQ_F32]     = { "rsqrt.approx.f32", { VC_F32, VC_F32 } },
    [NV_RCP_F32]     = { "rcp.approx.f32",   { VC_F32, VC_F32 } },
    [NV_SIN_F32]     = { "sin.approx.f32",   { VC_F32, VC_F32 } },
    [NV_COS_F32]     = { "cos.approx.f32",   { VC_F32, VC_F32 } },
    [NV_EX2_F32]     = { "ex2.approx.f32",   { VC_F32, VC_F32 } },
    [NV_LG2_F32]     = { "lg2.approx.f32",   { VC_F32, VC_F32 } },
    [NV_FLOOR_F32]   = { "cvt.rmi.f32.f32",  { VC_F32, VC_F32 } },
    [NV_CEIL_F32]    = { "cvt.rpi.f32.f32",  { VC_F32, VC_F32 } },
    [NV_TRUNC_F32]   = { "cvt.rzi.f32.f32",  { VC_F32, VC_F32 } },
    [NV_ROUND_F32]   = { "cvt.rni.f32.f32",  { VC_F32, VC_F32 } },
    [NV_MIN_F32]     = { "min.f32",     { VC_F32, VC_F32, VC_F32 } },
    [NV_MAX_F32]     = { "max.f32",     { VC_F32, VC_F32, VC_F32 } },
    [NV_MIN_U32]     = { "min.u32",     { VC_R32, VC_R32, VC_R32 } },
    [NV_MAX_U32]     = { "max.u32",     { VC_R32, VC_R32, VC_R32 } },
    [NV_MIN_S32]     = { "min.s32",     { VC_R32, VC_R32, VC_R32 } },
    [NV_MAX_S32]     = { "max.s32",     { VC_R32, VC_R32, VC_R32 } },

    [NV_RET]         = { "ret",           { VC_ANY } },
    [NV_EXIT]        = { "exit",          { VC_ANY } },
    [NV_TRAP]        = { "trap",          { VC_ANY } },
    [NV_BARWARP]     = { "bar.warp.sync", { VC_ANY } },
    [NV_MOV_F64_LIT] = { "mov.f64",  { VC_F64, VC_ANY, VC_ANY } },
    [NV_LEA_LOCAL]   = { "mov.u64",  { VC_R64, VC_ANY } },
    [NV_LEA_GLB]     = { "mov.u64",  { VC_R64, VC_ANY } },
    [NV_LEA_DSH]     = { "mov.u64",  { VC_R64 } },
    [NV_MOV_PK16]    = { "mov.b32",  { VC_B32, VC_X16, VC_X16 } },
    [NV_MMA]         = { "mma.sync", { VC_ANY, VC_ANY, VC_ANY, VC_ANY } },
    [NV_WLD]         = { "wmma.load",  { VC_ANY, VC_R64, VC_R32 } },
    [NV_WST]         = { "wmma.store", { VC_R64, VC_ANY, VC_R32 } },
    [NV_WMMA]        = { "wmma.mma",   { VC_ANY, VC_ANY, VC_ANY, VC_ANY } },
    [NV_ASM]         = { "asm",      { VC_ANY } },
    [NV_CVTA_GLB]    = { "cvta.global.u64", { VC_R64, VC_R64 } },
    [NV_CVTA_LOC]    = { "cvta.local.u64",  { VC_R64, VC_R64 } },
    [NV_CALL]        = { "call",     { VC_ANY } },
    [NV_ST_RETP]     = { "st.param", { VC_ANY } },

    [NV_SETP_LT_S64] = { "setp.lt.s64", { VC_PRD, VC_R64, VC_R64 } },
    [NV_SETP_LE_S64] = { "setp.le.s64", { VC_PRD, VC_R64, VC_R64 } },
    [NV_SETP_GT_S64] = { "setp.gt.s64", { VC_PRD, VC_R64, VC_R64 } },
    [NV_SETP_GE_S64] = { "setp.ge.s64", { VC_PRD, VC_R64, VC_R64 } },
    [NV_SETP_LT_U64] = { "setp.lt.u64", { VC_PRD, VC_R64, VC_R64 } },
    [NV_SETP_LE_U64] = { "setp.le.u64", { VC_PRD, VC_R64, VC_R64 } },
    [NV_SETP_GT_U64] = { "setp.gt.u64", { VC_PRD, VC_R64, VC_R64 } },
    [NV_SETP_GE_U64] = { "setp.ge.u64", { VC_PRD, VC_R64, VC_R64 } },
    [NV_DIV_U64]     = { "div.u64", { VC_R64, VC_R64, VC_R64 } },
    [NV_DIV_S64]     = { "div.s64", { VC_R64, VC_R64, VC_R64 } },
    [NV_REM_U64]     = { "rem.u64", { VC_R64, VC_R64, VC_R64 } },
    [NV_REM_S64]     = { "rem.s64", { VC_R64, VC_R64, VC_R64 } },
    [NV_NEG_S64]     = { "neg.s64", { VC_R64, VC_R64 } },
    [NV_SHR_S64]     = { "shr.s64", { VC_R64, VC_R64, VC_R32 } },
    [NV_MIN_U64]     = { "min.u64", { VC_R64, VC_R64, VC_R64 } },
    [NV_MAX_U64]     = { "max.u64", { VC_R64, VC_R64, VC_R64 } },
    [NV_MIN_S64]     = { "min.s64", { VC_R64, VC_R64, VC_R64 } },
    [NV_MAX_S64]     = { "max.s64", { VC_R64, VC_R64, VC_R64 } },
    [NV_MOV_B16]     = { "mov.b16", { VC_X16, VC_X16 } },
    [NV_MOV_B32]     = { "mov.b32", { VC_ANY, VC_ANY } },
    [NV_MOV_B64]     = { "mov.b64", { VC_ANY, VC_ANY } },
    [NV_CVT_S32_S16] = { "cvt.s32.s16",    { VC_R32, (uint8_t)(VC_R16 | VC_REQ) } },
    [NV_CVT_U16_U32] = { "cvt.u16.u32",    { VC_R16, (uint8_t)(VC_R32 | VC_REQ) } },
    [NV_CVT_F32_U64] = { "cvt.rn.f32.u64", { VC_F32, (uint8_t)(VC_R64 | VC_REQ) } },
    [NV_CVT_F32_S64] = { "cvt.rn.f32.s64", { VC_F32, (uint8_t)(VC_R64 | VC_REQ) } },
    [NV_CVT_U64_F32] = { "cvt.rzi.u64.f32",{ VC_R64, (uint8_t)(VC_F32 | VC_REQ) } },
    [NV_CVT_S64_F32] = { "cvt.rzi.s64.f32",{ VC_R64, (uint8_t)(VC_F32 | VC_REQ) } },
    [NV_CVT_F16_F64] = { "cvt.rn.f16.f64", { VC_H16, (uint8_t)(VC_F64 | VC_REQ) } },
    [NV_CVT_F64_F16] = { "cvt.f64.f16",    { VC_F64, (uint8_t)(VC_H16 | VC_REQ) } },
    [NV_CVT_F32_BF16] = { "cvt.f32.bf16",    { VC_F32, (uint8_t)(VC_R16 | VC_REQ) } },
    [NV_CVT_BF16_F32] = { "cvt.rn.bf16.f32", { VC_R16, (uint8_t)(VC_F32 | VC_REQ) } }
};

static const char *rfnm(uint8_t rf)
{
    switch (rf) {
    case NV_RF_U32:  return "u32";
    case NV_RF_U64:  return "u64";
    case NV_RF_F32:  return "f32";
    case NV_RF_F64:  return "f64";
    case NV_RF_PRED: return "pred";
    case NV_RF_U16:  return "u16";
    case NV_RF_F16:  return "f16";
    case NV_RF_B32:  return "b32";
    default:         return "unknown";
    }
}

static const char *vcnm(uint8_t c)
{
    switch (c & (uint8_t)~VC_REQ) {
    case VC_R32: return "u32";
    case VC_R64: return "u64";
    case VC_F32: return "f32";
    case VC_F64: return "f64";
    case VC_PRD: return "pred";
    case VC_R16: return "u16";
    case VC_H16: return "f16";
    case VC_B32: return "b32";
    case VC_X32: return "u32 or b32";
    case VC_X16: return "u16 or f16";
    case VC_I8:  return "u16 or u32";
    case VC_I16: return "u16 or u32";
    default:     return "any";
    }
}

static int vcok(uint8_t c, uint8_t rf)
{
    switch (c & (uint8_t)~VC_REQ) {
    case VC_R32: return rf == NV_RF_U32;
    case VC_R64: return rf == NV_RF_U64;
    case VC_F32: return rf == NV_RF_F32;
    case VC_F64: return rf == NV_RF_F64;
    case VC_PRD: return rf == NV_RF_PRED;
    case VC_R16: return rf == NV_RF_U16;
    case VC_H16: return rf == NV_RF_F16;
    case VC_B32: return rf == NV_RF_B32;
    case VC_X32: return rf == NV_RF_U32 || rf == NV_RF_B32;
    case VC_X16: return rf == NV_RF_U16 || rf == NV_RF_F16;
    case VC_I8:  return rf == NV_RF_U16 || rf == NV_RF_U32;
    case VC_I16: return rf == NV_RF_U16 || rf == NV_RF_U32;
    default:     return 1;
    }
}

static const char *vfnam(const nv_module_t *nv, uint32_t fi)
{
    uint32_t off;
    if (fi >= nv->num_mfunc) return "an unnamed function";
    off = nv->mfuncs[fi].name;
    if (nv->bir == NULL || off >= nv->bir->string_len)
        return "an unnamed function";
    return nv->bir->strings + off;
}

static int vretp(const nv_module_t *nv, uint32_t fi, const nv_minst_t *I)
{
    if (I->ops[0].kind != NV_MOP_REG) return 0;
    if (I->ops[0].rfile == (uint8_t)I->flags) return 0;
    (void)be_fail(BC_E842, vfnam(nv, fi), rfnm(I->ops[0].rfile),
                  rfnm((uint8_t)I->flags));
    return 1;
}

static int vcall(const nv_module_t *nv, uint32_t fi, const nv_minst_t *I)
{
    const nv_call_t *C = &nv->calls[I->flags % NV_MAX_CALL];
    if (C->hasret == 0 || C->ret.kind != NV_MOP_REG) return 0;
    if (C->ret.rfile == C->retrf) return 0;
    (void)be_fail(BC_E843, vfnam(nv, fi), rfnm(C->ret.rfile),
                  rfnm(C->retrf));
    return 1;
}

static int vinst(const nv_module_t *nv, uint32_t fi, const nv_minst_t *I)
{
    const nv_sig_t *G;
    uint32_t k;
    int bad = 0;

    if (I->op >= NV_OP_COUNT) {
        (void)be_fail(BC_E841, vfnam(nv, fi), (unsigned)I->op);
        return 1;
    }
    G = &nv_sig[I->op];
    if (G->nm == NULL) {
        (void)be_fail(BC_E841, vfnam(nv, fi), (unsigned)I->op);
        return 1;
    }

    if (I->op == NV_CALL)     return vcall(nv, fi, I);
    if (I->op == NV_ST_RETP)  return vretp(nv, fi, I);

    for (k = 0; k < NV_MAX_OPS; k++) {
        if (I->ops[k].kind != NV_MOP_REG) {
            if ((G->s[k] & VC_REQ) != 0
             && I->ops[k].kind != NV_MOP_NONE) {
                (void)be_fail(BC_E845, vfnam(nv, fi), G->nm, (unsigned)k);
                bad = 1;
            }
            continue;
        }
        if (vcok(G->s[k], I->ops[k].rfile)) continue;
        (void)be_fail(BC_E840, vfnam(nv, fi), G->nm, (unsigned)k,
                      vcnm(G->s[k]), rfnm(I->ops[k].rfile));
        bad = 1;
    }
    return bad;
}

int nv_vchk(const nv_module_t *nv)
{
    uint32_t fi, bi, j;
    uint32_t nbad = 0;

    if (nv == NULL) return BC_OK;
    for (fi = 0; fi < nv->num_mfunc; fi++) {
        const nv_mfunc_t *MF = &nv->mfuncs[fi];
        for (bi = 0; bi < MF->num_blks && MF->first_blk + bi < nv->num_mblk;
             bi++) {
            const nv_mblk_t *MB = &nv->mblks[MF->first_blk + bi];
            for (j = 0; j < MB->num_insts
                     && MB->first_inst + j < nv->num_minst; j++) {
                if (vinst(nv, fi, &nv->minsts[MB->first_inst + j]) == 0)
                    continue;
                nbad++;
                if (nbad >= 64u) return BC_ERR_VERIFY;
            }
        }
    }
    return nbad != 0u ? BC_ERR_VERIFY : BC_OK;
}
