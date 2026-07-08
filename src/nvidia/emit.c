#include "nvidia.h"
#include <stdio.h>
#include <string.h>
#include <stdarg.h>

/* PTX text emission. We generate polite ASCII that ptxas will JIT into
 * whatever SASS NVIDIA deems appropriate. Like writing a prayer and
 * sliding it under the cathedral door. */

/* ---- Output Buffer ---- */

static void nv_apnd(nv_module_t *nv, const char *fmt, ...)
{
    if (nv->out_len >= NV_MAX_OUT - 1) return;
    va_list ap;
    va_start(ap, fmt);
    int n = vsnprintf(nv->out_buf + nv->out_len,
                      NV_MAX_OUT - nv->out_len, fmt, ap);
    va_end(ap);
    if (n > 0)
        nv->out_len += (uint32_t)n;
    if (nv->out_len >= NV_MAX_OUT)
        nv->out_len = NV_MAX_OUT - 1;
}

/* ---- Special Register Names ---- */

static const char *spec_name(int32_t id)
{
    switch (id) {
    case NV_SPEC_TID_X:    return "%tid.x";
    case NV_SPEC_TID_Y:    return "%tid.y";
    case NV_SPEC_TID_Z:    return "%tid.z";
    case NV_SPEC_CTAID_X:  return "%ctaid.x";
    case NV_SPEC_CTAID_Y:  return "%ctaid.y";
    case NV_SPEC_CTAID_Z:  return "%ctaid.z";
    case NV_SPEC_NTID_X:   return "%ntid.x";
    case NV_SPEC_NTID_Y:   return "%ntid.y";
    case NV_SPEC_NTID_Z:   return "%ntid.z";
    case NV_SPEC_NCTAID_X: return "%nctaid.x";
    case NV_SPEC_NCTAID_Y: return "%nctaid.y";
    case NV_SPEC_NCTAID_Z: return "%nctaid.z";
    default:               return "%tid.x";
    }
}

/* ---- Operand Formatting ---- */

static void em_opnd(nv_module_t *nv, const nv_opnd_t *op)
{
    switch (op->kind) {
    case NV_MOP_REG:
        switch (op->rfile) {
        case NV_RF_U32:  nv_apnd(nv, "%%r%u", op->reg_num);  break;
        case NV_RF_U64:  nv_apnd(nv, "%%rd%u", op->reg_num); break;
        case NV_RF_F32:  nv_apnd(nv, "%%f%u", op->reg_num);  break;
        case NV_RF_F64:  nv_apnd(nv, "%%fd%u", op->reg_num); break;
        case NV_RF_PRED: nv_apnd(nv, "%%p%u", op->reg_num);  break;
        case NV_RF_U16:  nv_apnd(nv, "%%rh%u", op->reg_num); break;
        case NV_RF_F16:  nv_apnd(nv, "%%h%u", op->reg_num);  break;
        default:         nv_apnd(nv, "%%r%u", op->reg_num);   break;
        }
        break;
    case NV_MOP_IMM:
        nv_apnd(nv, "%d", op->imm);
        break;
    case NV_MOP_LABEL:
        nv_apnd(nv, "$BB%u", (unsigned)op->imm);
        break;
    case NV_MOP_SPEC:
        nv_apnd(nv, "%s", spec_name(op->imm));
        break;
    case NV_MOP_NONE:
    default:
        break;
    }
}

/* ---- Float Immediate Formatting ---- */
/* PTX accepts 0fXXXXXXXX for IEEE 754 hex float literals */

static void em_fimm(nv_module_t *nv, const nv_opnd_t *op)
{
    if (op->kind == NV_MOP_IMM) {
        union { int32_t i; float f; } pun;
        pun.i = op->imm;
        if (pun.f == 0.0f)
            nv_apnd(nv, "0f00000000");
        else if (pun.f == 1.0f)
            nv_apnd(nv, "0f3F800000");
        else
            nv_apnd(nv, "0f%08X", (unsigned)(uint32_t)op->imm);
    } else {
        em_opnd(nv, op);
    }
}

/* ---- Type Suffix for PTX Instructions ---- */

static const char *rf_param(uint8_t rf)
{
    switch (rf) {
    case NV_RF_U64:  return ".u64";
    case NV_RF_F32:  return ".f32";
    case NV_RF_F64:  return ".f64";
    default:         return ".u32";
    }
}

/* ---- Per-Instruction Emission ---- */

static void em_inst(nv_module_t *nv, const nv_minst_t *I)
{
    nv_apnd(nv, "\t");

    switch (I->op) {

    /* ---- Integer Arithmetic ---- */
    case NV_ADD_U32:
        nv_apnd(nv, "add.u32 ");
        em_opnd(nv, &I->ops[0]); nv_apnd(nv, ", ");
        em_opnd(nv, &I->ops[1]); nv_apnd(nv, ", ");
        em_opnd(nv, &I->ops[2]);
        break;
    case NV_ADD_U64:
        nv_apnd(nv, "add.u64 ");
        em_opnd(nv, &I->ops[0]); nv_apnd(nv, ", ");
        em_opnd(nv, &I->ops[1]); nv_apnd(nv, ", ");
        em_opnd(nv, &I->ops[2]);
        break;
    case NV_ADD_S32:
        nv_apnd(nv, "add.s32 ");
        em_opnd(nv, &I->ops[0]); nv_apnd(nv, ", ");
        em_opnd(nv, &I->ops[1]); nv_apnd(nv, ", ");
        em_opnd(nv, &I->ops[2]);
        break;
    case NV_SUB_U32: case NV_SUB_S32:
        nv_apnd(nv, "sub%s ", I->op == NV_SUB_S32 ? ".s32" : ".u32");
        em_opnd(nv, &I->ops[0]); nv_apnd(nv, ", ");
        em_opnd(nv, &I->ops[1]); nv_apnd(nv, ", ");
        em_opnd(nv, &I->ops[2]);
        break;
    case NV_SUB_S64:
        nv_apnd(nv, "sub.s64 ");
        em_opnd(nv, &I->ops[0]); nv_apnd(nv, ", ");
        em_opnd(nv, &I->ops[1]); nv_apnd(nv, ", ");
        em_opnd(nv, &I->ops[2]);
        break;
    case NV_MUL_LO_U32:
        nv_apnd(nv, "mul.lo.u32 ");
        em_opnd(nv, &I->ops[0]); nv_apnd(nv, ", ");
        em_opnd(nv, &I->ops[1]); nv_apnd(nv, ", ");
        em_opnd(nv, &I->ops[2]);
        break;
    case NV_MUL_LO_S32:
        nv_apnd(nv, "mul.lo.s32 ");
        em_opnd(nv, &I->ops[0]); nv_apnd(nv, ", ");
        em_opnd(nv, &I->ops[1]); nv_apnd(nv, ", ");
        em_opnd(nv, &I->ops[2]);
        break;
    case NV_MUL_LO_U64:
        nv_apnd(nv, "mul.lo.u64 ");
        em_opnd(nv, &I->ops[0]); nv_apnd(nv, ", ");
        em_opnd(nv, &I->ops[1]); nv_apnd(nv, ", ");
        em_opnd(nv, &I->ops[2]);
        break;
    case NV_MUL_HI_U32:
        nv_apnd(nv, "mul.hi.u32 ");
        em_opnd(nv, &I->ops[0]); nv_apnd(nv, ", ");
        em_opnd(nv, &I->ops[1]); nv_apnd(nv, ", ");
        em_opnd(nv, &I->ops[2]);
        break;
    case NV_MUL_HI_S32:
        nv_apnd(nv, "mul.hi.s32 ");
        em_opnd(nv, &I->ops[0]); nv_apnd(nv, ", ");
        em_opnd(nv, &I->ops[1]); nv_apnd(nv, ", ");
        em_opnd(nv, &I->ops[2]);
        break;
    case NV_MAD_LO_U64:
        nv_apnd(nv, "mad.lo.u64 ");
        em_opnd(nv, &I->ops[0]); nv_apnd(nv, ", ");
        em_opnd(nv, &I->ops[1]); nv_apnd(nv, ", ");
        em_opnd(nv, &I->ops[2]); nv_apnd(nv, ", ");
        em_opnd(nv, &I->ops[3]);
        break;
    case NV_DIV_U32:
        nv_apnd(nv, "div.u32 ");
        em_opnd(nv, &I->ops[0]); nv_apnd(nv, ", ");
        em_opnd(nv, &I->ops[1]); nv_apnd(nv, ", ");
        em_opnd(nv, &I->ops[2]);
        break;
    case NV_DIV_S32:
        nv_apnd(nv, "div.s32 ");
        em_opnd(nv, &I->ops[0]); nv_apnd(nv, ", ");
        em_opnd(nv, &I->ops[1]); nv_apnd(nv, ", ");
        em_opnd(nv, &I->ops[2]);
        break;
    case NV_REM_U32:
        nv_apnd(nv, "rem.u32 ");
        em_opnd(nv, &I->ops[0]); nv_apnd(nv, ", ");
        em_opnd(nv, &I->ops[1]); nv_apnd(nv, ", ");
        em_opnd(nv, &I->ops[2]);
        break;
    case NV_REM_S32:
        nv_apnd(nv, "rem.s32 ");
        em_opnd(nv, &I->ops[0]); nv_apnd(nv, ", ");
        em_opnd(nv, &I->ops[1]); nv_apnd(nv, ", ");
        em_opnd(nv, &I->ops[2]);
        break;
    case NV_NEG_S32:
        nv_apnd(nv, "neg.s32 ");
        em_opnd(nv, &I->ops[0]); nv_apnd(nv, ", ");
        em_opnd(nv, &I->ops[1]);
        break;

    /* ---- FP Arithmetic ---- */
    case NV_ADD_F32:
        nv_apnd(nv, "add.rn.f32 ");
        em_opnd(nv, &I->ops[0]); nv_apnd(nv, ", ");
        em_fimm(nv, &I->ops[1]); nv_apnd(nv, ", ");
        em_fimm(nv, &I->ops[2]);
        break;
    case NV_ADD_F64:
        nv_apnd(nv, "add.rn.f64 ");
        em_opnd(nv, &I->ops[0]); nv_apnd(nv, ", ");
        em_fimm(nv, &I->ops[1]); nv_apnd(nv, ", ");
        em_fimm(nv, &I->ops[2]);
        break;
    case NV_SUB_F32:
        nv_apnd(nv, "sub.rn.f32 ");
        em_opnd(nv, &I->ops[0]); nv_apnd(nv, ", ");
        em_fimm(nv, &I->ops[1]); nv_apnd(nv, ", ");
        em_fimm(nv, &I->ops[2]);
        break;
    case NV_SUB_F64:
        nv_apnd(nv, "sub.rn.f64 ");
        em_opnd(nv, &I->ops[0]); nv_apnd(nv, ", ");
        em_fimm(nv, &I->ops[1]); nv_apnd(nv, ", ");
        em_fimm(nv, &I->ops[2]);
        break;
    case NV_MUL_F32:
        nv_apnd(nv, "mul.rn.f32 ");
        em_opnd(nv, &I->ops[0]); nv_apnd(nv, ", ");
        em_fimm(nv, &I->ops[1]); nv_apnd(nv, ", ");
        em_fimm(nv, &I->ops[2]);
        break;
    case NV_MUL_F64:
        nv_apnd(nv, "mul.rn.f64 ");
        em_opnd(nv, &I->ops[0]); nv_apnd(nv, ", ");
        em_fimm(nv, &I->ops[1]); nv_apnd(nv, ", ");
        em_fimm(nv, &I->ops[2]);
        break;
    case NV_DIV_F32:
        nv_apnd(nv, "div.rn.f32 ");
        em_opnd(nv, &I->ops[0]); nv_apnd(nv, ", ");
        em_fimm(nv, &I->ops[1]); nv_apnd(nv, ", ");
        em_fimm(nv, &I->ops[2]);
        break;
    case NV_DIV_F64:
        nv_apnd(nv, "div.rn.f64 ");
        em_opnd(nv, &I->ops[0]); nv_apnd(nv, ", ");
        em_opnd(nv, &I->ops[1]); nv_apnd(nv, ", ");
        em_opnd(nv, &I->ops[2]);
        break;
    case NV_FMA_F32:
        nv_apnd(nv, "fma.rn.f32 ");
        em_opnd(nv, &I->ops[0]); nv_apnd(nv, ", ");
        em_fimm(nv, &I->ops[1]); nv_apnd(nv, ", ");
        em_fimm(nv, &I->ops[2]); nv_apnd(nv, ", ");
        em_fimm(nv, &I->ops[3]);
        break;
    case NV_FMA_F64:
        nv_apnd(nv, "fma.rn.f64 ");
        em_opnd(nv, &I->ops[0]); nv_apnd(nv, ", ");
        em_opnd(nv, &I->ops[1]); nv_apnd(nv, ", ");
        em_opnd(nv, &I->ops[2]); nv_apnd(nv, ", ");
        em_opnd(nv, &I->ops[3]);
        break;
    case NV_NEG_F32:
        nv_apnd(nv, "neg.f32 ");
        em_opnd(nv, &I->ops[0]); nv_apnd(nv, ", ");
        em_opnd(nv, &I->ops[1]);
        break;
    case NV_NEG_F64:
        nv_apnd(nv, "neg.f64 ");
        em_opnd(nv, &I->ops[0]); nv_apnd(nv, ", ");
        em_opnd(nv, &I->ops[1]);
        break;
    case NV_ABS_F32:
        nv_apnd(nv, "abs.f32 ");
        em_opnd(nv, &I->ops[0]); nv_apnd(nv, ", ");
        em_opnd(nv, &I->ops[1]);
        break;
    case NV_ABS_F64:
        nv_apnd(nv, "abs.f64 ");
        em_opnd(nv, &I->ops[0]); nv_apnd(nv, ", ");
        em_opnd(nv, &I->ops[1]);
        break;

    /* ---- Logic / Shift ---- */
    case NV_AND_B32: case NV_AND_B64:
        nv_apnd(nv, "and%s ", I->op == NV_AND_B64 ? ".b64" : ".b32");
        em_opnd(nv, &I->ops[0]); nv_apnd(nv, ", ");
        em_opnd(nv, &I->ops[1]); nv_apnd(nv, ", ");
        em_opnd(nv, &I->ops[2]);
        break;
    case NV_OR_B32: case NV_OR_B64:
        nv_apnd(nv, "or%s ", I->op == NV_OR_B64 ? ".b64" : ".b32");
        em_opnd(nv, &I->ops[0]); nv_apnd(nv, ", ");
        em_opnd(nv, &I->ops[1]); nv_apnd(nv, ", ");
        em_opnd(nv, &I->ops[2]);
        break;
    case NV_XOR_B32: case NV_XOR_B64:
        nv_apnd(nv, "xor%s ", I->op == NV_XOR_B64 ? ".b64" : ".b32");
        em_opnd(nv, &I->ops[0]); nv_apnd(nv, ", ");
        em_opnd(nv, &I->ops[1]); nv_apnd(nv, ", ");
        em_opnd(nv, &I->ops[2]);
        break;
    case NV_NOT_B32: case NV_NOT_B64:
        nv_apnd(nv, "not%s ", I->op == NV_NOT_B64 ? ".b64" : ".b32");
        em_opnd(nv, &I->ops[0]); nv_apnd(nv, ", ");
        em_opnd(nv, &I->ops[1]);
        break;
    case NV_SHL_B32: case NV_SHL_B64:
        nv_apnd(nv, "shl%s ", I->op == NV_SHL_B64 ? ".b64" : ".b32");
        em_opnd(nv, &I->ops[0]); nv_apnd(nv, ", ");
        em_opnd(nv, &I->ops[1]); nv_apnd(nv, ", ");
        em_opnd(nv, &I->ops[2]);
        break;
    case NV_SHR_U32:
        nv_apnd(nv, "shr.u32 ");
        em_opnd(nv, &I->ops[0]); nv_apnd(nv, ", ");
        em_opnd(nv, &I->ops[1]); nv_apnd(nv, ", ");
        em_opnd(nv, &I->ops[2]);
        break;
    case NV_SHR_S32:
        nv_apnd(nv, "shr.s32 ");
        em_opnd(nv, &I->ops[0]); nv_apnd(nv, ", ");
        em_opnd(nv, &I->ops[1]); nv_apnd(nv, ", ");
        em_opnd(nv, &I->ops[2]);
        break;
    case NV_SHR_U64:
        nv_apnd(nv, "shr.u64 ");
        em_opnd(nv, &I->ops[0]); nv_apnd(nv, ", ");
        em_opnd(nv, &I->ops[1]); nv_apnd(nv, ", ");
        em_opnd(nv, &I->ops[2]);
        break;

    /* ---- Comparison (setp) ---- */
    case NV_SETP_EQ_U32: case NV_SETP_NE_U32:
    case NV_SETP_LT_U32: case NV_SETP_LE_U32:
    case NV_SETP_GT_U32: case NV_SETP_GE_U32: {
        const char *cmp;
        switch (I->op) {
        case NV_SETP_EQ_U32: cmp = "eq"; break;
        case NV_SETP_NE_U32: cmp = "ne"; break;
        case NV_SETP_LT_U32: cmp = "lt"; break;
        case NV_SETP_LE_U32: cmp = "le"; break;
        case NV_SETP_GT_U32: cmp = "gt"; break;
        case NV_SETP_GE_U32: cmp = "ge"; break;
        default: cmp = "ne"; break;
        }
        nv_apnd(nv, "setp.%s.u32 ", cmp);
        em_opnd(nv, &I->ops[0]); nv_apnd(nv, ", ");
        em_opnd(nv, &I->ops[1]); nv_apnd(nv, ", ");
        em_opnd(nv, &I->ops[2]);
        break;
    }
    case NV_SETP_LT_S32: case NV_SETP_LE_S32:
    case NV_SETP_GT_S32: case NV_SETP_GE_S32: {
        const char *cmp;
        switch (I->op) {
        case NV_SETP_LT_S32: cmp = "lt"; break;
        case NV_SETP_LE_S32: cmp = "le"; break;
        case NV_SETP_GT_S32: cmp = "gt"; break;
        case NV_SETP_GE_S32: cmp = "ge"; break;
        default: cmp = "ne"; break;
        }
        nv_apnd(nv, "setp.%s.s32 ", cmp);
        em_opnd(nv, &I->ops[0]); nv_apnd(nv, ", ");
        em_opnd(nv, &I->ops[1]); nv_apnd(nv, ", ");
        em_opnd(nv, &I->ops[2]);
        break;
    }
    case NV_SETP_EQ_F32: case NV_SETP_NE_F32:
    case NV_SETP_LT_F32: case NV_SETP_LE_F32:
    case NV_SETP_GT_F32: case NV_SETP_GE_F32: {
        const char *cmp;
        switch (I->op) {
        case NV_SETP_EQ_F32: cmp = "eq"; break;
        case NV_SETP_NE_F32: cmp = "ne"; break;
        case NV_SETP_LT_F32: cmp = "lt"; break;
        case NV_SETP_LE_F32: cmp = "le"; break;
        case NV_SETP_GT_F32: cmp = "gt"; break;
        case NV_SETP_GE_F32: cmp = "ge"; break;
        default: cmp = "ne"; break;
        }
        nv_apnd(nv, "setp.%s.f32 ", cmp);
        em_opnd(nv, &I->ops[0]); nv_apnd(nv, ", ");
        em_fimm(nv, &I->ops[1]); nv_apnd(nv, ", ");
        em_fimm(nv, &I->ops[2]);
        break;
    }
    case NV_SETP_EQ_F64: case NV_SETP_NE_F64:
    case NV_SETP_LT_F64: case NV_SETP_LE_F64:
    case NV_SETP_GT_F64: case NV_SETP_GE_F64: {
        const char *cmp;
        switch (I->op) {
        case NV_SETP_EQ_F64: cmp = "eq"; break;
        case NV_SETP_NE_F64: cmp = "ne"; break;
        case NV_SETP_LT_F64: cmp = "lt"; break;
        case NV_SETP_LE_F64: cmp = "le"; break;
        case NV_SETP_GT_F64: cmp = "gt"; break;
        case NV_SETP_GE_F64: cmp = "ge"; break;
        default: cmp = "ne"; break;
        }
        nv_apnd(nv, "setp.%s.f64 ", cmp);
        em_opnd(nv, &I->ops[0]); nv_apnd(nv, ", ");
        em_opnd(nv, &I->ops[1]); nv_apnd(nv, ", ");
        em_opnd(nv, &I->ops[2]);
        break;
    }
    case NV_SETP_EQ_U64: case NV_SETP_NE_U64:
        nv_apnd(nv, "setp.%s.u64 ", I->op == NV_SETP_EQ_U64 ? "eq" : "ne");
        em_opnd(nv, &I->ops[0]); nv_apnd(nv, ", ");
        em_opnd(nv, &I->ops[1]); nv_apnd(nv, ", ");
        em_opnd(nv, &I->ops[2]);
        break;

    /* ---- Select ---- */
    case NV_SELP_U32: case NV_SELP_U64:
    case NV_SELP_F32: case NV_SELP_F64: {
        const char *tsuf;
        switch (I->op) {
        case NV_SELP_U64: tsuf = ".u64"; break;
        case NV_SELP_F32: tsuf = ".f32"; break;
        case NV_SELP_F64: tsuf = ".f64"; break;
        default:          tsuf = ".u32"; break;
        }
        nv_apnd(nv, "selp%s ", tsuf);
        em_opnd(nv, &I->ops[0]); nv_apnd(nv, ", ");
        /* src1/src2: use em_fimm for float types to emit 0fXXXXXXXX */
        if (I->op == NV_SELP_F32 || I->op == NV_SELP_F64) {
            em_fimm(nv, &I->ops[1]); nv_apnd(nv, ", ");
            em_fimm(nv, &I->ops[2]); nv_apnd(nv, ", ");
        } else {
            em_opnd(nv, &I->ops[1]); nv_apnd(nv, ", ");
            em_opnd(nv, &I->ops[2]); nv_apnd(nv, ", ");
        }
        em_opnd(nv, &I->ops[3]);
        break;
    }

    /* ---- Moves ---- */
    case NV_MOV_U32:
        nv_apnd(nv, "mov.u32 ");
        em_opnd(nv, &I->ops[0]); nv_apnd(nv, ", ");
        em_opnd(nv, &I->ops[1]);
        break;
    case NV_MOV_U64:
        nv_apnd(nv, "mov.u64 ");
        em_opnd(nv, &I->ops[0]); nv_apnd(nv, ", ");
        em_opnd(nv, &I->ops[1]);
        break;
    case NV_MOV_F32:
        nv_apnd(nv, "mov.f32 ");
        em_opnd(nv, &I->ops[0]); nv_apnd(nv, ", ");
        em_fimm(nv, &I->ops[1]);
        break;
    case NV_MOV_F64:
        nv_apnd(nv, "mov.f64 ");
        em_opnd(nv, &I->ops[0]); nv_apnd(nv, ", ");
        /* PTX f64 immediates need 0dXXXX hex format, not bare int */
        if (I->ops[1].kind == NV_MOP_IMM && I->ops[1].imm == 0)
            nv_apnd(nv, "0d0000000000000000");
        else
            em_opnd(nv, &I->ops[1]);
        break;
    case NV_MOV_PRED:
        nv_apnd(nv, "mov.pred ");
        em_opnd(nv, &I->ops[0]); nv_apnd(nv, ", ");
        em_opnd(nv, &I->ops[1]);
        break;

    /* ---- Conversions ---- */
    case NV_CVT_U32_F32:
        nv_apnd(nv, "cvt.rzi.u32.f32 ");
        em_opnd(nv, &I->ops[0]); nv_apnd(nv, ", ");
        em_opnd(nv, &I->ops[1]);
        break;
    case NV_CVT_S32_F32:
        nv_apnd(nv, "cvt.rzi.s32.f32 ");
        em_opnd(nv, &I->ops[0]); nv_apnd(nv, ", ");
        em_opnd(nv, &I->ops[1]);
        break;
    case NV_CVT_S32_F64:
        nv_apnd(nv, "cvt.rzi.s32.f64 ");
        em_opnd(nv, &I->ops[0]); nv_apnd(nv, ", ");
        em_opnd(nv, &I->ops[1]);
        break;
    case NV_CVT_U32_F64:
        nv_apnd(nv, "cvt.rzi.u32.f64 ");
        em_opnd(nv, &I->ops[0]); nv_apnd(nv, ", ");
        em_opnd(nv, &I->ops[1]);
        break;
    case NV_CVT_F32_U32:
        nv_apnd(nv, "cvt.rn.f32.u32 ");
        em_opnd(nv, &I->ops[0]); nv_apnd(nv, ", ");
        em_opnd(nv, &I->ops[1]);
        break;
    case NV_CVT_F32_S32:
        nv_apnd(nv, "cvt.rn.f32.s32 ");
        em_opnd(nv, &I->ops[0]); nv_apnd(nv, ", ");
        em_opnd(nv, &I->ops[1]);
        break;
    case NV_CVT_F32_F64:
        nv_apnd(nv, "cvt.rn.f32.f64 ");
        em_opnd(nv, &I->ops[0]); nv_apnd(nv, ", ");
        em_opnd(nv, &I->ops[1]);
        break;
    case NV_CVT_F64_F32:
        nv_apnd(nv, "cvt.f64.f32 ");
        em_opnd(nv, &I->ops[0]); nv_apnd(nv, ", ");
        em_opnd(nv, &I->ops[1]);
        break;
    case NV_CVT_U64_U32:
        nv_apnd(nv, "cvt.u64.u32 ");
        em_opnd(nv, &I->ops[0]); nv_apnd(nv, ", ");
        em_opnd(nv, &I->ops[1]);
        break;
    case NV_CVT_S64_S32:
        nv_apnd(nv, "cvt.s64.s32 ");
        em_opnd(nv, &I->ops[0]); nv_apnd(nv, ", ");
        em_opnd(nv, &I->ops[1]);
        break;
    case NV_CVT_U32_U64:
        nv_apnd(nv, "cvt.u32.u64 ");
        em_opnd(nv, &I->ops[0]); nv_apnd(nv, ", ");
        em_opnd(nv, &I->ops[1]);
        break;
    case NV_CVT_U64_F64:
        nv_apnd(nv, "cvt.rzi.u64.f64 ");
        em_opnd(nv, &I->ops[0]); nv_apnd(nv, ", ");
        em_opnd(nv, &I->ops[1]);
        break;
    case NV_CVT_S64_F64:
        nv_apnd(nv, "cvt.rzi.s64.f64 ");
        em_opnd(nv, &I->ops[0]); nv_apnd(nv, ", ");
        em_opnd(nv, &I->ops[1]);
        break;
    case NV_CVT_F64_U64:
        nv_apnd(nv, "cvt.rn.f64.u64 ");
        em_opnd(nv, &I->ops[0]); nv_apnd(nv, ", ");
        em_opnd(nv, &I->ops[1]);
        break;
    case NV_CVT_F64_S64:
        nv_apnd(nv, "cvt.rn.f64.s64 ");
        em_opnd(nv, &I->ops[0]); nv_apnd(nv, ", ");
        em_opnd(nv, &I->ops[1]);
        break;
    case NV_CVT_F64_S32:
        nv_apnd(nv, "cvt.rn.f64.s32 ");
        em_opnd(nv, &I->ops[0]); nv_apnd(nv, ", ");
        em_opnd(nv, &I->ops[1]);
        break;
    case NV_CVT_F64_U32:
        nv_apnd(nv, "cvt.rn.f64.u32 ");
        em_opnd(nv, &I->ops[0]); nv_apnd(nv, ", ");
        em_opnd(nv, &I->ops[1]);
        break;
    case NV_CVT_F32_F16:
        nv_apnd(nv, "cvt.f32.f16 ");
        em_opnd(nv, &I->ops[0]); nv_apnd(nv, ", ");
        em_opnd(nv, &I->ops[1]);
        break;
    case NV_CVT_F16_F32:
        nv_apnd(nv, "cvt.rn.f16.f32 ");
        em_opnd(nv, &I->ops[0]); nv_apnd(nv, ", ");
        em_opnd(nv, &I->ops[1]);
        break;

    /* ---- Loads: Global ---- */
    case NV_LD_GLB_U32: case NV_LD_GLB_U64:
    case NV_LD_GLB_F32: case NV_LD_GLB_F64:
    case NV_LD_GLB_U8:  case NV_LD_GLB_U16: {
        const char *tsuf;
        switch (I->op) {
        case NV_LD_GLB_U64: tsuf = ".u64"; break;
        case NV_LD_GLB_F32: tsuf = ".f32"; break;
        case NV_LD_GLB_F64: tsuf = ".f64"; break;
        case NV_LD_GLB_U8:  tsuf = ".u8";  break;
        case NV_LD_GLB_U16: tsuf = ".u16"; break;
        default:            tsuf = ".u32"; break;
        }
        nv_apnd(nv, "ld.global%s ", tsuf);
        em_opnd(nv, &I->ops[0]); nv_apnd(nv, ", [");
        em_opnd(nv, &I->ops[1]); nv_apnd(nv, "]");
        break;
    }

    /* ---- Stores: Global ---- */
    case NV_ST_GLB_U32: case NV_ST_GLB_U64:
    case NV_ST_GLB_F32: case NV_ST_GLB_F64:
    case NV_ST_GLB_U8:  case NV_ST_GLB_U16: {
        const char *tsuf;
        switch (I->op) {
        case NV_ST_GLB_U64: tsuf = ".u64"; break;
        case NV_ST_GLB_F32: tsuf = ".f32"; break;
        case NV_ST_GLB_F64: tsuf = ".f64"; break;
        case NV_ST_GLB_U8:  tsuf = ".u8";  break;
        case NV_ST_GLB_U16: tsuf = ".u16"; break;
        default:            tsuf = ".u32"; break;
        }
        nv_apnd(nv, "st.global%s [", tsuf);
        em_opnd(nv, &I->ops[0]); nv_apnd(nv, "], ");
        em_opnd(nv, &I->ops[1]);
        break;
    }

    /* ---- Loads/Stores: Shared ---- */
    case NV_LD_SHR_U32: case NV_LD_SHR_F32: {
        const char *tsuf = (I->op == NV_LD_SHR_F32) ? ".f32" : ".u32";
        nv_apnd(nv, "ld.shared%s ", tsuf);
        em_opnd(nv, &I->ops[0]); nv_apnd(nv, ", [");
        em_opnd(nv, &I->ops[1]); nv_apnd(nv, "]");
        break;
    }
    case NV_ST_SHR_U32: case NV_ST_SHR_F32: {
        const char *tsuf = (I->op == NV_ST_SHR_F32) ? ".f32" : ".u32";
        nv_apnd(nv, "st.shared%s [", tsuf);
        em_opnd(nv, &I->ops[0]); nv_apnd(nv, "], ");
        em_opnd(nv, &I->ops[1]);
        break;
    }

    /* ---- Loads/Stores: Local ---- */
    case NV_LD_LOC_U32: case NV_LD_LOC_U64:
    case NV_LD_LOC_F32: case NV_LD_LOC_F64: {
        const char *tsuf;
        switch (I->op) {
        case NV_LD_LOC_U64: tsuf = ".u64"; break;
        case NV_LD_LOC_F32: tsuf = ".f32"; break;
        case NV_LD_LOC_F64: tsuf = ".f64"; break;
        default:            tsuf = ".u32"; break;
        }
        nv_apnd(nv, "ld.local%s ", tsuf);
        em_opnd(nv, &I->ops[0]); nv_apnd(nv, ", [");
        em_opnd(nv, &I->ops[1]); nv_apnd(nv, "]");
        break;
    }
    case NV_ST_LOC_U32: case NV_ST_LOC_U64:
    case NV_ST_LOC_F32: case NV_ST_LOC_F64: {
        const char *tsuf;
        switch (I->op) {
        case NV_ST_LOC_U64: tsuf = ".u64"; break;
        case NV_ST_LOC_F32: tsuf = ".f32"; break;
        case NV_ST_LOC_F64: tsuf = ".f64"; break;
        default:            tsuf = ".u32"; break;
        }
        nv_apnd(nv, "st.local%s [", tsuf);
        em_opnd(nv, &I->ops[0]); nv_apnd(nv, "], ");
        em_opnd(nv, &I->ops[1]);
        break;
    }

    /* ---- Parameter Loads ---- */
    case NV_LD_PARAM_U32: case NV_LD_PARAM_U64:
    case NV_LD_PARAM_F32: case NV_LD_PARAM_F64: {
        const char *tsuf;
        switch (I->op) {
        case NV_LD_PARAM_U64: tsuf = ".u64"; break;
        case NV_LD_PARAM_F32: tsuf = ".f32"; break;
        case NV_LD_PARAM_F64: tsuf = ".f64"; break;
        default:              tsuf = ".u32"; break;
        }
        nv_apnd(nv, "ld.param%s ", tsuf);
        em_opnd(nv, &I->ops[0]); nv_apnd(nv, ", [param%d]",
                I->ops[1].imm);
        break;
    }

    /* ---- Atomics ---- */
    case NV_ATOM_ADD_U32:
        nv_apnd(nv, "atom.global.add.u32 ");
        em_opnd(nv, &I->ops[0]); nv_apnd(nv, ", [");
        em_opnd(nv, &I->ops[1]); nv_apnd(nv, "], ");
        em_opnd(nv, &I->ops[2]);
        break;
    case NV_ATOM_ADD_F32:
        nv_apnd(nv, "atom.global.add.f32 ");
        em_opnd(nv, &I->ops[0]); nv_apnd(nv, ", [");
        em_opnd(nv, &I->ops[1]); nv_apnd(nv, "], ");
        em_opnd(nv, &I->ops[2]);
        break;
    case NV_ATOM_MIN_U32:
        nv_apnd(nv, "atom.global.min.u32 ");
        em_opnd(nv, &I->ops[0]); nv_apnd(nv, ", [");
        em_opnd(nv, &I->ops[1]); nv_apnd(nv, "], ");
        em_opnd(nv, &I->ops[2]);
        break;
    case NV_ATOM_MAX_U32:
        nv_apnd(nv, "atom.global.max.u32 ");
        em_opnd(nv, &I->ops[0]); nv_apnd(nv, ", [");
        em_opnd(nv, &I->ops[1]); nv_apnd(nv, "], ");
        em_opnd(nv, &I->ops[2]);
        break;
    case NV_ATOM_AND_B32:
        nv_apnd(nv, "atom.global.and.b32 ");
        em_opnd(nv, &I->ops[0]); nv_apnd(nv, ", [");
        em_opnd(nv, &I->ops[1]); nv_apnd(nv, "], ");
        em_opnd(nv, &I->ops[2]);
        break;
    case NV_ATOM_OR_B32:
        nv_apnd(nv, "atom.global.or.b32 ");
        em_opnd(nv, &I->ops[0]); nv_apnd(nv, ", [");
        em_opnd(nv, &I->ops[1]); nv_apnd(nv, "], ");
        em_opnd(nv, &I->ops[2]);
        break;
    case NV_ATOM_XOR_B32:
        nv_apnd(nv, "atom.global.xor.b32 ");
        em_opnd(nv, &I->ops[0]); nv_apnd(nv, ", [");
        em_opnd(nv, &I->ops[1]); nv_apnd(nv, "], ");
        em_opnd(nv, &I->ops[2]);
        break;
    case NV_ATOM_XCHG_B32:
        nv_apnd(nv, "atom.global.exch.b32 ");
        em_opnd(nv, &I->ops[0]); nv_apnd(nv, ", [");
        em_opnd(nv, &I->ops[1]); nv_apnd(nv, "], ");
        em_opnd(nv, &I->ops[2]);
        break;
    case NV_ATOM_CAS_B32:
        nv_apnd(nv, "atom.global.cas.b32 ");
        em_opnd(nv, &I->ops[0]); nv_apnd(nv, ", [");
        em_opnd(nv, &I->ops[1]); nv_apnd(nv, "], ");
        em_opnd(nv, &I->ops[2]); nv_apnd(nv, ", ");
        if (I->num_uses > 2)
            em_opnd(nv, &I->ops[3]);
        else
            nv_apnd(nv, "0");
        break;

    /* ---- Branches ---- */
    case NV_BRA:
        nv_apnd(nv, "bra ");
        em_opnd(nv, &I->ops[0]);
        break;
    case NV_BRA_PRED:
        nv_apnd(nv, "@");
        em_opnd(nv, &I->ops[0]);
        nv_apnd(nv, " bra ");
        em_opnd(nv, &I->ops[1]);
        break;

    /* ---- Barriers ---- */
    case NV_BAR_SYNC:
        nv_apnd(nv, "bar.sync 0");
        break;

    /* ---- Warp Ops ---- */
    case NV_SHFL_IDX: case NV_SHFL_UP:
    case NV_SHFL_DOWN: case NV_SHFL_XOR: {
        const char *mode;
        switch (I->op) {
        case NV_SHFL_IDX:  mode = "idx";  break;
        case NV_SHFL_UP:   mode = "up";   break;
        case NV_SHFL_DOWN: mode = "down"; break;
        case NV_SHFL_XOR:  mode = "bfly"; break;
        default:           mode = "idx";  break;
        }
        nv_apnd(nv, "shfl.sync.%s.b32 ", mode);
        em_opnd(nv, &I->ops[0]); nv_apnd(nv, ", ");
        em_opnd(nv, &I->ops[1]); nv_apnd(nv, ", ");
        em_opnd(nv, &I->ops[2]); nv_apnd(nv, ", 31, -1");
        break;
    }
    case NV_VOTE_BALLOT:
        nv_apnd(nv, "vote.sync.ballot.b32 ");
        em_opnd(nv, &I->ops[0]); nv_apnd(nv, ", ");
        em_opnd(nv, &I->ops[1]); nv_apnd(nv, ", -1");
        break;
    case NV_VOTE_ANY:
        nv_apnd(nv, "vote.sync.any.pred ");
        em_opnd(nv, &I->ops[0]); nv_apnd(nv, ", ");
        em_opnd(nv, &I->ops[1]); nv_apnd(nv, ", -1");
        break;
    case NV_VOTE_ALL:
        nv_apnd(nv, "vote.sync.all.pred ");
        em_opnd(nv, &I->ops[0]); nv_apnd(nv, ", ");
        em_opnd(nv, &I->ops[1]); nv_apnd(nv, ", -1");
        break;

    /* ---- Math Builtins ---- */
    case NV_SQRT_F32:
        nv_apnd(nv, "sqrt.approx.f32 ");
        em_opnd(nv, &I->ops[0]); nv_apnd(nv, ", ");
        em_opnd(nv, &I->ops[1]);
        break;
    case NV_SQRT_F64:
        nv_apnd(nv, "sqrt.rn.f64 ");
        em_opnd(nv, &I->ops[0]); nv_apnd(nv, ", ");
        em_opnd(nv, &I->ops[1]);
        break;
    case NV_RSQ_F32:
        nv_apnd(nv, "rsqrt.approx.f32 ");
        em_opnd(nv, &I->ops[0]); nv_apnd(nv, ", ");
        em_opnd(nv, &I->ops[1]);
        break;
    case NV_RCP_F32:
        nv_apnd(nv, "rcp.approx.f32 ");
        em_opnd(nv, &I->ops[0]); nv_apnd(nv, ", ");
        em_opnd(nv, &I->ops[1]);
        break;
    case NV_SIN_F32:
        nv_apnd(nv, "sin.approx.f32 ");
        em_opnd(nv, &I->ops[0]); nv_apnd(nv, ", ");
        em_opnd(nv, &I->ops[1]);
        break;
    case NV_COS_F32:
        nv_apnd(nv, "cos.approx.f32 ");
        em_opnd(nv, &I->ops[0]); nv_apnd(nv, ", ");
        em_opnd(nv, &I->ops[1]);
        break;
    case NV_EX2_F32:
        nv_apnd(nv, "ex2.approx.f32 ");
        em_opnd(nv, &I->ops[0]); nv_apnd(nv, ", ");
        em_opnd(nv, &I->ops[1]);
        break;
    case NV_LG2_F32:
        nv_apnd(nv, "lg2.approx.f32 ");
        em_opnd(nv, &I->ops[0]); nv_apnd(nv, ", ");
        em_opnd(nv, &I->ops[1]);
        break;
    case NV_FLOOR_F32:
        nv_apnd(nv, "cvt.rmi.f32.f32 ");
        em_opnd(nv, &I->ops[0]); nv_apnd(nv, ", ");
        em_opnd(nv, &I->ops[1]);
        break;
    case NV_CEIL_F32:
        nv_apnd(nv, "cvt.rpi.f32.f32 ");
        em_opnd(nv, &I->ops[0]); nv_apnd(nv, ", ");
        em_opnd(nv, &I->ops[1]);
        break;
    case NV_TRUNC_F32:
        nv_apnd(nv, "cvt.rzi.f32.f32 ");
        em_opnd(nv, &I->ops[0]); nv_apnd(nv, ", ");
        em_opnd(nv, &I->ops[1]);
        break;
    case NV_ROUND_F32:
        nv_apnd(nv, "cvt.rni.f32.f32 ");
        em_opnd(nv, &I->ops[0]); nv_apnd(nv, ", ");
        em_opnd(nv, &I->ops[1]);
        break;
    case NV_MIN_F32:
        nv_apnd(nv, "min.f32 ");
        em_opnd(nv, &I->ops[0]); nv_apnd(nv, ", ");
        em_fimm(nv, &I->ops[1]); nv_apnd(nv, ", ");
        em_fimm(nv, &I->ops[2]);
        break;
    case NV_MAX_F32:
        nv_apnd(nv, "max.f32 ");
        em_opnd(nv, &I->ops[0]); nv_apnd(nv, ", ");
        em_fimm(nv, &I->ops[1]); nv_apnd(nv, ", ");
        em_fimm(nv, &I->ops[2]);
        break;
    case NV_MIN_U32: case NV_MIN_S32:
        nv_apnd(nv, "min%s ", I->op == NV_MIN_S32 ? ".s32" : ".u32");
        em_opnd(nv, &I->ops[0]); nv_apnd(nv, ", ");
        em_opnd(nv, &I->ops[1]); nv_apnd(nv, ", ");
        em_opnd(nv, &I->ops[2]);
        break;
    case NV_MAX_U32: case NV_MAX_S32:
        nv_apnd(nv, "max%s ", I->op == NV_MAX_S32 ? ".s32" : ".u32");
        em_opnd(nv, &I->ops[0]); nv_apnd(nv, ", ");
        em_opnd(nv, &I->ops[1]); nv_apnd(nv, ", ");
        em_opnd(nv, &I->ops[2]);
        break;

    /* ---- Control ---- */
    case NV_RET:
        nv_apnd(nv, "ret");
        break;
    case NV_EXIT:
        nv_apnd(nv, "exit");
        break;

    case NV_MOV_F64_LIT: {
        /* Reassemble 64-bit double from two 32-bit halves.
         * ops[1].imm = high 32, ops[2].imm = low 32.
         * Emits: mov.f64 %fdN, 0dXXXXXXXXXXXXXXXX */
        uint64_t hi = (uint32_t)I->ops[1].imm;
        uint64_t lo = (uint32_t)I->ops[2].imm;
        uint64_t bits = (hi << 32) | lo;
        nv_apnd(nv, "mov.f64 ");
        em_opnd(nv, &I->ops[0]);
        nv_apnd(nv, ", 0d%016llX", (unsigned long long)bits);
        break;
    }

    case NV_LEA_LOCAL: {
        /* Load address of .local variable + byte offset.
         * PTX local memory is NOT zero-based — you must reference
         * the declared __local symbol. Bare 0 crashes at launch.
         * ops[0] = dst u64, ops[1].imm = byte offset */
        int32_t off = I->ops[1].imm;
        nv_apnd(nv, "mov.u64 ");
        em_opnd(nv, &I->ops[0]);
        if (off == 0)
            nv_apnd(nv, ", __local");
        else
            nv_apnd(nv, ", __local+%d", off);
        break;
    }

    default:
        nv_apnd(nv, "/* unknown op %u */", I->op);
        break;
    }

    nv_apnd(nv, ";\n");
}

/* ---- Per-Function Emission ---- */

static void em_func(nv_module_t *nv, uint32_t fi)
{
    const nv_mfunc_t *MF = &nv->mfuncs[fi];
    const char *name = nv->bir->strings + MF->name;

    /* .maxntid directive for launch bounds */
    if (MF->launch_max > 0)
        nv_apnd(nv, ".maxntid %u\n", MF->launch_max);

    /* Entry or function header */
    if (MF->is_kern)
        nv_apnd(nv, ".entry %s (\n", name);
    else
        nv_apnd(nv, ".func %s (\n", name);

    /* .param list */
    for (uint32_t pi = 0; pi < MF->num_params; pi++) {
        const char *tsuf = rf_param(MF->params[pi].rfile);
        if (pi > 0) nv_apnd(nv, ",\n");
        nv_apnd(nv, "\t.param %s param%u", tsuf, pi);
    }
    if (nv->bkhit)
        nv_apnd(nv, ",\n\t.param .u64 __bkhit");
    nv_apnd(nv, "\n)\n");

    /* Function body */
    nv_apnd(nv, "{\n");

    /* .reg declarations — one per used register file */
    if (MF->rc[NV_RF_U32] > 1)
        nv_apnd(nv, "\t.reg .u32  %%r<%u>;\n",  MF->rc[NV_RF_U32]);
    if (MF->rc[NV_RF_U64] > 1)
        nv_apnd(nv, "\t.reg .u64  %%rd<%u>;\n", MF->rc[NV_RF_U64]);
    if (MF->rc[NV_RF_F32] > 1)
        nv_apnd(nv, "\t.reg .f32  %%f<%u>;\n",  MF->rc[NV_RF_F32]);
    if (MF->rc[NV_RF_F64] > 1)
        nv_apnd(nv, "\t.reg .f64  %%fd<%u>;\n", MF->rc[NV_RF_F64]);
    if (MF->rc[NV_RF_PRED] > 1)
        nv_apnd(nv, "\t.reg .pred %%p<%u>;\n",  MF->rc[NV_RF_PRED]);
    if (MF->rc[NV_RF_U16] > 1)
        nv_apnd(nv, "\t.reg .u16  %%rh<%u>;\n", MF->rc[NV_RF_U16]);
    if (MF->rc[NV_RF_F16] > 1)
        nv_apnd(nv, "\t.reg .f16  %%h<%u>;\n",  MF->rc[NV_RF_F16]);

    /* Local (stack) memory — without this declaration, ld.local/st.local
     * access unmapped memory and the driver gets very cross with us */
    if (MF->lcl_bytes > 0)
        nv_apnd(nv, "\t.local .align 8 .b8 __local[%u];\n", MF->lcl_bytes);

    /* Shared memory declaration */
    if (MF->lds_bytes > 0)
        nv_apnd(nv, "\t.shared .align 4 .b8 shmem[%u];\n", MF->lds_bytes);

    /* Block-hit instrumentation — load pointer, declare scratch regs.
     * Each block atomically increments bkhit[block_index].
     * Like putting a turnstile at every corridor junction
     * in a building you suspect has a ghost. */
    if (nv->bkhit) {
        nv_apnd(nv, "\t.reg .u64  %%rd_bk;\n");
        nv_apnd(nv, "\t.reg .u32  %%r_bk;\n");
        nv_apnd(nv, "\tld.param.u64 %%rd_bk, [__bkhit];\n");
    }

    nv_apnd(nv, "\n");

    /* Blocks and instructions */
    for (uint32_t bi = 0; bi < MF->num_blks; bi++) {
        const nv_mblk_t *MB = &nv->mblks[MF->first_blk + bi];
        uint32_t mbi = MF->first_blk + bi;

        nv_apnd(nv, "$BB%u:\n", mbi);

        /* Block-hit counter: atom.global.add bkhit[bi] */
        if (nv->bkhit)
            nv_apnd(nv, "\tatom.global.add.u32 %%r_bk, "
                    "[%%rd_bk+%u], 1;\n", bi * 4);

        int guard = 65536;
        for (uint32_t ii = 0; ii < MB->num_insts && guard > 0;
             ii++, guard--) {
            em_inst(nv, &nv->minsts[MB->first_inst + ii]);
        }
    }

    nv_apnd(nv, "}\n\n");
}

/* ---- Public API ---- */

int nv_emit_ptx(nv_module_t *nv, const char *path)
{
    nv->out_len = 0;

    /* PTX header — targeting Ada Lovelace (SM 8.9, PTX 8.0) */
    nv_apnd(nv,
        "// Generated by BarraCUDA - NVIDIA PTX backend\n"
        "// Open-source CUDA compilation: because NVCC said \"trust me, bro\"\n"
        "// and we said \"show us the source\"\n"
        "\n"
        ".version 8.0\n"
        ".target sm_89\n"
        ".address_size 64\n"
        "\n");

    /* Emit each function */
    int guard = 8192;
    for (uint32_t fi = 0; fi < nv->num_mfunc && guard > 0; fi++, guard--)
        em_func(nv, fi);

    /* Write to file */
    FILE *fp = fopen(path, "w");
    if (!fp) {
        fprintf(stderr, "error: cannot open '%s' for writing\n", path);
        return BC_ERR_IO;
    }
    fwrite(nv->out_buf, 1, nv->out_len, fp);
    fclose(fp);

    /* Stats */
    uint32_t nk = 0;
    int kg = 8192;
    for (uint32_t fi = 0; fi < nv->num_mfunc && kg > 0; fi++, kg--) {
        if (nv->mfuncs[fi].is_kern) nk++;
    }
    printf("wrote %s (%u bytes, %u kernel%s, %u instructions)\n",
           path, nv->out_len, nk, nk == 1 ? "" : "s", nv->num_minst);

    return BC_OK;
}
