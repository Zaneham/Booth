#include "nvidia.h"

#define NV_GBSYM "$bgbar"
#include "backend.h"
#include <stdio.h>
#include <string.h>
#include <stdarg.h>

/* PTX text emission. We generate polite ASCII that ptxas will JIT into
 * whatever SASS NVIDIA deems appropriate. Like writing a prayer and
 * sliding it under the cathedral door. */

/* ---- Output Buffer ---- */

static void nv_flsh(nv_module_t *nv)
{
    if (nv->out_fp == NULL || nv->out_len == 0) return;
    if (fwrite(nv->out_buf, 1, nv->out_len, nv->out_fp) != nv->out_len)
        nv->werr = 1;
    nv->out_len = 0;
}

static void nv_apnd(nv_module_t *nv, const char *fmt, ...)
{
    va_list ap;
    int try;

    if (nv->out_fp != NULL && nv->out_len > NV_MAX_OUT / 2)
        nv_flsh(nv);

    for (try = 0; try < 2; try++) {
        uint32_t room = NV_MAX_OUT - nv->out_len;
        int n = -1;
        if (nv->out_len < NV_MAX_OUT) {
            va_start(ap, fmt);
            n = vsnprintf(nv->out_buf + nv->out_len, room, fmt, ap);
            va_end(ap);
        }
        if (n >= 0 && (uint32_t)n < room) {
            nv->out_len += (uint32_t)n;
            nv->out_tot += (uint32_t)n;
            return;
        }
        if (nv->out_fp == NULL || nv->out_len == 0) break;
        nv_flsh(nv);
    }
    nv->out_buf[nv->out_len < NV_MAX_OUT ? nv->out_len : NV_MAX_OUT - 1] = 0;
    nv->ovf = 1;
}

static const char *asps(uint16_t f)
{
    return (f == NV_ASP_SHR) ? "shared." : (f == NV_ASP_GEN) ? "" : "global.";
}

static const char *atmn(uint16_t op)
{
    switch (op) {
    case NV_ATOM_ADD_U32:  return "add.u32";
    case NV_ATOM_ADD_F32:  return "add.f32";
    case NV_ATOM_ADD_U64:  return "add.u64";
    case NV_ATOM_ADD_F64:  return "add.f64";
    case NV_ATOM_MIN_U32:  return "min.u32";
    case NV_ATOM_MAX_U32:  return "max.u32";
    case NV_ATOM_AND_B32:  return "and.b32";
    case NV_ATOM_OR_B32:   return "or.b32";
    case NV_ATOM_XOR_B32:  return "xor.b32";
    case NV_ATOM_AND_B64:  return "and.b64";
    case NV_ATOM_OR_B64:   return "or.b64";
    case NV_ATOM_XOR_B64:  return "xor.b64";
    case NV_ATOM_XCHG_B32: return "exch.b32";
    case NV_ATOM_XCHG_B64: return "exch.b64";
    case NV_ATOM_CAS_B32:  return "cas.b32";
    case NV_ATOM_CAS_B64:  return "cas.b64";
    default:               return NULL;
    }
}

static int gsafe(char c)
{
    return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z')
        || (c >= '0' && c <= '9') || c == '_' || c == '$';
}

int nv_gsym(const bir_module_t *M, uint32_t gi, char *out, int size)
{
    if (out == NULL || size < 8) return 1;
    out[0] = '\0';
    if (M == NULL || gi >= M->num_globals) return 1;

    uint32_t noff = M->globals[gi].name;
    const char *nm = (noff < M->string_len) ? &M->strings[noff] : "g";

    int w = 0;
    if (!gsafe(nm[0]) || (nm[0] >= '0' && nm[0] <= '9')) out[w++] = '$';
    for (int i = 0; nm[i] != '\0' && w < size - 8; i++)
        out[w++] = gsafe(nm[i]) ? nm[i] : '_';
    out[w] = '\0';
    if (w == 0) { out[w++] = '$'; out[w++] = 'g'; out[w] = '\0'; }

    uint32_t dup = 0;
    for (uint32_t j = 0; j < gi; j++) {
        uint32_t o2 = M->globals[j].name;
        const char *n2 = (o2 < M->string_len) ? &M->strings[o2] : "g";
        if (strcmp(n2, nm) == 0) dup++;
    }
    if (dup > 0) {
        int n = snprintf(out + w, (size_t)(size - w), "$%u", (unsigned)dup);
        if (n < 0 || n >= size - w) { out[0] = '\0'; return 1; }
    }
    return 0;
}

const nv_mmash_t nv_mmash[NV_MMA_NSHAPE] = {
    { "m16n8k16.row.col.f32.f16.f16.f32",   8, 4 },
    { "m16n8k16.row.col.f32.bf16.bf16.f32", 8, 4 },
    { "m16n8k8.row.col.f32.f16.f16.f32",    4, 2 },
    { "m16n8k8.row.col.f32.bf16.bf16.f32",  4, 2 },
};

uint8_t nv_wmrf(uint32_t row)
{
    const bir_wmma_t *W = &bir_wmma[row % BIR_WM_NROW];
    return (uint8_t)((strcmp(W->abt, "f64") == 0) ? NV_RF_F64 : NV_RF_B32);
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
    case NV_SPEC_LANEID:   return "%laneid";
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
        case NV_RF_B32:  nv_apnd(nv, "%%rb%u", op->reg_num); break;
        default:         nv_apnd(nv, "%%r%u", op->reg_num);   break;
        }
        break;
    case NV_MOP_IMM:
        nv_apnd(nv, "%d", op->imm);
        break;
    case NV_MOP_LABEL:
        nv_apnd(nv, "$L%u_%u", (unsigned)nv->cur_emf, (unsigned)op->imm);
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
    case NV_RF_U16:  return ".b16";
    case NV_RF_F16:  return ".b16";
    default:         return ".u32";
    }
}

static const char *rf_ptyp(uint8_t rf)
{
    switch (rf) {
    case NV_RF_U64:  return ".u64";
    case NV_RF_F32:  return ".f32";
    case NV_RF_F64:  return ".f64";
    case NV_RF_U16:  return ".u16";
    case NV_RF_F16:  return ".b16";
    case NV_RF_B32:  return ".b32";
    default:         return ".u32";
    }
}

static const char *rf_bits(uint8_t rf)
{
    switch (rf) {
    case NV_RF_U64: case NV_RF_F64:  return ".b64";
    case NV_RF_U16: case NV_RF_F16:  return ".b16";
    default:                         return ".b32";
    }
}

static const char *nv_fnam(const nv_module_t *nv, const nv_call_t *C)
{
    if (C->vprt) return "vprintf";
    if (C->fn >= nv->bir->num_funcs) return "$booth_unnamed_func";
    if (nv->bir->funcs[C->fn].name >= nv->bir->string_len)
        return "$booth_unnamed_func";
    return nv->bir->strings + nv->bir->funcs[C->fn].name;
}

/* ---- Per-Instruction Emission ---- */

static void em_wtup(nv_module_t *nv, const nv_opnd_t *base, uint8_t n)
{
    nv_apnd(nv, "{");
    for (uint8_t k = 0; k < n; k++) {
        nv_opnd_t o = *base;
        o.reg_num = (uint16_t)(o.reg_num + k);
        if (k > 0) nv_apnd(nv, ", ");
        em_opnd(nv, &o);
    }
    nv_apnd(nv, "}");
}

static void em_wld(nv_module_t *nv, const nv_minst_t *I)
{
    uint32_t r = NV_WM_ROW(I->flags);
    const bir_wmma_t *W = &bir_wmma[r % BIR_WM_NROW];
    unsigned role = NV_WM_ROLE(I->flags);
    const char *rn = (role == NV_WM_ROLE_B) ? "b"
                   : (role == NV_WM_ROLE_C) ? "c" : "a";
    const char *ty = (role == NV_WM_ROLE_C) ? W->act : W->abt;

    nv_apnd(nv, "wmma.load.%s.sync.aligned.%s.%s.%s ", rn,
            NV_WM_ALAY(I->flags) ? "col" : "row", W->shp, ty);
    em_wtup(nv, &I->ops[0], bir_wmn(r, role));
    nv_apnd(nv, ", [");
    em_opnd(nv, &I->ops[1]);
    nv_apnd(nv, "], ");
    em_opnd(nv, &I->ops[2]);
}

static void em_wst(nv_module_t *nv, const nv_minst_t *I)
{
    uint32_t r = NV_WM_ROW(I->flags);
    const bir_wmma_t *W = &bir_wmma[r % BIR_WM_NROW];

    nv_apnd(nv, "wmma.store.d.sync.aligned.%s.%s.%s [",
            NV_WM_ALAY(I->flags) ? "col" : "row", W->shp, W->act);
    em_opnd(nv, &I->ops[0]);
    nv_apnd(nv, "], ");
    em_wtup(nv, &I->ops[1], W->nc);
    nv_apnd(nv, ", ");
    em_opnd(nv, &I->ops[2]);
}

static void em_wmma(nv_module_t *nv, const nv_minst_t *I)
{
    uint32_t r = NV_WM_ROW(I->flags);
    const bir_wmma_t *W = &bir_wmma[r % BIR_WM_NROW];
    const char *al = NV_WM_ALAY(I->flags) ? "col" : "row";
    const char *bl = NV_WM_BLAY(I->flags) ? "col" : "row";

    if (strcmp(W->abt, "b1") == 0)
        nv_apnd(nv, "wmma.mma.%s.popc.sync.aligned.%s.%s.%s.s32.b1.b1.s32 ",
                NV_WM_BOP(I->flags) ? "and" : "xor", al, bl, W->shp);
    else if (strcmp(W->abt, "f16") == 0)
        nv_apnd(nv, "wmma.mma.sync.aligned.%s.%s.%s.%s.%s ",
                al, bl, W->shp, W->act, W->act);
    else
        nv_apnd(nv, "wmma.mma.sync.aligned.%s.%s.%s.%s.%s.%s.%s ",
                al, bl, W->shp, W->act, W->abt, W->abt, W->act);
    em_wtup(nv, &I->ops[0], W->nc);
    nv_apnd(nv, ", ");
    em_wtup(nv, &I->ops[1], W->na);
    nv_apnd(nv, ", ");
    em_wtup(nv, &I->ops[2], W->nb);
    nv_apnd(nv, ", ");
    em_wtup(nv, &I->ops[3], W->nc);
}

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
    case NV_MUL_HI_U64:
        nv_apnd(nv, "mul.hi.u64 ");
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
    /* popc and clz both hand back a .u32 count whatever they counted over */
    case NV_POPC_B32: case NV_POPC_B64:
        nv_apnd(nv, "popc%s ", I->op == NV_POPC_B64 ? ".b64" : ".b32");
        em_opnd(nv, &I->ops[0]); nv_apnd(nv, ", ");
        em_opnd(nv, &I->ops[1]);
        break;
    case NV_CLZ_B32: case NV_CLZ_B64:
        nv_apnd(nv, "clz%s ", I->op == NV_CLZ_B64 ? ".b64" : ".b32");
        em_opnd(nv, &I->ops[0]); nv_apnd(nv, ", ");
        em_opnd(nv, &I->ops[1]);
        break;
    case NV_BREV_B32: case NV_BREV_B64:
        nv_apnd(nv, "brev%s ", I->op == NV_BREV_B64 ? ".b64" : ".b32");
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
    case NV_LD_GLB_U8:  case NV_LD_GLB_U16: case NV_LD_GLB_B16: {
        const char *tsuf;
        switch (I->op) {
        case NV_LD_GLB_U64: tsuf = ".u64"; break;
        case NV_LD_GLB_F32: tsuf = ".f32"; break;
        case NV_LD_GLB_F64: tsuf = ".f64"; break;
        case NV_LD_GLB_U8:  tsuf = ".u8";  break;
        case NV_LD_GLB_U16: tsuf = ".u16"; break;
        case NV_LD_GLB_B16: tsuf = ".b16"; break;
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
    case NV_ST_GLB_U8:  case NV_ST_GLB_U16: case NV_ST_GLB_B16: {
        const char *tsuf;
        switch (I->op) {
        case NV_ST_GLB_U64: tsuf = ".u64"; break;
        case NV_ST_GLB_F32: tsuf = ".f32"; break;
        case NV_ST_GLB_F64: tsuf = ".f64"; break;
        case NV_ST_GLB_U8:  tsuf = ".u8";  break;
        case NV_ST_GLB_U16: tsuf = ".u16"; break;
        case NV_ST_GLB_B16: tsuf = ".b16"; break;
        default:            tsuf = ".u32"; break;
        }
        nv_apnd(nv, "st.global%s [", tsuf);
        em_opnd(nv, &I->ops[0]); nv_apnd(nv, "], ");
        em_opnd(nv, &I->ops[1]);
        break;
    }

    /* ---- Loads/Stores: Shared ---- */
    case NV_LD_SHR_U32: case NV_LD_SHR_F32: case NV_LD_SHR_U8:
    case NV_LD_SHR_U16: case NV_LD_SHR_B16:
    case NV_LD_SHR_U64: case NV_LD_SHR_F64: {
        const char *tsuf = (I->op == NV_LD_SHR_F32) ? ".f32"
                         : (I->op == NV_LD_SHR_U16) ? ".u16"
                         : (I->op == NV_LD_SHR_B16) ? ".b16"
                         : (I->op == NV_LD_SHR_U64) ? ".u64"
                         : (I->op == NV_LD_SHR_F64) ? ".f64"
                         : (I->op == NV_LD_SHR_U8)  ? ".u8" : ".u32";
        nv_apnd(nv, "ld.shared%s ", tsuf);
        em_opnd(nv, &I->ops[0]); nv_apnd(nv, ", [");
        em_opnd(nv, &I->ops[1]); nv_apnd(nv, "]");
        break;
    }
    case NV_ST_SHR_U32: case NV_ST_SHR_F32: case NV_ST_SHR_U8:
    case NV_ST_SHR_U16: case NV_ST_SHR_B16:
    case NV_ST_SHR_U64: case NV_ST_SHR_F64: {
        const char *tsuf = (I->op == NV_ST_SHR_F32) ? ".f32"
                         : (I->op == NV_ST_SHR_U16) ? ".u16"
                         : (I->op == NV_ST_SHR_B16) ? ".b16"
                         : (I->op == NV_ST_SHR_U64) ? ".u64"
                         : (I->op == NV_ST_SHR_F64) ? ".f64"
                         : (I->op == NV_ST_SHR_U8)  ? ".u8" : ".u32";
        nv_apnd(nv, "st.shared%s [", tsuf);
        em_opnd(nv, &I->ops[0]); nv_apnd(nv, "], ");
        em_opnd(nv, &I->ops[1]);
        break;
    }

    /* ---- Loads/Stores: Local ---- */
    case NV_LD_LOC_U32: case NV_LD_LOC_U64:
    case NV_LD_LOC_F32: case NV_LD_LOC_F64: case NV_LD_LOC_U8:
    case NV_LD_LOC_U16: case NV_LD_LOC_B16: {
        const char *tsuf;
        switch (I->op) {
        case NV_LD_LOC_U64: tsuf = ".u64"; break;
        case NV_LD_LOC_F32: tsuf = ".f32"; break;
        case NV_LD_LOC_F64: tsuf = ".f64"; break;
        case NV_LD_LOC_U8:  tsuf = ".u8";  break;
        case NV_LD_LOC_U16: tsuf = ".u16"; break;
        case NV_LD_LOC_B16: tsuf = ".b16"; break;
        default:            tsuf = ".u32"; break;
        }
        nv_apnd(nv, "ld.local%s ", tsuf);
        em_opnd(nv, &I->ops[0]); nv_apnd(nv, ", [");
        em_opnd(nv, &I->ops[1]); nv_apnd(nv, "]");
        break;
    }
    case NV_ST_LOC_U32: case NV_ST_LOC_U64:
    case NV_ST_LOC_F32: case NV_ST_LOC_F64: case NV_ST_LOC_U8:
    case NV_ST_LOC_U16: case NV_ST_LOC_B16: {
        const char *tsuf;
        switch (I->op) {
        case NV_ST_LOC_U64: tsuf = ".u64"; break;
        case NV_ST_LOC_F32: tsuf = ".f32"; break;
        case NV_ST_LOC_F64: tsuf = ".f64"; break;
        case NV_ST_LOC_U8:  tsuf = ".u8";  break;
        case NV_ST_LOC_U16: tsuf = ".u16"; break;
        case NV_ST_LOC_B16: tsuf = ".b16"; break;
        default:            tsuf = ".u32"; break;
        }
        nv_apnd(nv, "st.local%s [", tsuf);
        em_opnd(nv, &I->ops[0]); nv_apnd(nv, "], ");
        em_opnd(nv, &I->ops[1]);
        break;
    }

    /* ---- Parameter Loads ---- */
    case NV_LD_PARAM_U32: case NV_LD_PARAM_U64:
    case NV_LD_PARAM_F32: case NV_LD_PARAM_F64:
    case NV_LD_PARAM_B16: {
        const char *tsuf;
        switch (I->op) {
        case NV_LD_PARAM_U64: tsuf = ".u64"; break;
        case NV_LD_PARAM_F32: tsuf = ".f32"; break;
        case NV_LD_PARAM_F64: tsuf = ".f64"; break;
        case NV_LD_PARAM_B16: tsuf = ".b16"; break;
        default:              tsuf = ".u32"; break;
        }
        nv_apnd(nv, "ld.param%s ", tsuf);
        em_opnd(nv, &I->ops[0]); nv_apnd(nv, ", [param%d]",
                I->ops[1].imm);
        break;
    }

    /* ---- Atomics ---- */
    case NV_ATOM_ADD_U32:  case NV_ATOM_ADD_F32:
    case NV_ATOM_ADD_U64:  case NV_ATOM_ADD_F64:
    case NV_ATOM_MIN_U32:  case NV_ATOM_MAX_U32:
    case NV_ATOM_AND_B32:  case NV_ATOM_OR_B32:   case NV_ATOM_XOR_B32:
    case NV_ATOM_AND_B64:  case NV_ATOM_OR_B64:   case NV_ATOM_XOR_B64:
    case NV_ATOM_XCHG_B32: case NV_ATOM_XCHG_B64:
    case NV_ATOM_CAS_B32:  case NV_ATOM_CAS_B64: {
        const char *mn = atmn(I->op);
        if (!mn) {
            nv->badop = 1;
            (void)be_fail(BC_E544, "nvptx", (unsigned)I->op);
            break;
        }
        nv_apnd(nv, "atom.%s%s ", asps(I->flags), mn);
        em_opnd(nv, &I->ops[0]); nv_apnd(nv, ", [");
        em_opnd(nv, &I->ops[1]); nv_apnd(nv, "], ");
        em_opnd(nv, &I->ops[2]);
        if (I->op == NV_ATOM_CAS_B32 || I->op == NV_ATOM_CAS_B64) {
            nv_apnd(nv, ", ");
            if (I->num_uses > 2)
                em_opnd(nv, &I->ops[3]);
            else
                nv_apnd(nv, "0");
        }
        break;
    }

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
    case NV_MEMBAR: {
        /* per PTX ISA 9.2 9.7.13.4, levels cta, gl and sys */
        static const char *const lvl[3] = { "cta", "gl", "sys" };
        nv_apnd(nv, "membar.%s", lvl[I->flags % 3u]);
        break;
    }
    case NV_NANOSLP:
        nv_apnd(nv, "nanosleep.u32 ");
        em_opnd(nv, &I->ops[0]);
        break;
    case NV_BARRED_OR: case NV_BARRED_AND: case NV_BARRED_POPC: {
        /* per PTX ISA 9.2 9.7.13.1, bar.red.op.pred d, a, c */
        const char *rd = (I->op == NV_BARRED_AND) ? "and.pred"
                       : (I->op == NV_BARRED_POPC) ? "popc.u32" : "or.pred";
        nv_apnd(nv, "bar.red.%s ", rd);
        em_opnd(nv, &I->ops[0]); nv_apnd(nv, ", 0, ");
        em_opnd(nv, &I->ops[1]);
        break;
    }
    case NV_CVTA_GLB: case NV_CVTA_LOC:
        nv_apnd(nv, "cvta.%s.u64 ",
                I->op == NV_CVTA_GLB ? "global" : "local");
        em_opnd(nv, &I->ops[0]); nv_apnd(nv, ", ");
        em_opnd(nv, &I->ops[1]);
        break;
    case NV_CVT_U32_U16:
        nv_apnd(nv, "cvt.u32.u16 ");
        em_opnd(nv, &I->ops[0]); nv_apnd(nv, ", ");
        em_opnd(nv, &I->ops[1]);
        break;
    case NV_CVT_F32_BF16:
        nv_apnd(nv, "cvt.f32.bf16 ");
        em_opnd(nv, &I->ops[0]); nv_apnd(nv, ", ");
        em_opnd(nv, &I->ops[1]);
        break;
    case NV_CVT_BF16_F32:
        nv_apnd(nv, "cvt.rn.bf16.f32 ");
        em_opnd(nv, &I->ops[0]); nv_apnd(nv, ", ");
        em_opnd(nv, &I->ops[1]);
        break;
    case NV_CVT_S32_S16:
        nv_apnd(nv, "cvt.s32.s16 ");
        em_opnd(nv, &I->ops[0]); nv_apnd(nv, ", ");
        em_opnd(nv, &I->ops[1]);
        break;
    case NV_CVT_U16_U32:
        nv_apnd(nv, "cvt.u16.u32 ");
        em_opnd(nv, &I->ops[0]); nv_apnd(nv, ", ");
        em_opnd(nv, &I->ops[1]);
        break;
    case NV_CVT_F32_U64:
        nv_apnd(nv, "cvt.rn.f32.u64 ");
        em_opnd(nv, &I->ops[0]); nv_apnd(nv, ", ");
        em_opnd(nv, &I->ops[1]);
        break;
    case NV_CVT_F32_S64:
        nv_apnd(nv, "cvt.rn.f32.s64 ");
        em_opnd(nv, &I->ops[0]); nv_apnd(nv, ", ");
        em_opnd(nv, &I->ops[1]);
        break;
    case NV_CVT_U64_F32:
        nv_apnd(nv, "cvt.rzi.u64.f32 ");
        em_opnd(nv, &I->ops[0]); nv_apnd(nv, ", ");
        em_opnd(nv, &I->ops[1]);
        break;
    case NV_CVT_S64_F32:
        nv_apnd(nv, "cvt.rzi.s64.f32 ");
        em_opnd(nv, &I->ops[0]); nv_apnd(nv, ", ");
        em_opnd(nv, &I->ops[1]);
        break;
    case NV_CVT_F16_F64:
        nv_apnd(nv, "cvt.rn.f16.f64 ");
        em_opnd(nv, &I->ops[0]); nv_apnd(nv, ", ");
        em_opnd(nv, &I->ops[1]);
        break;
    case NV_CVT_F64_F16:
        nv_apnd(nv, "cvt.f64.f16 ");
        em_opnd(nv, &I->ops[0]); nv_apnd(nv, ", ");
        em_opnd(nv, &I->ops[1]);
        break;
    case NV_MOV_B16: case NV_MOV_B32: case NV_MOV_B64:
        nv_apnd(nv, "mov%s ", I->op == NV_MOV_B64 ? ".b64"
                            : I->op == NV_MOV_B16 ? ".b16" : ".b32");
        em_opnd(nv, &I->ops[0]); nv_apnd(nv, ", ");
        em_opnd(nv, &I->ops[1]);
        break;
    case NV_NEG_S64:
        nv_apnd(nv, "neg.s64 ");
        em_opnd(nv, &I->ops[0]); nv_apnd(nv, ", ");
        em_opnd(nv, &I->ops[1]);
        break;
    case NV_SHR_S64:
        nv_apnd(nv, "shr.s64 ");
        em_opnd(nv, &I->ops[0]); nv_apnd(nv, ", ");
        em_opnd(nv, &I->ops[1]); nv_apnd(nv, ", ");
        em_opnd(nv, &I->ops[2]);
        break;
    case NV_DIV_U64: case NV_DIV_S64:
    case NV_REM_U64: case NV_REM_S64: {
        const char *mn;
        switch (I->op) {
        case NV_DIV_U64: mn = "div.u64"; break;
        case NV_DIV_S64: mn = "div.s64"; break;
        case NV_REM_U64: mn = "rem.u64"; break;
        default:         mn = "rem.s64"; break;
        }
        nv_apnd(nv, "%s ", mn);
        em_opnd(nv, &I->ops[0]); nv_apnd(nv, ", ");
        em_opnd(nv, &I->ops[1]); nv_apnd(nv, ", ");
        em_opnd(nv, &I->ops[2]);
        break;
    }
    case NV_MIN_U64: case NV_MAX_U64:
    case NV_MIN_S64: case NV_MAX_S64: {
        const char *mn;
        switch (I->op) {
        case NV_MIN_U64: mn = "min.u64"; break;
        case NV_MAX_U64: mn = "max.u64"; break;
        case NV_MIN_S64: mn = "min.s64"; break;
        default:         mn = "max.s64"; break;
        }
        nv_apnd(nv, "%s ", mn);
        em_opnd(nv, &I->ops[0]); nv_apnd(nv, ", ");
        em_opnd(nv, &I->ops[1]); nv_apnd(nv, ", ");
        em_opnd(nv, &I->ops[2]);
        break;
    }
    case NV_SETP_LT_S64: case NV_SETP_LE_S64:
    case NV_SETP_GT_S64: case NV_SETP_GE_S64:
    case NV_SETP_LT_U64: case NV_SETP_LE_U64:
    case NV_SETP_GT_U64: case NV_SETP_GE_U64: {
        const char *cmp, *ty;
        switch (I->op) {
        case NV_SETP_LT_S64: cmp = "lt"; ty = "s64"; break;
        case NV_SETP_LE_S64: cmp = "le"; ty = "s64"; break;
        case NV_SETP_GT_S64: cmp = "gt"; ty = "s64"; break;
        case NV_SETP_GE_S64: cmp = "ge"; ty = "s64"; break;
        case NV_SETP_LT_U64: cmp = "lt"; ty = "u64"; break;
        case NV_SETP_LE_U64: cmp = "le"; ty = "u64"; break;
        case NV_SETP_GT_U64: cmp = "gt"; ty = "u64"; break;
        default:             cmp = "ge"; ty = "u64"; break;
        }
        nv_apnd(nv, "setp.%s.%s ", cmp, ty);
        em_opnd(nv, &I->ops[0]); nv_apnd(nv, ", ");
        em_opnd(nv, &I->ops[1]); nv_apnd(nv, ", ");
        em_opnd(nv, &I->ops[2]);
        break;
    }
    case NV_ST_RETP:
        nv_apnd(nv, "st.param%s [func_retval0], ", rf_ptyp((uint8_t)I->flags));
        em_opnd(nv, &I->ops[0]);
        break;
    case NV_CALL: {
        const nv_call_t *C = &nv->calls[I->flags % NV_MAX_CALL];
        const char *nm = nv_fnam(nv, C);
        uint8_t k;
        nv_apnd(nv, "{\n");
        for (k = 0; k < C->nargs && k < NV_MAX_CARG; k++) {
            nv_apnd(nv, "\t.param %s cp%u;\n",
                    rf_bits(C->args[k].rfile), (unsigned)k);
            nv_apnd(nv, "\tst.param%s [cp%u], ",
                    rf_ptyp(C->args[k].rfile), (unsigned)k);
            em_opnd(nv, &C->args[k]);
            nv_apnd(nv, ";\n");
        }
        if (C->hasret)
            nv_apnd(nv, "\t.param %s cr;\n", rf_bits(C->retrf));
        nv_apnd(nv, "\tcall.uni ");
        if (C->hasret) nv_apnd(nv, "(cr), ");
        nv_apnd(nv, "%s", nm);
        if (C->nargs > 0) {
            nv_apnd(nv, ", (");
            for (k = 0; k < C->nargs && k < NV_MAX_CARG; k++)
                nv_apnd(nv, k == 0 ? "cp%u" : ", cp%u", (unsigned)k);
            nv_apnd(nv, ")");
        }
        nv_apnd(nv, ";\n");
        if (C->hasret) {
            nv_apnd(nv, "\tld.param%s ", rf_ptyp(C->retrf));
            em_opnd(nv, &C->ret);
            nv_apnd(nv, ", [cr];\n");
        }
        nv_apnd(nv, "\t}\n");
        return;
    }

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

    case NV_LEA_DSH:
        nv_apnd(nv, "mov.u64 ");
        em_opnd(nv, &I->ops[0]);
        nv_apnd(nv, ", __dynshmem");
        break;

    case NV_LEA_GLB: {
        char sym[NV_SYM_MAX];
        const char *nm = sym;
        uint32_t gi = (uint32_t)I->ops[1].imm;
        if (nv_gsym(nv->bir, gi, sym, (int)sizeof sym) != 0)
            nm = "$booth_unnamed_global";
        nv_apnd(nv, "mov.u64 ");
        em_opnd(nv, &I->ops[0]);
        nv_apnd(nv, ", %s", nm);
        break;
    }

    case NV_MOV_PK16:
        nv_apnd(nv, "mov.b32 ");
        em_opnd(nv, &I->ops[0]); nv_apnd(nv, ", {");
        em_opnd(nv, &I->ops[1]); nv_apnd(nv, ", ");
        em_opnd(nv, &I->ops[2]); nv_apnd(nv, "}");
        break;
    case NV_WLD:
        em_wld(nv, I);
        break;
    case NV_WST:
        em_wst(nv, I);
        break;
    case NV_WMMA:
        em_wmma(nv, I);
        break;
    case NV_ASM: {
        const nv_asm_t *A = &nv->asms[I->flags % NV_MAX_ASM];
        const char *t = nv->bir->strings + A->tmpl;
        uint32_t k = 0;
        KA_GUARD(g, BIR_ASM_TMPLZ);
        while (g-- && t[k]) {
            if (t[k] == '%' && t[k + 1] == '%') {
                nv_apnd(nv, "%%");
                k += 2;
            } else if (t[k] == '%' && t[k + 1] >= '0' && t[k + 1] <= '9') {
                uint32_t d = 0, h = 0;
                k++;
                while (h++ < 3 && t[k] >= '0' && t[k] <= '9')
                    d = d * 10u + (uint32_t)(t[k++] - '0');
                if (d < A->nops) em_opnd(nv, &A->ops[d]);
                else nv_apnd(nv, "%%%u", d);
            } else {
                nv_apnd(nv, "%c", t[k++]);
            }
        }
        nv_apnd(nv, "\n");
        return;
    }
    case NV_GBAR: {
        unsigned n = nv->gbn++;
        nv_apnd(nv,
            "// grid barrier: holds only while every block is resident,\n"
            "\t// which is what cudaLaunchCooperativeKernel promises.\n"
            "\t// a plain launch that overfills the device hangs here.\n");
        nv_apnd(nv, "\tbar.sync 0;\n");
        nv_apnd(nv,
            "\tmov.u32 %%r_gb0, %%tid.x;\n"
            "\tmov.u32 %%r_gb1, %%tid.y;\n"
            "\tor.b32  %%r_gb0, %%r_gb0, %%r_gb1;\n"
            "\tmov.u32 %%r_gb1, %%tid.z;\n"
            "\tor.b32  %%r_gb0, %%r_gb0, %%r_gb1;\n"
            "\tsetp.ne.u32 %%p_gb0, %%r_gb0, 0;\n");
        nv_apnd(nv, "\t@%%p_gb0 bra $GBW%u;\n", n);
        nv_apnd(nv, "\tmov.u64 %%rd_gb, %s;\n", NV_GBSYM);
        nv_apnd(nv,  /* per PTX ISA 9.2 9.7.13.4, .gpu is device scope */
            "\tld.acquire.gpu.global.u32 %%r_gb2, [%%rd_gb+4];\n"
            "\tmov.u32 %%r_gb0, %%nctaid.x;\n"
            "\tmov.u32 %%r_gb1, %%nctaid.y;\n"
            "\tmul.lo.u32 %%r_gb0, %%r_gb0, %%r_gb1;\n"
            "\tmov.u32 %%r_gb1, %%nctaid.z;\n"
            "\tmul.lo.u32 %%r_gb0, %%r_gb0, %%r_gb1;\n"
            "\tsub.u32 %%r_gb0, %%r_gb0, 1;\n");
        nv_apnd(nv,  /* per PTX ISA 9.2 9.7.13.5, atom takes .sem */
            "\tatom.acq_rel.gpu.global.add.u32 %%r_gb1, [%%rd_gb], 1;\n"
            "\tsetp.ne.u32 %%p_gb0, %%r_gb1, %%r_gb0;\n");
        nv_apnd(nv, "\t@%%p_gb0 bra $GBS%u;\n", n);
        nv_apnd(nv,
            "\tst.relaxed.gpu.global.u32 [%%rd_gb], 0;\n"
            "\tatom.release.gpu.global.add.u32 %%r_gb1, [%%rd_gb+4], 1;\n");
        nv_apnd(nv, "\tbra $GBW%u;\n", n);
        nv_apnd(nv, "$GBS%u:\n", n);
        nv_apnd(nv,
            "\tld.acquire.gpu.global.u32 %%r_gb1, [%%rd_gb+4];\n"
            "\tsetp.eq.u32 %%p_gb0, %%r_gb1, %%r_gb2;\n");
        nv_apnd(nv, "\t@%%p_gb0 bra $GBS%u;\n", n);
        nv_apnd(nv, "$GBW%u:\n", n);
        nv_apnd(nv, "\tbar.sync 0;\n");
        return;
    }
    case NV_BARWARP:
        nv_apnd(nv, "bar.warp.sync 0xffffffff");
        break;
    case NV_TRAP:
        nv_apnd(nv, "trap");
        break;
    case NV_MMA: {
        const nv_mmash_t *sh = &nv_mmash[I->flags % NV_MMA_NSHAPE];
        uint8_t nreg[4];
        nreg[0] = 4; nreg[1] = (uint8_t)(sh->na / 2);
        nreg[2] = (uint8_t)(sh->nb / 2); nreg[3] = 4;
        nv_apnd(nv, "mma.sync.aligned.%s ", sh->sfx);
        for (uint8_t g = 0; g < 4; g++) {
            if (g > 0) nv_apnd(nv, ", ");
            nv_apnd(nv, "{");
            for (uint8_t k = 0; k < nreg[g]; k++) {
                nv_opnd_t o = I->ops[g];
                o.reg_num = (uint16_t)(o.reg_num + k);
                if (k > 0) nv_apnd(nv, ", ");
                em_opnd(nv, &o);
            }
            nv_apnd(nv, "}");
        }
        break;
    }
    default:
        nv->badop = 1;
        (void)be_fail(BC_E544, "nvptx", (unsigned)I->op);
        break;
    }

    nv_apnd(nv, ";\n");
}

/* ---- Per-Function Emission ---- */

static void em_fhdr(nv_module_t *nv, const nv_mfunc_t *MF)
{
    const char *name = nv->bir->strings + MF->name;

    if (MF->is_kern) {
        nv_apnd(nv, ".entry %s (\n", name);
    } else if (MF->hasret) {
        nv_apnd(nv, ".func (.param %s func_retval0) %s (\n",
                rf_bits(MF->retrf), name);
    } else {
        nv_apnd(nv, ".func %s (\n", name);
    }

    for (uint32_t pi = 0; pi < MF->num_params; pi++) {
        const char *tsuf = MF->is_kern ? rf_param(MF->params[pi].rfile)
                                       : rf_bits(MF->params[pi].rfile);
        if (pi > 0) nv_apnd(nv, ",\n");
        nv_apnd(nv, "\t.param %s param%u", tsuf, pi);
    }
    if (nv->bkhit && MF->is_kern)
        nv_apnd(nv, ",\n\t.param .u64 __bkhit");
    nv_apnd(nv, "\n)");
}

static void em_fdecl(nv_module_t *nv, uint32_t fi)
{
    const nv_mfunc_t *MF = &nv->mfuncs[fi];
    if (MF->is_kern) return;
    em_fhdr(nv, MF);
    nv_apnd(nv, ";\n\n");
}

static int em_hasgb(const nv_module_t *nv, const nv_mfunc_t *MF)
{
    if (!nv->gbar) return 0;
    for (uint32_t bi = 0; bi < MF->num_blks; bi++) {
        const nv_mblk_t *MB = &nv->mblks[MF->first_blk + bi];
        for (uint32_t ii = 0; ii < MB->num_insts; ii++)
            if (nv->minsts[MB->first_inst + ii].op == NV_GBAR)
                return 1;
    }
    return 0;
}

static void em_func(nv_module_t *nv, uint32_t fi)
{
    const nv_mfunc_t *MF = &nv->mfuncs[fi];

    nv->cur_emf = fi;

    em_fhdr(nv, MF);
    nv_apnd(nv, "\n");

    /* per PTX ISA 9.2 11.4.2 .maxntid sits between .entry and its body */
    if (MF->launch_max > 0 && MF->is_kern)
        nv_apnd(nv, ".maxntid %u\n", MF->launch_max);

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
        nv_apnd(nv, "\t.reg .b16  %%rh<%u>;\n", MF->rc[NV_RF_U16]);
    if (MF->rc[NV_RF_F16] > 1)
        nv_apnd(nv, "\t.reg .f16  %%h<%u>;\n",  MF->rc[NV_RF_F16]);
    if (MF->rc[NV_RF_B32] > 1)
        nv_apnd(nv, "\t.reg .b32  %%rb<%u>;\n", MF->rc[NV_RF_B32]);

    /* Local (stack) memory — without this declaration, ld.local/st.local
     * access unmapped memory and the driver gets very cross with us */
    if (MF->lcl_bytes > 0)
        nv_apnd(nv, "\t.local .align %u .b8 __local[%u];\n",
                MF->lcl_alg ? MF->lcl_alg : 8u, MF->lcl_bytes);

    /* Shared memory declaration */
    if (MF->lds_bytes > 0)
        nv_apnd(nv, "\t.shared .align %u .b8 shmem[%u];\n",
                MF->lds_alg ? MF->lds_alg : 4u, MF->lds_bytes);

    /* Block-hit instrumentation — load pointer, declare scratch regs.
     * Each block atomically increments bkhit[block_index].
     * Like putting a turnstile at every corridor junction
     * in a building you suspect has a ghost. */
    if (em_hasgb(nv, MF)) {
        nv_apnd(nv, "\t.reg .u32  %%r_gb0;\n");
        nv_apnd(nv, "\t.reg .u32  %%r_gb1;\n");
        nv_apnd(nv, "\t.reg .u32  %%r_gb2;\n");
        nv_apnd(nv, "\t.reg .u64  %%rd_gb;\n");
        nv_apnd(nv, "\t.reg .pred %%p_gb0;\n");
    }

    if (nv->bkhit && MF->is_kern) {
        nv_apnd(nv, "\t.reg .u64  %%rd_bk;\n");
        nv_apnd(nv, "\t.reg .u32  %%r_bk;\n");
        nv_apnd(nv, "\tld.param.u64 %%rd_bk, [__bkhit];\n");
    }

    nv_apnd(nv, "\n");

    /* Blocks and instructions */
    for (uint32_t bi = 0; bi < MF->num_blks; bi++) {
        const nv_mblk_t *MB = &nv->mblks[MF->first_blk + bi];
        uint32_t mbi = MF->first_blk + bi;

        nv_apnd(nv, "$L%u_%u:\n", fi, mbi);

        /* Block-hit counter: atom.global.add bkhit[bi] */
        if (nv->bkhit && MF->is_kern)
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

static int em_gdata(nv_module_t *nv, uint32_t gi, uint32_t nb)
{
    const bir_module_t *M = nv->bir;
    uint32_t init = M->globals[gi].initializer;
    if (init == BIR_VAL_NONE || !BIR_VAL_IS_CONST(init)) return 0;
    uint32_t ci = BIR_VAL_INDEX(init);
    if (ci >= M->num_consts) return 1;
    if (M->consts[ci].kind != BIR_CONST_BYTES) return 0;

    uint32_t off = M->consts[ci].d.bytes.off;
    uint32_t len = M->consts[ci].d.bytes.len;
    if (off >= M->string_len) return 1;
    if (len > M->string_len - off) return 1;
    if (len > nb) return 1;

    nv_apnd(nv, " = {");
    for (uint32_t i = 0; i < nb; i++) {
        unsigned char b = (i < len) ? (unsigned char)M->strings[off + i] : 0u;
        nv_apnd(nv, i == 0 ? "%u" : ", %u", (unsigned)b);
    }
    nv_apnd(nv, "}");
    return 0;
}

static int em_globs(nv_module_t *nv)
{
    const bir_module_t *M = nv->bir;
    uint32_t n = M->num_globals;
    if (n > BIR_MAX_GLOBALS) return 1;
    if (n == 0) return 0;

    for (uint32_t gi = 0; gi < n; gi++) {
        char sym[NV_SYM_MAX];
        uint32_t nb = bir_bsz(M, M->globals[gi].type, 8);
        if (nb == 0) continue;
        if (nv_gsym(M, gi, sym, (int)sizeof sym) != 0) return 1;

        const char *sp = (M->globals[gi].addrspace == BIR_AS_SHARED)
                       ? ".shared" : ".global";
        nv_apnd(nv, "%s .align 8 .b8 %s[%u]", sp, sym, (unsigned)nb);
        if (em_gdata(nv, gi, nb) != 0) return 1;
        nv_apnd(nv, ";\n");
    }
    nv_apnd(nv, "\n");
    return 0;
}

/* ---- Public API ---- */

static int nv_bail(nv_module_t *nv, const char *path, int rc)
{
    if (nv->out_fp != NULL) {
        fclose(nv->out_fp);
        nv->out_fp = NULL;
    }
    remove(path);
    return rc;
}

static int em_pre(nv_module_t *nv)
{
    /* PTX header — targeting Ada Lovelace (SM 8.9, PTX 8.0) */
    nv_apnd(nv,
        "// Generated by Booth - NVIDIA PTX backend\n"
        "// Open-source CUDA compilation: because NVCC said \"trust me, bro\"\n"
        "// and we said \"show us the source\"\n"
        "\n"
        ".version 8.0\n"
        ".target sm_89\n"
        ".address_size 64\n"
        "\n");

    if (nv->dyn_alg)
        nv_apnd(nv, ".extern .shared .align %u .b8 __dynshmem[];\n\n",
                nv->dyn_alg);

    if (nv->gbar)
        nv_apnd(nv, ".global .align 4 .u32 %s[2] = {0, 0};\n\n",
                NV_GBSYM);

    if (em_globs(nv) != 0) {
        (void)be_fail(BC_E546, "nvptx");
        return 1;
    }

    if (nv->vprt)
        nv_apnd(nv,
            ".extern .func (.param .b32 status) vprintf\n"
            "(\n\t.param .b64 fmt,\n\t.param .b64 args\n);\n\n");

    {
        int dg = 8192;
        for (uint32_t fi = 0; fi < nv->num_mfunc && dg > 0; fi++, dg--)
            em_fdecl(nv, fi);
    }
    return 0;
}

int nv_emit_ptx(nv_module_t *nv, const char *path)
{
    nv->out_len = 0;
    nv->out_tot = 0;
    nv->werr = 0;
    nv->out_fp = fopen(path, "w");
    if (nv->out_fp == NULL) {
        fprintf(stderr, "error: cannot open '%s' for writing\n", path);
        return BC_ERR_IO;
    }

    if (em_pre(nv) != 0)
        return nv_bail(nv, path, BC_ERR_NVIDIA);

    /* Emit each function */
    int guard = 8192;
    for (uint32_t fi = 0; fi < nv->num_mfunc && guard > 0; fi++, guard--)
        em_func(nv, fi);

    if (nv->badop)
        return nv_bail(nv, path, BC_ERR_NVIDIA);
    if (nv->ovf) {
        (void)be_fail(BC_E545, "nvptx", "one PTX instruction",
                      (unsigned)NV_MAX_OUT);
        return nv_bail(nv, path, BC_ERR_NVIDIA);
    }

    nv_flsh(nv);
    if (fclose(nv->out_fp) != 0) nv->werr = 1;
    nv->out_fp = NULL;
    if (nv->werr) {
        fprintf(stderr, "error: cannot write '%s'\n", path);
        remove(path);
        return BC_ERR_IO;
    }

    /* Stats */
    uint32_t nk = 0;
    int kg = 8192;
    for (uint32_t fi = 0; fi < nv->num_mfunc && kg > 0; fi++, kg--) {
        if (nv->mfuncs[fi].is_kern) nk++;
    }
    printf("wrote %s (%u bytes, %u kernel%s, %u instructions)\n",
           path, nv->out_tot, nk, nk == 1 ? "" : "s", nv->num_minst);

    return BC_OK;
}
