#include "encode.h"
#include <string.h>

/*
 * AMDGCN binary instruction encoding.
 * Turns machine instructions into the dwords the hardware actually chews on.
 * Each format has its own encoder, because AMD's ISA designers apparently
 * considered "consistency" a four-letter word.
 */

/* ---- Helpers ---- */

/* Inline constant range: 0-64, -1..-16, and special float values */
static int fits_inline(int32_t val)
{
    if (val >= 0 && val <= 64) return 1;
    if (val >= -16 && val <= -1) return 1;
    return 0;
}

/* Encode an SGPR/inline-constant/literal source for scalar formats */
static uint8_t encode_ssrc(const moperand_t *op, uint32_t *literal, int *need_lit)
{
    switch (op->kind) {
    case MOP_SGPR:   return (uint8_t)op->reg_num;
    case MOP_SPECIAL:
        switch (op->imm) {
        case AMD_SPEC_VCC:       return 106;  /* VCC_LO */
        case AMD_SPEC_EXEC:      return 126;  /* EXEC_LO */
        case AMD_SPEC_SCC:       return 253;
        case AMD_SPEC_M0:        return 124;
        case AMD_SPEC_PRIV_BASE: return 0xED; /* src_private_base */
        default: return 0;
        }
    case MOP_IMM:
        if (fits_inline(op->imm)) {
            if (op->imm >= 0) return (uint8_t)(128 + op->imm);
            return (uint8_t)(192 + (-op->imm - 1));
        }
        *literal = (uint32_t)op->imm;
        *need_lit = 1;
        return 255; /* literal constant */
    default:
        return 0;
    }
}

/* Encode source for VOP formats (can be SGPR, VGPR, inline, literal) */
static uint16_t encode_vsrc(const moperand_t *op, uint32_t *literal, int *need_lit)
{
    switch (op->kind) {
    case MOP_VGPR:   return (uint16_t)(256 + op->reg_num);
    case MOP_SGPR:   return (uint16_t)op->reg_num;
    case MOP_SPECIAL:
        switch (op->imm) {
        case AMD_SPEC_VCC:  return 106;
        case AMD_SPEC_EXEC: return 126;
        case AMD_SPEC_SCC:  return 253;
        default: return 0;
        }
    case MOP_IMM:
        if (fits_inline(op->imm)) {
            if (op->imm >= 0) return (uint16_t)(128 + op->imm);
            return (uint16_t)(192 + (-op->imm - 1));
        }
        *literal = (uint32_t)op->imm;
        *need_lit = 1;
        return 255;
    default:
        return 0;
    }
}

static void emit_dword(amd_module_t *A, uint32_t dw)
{
    if (A->code_len + 4 > AMD_CODE_SIZE) return;
    memcpy(A->code + A->code_len, &dw, 4);
    A->code_len += 4;
}

const amd_enc_entry_t *get_enc_table(const amd_module_t *A)
{
    if (A->target <= AMD_TARGET_GFX942)
        return amd_enc_table_gfx9;
    if (A->target <= AMD_TARGET_GFX1030)
        return amd_enc_table_gfx10;
    return amd_enc_table;
}

/* Encode SDST field: handles both physical SGPRs and special registers */
static uint8_t encode_sdst(const moperand_t *op)
{
    if (op->kind == MOP_SGPR) return (uint8_t)op->reg_num;
    if (op->kind == MOP_SPECIAL) {
        switch (op->imm) {
        case AMD_SPEC_VCC:  return 106;
        case AMD_SPEC_EXEC: return 126;
        case AMD_SPEC_SCC:  return 253;
        case AMD_SPEC_M0:   return 124;
        default: break;
        }
    }
    return 0;
}

/* ---- Format Encoders ---- */

static void encode_sop2(amd_module_t *A, const minst_t *mi, uint16_t hw_op)
{
    /* [31:30]=10 [29:23]=OP [22:16]=SDST [15:8]=SSRC1 [7:0]=SSRC0 */
    uint32_t literal = 0;
    int need_lit = 0;
    uint8_t sdst = (mi->num_defs > 0) ? encode_sdst(&mi->operands[0]) : 0;
    uint8_t ssrc0 = encode_ssrc(&mi->operands[mi->num_defs], &literal, &need_lit);
    uint8_t ssrc1 = (mi->num_uses > 1) ?
                    encode_ssrc(&mi->operands[mi->num_defs + 1], &literal, &need_lit) : 0;

    uint32_t dw = (2u << 30) | ((uint32_t)(hw_op & 0x7F) << 23) |
                  ((uint32_t)sdst << 16) | ((uint32_t)ssrc1 << 8) | ssrc0;
    emit_dword(A, dw);
    if (need_lit) emit_dword(A, literal);
}

static void encode_sop1(amd_module_t *A, const minst_t *mi, uint16_t hw_op)
{
    /* [31:23]=101111101 [22:16]=SDST [15:8]=OP [7:0]=SSRC0 */
    uint32_t literal = 0;
    int need_lit = 0;
    uint8_t sdst = (mi->num_defs > 0) ? encode_sdst(&mi->operands[0]) : 0;
    uint8_t ssrc0 = (mi->num_uses > 0) ?
                    encode_ssrc(&mi->operands[mi->num_defs], &literal, &need_lit) : 0;

    uint32_t dw = 0xBE800000u | ((uint32_t)sdst << 16) |
                  ((uint32_t)(hw_op & 0xFF) << 8) | ssrc0;
    emit_dword(A, dw);
    if (need_lit) emit_dword(A, literal);
}

static void encode_sopc(amd_module_t *A, const minst_t *mi, uint16_t hw_op)
{
    /* [31:23]=101111110 [22:16]=OP [15:8]=SSRC1 [7:0]=SSRC0 */
    uint32_t literal = 0;
    int need_lit = 0;
    uint8_t ssrc0 = encode_ssrc(&mi->operands[0], &literal, &need_lit);
    uint8_t ssrc1 = (mi->num_uses > 1) ?
                    encode_ssrc(&mi->operands[1], &literal, &need_lit) : 0;

    uint32_t dw = 0xBF000000u | ((uint32_t)(hw_op & 0x7F) << 16) |
                  ((uint32_t)ssrc1 << 8) | ssrc0;
    emit_dword(A, dw);
    if (need_lit) emit_dword(A, literal);
}

static void encode_sopp(amd_module_t *A, const minst_t *mi, uint16_t hw_op)
{
    /* [31:23]=101111111 [22:16]=OP [15:0]=SIMM16 */
    uint16_t simm16 = 0;
    if (mi->op == AMD_S_WAITCNT) {
        uint16_t vm   = 63;   /* max: don't wait */
        uint16_t lgkm = 63;
        uint16_t exp  = 7;
        if (mi->flags & AMD_WAIT_VMCNT0)   vm   = 0;
        if (mi->flags & AMD_WAIT_LGKMCNT0) lgkm = 0;

        if (A->target <= AMD_TARGET_GFX1030) {
            /* GFX10 waitcnt SIMM16 (verified against llvm-mc):
               [15:14]=vmcnt[5:4] [13:8]=lgkmcnt[5:0] [6:4]=expcnt[2:0] [3:0]=vmcnt[3:0] */
            simm16 = (uint16_t)(((vm >> 4) & 0x3) << 14) |
                     (uint16_t)((lgkm & 0x3F) << 8) |
                     (uint16_t)((exp & 0x7) << 4) |
                     (uint16_t)(vm & 0xF);
        } else {
            /* GFX11 waitcnt SIMM16: [15:10]=vmcnt [9:4]=lgkmcnt [3:0]=expcnt */
            simm16 = (uint16_t)((vm << 10) | (lgkm << 4) | exp);
        }
    } else if (mi->num_uses > 0 && mi->operands[0].kind == MOP_LABEL) {
        /* Branch target: offset will be patched in fixup pass */
        simm16 = (uint16_t)mi->operands[0].imm;
    } else if (mi->num_uses > 0 && mi->operands[0].kind == MOP_IMM) {
        simm16 = (uint16_t)mi->operands[0].imm;
    }

    uint32_t dw = 0xBF800000u | ((uint32_t)(hw_op & 0x7F) << 16) | simm16;
    emit_dword(A, dw);
}

static void encode_smem(amd_module_t *A, const minst_t *mi, uint16_t hw_op)
{
    uint8_t sdata = (mi->num_defs > 0 && mi->operands[0].kind == MOP_SGPR) ?
                    (uint8_t)mi->operands[0].reg_num : 0;
    uint8_t sbase = 0;
    if (mi->num_uses > 0 && mi->operands[mi->num_defs].kind == MOP_SGPR)
        sbase = (uint8_t)(mi->operands[mi->num_defs].reg_num / 2);
    int32_t offset = 0;
    if (mi->num_uses > 1)
        offset = mi->operands[mi->num_defs + 1].imm;

    if (A->target <= AMD_TARGET_GFX942) {
        /* GFX9 SMEM: 2 dwords
           DW0: [31:26]=110000 [25:18]=OP [17]=IMM [16]=GLC [12:6]=SDATA [5:0]=SBASE
           DW1: [20:0]=OFFSET (when IMM=1) */
        uint32_t dw0 = (0x30u << 26) | ((uint32_t)(hw_op & 0xFF) << 18) |
                       (1u << 17) | /* IMM=1: use immediate offset */
                       ((uint32_t)(sdata & 0x7F) << 6) | (uint32_t)(sbase & 0x3F);
        uint32_t dw1 = (uint32_t)offset & 0x1FFFFF;
        emit_dword(A, dw0);
        emit_dword(A, dw1);
    } else if (A->target >= AMD_TARGET_GFX1200) {
        /* GFX12 SMEM: 2 dwords
           DW0: [31:26]=111101 [24:23]=TH [22:21]=SCOPE [18:13]=OP(6) [12:6]=SDATA [5:0]=SBASE
           DW1: [31:25]=SOFFSET(0x7C=null) [23:0]=IOFFSET(24-bit) */
        uint32_t dw0 = (0x3Du << 26) | ((uint32_t)(hw_op & 0x3F) << 13) |
                       ((uint32_t)(sdata & 0x7F) << 6) | (uint32_t)(sbase & 0x3F);
        uint32_t dw1 = (0x7Cu << 25) | ((uint32_t)offset & 0xFFFFFF);
        emit_dword(A, dw0);
        emit_dword(A, dw1);
    } else {
        /* GFX10/GFX11 SMEM: 2 dwords
           DW0: [31:26]=111101 [25:18]=OP(8) [12:6]=SDATA [5:0]=SBASE
           DW1: [31:25]=SOFFSET [20:0]=OFFSET(21-bit) */
        uint32_t soff_null = (A->target <= AMD_TARGET_GFX1030) ? 0x7Du : 0x7Cu;
        uint32_t dw0 = (0x3Du << 26) | ((uint32_t)(hw_op & 0xFF) << 18) |
                       ((uint32_t)(sdata & 0x7F) << 6) | (uint32_t)(sbase & 0x3F);
        uint32_t dw1 = (soff_null << 25) | ((uint32_t)offset & 0x1FFFFF);
        emit_dword(A, dw0);
        emit_dword(A, dw1);
    }
}

static void encode_vop1(amd_module_t *A, const minst_t *mi, uint16_t hw_op)
{
    /* [31:25]=0111111 [24:17]=VDST [16:9]=OP [8:0]=SRC0 */
    uint32_t literal = 0;
    int need_lit = 0;
    /* v_readfirstlane_b32 has an SGPR destination — accept both files */
    uint8_t vdst = (mi->num_defs > 0 &&
                    (mi->operands[0].kind == MOP_VGPR ||
                     mi->operands[0].kind == MOP_SGPR)) ?
                   (uint8_t)mi->operands[0].reg_num : 0;
    uint16_t src0 = (mi->num_uses > 0) ?
                    encode_vsrc(&mi->operands[mi->num_defs], &literal, &need_lit) : 0;

    uint32_t dw = (0x3Fu << 25) | ((uint32_t)vdst << 17) |
                  ((uint32_t)(hw_op & 0xFF) << 9) | (src0 & 0x1FF);
    emit_dword(A, dw);
    if (need_lit) emit_dword(A, literal);
}

static void encode_vop2(amd_module_t *A, const minst_t *mi, uint16_t hw_op)
{
    /* [31]=0 [30:25]=OP [24:17]=VDST [16:9]=VSRC1 [8:0]=SRC0 */
    uint32_t literal = 0;
    int need_lit = 0;
    uint8_t vdst = (mi->num_defs > 0 && mi->operands[0].kind == MOP_VGPR) ?
                   (uint8_t)mi->operands[0].reg_num : 0;
    uint16_t src0 = encode_vsrc(&mi->operands[mi->num_defs], &literal, &need_lit);
    uint8_t vsrc1 = 0;
    if (mi->num_uses > 1 && mi->operands[mi->num_defs + 1].kind == MOP_VGPR)
        vsrc1 = (uint8_t)mi->operands[mi->num_defs + 1].reg_num;

    uint32_t dw = ((uint32_t)(hw_op & 0x3F) << 25) | ((uint32_t)vdst << 17) |
                  ((uint32_t)vsrc1 << 9) | (src0 & 0x1FF);
    emit_dword(A, dw);
    if (need_lit) emit_dword(A, literal);
}

static void encode_vop3(amd_module_t *A, const minst_t *mi, uint16_t hw_op)
{
    /* VOP3: 2 dwords */
    uint32_t literal = 0;
    int need_lit = 0;
    uint8_t vdst = (mi->num_defs > 0) ?
                   (uint8_t)mi->operands[0].reg_num : 0;
    uint16_t src0 = (mi->num_uses > 0) ?
                    encode_vsrc(&mi->operands[mi->num_defs], &literal, &need_lit) : 0;
    uint16_t src1 = (mi->num_uses > 1) ?
                    encode_vsrc(&mi->operands[mi->num_defs + 1], &literal, &need_lit) : 0;
    uint16_t src2 = (mi->num_uses > 2) ?
                    encode_vsrc(&mi->operands[mi->num_defs + 2], &literal, &need_lit) : 0;

    uint32_t dw0, dw1;
    if (A->target <= AMD_TARGET_GFX942) {
        /* GFX9: DW0: [31:26]=110100 [25:16]=OP [7:0]=VDST */
        dw0 = (0x34u << 26) | ((uint32_t)(hw_op & 0x3FF) << 16) |
              (uint32_t)vdst;
    } else {
        /* GFX10+: DW0: [31:26]=110101 [25:16]=OP [7:0]=VDST */
        dw0 = (0x35u << 26) | ((uint32_t)(hw_op & 0x3FF) << 16) |
              (uint32_t)vdst;
    }
    dw1 = ((src2 & 0x1FF) << 18) | ((src1 & 0x1FF) << 9) | (src0 & 0x1FF);
    emit_dword(A, dw0);
    emit_dword(A, dw1);
}

static void encode_vopc(amd_module_t *A, const minst_t *mi, uint16_t hw_op)
{
    /* [31:25]=0111110 [24:17]=OP [16:9]=VSRC1 [8:0]=SRC0 */
    uint32_t literal = 0;
    int need_lit = 0;
    uint16_t src0 = (mi->num_uses > 0) ?
                    encode_vsrc(&mi->operands[0], &literal, &need_lit) : 0;
    uint8_t vsrc1 = 0;
    if (mi->num_uses > 1 && mi->operands[1].kind == MOP_VGPR)
        vsrc1 = (uint8_t)mi->operands[1].reg_num;

    uint32_t dw = (0x3Eu << 25) | ((uint32_t)(hw_op & 0xFF) << 17) |
                  ((uint32_t)vsrc1 << 9) | (src0 & 0x1FF);
    emit_dword(A, dw);
    if (need_lit) emit_dword(A, literal);
}

static void encode_ds(amd_module_t *A, const minst_t *mi, uint16_t hw_op)
{
    /* DS: 2 dwords
       DW0: [31:26]=110110 [25:18]=OP [17:16]=GDS [15:0]=OFFSET
       DW1: [31:24]=VDST [23:16]=DATA1 [15:8]=DATA0 [7:0]=ADDR */
    uint8_t vdst = 0, addr = 0, data0 = 0;
    if (mi->num_defs > 0 && mi->operands[0].kind == MOP_VGPR)
        vdst = (uint8_t)mi->operands[0].reg_num;
    /* First use operand is addr */
    uint8_t use_base = mi->num_defs;
    if (mi->num_uses > 0 && mi->operands[use_base].kind == MOP_VGPR)
        addr = (uint8_t)mi->operands[use_base].reg_num;
    if (mi->num_uses > 1 && mi->operands[use_base + 1].kind == MOP_VGPR)
        data0 = (uint8_t)mi->operands[use_base + 1].reg_num;

    uint16_t offset = 0;
    /* Check for immediate offset in last use operand */
    uint8_t last_use = (uint8_t)(use_base + mi->num_uses - 1);
    if (mi->num_uses > 0 && last_use < MINST_MAX_OPS &&
        mi->operands[last_use].kind == MOP_IMM)
        offset = (uint16_t)mi->operands[last_use].imm;

    uint32_t dw0 = (0x36u << 26) | ((uint32_t)(hw_op & 0xFF) << 18) | offset;
    uint32_t dw1 = ((uint32_t)vdst << 24) | ((uint32_t)data0 << 8) | addr;
    emit_dword(A, dw0);
    emit_dword(A, dw1);
}

static void encode_flat_global(amd_module_t *A, const minst_t *mi, uint16_t hw_op)
{
    uint8_t vdst = 0, addr = 0, data = 0;
    int32_t offset = 0;
    const amd_enc_entry_t *tbl = get_enc_table(A);
    int is_scratch = (tbl[mi->op].fmt == AMD_FMT_FLAT_SCR);
    int is_flat = (tbl[mi->op].fmt == AMD_FMT_FLAT);

    /* Extract def: VDST */
    if (mi->num_defs > 0 && mi->operands[0].kind == MOP_VGPR)
        vdst = (uint8_t)mi->operands[0].reg_num;

    /* Default null SADDR (GFX11 default, adjusted per-target below) */
    uint32_t saddr = 0x7C;

    /* Walk use operands: 1st VGPR=addr, 2nd VGPR=data, SGPR=saddr, IMM=offset.
     * Scratch SADDR-only stores have one VGPR (data only, no VADDR) —
     * detect this and put the single VGPR in data, leave addr=0. */
    uint8_t use_base = mi->num_defs;
    int got_addr = 0;
    int n_vgprs = 0;
    int has_sgpr = 0;
    for (uint8_t k = use_base; k < (uint8_t)(use_base + mi->num_uses) && k < MINST_MAX_OPS; k++) {
        if (mi->operands[k].kind == MOP_VGPR) n_vgprs++;
        if (mi->operands[k].kind == MOP_SGPR) has_sgpr = 1;
    }
    /* SADDR-only scratch store: single VGPR is data, addr=0 (off) */
    int saddr_only_store = (is_scratch && mi->num_defs == 0 && n_vgprs == 1 && has_sgpr);
    for (uint8_t k = use_base; k < (uint8_t)(use_base + mi->num_uses) && k < MINST_MAX_OPS; k++) {
        const moperand_t *op = &mi->operands[k];
        if (op->kind == MOP_VGPR) {
            if (saddr_only_store) { data = (uint8_t)op->reg_num; }
            else if (!got_addr) { addr = (uint8_t)op->reg_num; got_addr = 1; }
            else { data = (uint8_t)op->reg_num; }
        } else if (op->kind == MOP_SGPR) {
            saddr = op->reg_num;
        } else if (op->kind == MOP_IMM) {
            offset = op->imm;
        }
    }

    if (A->target >= AMD_TARGET_GFX1200) {
        /* GFX12 FLAT/GLOBAL/SCRATCH: 3 dwords (96-bit)
           DW0: [31:24]=ENCODING [21:14]=OP [7]=NV [6:0]=SADDR
           DW1: [30:23]=VSRC [22:20]=TH [19:18]=SCOPE [17]=SVE [7:0]=VDST
           DW2: [31:8]=IOFFSET(24-bit) [7:0]=VADDR
           Encoding: global=0xEE, scratch=0xED
           (PDF says OP[20:13] but AMD ISA XML has OP[21:14]) */
        uint8_t enc = is_scratch ? 0xED : 0xEE;
        uint8_t sve = is_scratch ? 1 : 0;

        uint32_t dw0 = ((uint32_t)enc << 24) |
                       ((uint32_t)(hw_op & 0xFF) << 14) |
                       (saddr & 0x7F);
        uint32_t dw1 = ((uint32_t)data << 23) |
                       ((uint32_t)sve << 17) |
                       (uint32_t)vdst;
        uint32_t dw2 = (((uint32_t)offset & 0xFFFFFF) << 8) |
                       (uint32_t)addr;

        if (mi->flags & AMD_FLAG_GLC) dw1 |= (1u << 20); /* TH[0] */

        emit_dword(A, dw0);
        emit_dword(A, dw1);
        emit_dword(A, dw2);
    } else if (A->target <= AMD_TARGET_GFX942) {
        /* GFX9 FLAT/GLOBAL/SCRATCH: 2 dwords (64-bit)
           DW0: [31:26]=0x37 [24:18]=OP(7b) [16]=GLC
                [15:14]=SEG [12:0]=OFFSET(13b signed)
           DW1: [31:24]=VDST [23:16]=SADDR [15:8]=DATA [7:0]=ADDR
           Null SADDR=0x7F. SEG: 0=flat 1=scratch 2=global */
        uint8_t seg = is_flat ? 0 : (is_scratch ? 1 : 2);
        if (saddr == 0x7C) saddr = 0x7F;

        uint32_t off_lo = (uint32_t)offset & 0x1FFF;
        uint32_t dw0 = (0x37u << 26) | ((uint32_t)(hw_op & 0x7F) << 18) |
                       ((uint32_t)seg << 14) | off_lo;
        uint32_t dw1 = ((uint32_t)vdst << 24) | ((saddr & 0xFF) << 16) |
                       ((uint32_t)data << 8) | addr;
        if (mi->flags & AMD_FLAG_GLC) dw0 |= (1u << 16);
        emit_dword(A, dw0);
        emit_dword(A, dw1);
    } else if (A->target <= AMD_TARGET_GFX1030) {
        /* GFX10 FLAT/GLOBAL/SCRATCH: 2 dwords (64-bit)
           DW0: [31:26]=0x37 [25]=DLC [24:18]=OP(7b) [17]=SLC [16]=GLC
                [15:14]=SEG [12:0]=OFFSET(13b signed)
           DW1: [31:24]=VDST [23:16]=SADDR [15:8]=DATA [7:0]=ADDR
           Null SADDR=0x7D for both global and scratch */
        uint8_t seg = is_scratch ? 1 : 2;
        if (saddr == 0x7C) saddr = 0x7D;

        uint32_t off_lo = (uint32_t)offset & 0x1FFF;
        uint32_t dw0 = (0x37u << 26) | ((uint32_t)(hw_op & 0x7F) << 18) |
                       ((uint32_t)seg << 14) | off_lo;
        uint32_t dw1 = ((uint32_t)vdst << 24) | ((saddr & 0xFF) << 16) |
                       ((uint32_t)data << 8) | addr;
        if (mi->flags & AMD_FLAG_GLC) dw0 |= (1u << 16);
        emit_dword(A, dw0);
        emit_dword(A, dw1);
    } else {
        /* GFX11 FLAT/GLOBAL/SCRATCH: 2 dwords (64-bit)
           DW0: [31:26]=0x37 [25:18]=OP(8b) [17:16]=SEG [14]=GLC
                [12:0]=OFFSET(13b signed)
           DW1: [31:24]=VDST [23:16]=SADDR [15:8]=DATA [7:0]=ADDR */
        uint8_t seg = is_scratch ? 1 : 2;
        if (is_scratch && saddr == 0x7C) saddr = 0xFC;

        uint32_t off_lo = (uint32_t)offset & 0x1FFF;
        uint32_t dw0 = (0x37u << 26) | ((uint32_t)(hw_op & 0xFF) << 18) |
                       ((uint32_t)seg << 16) | off_lo;
        uint32_t dw1 = ((uint32_t)vdst << 24) | ((saddr & 0xFF) << 16) |
                       ((uint32_t)data << 8) | addr;
        if (mi->flags & AMD_FLAG_GLC) dw0 |= (1u << 14);
        emit_dword(A, dw0);
        emit_dword(A, dw1);
    }
}

static void encode_vop3p_mai(amd_module_t *A, const minst_t *mi, uint16_t hw_op)
{
    const amd_mfma_t *sh = amd_mfsh(mi->op);
    if (sh) {
        uint8_t t = amd_mfop(sh, A->target);
        if (t == MFMA_NONE) { A->enc_err = 1; return; }
        hw_op = t;
    }
    /* VOP3P-MAI: 64-bit encoding for MFMA matrix instructions.
       DW0: [31:23]=0x1A7(prefix) [22:16]=OP(7b) [15]=ACC_CD [14:11]=ABID
            [10:8]=CBSZ [7:0]=VDST
       DW1: [63:61]=BLGP [60]=SRC1_ACC [59]=SRC0_ACC [58:50]=SRC2
            [49:41]=SRC1 [40:32]=SRC0
       On gfx942, unified VGPR file means ACC flags are 0. Lovely. */
    uint8_t vdst = (mi->num_defs > 0 && mi->operands[0].kind == MOP_VGPR) ?
                   (uint8_t)mi->operands[0].reg_num : 0;
    uint32_t literal = 0;
    int need_lit = 0;
    uint16_t src0 = (mi->num_uses > 0) ?
        encode_vsrc(&mi->operands[mi->num_defs], &literal, &need_lit) : 0;
    uint16_t src1 = (mi->num_uses > 1) ?
        encode_vsrc(&mi->operands[mi->num_defs + 1], &literal, &need_lit) : 0;
    uint16_t src2 = (mi->num_uses > 2) ?
        encode_vsrc(&mi->operands[mi->num_defs + 2], &literal, &need_lit) : 0;

    /* gfx942: ACC_CD=0 (VGPR dest), CBSZ=0, ABID=0, BLGP=0,
       SRC0_ACC=0, SRC1_ACC=0 — unified register file, no AGPR faff */
    uint32_t dw0 = (0x1A7u << 23) | ((uint32_t)(hw_op & 0x7F) << 16) |
                   (uint32_t)vdst;
    uint32_t dw1 = ((src2 & 0x1FF) << 18) | ((src1 & 0x1FF) << 9) |
                   (src0 & 0x1FF);
    emit_dword(A, dw0);
    emit_dword(A, dw1);
}

/* ---- Function Encoder ---- */

/* Instruction offsets for branch fixup */
static uint32_t inst_offsets[AMD_MAX_MINSTS];
static uint32_t block_offsets[AMD_MAX_MBLOCKS];

void encode_function(amd_module_t *A, uint32_t mf_idx)
{
    const mfunc_t *F = &A->mfuncs[mf_idx];
    const amd_enc_entry_t *tbl = get_enc_table(A);

    /* Pass 1: compute instruction byte offsets */
    uint32_t offset = A->code_len;
    for (uint32_t bi = 0; bi < F->num_blocks; bi++) {
        uint32_t mb_idx = F->first_block + bi;
        if (mb_idx >= AMD_MAX_MBLOCKS) break;
        const mblock_t *MB = &A->mblocks[mb_idx];
        block_offsets[mb_idx] = offset;

        for (uint32_t ii = 0; ii < MB->num_insts; ii++) {
            uint32_t mi_idx = MB->first_inst + ii;
            if (mi_idx >= AMD_MAX_MINSTS) break;
            const minst_t *mi = &A->minsts[mi_idx];
            inst_offsets[mi_idx] = offset;

            if (mi->op >= AMD_OP_COUNT) { offset += 4; continue; }
            const amd_enc_entry_t *enc = &tbl[mi->op];

            switch (enc->fmt) {
            case AMD_FMT_SOP2: case AMD_FMT_SOP1: case AMD_FMT_SOPC:
            case AMD_FMT_SOPP: case AMD_FMT_VOP1: case AMD_FMT_VOP2:
            case AMD_FMT_VOPC:
                offset += 4;
                /* Check if a literal constant is needed */
                for (uint8_t k = 0; k < mi->num_defs + mi->num_uses && k < MINST_MAX_OPS; k++) {
                    if (mi->operands[k].kind == MOP_IMM && !fits_inline(mi->operands[k].imm)) {
                        offset += 4;
                        break;
                    }
                }
                break;
            case AMD_FMT_SMEM: case AMD_FMT_VOP3: case AMD_FMT_DS:
            case AMD_FMT_VOP3P_MAI:
                offset += 8;
                break;
            case AMD_FMT_FLAT_GBL: case AMD_FMT_FLAT_SCR: case AMD_FMT_FLAT:
                offset += (A->target >= AMD_TARGET_GFX1200) ? 12 : 8;
                break;
            case AMD_FMT_PSEUDO:
                break; /* no bytes */
            default:
                offset += 4;
                break;
            }
        }
    }

    /* Copy offsets into module for .debug_bc emission */
    for (uint32_t bi = 0; bi < F->num_blocks; bi++) {
        uint32_t mb_idx = F->first_block + bi;
        if (mb_idx >= AMD_MAX_MBLOCKS) break;
        const mblock_t *MB = &A->mblocks[mb_idx];
        for (uint32_t ii = 0; ii < MB->num_insts; ii++) {
            uint32_t mi_idx = MB->first_inst + ii;
            if (mi_idx < AMD_MAX_MINSTS)
                A->inst_off[mi_idx] = inst_offsets[mi_idx];
        }
    }

    /* Pass 2: encode with branch fixup */
    for (uint32_t bi = 0; bi < F->num_blocks; bi++) {
        uint32_t mb_idx = F->first_block + bi;
        if (mb_idx >= AMD_MAX_MBLOCKS) break;
        const mblock_t *MB = &A->mblocks[mb_idx];

        for (uint32_t ii = 0; ii < MB->num_insts; ii++) {
            uint32_t mi_idx = MB->first_inst + ii;
            if (mi_idx >= AMD_MAX_MINSTS) break;
            minst_t *mi = &A->minsts[mi_idx]; /* non-const for branch fixup */

            if (mi->op >= AMD_OP_COUNT) continue;
            const amd_enc_entry_t *enc = &tbl[mi->op];

            /* Fix up branch targets: convert block index to PC-relative offset */
            if (enc->fmt == AMD_FMT_SOPP && mi->num_uses > 0 &&
                mi->operands[0].kind == MOP_LABEL) {
                uint32_t target_mb = (uint32_t)mi->operands[0].imm;
                if (target_mb < AMD_MAX_MBLOCKS) {
                    int32_t pc_offset = (int32_t)(block_offsets[target_mb] -
                                                  (inst_offsets[mi_idx] + 4));
                    pc_offset /= 4; /* SOPP offset is in dwords */
                    mi->operands[0].imm = pc_offset;
                }
            }

            switch (enc->fmt) {
            case AMD_FMT_SOP2: encode_sop2(A, mi, enc->hw_opcode); break;
            case AMD_FMT_SOP1: encode_sop1(A, mi, enc->hw_opcode); break;
            case AMD_FMT_SOPC: encode_sopc(A, mi, enc->hw_opcode); break;
            case AMD_FMT_SOPP: encode_sopp(A, mi, enc->hw_opcode); break;
            case AMD_FMT_SMEM: encode_smem(A, mi, enc->hw_opcode); break;
            case AMD_FMT_VOP1: encode_vop1(A, mi, enc->hw_opcode); break;
            case AMD_FMT_VOP2: encode_vop2(A, mi, enc->hw_opcode); break;
            case AMD_FMT_VOP3: encode_vop3(A, mi, enc->hw_opcode); break;
            case AMD_FMT_VOPC: encode_vopc(A, mi, enc->hw_opcode); break;
            case AMD_FMT_DS:   encode_ds(A, mi, enc->hw_opcode);   break;
            case AMD_FMT_VOP3P_MAI:
                encode_vop3p_mai(A, mi, enc->hw_opcode); break;
            case AMD_FMT_FLAT_GBL:
            case AMD_FMT_FLAT_SCR:
            case AMD_FMT_FLAT:
                encode_flat_global(A, mi, enc->hw_opcode);
                break;
            case AMD_FMT_PSEUDO:
                break;
            default:
                emit_dword(A, 0); /* placeholder */
                break;
            }
        }
    }
}
