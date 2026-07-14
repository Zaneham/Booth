#include "tensix.h"
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>

/* Register allocation, binary encoding, C++ output. Eight LRegs on Tensix,
 * eight GPRs on System/360. The wheel turns. */

/* ---- Encoding Table (Wormhole B0) ---- */
static const tt_enc_entry_t enc_table_sparse[] = {
    /* Data movement */
    { TT_FMT_D, 0x70, "sfpload"    },  /* TT_SFPLOAD    */
    { TT_FMT_E, 0x71, "sfploadi"   },  /* TT_SFPLOADI   */
    { TT_FMT_D, 0x72, "sfpstore"   },  /* TT_SFPSTORE   */
    { TT_FMT_C, 0x73, "sfplut"     },  /* TT_SFPLUT     */
    { TT_FMT_B, 0x7C, "sfpmov"     },  /* TT_SFPMOV     */

    /* FP arithmetic */
    { TT_FMT_A, 0x84, "sfpmad"     },  /* TT_SFPMAD     */
    { TT_FMT_A, 0x85, "sfpadd"     },  /* TT_SFPADD     */
    { TT_FMT_A, 0x86, "sfpmul"     },  /* TT_SFPMUL     */

    /* Immediate arithmetic */
    { TT_FMT_B, 0x74, "sfpmuli"    },  /* TT_SFPMULI    */
    { TT_FMT_C, 0x75, "sfpaddi"    },  /* TT_SFPADDI    */
    { TT_FMT_B, 0x79, "sfpiadd"    },  /* TT_SFPIADD    */

    /* Bitwise */
    { TT_FMT_B, 0x7E, "sfpand"     },  /* TT_SFPAND     */
    { TT_FMT_B, 0x7F, "sfpor"      },  /* TT_SFPOR      */
    { TT_FMT_B, 0x80, "sfpnot"     },  /* TT_SFPNOT     */
    { TT_FMT_B, 0x8D, "sfpxor"     },  /* TT_SFPXOR     */
    { TT_FMT_B, 0x7A, "sfpshft"    },  /* TT_SFPSHFT    */

    /* Comparison / predication */
    { TT_FMT_B, 0x7B, "sfpsetcc"   },  /* TT_SFPSETCC   */
    { TT_FMT_C, 0x8A, "sfpencc"    },  /* TT_SFPENCC    */
    { TT_FMT_C, 0x87, "sfppushc"   },  /* TT_SFPPUSHC   */
    { TT_FMT_C, 0x88, "sfppopc"    },  /* TT_SFPPOPC    */
    { TT_FMT_C, 0x8B, "sfpcompc"   },  /* TT_SFPCOMPC   */

    /* Float manipulation */
    { TT_FMT_B, 0x7D, "sfpabs"     },  /* TT_SFPABS     */
    { TT_FMT_B, 0x89, "sfpsetsgn"  },  /* TT_SFPSETSGN  */
    { TT_FMT_B, 0x76, "sfpdivp2"   },  /* TT_SFPDIVP2   */
    { TT_FMT_B, 0x77, "sfpexexp"   },  /* TT_SFPEXEXP   */
    { TT_FMT_B, 0x78, "sfpexman"   },  /* TT_SFPEXMAN   */
    { TT_FMT_B, 0x82, "sfpsetexp"  },  /* TT_SFPSETEXP  */
    { TT_FMT_B, 0x83, "sfpsetman"  },  /* TT_SFPSETMAN  */
    { TT_FMT_B, 0x81, "sfplz"      },  /* TT_SFPLZ      */

    /* Conversion */
    { TT_FMT_B, 0x90, "sfpcast"    },  /* TT_SFPCAST    */
    { TT_FMT_B, 0x8E, "sfpstochrnd"},  /* TT_SFPSTOCHRND */

    /* Transpose / permute */
    { TT_FMT_A, 0x8C, "sfptransp"  },  /* TT_SFPTRANSP  */
    { TT_FMT_B, 0x92, "sfpswap"    },  /* TT_SFPSWAP    */

    /* Configuration */
    { TT_FMT_C, 0x91, "sfpconfig"  },  /* TT_SFPCONFIG  */
    { TT_FMT_C, 0x93, "sfpldmacro" },  /* TT_SFPLDMACRO */

    /* Blackhole+ */
    { TT_FMT_B, 0x94, "sfpshft2"   },  /* TT_SFPSHFT2   */
    { TT_FMT_C, 0x95, "sfplutfp32" },  /* TT_SFPLUTFP32 */
    { TT_FMT_C, 0x96, "sfple"      },  /* TT_SFPLE      */
    { TT_FMT_C, 0x97, "sfpgt"      },  /* TT_SFPGT      */
    { TT_FMT_A, 0x98, "sfpmul24"   },  /* TT_SFPMUL24   */
    { TT_FMT_A, 0x99, "sfparecip"  },  /* TT_SFPARECIP  */

    /* Control */
    { TT_FMT_C, 0x02, "sfpnop"     },  /* TT_SFPNOP     */
    { TT_FMT_C, 0x8F, "sfpwnop"    },  /* TT_SFPWNOP    */
};

#define ENC_TABLE_SPARSE_LEN \
    (sizeof(enc_table_sparse) / sizeof(enc_table_sparse[0]))

static const tt_enc_entry_t *enc_lookup(uint16_t op)
{
    if (op >= 0x02 && op <= 0x99) {
        int guard = 256;
        for (uint32_t i = 0; i < ENC_TABLE_SPARSE_LEN && guard > 0;
             i++, guard--) {
            if (enc_table_sparse[i].hw_opcode == (uint8_t)op)
                return &enc_table_sparse[i];
        }
    }
    return NULL;
}

/* ---- Output ---- */

static void out_append(tt_module_t *tt, const char *fmt, ...)
{
    if (tt->out_len >= TT_MAX_CODE_SIZE - 1) return;
    va_list ap;
    va_start(ap, fmt);
    int n = vsnprintf(tt->out_buf + tt->out_len,
                      TT_MAX_CODE_SIZE - tt->out_len, fmt, ap);
    va_end(ap);
    if (n > 0)
        tt->out_len += (uint32_t)n;
    if (tt->out_len >= TT_MAX_CODE_SIZE)
        tt->out_len = TT_MAX_CODE_SIZE - 1;
}

/* ---- Register Allocator ---- */

typedef struct {
    uint32_t first_def;
    uint32_t last_use;
    uint16_t activity;
    uint8_t  phys_reg;      /* 0 = unassigned, 1-8 = L0-L7 */
    uint8_t  spilled;
    uint16_t spill_row;     /* 0xFFFF = no slot yet */
    uint16_t pad;
} ra_vreg_info_t;

#define RA_SCRATCH_SIZE 32768
static tt_minst_t ra_scratch[RA_SCRATCH_SIZE];
static uint32_t ra_out_pos;

static ra_vreg_info_t ra_info[TT_MAX_VREGS];

static struct {
    uint32_t vreg;          /* vreg in this LReg, 0 = free */
} ra_lreg[TT_NUM_LREGS];

static uint32_t ra_spill_offset;

#define RA_WEIGHT_DEF    1
#define RA_WEIGHT_USE    2
#define RA_WEIGHT_FMTA   3

static void ra_forward_pass(const tt_module_t *tt, const tt_mfunc_t *MF)
{
    memset(ra_info, 0, sizeof(ra_info));

    int guard = 131072;
    for (uint32_t bi = 0; bi < MF->num_blocks && guard > 0; bi++, guard--) {
        const tt_mblock_t *MB = &tt->mblocks[MF->first_block + bi];
        int ig = 65536;
        for (uint32_t ii = 0; ii < MB->num_insts && ig > 0; ii++, ig--) {
            uint32_t abs_idx = MB->first_inst + ii;
            const tt_minst_t *I = &tt->minsts[abs_idx];

            int total = I->num_defs + I->num_uses;
            if (total > TT_MINST_MAX_OPS) total = TT_MINST_MAX_OPS;

            for (int k = 0; k < total; k++) {
                const tt_operand_t *op = &I->operands[k];
                if (op->kind != TT_MOP_VREG) continue;
                if (op->reg_num == 0) continue;

                ra_vreg_info_t *vi = &ra_info[op->reg_num];
                if (k < I->num_defs) {
                    if (vi->first_def == 0)
                        vi->first_def = abs_idx + 1;
                    vi->activity += RA_WEIGHT_DEF;
                } else {
                    vi->last_use = abs_idx + 1;
                    vi->activity = (uint16_t)(vi->activity +
                        ((I->fmt == TT_FMT_A)
                         ? RA_WEIGHT_FMTA : RA_WEIGHT_USE));
                }
            }
        }
    }
}

static uint32_t ra_find_next_use(const tt_module_t *tt, const tt_mfunc_t *MF,
                                  uint16_t vreg, uint32_t after_idx)
{
    int guard = 65536;
    for (uint32_t bi = 0; bi < MF->num_blocks && guard > 0; bi++, guard--) {
        const tt_mblock_t *MB = &tt->mblocks[MF->first_block + bi];
        int ig = 65536;
        for (uint32_t ii = 0; ii < MB->num_insts && ig > 0; ii++, ig--) {
            uint32_t abs = MB->first_inst + ii;
            if (abs <= after_idx) continue;
            const tt_minst_t *I = &tt->minsts[abs];
            int total = I->num_defs + I->num_uses;
            if (total > TT_MINST_MAX_OPS) total = TT_MINST_MAX_OPS;
            for (int k = I->num_defs; k < total; k++) {
                if (I->operands[k].kind == TT_MOP_VREG &&
                    I->operands[k].reg_num == vreg)
                    return abs;
            }
        }
    }
    return UINT32_MAX;
}

static void ra_emit_load(uint8_t lreg, uint16_t dst_row)
{
    if (ra_out_pos >= RA_SCRATCH_SIZE) return;
    tt_minst_t *I = &ra_scratch[ra_out_pos++];
    memset(I, 0, sizeof(*I));
    I->op = TT_SFPLOAD;
    I->fmt = TT_FMT_D;
    I->num_defs = 1;
    I->num_uses = 3;
    I->operands[0].kind = TT_MOP_LREG;
    I->operands[0].reg_num = lreg;
    I->operands[1].kind = TT_MOP_IMM;
    I->operands[1].imm = TT_LDST_MOD0_FP32;
    I->operands[2].kind = TT_MOP_IMM;
    I->operands[2].imm = 0;
    I->operands[3].kind = TT_MOP_DST_ROW;
    I->operands[3].reg_num = dst_row;
    I->flags = TT_LDST_MOD0_FP32;
}

static void ra_emit_store(uint8_t lreg, uint16_t dst_row)
{
    if (ra_out_pos >= RA_SCRATCH_SIZE) return;
    tt_minst_t *I = &ra_scratch[ra_out_pos++];
    memset(I, 0, sizeof(*I));
    I->op = TT_SFPSTORE;
    I->fmt = TT_FMT_D;
    I->num_defs = 0;
    I->num_uses = 4;
    I->operands[0].kind = TT_MOP_LREG;
    I->operands[0].reg_num = lreg;
    I->operands[1].kind = TT_MOP_IMM;
    I->operands[1].imm = TT_LDST_MOD0_FP32;
    I->operands[2].kind = TT_MOP_IMM;
    I->operands[2].imm = 0;
    I->operands[3].kind = TT_MOP_DST_ROW;
    I->operands[3].reg_num = dst_row;
    I->flags = TT_LDST_MOD0_FP32;
}

static int ra_find_free(void)
{
    for (int r = 0; r < TT_NUM_LREGS; r++) {
        if (ra_lreg[r].vreg == 0) return r;
    }
    return -1;
}

static int ra_evict(tt_module_t *tt, const tt_mfunc_t *MF, uint32_t cur_idx)
{
    uint32_t farthest = 0;
    int evict_r = 0;
    for (int r = 0; r < TT_NUM_LREGS; r++) {
        uint32_t vr = ra_lreg[r].vreg;
        if (vr == 0) { evict_r = r; break; }
        uint32_t nu = ra_find_next_use(tt, MF, (uint16_t)vr, cur_idx);
        if (nu > farthest) { farthest = nu; evict_r = r; }
    }

    uint32_t ev_vr = ra_lreg[evict_r].vreg;
    if (ev_vr > 0 && ev_vr < TT_MAX_VREGS) {
        ra_vreg_info_t *evi = &ra_info[ev_vr];
        if (evi->spill_row == 0xFFFF) {
            if (ra_spill_offset < TT_DST_ROWS)
                evi->spill_row = (uint16_t)ra_spill_offset++;
            else
                evi->spill_row = 0; /* overflow — silently degrade */
        }
        ra_emit_store((uint8_t)evict_r, evi->spill_row);
        evi->spilled = 1;
        evi->phys_reg = 0;
    }
    ra_lreg[evict_r].vreg = 0;
    return evict_r;
}

static int ra_ensure_lreg(tt_module_t *tt, const tt_mfunc_t *MF,
                           uint16_t vreg, uint32_t cur_idx)
{
    if (vreg == 0) return 0;
    ra_vreg_info_t *vi = &ra_info[vreg];

    if (vi->phys_reg > 0 && !vi->spilled) {
        return vi->phys_reg - 1;
    }

    int lreg = ra_find_free();
    if (lreg < 0)
        lreg = ra_evict(tt, MF, cur_idx);

    if (vi->spilled && vi->spill_row != 0xFFFF) {
        ra_emit_load((uint8_t)lreg, vi->spill_row);
        vi->spilled = 0;
    }

    vi->phys_reg = (uint8_t)(lreg + 1);
    ra_lreg[lreg].vreg = vreg;
    return lreg;
}

static void ra_allocate_function(tt_module_t *tt, tt_mfunc_t *MF)
{
    memset(ra_lreg, 0, sizeof(ra_lreg));
    ra_spill_offset = MF->dst_rows_used;
    uint8_t lreg_used_mask = 0;

    int guard = 131072;
    for (uint32_t bi = 0; bi < MF->num_blocks && guard > 0; bi++, guard--) {
        tt_mblock_t *MB = &tt->mblocks[MF->first_block + bi];
        uint32_t new_first = ra_out_pos;

        int ig = 65536;
        for (uint32_t ii = 0; ii < MB->num_insts && ig > 0; ii++, ig--) {
            uint32_t abs_idx = MB->first_inst + ii;
            const tt_minst_t *orig = &tt->minsts[abs_idx];

            if (orig->fmt == TT_FMT_PSEUDO) continue;
            tt_minst_t inst = *orig;
            int total = inst.num_defs + inst.num_uses;
            if (total > TT_MINST_MAX_OPS) total = TT_MINST_MAX_OPS;

            /* Uses */
            for (int k = inst.num_defs; k < total; k++) {
                tt_operand_t *op = &inst.operands[k];
                if (op->kind != TT_MOP_VREG) continue;
                int lreg = ra_ensure_lreg(tt, MF, op->reg_num, abs_idx);
                lreg_used_mask |= (uint8_t)(1 << lreg);
                op->kind = TT_MOP_LREG;
                op->reg_num = (uint16_t)lreg;
            }

            /* Defs */
            for (int k = 0; k < inst.num_defs; k++) {
                tt_operand_t *op = &inst.operands[k];
                if (op->kind != TT_MOP_VREG) continue;
                uint16_t vr = op->reg_num;

                /* Reuse LReg of a dying source */
                int shared = -1;
                for (int u = inst.num_defs; u < total; u++) {
                    if (inst.operands[u].kind != TT_MOP_LREG) continue;
                    uint32_t occ = ra_lreg[inst.operands[u].reg_num].vreg;
                    if (occ > 0 && occ < TT_MAX_VREGS &&
                        ra_info[occ].last_use <= abs_idx + 1) {
                        shared = (int)inst.operands[u].reg_num;
                        break;
                    }
                }

                int lreg;
                if (shared >= 0) {
                    uint32_t old_vr = ra_lreg[shared].vreg;
                    if (old_vr > 0 && old_vr < TT_MAX_VREGS)
                        ra_info[old_vr].phys_reg = 0;
                    lreg = shared;
                } else {
                    lreg = ra_find_free();
                    if (lreg < 0)
                        lreg = ra_evict(tt, MF, abs_idx);
                }

                ra_info[vr].phys_reg = (uint8_t)(lreg + 1);
                ra_info[vr].spilled = 0;
                ra_lreg[lreg].vreg = vr;
                lreg_used_mask |= (uint8_t)(1 << lreg);
                op->kind = TT_MOP_LREG;
                op->reg_num = (uint16_t)lreg;
            }

            if (ra_out_pos < RA_SCRATCH_SIZE)
                ra_scratch[ra_out_pos++] = inst;

            /* Free dead */
            for (int r = 0; r < TT_NUM_LREGS; r++) {
                uint32_t vr = ra_lreg[r].vreg;
                if (vr == 0) continue;
                if (ra_info[vr].last_use > 0 &&
                    ra_info[vr].last_use <= abs_idx + 1) {
                    ra_lreg[r].vreg = 0;
                }
            }
        }

        MB->first_inst = new_first;
        MB->num_insts = ra_out_pos - new_first;
    }

    MF->dst_rows_used = (uint16_t)ra_spill_offset;

    uint8_t m = lreg_used_mask;
    uint16_t count = 0;
    while (m) { count += (m & 1); m >>= 1; }
    MF->num_lregs_used = count;
}

void tensix_regalloc(tt_module_t *tt)
{
    ra_out_pos = 0;

    int guard = 8192;
    for (uint32_t fi = 0; fi < tt->num_mfuncs && guard > 0; fi++, guard--) {
        tt_mfunc_t *MF = &tt->mfuncs[fi];
        ra_forward_pass(tt, MF);
        ra_allocate_function(tt, MF);
    }

    if (ra_out_pos > 0 && ra_out_pos <= TT_MAX_MINSTS) {
        memcpy(tt->minsts, ra_scratch,
               ra_out_pos * sizeof(tt_minst_t));
        tt->num_minsts = ra_out_pos;
    }
}

/* ---- Encoding ---- */

static int get_reg(const tt_operand_t *op)
{
    if (op->kind == TT_MOP_LREG) return (int)op->reg_num;
    return 0;
}

static int32_t get_imm(const tt_operand_t *op)
{
    if (op->kind == TT_MOP_IMM) return op->imm;
    return 0;
}

static int get_dst_row(const tt_operand_t *op)
{
    if (op->kind == TT_MOP_DST_ROW) return (int)op->reg_num;
    if (op->kind == TT_MOP_IMM) return (int)op->imm;
    return 0;
}

static uint32_t encode_inst(const tt_minst_t *I)
{
    const tt_enc_entry_t *enc = enc_lookup(I->op);
    if (!enc) return 0;

    uint32_t op = (uint32_t)enc->hw_opcode << 24;

    switch (enc->fmt) {
    case TT_FMT_A: {
        int dst   = get_reg(&I->operands[0]);
        int src_a = get_reg(&I->operands[1]);
        int src_b = get_reg(&I->operands[2]);
        int src_c = get_reg(&I->operands[3]);
        int mod1  = (I->num_uses > 3) ? (int)get_imm(&I->operands[4]) : 0;
        return op | ((src_a & 0xF) << 16) | ((src_b & 0xF) << 12)
                  | ((src_c & 0xF) << 8) | ((dst & 0xF) << 4)
                  | (mod1 & 0xF);
    }
    case TT_FMT_B: {
        int dst  = get_reg(&I->operands[0]);
        int src  = (I->num_uses > 0) ? get_reg(&I->operands[1]) : 0;
        int imm  = (I->num_uses > 1) ? (int)get_imm(&I->operands[2]) : 0;
        int mod1 = (I->num_uses > 2) ? (int)get_imm(&I->operands[3]) : 0;
        return op | ((imm & 0xFFF) << 12) | ((src & 0xF) << 8)
                  | ((dst & 0xF) << 4) | (mod1 & 0xF);
    }
    case TT_FMT_C: {
        int imm  = (I->num_uses > 0) ? (int)get_imm(&I->operands[0]) : 0;
        int mod1 = (I->num_uses > 1) ? (int)get_imm(&I->operands[1]) : 0;
        return op | ((imm & 0xFFF) << 12) | (mod1 & 0xF);
    }
    case TT_FMT_D: {
        int lreg = get_reg(&I->operands[0]);
        int mod0 = (int)get_imm(&I->operands[1]);
        int addr = (int)get_imm(&I->operands[2]);
        int dest = get_dst_row(&I->operands[3]);
        return op | ((lreg & 0xF) << 20) | ((mod0 & 0xF) << 16)
                  | ((addr & 0x3) << 14) | (dest & 0x3FFF);
    }
    case TT_FMT_E: {
        int lreg = get_reg(&I->operands[0]);
        int mod0 = (int)get_imm(&I->operands[1]);
        int imm  = (int)get_imm(&I->operands[2]);
        return op | ((lreg & 0xF) << 20) | ((mod0 & 0xF) << 16)
                  | (imm & 0xFFFF);
    }
    default:
        return op;
    }
}

/* ---- Disassembly ---- */

static void disasm_inst(const tt_minst_t *I, char *buf, size_t bufsz)
{
    const tt_enc_entry_t *enc = enc_lookup(I->op);
    if (!enc) {
        snprintf(buf, bufsz, "pseudo");
        return;
    }

    switch (enc->fmt) {
    case TT_FMT_A:
        snprintf(buf, bufsz, "%s L%d, L%d, L%d, L%d, %d",
                 enc->mnemonic,
                 get_reg(&I->operands[0]),
                 get_reg(&I->operands[1]),
                 get_reg(&I->operands[2]),
                 get_reg(&I->operands[3]),
                 (I->num_uses > 3) ? (int)get_imm(&I->operands[4]) : 0);
        break;
    case TT_FMT_B: {
        int imm = (I->num_uses > 1) ? (int)get_imm(&I->operands[2]) : 0;
        int mod = (I->num_uses > 2) ? (int)get_imm(&I->operands[3]) : 0;
        if (imm != 0 || mod != 0)
            snprintf(buf, bufsz, "%s L%d, L%d, %d, %d",
                     enc->mnemonic,
                     get_reg(&I->operands[0]),
                     (I->num_uses > 0) ? get_reg(&I->operands[1]) : 0,
                     imm, mod);
        else
            snprintf(buf, bufsz, "%s L%d, L%d",
                     enc->mnemonic,
                     get_reg(&I->operands[0]),
                     (I->num_uses > 0) ? get_reg(&I->operands[1]) : 0);
        break;
    }
    case TT_FMT_C: {
        int imm = (I->num_uses > 0) ? (int)get_imm(&I->operands[0]) : 0;
        int mod = (I->num_uses > 1) ? (int)get_imm(&I->operands[1]) : 0;
        if (imm != 0 || mod != 0)
            snprintf(buf, bufsz, "%s %d, %d", enc->mnemonic, imm, mod);
        else
            snprintf(buf, bufsz, "%s", enc->mnemonic);
        break;
    }
    case TT_FMT_D:
        snprintf(buf, bufsz, "%s L%d, mod=%d, addr=%d, dst=%d",
                 enc->mnemonic,
                 get_reg(&I->operands[0]),
                 (int)get_imm(&I->operands[1]),
                 (int)get_imm(&I->operands[2]),
                 get_dst_row(&I->operands[3]));
        break;
    case TT_FMT_E:
        snprintf(buf, bufsz, "%s L%d, mod=%d, 0x%04X",
                 enc->mnemonic,
                 get_reg(&I->operands[0]),
                 (int)get_imm(&I->operands[1]),
                 (int)get_imm(&I->operands[2]) & 0xFFFF);
        break;
    default:
        snprintf(buf, bufsz, "???");
        break;
    }
}

/* ---- TT-Metalium C++ Emission ---- */

int tensix_emit_metalium(tt_module_t *tt, const char *path)
{
    tt->out_len = 0;

    out_append(tt,
        "/* Generated by Booth — Tensix backend */\n"
        "/* Do not edit. If you must, at least feel bad about it. */\n"
        "\n"
        "#include \"compute_kernel_api.h\"\n"
        "#include \"compute_kernel_api/common.h\"\n"
        "#include \"compute_kernel_api/tile_move_copy.h\"\n"
        "\n"
        "#ifndef TT_SFPINSTWORD\n"
        "#define TT_SFPINSTWORD(w) \\\n"
        "    ckernel::instrn_buffer[0] = (w); \\\n"
        "    ckernel::instrn_buffer[0].push()\n"
        "#endif\n"
        "\n");

    int guard = 8192;
    for (uint32_t fi = 0; fi < tt->num_mfuncs && guard > 0; fi++, guard--) {
        const tt_mfunc_t *MF = &tt->mfuncs[fi];
        if (!MF->is_kernel) continue;

        const char *name = tt->bir->strings + MF->name;
        static const char *pat_names[] = {
            "generic", "element-wise", "reduction"
        };
        const char *pat = (unsigned)MF->coarsen_pattern < TT_PATTERN_COUNT
                          ? pat_names[MF->coarsen_pattern] : "generic";

        out_append(tt,
            "namespace NAMESPACE {\n"
            "void MAIN {\n"
            "    /* kernel: %s, %u LRegs, %u Dst rows, %s, %u tiles/core */\n"
            "    uint32_t num_tiles = get_arg_val<uint32_t>(0);\n"
            "\n"
            "    for (uint32_t tile = 0; tile < num_tiles; tile++) {\n",
            name,
            (unsigned)MF->num_lregs_used,
            (unsigned)MF->dst_rows_used,
            pat,
            (unsigned)MF->tiles_per_core);

        /* Wait on input CBs, unpack tiles into Dst */
        {
            const tt_dmov_t *dm = &tt->dmov;
            if (dm->num_bufs > 0) {
                int dg = TT_MAX_DMOV_BUFS;
                for (uint32_t di = 0; di < dm->num_bufs && dg > 0; di++, dg--) {
                    if (dm->bufs[di].is_output) continue;
                    out_append(tt,
                        "        cb_wait_front(tt::CBIndex::c_%u, 1);\n",
                        dm->bufs[di].cb_index);
                }
                out_append(tt, "        tile_regs_acquire();\n");
                dg = TT_MAX_DMOV_BUFS;
                for (uint32_t di = 0; di < dm->num_bufs && dg > 0; di++, dg--) {
                    if (dm->bufs[di].is_output) continue;
                    out_append(tt,
                        "        copy_tile(tt::CBIndex::c_%u, 0, %u);\n",
                        dm->bufs[di].cb_index, dm->bufs[di].dst_row / 16);
                }
            } else {
                /* Fallback: single CB, no copy_tile (pre-tier4 compat) */
                out_append(tt,
                    "        cb_wait_front(tt::CBIndex::c_0, 1);\n"
                    "        tile_regs_acquire();\n");
            }
        }

        out_append(tt, "\n");

        uint32_t inst_count = 0;
        for (uint32_t bi = 0; bi < MF->num_blocks; bi++) {
            const tt_mblock_t *MB = &tt->mblocks[MF->first_block + bi];

            if (MF->num_blocks > 1)
                out_append(tt, "        /* block %u */\n", bi);

            int blk_guard = 65536;
            for (uint32_t ii = 0; ii < MB->num_insts && blk_guard > 0;
                 ii++, blk_guard--) {
                const tt_minst_t *I = &tt->minsts[MB->first_inst + ii];
                const tt_enc_entry_t *enc = enc_lookup(I->op);
                if (!enc) continue;

                uint32_t word = encode_inst(I);
                char comment[128];
                disasm_inst(I, comment, sizeof(comment));

                out_append(tt,
                    "        TT_SFPINSTWORD(0x%08Xu); /* %s */\n",
                    word, comment);
                inst_count++;
            }
        }

        out_append(tt,
            "\n"
            "        tile_regs_commit();\n"
            "        tile_regs_wait();\n");

        /* Pack results + release + CB handshake */
        {
            const tt_dmov_t *dm = &tt->dmov;
            if (dm->num_bufs > 0) {
                int dg = TT_MAX_DMOV_BUFS;
                for (uint32_t di = 0; di < dm->num_bufs && dg > 0; di++, dg--) {
                    if (!dm->bufs[di].is_output) continue;
                    out_append(tt,
                        "        pack_tile(0, tt::CBIndex::c_%u);\n",
                        dm->bufs[di].cb_index);
                }
                out_append(tt, "        tile_regs_release();\n");
                dg = TT_MAX_DMOV_BUFS;
                for (uint32_t di = 0; di < dm->num_bufs && dg > 0; di++, dg--) {
                    if (dm->bufs[di].is_output) continue;
                    out_append(tt,
                        "        cb_pop_front(tt::CBIndex::c_%u, 1);\n",
                        dm->bufs[di].cb_index);
                }
                dg = TT_MAX_DMOV_BUFS;
                for (uint32_t di = 0; di < dm->num_bufs && dg > 0; di++, dg--) {
                    if (!dm->bufs[di].is_output) continue;
                    out_append(tt,
                        "        cb_push_back(tt::CBIndex::c_%u, 1);\n",
                        dm->bufs[di].cb_index);
                }
            } else {
                /* Fallback: single CB pair (pre-tier4 compat) */
                out_append(tt,
                    "        pack_tile(0, tt::CBIndex::c_16);\n"
                    "        tile_regs_release();\n"
                    "        cb_pop_front(tt::CBIndex::c_0, 1);\n"
                    "        cb_push_back(tt::CBIndex::c_16, 1);\n");
            }
        }

        out_append(tt,
            "    }\n"
            "}\n"
            "} /* namespace */\n"
            "\n");

        printf("  %s: %u instructions, %u Dst rows\n",
               name, inst_count, (unsigned)MF->dst_rows_used);
    }

    FILE *fp = fopen(path, "w");
    if (!fp) {
        fprintf(stderr, "error: cannot open '%s' for writing\n", path);
        return BC_ERR_IO;
    }
    fwrite(tt->out_buf, 1, tt->out_len, fp);
    fclose(fp);

    uint32_t num_kernels = 0;
    int kg = 8192;
    for (uint32_t fi = 0; fi < tt->num_mfuncs && kg > 0; fi++, kg--) {
        if (tt->mfuncs[fi].is_kernel) num_kernels++;
    }
    printf("wrote %s (%u bytes, %u kernel%s)\n",
           path, tt->out_len, num_kernels, num_kernels == 1 ? "" : "s");

    return BC_OK;
}
