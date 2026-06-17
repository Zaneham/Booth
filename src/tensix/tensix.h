#ifndef BARRACUDA_TENSIX_H
#define BARRACUDA_TENSIX_H

#include "bir.h"

/* Tenstorrent Tensix backend. 32-lane SFPU, 8 writable LRegs, predicated
 * execution, no branches, and a register that contains 0.8373.
 * ISA from sfpi-binutils + corsix.org. */

#define BC_ERR_TENSIX   -7

/* ---- Target Constants ---- */

#define TT_SFPU_LANES       32      /* lanes per SFPU vector op */
#define TT_NUM_LREGS        8       /* writable vector registers: L0-L7 */
#define TT_TOTAL_LREGS      16      /* L0-L15, but only L0-L7 are writable */
#define TT_DST_ROWS         1024    /* Dst register file rows (spill target) */

/* ---- Constant Registers (L8-L15, read-only) ---- */

#define TT_LREG_CONST_0873  8       /* L8  = 0.8373 (don't ask) */
#define TT_LREG_ZERO        9       /* L9  = 0.0 */
#define TT_LREG_ONE         10      /* L10 = 1.0 */
#define TT_LREG_PROG0       11      /* L11-L14 = programmable via SFPCONFIG */
#define TT_LREG_PROG1       12
#define TT_LREG_PROG2       13
#define TT_LREG_PROG3       14
#define TT_LREG_LANE_ID     15      /* L15 = lane ID (0,2,4,...,62) */

/* ---- Encoding Formats (all 32-bit, opcode at [31:24]) ---- */

typedef enum {
    TT_FMT_A,          /* [op:8][_:4][src_a:4][src_b:4][src_c:4][dst:4][mod1:4] */
    TT_FMT_B,          /* [op:8][imm12:12][src:4][dst:4][mod1:4] */
    TT_FMT_C,          /* [op:8][imm16:16][dst:4][mod1:4] */
    TT_FMT_D,          /* [op:8][lreg:4][mod0:4][addr_mode:2][dest_addr:14] */
    TT_FMT_E,          /* [op:8][lreg:4][mod0:4][imm16:16] */
    TT_FMT_PSEUDO,
    TT_FMT_COUNT
} tt_fmt_t;

/* ---- SFPU Opcodes (Wormhole B0, from sfpi-binutils) ---- */

typedef enum {
    TT_SFPLOAD          = 0x70,
    TT_SFPLOADI         = 0x71,
    TT_SFPSTORE         = 0x72,
    TT_SFPLUT           = 0x73,
    TT_SFPMOV           = 0x7C,

    TT_SFPMAD           = 0x84,     /* dst = a * b + c */
    TT_SFPADD           = 0x85,     /* dst = a + c */
    TT_SFPMUL           = 0x86,     /* dst = a * b */

    TT_SFPMULI          = 0x74,
    TT_SFPADDI          = 0x75,
    TT_SFPIADD          = 0x79,

    TT_SFPAND           = 0x7E,
    TT_SFPOR            = 0x7F,
    TT_SFPNOT           = 0x80,
    TT_SFPXOR           = 0x8D,
    TT_SFPSHFT          = 0x7A,

    TT_SFPSETCC         = 0x7B,
    TT_SFPENCC          = 0x8A,
    TT_SFPPUSHC         = 0x87,
    TT_SFPPOPC          = 0x88,
    TT_SFPCOMPC         = 0x8B,

    TT_SFPABS           = 0x7D,
    TT_SFPSETSGN        = 0x89,
    TT_SFPDIVP2         = 0x76,
    TT_SFPEXEXP         = 0x77,
    TT_SFPEXMAN         = 0x78,
    TT_SFPSETEXP        = 0x82,
    TT_SFPSETMAN        = 0x83,
    TT_SFPLZ            = 0x81,

    TT_SFPCAST          = 0x90,
    TT_SFPSTOCHRND      = 0x8E,

    TT_SFPTRANSP        = 0x8C,
    TT_SFPSWAP          = 0x92,

    TT_SFPCONFIG        = 0x91,
    TT_SFPLDMACRO       = 0x93,

    /* Blackhole+ */
    TT_SFPSHFT2         = 0x94,
    TT_SFPLUTFP32       = 0x95,
    TT_SFPLE            = 0x96,
    TT_SFPGT            = 0x97,
    TT_SFPMUL24         = 0x98,
    TT_SFPARECIP        = 0x99,

    TT_SFPNOP           = 0x02,
    TT_SFPWNOP          = 0x8F,

    TT_PSEUDO_PHI,
    TT_PSEUDO_COPY,
    TT_PSEUDO_DEF,

    TT_OP_COUNT
} tt_sfpu_op_t;

/* ---- Modifier Constants ---- */

#define TT_LOADI_MOD0_FLOATA  0     /* FP32 upper */
#define TT_LOADI_MOD0_FLOATB  1     /* FP32 lower */
#define TT_LOADI_MOD0_INTA    2     /* INT32 upper */
#define TT_LOADI_MOD0_INTB    4     /* INT32 lower */

#define TT_LDST_MOD0_SRCB     0
#define TT_LDST_MOD0_FP16A    1
#define TT_LDST_MOD0_FP16B    2
#define TT_LDST_MOD0_FP32     3

#define TT_CC_LT             0
#define TT_CC_GE             1
#define TT_CC_EQ             2
#define TT_CC_NE             3

#define TT_ENCC_INIT         0
#define TT_ENCC_COMP         1
#define TT_ENCC_AND          2

/* ---- Machine Operand ---- */

typedef enum {
    TT_MOP_NONE,
    TT_MOP_LREG,
    TT_MOP_VREG,
    TT_MOP_IMM,
    TT_MOP_DST_ROW,
    TT_MOP_LABEL,
} tt_mop_kind_t;

typedef struct {
    uint8_t  kind;          /* tt_mop_kind_t */
    uint8_t  pad;
    uint16_t reg_num;
    int32_t  imm;
} tt_operand_t;

/* ---- Machine Instruction ---- */

#define TT_MINST_MAX_OPS 6

typedef struct {
    uint16_t    op;
    uint8_t     num_defs;
    uint8_t     num_uses;
    tt_operand_t operands[TT_MINST_MAX_OPS];
    uint16_t    flags;
    uint8_t     fmt;
    uint8_t     pad;
} tt_minst_t;

/* ---- Thread Coarsening ---- */

#define TT_WH_NUM_CORES    64       /* Wormhole B0: 8x8 */
#define TT_WH_CORES_X       8
#define TT_WH_CORES_Y       8

typedef enum {
    TT_PATTERN_GENERIC,
    TT_PATTERN_ELEMENTWISE,
    TT_PATTERN_REDUCTION,
    TT_PATTERN_COUNT
} tt_coarsen_pattern_t;

/* ---- Data Movement ---- */

#define TT_MAX_DMOV_BUFS  16

typedef struct {
    uint32_t bir_param;       /* BIR param index (BIR_PARAM subop) */
    uint32_t cb_index;        /* circular buffer ID (0-15 in, 16-31 out) */
    uint32_t tile_size;       /* bytes per tile (2048 BF16, 4096 FP32) */
    uint32_t dst_row;         /* Dst register file row for unpack */
    uint32_t name;            /* string table offset for param name */
    uint8_t  is_output;       /* 0 = input (loaded), 1 = output (stored) */
    uint8_t  pad[3];
} tt_dmov_buf_t;

typedef struct {
    tt_dmov_buf_t bufs[TT_MAX_DMOV_BUFS];
    uint32_t      num_bufs;
    uint32_t      num_inputs;
    uint32_t      num_outputs;
} tt_dmov_t;

/* ---- Machine Block / Function ---- */

typedef struct {
    uint32_t first_inst;
    uint32_t num_insts;
    uint32_t bir_block;
} tt_mblock_t;

typedef struct {
    uint32_t name;
    uint32_t first_block;
    uint16_t num_blocks;
    uint16_t num_lregs_used;
    uint16_t dst_rows_used;
    uint16_t is_kernel;
    uint32_t launch_bounds_max;
    uint32_t launch_bounds_min;
    uint32_t tiles_per_core;
    uint32_t bir_func;
    uint8_t  coarsen_pattern;
    uint8_t  coarsen_pad[3];
} tt_mfunc_t;

/* ---- Module ---- */

#define TT_MAX_MINSTS    (1 << 18)
#define TT_MAX_MBLOCKS   (1 << 16)
#define TT_MAX_MFUNCS    (1 << 12)
#define TT_MAX_VREGS     (1 << 16)
#define TT_MAX_CODE_SIZE (1024 * 1024)

typedef struct {
    const bir_module_t *bir;

    tt_minst_t  minsts[TT_MAX_MINSTS];
    uint32_t    num_minsts;

    tt_mblock_t mblocks[TT_MAX_MBLOCKS];
    uint32_t    num_mblocks;

    tt_mfunc_t  mfuncs[TT_MAX_MFUNCS];
    uint32_t    num_mfuncs;

    uint32_t    vreg_count;
    uint8_t     reg_map[TT_MAX_VREGS];

    uint32_t    val_vreg[BIR_MAX_INSTS];

    char        out_buf[TT_MAX_CODE_SIZE];
    uint32_t    out_len;

    tt_dmov_t   dmov;
} tt_module_t;

/* ---- Encoding Table Entry ---- */

typedef struct {
    uint8_t     fmt;
    uint8_t     hw_opcode;
    const char *mnemonic;
} tt_enc_entry_t;

/* ---- Public API ---- */

int  tensix_compile(const bir_module_t *bir, tt_module_t *tt);
void tensix_coarsen(tt_module_t *tt);
void tensix_regalloc(tt_module_t *tt);
int  tensix_emit_metalium(tt_module_t *tt, const char *path);
int  tensix_emit_binary(tt_module_t *tt, const char *path);   /* raw Tensix words */
int  tensix_emit_ttinsn(tt_module_t *tt, const char *path);   /* RISC-V .ttinsn stream */
/* Data movement — Tier 4 */
void tensix_analyze_datamov(const bir_module_t *bir, const tt_module_t *tt,
                            tt_dmov_t *dmov);
int  tensix_emit_reader(const tt_module_t *tt, const tt_dmov_t *dmov,
                        const char *path);
int  tensix_emit_writer(const tt_module_t *tt, const tt_dmov_t *dmov,
                        const char *path);
int  tensix_emit_host_full(const tt_module_t *tt, const tt_dmov_t *dmov,
                           const char *host_path, const char *reader_path,
                           const char *compute_path, const char *writer_path);

#endif /* BARRACUDA_TENSIX_H */
