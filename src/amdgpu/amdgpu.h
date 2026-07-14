#ifndef BARRACUDA_AMDGPU_H
#define BARRACUDA_AMDGPU_H

#include "bir.h"

/*
 * AMDGPU backend for Booth.
 * Targets CDNA 2/3 (gfx90a/gfx942), RDNA 2 (gfx1030), RDNA 3 (gfx1100), RDNA 4 (gfx1200).
 * Compiles BIR SSA to AMDGCN machine IR, then emits assembly text
 * or binary ELF code objects (.hsaco).
 * Built with the quiet confidence of someone who reads ISA manuals for fun.
 */

#define BC_ERR_AMDGPU   -6

/* ---- Target Selection ---- */

typedef enum {
    AMD_TARGET_GFX90A,    /* CDNA 2 (MI250, Wave64) */
    AMD_TARGET_GFX942,    /* CDNA 3 (MI300X, Wave64) */
    AMD_TARGET_GFX1030,   /* RDNA 2 */
    AMD_TARGET_GFX1100,   /* RDNA 3 */
    AMD_TARGET_GFX1200,   /* RDNA 4 */
} amd_target_t;

/* ---- Target Constants ---- */

#define AMD_MAX_SGPRS       102
#define AMD_MAX_VGPRS       256
#define AMD_WAVE_SIZE       32
#define AMD_WAVE64          64

/* Pre-loaded SGPRs — layout depends on kernel needs.
 * Default (no dispatch_ptr): s[0:1]=kernarg, s2+=TGID → reserved=3
 * With dispatch_ptr:         s[0:1]=dispatch, s[2:3]=kernarg, s4+=TGID → reserved=5
 * Actual positions computed per-kernel in isel. */
#define AMD_KERN_MIN_RESERVED  2  /* absolute floor: kernarg pair */

/* Pre-loaded VGPRs */
#define AMD_VGPR_THREAD_X     0   /* v0: thread ID X */
#define AMD_VGPR_THREAD_Y     1   /* v1: thread ID Y */
#define AMD_VGPR_THREAD_Z     2   /* v2: thread ID Z */

/* CDNA flat scratch staging (reserved physical VGPRs) */
#define AMD_VGPR_SCR_LO     250   /* v250: scratch addr lo */
#define AMD_VGPR_SCR_HI     251   /* v251: private_base hi */

/* Special register IDs (for MOP_SPECIAL) */
#define AMD_SPEC_VCC        0
#define AMD_SPEC_EXEC       1
#define AMD_SPEC_SCC        2
#define AMD_SPEC_M0         3
#define AMD_SPEC_PRIV_BASE  4   /* src_private_base (SSRC=0xED) */

/* ELF constants */
#define EM_AMDGPU                224
#define ELFOSABI_AMDGPU_HSA      64
#define EF_AMDGPU_MACH_AMDGCN_GFX90A   0x3F
#define EF_AMDGPU_MACH_AMDGCN_GFX942   0x4C
#define EF_AMDGPU_MACH_AMDGCN_GFX1030  0x36
#define EF_AMDGPU_MACH_AMDGCN_GFX1100  0x41
#define EF_AMDGPU_MACH_AMDGCN_GFX1200  0x48

/* ---- Instruction Encoding Formats ---- */

typedef enum {
    AMD_FMT_SOP1,       /* scalar one-input */
    AMD_FMT_SOP2,       /* scalar two-input */
    AMD_FMT_SOPC,       /* scalar compare (sets SCC) */
    AMD_FMT_SOPK,       /* scalar with 16-bit inline constant */
    AMD_FMT_SOPP,       /* scalar program (branches, barriers, waitcnt) */
    AMD_FMT_SMEM,       /* scalar memory */
    AMD_FMT_VOP1,       /* vector one-input */
    AMD_FMT_VOP2,       /* vector two-input */
    AMD_FMT_VOP3,       /* vector three-input (64-bit encoding) */
    AMD_FMT_VOPC,       /* vector compare (sets VCC) */
    AMD_FMT_DS,         /* data share (LDS) */
    AMD_FMT_FLAT_GBL,   /* flat/global memory */
    AMD_FMT_FLAT_SCR,   /* flat/scratch memory */
    AMD_FMT_FLAT,       /* true flat (SEG=0, private aperture) */
    AMD_FMT_VOP3P_MAI,  /* MFMA matrix instructions (64-bit, CDNA only) */
    AMD_FMT_PSEUDO,     /* pseudo-instruction (eliminated before emit) */
    AMD_FMT_COUNT
} amd_fmt_t;

/* ---- Machine Opcodes ---- */

typedef enum {
    /* -- SOP2: scalar two-input -- */
    AMD_S_ADD_U32,
    AMD_S_ADD_I32,
    AMD_S_SUB_U32,
    AMD_S_MUL_I32,
    AMD_S_AND_B32,
    AMD_S_OR_B32,
    AMD_S_XOR_B32,
    AMD_S_LSHL_B32,
    AMD_S_LSHR_B32,
    AMD_S_ASHR_I32,
    AMD_S_ANDN2_B32,
    AMD_S_ORN2_B32,
    AMD_S_BFE_I32,
    AMD_S_CSELECT_B32,

    /* -- SOP2: scalar two-input, 64-bit (CDNA/Wave64) -- */
    AMD_S_AND_B64,
    AMD_S_OR_B64,
    AMD_S_XOR_B64,
    AMD_S_ANDN2_B64,
    AMD_S_ORN2_B64,

    /* -- SOP1: scalar one-input -- */
    AMD_S_MOV_B32,
    AMD_S_NOT_B32,
    AMD_S_AND_SAVEEXEC_B32,
    AMD_S_MOV_B64,
    AMD_S_NOT_B64,
    AMD_S_AND_SAVEEXEC_B64,
    AMD_S_SETPC_B64,
    AMD_S_SWAPPC_B64,
    AMD_S_GETPC_B64,

    /* -- SOPC: scalar compare -- */
    AMD_S_CMP_EQ_U32,
    AMD_S_CMP_NE_U32,
    AMD_S_CMP_LT_U32,
    AMD_S_CMP_LE_U32,
    AMD_S_CMP_GT_U32,
    AMD_S_CMP_GE_U32,
    AMD_S_CMP_LT_I32,
    AMD_S_CMP_LE_I32,
    AMD_S_CMP_GT_I32,
    AMD_S_CMP_GE_I32,
    AMD_S_CMP_EQ_I32,
    AMD_S_CMP_NE_I32,

    /* -- SOPP: scalar program -- */
    AMD_S_BRANCH,
    AMD_S_CBRANCH_SCC0,
    AMD_S_CBRANCH_SCC1,
    AMD_S_CBRANCH_EXECZ,
    AMD_S_CBRANCH_EXECNZ,
    AMD_S_ENDPGM,
    AMD_S_BARRIER,
    AMD_S_WAITCNT,
    AMD_S_NOP,
    AMD_S_WAIT_LOADCNT,   /* GFX12: wait for VMEM loads */
    AMD_S_WAIT_STORECNT,  /* GFX12: wait for VMEM stores */
    AMD_S_WAIT_DSCNT,     /* GFX12: wait for DS/LDS */
    AMD_S_WAIT_KMCNT,     /* GFX12: wait for scalar memory */

    /* -- SMEM: scalar memory -- */
    AMD_S_LOAD_DWORD,
    AMD_S_LOAD_DWORDX2,
    AMD_S_LOAD_DWORDX4,

    /* -- VOP2: vector two-input -- */
    AMD_V_ADD_U32,
    AMD_V_SUB_U32,
    AMD_V_MUL_LO_U32,
    AMD_V_AND_B32,
    AMD_V_OR_B32,
    AMD_V_XOR_B32,
    AMD_V_LSHLREV_B32,
    AMD_V_LSHRREV_B32,
    AMD_V_ASHRREV_I32,
    AMD_V_ADD_F32,
    AMD_V_SUB_F32,
    AMD_V_MUL_F32,
    AMD_V_CNDMASK_B32,
    AMD_V_MIN_F32,
    AMD_V_MAX_F32,

    /* -- VOP1: vector one-input -- */
    AMD_V_MOV_B32,
    AMD_V_CVT_F32_I32,
    AMD_V_CVT_F32_U32,
    AMD_V_CVT_I32_F32,
    AMD_V_CVT_U32_F32,
    AMD_V_CVT_F32_F16,
    AMD_V_CVT_F16_F32,
    AMD_V_CVT_F64_F32,
    AMD_V_CVT_F32_F64,
    AMD_V_CVT_F32_BF16,
    AMD_V_CVT_BF16_F32,
    AMD_V_RCP_F32,
    AMD_V_SQRT_F32,
    AMD_V_RSQ_F32,
    AMD_V_EXP_F32,
    AMD_V_LOG_F32,
    AMD_V_SIN_F32,
    AMD_V_COS_F32,
    AMD_V_FLOOR_F32,
    AMD_V_CEIL_F32,
    AMD_V_TRUNC_F32,
    AMD_V_RNDNE_F32,
    AMD_V_FRACT_F32,
    AMD_V_NOT_B32,
    AMD_V_READFIRSTLANE_B32,

    /* -- VOP3: vector three-input -- */
    AMD_V_MAD_U32_U24,
    AMD_V_BFE_I32,
    AMD_V_BFE_U32,
    AMD_V_LSHL_ADD_U32,
    AMD_V_ADD3_U32,

    /* -- VOPC: vector compare -- */
    AMD_V_CMP_EQ_U32,
    AMD_V_CMP_NE_U32,
    AMD_V_CMP_LT_U32,
    AMD_V_CMP_LE_U32,
    AMD_V_CMP_GT_U32,
    AMD_V_CMP_GE_U32,
    AMD_V_CMP_LT_I32,
    AMD_V_CMP_LE_I32,
    AMD_V_CMP_GT_I32,
    AMD_V_CMP_GE_I32,
    AMD_V_CMP_EQ_I32,
    AMD_V_CMP_NE_I32,
    AMD_V_CMP_EQ_F32,
    AMD_V_CMP_NE_F32,
    AMD_V_CMP_LT_F32,
    AMD_V_CMP_LE_F32,
    AMD_V_CMP_GT_F32,
    AMD_V_CMP_GE_F32,
    AMD_V_CMP_O_F32,
    AMD_V_CMP_U_F32,
    AMD_V_CMP_NLT_F32,
    AMD_V_CMP_NLE_F32,
    AMD_V_CMP_NGT_F32,
    AMD_V_CMP_NGE_F32,
    AMD_V_CMP_NEQ_F32,

    /* -- DS: data share (LDS) -- */
    AMD_DS_READ_B32,
    AMD_DS_WRITE_B32,
    AMD_DS_ADD_RTN_U32,
    AMD_DS_SUB_RTN_U32,
    AMD_DS_AND_RTN_B32,
    AMD_DS_OR_RTN_B32,
    AMD_DS_XOR_RTN_B32,
    AMD_DS_MIN_RTN_I32,
    AMD_DS_MAX_RTN_I32,
    AMD_DS_SWIZZLE_B32,
    AMD_DS_BPERMUTE_B32,

    /* -- FLAT_GBL: global memory -- */
    AMD_GLOBAL_LOAD_DWORD,
    AMD_GLOBAL_STORE_DWORD,
    AMD_GLOBAL_LOAD_DWORDX2,
    AMD_GLOBAL_STORE_DWORDX2,
    AMD_GLOBAL_ATOMIC_ADD,
    AMD_GLOBAL_ATOMIC_SUB,
    AMD_GLOBAL_ATOMIC_AND,
    AMD_GLOBAL_ATOMIC_OR,
    AMD_GLOBAL_ATOMIC_XOR,
    AMD_GLOBAL_ATOMIC_SMIN,
    AMD_GLOBAL_ATOMIC_SMAX,
    AMD_GLOBAL_ATOMIC_SWAP,
    AMD_GLOBAL_ATOMIC_CMPSWAP,

    /* -- FLAT_SCR: scratch memory -- */
    AMD_SCRATCH_LOAD_DWORD,
    AMD_SCRATCH_STORE_DWORD,

    /* -- FLAT: true flat (SEG=0, CDNA private aperture scratch) -- */
    AMD_FLAT_LOAD_DWORD,
    AMD_FLAT_STORE_DWORD,

    /* -- VOP3P-MAI: Matrix Fused Multiply-Add (CDNA only) -- */
    AMD_V_MFMA_F32_4X4X4_F16,
    AMD_V_MFMA_F32_16X16X16_F16,
    AMD_V_MFMA_F32_32X32X8_F16,
    AMD_V_MFMA_F32_4X4X4_BF16_1K,
    AMD_V_MFMA_F32_16X16X16_BF16_1K,
    AMD_V_MFMA_F32_32X32X8_BF16_1K,
    AMD_V_MFMA_F32_4X4X1_F32,
    AMD_V_MFMA_F32_16X16X4_F32,
    AMD_V_MFMA_F32_32X32X2_F32,
    AMD_V_MFMA_I32_4X4X4_I8,
    AMD_V_MFMA_I32_16X16X16_I8,
    AMD_V_MFMA_I32_32X32X8_I8,
    /* FP8/BF8 mixed-precision (gfx942 CDNA3) */
    AMD_V_MFMA_F32_16X16X32_FP8_FP8,
    AMD_V_MFMA_F32_16X16X32_FP8_BF8,
    AMD_V_MFMA_F32_16X16X32_BF8_FP8,
    AMD_V_MFMA_F32_16X16X32_BF8_BF8,
    AMD_V_MFMA_F32_32X32X16_FP8_FP8,
    AMD_V_MFMA_F32_32X32X16_FP8_BF8,
    AMD_V_MFMA_F32_32X32X16_BF8_FP8,
    AMD_V_MFMA_F32_32X32X16_BF8_BF8,
    /* F64 matrix (gfx942 CDNA3) */
    AMD_V_MFMA_F64_4X4X4_F64,
    AMD_V_MFMA_F64_16X16X4_F64,

    /* -- Pseudo-instructions (eliminated before emit) -- */
    AMD_PSEUDO_PHI,
    AMD_PSEUDO_COPY,
    AMD_PSEUDO_DEF,       /* defines a vreg (for phi elimination temps) */

    AMD_OP_COUNT
} amd_op_t;

/* ---- Machine Operand ---- */

typedef enum {
    MOP_NONE,
    MOP_SGPR,           /* physical SGPR */
    MOP_VGPR,           /* physical VGPR */
    MOP_VREG_S,         /* virtual scalar reg (maps to SGPR) */
    MOP_VREG_V,         /* virtual vector reg (maps to VGPR) */
    MOP_IMM,            /* 32-bit immediate */
    MOP_LABEL,          /* machine block index */
    MOP_SPECIAL         /* VCC, EXEC, SCC, M0 */
} mop_kind_t;

typedef struct {
    uint8_t  kind;       /* mop_kind_t */
    uint8_t  pad;
    uint16_t reg_num;    /* physical or virtual reg number */
    int32_t  imm;        /* immediate value or special reg ID */
} moperand_t;            /* 8 bytes */

/* ---- Machine Instruction ---- */

#define MINST_MAX_OPS 6

typedef struct {
    uint16_t    op;                      /* amd_op_t */
    uint8_t     num_defs;                /* outputs (0-2) */
    uint8_t     num_uses;                /* inputs (0-4) */
    moperand_t  operands[MINST_MAX_OPS]; /* [0..num_defs-1]=defs, rest=uses */
    uint16_t    flags;                   /* waitcnt encoding, glc, etc. */
} minst_t;   /* 56 bytes */

/* Instruction flags */
#define AMD_FLAG_GLC     0x0001
#define AMD_FLAG_SLC     0x0002
#define AMD_FLAG_DLC     0x0004

/* Waitcnt flag values (stored in flags for s_waitcnt) */
#define AMD_WAIT_VMCNT0       0x0100   /* wait for all VM ops */
#define AMD_WAIT_LGKMCNT0     0x0200   /* wait for all LDS/GDS/SMEM ops */
#define AMD_WAIT_ALL          0x0300   /* both */

/* ---- Machine Block / Function ---- */

typedef struct {
    uint32_t first_inst;
    uint32_t num_insts;
    uint32_t bir_block;   /* source BIR block index */
} mblock_t;  /* 12 bytes */

typedef struct {
    uint32_t name;             /* string table offset */
    uint32_t first_block;
    uint16_t num_blocks;
    uint16_t num_sgprs;        /* after regalloc */
    uint16_t num_vgprs;
    uint16_t lds_bytes;
    uint32_t scratch_bytes;
    uint32_t kernarg_bytes;
    uint16_t is_kernel;        /* 1 for __global__ */
    uint16_t wavefront_size;   /* 32 */
    uint16_t first_alloc_sgpr; /* first SGPR available to regalloc (after param pairs) */
    uint16_t bir_func;         /* BIR func index (for rplan BIR scan) */
    uint8_t  needs_dispatch;   /* 1 if kernel uses blockDim/gridDim (dispatch_ptr) */
    uint8_t  max_dim;          /* highest dim used: 0=x, 1=xy, 2=xyz */
    uint32_t launch_bounds_max; /* 0 = unconstrained. >0 = programmer's optimistic thread count */
    uint32_t launch_bounds_min; /* 0 = not set */

    /* Resource plan — stamped by amd_rplan(), read by isel + emit.
     * Target decisions made once.  No is_cdna() downstream.
     * Like pre-flight checks: argue with the checklist, not the runway. */
    uint8_t  exec_w;     /* 0=B32 (Wave32), 1=B64 (Wave64) */
    uint8_t  smem_hz;    /* 1=SMEM→SALU hazard, promote to VALU */
    uint8_t  scr_afs;    /* 1=architected flat scratch (no prologue) */
    uint8_t  rp_pad;
    uint16_t imp_sgp;    /* implicit system SGPRs (6 on CDNA, 0 on RDNA) */
    uint16_t sgp_min;    /* min SGPR block count for KD */
    uint32_t r1_mode;    /* RSRC1 static mode bits (IEEE, DX10, WGP, etc) */
} mfunc_t;

/* ---- Kernel Descriptor (64 bytes, AMD spec) ---- */

typedef struct {
    uint32_t group_segment_fixed_size;
    uint32_t private_segment_fixed_size;
    uint32_t kernarg_size;
    uint32_t reserved0;
    int64_t  kernel_code_entry_byte_offset;
    uint8_t  reserved1[20];
    uint32_t compute_pgm_rsrc3;
    uint32_t compute_pgm_rsrc1;
    uint32_t compute_pgm_rsrc2;
    uint16_t kernel_code_properties;
    uint8_t  reserved2[6];
} amd_kernel_descriptor_t;  /* 64 bytes */

/* ---- Module ---- */

#define AMD_MAX_MINSTS    (1 << 18)   /* 256K machine instructions */
#define AMD_MAX_MBLOCKS   (1 << 16)
#define AMD_MAX_MFUNCS    (1 << 12)
#define AMD_MAX_VREGS     (1 << 16)
#define AMD_CODE_SIZE     (4*1024*1024)
#define AMD_ASM_SIZE      (4*1024*1024)

typedef struct {
    const bir_module_t *bir;
    amd_target_t target;

    /* Chip-specific ELF metadata (set once by main, read by emit) */
    uint32_t    elf_mach;       /* e_flags value for this exact chip */
    char        chip_name[12];  /* "gfx1151" etc. */
    uint8_t     snap_mode;      /* 1 = photograph the suspects on entry */

    minst_t     minsts[AMD_MAX_MINSTS];
    uint32_t    num_minsts;

    mblock_t    mblocks[AMD_MAX_MBLOCKS];
    uint32_t    num_mblocks;

    mfunc_t     mfuncs[AMD_MAX_MFUNCS];
    uint32_t    num_mfuncs;

    /* Divergence: 1 bit per BIR instruction */
    uint32_t    divergent[AMD_MAX_MINSTS / 32];

    /* Virtual register tracking */
    uint32_t    vreg_count;
    uint16_t    reg_map[AMD_MAX_VREGS];    /* vreg -> phys reg */
    uint8_t     reg_file[AMD_MAX_VREGS];   /* 0=SGPR, 1=VGPR */
    uint32_t    vr_divg[AMD_MAX_VREGS / 32]; /* 1=divergent per vreg */

    /* BIR value -> machine vreg mapping */
    uint32_t    val_vreg[BIR_MAX_INSTS];   /* BIR inst index -> vreg */
    uint8_t     val_file[BIR_MAX_INSTS];   /* 0=scalar, 1=vector */
    uint16_t    val_sbase[BIR_MAX_INSTS];  /* SGPR pair base for pointers, 0xFFFF=none */
    int32_t     val_scroff[BIR_MAX_INSTS]; /* scratch byte offset, -1=dynamic */

    /* Instruction byte offsets (populated by encode_function) */
    uint32_t    inst_off[AMD_MAX_MINSTS];

    /* Output buffers */
    uint8_t     code[AMD_CODE_SIZE];
    uint32_t    code_len;

    char        asm_buf[AMD_ASM_SIZE];
    uint32_t    asm_len;
} amd_module_t;

/* ---- Encoding Table Entry ---- */

typedef struct {
    uint8_t  fmt;          /* amd_fmt_t */
    uint16_t hw_opcode;    /* hardware opcode field */
    const char *mnemonic;  /* assembly text name */
} amd_enc_entry_t;

/* ---- Public API ---- */

/* Resource planning: stamp target decisions onto mfuncs (before isel) */
void amd_rplan(amd_module_t *A);

/* Compile BIR module to AMDGCN machine IR (divergence + isel) */
int  amdgpu_compile(const bir_module_t *bir, amd_module_t *amd);

/* Post-isel passes: phi elimination + register allocation */
void amdgpu_phi_elim(amd_module_t *amd);
void amdgpu_regalloc(amd_module_t *amd);

/* Emit assembly text to stdout or file */
void amdgpu_emit_asm(const amd_module_t *amd, FILE *out);

/* Emit binary ELF code object (.hsaco) to file */
int  amdgpu_emit_elf(amd_module_t *amd, const char *path);

/* Set to 1 to force linear scan instead of graph coloring */
extern int amd_ra_lin;

/* Set to 1 for divergence-aware SSA register allocation */
extern int amd_ra_ssa;

/* If non-zero, cap available VGPRs for regalloc (forces spills for testing) */
extern int amd_max_vgpr;

/* ---- Divergence Helpers ---- */

static inline int vr_div(const amd_module_t *A, uint16_t vr)
{ return (int)((A->vr_divg[vr / 32] >> (vr % 32)) & 1u); }

static inline void vr_sdiv(amd_module_t *A, uint16_t vr)
{ A->vr_divg[vr / 32] |= 1u << (vr % 32); }

/* ---- Shared Helpers (emit.c, un-static for ra_ssa.c) ---- */

uint16_t op_vreg(const moperand_t *op);
void     rw_ops(amd_module_t *A, const mfunc_t *F);
void     dce_copy(amd_module_t *A, const mfunc_t *F);
void     fin_regs(const amd_module_t *A, mfunc_t *F);

/* SSA register allocator (ra_ssa.c) */
void ra_ssa(amd_module_t *A, uint32_t mf_idx);

/* Encoding table (defined in amdgpu_emit.c) */
extern const amd_enc_entry_t amd_enc_table[AMD_OP_COUNT];
extern const amd_enc_entry_t amd_enc_table_gfx10[AMD_OP_COUNT];
extern const amd_enc_entry_t amd_enc_table_gfx9[AMD_OP_COUNT];

#endif /* BARRACUDA_AMDGPU_H */
