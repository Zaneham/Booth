#ifndef BARRACUDA_NVIDIA_H
#define BARRACUDA_NVIDIA_H

#include "bir.h"
#include <stdio.h>

/* NVIDIA PTX backend. The irony of an open-source CUDA compiler targeting
 * NVIDIA hardware is not lost on us. Think of it as returning a library
 * book — to a library that charges admission and checks your bag. */


/* ---- PTX Opcodes ---- */
/* Tags for the text emitter, not real machine opcodes. PTX is already
 * a text IR — we're generating text from an IR to feed to a JIT that
 * generates the actual machine code. It's turtles all the way down. */

typedef enum {
    /* Integer arithmetic */
    NV_ADD_U32 = 0,  NV_ADD_U64,  NV_ADD_S32,
    NV_SUB_U32,      NV_SUB_S32,  NV_SUB_S64,
    NV_MUL_LO_U32,   NV_MUL_LO_S32,  NV_MUL_LO_U64,
    NV_MUL_HI_U32,   NV_MUL_HI_S32,  NV_MUL_HI_U64,
    NV_MAD_LO_U64,   /* mad.lo.u64 for GEP */
    NV_DIV_U32,       NV_DIV_S32,
    NV_REM_U32,       NV_REM_S32,
    NV_NEG_S32,

    /* FP arithmetic */
    NV_ADD_F32,  NV_ADD_F64,
    NV_SUB_F32,  NV_SUB_F64,
    NV_MUL_F32,  NV_MUL_F64,
    NV_DIV_F32,  NV_DIV_F64,
    NV_FMA_F32,  NV_FMA_F64,
    NV_NEG_F32,  NV_NEG_F64,
    NV_ABS_F32,  NV_ABS_F64,

    /* Logic / shift */
    NV_AND_B32,  NV_AND_B64,
    NV_OR_B32,   NV_OR_B64,
    NV_XOR_B32,  NV_XOR_B64,
    NV_NOT_B32,  NV_NOT_B64,
    NV_SHL_B32,  NV_SHL_B64,
    NV_SHR_U32,  NV_SHR_S32,  NV_SHR_U64,

    /* Bit counting. No ctz in PTX, so that one is built from brev + clz. */
    NV_POPC_B32, NV_POPC_B64,
    NV_CLZ_B32,  NV_CLZ_B64,
    NV_BREV_B32, NV_BREV_B64,

    /* Comparison — setp */
    NV_SETP_EQ_U32,  NV_SETP_NE_U32,
    NV_SETP_LT_U32,  NV_SETP_LE_U32,
    NV_SETP_GT_U32,  NV_SETP_GE_U32,
    NV_SETP_LT_S32,  NV_SETP_LE_S32,
    NV_SETP_GT_S32,  NV_SETP_GE_S32,
    NV_SETP_EQ_F32,  NV_SETP_NE_F32,
    NV_SETP_LT_F32,  NV_SETP_LE_F32,
    NV_SETP_GT_F32,  NV_SETP_GE_F32,
    NV_SETP_EQ_F64,  NV_SETP_NE_F64,
    NV_SETP_LT_F64,  NV_SETP_LE_F64,
    NV_SETP_GT_F64,  NV_SETP_GE_F64,
    NV_SETP_EQ_U64,  NV_SETP_NE_U64,

    /* Select / predicated move */
    NV_SELP_U32,  NV_SELP_U64,
    NV_SELP_F32,  NV_SELP_F64,

    /* Moves */
    NV_MOV_U32,  NV_MOV_U64,
    NV_MOV_F32,  NV_MOV_F64,
    NV_MOV_PRED,

    /* Conversions */
    NV_CVT_U32_F32,  NV_CVT_S32_F32,  /* fptosi/fptoui (f32 src) */
    NV_CVT_U32_F64,  NV_CVT_S32_F64,  /* fptosi/fptoui (f64 src) */
    NV_CVT_F32_U32,  NV_CVT_F32_S32,  /* uitofp/sitofp */
    NV_CVT_F32_F64,  NV_CVT_F64_F32,  /* fptrunc/fpext */
    NV_CVT_U64_U32,  NV_CVT_S64_S32,  /* zext/sext to 64 */
    NV_CVT_U32_U64,                    /* trunc 64->32 */
    NV_CVT_U64_F64,  NV_CVT_S64_F64,  /* fp64->int64 */
    NV_CVT_F64_U64,  NV_CVT_F64_S64,  /* int64->fp64 */
    NV_CVT_F64_U32,  NV_CVT_F64_S32,  /* int32->fp64 */
    NV_CVT_F32_F16,  NV_CVT_F16_F32,  /* half conversions */

    /* Loads / stores — global */
    NV_LD_GLB_U32,  NV_LD_GLB_U64,
    NV_LD_GLB_F32,  NV_LD_GLB_F64,
    NV_LD_GLB_U8,   NV_LD_GLB_U16,  NV_LD_GLB_B16,
    NV_ST_GLB_U32,  NV_ST_GLB_U64,
    NV_ST_GLB_F32,  NV_ST_GLB_F64,
    NV_ST_GLB_U8,   NV_ST_GLB_U16,  NV_ST_GLB_B16,

    /* Loads / stores — shared */
    NV_LD_SHR_U32,  NV_LD_SHR_F32,  NV_LD_SHR_U8,
    NV_LD_SHR_U16,  NV_LD_SHR_B16,
    NV_ST_SHR_U32,  NV_ST_SHR_F32,  NV_ST_SHR_U8,
    NV_ST_SHR_U16,  NV_ST_SHR_B16,

    /* Loads / stores — local (scratch / alloca) */
    NV_LD_LOC_U32,  NV_LD_LOC_U64,
    NV_LD_LOC_F32,  NV_LD_LOC_F64,  NV_LD_LOC_U8,
    NV_LD_LOC_U16,  NV_LD_LOC_B16,
    NV_ST_LOC_U32,  NV_ST_LOC_U64,
    NV_ST_LOC_F32,  NV_ST_LOC_F64,  NV_ST_LOC_U8,
    NV_ST_LOC_U16,  NV_ST_LOC_B16,

    /* Parameter loads */
    NV_LD_PARAM_U32,  NV_LD_PARAM_U64,
    NV_LD_PARAM_F32,  NV_LD_PARAM_F64,

    /* Atomics — flags carry the address space, NV_ASP_* below */
    NV_ATOM_ADD_U32,  NV_ATOM_ADD_F32,
    NV_ATOM_MIN_U32,  NV_ATOM_MAX_U32,
    NV_ATOM_AND_B32,  NV_ATOM_OR_B32,  NV_ATOM_XOR_B32,
    NV_ATOM_XCHG_B32, NV_ATOM_CAS_B32,
    NV_ATOM_ADD_U64,  NV_ATOM_ADD_F64,
    NV_ATOM_AND_B64,  NV_ATOM_OR_B64,  NV_ATOM_XOR_B64,
    NV_ATOM_XCHG_B64, NV_ATOM_CAS_B64,

    /* Branches */
    NV_BRA,           /* bra $label */
    NV_BRA_PRED,      /* @%p bra $label */

    /* Barriers */
    NV_BAR_SYNC,      /* bar.sync 0 */
    NV_MEMBAR,
    NV_NANOSLP,
    NV_BARRED_OR,
    NV_BARRED_AND,
    NV_BARRED_POPC,

    /* Warp ops */
    NV_SHFL_IDX,      /* shfl.sync.idx.b32 */
    NV_SHFL_UP,       /* shfl.sync.up.b32 */
    NV_SHFL_DOWN,     /* shfl.sync.down.b32 */
    NV_SHFL_XOR,      /* shfl.sync.bfly.b32 */
    NV_VOTE_BALLOT,   /* vote.sync.ballot.b32 */
    NV_VOTE_ANY,      /* vote.sync.any.pred */
    NV_VOTE_ALL,      /* vote.sync.all.pred */

    /* Math builtins */
    NV_SQRT_F32,      /* sqrt.approx.f32 */
    NV_SQRT_F64,      /* sqrt.rn.f64 */
    NV_RSQ_F32,       /* rsqrt.approx.f32 */
    NV_RCP_F32,       /* rcp.approx.f32 */
    NV_SIN_F32,       /* sin.approx.f32 */
    NV_COS_F32,       /* cos.approx.f32 */
    NV_EX2_F32,       /* ex2.approx.f32 (no f64 in PTX) */
    NV_LG2_F32,       /* lg2.approx.f32 */
    NV_FLOOR_F32,     /* cvt.rmi.f32.f32 (floor) */
    NV_CEIL_F32,      /* cvt.rpi.f32.f32 (ceil) */
    NV_TRUNC_F32,     /* cvt.rzi.f32.f32 (trunc) */
    NV_ROUND_F32,     /* cvt.rni.f32.f32 (round nearest) */
    NV_MIN_F32,       /* min.f32 */
    NV_MAX_F32,       /* max.f32 */
    NV_MIN_U32,       /* min.u32 */
    NV_MAX_U32,       /* max.u32 */
    NV_MIN_S32,       /* min.s32 */
    NV_MAX_S32,       /* max.s32 */

    /* Pseudo-ops */
    NV_RET,           /* ret; */
    NV_EXIT,          /* exit; */
    NV_MOV_F64_LIT,   /* mov.f64 %fd, 0dXXXX — ops[0]=dst, ops[1].imm=hi32, ops[2].imm=lo32 */
    NV_LEA_LOCAL,     /* mov.u64 %rd, __local+off — ops[0]=dst, ops[1].imm=byte offset */
    NV_LEA_GLB,       /* mov.u64 %rd, <sym> — ops[0]=dst, ops[1].imm=global index */
    NV_LEA_DSH,       /* mov.u64 %rd, __dynshmem - the dynamic shared base */
    NV_MOV_PK16,      /* mov.b32 %rb, {%rh_lo, %rh_hi} — packs two halves */
    NV_MMA,           /* mma.sync.aligned.<shape>.row.col.f32.<t>.<t>.f32.
                       * ops are the BASE register of each fragment tuple:
                       * [0]=D [1]=A [2]=B [3]=C; flags = nv_mmash_t index */
    NV_WLD,
    NV_WST,
    NV_WMMA,
    NV_ASM,           /* verbatim inline asm text; flags = nv->asms index */
    NV_BARWARP,       /* bar.warp.sync 0xffffffff */
    NV_TRAP,          /* trap; */
    NV_CVTA_GLB,
    NV_CVTA_LOC,
    NV_CALL,
    NV_ST_RETP,
    NV_CVT_U32_U16,

    NV_SETP_LT_S64,  NV_SETP_LE_S64,  NV_SETP_GT_S64,  NV_SETP_GE_S64,
    NV_SETP_LT_U64,  NV_SETP_LE_U64,  NV_SETP_GT_U64,  NV_SETP_GE_U64,
    NV_DIV_U64,      NV_DIV_S64,      NV_REM_U64,      NV_REM_S64,
    NV_NEG_S64,      NV_SHR_S64,
    NV_MOV_B16,      NV_MOV_B32,      NV_MOV_B64,
    NV_CVT_S32_S16,  NV_CVT_U16_U32,
    NV_CVT_F32_U64,  NV_CVT_F32_S64,  NV_CVT_U64_F32,  NV_CVT_S64_F32,
    NV_CVT_F16_F64,  NV_CVT_F64_F16,
    NV_CVT_F32_BF16, NV_CVT_BF16_F32,
    NV_LD_PARAM_B16,
    NV_LD_SHR_U64,   NV_LD_SHR_F64,   NV_ST_SHR_U64,   NV_ST_SHR_F64,
    NV_MIN_U64,      NV_MAX_U64,      NV_MIN_S64,      NV_MAX_S64,
    NV_GBAR,

    NV_OP_COUNT
} nv_ptx_op_t;

#define NV_ASP_GLB  0
#define NV_ASP_SHR  1
#define NV_ASP_GEN  2

#define NV_SCOPE_CTA  0
#define NV_SCOPE_GPU  1
#define NV_SCOPE_SYS  2

/* ---- Register File Codes ---- */

typedef enum {
    NV_RF_U32  = 0,   /* %r<N>  — 32-bit integer */
    NV_RF_U64  = 1,   /* %rd<N> — 64-bit integer */
    NV_RF_F32  = 2,   /* %f<N>  — 32-bit float */
    NV_RF_F64  = 3,   /* %fd<N> — 64-bit float */
    NV_RF_PRED = 4,   /* %p<N>  — predicate */
    NV_RF_U16  = 5,   /* %rh<N> — 16-bit integer */
    NV_RF_F16  = 6,   /* %h<N>  — 16-bit float */
    NV_RF_B32  = 7,   /* %rb<N> — untyped 32-bit, mma fragment halves */
    NV_RF_COUNT
} nv_rfile_t;

/* ---- Machine Operand ---- */

typedef enum {
    NV_MOP_NONE = 0,
    NV_MOP_REG,        /* virtual register */
    NV_MOP_IMM,        /* 32-bit immediate */
    NV_MOP_IMM64,      /* 64-bit immediate (stored in imm64) */
    NV_MOP_LABEL,      /* block label index */
    NV_MOP_SPEC,       /* special register (tid, ctaid, ntid) */
} nv_mop_t;

/* Special register IDs — kept flat, not an enum, so
 * we can pack them into the imm field. */
#define NV_SPEC_TID_X     0
#define NV_SPEC_TID_Y     1
#define NV_SPEC_TID_Z     2
#define NV_SPEC_CTAID_X   3
#define NV_SPEC_CTAID_Y   4
#define NV_SPEC_CTAID_Z   5
#define NV_SPEC_NTID_X    6
#define NV_SPEC_NTID_Y    7
#define NV_SPEC_NTID_Z    8
#define NV_SPEC_NCTAID_X  9
#define NV_SPEC_NCTAID_Y  10
#define NV_SPEC_NCTAID_Z  11
#define NV_SPEC_LANEID    12

typedef struct {
    const char *sfx;     /* everything between "aligned." and the operands */
    uint8_t     na, nb;  /* A and B elements per lane */
} nv_mmash_t;

#define NV_MMA_NSHAPE 4
extern const nv_mmash_t nv_mmash[NV_MMA_NSHAPE];

#define NV_WM_ROLE_A BIR_WM_A
#define NV_WM_ROLE_B BIR_WM_B
#define NV_WM_ROLE_C BIR_WM_C
#define NV_WM_ROLE_D BIR_WM_D

#define NV_WM_MKF(r, al, bl, ro, bo) ((uint16_t)(((r) & 31u) | (((al) & 1u) << 5) | (((bl) & 1u) << 6) | (((ro) & 3u) << 7) | (((bo) & 1u) << 9)))

#define NV_WM_ROW(f)  ((unsigned)(f) & 31u)
#define NV_WM_ALAY(f) (((unsigned)(f) >> 5) & 1u)
#define NV_WM_BLAY(f) (((unsigned)(f) >> 6) & 1u)
#define NV_WM_ROLE(f) (((unsigned)(f) >> 7) & 3u)
#define NV_WM_BOP(f)  (((unsigned)(f) >> 9) & 1u)

uint8_t nv_wmrf(uint32_t row);

typedef struct {
    uint8_t  kind;       /* nv_mop_t */
    uint8_t  rfile;      /* nv_rfile_t */
    uint16_t reg_num;
    int32_t  imm;
} nv_opnd_t;

/* ---- Machine Instruction ---- */

#define NV_MAX_FRAME (512u * 1024u)
#define NV_MAX_ASM    4096
#define NV_MAX_ASMOP  BIR_ASM_MAXOP

#define NV_MAX_OPS  6

typedef struct {
    uint16_t  op;        /* nv_ptx_op_t */
    uint8_t   num_defs;
    uint8_t   num_uses;
    nv_opnd_t ops[NV_MAX_OPS];
    uint16_t  flags;
    uint16_t  pad;
} nv_minst_t;

#define NV_MAX_CALL  4096
#define NV_MAX_CARG    24

typedef struct {
    uint32_t   fn;
    uint8_t    nargs;
    uint8_t    retrf;
    uint8_t    vprt;
    uint8_t    hasret;
    nv_opnd_t  ret;
    nv_opnd_t  args[NV_MAX_CARG];
} nv_call_t;

typedef struct {
    uint32_t   tmpl;     /* BIR strings offset of the template text */
    uint8_t    nops;
    uint8_t    pad[3];
    nv_opnd_t  ops[NV_MAX_ASMOP];
} nv_asm_t;

/* ---- Machine Block / Function ---- */

typedef struct {
    uint32_t first_inst;
    uint32_t num_insts;
    uint32_t bir_block;
} nv_mblk_t;

/* Per-function param descriptor for PTX .param declarations */
#define NV_MAX_PARAMS  128

typedef struct {
    uint32_t name;       /* string table offset */
    uint8_t  rfile;      /* NV_RF_U32/U64/F32/F64 */
    uint8_t  pad[3];
} nv_param_t;

typedef struct {
    uint32_t    name;        /* string table offset */
    uint32_t    first_blk;
    uint16_t    num_blks;
    uint16_t    is_kern;
    uint32_t    num_params;
    nv_param_t  params[NV_MAX_PARAMS];
    uint32_t    lds_bytes;   /* shared memory size */
    uint32_t    lcl_bytes;   /* local (stack) memory per thread */
    uint32_t    lds_alg;
    uint32_t    lcl_alg;
    uint32_t    launch_max;  /* launch_bounds max threads */
    uint32_t    launch_min;  /* launch_bounds min blocks */
    uint32_t    bir_func;
    uint16_t    rc[NV_RF_COUNT];  /* per-rfile vreg high-water */
    uint8_t     retrf;
    uint8_t     hasret;
} nv_mfunc_t;

/* ---- Module ---- */

#define NV_MAX_MINST  (1 << 18)   /* 262144 */
#define NV_MAX_MBLK   (1 << 16)   /* 65536 */
#define NV_MAX_MFUNC  (1 << 12)   /* 4096 */
#define NV_MAX_OUT    (2 * 1024 * 1024)  /* 2 MB output buffer */

typedef struct {
    const bir_module_t *bir;

    nv_minst_t  minsts[NV_MAX_MINST];
    uint32_t    num_minst;

    nv_mblk_t   mblks[NV_MAX_MBLK];
    uint32_t    num_mblk;

    nv_mfunc_t  mfuncs[NV_MAX_MFUNC];
    uint32_t    num_mfunc;

    /* Virtual register counters — one per register file */
    uint16_t    rc[NV_RF_COUNT];

    /* BIR inst index → vreg number + rfile */
    uint16_t    val_vreg[BIR_MAX_INSTS];
    uint8_t     val_rfile[BIR_MAX_INSTS];

    nv_asm_t    asms[NV_MAX_ASM];
    uint32_t    num_asm;

    nv_call_t   calls[NV_MAX_CALL];
    uint32_t    num_call;
    uint32_t    cur_emf;
    uint8_t     vprt;
    uint8_t     vp_pad[3];

    /* Block-hit instrumentation (diagnostic) */
    uint8_t     bkhit;      /* emit atom.add at each block label */
    uint8_t     badop;      /* emitter met a machine op it cannot print */
    uint8_t     ovf;        /* the PTX text outgrew out_buf */
    uint8_t     gbar;

    uint32_t    gbn;
    uint32_t    dyn_alg;    /* extern __shared__ alignment, 0 if none */

    /* Output text buffer */
    char        out_buf[NV_MAX_OUT];
    uint32_t    out_len;
    uint32_t    out_tot;
    FILE       *out_fp;
    uint8_t     werr;
    uint8_t     wr_pad[3];
} nv_module_t;

/* ---- Public API ---- */

#define NV_SYM_MAX  BIR_SYM_MAX

int  nv_gsym(const bir_module_t *M, uint32_t gi, char *out, int size);
int  nv_compile(const bir_module_t *bir, nv_module_t *nv);
int  nv_vchk(const nv_module_t *nv);
int  nv_sass(nv_module_t *nv, const char *path, int text);
int  nv_emit_ptx(nv_module_t *nv, const char *path);

#endif /* BARRACUDA_NVIDIA_H */
