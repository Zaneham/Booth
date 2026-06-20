#ifndef BARRACUDA_BIR_H
#define BARRACUDA_BIR_H

#include "barracuda.h"

/*
 * BarraCUDA Intermediate Representation.
 * SSA form, flat arrays, index-based references.
 * No malloc. No mercy. No hardware assumptions.
 */

/* ---- Limits ---- */

#define BIR_MAX_TYPES       (1 << 14)
#define BIR_MAX_TYPE_FIELDS (1 << 16)
#define BIR_MAX_CONSTS      (1 << 16)
#define BIR_MAX_INSTS       (1 << 18)
#define BIR_MAX_EXTRA_OPS   (1 << 16)
#define BIR_MAX_BLOCKS      (1 << 16)
#define BIR_MAX_FUNCS       (1 << 12)
#define BIR_MAX_GLOBALS     (1 << 12)
#define BIR_MAX_STRINGS     (1 << 20)

/* ---- Value References ---- */

/*
 * Operand slots are 32-bit tagged references.
 * Bit 31 distinguishes instruction values from constants.
 * Block references in branch targets are plain block indices,
 * identified by operand position (the opcode says which slot is a block).
 */

#define BIR_VAL_CONST_BIT   0x80000000u
#define BIR_VAL_INDEX(v)    ((v) & 0x7FFFFFFFu)
#define BIR_VAL_IS_CONST(v) ((v) & BIR_VAL_CONST_BIT)
#define BIR_MAKE_CONST(i)   (BIR_VAL_CONST_BIT | (i))
#define BIR_MAKE_VAL(i)     (i)
#define BIR_VAL_NONE        0xFFFFFFFFu

/* ---- Address Spaces ---- */

typedef enum {
    BIR_AS_PRIVATE,     /* per-thread stack */
    BIR_AS_SHARED,      /* per-block scratchpad */
    BIR_AS_GLOBAL,      /* device DRAM */
    BIR_AS_CONSTANT,    /* read-only, cached */
    BIR_AS_GENERIC,     /* unresolved — backend decides */
    BIR_AS_COUNT
} bir_addrspace_t;

/* ---- Type System ---- */

typedef enum {
    BIR_TYPE_VOID,
    BIR_TYPE_INT,       /* width: 1, 8, 16, 32, 64 bits */
    BIR_TYPE_FLOAT,     /* width: 16, 32, 64 bits */
    BIR_TYPE_BFLOAT,    /* width: 16 (brain float) */
    BIR_TYPE_PTR,       /* addrspace + pointee */
    BIR_TYPE_VECTOR,    /* <N x element> where N = width field */
    BIR_TYPE_STRUCT,    /* named or anonymous aggregate */
    BIR_TYPE_ARRAY,     /* [N x element] where N = count field */
    BIR_TYPE_FUNC,      /* return type + param types */
    BIR_TYPE_KIND_COUNT
} bir_type_kind_t;

typedef struct {
    uint8_t     kind;       /* bir_type_kind_t */
    uint8_t     addrspace;  /* PTR: address space */
    uint16_t    width;      /* INT/FLOAT: bit width. VECTOR: lane count */
    uint32_t    inner;      /* PTR: pointee. ARRAY/VECTOR: element. FUNC: return type */
    uint32_t    count;      /* ARRAY: element count. STRUCT/FUNC: fields_start index */
    uint16_t    num_fields; /* STRUCT/FUNC: field/param count */
    uint16_t    pad;
} bir_type_t;   /* 16 bytes */

/* ---- Comparison Predicates ---- */

typedef enum {
    /* Integer */
    BIR_ICMP_EQ, BIR_ICMP_NE,
    BIR_ICMP_SLT, BIR_ICMP_SLE, BIR_ICMP_SGT, BIR_ICMP_SGE,
    BIR_ICMP_ULT, BIR_ICMP_ULE, BIR_ICMP_UGT, BIR_ICMP_UGE,
    /* Float — O = ordered (NaN → false), U = unordered (NaN → true) */
    BIR_FCMP_OEQ, BIR_FCMP_ONE, BIR_FCMP_OLT, BIR_FCMP_OLE, BIR_FCMP_OGT, BIR_FCMP_OGE,
    BIR_FCMP_UEQ, BIR_FCMP_UNE, BIR_FCMP_ULT, BIR_FCMP_ULE, BIR_FCMP_UGT, BIR_FCMP_UGE,
    BIR_FCMP_ORD, BIR_FCMP_UNO,
    BIR_CMP_COUNT
} bir_cmp_pred_t;

/* ---- Memory Ordering ---- */

typedef enum {
    BIR_ORDER_RELAXED,
    BIR_ORDER_ACQUIRE,
    BIR_ORDER_RELEASE,
    BIR_ORDER_ACQ_REL,
    BIR_ORDER_SEQ_CST,
    BIR_ORDER_COUNT
} bir_mem_order_t;

/* ---- Opcodes ---- */

/* The vocabulary of a GPU that doesn't exist yet. */
typedef enum {
    /* Arithmetic */
    BIR_ADD, BIR_SUB, BIR_MUL,
    BIR_SDIV, BIR_UDIV, BIR_SREM, BIR_UREM,
    BIR_FADD, BIR_FSUB, BIR_FMUL, BIR_FDIV, BIR_FREM,

    /* Bitwise */
    BIR_AND, BIR_OR, BIR_XOR, BIR_SHL, BIR_LSHR, BIR_ASHR,

    /* Comparison — subop holds the predicate */
    BIR_ICMP, BIR_FCMP,

    /* Conversion */
    BIR_TRUNC, BIR_ZEXT, BIR_SEXT,
    BIR_FPTRUNC, BIR_FPEXT,
    BIR_FPTOSI, BIR_FPTOUI, BIR_SITOFP, BIR_UITOFP,
    BIR_PTRTOINT, BIR_INTTOPTR, BIR_BITCAST,

    /* Math */
    BIR_SQRT, BIR_RSQ, BIR_RCP,
    BIR_EXP2, BIR_LOG2,
    BIR_SIN, BIR_COS,
    BIR_FABS, BIR_FLOOR, BIR_CEIL, BIR_FTRUNC, BIR_RNDNE,
    BIR_FMAX, BIR_FMIN,

    /* Memory */
    BIR_ALLOCA,                 /* per-thread stack. subop = log2(align) */
    BIR_SHARED_ALLOC,           /* per-block SRAM. Not stack, not heap. Its own thing. */
    BIR_GLOBAL_REF,             /* subop = global index. Yields ptr to __device__/__constant__ global. */
    BIR_LOAD,                   /* subop: 0 = normal, 1 = volatile */
    BIR_STORE,                  /* subop: 0 = normal, 1 = volatile */
    BIR_GEP,                    /* base ptr + typed indices. Simpler than LLVM's. */

    /* Control flow */
    BIR_BR,                     /* ops[0] = block */
    BIR_BR_COND,                /* ops[0] = cond, [1] = true, [2] = false, [3] = merge */
    BIR_SWITCH,                 /* ops[0] = val, ops[1] = default block, rest in extra */
    BIR_RET,                    /* ops[0] = value (or num_operands=0 for void) */
    BIR_UNREACHABLE,

    /* SSA */
    BIR_PHI,                    /* (block, value) pairs */
    BIR_PARAM,                  /* function parameter. subop = param index */

    /* Thread model — hardware-agnostic. subop = dimension (0=x, 1=y, 2=z) */
    BIR_THREAD_ID,
    BIR_BLOCK_ID,
    BIR_BLOCK_DIM,
    BIR_GRID_DIM,

    /* Barriers — the IR says "synchronize here", backends figure out how */
    BIR_BARRIER,
    BIR_BARRIER_GROUP,

    /* Atomics — subop = memory ordering */
    BIR_ATOMIC_ADD, BIR_ATOMIC_SUB,
    BIR_ATOMIC_AND, BIR_ATOMIC_OR, BIR_ATOMIC_XOR,
    BIR_ATOMIC_MIN, BIR_ATOMIC_MAX,
    BIR_ATOMIC_XCHG, BIR_ATOMIC_CAS,
    BIR_ATOMIC_LOAD, BIR_ATOMIC_STORE,

    /* Warp-level primitives — Tenstorrent backend lowers to SRAM movement */
    BIR_SHFL, BIR_SHFL_UP, BIR_SHFL_DOWN, BIR_SHFL_XOR,
    BIR_BALLOT, BIR_VOTE_ANY, BIR_VOTE_ALL,

    /* Matrix */
    BIR_MFMA,                   /* subop = variant ID, ops: [0]=A, [1]=B, [2]=C(accum) */

    /* Misc */
    BIR_CALL,                   /* ops[0] = callee func index, rest = args */
    BIR_SELECT,                 /* ops[0] = cond, ops[1] = true, ops[2] = false */
    BIR_INLINE_ASM,             /* for when you need to talk directly to the metal */

    /* Wide multiply — high half of a full-width product. Keystone for
       256-bit field arithmetic (Montgomery): mul-lo + mul-hi + carry. */
    BIR_UMULHI,

    BIR_OP_IMPLEMENTED,

    /* Reserved: texture/surface ops */
    BIR_TEX_BASE = 128,
    BIR_TEX_MAX  = 143,

    BIR_OP_COUNT = 144
} bir_op_t;

/* ---- Instruction ---- */

/*
 * 32 bytes. Two per cache line. The prophecy of good alignment.
 *
 * Six inline operands covers 99% of real instructions.
 * When num_operands == OVERFLOW (phi nodes, large calls):
 *   ops[0] = index into extra_operands[]
 *   ops[1] = count of extra operands
 */

#define BIR_OPERANDS_INLINE     6
#define BIR_OPERANDS_OVERFLOW   0xFF

typedef struct {
    uint16_t    op;             /* bir_op_t */
    uint8_t     num_operands;   /* inline count (0-6), or OVERFLOW */
    uint8_t     subop;          /* predicate, dimension, ordering, volatile, align */
    uint32_t    type;           /* result type index */
    uint32_t    operands[BIR_OPERANDS_INLINE];
} bir_inst_t;   /* 32 bytes */

/* ---- Constants ---- */

/* Values that were decided before time began. */
typedef enum {
    BIR_CONST_INT,
    BIR_CONST_FLOAT,
    BIR_CONST_NULL,
    BIR_CONST_UNDEF,
    BIR_CONST_ZERO,
    BIR_CONST_BYTES,            /* d.bytes.off/len index the strings table */
    BIR_CONST_KIND_COUNT
} bir_const_kind_t;

typedef struct {
    uint8_t     kind;
    uint8_t     pad[3];
    uint32_t    type;
    union {
        int64_t     ival;
        double      fval;
        struct { uint32_t off; uint32_t len; } bytes;
    } d;
} bir_const_t;  /* 16 bytes */

/* ---- Basic Blocks ---- */

typedef struct {
    uint32_t    name;           /* string table offset (null-terminated) */
    uint32_t    first_inst;     /* index into insts[] */
    uint32_t    num_insts;
} bir_block_t;  /* 12 bytes */

/* ---- Functions ---- */

typedef struct {
    uint32_t    name;           /* string table offset */
    uint32_t    type;           /* function type index */
    uint32_t    first_block;    /* index into blocks[] */
    uint32_t    total_insts;    /* sum of all block num_insts */
    uint16_t    num_blocks;
    uint16_t    num_params;
    uint16_t    cuda_flags;     /* CUDA_GLOBAL, CUDA_DEVICE, CUDA_HOST from ast.h */
    uint16_t    pad;
    uint32_t    launch_bounds_max;
    uint32_t    launch_bounds_min;
} bir_func_t;   /* 32 bytes */

/* ---- Global Variables ---- */

typedef struct {
    uint32_t    name;           /* string table offset */
    uint32_t    type;           /* type index */
    uint32_t    initializer;    /* constant ref (BIR_MAKE_CONST), or BIR_VAL_NONE */
    uint16_t    cuda_flags;
    uint8_t     addrspace;      /* bir_addrspace_t */
    uint8_t     is_const;
} bir_global_t;  /* 16 bytes */

/* ---- Module ---- */

/* The whole program in one struct. No malloc. Deterministic layout. */
typedef struct {
    bir_type_t      types[BIR_MAX_TYPES];
    uint32_t        type_fields[BIR_MAX_TYPE_FIELDS];
    uint32_t        num_types;
    uint32_t        num_type_fields;

    bir_const_t     consts[BIR_MAX_CONSTS];
    uint32_t        num_consts;

    bir_inst_t      insts[BIR_MAX_INSTS];
    uint32_t        inst_lines[BIR_MAX_INSTS];  /* source line per instruction — the breadcrumbs home */
    uint32_t        extra_operands[BIR_MAX_EXTRA_OPS];
    uint32_t        num_insts;
    uint32_t        num_extra_ops;

    bir_block_t     blocks[BIR_MAX_BLOCKS];
    uint32_t        num_blocks;

    bir_func_t      funcs[BIR_MAX_FUNCS];
    uint32_t        num_funcs;

    bir_global_t    globals[BIR_MAX_GLOBALS];
    uint32_t        num_globals;

    char            strings[BIR_MAX_STRINGS];
    uint32_t        string_len;
} bir_module_t;

/* ---- API ---- */

void        bir_module_init(bir_module_t *M);

/* Type interning — returns index of existing or newly created type */
uint32_t    bir_type_void(bir_module_t *M);
uint32_t    bir_type_int(bir_module_t *M, int width_bits);
uint32_t    bir_type_float(bir_module_t *M, int width_bits);
uint32_t    bir_type_bfloat(bir_module_t *M);
uint32_t    bir_type_ptr(bir_module_t *M, uint32_t pointee, int addrspace);
uint32_t    bir_type_array(bir_module_t *M, uint32_t elem, uint32_t count);
uint32_t    bir_type_vector(bir_module_t *M, uint32_t elem, uint32_t count);
uint32_t    bir_type_struct(bir_module_t *M, const uint32_t *fields, int nfields);
uint32_t    bir_type_func(bir_module_t *M, uint32_t ret,
                          const uint32_t *params, int nparams);

/* String table */
uint32_t    bir_add_string(bir_module_t *M, const char *s, uint32_t len);

/* Constants — returns index into consts[], use BIR_MAKE_CONST() for operand slots */
uint32_t    bir_const_int(bir_module_t *M, uint32_t type, int64_t val);
uint32_t    bir_const_float(bir_module_t *M, uint32_t type, double val);
uint32_t    bir_const_null(bir_module_t *M, uint32_t type);
uint32_t    bir_const_bytes(bir_module_t *M, uint32_t type,
                            uint32_t off, uint32_t len);

/* True if a global has a BIR_CONST_BYTES initializer (i.e. is a
 * string literal or other read-only byte array). Backends without
 * Phase-2 support gate on this and refuse rather than producing
 * silent wrong output. */
int         bir_global_is_bytes(const bir_module_t *M, uint32_t gi);

/* Name tables */
const char *bir_op_name(int op);
const char *bir_type_kind_name(int kind);
const char *bir_cmp_name(int pred);
const char *bir_addrspace_name(int as);
const char *bir_order_name(int ord);

/* Printer */
int  bir_type_str(const bir_module_t *M, uint32_t tidx, char *buf, int size);
void bir_print_module(const bir_module_t *M, FILE *out);

#endif /* BARRACUDA_BIR_H */
