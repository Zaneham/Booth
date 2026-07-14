#ifndef BARRACUDA_SEMA_H
#define BARRACUDA_SEMA_H

#include "barracuda.h"
#include "ast.h"
#include "parser.h"

/*
 * Semantic analysis — the adult supervision between parser and codegen.
 * Separate type system from BIR: preserves signedness, qualifiers, and
 * other details the lowerer was happily ignoring.
 */

/* ---- Limits ---- */

#define SEMA_MAX_TYPES    4096
#define SEMA_MAX_SYMS     4096
#define SEMA_MAX_SCOPES   128
#define SEMA_MAX_ERRORS   64
#define SEMA_MAX_STRUCTS  128
#define SEMA_MAX_FIELDS   16

#define BC_ERR_SEMA       -6

/* ---- Type System ---- */

typedef enum {
    STYPE_VOID,
    STYPE_BOOL,
    STYPE_CHAR,  STYPE_SCHAR,  STYPE_UCHAR,
    STYPE_SHORT, STYPE_USHORT,
    STYPE_INT,   STYPE_UINT,
    STYPE_LONG,  STYPE_ULONG,
    STYPE_LLONG, STYPE_ULLONG,
    STYPE_FLOAT, STYPE_DOUBLE, STYPE_LDOUBLE,
    STYPE_HALF,
    STYPE_BF16,
    STYPE_PTR,
    STYPE_ARRAY,
    STYPE_STRUCT,
    STYPE_ENUM,
    STYPE_FUNC,
    STYPE_VECTOR,
    STYPE_ERROR,   /* poison — any op on this silently produces STYPE_ERROR */
    STYPE_COUNT
} stype_kind_t;

/* Qualifier bits */
#define SQUAL_CONST     0x01
#define SQUAL_VOLATILE  0x02

/* 12 bytes, interned in pool. Deduplicated on insertion. */
typedef struct {
    uint8_t     kind;        /* stype_kind_t */
    uint8_t     qualifiers;  /* SQUAL_CONST, SQUAL_VOLATILE */
    uint16_t    width;       /* vector: lane count. array: element count (small) */
    uint32_t    inner;       /* ptr: pointee. array/vector: element. func: return type */
    uint32_t    extra;       /* struct: struct_def index. func: param list start in param_pool */
} stype_t;

/* ---- Struct Definition ---- */

typedef struct {
    char        name[128];
    uint32_t    field_types[SEMA_MAX_FIELDS];
    char        field_names[SEMA_MAX_FIELDS][64];
    int         num_fields;
} sema_struct_t;

/* ---- Symbol Table ---- */

typedef enum {
    SYM_VAR, SYM_FUNC, SYM_PARAM, SYM_TYPEDEF,
    SYM_STRUCT, SYM_ENUM, SYM_ENUM_CONST, SYM_TEMPLATE
} sym_kind_t;

typedef struct {
    char        name[128];
    uint32_t    type;        /* stype index */
    uint32_t    ast_node;    /* defining AST node */
    uint8_t     kind;        /* sym_kind_t */
    uint8_t     scope;       /* scope depth at definition */
    uint16_t    cuda_flags;  /* __device__, __global__, __shared__ etc. */
} sema_sym_t;

/* ---- Error (unified bc_error_t from barracuda.h) ---- */

/* ---- Function param pool ---- */

#define SEMA_MAX_PARAMS   2048

/* ---- Context ---- */

typedef struct {
    const parser_t  *P;
    const char      *src;

    /* Type pool — interned */
    stype_t         types[SEMA_MAX_TYPES];
    uint32_t        num_types;

    /* Symbol table — scoped, flat array */
    sema_sym_t      syms[SEMA_MAX_SYMS];
    uint32_t        num_syms;
    uint32_t        scope_stack[SEMA_MAX_SCOPES];
    uint32_t        scope_depth;

    /* Struct definitions */
    sema_struct_t   structs[SEMA_MAX_STRUCTS];
    uint32_t        num_structs;

    /* Function parameter type pool (for func types) */
    uint32_t        param_pool[SEMA_MAX_PARAMS];
    uint32_t        num_params;

    /* Side table: AST node index → stype index. 0 = unresolved. */
    uint32_t        node_types[BC_MAX_NODES];

    /* Current function return type (for return-statement checking) */
    uint32_t        cur_ret_type;

    /* Target setting for wavefront size */
    int             amd_target;

    /* Errors */
    bc_error_t      errors[SEMA_MAX_ERRORS];
    int             num_errors;
} sema_ctx_t;

/* ---- Public API ---- */

void sema_init(sema_ctx_t *S, const parser_t *P, uint32_t root, int amd_target);
int  sema_check(sema_ctx_t *S, uint32_t root);
void sema_dump(const sema_ctx_t *S, uint32_t root);

int  stype_str(const sema_ctx_t *S, uint32_t tidx, char *buf, int size);

#endif /* BARRACUDA_SEMA_H */
