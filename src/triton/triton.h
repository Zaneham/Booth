#ifndef BARRACUDA_TRITON_H
#define BARRACUDA_TRITON_H

#include "barracuda.h"
#include "bir.h"
#include <stdio.h>

/* Triton frontend, by which we mean the part of BarraCUDA that accepts
 * the dialect of Python that OpenAI saw fit to bolt a GPU compiler onto
 * and translates it into BIR, the same way our existing frontend
 * translates CUDA. Triton is the lingua franca of contemporary AI
 * kernel work, and the cost of admission to that conversation is that
 * we now have to parse Python, a language which decided that
 * whitespace ought to be load bearing and that the resulting world
 * would be a better place. We disagree, but we are not in charge of
 * what the world has settled on, and so here we are.
 *
 * The frontend is deliberately self contained. We do not embed CPython,
 * we do not call out to a Python process at build time, and we do not
 * vendor in any third party Python parser. We write our own tokenizer
 * and our own recursive descent parser in C99, in the same shape as the
 * existing C99 frontend, because that is what every other part of the
 * compiler does and the project's character is consistency. The subset
 * of Python that Triton accepts is small enough to make this tractable:
 * no classes inside kernels, no exceptions, no generators, no async,
 * no lambdas, no decorators other than @triton.jit, and almost none of
 * Python's enormous standard library. What survives is closer to
 * ALGOL with colons than to Python the language. */

#define BC_ERR_TRITON   -10

/* ---- Limits ----
 * Fixed pools, no malloc on the hot path. The Triton kernels people
 * actually write fit comfortably inside these. The numbers are sized
 * the same way the C99 frontend's are: liberal enough to never trip
 * on real code, mean enough that an attacker cannot waste our day. */

#define TN_MAX_TOKENS       (1 << 18)
#define TN_MAX_NODES        (1 << 18)
#define TN_MAX_INDENTS      64
#define TN_MAX_PAREN_DEPTH  64
#define TN_MAX_STRINGS      (1 << 16)

/* ---- Token Kinds ----
 * Python tokens, with our own naming to keep them clear of the C99
 * token enum living next door in fe/token.h. The four indentation
 * related kinds (NEWLINE, INDENT, DEDENT, EOF) are how the lexer
 * tells the parser about block structure, because Python decided
 * curly braces were vulgar. */

typedef enum {
    TN_TOK_NEWLINE = 0,         /* end of a logical line */
    TN_TOK_INDENT,              /* block opens, deeper indentation */
    TN_TOK_DEDENT,              /* block closes, shallower indentation */

    TN_TOK_IDENT,               /* names and reserved words live here */
    TN_TOK_INT,                 /* decimal, hex 0x, octal 0o, binary 0b */
    TN_TOK_FLOAT,               /* 1.5, 1.5e10, .5, 5. */
    TN_TOK_STRING,              /* "...", '...', simple form only for now */

    /* Reserved words. We tokenize identifiers and then look them up in
     * the keyword table during the same pass, which keeps the token
     * count down and the parser case statements honest. */
    TN_TOK_KW_DEF, TN_TOK_KW_RETURN, TN_TOK_KW_PASS,
    TN_TOK_KW_IF, TN_TOK_KW_ELIF, TN_TOK_KW_ELSE,
    TN_TOK_KW_FOR, TN_TOK_KW_WHILE, TN_TOK_KW_BREAK, TN_TOK_KW_CONTINUE,
    TN_TOK_KW_IN, TN_TOK_KW_NOT, TN_TOK_KW_AND, TN_TOK_KW_OR, TN_TOK_KW_IS,
    TN_TOK_KW_NONE, TN_TOK_KW_TRUE, TN_TOK_KW_FALSE,
    TN_TOK_KW_IMPORT, TN_TOK_KW_FROM, TN_TOK_KW_AS,
    TN_TOK_KW_LAMBDA, TN_TOK_KW_GLOBAL, TN_TOK_KW_NONLOCAL,
    TN_TOK_KW_ASSERT, TN_TOK_KW_DEL,

    /* Punctuation */
    TN_TOK_LPAREN, TN_TOK_RPAREN,         /* ( ) */
    TN_TOK_LBRACK, TN_TOK_RBRACK,         /* [ ] */
    TN_TOK_LBRACE, TN_TOK_RBRACE,         /* { } */
    TN_TOK_COLON, TN_TOK_COMMA,           /* : , */
    TN_TOK_SEMI, TN_TOK_DOT,              /* ; . */
    TN_TOK_AT, TN_TOK_ARROW,              /* @ -> */
    TN_TOK_ASSIGN,                        /* = */

    /* Augmented assignment */
    TN_TOK_AADD, TN_TOK_ASUB, TN_TOK_AMUL, TN_TOK_ADIV,
    TN_TOK_AFDIV, TN_TOK_AMOD, TN_TOK_APOW,
    TN_TOK_AAND, TN_TOK_AOR, TN_TOK_AXOR,
    TN_TOK_ASHL, TN_TOK_ASHR,
    TN_TOK_AMATMUL,                       /* @= */

    /* Binary operators */
    TN_TOK_PLUS, TN_TOK_MINUS, TN_TOK_STAR, TN_TOK_SLASH,
    TN_TOK_FSLASH,                        /* // floor division */
    TN_TOK_PERCENT, TN_TOK_DSTAR,         /* ** power */
    TN_TOK_AMP, TN_TOK_PIPE, TN_TOK_CARET,
    TN_TOK_TILDE, TN_TOK_SHL, TN_TOK_SHR,

    /* Comparisons */
    TN_TOK_LT, TN_TOK_GT, TN_TOK_LE, TN_TOK_GE,
    TN_TOK_EQ, TN_TOK_NE,

    /* Misc */
    TN_TOK_WALRUS,                        /* := */
    TN_TOK_EOF,

    TN_TOK_COUNT
} tn_tok_kind_t;

typedef struct {
    int         kind;       /* tn_tok_kind_t */
    uint32_t    off;        /* offset into source */
    uint32_t    len;        /* length in source */
    uint32_t    line;
    uint16_t    col;
    uint16_t    pad;
} tn_tok_t;

/* ---- Lexer State ----
 * The Python tokenizer keeps a stack of column positions for the
 * currently open indentation levels, plus a separate count of paren
 * depth because lines inside brackets or parens are joined together
 * across newlines without emitting NEWLINE tokens, on the grounds
 * that the programmer evidently has not finished saying what they
 * meant to say. The pending_dedents counter handles the case where
 * dedenting closes several block levels at once and we need to emit
 * a run of DEDENT tokens before the next real token shows up. */

typedef struct {
    const char *src;
    uint32_t    src_len;
    uint32_t    pos;
    uint32_t    line;
    uint32_t    line_start;

    int         indents[TN_MAX_INDENTS];
    int         indent_depth;
    int         paren_depth;
    int         pending_dedents;
    int         at_line_start;          /* expecting indentation at next read */

    tn_tok_t   *tokens;
    uint32_t    num_tokens;
    uint32_t    max_tokens;

    bc_error_t  errors[BC_MAX_ERRORS];
    int         num_errors;
} tn_lex_t;

/* ---- Public API: Lexer ---- */

void        tn_lex_init(tn_lex_t *L, const char *src, uint32_t len,
                        tn_tok_t *tokens, uint32_t max_tokens);
int         tn_tokenize(tn_lex_t *L);
const char *tn_tok_name(int kind);
int         tn_tok_text(const tn_lex_t *L, const tn_tok_t *tok,
                        char *buf, int bufsize);

/* ---- AST Node Kinds ----
 * The shape of the tree the parser builds. Function definitions,
 * statements, parameters, dotted names, and the opaque expression
 * span used by sitting one to defer the real expression parser to a
 * later visit. Anything that does not have a more specific node kind
 * arrives at the sema stage as a TN_NK_EXPR_SPAN, which records the
 * first and last tokens of an expression and trusts the next pass
 * to do something more interesting with them. */

typedef enum {
    TN_NK_MODULE = 0,       /* program root */
    TN_NK_IMPORT,           /* import dotted_name [as ident] (, ...)* */
    TN_NK_IMPORT_FROM,      /* from dotted_name import names */
    TN_NK_FUNCDEF,          /* def NAME ( params ) [ -> ret ] : suite */
    TN_NK_DECORATOR,        /* @ dotted_name [ ( args ) ] */
    TN_NK_PARAM,            /* IDENT [: anno] [= default] */
    TN_NK_BLOCK,            /* INDENT-delimited statement suite */
    TN_NK_ASSIGN,           /* target = expr */
    TN_NK_AUG_ASSIGN,       /* target op= expr; flags carries op kind */
    TN_NK_EXPR_STMT,        /* expression evaluated for its side effect */
    TN_NK_IF,               /* if test : suite [elif...]* [else] */
    TN_NK_FOR,              /* for target in iter : suite [else] */
    TN_NK_WHILE,            /* while test : suite [else] */
    TN_NK_RETURN,           /* return [expr] */
    TN_NK_PASS,
    TN_NK_BREAK,
    TN_NK_CONTINUE,
    TN_NK_DOTTED_NAME,      /* ident (.ident)* with no further structure */
    TN_NK_EXPR_SPAN,        /* opaque expression: tok_off..tok_off+tok_len */

    /* Expression nodes added in parser sitting two. They live below
     * EXPR_SPAN in the enum so dump tables stay readable in numeric
     * order. The expression parser produces these instead of opaque
     * spans wherever the syntax is one of the forms a sane Triton
     * kernel would actually contain. */
    TN_NK_NAME,             /* identifier reference; tok_off names it */
    TN_NK_LITERAL,          /* int / float / string / None / True / False */
    TN_NK_TUPLE,            /* (a, b, c) or top-level a, b, c */
    TN_NK_BINOP,            /* a OP b; flags carries the op code */
    TN_NK_UNOP,             /* OP a; flags carries the op code */
    TN_NK_BOOLOP,           /* a and b, a or b; flags = AND or OR */
    TN_NK_COMPARE,          /* a < b (sitting two does not chain) */
    TN_NK_CALL,             /* f(args); kids[0] = callee, rest = args */
    TN_NK_KEYWORD,          /* name=value inside a Call's argument list */
    TN_NK_ATTR,             /* a.b; kids[0] = a, name in tok span */
    TN_NK_SUBSCRIPT,        /* a[i]; kids[0] = base, kids[1..] = index */
    TN_NK_SLICE,            /* i:j:k inside subscripts */
    TN_NK_IFEXPR,           /* x if cond else y */
    TN_NK_LIST,             /* [a, b, c] */

    TN_NK_COUNT
} tn_nk_t;

/* Operator subcodes for BINOP. Packed into flags. */

enum {
    TN_BOP_ADD = 0, TN_BOP_SUB, TN_BOP_MUL, TN_BOP_DIV, TN_BOP_FDIV,
    TN_BOP_MOD, TN_BOP_POW, TN_BOP_MATMUL,
    TN_BOP_AND, TN_BOP_OR, TN_BOP_XOR,
    TN_BOP_SHL, TN_BOP_SHR,
    TN_BOP_COUNT
};

/* Unary operator subcodes for UNOP. */

enum {
    TN_UOP_POS = 0, TN_UOP_NEG, TN_UOP_INV, TN_UOP_NOT,
    TN_UOP_COUNT
};

/* Boolean operator subcodes for BOOLOP. */

enum {
    TN_LOP_AND = 0, TN_LOP_OR,
    TN_LOP_COUNT
};

/* Comparison operator subcodes for COMPARE. */

enum {
    TN_CMP_LT = 0, TN_CMP_LE, TN_CMP_GT, TN_CMP_GE,
    TN_CMP_EQ, TN_CMP_NE,
    TN_CMP_IS, TN_CMP_ISNOT,
    TN_CMP_IN, TN_CMP_NOTIN,
    TN_CMP_COUNT
};

/* Literal kind subcodes for LITERAL. */

enum {
    TN_LIT_INT = 0, TN_LIT_FLOAT, TN_LIT_STRING,
    TN_LIT_NONE, TN_LIT_TRUE, TN_LIT_FALSE,
    TN_LIT_COUNT
};

/* Augmented assignment subcodes, packed into tn_node_t::flags. */

enum {
    TN_AUG_ADD = 0, TN_AUG_SUB, TN_AUG_MUL, TN_AUG_DIV,
    TN_AUG_FDIV,    TN_AUG_MOD, TN_AUG_POW,
    TN_AUG_AND,     TN_AUG_OR,  TN_AUG_XOR,
    TN_AUG_SHL,     TN_AUG_SHR, TN_AUG_MATMUL,
    TN_AUG_COUNT
};

/* ---- AST Node ----
 * Fixed 32 bytes, six inline child indices, overflow flag for the
 * rare case (function bodies with more than six statements, mainly)
 * which redirects children into an external pool the same way BIR
 * handles instructions with more than six operands. The pattern is
 * deliberate: it keeps the common-case node small and cache-friendly
 * while still letting the unusual node spend extra space when it
 * really has to. */

#define TN_NODE_INLINE_KIDS    6
#define TN_NODE_KIDS_OVERFLOW  0xFF

typedef struct {
    uint8_t  kind;                              /* tn_nk_t */
    uint8_t  num_kids;                          /* or TN_NODE_KIDS_OVERFLOW */
    uint16_t flags;                             /* per-kind subcode */
    uint32_t tok_off;                           /* first token in this node */
    uint32_t tok_len;                           /* token count in this node */
    uint32_t kids[TN_NODE_INLINE_KIDS];         /* inline child indices */
} tn_node_t;                                    /* 32 bytes */

#define TN_MAX_EXTRA_KIDS  (1 << 18)
#define TN_MAX_KID_SCRATCH (1 << 16)

/* ---- Parser State ----
 * Heap-allocated so the embedded arrays do not blow main's stack
 * frame. The shape follows the same convention as sema_ctx_t in the
 * C99 frontend: one big context struct, freed at the end of the
 * pass, no internal allocations during parsing.
 *
 * kid_scratch is a stack of child indices used during parsing. Each
 * grammar function records kid_scratch_top on entry, pushes its
 * children onto the scratch as it parses them, then commits the
 * whole contiguous range into either the node's inline kids slots
 * or into the extra_kids overflow pool at the end. The pattern
 * stops nested nodes from interleaving their overflow writes with
 * one another, which is the bug that the simpler incremental
 * add_kid had: a child node's overflow could happily scribble
 * inside its parent's reserved overflow range. */

typedef struct {
    const tn_lex_t *lex;
    uint32_t        cur;                        /* current token index */

    tn_node_t       nodes[TN_MAX_NODES];
    uint32_t        num_nodes;

    uint32_t        extra_kids[TN_MAX_EXTRA_KIDS];
    uint32_t        num_extra;

    uint32_t        kid_scratch[TN_MAX_KID_SCRATCH];
    uint32_t        kid_scratch_top;

    bc_error_t      errors[BC_MAX_ERRORS];
    int             num_errors;

    uint32_t        root;                       /* module node index */
} tn_parse_t;

/* ---- Public API: Parser ---- */

void        tn_parse_init(tn_parse_t *P, const tn_lex_t *L);
int         tn_parse(tn_parse_t *P);
const char *tn_nk_name(int kind);
void        tn_ast_dump(const tn_parse_t *P, FILE *out);

/* ---- Sema: Symbol Kinds ----
 * The categories of thing a Name node can resolve to. UNBOUND is
 * the sentinel for names that did not bind, which is reported as a
 * diagnostic but does not stop the pass (we keep going so later
 * names can still be checked). */

typedef enum {
    TN_SYM_UNBOUND = 0,
    TN_SYM_PARAM,           /* kernel parameter; aux = param index */
    TN_SYM_LOCAL,           /* assigned local; aux = declaring node */
    TN_SYM_LOOPVAR,         /* for-loop target; aux = declaring node */
    TN_SYM_MODULE,          /* import alias; aux = builtin module id */
    TN_SYM_INTRINSIC,       /* tl.* function; aux = tn_intrinsic_t */
    TN_SYM_TYPE,            /* tl.float32 etc.; aux = tn_intrinsic_t */
    TN_SYM_KIND_COUNT
} tn_sym_kind_t;

/* ---- Sema: Builtin Module Ids ----
 * The handful of modules a Triton kernel imports. We do not run the
 * import statements; we just recognise the canonical aliases. */

enum {
    TN_MOD_NONE = 0,
    TN_MOD_TRITON,          /* `import triton` */
    TN_MOD_TL,              /* `import triton.language as tl` */
    TN_MOD_MATH,            /* `import math` (rare in kernels) */
    TN_MOD_COUNT
};

/* ---- Sema: tl.* Intrinsic Ids ----
 * Hardcoded enumeration of the Triton language functions and types
 * the sema knows about. The list covers the intrinsics that appear
 * in real Triton kernels in the wild and the canonical numeric
 * dtypes; adding more is a one-line table entry away. */

typedef enum {
    TN_TLI_NONE = 0,

    /* Thread model */
    TN_TLI_PROGRAM_ID,
    TN_TLI_NUM_PROGRAMS,

    /* Memory */
    TN_TLI_LOAD,
    TN_TLI_STORE,
    TN_TLI_MAKE_BLOCK_PTR,
    TN_TLI_ADVANCE,

    /* Tile construction */
    TN_TLI_ARANGE,
    TN_TLI_ZEROS,
    TN_TLI_ZEROS_LIKE,
    TN_TLI_FULL,
    TN_TLI_BROADCAST_TO,
    TN_TLI_RESHAPE,
    TN_TLI_TRANS,
    TN_TLI_WHERE,

    /* Reductions */
    TN_TLI_SUM,
    TN_TLI_MAX,
    TN_TLI_MIN,
    TN_TLI_ARGMAX,
    TN_TLI_ARGMIN,

    /* Matrix */
    TN_TLI_DOT,

    /* Math (single-arg unless noted) */
    TN_TLI_EXP, TN_TLI_EXP2, TN_TLI_LOG, TN_TLI_LOG2,
    TN_TLI_SIN, TN_TLI_COS, TN_TLI_TAN, TN_TLI_TANH,
    TN_TLI_SQRT, TN_TLI_RSQRT, TN_TLI_ABS,
    TN_TLI_FLOOR, TN_TLI_CEIL, TN_TLI_ERF,
    TN_TLI_MAXIMUM, TN_TLI_MINIMUM,
    TN_TLI_FDIV, TN_TLI_CDIV,

    /* Compile-time helpers */
    TN_TLI_STATIC_ASSERT,
    TN_TLI_STATIC_PRINT,
    TN_TLI_DEVICE_ASSERT,
    TN_TLI_DEVICE_PRINT,
    TN_TLI_CONSTEXPR,       /* type marker, not a function */

    /* Numeric types */
    TN_TLI_FLOAT16, TN_TLI_FLOAT32, TN_TLI_FLOAT64, TN_TLI_BFLOAT16,
    TN_TLI_INT1, TN_TLI_INT8, TN_TLI_INT16, TN_TLI_INT32, TN_TLI_INT64,
    TN_TLI_UINT8, TN_TLI_UINT16, TN_TLI_UINT32, TN_TLI_UINT64,

    TN_TLI_COUNT
} tn_intrinsic_t;

/* ---- Sema: Symbol Table Entry ---- */

typedef struct {
    uint32_t name_off;      /* source offset of the name token */
    uint16_t name_len;
    uint8_t  kind;          /* tn_sym_kind_t */
    uint8_t  pad;
    uint32_t aux;           /* param idx, intrinsic id, module id, etc. */
    uint32_t decl_node;     /* AST node that introduced the binding */
} tn_sym_t;

typedef struct {
    uint32_t start;         /* first symbol in this scope (index into syms) */
    uint32_t end;           /* one past last */
} tn_scope_t;

#define TN_MAX_SYMBOLS  8192
#define TN_MAX_SCOPES   128

/* ---- Sema: Tile Shape ----
 * rank 0 = scalar, rank 1 = vec[dims[0]], rank 2 = mat[dims[0],
 * dims[1]]. dim = -1 means dynamic. dtype is a tn_intrinsic_t code
 * (tl.float32 etc.), or 0 if not yet inferred. */

typedef struct {
    uint8_t  rank;
    uint8_t  dtype;
    uint8_t  _pad[2];
    int32_t  dims[2];
} tn_shape_t;                                       /* 12 bytes */

/* ---- Sema: Per-Node Annotation ----
 * Parallel arrays over the parser node pool. node_sym_kind +
 * node_sym_aux carry the resolved symbol for Name / Attr. node_shape
 * carries the inferred tile shape; non-expression nodes leave it
 * zero. */

typedef struct {
    const tn_parse_t *parser;

    tn_sym_t        syms[TN_MAX_SYMBOLS];
    uint32_t        num_syms;

    tn_scope_t      scopes[TN_MAX_SCOPES];
    int             num_scopes;

    uint8_t         node_sym_kind[TN_MAX_NODES];
    uint32_t        node_sym_aux[TN_MAX_NODES];
    tn_shape_t      node_shape[TN_MAX_NODES];
    int32_t         node_const_val[TN_MAX_NODES];   /* >=0 = known; -1 = dynamic */

    bc_error_t      errors[BC_MAX_ERRORS];
    int             num_errors;
} tn_sema_t;

/* ---- Public API: Sema ---- */

void        tn_sema_init(tn_sema_t *S, const tn_parse_t *P);
int         tn_sema(tn_sema_t *S);
void        tn_sema_dump(const tn_sema_t *S, FILE *out);
const char *tn_sym_kind_name(int kind);
const char *tn_intrinsic_name(int id);

/* Format a shape as "scalar", "vec[N]" or "mat[M, N]" with an
 * optional :dtype suffix. ? appears where a dim is dynamic. */
int         tn_shape_format(tn_shape_t sh, char *buf, int bufsize);

/* ---- Lowering Context ----
 * Walks the sema-annotated AST and produces BIR. Heap-allocated to
 * keep the embedded node-to-value map off main's stack frame. The
 * per-node map records, for any AST node whose lowering yielded a
 * BIR value, the value reference (with the BIR_VAL_CONST_BIT flag
 * for constants). Subsequent references through that node can read
 * the value back in one lookup. */

/* A materialized tile: per-element BIR value refs, row-major. vec[N]
 * is stored as rank 1, d0=N, d1=1. Capped at 32x32 for the first cut. */
#define TN_TILE_MAX  1024
#define TN_TILE_POOL 128

typedef struct {
    int      rank;
    int      d0, d1;
    uint32_t elem[TN_TILE_MAX];
} tn_tile_t;

typedef struct {
    const tn_parse_t *parser;
    const tn_sema_t  *sema;
    bir_module_t     *bir;

    uint32_t        cur_func;       /* BIR function index in flight */
    uint32_t        cur_block;      /* BIR block index in flight */
    uint32_t        cur_param_base; /* index of first BIR_PARAM in current func */

    /* Map from AST node index to BIR value reference, BIR_VAL_NONE
     * if not yet produced. Used to resolve Name references back to
     * the value their declaring node generated. */
    uint32_t        node_val[TN_MAX_NODES];

    /* Rank-2 (and rank-1) tiles in a kernel that uses tl.dot are
     * materialized and fully unrolled: each tile is an array of
     * per-element BIR scalar values. tile_mode turns the whole kernel
     * onto this path (arange becomes constants, not thread_id, and the
     * launch runs one thread). node_tile maps a binding node (Assign)
     * to its pool slot. */
    int             tile_mode;
    int             n_tiles;
    tn_tile_t       tile_pool[TN_TILE_POOL];
    int32_t         node_tile[TN_MAX_NODES];

    /* Common types, looked up once per module so we are not building
     * the same type entry over and over. */
    uint32_t        t_i32;
    uint32_t        t_f32;
    uint32_t        t_ptr_i32;
    uint32_t        t_ptr_f32;
    uint32_t        t_void;

    bc_error_t      errors[BC_MAX_ERRORS];
    int             num_errors;
} tn_lower_t;

/* ---- Public API: Lowering ---- */

void tn_lower_init(tn_lower_t *L, const tn_parse_t *P,
                   const tn_sema_t *S, bir_module_t *M);
int  tn_lower(tn_lower_t *L);

#endif /* BARRACUDA_TRITON_H */
