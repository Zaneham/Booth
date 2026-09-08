#ifndef BARRACUDA_AST_H
#define BARRACUDA_AST_H

#include "barracuda.h"

#define BC_MAX_NODES    (1 << 18)
#define BC_MAX_ASMOP    24

typedef enum {
    AST_NONE = 0,

    AST_INT_LIT,
    AST_FLOAT_LIT,
    AST_STRING_LIT,
    AST_CHAR_LIT,
    AST_BOOL_LIT,
    AST_NULL_LIT,
    AST_IDENT,
    AST_BINARY,
    AST_UNARY_PREFIX,
    AST_UNARY_POSTFIX,
    AST_TERNARY,
    AST_CALL,
    AST_LAUNCH,
    AST_MEMBER,
    AST_SUBSCRIPT,
    AST_CAST,
    AST_SIZEOF,
    AST_PAREN,
    AST_INIT_LIST,
    AST_SCOPE_RES,
    AST_TEMPLATE_ARGS,
    AST_PACK_EXP,
    AST_PACK_SIZE,
    AST_FOLD,
    AST_LAMBDA,
    AST_NEW,
    AST_DELETE,

    AST_EXPR_STMT,
    AST_BLOCK,
    AST_IF,
    AST_FOR,
    AST_RANGE_FOR,
    AST_WHILE,
    AST_DO_WHILE,
    AST_SWITCH,
    AST_CASE,
    AST_DEFAULT,
    AST_RETURN,
    AST_BREAK,
    AST_CONTINUE,
    AST_GOTO,
    AST_LABEL,
    AST_ASM,
    AST_ASM_OP,

    AST_TYPE_SPEC,
    AST_DECLARATOR,
    AST_PARAM,
    AST_VAR_DECL,
    AST_FUNC_DEF,
    AST_FUNC_DECL,
    AST_STRUCT_DEF,
    AST_ENUM_DEF,
    AST_ENUMERATOR,
    AST_TYPEDEF,
    AST_USING,
    AST_NAMESPACE,
    AST_TEMPLATE_DECL,
    AST_TEMPLATE_PARAM,

    AST_BASE,

    AST_PP_DIRECTIVE,
    AST_TRANSLATION_UNIT,

    AST_TYPE_COUNT
} ast_type_t;

#define TP_NTYP         0x01
#define TP_PACK         0x02

#define PRM_PACK        0x01
#define PRM_VARG        0x02

#define IF_CEXP         0x01

#define LAM_BADC        0x01
#define LAM_RREF        0x02
#define LAM_PDSH        4

#define FLD_UL          0x01
#define FLD_UR          0x02
#define FLD_BL          0x03
#define FLD_BR          0x04

#define QUAL_CONST      0x01
#define QUAL_VOLATILE   0x02
#define QUAL_STATIC     0x04
#define QUAL_EXTERN     0x08
#define QUAL_INLINE     0x10
#define QUAL_REGISTER   0x20
#define QUAL_CONSTEXPR  0x40
#define QUAL_TYPEDEF    0x80
#define QUAL_SCOPED     0x100
#define QUAL_PINIT      0x200
#define QUAL_REF        0x400
#define QUAL_RREF       0x800
#define QUAL_DECOR      0x1000
#define QUAL_PDEF       0x2000
#define QUAL_PTR1       0x4000
#define QUAL_PTR2       0x8000

typedef enum {
    TYPE_VOID, TYPE_BOOL, TYPE_CHAR, TYPE_SHORT, TYPE_INT, TYPE_LONG,
    TYPE_LLONG, TYPE_FLOAT, TYPE_DOUBLE, TYPE_LDOUBLE,
    TYPE_UNSIGNED, TYPE_SIGNED,
    TYPE_STRUCT, TYPE_ENUM, TYPE_UNION, TYPE_CLASS,
    TYPE_AUTO,
    TYPE_NAME,
    TYPE_DECLTYPE,
} basic_type_t;

/* Left-child / right-sibling. 0 = no node. Simple as a tree should be. */
typedef struct {
    uint16_t    type;
    uint16_t    cuda_flags;
    uint32_t    line;
    uint16_t    col;
    uint16_t    qualifiers;
    uint32_t    first_child;
    uint32_t    next_sibling;
    uint32_t    launch_bounds_max;  /* 0 = not set. Outside union to avoid the oper.flags conflict. */
    uint32_t    launch_bounds_min;  /* 0 = not set */
    uint32_t    algn;
    union {
        int64_t     ival;
        double      fval;
        struct { uint32_t offset; uint32_t len; } text;
        struct { int op; int flags; } oper;
        struct { int kind; int is_unsigned; } btype;
        struct { int is_arrow; } member;
        struct { uint32_t lb_max; uint32_t lb_min; } launch_bounds;
    } d;
} ast_node_t;

const char *ast_type_name(int type);

#endif /* BARRACUDA_AST_H */
