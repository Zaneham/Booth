#ifndef BARRACUDA_TOKEN_H
#define BARRACUDA_TOKEN_H

#include "barracuda.h"

typedef enum {
    TOK_INT_LIT,
    TOK_FLOAT_LIT,
    TOK_STRING_LIT,
    TOK_CHAR_LIT,
    TOK_IDENT,

    TOK_LPAREN,             /* ( */
    TOK_RPAREN,             /* ) */
    TOK_LBRACKET,           /* [ */
    TOK_RBRACKET,           /* ] */
    TOK_LBRACE,             /* { */
    TOK_RBRACE,             /* } */
    TOK_SEMI,               /* ; */
    TOK_COMMA,              /* , */
    TOK_DOT,                /* . */
    TOK_ARROW,              /* -> */
    TOK_COLON,              /* : */
    TOK_DCOLON,             /* :: */
    TOK_QUESTION,           /* ? */
    TOK_ELLIPSIS,           /* ... */

    TOK_PLUS,               /* + */
    TOK_MINUS,              /* - */
    TOK_STAR,               /* * */
    TOK_SLASH,              /* / */
    TOK_PERCENT,            /* % */
    TOK_INC,                /* ++ */
    TOK_DEC,                /* -- */

    TOK_AMP,                /* & */
    TOK_PIPE,               /* | */
    TOK_CARET,              /* ^ */
    TOK_TILDE,              /* ~ */
    TOK_SHL,                /* << */
    TOK_SHR,                /* >> */

    TOK_LT,                 /* < */
    TOK_GT,                 /* > */
    TOK_LE,                 /* <= */
    TOK_GE,                 /* >= */
    TOK_EQ,                 /* == */
    TOK_NE,                 /* != */

    TOK_LAND,               /* && */
    TOK_LOR,                /* || */
    TOK_BANG,               /* ! */

    TOK_ASSIGN,             /* = */
    TOK_PLUS_EQ,            /* += */
    TOK_MINUS_EQ,           /* -= */
    TOK_STAR_EQ,            /* *= */
    TOK_SLASH_EQ,           /* /= */
    TOK_PERCENT_EQ,         /* %= */
    TOK_AMP_EQ,             /* &= */
    TOK_PIPE_EQ,            /* |= */
    TOK_CARET_EQ,           /* ^= */
    TOK_SHL_EQ,             /* <<= */
    TOK_SHR_EQ,             /* >>= */

    TOK_LAUNCH_OPEN,        /* <<< */
    TOK_LAUNCH_CLOSE,       /* >>> */

    TOK_HASH,               /* # */
    TOK_DHASH,              /* ## */
    TOK_PP_LINE,

    TOK_AUTO,
    TOK_BREAK,
    TOK_CASE,
    TOK_CHAR,
    TOK_CONST,
    TOK_CONTINUE,
    TOK_DEFAULT,
    TOK_DO,
    TOK_DOUBLE,
    TOK_ELSE,
    TOK_ENUM,
    TOK_EXTERN,
    TOK_FLOAT,
    TOK_FOR,
    TOK_GOTO,
    TOK_IF,
    TOK_INLINE,
    TOK_INT,
    TOK_LONG,
    TOK_REGISTER,
    TOK_RESTRICT,
    TOK_RETURN,
    TOK_SHORT,
    TOK_SIGNED,
    TOK_SIZEOF,
    TOK_STATIC,
    TOK_STRUCT,
    TOK_SWITCH,
    TOK_TYPEDEF,
    TOK_UNION,
    TOK_UNSIGNED,
    TOK_VOID,
    TOK_VOLATILE,
    TOK_WHILE,

    TOK_ALIGNAS,
    TOK_ALIGNOF,
    TOK_BOOL,
    TOK_CATCH,
    TOK_CLASS,
    TOK_CONST_CAST,
    TOK_CONSTEXPR,
    TOK_DECLTYPE,
    TOK_DELETE,
    TOK_DYNAMIC_CAST,
    TOK_EXPLICIT,
    TOK_FALSE,
    TOK_FINAL,
    TOK_FRIEND,
    TOK_MUTABLE,
    TOK_NAMESPACE,
    TOK_NEW,
    TOK_NOEXCEPT,
    TOK_NULLPTR,
    TOK_OPERATOR,
    TOK_OVERRIDE,
    TOK_PRIVATE,
    TOK_PROTECTED,
    TOK_PUBLIC,
    TOK_REINTERPRET_CAST,
    TOK_STATIC_ASSERT,
    TOK_STATIC_CAST,
    TOK_TEMPLATE,
    TOK_THIS,
    TOK_THROW,
    TOK_TRUE,
    TOK_TRY,
    TOK_TYPENAME,
    TOK_USING,
    TOK_VIRTUAL,

    TOK_CU_GLOBAL,          /* __global__ */
    TOK_CU_DEVICE,          /* __device__ */
    TOK_CU_HOST,            /* __host__ */
    TOK_CU_SHARED,          /* __shared__ */
    TOK_CU_CONSTANT,        /* __constant__ */
    TOK_CU_MANAGED,         /* __managed__ */
    TOK_CU_GRID_CONSTANT,   /* __grid_constant__ */
    TOK_CU_LAUNCH_BOUNDS,   /* __launch_bounds__ */
    TOK_CU_RESTRICT,        /* __restrict__ */
    TOK_CU_FORCEINLINE,     /* __forceinline__ */
    TOK_CU_NOINLINE,        /* __noinline__ */
    TOK_CU_ALIGN,           /* __align__ */
    TOK_ASM,                /* asm, __asm, __asm__ */

    TOK_NORETURN,           /* _Noreturn */

    TOK_EOF,
    TOK_ERROR,
    TOK_COUNT
} token_type_t;

typedef struct {
    int         type;
    uint32_t    offset;
    uint32_t    len;
    uint32_t    line;
    uint16_t    col;
} token_t;

const char *token_type_name(int type);

#endif /* BARRACUDA_TOKEN_H */
