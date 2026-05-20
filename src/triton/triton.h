#ifndef BARRACUDA_TRITON_H
#define BARRACUDA_TRITON_H

#include "barracuda.h"
#include "bir.h"

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

/* ---- Public API ---- */

void        tn_lex_init(tn_lex_t *L, const char *src, uint32_t len,
                        tn_tok_t *tokens, uint32_t max_tokens);
int         tn_tokenize(tn_lex_t *L);
const char *tn_tok_name(int kind);
int         tn_tok_text(const tn_lex_t *L, const tn_tok_t *tok,
                        char *buf, int bufsize);

/* Parser, sema, and lowering live in parallel files and arrive in
 * subsequent sittings. For now they exist as stubs so the build can
 * compile against the public surface and main.c can route through. */

int  tn_parse(const tn_lex_t *L);
int  tn_sema(void);
int  tn_lower(bir_module_t *out);

#endif /* BARRACUDA_TRITON_H */
