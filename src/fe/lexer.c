#include "lexer.h"
#include <ctype.h>
#include <stdarg.h>

/* 80 keywords walk into a sorted bar */

typedef struct {
    const char *name;
    int         type;
} kw_entry_t;

static const kw_entry_t keywords[] = {
    {"_Noreturn",          TOK_NORETURN},
    {"__align__",          TOK_CU_ALIGN},
    {"__asm",              TOK_ASM},
    {"__asm__",            TOK_ASM},
    {"__constant__",       TOK_CU_CONSTANT},
    {"__device__",         TOK_CU_DEVICE},
    {"__forceinline__",    TOK_CU_FORCEINLINE},
    {"__global__",         TOK_CU_GLOBAL},
    {"__grid_constant__",  TOK_CU_GRID_CONSTANT},
    {"__host__",           TOK_CU_HOST},
    {"__inline",           TOK_INLINE},
    {"__inline__",         TOK_INLINE},
    {"__launch_bounds__",  TOK_CU_LAUNCH_BOUNDS},
    {"__managed__",        TOK_CU_MANAGED},
    {"__noinline__",       TOK_CU_NOINLINE},
    {"__restrict__",       TOK_CU_RESTRICT},
    {"__shared__",         TOK_CU_SHARED},
    {"alignas",            TOK_ALIGNAS},
    {"alignof",            TOK_ALIGNOF},
    {"asm",                TOK_ASM},
    {"auto",               TOK_AUTO},
    {"bool",               TOK_BOOL},
    {"break",              TOK_BREAK},
    {"case",               TOK_CASE},
    {"catch",              TOK_CATCH},
    {"char",               TOK_CHAR},
    {"class",              TOK_CLASS},
    {"const",              TOK_CONST},
    {"const_cast",         TOK_CONST_CAST},
    {"constexpr",          TOK_CONSTEXPR},
    {"continue",           TOK_CONTINUE},
    {"decltype",           TOK_DECLTYPE},
    {"default",            TOK_DEFAULT},
    {"delete",             TOK_DELETE},
    {"do",                 TOK_DO},
    {"double",             TOK_DOUBLE},
    {"dynamic_cast",       TOK_DYNAMIC_CAST},
    {"else",               TOK_ELSE},
    {"enum",               TOK_ENUM},
    {"explicit",           TOK_EXPLICIT},
    {"extern",             TOK_EXTERN},
    {"false",              TOK_FALSE},
    {"final",              TOK_FINAL},
    {"float",              TOK_FLOAT},
    {"for",                TOK_FOR},
    {"friend",             TOK_FRIEND},
    {"goto",               TOK_GOTO},
    {"if",                 TOK_IF},
    {"inline",             TOK_INLINE},
    {"int",                TOK_INT},
    {"long",               TOK_LONG},
    {"mutable",            TOK_MUTABLE},
    {"namespace",          TOK_NAMESPACE},
    {"new",                TOK_NEW},
    {"noexcept",           TOK_NOEXCEPT},
    {"nullptr",            TOK_NULLPTR},
    {"operator",           TOK_OPERATOR},
    {"override",           TOK_OVERRIDE},
    {"private",            TOK_PRIVATE},
    {"protected",          TOK_PROTECTED},
    {"public",             TOK_PUBLIC},
    {"register",           TOK_REGISTER},
    {"reinterpret_cast",   TOK_REINTERPRET_CAST},
    {"restrict",           TOK_RESTRICT},
    {"return",             TOK_RETURN},
    {"short",              TOK_SHORT},
    {"signed",             TOK_SIGNED},
    {"sizeof",             TOK_SIZEOF},
    {"static",             TOK_STATIC},
    {"static_assert",      TOK_STATIC_ASSERT},
    {"static_cast",        TOK_STATIC_CAST},
    {"struct",             TOK_STRUCT},
    {"switch",             TOK_SWITCH},
    {"template",           TOK_TEMPLATE},
    {"this",               TOK_THIS},
    {"throw",              TOK_THROW},
    {"true",               TOK_TRUE},
    {"try",                TOK_TRY},
    {"typedef",            TOK_TYPEDEF},
    {"typename",           TOK_TYPENAME},
    {"union",              TOK_UNION},
    {"unsigned",           TOK_UNSIGNED},
    {"using",              TOK_USING},
    {"virtual",            TOK_VIRTUAL},
    {"void",               TOK_VOID},
    {"volatile",           TOK_VOLATILE},
    {"while",              TOK_WHILE},
};

#define NUM_KEYWORDS ((int)(sizeof(keywords) / sizeof(keywords[0])))

static int lookup_keyword(const char *src, uint32_t len)
{
    int lo = 0, hi = NUM_KEYWORDS - 1;
    while (lo <= hi) {
        int mid = (lo + hi) / 2;
        int cmp = strncmp(src, keywords[mid].name, len);
        if (cmp == 0) {
            /* Check exact length match */
            if (keywords[mid].name[len] == '\0')
                return keywords[mid].type;
            cmp = -1;

        }
        if (cmp < 0)
            hi = mid - 1;
        else
            lo = mid + 1;
    }
    return TOK_IDENT;
}

/* The table stays static; the suite only needs to read the order. */
int lexer_kw_count(void) { return NUM_KEYWORDS; }

const char *lexer_kw_at(int i)
{
    if (i < 0 || i >= NUM_KEYWORDS) return NULL;
    return keywords[i].name;
}

static const char *tok_names[] = {
    [TOK_INT_LIT]       = "INT_LIT",
    [TOK_FLOAT_LIT]     = "FLOAT_LIT",
    [TOK_STRING_LIT]    = "STRING_LIT",
    [TOK_CHAR_LIT]      = "CHAR_LIT",
    [TOK_IDENT]         = "IDENT",
    [TOK_LPAREN]        = "(",
    [TOK_RPAREN]        = ")",
    [TOK_LBRACKET]      = "[",
    [TOK_RBRACKET]      = "]",
    [TOK_LBRACE]        = "{",
    [TOK_RBRACE]        = "}",
    [TOK_SEMI]          = ";",
    [TOK_COMMA]         = ",",
    [TOK_DOT]           = ".",
    [TOK_ARROW]         = "->",
    [TOK_COLON]         = ":",
    [TOK_DCOLON]        = "::",
    [TOK_QUESTION]      = "?",
    [TOK_ELLIPSIS]      = "...",
    [TOK_PLUS]          = "+",
    [TOK_MINUS]         = "-",
    [TOK_STAR]          = "*",
    [TOK_SLASH]         = "/",
    [TOK_PERCENT]       = "%",
    [TOK_INC]           = "++",
    [TOK_DEC]           = "--",
    [TOK_AMP]           = "&",
    [TOK_PIPE]          = "|",
    [TOK_CARET]         = "^",
    [TOK_TILDE]         = "~",
    [TOK_SHL]           = "<<",
    [TOK_SHR]           = ">>",
    [TOK_LT]            = "<",
    [TOK_GT]            = ">",
    [TOK_LE]            = "<=",
    [TOK_GE]            = ">=",
    [TOK_EQ]            = "==",
    [TOK_NE]            = "!=",
    [TOK_LAND]          = "&&",
    [TOK_LOR]           = "||",
    [TOK_BANG]          = "!",
    [TOK_ASSIGN]        = "=",
    [TOK_PLUS_EQ]       = "+=",
    [TOK_MINUS_EQ]      = "-=",
    [TOK_STAR_EQ]       = "*=",
    [TOK_SLASH_EQ]      = "/=",
    [TOK_PERCENT_EQ]    = "%=",
    [TOK_AMP_EQ]        = "&=",
    [TOK_PIPE_EQ]       = "|=",
    [TOK_CARET_EQ]      = "^=",
    [TOK_SHL_EQ]        = "<<=",
    [TOK_SHR_EQ]        = ">>=",
    [TOK_LAUNCH_OPEN]   = "<<<",
    [TOK_LAUNCH_CLOSE]  = ">>>",
    [TOK_HASH]          = "#",
    [TOK_DHASH]         = "##",
    [TOK_PP_LINE]       = "PP_LINE",
    [TOK_AUTO]          = "auto",
    [TOK_BREAK]         = "break",
    [TOK_CASE]          = "case",
    [TOK_CHAR]          = "char",
    [TOK_CONST]         = "const",
    [TOK_CONTINUE]      = "continue",
    [TOK_DEFAULT]       = "default",
    [TOK_DO]            = "do",
    [TOK_DOUBLE]        = "double",
    [TOK_ELSE]          = "else",
    [TOK_ENUM]          = "enum",
    [TOK_EXTERN]        = "extern",
    [TOK_FLOAT]         = "float",
    [TOK_FOR]           = "for",
    [TOK_GOTO]          = "goto",
    [TOK_IF]            = "if",
    [TOK_INLINE]        = "inline",
    [TOK_INT]           = "int",
    [TOK_LONG]          = "long",
    [TOK_REGISTER]      = "register",
    [TOK_RESTRICT]      = "restrict",
    [TOK_RETURN]        = "return",
    [TOK_SHORT]         = "short",
    [TOK_SIGNED]        = "signed",
    [TOK_SIZEOF]        = "sizeof",
    [TOK_STATIC]        = "static",
    [TOK_STRUCT]        = "struct",
    [TOK_SWITCH]        = "switch",
    [TOK_TYPEDEF]       = "typedef",
    [TOK_UNION]         = "union",
    [TOK_UNSIGNED]      = "unsigned",
    [TOK_VOID]          = "void",
    [TOK_VOLATILE]      = "volatile",
    [TOK_WHILE]         = "while",
    [TOK_ALIGNAS]       = "alignas",
    [TOK_ALIGNOF]       = "alignof",
    [TOK_BOOL]          = "bool",
    [TOK_CATCH]         = "catch",
    [TOK_CLASS]         = "class",
    [TOK_CONST_CAST]    = "const_cast",
    [TOK_CONSTEXPR]     = "constexpr",
    [TOK_DECLTYPE]      = "decltype",
    [TOK_DELETE]        = "delete",
    [TOK_DYNAMIC_CAST]  = "dynamic_cast",
    [TOK_EXPLICIT]      = "explicit",
    [TOK_FALSE]         = "false",
    [TOK_FINAL]         = "final",
    [TOK_FRIEND]        = "friend",
    [TOK_MUTABLE]       = "mutable",
    [TOK_NAMESPACE]     = "namespace",
    [TOK_NEW]           = "new",
    [TOK_NOEXCEPT]      = "noexcept",
    [TOK_NULLPTR]       = "nullptr",
    [TOK_OPERATOR]      = "operator",
    [TOK_OVERRIDE]      = "override",
    [TOK_PRIVATE]       = "private",
    [TOK_PROTECTED]     = "protected",
    [TOK_PUBLIC]        = "public",
    [TOK_REINTERPRET_CAST] = "reinterpret_cast",
    [TOK_STATIC_ASSERT] = "static_assert",
    [TOK_STATIC_CAST]   = "static_cast",
    [TOK_TEMPLATE]      = "template",
    [TOK_THIS]          = "this",
    [TOK_THROW]         = "throw",
    [TOK_TRUE]          = "true",
    [TOK_TRY]           = "try",
    [TOK_TYPENAME]      = "typename",
    [TOK_USING]         = "using",
    [TOK_VIRTUAL]       = "virtual",
    [TOK_CU_GLOBAL]     = "__global__",
    [TOK_CU_DEVICE]     = "__device__",
    [TOK_CU_HOST]       = "__host__",
    [TOK_CU_SHARED]     = "__shared__",
    [TOK_CU_CONSTANT]   = "__constant__",
    [TOK_CU_MANAGED]    = "__managed__",
    [TOK_CU_GRID_CONSTANT] = "__grid_constant__",
    [TOK_CU_LAUNCH_BOUNDS] = "__launch_bounds__",
    [TOK_CU_RESTRICT]   = "__restrict__",
    [TOK_CU_FORCEINLINE] = "__forceinline__",
    [TOK_CU_NOINLINE]   = "__noinline__",
    [TOK_CU_ALIGN]      = "__align__",
    [TOK_ASM]           = "asm",
    [TOK_NORETURN]      = "_Noreturn",
    [TOK_EOF]           = "EOF",
    [TOK_ERROR]         = "ERROR",
};

const char *token_type_name(int type)
{
    if (type >= 0 && type < TOK_COUNT)
        return tok_names[type] ? tok_names[type] : "???";
    return "???";
}

void lexer_init(lexer_t *L, const char *src, uint32_t len,
                token_t *tokens, uint32_t max_tokens)
{
    memset(L, 0, sizeof(*L));
    L->src = src;
    L->src_len = len;
    L->pos = 0;
    L->line = 1;
    L->line_start = 0;
    L->tokens = tokens;
    L->num_tokens = 0;
    L->max_tokens = max_tokens;
    L->num_errors = 0;
}

static inline int at_end(const lexer_t *L)
{
    return L->pos >= L->src_len;
}

static inline char cur(const lexer_t *L)
{
    return at_end(L) ? '\0' : L->src[L->pos];
}

static inline char peek(const lexer_t *L, int ahead)
{
    uint32_t p = L->pos + (uint32_t)ahead;
    return (p < L->src_len) ? L->src[p] : '\0';
}

static inline void advance(lexer_t *L)
{
    if (L->pos < L->src_len) {
        if (L->src[L->pos] == '\n') {
            L->line++;
            L->line_start = L->pos + 1;
        }
        L->pos++;
    }
}

static void emit(lexer_t *L, int type, uint32_t start, uint32_t len,
                 uint32_t line, uint16_t col)
{
    if (L->num_tokens >= L->max_tokens) {
        if (L->num_errors < BC_MAX_ERRORS) {
            bc_error_t *e = &L->errors[L->num_errors++];
            e->loc.line = line;
            e->loc.col = col;
            e->loc.offset = start;
            e->code = BC_ERR_OVERFLOW;
            e->eid  = (uint16_t)BC_E001;
            snprintf(e->msg, sizeof(e->msg), "%s", bc_efmt(BC_E001));
        }
        return;
    }
    token_t *t = &L->tokens[L->num_tokens++];
    t->type = type;
    t->offset = start;
    t->len = len;
    t->line = line;
    t->col = col;
}

static void lex_error(lexer_t *L, bc_eid_t eid, ...)
{
    if (L->num_errors < BC_MAX_ERRORS) {
        bc_error_t *e = &L->errors[L->num_errors++];
        e->loc.line = L->line;
        e->loc.col = (uint16_t)(L->pos - L->line_start + 1);
        e->loc.offset = L->pos;
        e->code = BC_ERR_LEX;
        e->eid  = (uint16_t)eid;
        va_list ap;
        va_start(ap, eid);
        vsnprintf(e->msg, sizeof(e->msg), bc_efmt(eid), ap);
        va_end(ap);
    }
}

int lexer_token_text(const lexer_t *L, const token_t *tok,
                     char *buf, int bufsize)
{
    int len = (int)tok->len;
    if (len >= bufsize) len = bufsize - 1;
    memcpy(buf, L->src + tok->offset, (size_t)len);
    buf[len] = '\0';
    return len;
}

static uint32_t lx_splc(const lexer_t *L)
{
    if (cur(L) != '\\') return 0;
    if (peek(L, 1) == '\n') return 2;
    if (peek(L, 1) == '\r' && peek(L, 2) == '\n') return 3;
    return 0;
}

static void skip_whitespace(lexer_t *L)
{
    while (!at_end(L)) {
        char c = cur(L);
        if (c == ' ' || c == '\t' || c == '\r' || c == '\n')
            advance(L);
        else
            break;
    }
}

static void skip_line_comment(lexer_t *L)
{
    while (!at_end(L) && cur(L) != '\n')
        advance(L);
}

static void skip_block_comment(lexer_t *L)
{
    advance(L); /* skip / */
    advance(L); /* skip * */
    while (!at_end(L)) {
        if (cur(L) == '*' && peek(L, 1) == '/') {
            advance(L);
            advance(L);
            return;
        }
        advance(L);
    }
    lex_error(L, BC_E002);
}

static void scan_string(lexer_t *L, char quote)
{
    uint32_t start = L->pos;
    uint32_t start_line = L->line;
    uint16_t start_col = (uint16_t)(L->pos - L->line_start + 1);
    int type = (quote == '"') ? TOK_STRING_LIT : TOK_CHAR_LIT;

    advance(L);
    while (!at_end(L) && cur(L) != quote) {
        if (cur(L) == '\\')
            advance(L);
        if (cur(L) == '\n') {
            lex_error(L, BC_E003);
            break;
        }
        advance(L);
    }
    if (!at_end(L))
        advance(L);
    else
        lex_error(L, BC_E004);

    emit(L, type, start, L->pos - start, start_line, start_col);
}

static void scan_number(lexer_t *L)
{
    uint32_t start = L->pos;
    uint32_t start_line = L->line;
    uint16_t start_col = (uint16_t)(L->pos - L->line_start + 1);
    int is_float = 0;

    if (cur(L) == '0' && (peek(L, 1) == 'x' || peek(L, 1) == 'X')) {
        advance(L); advance(L);
        while (!at_end(L) && (isxdigit((unsigned char)cur(L)) ||
               cur(L) == '\''))
            advance(L);
        if (cur(L) == '.') {
            is_float = 1;
            advance(L);
            while (!at_end(L) && isxdigit((unsigned char)cur(L)))
                advance(L);
        }
        if (cur(L) == 'p' || cur(L) == 'P') {
            is_float = 1;
            advance(L);
            if (cur(L) == '+' || cur(L) == '-') advance(L);
            while (!at_end(L) && isdigit((unsigned char)cur(L)))
                advance(L);
        }
    } else if (cur(L) == '0' && (peek(L, 1) == 'b' || peek(L, 1) == 'B')) {
        advance(L); advance(L);
        while (!at_end(L) && (cur(L) == '0' || cur(L) == '1' ||
               cur(L) == '\''))
            advance(L);
    } else {
        while (!at_end(L) && (isdigit((unsigned char)cur(L)) ||
               cur(L) == '\''))
            advance(L);
        if (cur(L) == '.' && peek(L, 1) != '.') {
            is_float = 1;
            advance(L);
            while (!at_end(L) && isdigit((unsigned char)cur(L)))
                advance(L);
        }
        if (cur(L) == 'e' || cur(L) == 'E') {
            is_float = 1;
            advance(L);
            if (cur(L) == '+' || cur(L) == '-') advance(L);
            while (!at_end(L) && isdigit((unsigned char)cur(L)))
                advance(L);
        }
    }

    while (!at_end(L) && (cur(L) == 'u' || cur(L) == 'U' ||
           cur(L) == 'l' || cur(L) == 'L' ||
           cur(L) == 'f' || cur(L) == 'F'))
        advance(L);

    emit(L, is_float ? TOK_FLOAT_LIT : TOK_INT_LIT,
         start, L->pos - start, start_line, start_col);
}

static void scan_ident(lexer_t *L)
{
    uint32_t start = L->pos;
    uint32_t start_line = L->line;
    uint16_t start_col = (uint16_t)(L->pos - L->line_start + 1);

    while (!at_end(L) && (isalnum((unsigned char)cur(L)) || cur(L) == '_'))
        advance(L);

    uint32_t len = L->pos - start;

    if (cur(L) == '"' || cur(L) == '\'') {
        if ((len == 1 && (L->src[start] == 'L' || L->src[start] == 'u' ||
             L->src[start] == 'U')) ||
            (len == 2 && L->src[start] == 'u' && L->src[start+1] == '8')) {
            scan_string(L, cur(L));
            L->tokens[L->num_tokens - 1].offset = start;
            L->tokens[L->num_tokens - 1].len = L->pos - start;
            return;
        }
    }

    int type = lookup_keyword(L->src + start, len);
    emit(L, type, start, len, start_line, start_col);
}

static void scan_pp_line(lexer_t *L)
{
    uint32_t start = L->pos;
    uint32_t start_line = L->line;
    uint16_t start_col = (uint16_t)(L->pos - L->line_start + 1);

    while (!at_end(L) && cur(L) != '\n') {
        uint32_t n = lx_splc(L);
        if (n == 0) n = 1;
        for (uint32_t i = 0; i < n; i++)
            advance(L);
    }

    emit(L, TOK_PP_LINE, start, L->pos - start, start_line, start_col);
}

/* Where every character learns its place in the hierarchy */

int lexer_tokenize(lexer_t *L)
{
    while (!at_end(L)) {
        skip_whitespace(L);
        if (at_end(L)) break;

        uint32_t start = L->pos;
        uint32_t start_line = L->line;
        uint16_t start_col = (uint16_t)(L->pos - L->line_start + 1);
        char c = cur(L);

        if (c == '/' && peek(L, 1) == '/') {
            skip_line_comment(L);
            continue;
        }
        if (c == '/' && peek(L, 1) == '*') {
            skip_block_comment(L);
            continue;
        }

        if (c == '#') {
            if (peek(L, 1) == '#') {
                advance(L); advance(L);
                emit(L, TOK_DHASH, start, 2, start_line, start_col);
            } else {
                scan_pp_line(L);
            }
            continue;
        }

        if (c == '"' || c == '\'') {
            scan_string(L, c);
            continue;
        }

        if (isdigit((unsigned char)c) ||
            (c == '.' && isdigit((unsigned char)peek(L, 1)))) {
            scan_number(L);
            continue;
        }

        if (isalpha((unsigned char)c) || c == '_') {
            scan_ident(L);
            continue;
        }

        advance(L);
        switch (c) {
        case '(':
            emit(L, TOK_LPAREN, start, 1, start_line, start_col);
            break;
        case ')':
            emit(L, TOK_RPAREN, start, 1, start_line, start_col);
            break;
        case '[':
            emit(L, TOK_LBRACKET, start, 1, start_line, start_col);
            break;
        case ']':
            emit(L, TOK_RBRACKET, start, 1, start_line, start_col);
            break;
        case '{':
            emit(L, TOK_LBRACE, start, 1, start_line, start_col);
            break;
        case '}':
            emit(L, TOK_RBRACE, start, 1, start_line, start_col);
            break;
        case ';':
            emit(L, TOK_SEMI, start, 1, start_line, start_col);
            break;
        case ',':
            emit(L, TOK_COMMA, start, 1, start_line, start_col);
            break;
        case '~':
            emit(L, TOK_TILDE, start, 1, start_line, start_col);
            break;
        case '?':
            emit(L, TOK_QUESTION, start, 1, start_line, start_col);
            break;
        case '.':
            if (cur(L) == '.' && peek(L, 1) == '.') {
                advance(L); advance(L);
                emit(L, TOK_ELLIPSIS, start, 3, start_line, start_col);
            } else {
                emit(L, TOK_DOT, start, 1, start_line, start_col);
            }
            break;
        case ':':
            if (cur(L) == ':') {
                advance(L);
                emit(L, TOK_DCOLON, start, 2, start_line, start_col);
            } else {
                emit(L, TOK_COLON, start, 1, start_line, start_col);
            }
            break;
        case '+':
            if (cur(L) == '+') {
                advance(L);
                emit(L, TOK_INC, start, 2, start_line, start_col);
            } else if (cur(L) == '=') {
                advance(L);
                emit(L, TOK_PLUS_EQ, start, 2, start_line, start_col);
            } else {
                emit(L, TOK_PLUS, start, 1, start_line, start_col);
            }
            break;
        case '-':
            if (cur(L) == '-') {
                advance(L);
                emit(L, TOK_DEC, start, 2, start_line, start_col);
            } else if (cur(L) == '=') {
                advance(L);
                emit(L, TOK_MINUS_EQ, start, 2, start_line, start_col);
            } else if (cur(L) == '>') {
                advance(L);
                emit(L, TOK_ARROW, start, 2, start_line, start_col);
            } else {
                emit(L, TOK_MINUS, start, 1, start_line, start_col);
            }
            break;
        case '*':
            if (cur(L) == '=') {
                advance(L);
                emit(L, TOK_STAR_EQ, start, 2, start_line, start_col);
            } else {
                emit(L, TOK_STAR, start, 1, start_line, start_col);
            }
            break;
        case '/':
            if (cur(L) == '=') {
                advance(L);
                emit(L, TOK_SLASH_EQ, start, 2, start_line, start_col);
            } else {
                emit(L, TOK_SLASH, start, 1, start_line, start_col);
            }
            break;
        case '%':
            if (cur(L) == '=') {
                advance(L);
                emit(L, TOK_PERCENT_EQ, start, 2, start_line, start_col);
            } else {
                emit(L, TOK_PERCENT, start, 1, start_line, start_col);
            }
            break;
        case '&':
            if (cur(L) == '&') {
                advance(L);
                emit(L, TOK_LAND, start, 2, start_line, start_col);
            } else if (cur(L) == '=') {
                advance(L);
                emit(L, TOK_AMP_EQ, start, 2, start_line, start_col);
            } else {
                emit(L, TOK_AMP, start, 1, start_line, start_col);
            }
            break;
        case '|':
            if (cur(L) == '|') {
                advance(L);
                emit(L, TOK_LOR, start, 2, start_line, start_col);
            } else if (cur(L) == '=') {
                advance(L);
                emit(L, TOK_PIPE_EQ, start, 2, start_line, start_col);
            } else {
                emit(L, TOK_PIPE, start, 1, start_line, start_col);
            }
            break;
        case '^':
            if (cur(L) == '=') {
                advance(L);
                emit(L, TOK_CARET_EQ, start, 2, start_line, start_col);
            } else {
                emit(L, TOK_CARET, start, 1, start_line, start_col);
            }
            break;
        case '!':
            if (cur(L) == '=') {
                advance(L);
                emit(L, TOK_NE, start, 2, start_line, start_col);
            } else {
                emit(L, TOK_BANG, start, 1, start_line, start_col);
            }
            break;
        case '=':
            if (cur(L) == '=') {
                advance(L);
                emit(L, TOK_EQ, start, 2, start_line, start_col);
            } else {
                emit(L, TOK_ASSIGN, start, 1, start_line, start_col);
            }
            break;
        case '<':
            if (cur(L) == '<' && peek(L, 1) == '<') {
                advance(L); advance(L);
                emit(L, TOK_LAUNCH_OPEN, start, 3, start_line, start_col);
            } else if (cur(L) == '<' && peek(L, 1) == '=') {
                advance(L); advance(L);
                emit(L, TOK_SHL_EQ, start, 3, start_line, start_col);
            } else if (cur(L) == '<') {
                advance(L);
                emit(L, TOK_SHL, start, 2, start_line, start_col);
            } else if (cur(L) == '=') {
                advance(L);
                emit(L, TOK_LE, start, 2, start_line, start_col);
            } else {
                emit(L, TOK_LT, start, 1, start_line, start_col);
            }
            break;
        case '>':
            if (cur(L) == '>' && peek(L, 1) == '>') {
                advance(L); advance(L);
                emit(L, TOK_LAUNCH_CLOSE, start, 3, start_line, start_col);
            } else if (cur(L) == '>' && peek(L, 1) == '=') {
                advance(L); advance(L);
                emit(L, TOK_SHR_EQ, start, 3, start_line, start_col);
            } else if (cur(L) == '>') {
                advance(L);
                emit(L, TOK_SHR, start, 2, start_line, start_col);
            } else if (cur(L) == '=') {
                advance(L);
                emit(L, TOK_GE, start, 2, start_line, start_col);
            } else {
                emit(L, TOK_GT, start, 1, start_line, start_col);
            }
            break;
        default:
            lex_error(L, BC_E005);
            emit(L, TOK_ERROR, start, 1, start_line, start_col);
            break;
        }
    }

    emit(L, TOK_EOF, L->pos, 0, L->line,
         (uint16_t)(L->pos - L->line_start + 1));

    return L->num_errors > 0 ? BC_ERR_LEX : BC_OK;
}
