/* Triton frontend lexer. Tokenises the Python subset that Triton
 * kernels are written in: recursive descent over characters, plus
 * Python's indentation tokeniser, which keeps a stack of column
 * positions and emits INDENT/DEDENT at block boundaries. */

#include "triton.h"

#include <string.h>
#include <stdio.h>

/* ---- Character classifiers ----
 * Inlined rather than via ctype.h: isalpha can be a real call that
 * touches locale data, and these run on every character. */

static int tx_alph(int c)
{
    return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || c == '_';
}

static int tx_dig(int c)
{
    return c >= '0' && c <= '9';
}

static int tx_idn(int c)
{
    return tx_alph(c) || tx_dig(c);
}

static int tx_hex(int c)
{
    return tx_dig(c) || (c >= 'a' && c <= 'f') || (c >= 'A' && c <= 'F');
}

/* ---- Keyword lookup ----
 * Map a scanned identifier to its reserved-word kind, or TN_TOK_IDENT
 * if it is none. Recognising keywords here lets the parser and sema
 * report misuse with a clear message rather than a generic one. */

typedef struct { const char *name; int kind; } tn_kw_t;

static const tn_kw_t tn_kws[] = {
    {"def",       TN_TOK_KW_DEF},
    {"return",    TN_TOK_KW_RETURN},
    {"pass",      TN_TOK_KW_PASS},
    {"if",        TN_TOK_KW_IF},
    {"elif",      TN_TOK_KW_ELIF},
    {"else",      TN_TOK_KW_ELSE},
    {"for",       TN_TOK_KW_FOR},
    {"while",     TN_TOK_KW_WHILE},
    {"break",     TN_TOK_KW_BREAK},
    {"continue",  TN_TOK_KW_CONTINUE},
    {"in",        TN_TOK_KW_IN},
    {"not",       TN_TOK_KW_NOT},
    {"and",       TN_TOK_KW_AND},
    {"or",        TN_TOK_KW_OR},
    {"is",        TN_TOK_KW_IS},
    {"None",      TN_TOK_KW_NONE},
    {"True",      TN_TOK_KW_TRUE},
    {"False",     TN_TOK_KW_FALSE},
    {"import",    TN_TOK_KW_IMPORT},
    {"from",      TN_TOK_KW_FROM},
    {"as",        TN_TOK_KW_AS},
    {"lambda",    TN_TOK_KW_LAMBDA},
    {"global",    TN_TOK_KW_GLOBAL},
    {"nonlocal",  TN_TOK_KW_NONLOCAL},
    {"assert",    TN_TOK_KW_ASSERT},
    {"del",       TN_TOK_KW_DEL},
    {NULL, 0}
};

static int tn_kw_lookup(const char *s, uint32_t len)
{
    for (int i = 0; tn_kws[i].name != NULL; i++) {
        const char *n = tn_kws[i].name;
        uint32_t nl = (uint32_t)strlen(n);
        if (nl == len && memcmp(n, s, len) == 0)
            return tn_kws[i].kind;
    }
    return TN_TOK_IDENT;
}

/* ---- Token name table ----
 * For --triton --lex output and diagnostics. Order must match the
 * token enum; nothing enforces that, so a new token kind needs its
 * name added in the same position. */

static const char *tn_tok_names[TN_TOK_COUNT] = {
    "NEWLINE", "INDENT", "DEDENT",
    "IDENT", "INT", "FLOAT", "STRING",
    "def", "return", "pass",
    "if", "elif", "else",
    "for", "while", "break", "continue",
    "in", "not", "and", "or", "is",
    "None", "True", "False",
    "import", "from", "as",
    "lambda", "global", "nonlocal",
    "assert", "del",
    "(", ")", "[", "]", "{", "}",
    ":", ",", ";", ".",
    "@", "->", "=",
    "+=", "-=", "*=", "/=", "//=", "%=", "**=",
    "&=", "|=", "^=", "<<=", ">>=", "@=",
    "+", "-", "*", "/", "//", "%", "**",
    "&", "|", "^", "~", "<<", ">>",
    "<", ">", "<=", ">=", "==", "!=",
    ":=", "EOF"
};

const char *tn_tok_name(int kind)
{
    if (kind < 0 || kind >= TN_TOK_COUNT) return "?";
    return tn_tok_names[kind];
}

/* ---- Lexer Bookkeeping ---- */

static void tn_err(tn_lex_t *L, uint16_t eid, const char *msg)
{
    if (L->num_errors >= BC_MAX_ERRORS) return;
    bc_error_t *e = &L->errors[L->num_errors++];
    e->eid = eid;
    e->loc.line = L->line;
    e->loc.col  = (uint16_t)(L->pos - L->line_start);
    snprintf(e->msg, sizeof(e->msg), "%s", msg);
}

static void tn_emit(tn_lex_t *L, int kind, uint32_t off, uint32_t len)
{
    if (L->num_tokens >= L->max_tokens) return;
    tn_tok_t *t = &L->tokens[L->num_tokens++];
    t->kind = kind;
    t->off  = off;
    t->len  = len;
    t->line = L->line;
    t->col  = (uint16_t)(off - L->line_start);
    t->pad  = 0;
}

/* Variant of tn_emit that records the token's position from the start
 * of the literal rather than the lexer's current location. Multi-line
 * scanners need it: by the time they finish, L->line and L->line_start
 * have moved past the token's start, so plain tn_emit would compute a
 * bogus column. */

static void tn_emit_at(tn_lex_t *L, int kind, uint32_t off, uint32_t len,
                       uint32_t line, uint32_t line_start)
{
    if (L->num_tokens >= L->max_tokens) return;
    tn_tok_t *t = &L->tokens[L->num_tokens++];
    t->kind = kind;
    t->off  = off;
    t->len  = len;
    t->line = line;
    t->col  = (uint16_t)(off - line_start);
    t->pad  = 0;
}

/* ---- Indentation handler ----
 * Called at the start of a logical line when not inside parens or
 * brackets. Counts leading spaces, skips blank and comment-only lines,
 * then emits INDENT, a run of DEDENTs, or nothing depending on how the
 * indentation compares to the stack. Tabs count as one column each. */

static int tn_indent(tn_lex_t *L)
{
    /* Skip blank and comment-only lines: they do not affect the
     * indentation calculation. */
    for (;;) {
        uint32_t start = L->pos;
        int col = 0;
        while (L->pos < L->src_len &&
               (L->src[L->pos] == ' ' || L->src[L->pos] == '\t')) {
            col++;
            L->pos++;
        }

        if (L->pos >= L->src_len) {
            /* End of file. Close any open blocks by queuing DEDENTs. */
            L->pending_dedents = L->indent_depth;
            L->indent_depth = 0;
            return 1;
        }

        char c = L->src[L->pos];
        if (c == '\r') {
            /* Lone carriage return (usually the first half of CRLF) is
             * inline whitespace. Skip it; a following '\n' is handled
             * by the blank-line case below. */
            L->pos++;
            continue;
        }
        if (c == '#') {
            /* Comment line, skip to newline and try again. */
            while (L->pos < L->src_len && L->src[L->pos] != '\n')
                L->pos++;
            if (L->pos < L->src_len) {
                L->pos++;
                L->line++;
                L->line_start = L->pos;
            }
            continue;
        }
        if (c == '\n') {
            L->pos++;
            L->line++;
            L->line_start = L->pos;
            continue;
        }

        /* A real token starts here. Compare col against the top of
         * the indent stack. */
        int top = (L->indent_depth > 0)
                  ? L->indents[L->indent_depth - 1]
                  : 0;
        if (col > top) {
            if (L->indent_depth >= TN_MAX_INDENTS) {
                tn_err(L, 50, "indentation too deeply nested");
                return 0;
            }
            L->indents[L->indent_depth++] = col;
            tn_emit(L, TN_TOK_INDENT, start, 0);
        } else if (col < top) {
            while (L->indent_depth > 0 &&
                   L->indents[L->indent_depth - 1] > col) {
                L->indent_depth--;
                tn_emit(L, TN_TOK_DEDENT, start, 0);
            }
            int new_top = (L->indent_depth > 0)
                          ? L->indents[L->indent_depth - 1]
                          : 0;
            if (new_top != col) {
                tn_err(L, 51, "inconsistent dedent");
                return 0;
            }
        }
        L->at_line_start = 0;
        return 1;
    }
}

/* ---- Identifier and Keyword Scan ---- */

static void tn_scan_ident(tn_lex_t *L)
{
    uint32_t start = L->pos;
    while (L->pos < L->src_len && tx_idn(L->src[L->pos]))
        L->pos++;
    uint32_t len = L->pos - start;
    int kind = tn_kw_lookup(L->src + start, len);
    tn_emit(L, kind, start, len);
}

/* ---- Numeric literal scan ----
 * Handles 0x/0o/0b integer prefixes, decimal integers, and floats with
 * optional fraction and exponent. Underscore digit separators are
 * accepted and ignored for now; sema can parse them if a kernel ever
 * uses one. */

static void tn_scan_num(tn_lex_t *L)
{
    uint32_t start = L->pos;
    int is_float = 0;

    if (L->pos + 1 < L->src_len && L->src[L->pos] == '0') {
        char p = L->src[L->pos + 1];
        if (p == 'x' || p == 'X') {
            L->pos += 2;
            while (L->pos < L->src_len &&
                   (tx_hex(L->src[L->pos]) || L->src[L->pos] == '_'))
                L->pos++;
            tn_emit(L, TN_TOK_INT, start, L->pos - start);
            return;
        }
        if (p == 'o' || p == 'O' || p == 'b' || p == 'B') {
            L->pos += 2;
            while (L->pos < L->src_len &&
                   (tx_dig(L->src[L->pos]) || L->src[L->pos] == '_'))
                L->pos++;
            tn_emit(L, TN_TOK_INT, start, L->pos - start);
            return;
        }
    }

    while (L->pos < L->src_len &&
           (tx_dig(L->src[L->pos]) || L->src[L->pos] == '_'))
        L->pos++;

    /* Fractional dot, greedy. This function is only entered on a
     * leading digit, or on a dot already known to precede a digit, so a
     * dot here belongs to the float and is never member access. Matches
     * CPython, which lexes `5.foo` as `5.` then `foo`. */
    if (L->pos < L->src_len && L->src[L->pos] == '.') {
        is_float = 1;
        L->pos++;
        while (L->pos < L->src_len &&
               (tx_dig(L->src[L->pos]) || L->src[L->pos] == '_'))
            L->pos++;
    }

    if (L->pos < L->src_len &&
        (L->src[L->pos] == 'e' || L->src[L->pos] == 'E')) {
        is_float = 1;
        L->pos++;
        if (L->pos < L->src_len &&
            (L->src[L->pos] == '+' || L->src[L->pos] == '-'))
            L->pos++;
        while (L->pos < L->src_len && tx_dig(L->src[L->pos]))
            L->pos++;
    }

    /* Imaginary 'j' suffix: accepted as part of the number and lexed as
     * FLOAT. The rest of the compiler ignores the suffix; sema handles
     * it later. */
    if (L->pos < L->src_len &&
        (L->src[L->pos] == 'j' || L->src[L->pos] == 'J')) {
        is_float = 1;
        L->pos++;
    }

    tn_emit(L, is_float ? TN_TOK_FLOAT : TN_TOK_INT, start, L->pos - start);
}

/* ---- String literal scan ----
 * Single-line strings are single or double quoted; triple-quoted
 * strings span lines and hold docstrings. String prefixes (r, b, f, u,
 * or two of them) are detected first and folded into the STRING token
 * length so the parser sees one token per literal; the prefix's
 * meaning (raw, bytes, formatted, unicode) is left to sema. f-strings
 * are accepted syntactically without parsing the embedded expressions;
 * sema diagnoses them if a kernel uses one. */

static void tn_scan_str_at(tn_lex_t *L, uint32_t start)
{
    /* Snapshot the line position at the start of the literal so a
     * triple-quoted string reports a sensible line and column when the
     * closing quote arrives many lines later. */
    uint32_t start_line       = L->line;
    uint32_t start_line_start = L->line_start;

    char quote = L->src[L->pos];

    /* Triple-quoted: three matching quotes opens a string that ends
     * only at three matching quotes again, with newlines counted but
     * not terminating the literal. Docstrings live here. */
    if (L->pos + 2 < L->src_len &&
        L->src[L->pos + 1] == quote &&
        L->src[L->pos + 2] == quote) {
        L->pos += 3;
        uint32_t guard = 0;
        while (L->pos + 2 < L->src_len && guard < L->src_len) {
            if (L->src[L->pos] == quote &&
                L->src[L->pos + 1] == quote &&
                L->src[L->pos + 2] == quote) {
                L->pos += 3;
                tn_emit_at(L, TN_TOK_STRING, start, L->pos - start,
                           start_line, start_line_start);
                return;
            }
            if (L->src[L->pos] == '\\' && L->pos + 1 < L->src_len) {
                if (L->src[L->pos + 1] == '\n') {
                    L->line++;
                    L->line_start = L->pos + 2;
                }
                L->pos += 2;
                guard++;
                continue;
            }
            if (L->src[L->pos] == '\n') {
                L->line++;
                L->line_start = L->pos + 1;
            }
            L->pos++;
            guard++;
        }
        tn_err(L, 56, "unterminated triple-quoted string");
        tn_emit_at(L, TN_TOK_STRING, start, L->pos - start,
                   start_line, start_line_start);
        return;
    }

    /* Single-line single or double quoted. */
    L->pos++;
    while (L->pos < L->src_len && L->src[L->pos] != quote) {
        if (L->src[L->pos] == '\\' && L->pos + 1 < L->src_len) {
            if (L->src[L->pos + 1] == '\n') {
                /* Explicit line continuation inside a string. Rare
                 * but legal. */
                L->line++;
                L->line_start = L->pos + 2;
            }
            L->pos += 2;
            continue;
        }
        if (L->src[L->pos] == '\n') {
            tn_err(L, 52, "unterminated string literal");
            tn_emit_at(L, TN_TOK_STRING, start, L->pos - start,
                       start_line, start_line_start);
            return;
        }
        L->pos++;
    }
    if (L->pos < L->src_len) L->pos++;
    tn_emit_at(L, TN_TOK_STRING, start, L->pos - start,
               start_line, start_line_start);
}

static void tn_scan_str(tn_lex_t *L)
{
    tn_scan_str_at(L, L->pos);
}

/* String prefix detection. Returns the prefix length (1 or 2) when the
 * character after the prefix is a quote, else zero. Accepts any one or
 * two of r/R, b/B, f/F, u/U. Python forbids some combinations (fb is
 * invalid, bf is not); the lexer stays permissive and lets sema reject
 * the odd ones, in keeping with the compiler's keep-going recovery. */

static int tn_is_str_prefix(const tn_lex_t *L)
{
    uint32_t p = L->pos;
    int n = 0;
    while (n < 2 && p < L->src_len) {
        char c = L->src[p];
        if (c == 'r' || c == 'R' || c == 'b' || c == 'B' ||
            c == 'f' || c == 'F' || c == 'u' || c == 'U') {
            p++;
            n++;
        } else {
            break;
        }
    }
    if (n == 0) return 0;
    if (p >= L->src_len) return 0;
    if (L->src[p] != '"' && L->src[p] != '\'') return 0;
    return n;
}

/* ---- Operator and punctuation scan ----
 * Maximum munch: multi-character operators are matched before their
 * prefixes, ** before *, // before /, == before =. Each case tests the
 * current character then looks ahead for a second-character extension. */

static int tn_match2(tn_lex_t *L, char c2)
{
    if (L->pos + 1 < L->src_len && L->src[L->pos + 1] == c2) {
        return 1;
    }
    return 0;
}

static void tn_emit_op(tn_lex_t *L, int kind, uint32_t n)
{
    tn_emit(L, kind, L->pos, n);
    L->pos += n;
}

static void tn_scan_op(tn_lex_t *L)
{
    char c = L->src[L->pos];
    switch (c) {
    case '(':
        tn_emit_op(L, TN_TOK_LPAREN, 1);
        L->paren_depth++;
        return;
    case ')':
        tn_emit_op(L, TN_TOK_RPAREN, 1);
        if (L->paren_depth > 0) L->paren_depth--;
        return;
    case '[':
        tn_emit_op(L, TN_TOK_LBRACK, 1);
        L->paren_depth++;
        return;
    case ']':
        tn_emit_op(L, TN_TOK_RBRACK, 1);
        if (L->paren_depth > 0) L->paren_depth--;
        return;
    case '{':
        tn_emit_op(L, TN_TOK_LBRACE, 1);
        L->paren_depth++;
        return;
    case '}':
        tn_emit_op(L, TN_TOK_RBRACE, 1);
        if (L->paren_depth > 0) L->paren_depth--;
        return;
    case ',': tn_emit_op(L, TN_TOK_COMMA, 1); return;
    case ';': tn_emit_op(L, TN_TOK_SEMI, 1); return;
    case '.': tn_emit_op(L, TN_TOK_DOT, 1); return;
    case '~': tn_emit_op(L, TN_TOK_TILDE, 1); return;
    case ':':
        if (tn_match2(L, '=')) { tn_emit_op(L, TN_TOK_WALRUS, 2); return; }
        tn_emit_op(L, TN_TOK_COLON, 1); return;
    case '@':
        if (tn_match2(L, '=')) { tn_emit_op(L, TN_TOK_AMATMUL, 2); return; }
        tn_emit_op(L, TN_TOK_AT, 1); return;
    case '+':
        if (tn_match2(L, '=')) { tn_emit_op(L, TN_TOK_AADD, 2); return; }
        tn_emit_op(L, TN_TOK_PLUS, 1); return;
    case '-':
        if (tn_match2(L, '=')) { tn_emit_op(L, TN_TOK_ASUB, 2); return; }
        if (tn_match2(L, '>')) { tn_emit_op(L, TN_TOK_ARROW, 2); return; }
        tn_emit_op(L, TN_TOK_MINUS, 1); return;
    case '*':
        if (L->pos + 1 < L->src_len && L->src[L->pos + 1] == '*') {
            if (L->pos + 2 < L->src_len && L->src[L->pos + 2] == '=') {
                tn_emit_op(L, TN_TOK_APOW, 3);
                return;
            }
            tn_emit_op(L, TN_TOK_DSTAR, 2);
            return;
        }
        if (tn_match2(L, '=')) { tn_emit_op(L, TN_TOK_AMUL, 2); return; }
        tn_emit_op(L, TN_TOK_STAR, 1); return;
    case '/':
        if (L->pos + 1 < L->src_len && L->src[L->pos + 1] == '/') {
            if (L->pos + 2 < L->src_len && L->src[L->pos + 2] == '=') {
                tn_emit_op(L, TN_TOK_AFDIV, 3);
                return;
            }
            tn_emit_op(L, TN_TOK_FSLASH, 2);
            return;
        }
        if (tn_match2(L, '=')) { tn_emit_op(L, TN_TOK_ADIV, 2); return; }
        tn_emit_op(L, TN_TOK_SLASH, 1); return;
    case '%':
        if (tn_match2(L, '=')) { tn_emit_op(L, TN_TOK_AMOD, 2); return; }
        tn_emit_op(L, TN_TOK_PERCENT, 1); return;
    case '&':
        if (tn_match2(L, '=')) { tn_emit_op(L, TN_TOK_AAND, 2); return; }
        tn_emit_op(L, TN_TOK_AMP, 1); return;
    case '|':
        if (tn_match2(L, '=')) { tn_emit_op(L, TN_TOK_AOR, 2); return; }
        tn_emit_op(L, TN_TOK_PIPE, 1); return;
    case '^':
        if (tn_match2(L, '=')) { tn_emit_op(L, TN_TOK_AXOR, 2); return; }
        tn_emit_op(L, TN_TOK_CARET, 1); return;
    case '<':
        if (L->pos + 1 < L->src_len && L->src[L->pos + 1] == '<') {
            if (L->pos + 2 < L->src_len && L->src[L->pos + 2] == '=') {
                tn_emit_op(L, TN_TOK_ASHL, 3);
                return;
            }
            tn_emit_op(L, TN_TOK_SHL, 2);
            return;
        }
        if (tn_match2(L, '=')) { tn_emit_op(L, TN_TOK_LE, 2); return; }
        tn_emit_op(L, TN_TOK_LT, 1); return;
    case '>':
        if (L->pos + 1 < L->src_len && L->src[L->pos + 1] == '>') {
            if (L->pos + 2 < L->src_len && L->src[L->pos + 2] == '=') {
                tn_emit_op(L, TN_TOK_ASHR, 3);
                return;
            }
            tn_emit_op(L, TN_TOK_SHR, 2);
            return;
        }
        if (tn_match2(L, '=')) { tn_emit_op(L, TN_TOK_GE, 2); return; }
        tn_emit_op(L, TN_TOK_GT, 1); return;
    case '=':
        if (tn_match2(L, '=')) { tn_emit_op(L, TN_TOK_EQ, 2); return; }
        tn_emit_op(L, TN_TOK_ASSIGN, 1); return;
    case '!':
        if (tn_match2(L, '=')) { tn_emit_op(L, TN_TOK_NE, 2); return; }
        /* A bare ! is not valid Python. Report it and advance so the
         * parser can keep going. */
        tn_err(L, 53, "unexpected character '!'");
        L->pos++;
        return;
    default:
        tn_err(L, 54, "unexpected character");
        L->pos++;
        return;
    }
}

/* ---- Initialisation ---- */

void tn_lex_init(tn_lex_t *L, const char *src, uint32_t len,
                 tn_tok_t *tokens, uint32_t max_tokens)
{
    memset(L, 0, sizeof(*L));
    L->src = src;
    L->src_len = len;
    L->pos = 0;
    L->line = 1;
    L->line_start = 0;
    L->tokens = tokens;
    L->max_tokens = max_tokens;
    L->num_tokens = 0;
    L->indent_depth = 0;
    L->paren_depth = 0;
    L->pending_dedents = 0;
    L->at_line_start = 1;
}

/* ---- Top-level tokenise loop ---- */

int tn_tokenize(tn_lex_t *L)
{
    uint32_t guard = 0;
    while (L->pos < L->src_len) {
        if (++guard > L->src_len * 4) {
            tn_err(L, 55, "lexer made no progress (internal bug)");
            return BC_ERR_TRITON;
        }

        if (L->pending_dedents > 0) {
            tn_emit(L, TN_TOK_DEDENT, L->pos, 0);
            L->pending_dedents--;
            continue;
        }

        if (L->at_line_start && L->paren_depth == 0) {
            if (!tn_indent(L)) return BC_ERR_TRITON;
            continue;
        }

        char c = L->src[L->pos];

        /* Skip blank space inside a line. */
        if (c == ' ' || c == '\t' || c == '\r') {
            L->pos++;
            continue;
        }

        /* Explicit line continuation via backslash newline. Eat both
         * and stay in the current logical line. */
        if (c == '\\' && L->pos + 1 < L->src_len &&
            L->src[L->pos + 1] == '\n') {
            L->pos += 2;
            L->line++;
            L->line_start = L->pos;
            continue;
        }

        /* Newline. Inside parens or brackets, it is just whitespace;
         * outside, it is a logical line terminator. */
        if (c == '\n') {
            if (L->paren_depth == 0) {
                tn_emit(L, TN_TOK_NEWLINE, L->pos, 0);
                L->at_line_start = 1;
            }
            L->pos++;
            L->line++;
            L->line_start = L->pos;
            continue;
        }

        /* Inline comment. Skip to newline but do not consume it. */
        if (c == '#') {
            while (L->pos < L->src_len && L->src[L->pos] != '\n')
                L->pos++;
            continue;
        }

        if (tx_alph(c)) {
            /* String prefixes (r, b, f, u, or two-letter combinations
             * like rb or fr) take the whole literal as one STRING token
             * even though they start with an identifier letter. Detect
             * them before the normal identifier scan. */
            int pre = tn_is_str_prefix(L);
            if (pre > 0) {
                uint32_t start = L->pos;
                L->pos += (uint32_t)pre;
                tn_scan_str_at(L, start);
                continue;
            }
            tn_scan_ident(L);
            continue;
        }
        if (tx_dig(c)) { tn_scan_num(L); continue; }

        /* A dot followed by a digit is a `.5`-style float. Spotted here
         * because tn_scan_num is otherwise only entered on a leading
         * digit, and a bare dot must stay member access everywhere
         * else. */
        if (c == '.' && L->pos + 1 < L->src_len &&
            tx_dig(L->src[L->pos + 1])) {
            tn_scan_num(L);
            continue;
        }

        if (c == '"' || c == '\'') { tn_scan_str(L); continue; }

        tn_scan_op(L);
    }

    /* End of source. Emit a final NEWLINE unless already on a fresh
     * line, then close remaining indent levels with DEDENTs before EOF,
     * so the parser sees the same shape whether or not the file ended
     * with a trailing newline. */
    if (L->num_tokens > 0 &&
        L->tokens[L->num_tokens - 1].kind != TN_TOK_NEWLINE) {
        tn_emit(L, TN_TOK_NEWLINE, L->pos, 0);
    }
    while (L->indent_depth > 0) {
        tn_emit(L, TN_TOK_DEDENT, L->pos, 0);
        L->indent_depth--;
    }
    tn_emit(L, TN_TOK_EOF, L->pos, 0);
    return BC_OK;
}

int tn_tok_text(const tn_lex_t *L, const tn_tok_t *tok,
                char *buf, int bufsize)
{
    int n = (int)tok->len;
    if (n >= bufsize) n = bufsize - 1;
    if (n > 0) memcpy(buf, L->src + tok->off, (size_t)n);
    buf[n > 0 ? n : 0] = '\0';
    return n;
}
