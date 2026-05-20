/* Triton frontend lexer: a tokenizer for the dialect of Python that
 * Triton kernels are written in. We do this entirely in C99, no
 * embedded Python, no shelled-out helper script, no third party
 * library, and no apologies. The grammar we have to handle is a
 * small subset of Python, the algorithm is a textbook recursive
 * descent over characters, and the only genuinely novel ingredient
 * is Python's indentation tokenizer, which maintains a stack of
 * column positions and emits INDENT and DEDENT tokens at the
 * boundaries of nested blocks. This is the same algorithm CPython
 * uses, written here in a more economical handful of lines because
 * we are doing less, do not have to retain bug-for-bug compatibility
 * with twenty years of CPython, and are not measured on how often we
 * occured in this morning's pandas trace.
 *
 * The historical irony is not lost on us. C was the language Python
 * was written in to begin with, and Python is the language OpenAI
 * decided every modern GPU kernel should be authored in, and now C
 * is the language we are using to parse the Python that produces
 * the GPU kernels. The wheel turns; the toolchain accumulates
 * layers of cake; somebody, somewhere, is eventually going to want
 * a Fortran frontend, and we will at least know how to type one. */

#include "triton.h"

#include <string.h>
#include <stdio.h>

/* ---- Character Classifiers ----
 * Inlined rather than calling out to ctype.h, because the project's
 * style is to keep the dependencies as close to zero as decency
 * allows, and isalpha is a surprisingly expensive function call on
 * some libc implementations once you start counting cache misses on
 * the locale data. */

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

/* ---- Keyword Lookup ----
 * After an identifier has been tokenized, we look it up here to see
 * whether it is a reserved word in Triton's Python subset. Triton
 * itself rejects most of Python's reserved words inside @triton.jit
 * functions, but the lexer's job is to recognise them so that the
 * parser or sema can complain about them in a more pleasant manner
 * than "unexpected garbage" further down the road. */

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

/* ---- Token Name Table ----
 * Used by --triton --lex and by diagnostic output. The order has to
 * match the enum, which is enforced by careful reading rather than by
 * the compiler, so anyone adding a token kind needs to add a name in
 * the same position or the diagnostics will print rubbish. */

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

/* tn_emit_at is the variant that lets a scanner record the token's
 * position from the start of the literal rather than from the
 * lexer's current location. Multi-line scanners need it because by
 * the time they finish the scan, L->line and L->line_start have
 * moved past the token's beginning and the regular tn_emit would
 * produce a column number that has rolled under into the millions.
 * Column 65227 is technically a valid column on a sufficiently wide
 * editor but is rarely what the programmer meant, and your error
 * messages tend to lose the sympathy of the reader once they cross
 * the four digit threshold. */

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

/* ---- Indentation Handler ----
 * Called when the lexer is at the start of a logical line and is not
 * inside any parentheses or brackets. It counts the leading spaces,
 * advances past blank or comment-only lines without emitting
 * anything, then either emits INDENT, a run of DEDENTs, or nothing
 * at all depending on how the new indentation compares to the
 * current stack. Tabs are converted to one column each, which is
 * not strictly what CPython does but is good enough for Triton
 * kernels written by people who set their editors up sensibly.
 *
 * If you are reading this comment because you discovered that mixing
 * tabs and spaces broke your file, the correct response is to pick
 * one and stick with it for the rest of your career, rather than to
 * file a bug. PEP 8 has been suggesting this for over twenty years,
 * and CPython has been getting progressively less patient with
 * people who refuse to listen, so the trajectory is clear. */

static int tn_indent(tn_lex_t *L)
{
    /* Skip blank and comment-only lines without taking them into
     * account for the indentation calculation. CPython does the same
     * thing for the same reason: a blank line is a typographical
     * convenience, not a structural statement. */
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

/* ---- Numeric Literal Scan ----
 * Python accepts 0x/0o/0b prefixes for non-decimal integers, a decimal
 * with optional fractional and exponent parts for floats, and
 * underscores as digit separators (which we accept silently and
 * ignore for now, on the grounds that none of the literals we will
 * meet in early Triton kernels will use them, and the day we do meet
 * one is the day we add proper parsing in the sema). */

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

    /* Fractional dot, greedy. The main loop only enters this function
     * with a leading digit or with a dot already known to be followed
     * by a digit, so any dot we meet here belongs to the float and
     * cannot possibly be the dot of a member access. This is also
     * how CPython's tokenizer treats `5.foo`, which it splits into
     * the float literal `5.` and the identifier `foo` rather than
     * into an integer followed by member access. The reasoning, if
     * you ask CPython about it definately enough, is that nobody
     * writes `5.foo` on purpose, and anyone who does is going to
     * get a syntax error from the parser shortly after this point
     * anyway, so we may as well decide here. */
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

    /* Imaginary suffix and the float-only j literal are accepted as
     * part of the number but the rest of the compiler does not know
     * what to do with them; we tokenize them as FLOAT and let sema
     * shrug at the suffix later. */
    if (L->pos < L->src_len &&
        (L->src[L->pos] == 'j' || L->src[L->pos] == 'J')) {
        is_float = 1;
        L->pos++;
    }

    tn_emit(L, is_float ? TN_TOK_FLOAT : TN_TOK_INT, start, L->pos - start);
}

/* ---- String Literal Scan ----
 * Single-line strings can be either single or double quoted, on the
 * principle that the people who designed Python could not seperate
 * the two camps in good conscience and decided the user could
 * choose. Triple quoted strings span multiple lines and exist
 * primarily to host the docstring at the top of every well behaved
 * Triton kernel, in which the author apologises in advance for what
 * is about to happen to the GPU in the rest of the function.
 *
 * String prefixes (r, b, f, u, plus any one of those followed by
 * another letter) are detected before the scan begins and folded
 * into the STRING token's length so that the parser sees one token
 * per literal regardless of how the user dressed it up that
 * morning. The actual semantic effect of the prefix (raw, bytes,
 * formatted, unicode) is deferred to sema, which is the natural
 * place to decide what to do about it because sema is the only
 * pass that has the time and the equanimity.
 *
 * We accept f-strings syntactically here without parsing the
 * embedded expressions, on the entirely reasonable grounds that
 * the kernels we are likely to meet do not actually use them and
 * the rare ones that do can be greeted with a polite diagnostic
 * when sema notices. F-strings are, for the record, a parser's
 * revenge for several centuries of being treated like an unfussy
 * filing clerk: the price of letting users write {x:.4f} inside a
 * string and have the runtime format it is that somebody now has
 * to parse that mess, and that somebody is not us, not today. */

static void tn_scan_str_at(tn_lex_t *L, uint32_t start)
{
    /* Snapshot the line position at the start of the literal so that
     * triple-quoted strings can still report a sensible line and
     * column when the closing quote arrives many lines later. */
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

/* String prefix detection. Returns the prefix length in characters
 * (1 or 2) if the next char after the prefix is a quote, otherwise
 * zero. We accept any combination of one or two letters from the
 * Python prefix set: r/R, b/B, f/F, u/U.
 *
 * Python's actual grammar forbids certain combinations (for
 * instance, fb is not legal, only bf or rb) but we are deliberately
 * permissive in the lexer and let sema complain about wierd
 * combinations later. This is consistent with the rest of the
 * error recovery in the compiler, which would rather keep
 * tokenising and tell you about three problems than die at the
 * first one and leave you guessing about the other two. */

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

/* ---- Operator and Punctuation Scan ----
 * Multi-character operators come first because of the maximum munch
 * principle: we want ** before *, // before /, == before =, and so
 * on. The structure is a sequence of small decisions on the current
 * character followed by lookahead for any second-character extension
 * that might be present. */

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
        /* A bare ! is not valid Python. Emit something so we make
         * forward progress and let the parser complain about it. */
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

/* ---- Top Level Tokenize Loop ---- */

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
        if (c == ' ' || c == '\t') {
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
            /* String prefixes (r, b, f, u, plus two-letter combinations
             * like rb or fr) want to take the whole literal as a single
             * STRING token even though they begin with an identifier
             * letter. Detect those before falling into the normal
             * identifier scan. */
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

        /* A dot followed directly by a digit is a float literal of the
         * `.5` variety. We have to spot it here because tn_scan_num is
         * otherwise only entered on a leading digit, and we want to
         * keep the bare dot meaning member access in every other
         * context. */
        if (c == '.' && L->pos + 1 < L->src_len &&
            tx_dig(L->src[L->pos + 1])) {
            tn_scan_num(L);
            continue;
        }

        if (c == '"' || c == '\'') { tn_scan_str(L); continue; }

        tn_scan_op(L);
    }

    /* End of source. Emit a final NEWLINE if we are not already on
     * a fresh line, then close any remaining indent levels with
     * DEDENT tokens before the EOF, so the parser receives the same
     * shape regardless of whether the file ended with a trailing
     * newline. */
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
