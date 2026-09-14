/* ============================================================
 * preproc.c — CUDA Preprocessor for Booth
 *
 * Runs before the lexer on raw source text.
 * Handles: #include, #define/#undef, macro expansion,
 *          #ifdef/#ifndef/#if/#elif/#else/#endif, #pragma, #error.
 * Output: single expanded source buffer ready for lexer.
 *
 * Not a full C preprocessor — just enough for CUDA headers and
 * user macros. Covers the subset that real .cu files use.
 * ============================================================ */

#include "preproc.h"
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>
#include <ctype.h>

/* ---- Error reporting ---- */

static void pp_error(preproc_t *pp, bc_eid_t eid, ...)
{
    if (pp->num_errors >= BC_MAX_ERRORS) return;
    bc_error_t *e = &pp->errors[pp->num_errors++];
    e->loc.line = pp->line;
    e->loc.col = 1;
    e->loc.offset = pp->pos;
    e->code = BC_ERR_PREPROC;
    e->eid  = (uint16_t)eid;
    va_list ap;
    va_start(ap, eid);
    vsnprintf(e->msg, sizeof(e->msg), bc_efmt(eid), ap);
    va_end(ap);
}

/* ---- Character-level utilities ---- */

static int pp_at_end(const preproc_t *pp)
{
    return pp->pos >= pp->src_len;
}

static char pp_cur(const preproc_t *pp)
{
    return pp->pos < pp->src_len ? pp->src[pp->pos] : '\0';
}

static char pp_peek(const preproc_t *pp, uint32_t ahead)
{
    uint32_t idx = pp->pos + ahead;
    return idx < pp->src_len ? pp->src[idx] : '\0';
}

static void pp_advance(preproc_t *pp)
{
    if (pp->pos < pp->src_len) {
        if (pp->src[pp->pos] == '\n')
            pp->line++;
        pp->pos++;
    }
}

static void pp_skip_hspace(preproc_t *pp)
{
    while (!pp_at_end(pp) && (pp_cur(pp) == ' ' || pp_cur(pp) == '\t'))
        pp_advance(pp);
}

static void pp_skip_to_eol(preproc_t *pp)
{
    while (!pp_at_end(pp) && pp_cur(pp) != '\n')
        pp_advance(pp);
}

static uint32_t pp_nlsz(const preproc_t *pp)
{
    if (pp_cur(pp) == '\n') return 1;
    if (pp_cur(pp) == '\r' && pp_peek(pp, 1) == '\n') return 2;
    return 0;
}

static uint32_t pp_splc(const preproc_t *pp)
{
    if (pp_cur(pp) != '\\') return 0;
    if (pp_peek(pp, 1) == '\n') return 2;
    if (pp_peek(pp, 1) == '\r' && pp_peek(pp, 2) == '\n') return 3;
    return 0;
}

static int pp_is_ident_start(char c)
{
    return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || c == '_';
}

static int pp_is_ident_char(char c)
{
    return pp_is_ident_start(c) || (c >= '0' && c <= '9');
}

/* Comments are gone before macros are seen, everywhere except here: a body was
 * kept whole, so '#define S 5 // why' commented out every line that used S, and
 * a joined invocation would lose everything after a // on its first line.
 * A block comment left open is not this function's business. */
static uint32_t stripc(char *s, uint32_t n)
{
    uint32_t r = 0, w = 0;
    while (r < n) {
        if (s[r] == '"' || s[r] == '\'') {
            char q = s[r];
            s[w++] = s[r++];
            while (r < n && s[r] != q) {
                if (s[r] == '\\' && r + 1 < n) s[w++] = s[r++];
                s[w++] = s[r++];
            }
            if (r < n) s[w++] = s[r++];
            continue;
        }
        if (s[r] == '/' && r + 1 < n && s[r+1] == '/') break;
        if (s[r] == '/' && r + 1 < n && s[r+1] == '*') {
            uint32_t e = r + 2;
            while (e + 1 < n && !(s[e] == '*' && s[e+1] == '/')) e++;
            if (e + 1 >= n) break;
            s[w++] = ' ';
            r = e + 2;
            continue;
        }
        s[w++] = s[r++];
    }
    return w;
}

/* ---- Output helpers ---- */

/* One byte of out_max is held back so the buffer can always be terminated;
 * the lexer reads until NUL and a full buffer used to run it into whatever
 * static followed. */
static int pp_room(preproc_t *pp, uint32_t need)
{
    if (pp->out_len + need < pp->out_max) return 1;
    if (!pp->ovflw) {
        pp->ovflw = 1;
        pp_error(pp, BC_E053, pp->out_max);
    }
    return 0;
}

static void pp_emit_char(preproc_t *pp, char c)
{
    if (pp_room(pp, 1))
        pp->out[pp->out_len++] = c;
}

static void pp_emit_str(preproc_t *pp, const char *s, uint32_t len)
{
    if (!pp_room(pp, len)) return;
    memcpy(pp->out + pp->out_len, s, len);
    pp->out_len += len;
}

static void pp_eatnl(preproc_t *pp)
{
    uint32_t n = pp_nlsz(pp);
    if (n == 0) return;
    for (uint32_t i = 0; i < n; i++)
        pp_advance(pp);
    for (uint32_t i = 0; i <= pp->nspl; i++)
        pp_emit_char(pp, '\n');
}

/* ---- String pool ---- */

static uint32_t pool_add(preproc_t *pp, const char *s, uint32_t len)
{
    if (pp->pool_len + len > PP_POOL_SIZE) {
        pp_error(pp, BC_E040);
        return 0;
    }
    uint32_t off = pp->pool_len;
    memcpy(pp->pool + off, s, len);
    pp->pool_len += len;
    return off;
}

/* ---- Macro table ---- */

static pp_macro_t *pp_find_macro(preproc_t *pp, const char *name, uint32_t len)
{
    for (uint32_t i = 0; i < pp->num_macros; i++) {
        pp_macro_t *m = &pp->macros[i];
        if (m->name_len == len &&
            memcmp(pp->pool + m->name_off, name, len) == 0)
            return m;
    }
    return NULL;
}

static int pp_define_macro(preproc_t *pp, const char *name, uint32_t name_len,
                           const char *body, uint32_t body_len,
                           int num_params,
                           const char params[][BC_MAX_IDENT],
                           const uint32_t param_lens[],
                           int varia)
{
    /* Check for redefinition — replace body if exists */
    pp_macro_t *existing = pp_find_macro(pp, name, name_len);
    if (existing) {
        /* A header seen twice redefines every macro in it identically. The
         * pool has no free list, so storing that again is pure waste. */
        if (existing->body_len == body_len &&
            existing->num_params == (int16_t)num_params &&
            existing->varia == (uint8_t)(varia != 0) &&
            memcmp(pp->pool + existing->body_off, body, body_len) == 0)
            return BC_OK;
        existing->body_off = pool_add(pp, body, body_len);
        existing->body_len = (uint16_t)body_len;
        existing->num_params = (int16_t)num_params;
        existing->varia = (uint8_t)(varia != 0);
        for (int i = 0; i < num_params && i < PP_MAX_PARAMS; i++) {
            existing->param_off[i] = pool_add(pp, params[i], param_lens[i]);
            existing->param_len[i] = (uint8_t)param_lens[i];
        }
        return BC_OK;
    }

    if (pp->num_macros >= PP_MAX_MACROS) {
        pp_error(pp, BC_E041, PP_MAX_MACROS);
        return BC_ERR_PREPROC;
    }

    pp_macro_t *m = &pp->macros[pp->num_macros++];
    m->name_off = pool_add(pp, name, name_len);
    m->name_len = (uint16_t)name_len;
    m->body_off = pool_add(pp, body, body_len);
    m->body_len = (uint16_t)body_len;
    m->num_params = (int16_t)num_params;
    m->varia = (uint8_t)(varia != 0);

    for (int i = 0; i < num_params && i < PP_MAX_PARAMS; i++) {
        m->param_off[i] = pool_add(pp, params[i], param_lens[i]);
        m->param_len[i] = (uint8_t)param_lens[i];
    }
    return BC_OK;
}

static void pp_undef_macro(preproc_t *pp, const char *name, uint32_t len)
{
    for (uint32_t i = 0; i < pp->num_macros; i++) {
        pp_macro_t *m = &pp->macros[i];
        if (m->name_len == len &&
            memcmp(pp->pool + m->name_off, name, len) == 0) {
            /* Swap with last and shrink */
            pp->macros[i] = pp->macros[pp->num_macros - 1];
            pp->num_macros--;
            return;
        }
    }
}

/* ---- Conditional stack ---- */

static int pp_is_active(const preproc_t *pp)
{
    if (pp->cond_depth == 0) return 1;
    return pp->cond_stack[pp->cond_depth - 1].active;
}

static void pp_push_cond(preproc_t *pp, int active)
{
    if (pp->cond_depth >= PP_MAX_COND_DEPTH) {
        pp_error(pp, BC_E042, PP_MAX_COND_DEPTH);
        return;
    }
    int parent = pp_is_active(pp);
    pp_cond_t *c = &pp->cond_stack[pp->cond_depth++];
    c->parent_active = parent;
    c->active = parent && active;
    c->seen_true = c->active;
}

static void pp_flip_else(preproc_t *pp)
{
    if (pp->cond_depth == 0) {
        pp_error(pp, BC_E043);
        return;
    }
    pp_cond_t *c = &pp->cond_stack[pp->cond_depth - 1];
    c->active = c->parent_active && !c->seen_true;
    c->seen_true = 1;
}

static void pp_flip_elif(preproc_t *pp, int expr_val)
{
    if (pp->cond_depth == 0) {
        pp_error(pp, BC_E044);
        return;
    }
    pp_cond_t *c = &pp->cond_stack[pp->cond_depth - 1];
    c->active = c->parent_active && !c->seen_true && expr_val;
    if (c->active) c->seen_true = 1;
}

static void pp_pop_cond(preproc_t *pp)
{
    if (pp->cond_depth == 0) {
        pp_error(pp, BC_E045);
        return;
    }
    pp->cond_depth--;
}

/* ---- Read an identifier from current position ---- */

static uint32_t pp_read_ident(const preproc_t *pp, char *buf, uint32_t max)
{
    uint32_t len = 0;
    uint32_t p = pp->pos;
    while (p < pp->src_len && pp_is_ident_char(pp->src[p]) && len + 1 < max) {
        buf[len++] = pp->src[p++];
    }
    buf[len] = '\0';
    return len;
}

/* ---- Collect a logical line (handling backslash-newline continuations) ---- */

static uint32_t pp_collect_line(preproc_t *pp, char *buf, uint32_t max)
{
    uint32_t len = 0;
    while (!pp_at_end(pp) && pp_nlsz(pp) == 0) {
        uint32_t n = pp_splc(pp);
        if (n > 0) {
            for (uint32_t i = 0; i < n; i++)
                pp_advance(pp);
            pp->nspl++;
            continue;
        }
        if (len + 1 < max)
            buf[len++] = pp_cur(pp);
        pp_advance(pp);
    }
    buf[len] = '\0';
    return len;
}

/* ---- Expression evaluator for #if ---- */

/*
 * Evaluate a preprocessor expression (integer arithmetic + defined()).
 * Uses shunting-yard for binary operators, handles unary prefix.
 * All arithmetic is done in int64_t.
 */

#define EXPR_MAX_TOKENS 256
#define EXPR_MAX_STACK  128

typedef enum {
    ETOK_NUM, ETOK_LPAREN, ETOK_RPAREN,
    ETOK_PLUS, ETOK_MINUS, ETOK_STAR, ETOK_SLASH, ETOK_PERCENT,
    ETOK_LT, ETOK_GT, ETOK_LE, ETOK_GE, ETOK_EQ, ETOK_NE,
    ETOK_LAND, ETOK_LOR, ETOK_NOT,
    ETOK_AND, ETOK_OR, ETOK_XOR, ETOK_TILDE,
    ETOK_SHL, ETOK_SHR,
    ETOK_UNARY_MINUS, ETOK_UNARY_PLUS,
    ETOK_END
} etok_type_t;

typedef struct {
    int     type;
    int64_t val;
} etok_t;

static int etok_prec(int type)
{
    switch (type) {
    case ETOK_LOR:          return 1;
    case ETOK_LAND:         return 2;
    case ETOK_OR:           return 3;
    case ETOK_XOR:          return 4;
    case ETOK_AND:          return 5;
    case ETOK_EQ: case ETOK_NE: return 6;
    case ETOK_LT: case ETOK_GT: case ETOK_LE: case ETOK_GE: return 7;
    case ETOK_SHL: case ETOK_SHR: return 8;
    case ETOK_PLUS: case ETOK_MINUS: return 9;
    case ETOK_STAR: case ETOK_SLASH: case ETOK_PERCENT: return 10;
    case ETOK_NOT: case ETOK_TILDE:
    case ETOK_UNARY_MINUS: case ETOK_UNARY_PLUS: return 11;
    default: return 0;
    }
}

static int etok_is_unary(int type)
{
    return type == ETOK_NOT || type == ETOK_TILDE ||
           type == ETOK_UNARY_MINUS || type == ETOK_UNARY_PLUS;
}

static int etok_is_right_assoc(int type)
{
    return etok_is_unary(type);
}

/* Tokenize a #if expression string. Handles defined(X) and defined X. */
static int expr_tokenize(preproc_t *pp, const char *s, uint32_t len,
                         etok_t *toks, int max_toks)
{
    int n = 0;
    uint32_t i = 0;
    int prev_was_operand = 0; /* for unary vs binary disambiguation */

    while (i < len && n < max_toks - 1) {
        /* skip whitespace */
        while (i < len && (s[i] == ' ' || s[i] == '\t')) i++;
        if (i >= len) break;

        /* number */
        if (s[i] >= '0' && s[i] <= '9') {
            int64_t val = 0;
            if (s[i] == '0' && i + 1 < len && (s[i+1] == 'x' || s[i+1] == 'X')) {
                i += 2;
                while (i < len && isxdigit((unsigned char)s[i])) {
                    char c = s[i++];
                    int d = (c >= '0' && c <= '9') ? c - '0' :
                            (c >= 'a' && c <= 'f') ? c - 'a' + 10 :
                            c - 'A' + 10;
                    val = val * 16 + d;
                }
            } else if (s[i] == '0' && i + 1 < len && s[i+1] >= '0' && s[i+1] <= '7') {
                while (i < len && s[i] >= '0' && s[i] <= '7')
                    val = val * 8 + (s[i++] - '0');
            } else {
                while (i < len && s[i] >= '0' && s[i] <= '9')
                    val = val * 10 + (s[i++] - '0');
            }
            /* skip integer suffix (U, L, UL, LL, ULL) */
            while (i < len && (s[i] == 'u' || s[i] == 'U' ||
                               s[i] == 'l' || s[i] == 'L'))
                i++;
            toks[n].type = ETOK_NUM;
            toks[n].val = val;
            n++;
            prev_was_operand = 1;
            continue;
        }

        /* character literal 'x' */
        if (s[i] == '\'') {
            i++;
            int64_t val = 0;
            if (i < len && s[i] == '\\') {
                i++;
                if (i < len) {
                    switch (s[i]) {
                    case 'n':  val = '\n'; break;
                    case 't':  val = '\t'; break;
                    case '\\': val = '\\'; break;
                    case '\'': val = '\''; break;
                    case '0':  val = '\0'; break;
                    default:   val = s[i]; break;
                    }
                    i++;
                }
            } else if (i < len) {
                val = (unsigned char)s[i++];
            }
            if (i < len && s[i] == '\'') i++;
            toks[n].type = ETOK_NUM;
            toks[n].val = val;
            n++;
            prev_was_operand = 1;
            continue;
        }

        /* identifier or "defined" */
        if (pp_is_ident_start(s[i])) {
            uint32_t start = i;
            while (i < len && pp_is_ident_char(s[i])) i++;
            uint32_t ilen = i - start;

            if (ilen == 7 && memcmp(s + start, "defined", 7) == 0) {
                /* defined(X) or defined X */
                while (i < len && (s[i] == ' ' || s[i] == '\t')) i++;
                int has_paren = 0;
                if (i < len && s[i] == '(') { has_paren = 1; i++; }
                while (i < len && (s[i] == ' ' || s[i] == '\t')) i++;
                uint32_t ns = i;
                while (i < len && pp_is_ident_char(s[i])) i++;
                int64_t val = pp_find_macro(pp, s + ns, i - ns) ? 1 : 0;
                while (i < len && (s[i] == ' ' || s[i] == '\t')) i++;
                if (has_paren && i < len && s[i] == ')') i++;
                toks[n].type = ETOK_NUM;
                toks[n].val = val;
                n++;
                prev_was_operand = 1;
            } else {
                /* Other identifier — try macro expansion, else 0 */
                pp_macro_t *m = pp_find_macro(pp, s + start, ilen);
                int64_t val = 0;
                if (m && m->num_params < 0 && m->body_len > 0) {
                    /* Simple object macro — try to parse as number */
                    const char *body = pp->pool + m->body_off;
                    char *end = NULL;
                    val = strtoll(body, &end, 0);
                    if (end == body) val = 0;
                }
                toks[n].type = ETOK_NUM;
                toks[n].val = val;
                n++;
                prev_was_operand = 1;
            }
            continue;
        }

        /* operators */
        char c = s[i];
        if (c == '(') {
            toks[n].type = ETOK_LPAREN; toks[n].val = 0; n++; i++;
            prev_was_operand = 0;
        } else if (c == ')') {
            toks[n].type = ETOK_RPAREN; toks[n].val = 0; n++; i++;
            prev_was_operand = 1;
        } else if (c == '+') {
            toks[n].type = prev_was_operand ? ETOK_PLUS : ETOK_UNARY_PLUS;
            toks[n].val = 0; n++; i++;
            prev_was_operand = 0;
        } else if (c == '-') {
            toks[n].type = prev_was_operand ? ETOK_MINUS : ETOK_UNARY_MINUS;
            toks[n].val = 0; n++; i++;
            prev_was_operand = 0;
        } else if (c == '*') {
            toks[n].type = ETOK_STAR; toks[n].val = 0; n++; i++;
            prev_was_operand = 0;
        } else if (c == '/') {
            toks[n].type = ETOK_SLASH; toks[n].val = 0; n++; i++;
            prev_was_operand = 0;
        } else if (c == '%') {
            toks[n].type = ETOK_PERCENT; toks[n].val = 0; n++; i++;
            prev_was_operand = 0;
        } else if (c == '&' && i + 1 < len && s[i+1] == '&') {
            toks[n].type = ETOK_LAND; toks[n].val = 0; n++; i += 2;
            prev_was_operand = 0;
        } else if (c == '|' && i + 1 < len && s[i+1] == '|') {
            toks[n].type = ETOK_LOR; toks[n].val = 0; n++; i += 2;
            prev_was_operand = 0;
        } else if (c == '<' && i + 1 < len && s[i+1] == '<') {
            toks[n].type = ETOK_SHL; toks[n].val = 0; n++; i += 2;
            prev_was_operand = 0;
        } else if (c == '>' && i + 1 < len && s[i+1] == '>') {
            toks[n].type = ETOK_SHR; toks[n].val = 0; n++; i += 2;
            prev_was_operand = 0;
        } else if (c == '<' && i + 1 < len && s[i+1] == '=') {
            toks[n].type = ETOK_LE; toks[n].val = 0; n++; i += 2;
            prev_was_operand = 0;
        } else if (c == '>' && i + 1 < len && s[i+1] == '=') {
            toks[n].type = ETOK_GE; toks[n].val = 0; n++; i += 2;
            prev_was_operand = 0;
        } else if (c == '=' && i + 1 < len && s[i+1] == '=') {
            toks[n].type = ETOK_EQ; toks[n].val = 0; n++; i += 2;
            prev_was_operand = 0;
        } else if (c == '!' && i + 1 < len && s[i+1] == '=') {
            toks[n].type = ETOK_NE; toks[n].val = 0; n++; i += 2;
            prev_was_operand = 0;
        } else if (c == '<') {
            toks[n].type = ETOK_LT; toks[n].val = 0; n++; i++;
            prev_was_operand = 0;
        } else if (c == '>') {
            toks[n].type = ETOK_GT; toks[n].val = 0; n++; i++;
            prev_was_operand = 0;
        } else if (c == '!') {
            toks[n].type = ETOK_NOT; toks[n].val = 0; n++; i++;
            prev_was_operand = 0;
        } else if (c == '~') {
            toks[n].type = ETOK_TILDE; toks[n].val = 0; n++; i++;
            prev_was_operand = 0;
        } else if (c == '&') {
            toks[n].type = ETOK_AND; toks[n].val = 0; n++; i++;
            prev_was_operand = 0;
        } else if (c == '|') {
            toks[n].type = ETOK_OR; toks[n].val = 0; n++; i++;
            prev_was_operand = 0;
        } else if (c == '^') {
            toks[n].type = ETOK_XOR; toks[n].val = 0; n++; i++;
            prev_was_operand = 0;
        } else {
            /* Unknown char — skip */
            i++;
        }
    }

    toks[n].type = ETOK_END;
    toks[n].val = 0;
    return n;
}

/* Evaluate tokenized expression using shunting-yard + postfix eval */
static int64_t expr_evaluate(etok_t *toks, int ntoks)
{
    /* Phase 1: shunting-yard → postfix */
    etok_t output[EXPR_MAX_TOKENS];
    int    op_stack[EXPR_MAX_STACK]; /* indices into toks */
    int    nout = 0, nops = 0;

    for (int i = 0; i < ntoks; i++) {
        etok_t *t = &toks[i];

        if (t->type == ETOK_NUM) {
            if (nout < EXPR_MAX_TOKENS) output[nout++] = *t;
        } else if (t->type == ETOK_LPAREN) {
            if (nops < EXPR_MAX_STACK) op_stack[nops++] = i;
        } else if (t->type == ETOK_RPAREN) {
            while (nops > 0 && toks[op_stack[nops-1]].type != ETOK_LPAREN) {
                if (nout < EXPR_MAX_TOKENS) output[nout++] = toks[op_stack[--nops]];
            }
            if (nops > 0) nops--; /* pop LPAREN */
        } else {
            /* Operator */
            int prec = etok_prec(t->type);
            int ra = etok_is_right_assoc(t->type);
            while (nops > 0 && toks[op_stack[nops-1]].type != ETOK_LPAREN) {
                int top_prec = etok_prec(toks[op_stack[nops-1]].type);
                if (ra ? (top_prec > prec) : (top_prec >= prec)) {
                    if (nout < EXPR_MAX_TOKENS) output[nout++] = toks[op_stack[--nops]];
                } else break;
            }
            if (nops < EXPR_MAX_STACK) op_stack[nops++] = i;
        }
    }
    while (nops > 0) {
        if (toks[op_stack[nops-1]].type != ETOK_LPAREN)
            if (nout < EXPR_MAX_TOKENS) output[nout++] = toks[op_stack[--nops]];
            else nops--;
        else nops--;
    }

    /* Phase 2: evaluate postfix */
    int64_t val_stack[EXPR_MAX_STACK];
    int nvs = 0;

    for (int i = 0; i < nout; i++) {
        etok_t *t = &output[i];
        if (t->type == ETOK_NUM) {
            if (nvs < EXPR_MAX_STACK) val_stack[nvs++] = t->val;
        } else if (etok_is_unary(t->type)) {
            if (nvs < 1) continue;
            int64_t a = val_stack[--nvs];
            int64_t r = 0;
            switch (t->type) {
            case ETOK_NOT:         r = !a; break;
            case ETOK_TILDE:       r = ~a; break;
            case ETOK_UNARY_MINUS: r = -a; break;
            case ETOK_UNARY_PLUS:  r = a; break;
            default: break;
            }
            val_stack[nvs++] = r;
        } else {
            /* Binary */
            if (nvs < 2) continue;
            int64_t b = val_stack[--nvs];
            int64_t a = val_stack[--nvs];
            int64_t r = 0;
            switch (t->type) {
            case ETOK_PLUS:    r = a + b; break;
            case ETOK_MINUS:   r = a - b; break;
            case ETOK_STAR:    r = a * b; break;
            case ETOK_SLASH:   r = b ? a / b : 0; break;
            case ETOK_PERCENT: r = b ? a % b : 0; break;
            case ETOK_LT:     r = a < b; break;
            case ETOK_GT:     r = a > b; break;
            case ETOK_LE:     r = a <= b; break;
            case ETOK_GE:     r = a >= b; break;
            case ETOK_EQ:     r = a == b; break;
            case ETOK_NE:     r = a != b; break;
            case ETOK_LAND:   r = a && b; break;
            case ETOK_LOR:    r = a || b; break;
            case ETOK_AND:    r = a & b; break;
            case ETOK_OR:     r = a | b; break;
            case ETOK_XOR:    r = a ^ b; break;
            case ETOK_SHL:    r = a << b; break;
            case ETOK_SHR:    r = a >> b; break;
            default: break;
            }
            val_stack[nvs++] = r;
        }
    }

    return nvs > 0 ? val_stack[0] : 0;
}

static int64_t pp_eval_expr(preproc_t *pp, const char *expr, uint32_t len)
{
    etok_t toks[EXPR_MAX_TOKENS];
    int ntoks = expr_tokenize(pp, expr, len, toks, EXPR_MAX_TOKENS);
    return expr_evaluate(toks, ntoks);
}

/* ---- Macro expansion ---- */

/*
 * Expand macros in text [in, in+in_len).
 * Write expanded result to [out, out+out_max).
 * blocked/num_blocked: macros currently being expanded (anti-recursion).
 * depth: recursion depth counter.
 * Returns bytes written to out.
 */
/*
 * Per-expansion buffer size.
 * Kept small so the recursive expand function doesn't blow the stack.
 * Bodies are rarely > 2KB; arguments are spans and cost nothing here.
 */
#define PP_TMP_BUF  8192

/* __VA_ARGS__ is the last parameter and it swallows every argument from its
 * own position onwards, so a parameter reference covers a range, not one arg. */
static int vlast(const pp_macro_t *m, int pi)
{
    return m->varia && pi >= 0 && pi == m->num_params - 1;
}

static int varg(const pp_macro_t *m, int pi, int nargs)
{
    if (pi < 0) return pi;
    int end = vlast(m, pi) ? nargs : pi + 1;
    return end > nargs ? nargs : end;
}

static uint32_t pp_expand_text(preproc_t *pp,
                               const char *in, uint32_t in_len,
                               char *out, uint32_t out_max,
                               const char **blocked, int num_blocked,
                               int depth)
{
    uint32_t olen = 0;
    uint32_t i = 0;

    if (depth > PP_MAX_EXPAND_DEPTH) {
        uint32_t n = in_len < out_max ? in_len : out_max;
        memcpy(out, in, n);
        return n;
    }

    /* Resume a block comment left open by an earlier line: copy through to its
     * close before any macro name in the prose can be substituted. */
    if (pp->in_bcmt) {
        while (i < in_len && !(in[i] == '*' && i + 1 < in_len && in[i+1] == '/')) {
            if (olen < out_max) out[olen++] = in[i++]; else i++;
        }
        if (i < in_len) {
            if (olen < out_max) out[olen++] = in[i++]; else i++;
            if (olen < out_max) out[olen++] = in[i++]; else i++;
            pp->in_bcmt = 0;
        }
    }

    while (i < in_len) {
        /* Skip C-style block comments */
        if (in[i] == '/' && i + 1 < in_len && in[i+1] == '*') {
            if (olen < out_max) out[olen++] = in[i++]; else i++;
            if (olen < out_max) out[olen++] = in[i++]; else i++;
            while (i < in_len && !(in[i] == '*' && i + 1 < in_len && in[i+1] == '/')) {
                if (olen < out_max) out[olen++] = in[i++]; else i++;
            }
            if (i >= in_len) {
                /* Unterminated on this line; the next one resumes inside it. */
                pp->in_bcmt = 1;
                continue;
            }
            if (olen < out_max) out[olen++] = in[i++]; else i++;
            if (i < in_len) { if (olen < out_max) out[olen++] = in[i++]; else i++; }
            continue;
        }

        /* Skip C++ line comments */
        if (in[i] == '/' && i + 1 < in_len && in[i+1] == '/') {
            while (i < in_len && in[i] != '\n') {
                if (olen < out_max) out[olen++] = in[i++]; else i++;
            }
            continue;
        }

        /* Skip string literals */
        if (in[i] == '"' || in[i] == '\'') {
            char q = in[i];
            if (olen < out_max) out[olen++] = in[i++];
            else i++;
            while (i < in_len && in[i] != q) {
                if (in[i] == '\\' && i + 1 < in_len) {
                    if (olen < out_max) out[olen++] = in[i++];
                    else i++;
                }
                if (olen < out_max) out[olen++] = in[i++];
                else i++;
            }
            if (i < in_len) {
                if (olen < out_max) out[olen++] = in[i++];
                else i++;
            }
            continue;
        }

        /* Check for identifier (potential macro) */
        if (pp_is_ident_start(in[i])) {
            uint32_t id_start = i;
            while (i < in_len && pp_is_ident_char(in[i])) i++;
            uint32_t id_len = i - id_start;

            /* Check __LINE__ */
            if (id_len == 8 && memcmp(in + id_start, "__LINE__", 8) == 0) {
                char num[32];
                int nlen = snprintf(num, sizeof(num), "%u", pp->line);
                for (int k = 0; k < nlen && olen < out_max; k++)
                    out[olen++] = num[k];
                continue;
            }

            /* Check __FILE__ */
            if (id_len == 8 && memcmp(in + id_start, "__FILE__", 8) == 0) {
                if (olen < out_max) out[olen++] = '"';
                for (uint32_t k = 0; pp->filename[k] && olen < out_max; k++)
                    out[olen++] = pp->filename[k];
                if (olen < out_max) out[olen++] = '"';
                continue;
            }

            /* Is it blocked? (anti-recursion) */
            int is_blocked = 0;
            for (int b = 0; b < num_blocked; b++) {
                if (strlen(blocked[b]) == id_len &&
                    memcmp(blocked[b], in + id_start, id_len) == 0) {
                    is_blocked = 1;
                    break;
                }
            }

            pp_macro_t *m = is_blocked ? NULL : pp_find_macro(pp, in + id_start, id_len);

            if (!m) {
                for (uint32_t k = id_start; k < id_start + id_len && olen < out_max; k++)
                    out[olen++] = in[k];
                continue;
            }

            if (m->num_params < 0) {
                /* Object-like macro: expand body with rescanning */
                const char *new_blocked[PP_MAX_EXPAND_DEPTH];
                int nb = 0;
                for (int b = 0; b < num_blocked && nb < PP_MAX_EXPAND_DEPTH - 1; b++)
                    new_blocked[nb++] = blocked[b];
                char name_tmp[BC_MAX_IDENT];
                memcpy(name_tmp, pp->pool + m->name_off, m->name_len);
                name_tmp[m->name_len] = '\0';
                new_blocked[nb++] = name_tmp;

                char tmp[PP_TMP_BUF];
                uint32_t tlen = pp_expand_text(pp,
                    pp->pool + m->body_off, m->body_len,
                    tmp, PP_TMP_BUF, new_blocked, nb, depth + 1);
                for (uint32_t k = 0; k < tlen && olen < out_max; k++)
                    out[olen++] = tmp[k];
                continue;
            }

            /* Function-like macro: need '(' after optional whitespace */
            uint32_t save_i = i;
            while (i < in_len && (in[i] == ' ' || in[i] == '\t')) i++;

            if (i >= in_len || in[i] != '(') {
                i = save_i;
                for (uint32_t k = id_start; k < id_start + id_len && olen < out_max; k++)
                    out[olen++] = in[k];
                continue;
            }
            i++; /* skip '(' */

            uint32_t aoff[PP_MAX_ARGS];
            uint32_t arg_lens[PP_MAX_ARGS];
            int nargs = 0;
            int paren_depth = 0;
            int closed = 0;
            int full = 0;
            uint32_t arg_beg = i;

            while (i < in_len) {
                if (in[i] == '(') {
                    paren_depth++;
                    i++;
                } else if (in[i] == ')' && paren_depth > 0) {
                    paren_depth--;
                    i++;
                } else if (in[i] == ')' && paren_depth == 0) {
                    if (nargs >= PP_MAX_ARGS) full = 1;
                    else {
                        aoff[nargs] = arg_beg;
                        arg_lens[nargs] = i - arg_beg;
                        nargs++;
                    }
                    i++;
                    closed = 1;
                    break;
                } else if (in[i] == ',' && paren_depth == 0) {
                    if (nargs >= PP_MAX_ARGS) full = 1;
                    else {
                        aoff[nargs] = arg_beg;
                        arg_lens[nargs] = i - arg_beg;
                        nargs++;
                    }
                    i++;
                    arg_beg = i;
                } else {
                    i++;
                }
            }

            if (!closed || full) {
                char mn[BC_MAX_IDENT + 1];
                uint32_t ml = m->name_len < BC_MAX_IDENT ? m->name_len
                                                         : BC_MAX_IDENT;
                memcpy(mn, pp->pool + m->name_off, ml);
                mn[ml] = 0;
                pp_error(pp, BC_E174, mn, PP_MAX_ARGS);
                continue;
            }

            /* Handle zero-argument case: macro() with 0 params */
            if (nargs == 1 && arg_lens[0] == 0 && m->num_params == 0)
                nargs = 0;

            /* Substitute parameters in body */
            char subst[PP_TMP_BUF];
            uint32_t slen = 0;
            const char *body = pp->pool + m->body_off;
            uint32_t blen = m->body_len;

            for (uint32_t j = 0; j < blen; ) {
                /* Check for ## (token paste) */
                if (j + 1 < blen && body[j] == '#' && body[j+1] == '#') {
                    while (slen > 0 && (subst[slen-1] == ' ' || subst[slen-1] == '\t'))
                        slen--;
                    j += 2;
                    while (j < blen && (body[j] == ' ' || body[j] == '\t'))
                        j++;
                    continue;
                }

                /* Check for # (stringify) */
                if (body[j] == '#' && j + 1 < blen && pp_is_ident_start(body[j+1])) {
                    j++;
                    uint32_t ps = j;
                    while (j < blen && pp_is_ident_char(body[j])) j++;
                    uint32_t plen = j - ps;
                    int pi = -1;
                    for (int p = 0; p < m->num_params; p++) {
                        if (m->param_len[p] == plen &&
                            memcmp(pp->pool + m->param_off[p], body + ps, plen) == 0) {
                            pi = p;
                            break;
                        }
                    }
                    int a0 = pi, a1 = varg(m, pi, nargs);
                    if (pi >= 0 && a0 < a1) {
                        if (slen < PP_TMP_BUF) subst[slen++] = '"';
                        for (int p = a0; p < a1; p++) {
                            if (p > a0 && slen < PP_TMP_BUF) subst[slen++] = ',';
                            for (uint32_t k = 0; k < arg_lens[p] && slen < PP_TMP_BUF; k++) {
                                char ac = in[aoff[p] + k];
                                if (ac == '"' || ac == '\\')
                                    if (slen < PP_TMP_BUF) subst[slen++] = '\\';
                                if (slen < PP_TMP_BUF) subst[slen++] = ac;
                            }
                        }
                        if (slen < PP_TMP_BUF) subst[slen++] = '"';
                    }
                    continue;
                }

                /* Check for parameter name in body */
                if (pp_is_ident_start(body[j])) {
                    uint32_t ps = j;
                    while (j < blen && pp_is_ident_char(body[j])) j++;
                    uint32_t plen = j - ps;
                    int pi = -1;
                    for (int p = 0; p < m->num_params; p++) {
                        if (m->param_len[p] == plen &&
                            memcmp(pp->pool + m->param_off[p], body + ps, plen) == 0) {
                            pi = p;
                            break;
                        }
                    }
                    int a0 = pi, a1 = varg(m, pi, nargs);
                    if (pi >= 0 && (pi < nargs || vlast(m, pi))) {
                        for (int p = a0; p < a1; p++) {
                            if (p > a0 && slen < PP_TMP_BUF) subst[slen++] = ',';
                            for (uint32_t k = 0; k < arg_lens[p] && slen < PP_TMP_BUF; k++)
                                subst[slen++] = in[aoff[p] + k];
                        }
                    } else {
                        for (uint32_t k = ps; k < ps + plen && slen < PP_TMP_BUF; k++)
                            subst[slen++] = body[k];
                    }
                    continue;
                }

                if (slen < PP_TMP_BUF) subst[slen++] = body[j];
                j++;
            }

            /* Rescan the substituted result */
            const char *new_blocked[PP_MAX_EXPAND_DEPTH];
            int nb = 0;
            for (int b = 0; b < num_blocked && nb < PP_MAX_EXPAND_DEPTH - 1; b++)
                new_blocked[nb++] = blocked[b];
            char name_tmp2[BC_MAX_IDENT];
            memcpy(name_tmp2, pp->pool + m->name_off, m->name_len);
            name_tmp2[m->name_len] = '\0';
            new_blocked[nb++] = name_tmp2;

            char tmp2[PP_TMP_BUF];
            uint32_t tlen2 = pp_expand_text(pp, subst, slen,
                tmp2, PP_TMP_BUF, new_blocked, nb, depth + 1);
            for (uint32_t k = 0; k < tlen2 && olen < out_max; k++)
                out[olen++] = tmp2[k];
            continue;
        }

        /* Plain character — copy */
        if (olen < out_max) out[olen++] = in[i];
        i++;
    }

    return olen;
}

/* Expand macros in a line, writing result to pp output buffer.
 * Uses pp->exp_buf as workspace (safe — this is the top-level entry point). */
static void pp_expand_and_emit(preproc_t *pp, const char *line, uint32_t len)
{
    const char *no_blocked = NULL;
    uint32_t elen = pp_expand_text(pp, line, len,
                                   pp->exp_buf, PP_EXPAND_BUF,
                                   &no_blocked, 0, 0);
    pp_emit_str(pp, pp->exp_buf, elen);
}

/* ---- Directive handlers ---- */

static void pp_dir_define(preproc_t *pp)
{
    pp_skip_hspace(pp);

    /* Read macro name */
    if (pp_at_end(pp) || !pp_is_ident_start(pp_cur(pp))) {
        pp_error(pp, BC_E046);
        pp_skip_to_eol(pp);
        return;
    }

    char name[BC_MAX_IDENT];
    uint32_t name_len = pp_read_ident(pp, name, BC_MAX_IDENT);
    pp->pos += name_len;

    /* Check for function-like macro: '(' immediately after name (no space) */
    char params[PP_MAX_PARAMS][BC_MAX_IDENT];
    uint32_t param_lens[PP_MAX_PARAMS];
    int num_params = -1; /* -1 = object-like */
    int varia = 0;

    if (!pp_at_end(pp) && pp_cur(pp) == '(') {
        num_params = 0;
        pp_advance(pp); /* skip '(' */
        pp_skip_hspace(pp);

        if (!pp_at_end(pp) && pp_cur(pp) != ')') {
            while (!pp_at_end(pp) && num_params < PP_MAX_PARAMS) {
                pp_skip_hspace(pp);
                if (pp_cur(pp) == '.' && pp_peek(pp, 1) == '.' &&
                    pp_peek(pp, 2) == '.') {
                    pp_advance(pp); pp_advance(pp); pp_advance(pp);
                    memcpy(params[num_params], "__VA_ARGS__", 11);
                    param_lens[num_params] = 11;
                    num_params++;
                    varia = 1;
                    break;
                }
                if (!pp_is_ident_start(pp_cur(pp))) break;
                uint32_t plen = pp_read_ident(pp, params[num_params], BC_MAX_IDENT);
                param_lens[num_params] = plen;
                pp->pos += plen;
                num_params++;
                pp_skip_hspace(pp);
                if (!pp_at_end(pp) && pp_cur(pp) == ',')
                    pp_advance(pp);
                else
                    break;
            }
            pp_skip_hspace(pp);
            if (num_params >= PP_MAX_PARAMS && !pp_at_end(pp)
                && pp_cur(pp) != ')') {
                pp_error(pp, BC_E300, name, PP_MAX_PARAMS);
                pp_skip_to_eol(pp);
                return;
            }
        }
        /* Anything left before the ')' is a parameter list this preprocessor
         * does not understand. Skipping it beats leaving it in the body,
         * which is how '#define F(...)' used to expand to '...)'. */
        KA_GUARD(g, PP_LINE_BUF);
        while (!pp_at_end(pp) && pp_cur(pp) != ')' && pp_cur(pp) != '\n' && g--)
            pp_advance(pp);
        if (!pp_at_end(pp) && pp_cur(pp) == ')')
            pp_advance(pp);
    }

    pp_skip_hspace(pp);

    /* Collect body (rest of logical line) */
    char body[PP_EXPAND_BUF];
    uint32_t body_len = pp_collect_line(pp, body, PP_EXPAND_BUF);
    body_len = stripc(body, body_len);

    /* Trim trailing whitespace from body */
    while (body_len > 0 && (body[body_len-1] == ' ' || body[body_len-1] == '\t'))
        body_len--;

    pp_define_macro(pp, name, name_len, body, body_len,
                    num_params, (const char (*)[BC_MAX_IDENT])params, param_lens,
                    varia);
}

static void pp_dir_undef(preproc_t *pp)
{
    pp_skip_hspace(pp);
    char name[BC_MAX_IDENT];
    uint32_t name_len = pp_read_ident(pp, name, BC_MAX_IDENT);
    pp->pos += name_len;
    pp_undef_macro(pp, name, name_len);
    pp_skip_to_eol(pp);
}

static void pp_dir_ifdef(preproc_t *pp, int negate)
{
    pp_skip_hspace(pp);
    char name[BC_MAX_IDENT];
    uint32_t name_len = pp_read_ident(pp, name, BC_MAX_IDENT);
    pp->pos += name_len;
    int defined = pp_find_macro(pp, name, name_len) != NULL;
    if (negate) defined = !defined;
    pp_push_cond(pp, defined);
    pp_skip_to_eol(pp);
}

static void pp_dir_if(preproc_t *pp)
{
    pp_skip_hspace(pp);
    char expr[PP_LINE_BUF];
    uint32_t elen = pp_collect_line(pp, expr, PP_LINE_BUF);
    int64_t val = pp_eval_expr(pp, expr, elen);
    pp_push_cond(pp, val != 0);
}

static void pp_dir_elif(preproc_t *pp)
{
    pp_skip_hspace(pp);
    char expr[PP_LINE_BUF];
    uint32_t elen = pp_collect_line(pp, expr, PP_LINE_BUF);
    int64_t val = pp_eval_expr(pp, expr, elen);
    pp_flip_elif(pp, val != 0);
}

static void pp_dir_else(preproc_t *pp)
{
    pp_flip_else(pp);
    pp_skip_to_eol(pp);
}

static void pp_dir_endif(preproc_t *pp)
{
    pp_pop_cond(pp);
    pp_skip_to_eol(pp);
}

static void pp_dir_error(preproc_t *pp)
{
    pp_skip_hspace(pp);
    char msg[256];
    uint32_t mlen = pp_collect_line(pp, msg, sizeof(msg));
    (void)mlen;
    pp_error(pp, BC_E047, msg);
}

/* ---- #pragma once ---- */

static int once_has(const preproc_t *pp, const char *path)
{
    for (uint32_t i = 0; i < pp->n_once; i++)
        if (strcmp(pp->once[i], path) == 0) return 1;
    return 0;
}

/* A full table only means the header gets read again, which is what every
 * header did before this existed. Slower, never wrong. */
static void once_add(preproc_t *pp, const char *path)
{
    if (once_has(pp, path)) return;
    if (pp->n_once >= PP_MAX_ONCE) return;
    snprintf(pp->once[pp->n_once++], BC_MAX_PATH, "%s", path);
}

static void pp_dir_pragma(preproc_t *pp)
{
    pp_skip_hspace(pp);
    if (pp->pos + 4 <= pp->src_len &&
        memcmp(pp->src + pp->pos, "once", 4) == 0 &&
        !pp_is_ident_char(pp_peek(pp, 4)))
        once_add(pp, pp->filename);
    pp_skip_to_eol(pp);
}

/* ---- #include handling ---- */

static int pp_read_file(const char *path, char **buf, uint32_t *out_len)
{
    FILE *fp = fopen(path, "rb");
    if (!fp) return -1;
    fseek(fp, 0, SEEK_END);
    long sz = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    if (sz < 0 || (uint32_t)sz >= BC_MAX_SOURCE) {
        fclose(fp);
        return -1;
    }
    *buf = (char *)malloc((uint32_t)sz + 1);
    if (!*buf) { fclose(fp); return -1; }
    *out_len = (uint32_t)fread(*buf, 1, (size_t)sz, fp);
    (*buf)[*out_len] = '\0';
    fclose(fp);
    return 0;
}

/* Extract directory from a file path */
static void path_dirname(const char *path, char *dir, uint32_t max)
{
    uint32_t len = (uint32_t)strlen(path);
    uint32_t last_sep = 0;
    int found = 0;
    for (uint32_t i = 0; i < len; i++) {
        if (path[i] == '/' || path[i] == '\\') {
            last_sep = i;
            found = 1;
        }
    }
    if (found) {
        uint32_t n = last_sep + 1 < max ? last_sep + 1 : max - 1;
        memcpy(dir, path, n);
        dir[n] = '\0';
    } else {
        dir[0] = '.';
        dir[1] = '/';
        dir[2] = '\0';
    }
}

static int pp_try_include(const char *dir, const char *name,
                          char *fullpath, uint32_t fmax)
{
    uint32_t dlen = (uint32_t)strlen(dir);
    uint32_t nlen = (uint32_t)strlen(name);
    if (dlen + nlen >= fmax) return 0;
    memcpy(fullpath, dir, dlen);
    memcpy(fullpath + dlen, name, nlen);
    fullpath[dlen + nlen] = '\0';
    FILE *fp = fopen(fullpath, "rb");
    if (fp) { fclose(fp); return 1; }
    return 0;
}

static void pp_dir_include(preproc_t *pp)
{
    pp_skip_hspace(pp);

    char include_name[BC_MAX_PATH];
    uint32_t nlen = 0;
    int is_angle = 0;

    if (pp_cur(pp) == '"') {
        pp_advance(pp);
        while (!pp_at_end(pp) && pp_cur(pp) != '"' && pp_cur(pp) != '\n') {
            if (nlen + 1 < BC_MAX_PATH) include_name[nlen++] = pp_cur(pp);
            pp_advance(pp);
        }
        if (pp_cur(pp) == '"') pp_advance(pp);
    } else if (pp_cur(pp) == '<') {
        is_angle = 1;
        pp_advance(pp);
        while (!pp_at_end(pp) && pp_cur(pp) != '>' && pp_cur(pp) != '\n') {
            if (nlen + 1 < BC_MAX_PATH) include_name[nlen++] = pp_cur(pp);
            pp_advance(pp);
        }
        if (pp_cur(pp) == '>') pp_advance(pp);
    } else {
        pp_error(pp, BC_E048);
        pp_skip_to_eol(pp);
        return;
    }
    include_name[nlen] = '\0';
    pp_skip_to_eol(pp);

    /* Search for the file */
    char fullpath[BC_MAX_PATH];
    int found = 0;

    if (!is_angle) {
        /* For "file", search relative to current file first */
        char dir[BC_MAX_PATH];
        path_dirname(pp->filename, dir, BC_MAX_PATH);
        found = pp_try_include(dir, include_name, fullpath, BC_MAX_PATH);
    }

    if (!found) {
        for (uint32_t i = 0; i < pp->num_include_paths && !found; i++) {
            char dir[BC_MAX_PATH];
            snprintf(dir, BC_MAX_PATH, "%s/", pp->include_paths[i]);
            found = pp_try_include(dir, include_name, fullpath, BC_MAX_PATH);
        }
    }

    if (!found) {
        /* Not found — skip with warning (allows test files with system includes) */
        /* Don't treat as error — common for <stdio.h>, <cuda_runtime.h> etc. */
        return;
    }

    if (once_has(pp, fullpath))
        return;

    /* Guard against include depth overflow */
    if (pp->file_depth >= PP_MAX_FILE_DEPTH) {
        pp_error(pp, BC_E049, PP_MAX_FILE_DEPTH);
        return;
    }

    /* Read the included file */
    char *inc_buf = NULL;
    uint32_t inc_len = 0;
    if (pp_read_file(fullpath, &inc_buf, &inc_len) != 0) {
        pp_error(pp, BC_E050, fullpath);
        return;
    }

    /* Push current file state */
    pp_file_entry_t *fe = &pp->file_stack[pp->file_depth++];
    fe->saved_src = pp->src;
    fe->saved_src_len = pp->src_len;
    fe->saved_pos = pp->pos;
    fe->saved_line = pp->line;
    snprintf(fe->saved_filename, BC_MAX_PATH, "%s", pp->filename);
    fe->buf = inc_buf;

    /* Switch to included file */
    pp->src = inc_buf;
    pp->src_len = inc_len;
    pp->pos = 0;
    pp->line = 1;
    snprintf(pp->filename, BC_MAX_PATH, "%s", fullpath);
}

/* Pop include file stack. Returns 1 if popped, 0 if at top level. */
static int pp_pop_file(preproc_t *pp)
{
    if (pp->file_depth == 0) return 0;
    pp->file_depth--;
    pp_file_entry_t *fe = &pp->file_stack[pp->file_depth];
    free(fe->buf);
    fe->buf = NULL;
    pp->src = fe->saved_src;
    pp->src_len = fe->saved_src_len;
    pp->pos = fe->saved_pos;
    pp->line = fe->saved_line;
    snprintf(pp->filename, BC_MAX_PATH, "%s", fe->saved_filename);
    return 1;
}

/* ---- Multi-line macro invocations ---- */

/* Expansion runs a line at a time, so an argument list that opens on one line
 * and closes on a later one has to be joined first. True when the line ends
 * inside the argument list of a function-like macro. */
static int ocall(preproc_t *pp, const char *s, uint32_t n)
{
    int depth = 0;
    int bcmt = pp->in_bcmt;
    uint32_t i = 0;

    while (i < n) {
        if (bcmt) {
            if (s[i] == '*' && i + 1 < n && s[i+1] == '/') { bcmt = 0; i += 2; }
            else i++;
            continue;
        }
        if (s[i] == '/' && i + 1 < n && s[i+1] == '*') { bcmt = 1; i += 2; continue; }
        if (s[i] == '/' && i + 1 < n && s[i+1] == '/') break;
        if (s[i] == '"' || s[i] == '\'') {
            char q = s[i++];
            while (i < n && s[i] != q) {
                if (s[i] == '\\' && i + 1 < n) i++;
                i++;
            }
            i++;
            continue;
        }
        if (depth > 0) {
            if (s[i] == '(') depth++;
            else if (s[i] == ')') depth--;
            i++;
            continue;
        }
        if (pp_is_ident_start(s[i])) {
            uint32_t st = i;
            while (i < n && pp_is_ident_char(s[i])) i++;
            uint32_t j = i;
            while (j < n && (s[j] == ' ' || s[j] == '\t')) j++;
            if (j < n && s[j] == '(') {
                pp_macro_t *m = pp_find_macro(pp, s + st, i - st);
                if (m && m->num_params >= 0) { depth = 1; i = j + 1; }
            }
            continue;
        }
        i++;
    }
    return depth > 0;
}


/* ---- Main processing loop ---- */

static void pp_process_directive(preproc_t *pp)
{
    pp_advance(pp); /* skip '#' */
    pp_skip_hspace(pp);

    /* Read directive name */
    char dir[64];
    uint32_t dlen = 0;
    while (!pp_at_end(pp) && pp_is_ident_char(pp_cur(pp)) && dlen + 1 < sizeof(dir)) {
        dir[dlen++] = pp_cur(pp);
        pp_advance(pp);
    }
    dir[dlen] = '\0';

    /* Directives that must be handled even in inactive blocks */
    if (strcmp(dir, "ifdef") == 0) {
        if (pp_is_active(pp))
            pp_dir_ifdef(pp, 0);
        else {
            pp_push_cond(pp, 0); /* nested inactive */
            pp_skip_to_eol(pp);
        }
        return;
    }
    if (strcmp(dir, "ifndef") == 0) {
        if (pp_is_active(pp))
            pp_dir_ifdef(pp, 1);
        else {
            pp_push_cond(pp, 0);
            pp_skip_to_eol(pp);
        }
        return;
    }
    if (strcmp(dir, "if") == 0) {
        if (pp_is_active(pp))
            pp_dir_if(pp);
        else {
            pp_push_cond(pp, 0);
            pp_skip_to_eol(pp);
        }
        return;
    }
    if (strcmp(dir, "elif") == 0) {
        if (pp->cond_depth > 0 && pp->cond_stack[pp->cond_depth - 1].parent_active)
            pp_dir_elif(pp);
        else
            pp_skip_to_eol(pp);
        return;
    }
    if (strcmp(dir, "else") == 0) {
        pp_dir_else(pp);
        return;
    }
    if (strcmp(dir, "endif") == 0) {
        pp_dir_endif(pp);
        return;
    }

    /* All other directives: only process if active */
    if (!pp_is_active(pp)) {
        pp_skip_to_eol(pp);
        return;
    }

    if (strcmp(dir, "define") == 0)       pp_dir_define(pp);
    else if (strcmp(dir, "undef") == 0)   pp_dir_undef(pp);
    else if (strcmp(dir, "include") == 0) pp_dir_include(pp);
    else if (strcmp(dir, "error") == 0)   pp_dir_error(pp);
    else if (strcmp(dir, "pragma") == 0)  pp_dir_pragma(pp);
    else if (strcmp(dir, "warning") == 0) pp_skip_to_eol(pp);
    else if (strcmp(dir, "line") == 0)    pp_skip_to_eol(pp);
    else if (dlen == 0) { /* null directive: just '#' on a line */ }
    else {
        pp_error(pp, BC_E051, dir);
        pp_skip_to_eol(pp);
    }
}

int pp_process(preproc_t *pp)
{
    while ((!pp_at_end(pp) || pp->file_depth > 0) && !pp->ovflw) {
        /* If we've reached the end of an included file, pop the file stack */
        if (pp_at_end(pp)) {
            if (!pp_pop_file(pp))
                break;
            continue;
        }

        /* Check for start of line — skip horizontal whitespace, look for '#' */
        uint32_t line_start = pp->pos;
        pp->nspl = 0;
        pp_skip_hspace(pp);

        if (pp_at_end(pp)) continue;

        if (pp_cur(pp) == '#' && pp_peek(pp, 1) != '#') {
            pp_process_directive(pp);
            pp_eatnl(pp);
            continue;
        }

        /* Non-directive line */
        if (!pp_is_active(pp)) {
            /* In inactive conditional block — skip line, emit newline */
            pp_skip_to_eol(pp);
            pp_eatnl(pp);
            continue;
        }

        /* Active non-directive line: collect, expand macros, emit */
        /* Reset position to line_start to include leading whitespace */
        pp->pos = line_start;
        char line[PP_LINE_BUF];
        uint32_t llen = pp_collect_line(pp, line, PP_LINE_BUF);

        uint32_t joins = 0;
        KA_GUARD(g, PP_MAX_JOIN);
        while (g-- && llen + 2 < PP_LINE_BUF && ocall(pp, line, llen)) {
            uint32_t nl = pp_cur(pp) == '\n' ? 1u
                        : (pp_cur(pp) == '\r' && pp_peek(pp, 1) == '\n') ? 2u : 0u;
            if (nl == 0) break;
            llen = stripc(line, llen);
            for (uint32_t k = 0; k < nl; k++)
                pp_advance(pp);
            line[llen++] = ' ';
            llen += pp_collect_line(pp, line + llen, PP_LINE_BUF - llen);
            joins++;
        }
        if (joins) llen = stripc(line, llen);

        pp_expand_and_emit(pp, line, llen);
        while (joins--) pp_emit_char(pp, '\n');

        pp_eatnl(pp);
    }

    /* Abandoning the run leaves the include stack loaded, and every entry
     * owns a malloc'd buffer. */
    KA_GUARD(g, PP_MAX_FILE_DEPTH + 1);
    while (g-- && pp_pop_file(pp)) { }

    /* Check for unterminated conditionals */
    if (pp->cond_depth > 0 && !pp->ovflw) {
        pp_error(pp, BC_E052, pp->cond_depth);
    }

    pp->out[pp->out_len < pp->out_max ? pp->out_len : pp->out_max - 1] = '\0';

    return pp->num_errors > 0 ? BC_ERR_PREPROC : BC_OK;
}

/* ---- Public API ---- */

static const struct { const char *n; const char *v; } pdlim[] = {
    {"CHAR_BIT",      "8"},
    {"SCHAR_MIN",     "(-128)"},
    {"SCHAR_MAX",     "127"},
    {"UCHAR_MAX",     "255"},
    {"CHAR_MIN",      "(-128)"},
    {"CHAR_MAX",      "127"},
    {"SHRT_MIN",      "(-32768)"},
    {"SHRT_MAX",      "32767"},
    {"USHRT_MAX",     "65535"},
    {"INT_MIN",       "(-2147483647-1)"},
    {"INT_MAX",       "2147483647"},
    {"UINT_MAX",      "4294967295U"},
    {"LONG_MIN",      "(-9223372036854775807LL-1)"},
    {"LONG_MAX",      "9223372036854775807LL"},
    {"ULONG_MAX",     "18446744073709551615ULL"},
    {"LLONG_MIN",     "(-9223372036854775807LL-1)"},
    {"LLONG_MAX",     "9223372036854775807LL"},
    {"ULLONG_MAX",    "18446744073709551615ULL"},
    {"INT8_MIN",      "(-128)"},
    {"INT8_MAX",      "127"},
    {"UINT8_MAX",     "255"},
    {"INT16_MIN",     "(-32768)"},
    {"INT16_MAX",     "32767"},
    {"UINT16_MAX",    "65535"},
    {"INT32_MIN",     "(-2147483647-1)"},
    {"INT32_MAX",     "2147483647"},
    {"UINT32_MAX",    "4294967295U"},
    {"INT64_MIN",     "(-9223372036854775807LL-1)"},
    {"INT64_MAX",     "9223372036854775807LL"},
    {"UINT64_MAX",    "18446744073709551615ULL"},
    {"INTPTR_MIN",    "(-9223372036854775807LL-1)"},
    {"INTPTR_MAX",    "9223372036854775807LL"},
    {"UINTPTR_MAX",   "18446744073709551615ULL"},
    {"PTRDIFF_MIN",   "(-9223372036854775807LL-1)"},
    {"PTRDIFF_MAX",   "9223372036854775807LL"},
    {"SIZE_MAX",      "18446744073709551615ULL"},
    {"FLT_RADIX",     "2"},
    {"FLT_MANT_DIG",  "24"},
    {"FLT_DIG",       "6"},
    {"FLT_MIN_EXP",   "(-125)"},
    {"FLT_MAX_EXP",   "128"},
    {"FLT_MIN_10_EXP","(-37)"},
    {"FLT_MAX_10_EXP","38"},
    {"FLT_MIN",       "1.17549435e-38F"},
    {"FLT_MAX",       "3.40282347e+38F"},
    {"FLT_EPSILON",   "1.19209290e-7F"},
    {"FLT_TRUE_MIN",  "1.40129846e-45F"},
    {"DBL_MANT_DIG",  "53"},
    {"DBL_DIG",       "15"},
    {"DBL_MIN_EXP",   "(-1021)"},
    {"DBL_MAX_EXP",   "1024"},
    {"DBL_MIN_10_EXP","(-307)"},
    {"DBL_MAX_10_EXP","308"},
    {"DBL_MIN",       "2.2250738585072014e-308"},
    {"DBL_MAX",       "1.7976931348623157e+308"},
    {"DBL_EPSILON",   "2.2204460492503131e-16"},
    {"INFINITY",      "(1.0f/0.0f)"},
    {"NAN",           "(0.0f/0.0f)"},
    {"HUGE_VALF",     "(1.0f/0.0f)"},
    {"HUGE_VAL",      "(1.0/0.0)"},
    {"M_PI",          "3.14159265358979323846"},
    {"M_PI_2",        "1.57079632679489661923"},
    {"M_PI_4",        "0.78539816339744830962"},
    {"M_E",           "2.7182818284590452354"},
    {"M_LN2",         "0.69314718055994530942"},
    {"M_LN10",        "2.30258509299404568402"},
    {"M_LOG2E",       "1.4426950408889634074"},
    {"M_SQRT2",       "1.41421356237309504880"}
};

static const struct { const char *n; const char *v; } pdhnd[] = {
    {"cudaStream_t",               "void *"},
    {"cudaEvent_t",                "void *"},
    {"cudaGraph_t",                "void *"},
    {"cudaGraphExec_t",            "void *"},
    {"cudaGraphNode_t",            "void *"},
    {"cudaGraphicsResource_t",     "void *"},
    {"cudaArray_t",                "void *"},
    {"cudaArray_const_t",          "const void *"},
    {"cudaMipmappedArray_t",       "void *"},
    {"cudaMipmappedArray_const_t", "const void *"},
    {"cudaMemPool_t",              "void *"},
    {"cudaFunction_t",             "void *"},
    {"cudaExternalMemory_t",       "void *"},
    {"cudaExternalSemaphore_t",    "void *"},
    {"cudaUserObject_t",           "void *"},
    {"cudaHostFn_t",               "void *"},
    {"CUstream_st",                "void"},
    {"CUevent_st",                 "void"}
};

static const struct { const char *sfx; const char *len; } pdiw[] = {
    {"8",       "hh"}, {"16",      "h"}, {"32",      ""},  {"64",      "ll"},
    {"LEAST8",  "hh"}, {"LEAST16", "h"}, {"LEAST32", ""},  {"LEAST64", "ll"},
    {"FAST8",   "hh"}, {"FAST16",  "h"}, {"FAST32",  ""},  {"FAST64",  "ll"},
    {"MAX",     "ll"}, {"PTR",     "ll"}
};

static void pdfmt(preproc_t *pp)
{
    static const char cnv[] = "diouxX";

    for (size_t i = 0; i < sizeof pdiw / sizeof pdiw[0]; i++)
        for (size_t j = 0; j < sizeof cnv - 1; j++) {
            char nm[32], vl[12];

            if (snprintf(vl, sizeof vl, "\"%s%c\"", pdiw[i].len, cnv[j]) < 0)
                continue;
            if (snprintf(nm, sizeof nm, "PRI%c%s", cnv[j], pdiw[i].sfx) > 0)
                (void)pp_define(pp, nm, vl);
            if (cnv[j] == 'X') continue;
            if (snprintf(nm, sizeof nm, "SCN%c%s", cnv[j], pdiw[i].sfx) > 0)
                (void)pp_define(pp, nm, vl);
        }
}

void pp_init(preproc_t *pp, const char *src, uint32_t len,
             char *out_buf, uint32_t out_max, const char *filename)
{
    memset(pp, 0, sizeof(*pp));
    pp->src = src;
    pp->src_len = len;
    pp->pos = 0;
    pp->line = 1;
    pp->out = out_buf;
    pp->out_len = 0;
    pp->out_max = out_max;
    if (filename)
        snprintf(pp->filename, BC_MAX_PATH, "%s", filename);

    pp_define(pp, "__BARRACUDA__", "1");
    pp_define(pp, "__CUDA_ARCH__", "1100");
    pp_define(pp, "__CUDACC__", "1");
    pp_define(pp, "__cplusplus", "201703L");

    for (size_t i = 0; i < sizeof pdlim / sizeof pdlim[0]; i++)
        (void)pp_define(pp, pdlim[i].n, pdlim[i].v);
    for (size_t i = 0; i < sizeof pdhnd / sizeof pdhnd[0]; i++)
        (void)pp_define(pp, pdhnd[i].n, pdhnd[i].v);
    pdfmt(pp);
}

int pp_add_include_path(preproc_t *pp, const char *path)
{
    if (pp->num_include_paths >= PP_MAX_INCLUDE_PATHS) return BC_ERR_OVERFLOW;
    snprintf(pp->include_paths[pp->num_include_paths], BC_MAX_PATH, "%s", path);
    pp->num_include_paths++;
    return BC_OK;
}

int pp_define(preproc_t *pp, const char *name, const char *value)
{
    uint32_t nlen = (uint32_t)strlen(name);
    uint32_t vlen = value ? (uint32_t)strlen(value) : 0;
    return pp_define_macro(pp, name, nlen, value ? value : "", vlen,
                           -1, NULL, NULL, 0);
}
