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

static int pp_is_ident_start(char c)
{
    return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || c == '_';
}

static int pp_is_ident_char(char c)
{
    return pp_is_ident_start(c) || (c >= '0' && c <= '9');
}

/* ---- Output helpers ---- */

static void pp_emit_char(preproc_t *pp, char c)
{
    if (pp->out_len < pp->out_max)
        pp->out[pp->out_len++] = c;
}

static void pp_emit_str(preproc_t *pp, const char *s, uint32_t len)
{
    for (uint32_t i = 0; i < len && pp->out_len < pp->out_max; i++)
        pp->out[pp->out_len++] = s[i];
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
                           const uint32_t param_lens[])
{
    /* Check for redefinition — replace body if exists */
    pp_macro_t *existing = pp_find_macro(pp, name, name_len);
    if (existing) {
        existing->body_off = pool_add(pp, body, body_len);
        existing->body_len = (uint16_t)body_len;
        existing->num_params = (int16_t)num_params;
        for (int i = 0; i < num_params && i < PP_MAX_PARAMS; i++) {
            existing->param_off[i] = (uint16_t)pool_add(pp, params[i], param_lens[i]);
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

    for (int i = 0; i < num_params && i < PP_MAX_PARAMS; i++) {
        m->param_off[i] = (uint16_t)pool_add(pp, params[i], param_lens[i]);
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
    while (!pp_at_end(pp) && pp_cur(pp) != '\n') {
        if (pp_cur(pp) == '\\' && pp_peek(pp, 1) == '\n') {
            pp_advance(pp); /* skip backslash */
            pp_advance(pp); /* skip newline (counts line) */
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
 * Per-arg and per-expansion buffer sizes.
 * Kept small so the recursive expand function doesn't blow the stack.
 * Macro args are rarely > 256 chars; bodies rarely > 2KB.
 */
#define PP_ARG_BUF  1024
#define PP_TMP_BUF  8192

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

            /* Collect arguments (1KB per arg — plenty for real macros) */
            char args[PP_MAX_PARAMS][PP_ARG_BUF];
            uint32_t arg_lens[PP_MAX_PARAMS];
            int nargs = 0;
            int paren_depth = 0;
            uint32_t arg_len = 0;

            while (i < in_len && nargs < PP_MAX_PARAMS) {
                if (in[i] == '(') {
                    paren_depth++;
                    if (arg_len < PP_ARG_BUF - 1) args[nargs][arg_len++] = in[i];
                    i++;
                } else if (in[i] == ')' && paren_depth > 0) {
                    paren_depth--;
                    if (arg_len < PP_ARG_BUF - 1) args[nargs][arg_len++] = in[i];
                    i++;
                } else if (in[i] == ')' && paren_depth == 0) {
                    args[nargs][arg_len] = '\0';
                    arg_lens[nargs] = arg_len;
                    nargs++;
                    i++;
                    break;
                } else if (in[i] == ',' && paren_depth == 0) {
                    args[nargs][arg_len] = '\0';
                    arg_lens[nargs] = arg_len;
                    nargs++;
                    arg_len = 0;
                    i++;
                } else {
                    if (arg_len < PP_ARG_BUF - 1) args[nargs][arg_len++] = in[i];
                    i++;
                }
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
                    if (pi >= 0 && pi < nargs) {
                        if (slen < PP_TMP_BUF) subst[slen++] = '"';
                        for (uint32_t k = 0; k < arg_lens[pi] && slen < PP_TMP_BUF; k++) {
                            if (args[pi][k] == '"' || args[pi][k] == '\\')
                                if (slen < PP_TMP_BUF) subst[slen++] = '\\';
                            if (slen < PP_TMP_BUF) subst[slen++] = args[pi][k];
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
                    if (pi >= 0 && pi < nargs) {
                        for (uint32_t k = 0; k < arg_lens[pi] && slen < PP_TMP_BUF; k++)
                            subst[slen++] = args[pi][k];
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

    if (!pp_at_end(pp) && pp_cur(pp) == '(') {
        num_params = 0;
        pp_advance(pp); /* skip '(' */
        pp_skip_hspace(pp);

        if (!pp_at_end(pp) && pp_cur(pp) != ')') {
            while (!pp_at_end(pp) && num_params < PP_MAX_PARAMS) {
                pp_skip_hspace(pp);
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
        }
        pp_skip_hspace(pp);
        if (!pp_at_end(pp) && pp_cur(pp) == ')')
            pp_advance(pp);
    }

    pp_skip_hspace(pp);

    /* Collect body (rest of logical line) */
    char body[PP_EXPAND_BUF];
    uint32_t body_len = pp_collect_line(pp, body, PP_EXPAND_BUF);

    /* Trim trailing whitespace from body */
    while (body_len > 0 && (body[body_len-1] == ' ' || body[body_len-1] == '\t'))
        body_len--;

    pp_define_macro(pp, name, name_len, body, body_len,
                    num_params, (const char (*)[BC_MAX_IDENT])params, param_lens);
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

static void pp_dir_pragma(preproc_t *pp)
{
    /* Just skip the line — could emit as a comment later */
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
    while (!pp_at_end(pp) || pp->file_depth > 0) {
        /* If we've reached the end of an included file, pop the file stack */
        if (pp_at_end(pp)) {
            if (!pp_pop_file(pp))
                break;
            continue;
        }

        /* Check for start of line — skip horizontal whitespace, look for '#' */
        uint32_t line_start = pp->pos;
        pp_skip_hspace(pp);

        if (pp_at_end(pp)) continue;

        if (pp_cur(pp) == '#' && pp_peek(pp, 1) != '#') {
            pp_process_directive(pp);
            /* Eat the trailing newline after the directive */
            if (!pp_at_end(pp) && pp_cur(pp) == '\n') {
                pp_emit_char(pp, '\n'); /* preserve line count */
                pp_advance(pp);
            }
            continue;
        }

        /* Non-directive line */
        if (!pp_is_active(pp)) {
            /* In inactive conditional block — skip line, emit newline */
            pp_skip_to_eol(pp);
            if (!pp_at_end(pp) && pp_cur(pp) == '\n') {
                pp_emit_char(pp, '\n');
                pp_advance(pp);
            }
            continue;
        }

        /* Active non-directive line: collect, expand macros, emit */
        /* Reset position to line_start to include leading whitespace */
        pp->pos = line_start;
        char line[PP_LINE_BUF];
        uint32_t llen = pp_collect_line(pp, line, PP_LINE_BUF);
        pp_expand_and_emit(pp, line, llen);

        if (!pp_at_end(pp) && pp_cur(pp) == '\n') {
            pp_emit_char(pp, '\n');
            pp_advance(pp);
        }
    }

    /* Check for unterminated conditionals */
    if (pp->cond_depth > 0) {
        pp_error(pp, BC_E052, pp->cond_depth);
    }

    /* Null-terminate output */
    if (pp->out_len < pp->out_max)
        pp->out[pp->out_len] = '\0';

    return pp->num_errors > 0 ? BC_ERR_PREPROC : BC_OK;
}

/* ---- Public API ---- */

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

    /* Predefined macros */
    pp_define(pp, "__BARRACUDA__", "1");
    pp_define(pp, "__CUDA_ARCH__", "1100");
    pp_define(pp, "__CUDACC__", "1");
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
                           -1, NULL, NULL);
}
