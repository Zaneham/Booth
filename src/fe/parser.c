#include "parser.h"
#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>

/* Every node gets a name. Even the ones that probably shouldn't exist. */

static const char *ast_names[] = {
    [AST_NONE]          = "NONE",
    [AST_INT_LIT]       = "int_lit",
    [AST_FLOAT_LIT]     = "float_lit",
    [AST_STRING_LIT]    = "string_lit",
    [AST_CHAR_LIT]      = "char_lit",
    [AST_BOOL_LIT]      = "bool_lit",
    [AST_NULL_LIT]      = "null_lit",
    [AST_IDENT]         = "ident",
    [AST_BINARY]        = "binary",
    [AST_UNARY_PREFIX]  = "unary_pre",
    [AST_UNARY_POSTFIX] = "unary_post",
    [AST_TERNARY]       = "ternary",
    [AST_CALL]          = "call",
    [AST_LAUNCH]        = "launch",
    [AST_MEMBER]        = "member",
    [AST_SUBSCRIPT]     = "subscript",
    [AST_CAST]          = "cast",
    [AST_SIZEOF]        = "sizeof",
    [AST_PAREN]         = "paren",
    [AST_INIT_LIST]     = "init_list",
    [AST_SCOPE_RES]     = "scope",
    [AST_TEMPLATE_ARGS] = "template_args",
    [AST_EXPR_STMT]     = "expr_stmt",
    [AST_BLOCK]         = "block",
    [AST_IF]            = "if",
    [AST_FOR]           = "for",
    [AST_WHILE]         = "while",
    [AST_DO_WHILE]      = "do_while",
    [AST_SWITCH]        = "switch",
    [AST_CASE]          = "case",
    [AST_DEFAULT]       = "default",
    [AST_RETURN]        = "return",
    [AST_BREAK]         = "break",
    [AST_CONTINUE]      = "continue",
    [AST_GOTO]          = "goto",
    [AST_LABEL]         = "label",
    [AST_TYPE_SPEC]     = "type",
    [AST_DECLARATOR]    = "declarator",
    [AST_PARAM]         = "param",
    [AST_VAR_DECL]      = "var_decl",
    [AST_FUNC_DEF]      = "func_def",
    [AST_FUNC_DECL]     = "func_decl",
    [AST_STRUCT_DEF]    = "struct_def",
    [AST_ENUM_DEF]      = "enum_def",
    [AST_ENUMERATOR]    = "enumerator",
    [AST_TYPEDEF]       = "typedef",
    [AST_USING]         = "using",
    [AST_NAMESPACE]     = "namespace",
    [AST_TEMPLATE_DECL] = "template_decl",
    [AST_TEMPLATE_PARAM]= "template_param",
    [AST_PP_DIRECTIVE]  = "pp_directive",
    [AST_TRANSLATION_UNIT] = "translation_unit",
};

const char *ast_type_name(int type)
{
    if (type >= 0 && type < AST_TYPE_COUNT)
        return ast_names[type] ? ast_names[type] : "???";
    return "???";
}

void parser_init(parser_t *P, const token_t *tokens, uint32_t num_tokens,
                 const char *src, ast_node_t *nodes, uint32_t max_nodes)
{
    memset(P, 0, sizeof(*P));
    P->tokens = tokens;
    P->num_tokens = num_tokens;
    P->pos = 0;
    P->src = src;
    P->nodes = nodes;
    P->num_nodes = 1; /* index 0: the void. Nothing lives here. */
    P->max_nodes = max_nodes;
    memset(&nodes[0], 0, sizeof(ast_node_t));
}

static const token_t *cur(const parser_t *P)
{
    return &P->tokens[P->pos < P->num_tokens ? P->pos : P->num_tokens - 1];
}

static int cur_type(const parser_t *P)
{
    return cur(P)->type;
}

static int peek_type(const parser_t *P, int ahead)
{
    uint32_t p = P->pos + (uint32_t)ahead;
    if (p >= P->num_tokens) p = P->num_tokens - 1;
    return P->tokens[p].type;
}

static void advance(parser_t *P)
{
    if (P->pos < P->num_tokens) P->pos++;
}

static void parse_error(parser_t *P, bc_eid_t eid, ...)
{
    if (P->num_errors < BC_MAX_ERRORS) {
        bc_error_t *e = &P->errors[P->num_errors++];
        const token_t *t = cur(P);
        e->loc.line = t->line;
        e->loc.col = t->col;
        e->loc.offset = t->offset;
        e->code = BC_ERR_PARSE;
        e->eid  = (uint16_t)eid;
        va_list ap;
        va_start(ap, eid);
        vsnprintf(e->msg, sizeof(e->msg), bc_efmt(eid), ap);
        va_end(ap);
    }
}

static int match(parser_t *P, int type)
{
    if (cur_type(P) == type) { advance(P); return 1; }
    return 0;
}

static int expect(parser_t *P, int type)
{
    if (cur_type(P) == type) { advance(P); return 1; }
    parse_error(P, BC_E020,
                token_type_name(type), token_type_name(cur_type(P)));
    return 0;
}

static uint32_t alloc_node(parser_t *P, int type)
{
    if (P->num_nodes >= P->max_nodes) {
        parse_error(P, BC_E021);
        return 0;
    }
    uint32_t idx = P->num_nodes++;
    ast_node_t *n = &P->nodes[idx];
    memset(n, 0, sizeof(*n));
    n->type = (uint16_t)type;
    const token_t *t = cur(P);
    n->line = t->line;
    n->col = t->col;
    return idx;
}

static void add_child(parser_t *P, uint32_t parent, uint32_t child)
{
    if (!parent || !child) return;
    ast_node_t *p = &P->nodes[parent];
    if (!p->first_child) {
        p->first_child = child;
    } else {
        uint32_t c = p->first_child;
        uint32_t guard = P->max_nodes;
        while (P->nodes[c].next_sibling && --guard)
            c = P->nodes[c].next_sibling;
        P->nodes[c].next_sibling = child;
    }
}

/* ---- Type name registry for cast disambiguation ---- */

static void reg_tname(parser_t *P, uint32_t off, uint16_t len)
{
    if (P->num_tnames >= 128) return;
    P->tnames[P->num_tnames].off = off;
    P->tnames[P->num_tnames].len = len;
    P->num_tnames++;
}

static int is_reg_type(const parser_t *P, uint32_t off, uint16_t len)
{
    for (int i = 0; i < P->num_tnames; i++) {
        if (P->tnames[i].len == len &&
            memcmp(P->src + P->tnames[i].off, P->src + off, len) == 0)
            return 1;
    }
    return 0;
}

static int is_type_keyword(int type)
{
    switch (type) {
    case TOK_VOID: case TOK_CHAR: case TOK_SHORT: case TOK_INT:
    case TOK_LONG: case TOK_FLOAT: case TOK_DOUBLE: case TOK_BOOL:
    case TOK_SIGNED: case TOK_UNSIGNED: case TOK_AUTO:
    case TOK_STRUCT: case TOK_UNION: case TOK_ENUM: case TOK_CLASS:
    case TOK_CONST: case TOK_VOLATILE: case TOK_CONSTEXPR:
        return 1;
    default:
        return 0;
    }
}

static int is_cuda_qualifier(int type)
{
    switch (type) {
    case TOK_CU_GLOBAL: case TOK_CU_DEVICE: case TOK_CU_HOST:
    case TOK_CU_SHARED: case TOK_CU_CONSTANT: case TOK_CU_MANAGED:
    case TOK_CU_GRID_CONSTANT: case TOK_CU_RESTRICT:
    case TOK_CU_FORCEINLINE: case TOK_CU_NOINLINE:
    case TOK_CU_LAUNCH_BOUNDS:
        return 1;
    default:
        return 0;
    }
}

static int is_storage_class(int type)
{
    switch (type) {
    case TOK_STATIC: case TOK_EXTERN: case TOK_REGISTER:
    case TOK_INLINE: case TOK_TYPEDEF:
        return 1;
    default:
        return 0;
    }
}

static uint16_t cuda_flag_for(int tok_type)
{
    switch (tok_type) {
    case TOK_CU_GLOBAL:        return CUDA_GLOBAL;
    case TOK_CU_DEVICE:        return CUDA_DEVICE;
    case TOK_CU_HOST:          return CUDA_HOST;
    case TOK_CU_SHARED:        return CUDA_SHARED;
    case TOK_CU_CONSTANT:      return CUDA_CONSTANT;
    case TOK_CU_MANAGED:       return CUDA_MANAGED;
    case TOK_CU_GRID_CONSTANT: return CUDA_GRID_CONSTANT;
    case TOK_CU_RESTRICT:      return CUDA_RESTRICT;
    case TOK_CU_FORCEINLINE:   return CUDA_FORCEINLINE;
    case TOK_CU_NOINLINE:      return CUDA_NOINLINE;
    case TOK_CU_LAUNCH_BOUNDS: return CUDA_LAUNCH_BOUNDS;
    default: return 0;
    }
}

static uint16_t storage_flag_for(int tok_type)
{
    switch (tok_type) {
    case TOK_CONST:     return QUAL_CONST;
    case TOK_VOLATILE:  return QUAL_VOLATILE;
    case TOK_STATIC:    return QUAL_STATIC;
    case TOK_EXTERN:    return QUAL_EXTERN;
    case TOK_INLINE:    return QUAL_INLINE;
    case TOK_REGISTER:  return QUAL_REGISTER;
    case TOK_CONSTEXPR: return QUAL_CONSTEXPR;
    case TOK_TYPEDEF:   return QUAL_TYPEDEF;
    default: return 0;
    }
}

/* ---- Error Recovery ---- */

/* Skip to next semicolon, consuming it. Stops at } or EOF without consuming.
   The parser equivalent of "have you tried turning it off and on again." */
static void sync_past_semi(parser_t *P)
{
    int guard = 10000;
    while (cur_type(P) != TOK_EOF && --guard) {
        int t = cur_type(P);
        if (t == TOK_SEMI) { advance(P); return; }
        if (t == TOK_RBRACE) return;
        advance(P);
    }
}

/* Skip to next declaration/statement start or block boundary.
   Like fast-forwarding through the boring bits of a meeting. */
static void sync_to_next_decl(parser_t *P)
{
    int guard = 10000;
    while (cur_type(P) != TOK_EOF && --guard) {
        int t = cur_type(P);
        if (t == TOK_SEMI) { advance(P); return; }
        if (t == TOK_RBRACE) { advance(P); return; }
        if (t == TOK_LBRACE) return;
        if (is_storage_class(t) || is_cuda_qualifier(t) ||
            t == TOK_IF || t == TOK_FOR || t == TOK_WHILE || t == TOK_DO ||
            t == TOK_RETURN || t == TOK_STRUCT || t == TOK_CLASS ||
            t == TOK_ENUM || t == TOK_TYPEDEF)
            return;
        advance(P);
    }
}

static uint32_t parse_expr(parser_t *P, int min_prec);
static uint32_t parse_stmt(parser_t *P);
static uint32_t parse_type_spec(parser_t *P, uint16_t *quals, uint16_t *cuda);
static uint32_t parse_decl_or_stmt(parser_t *P);

static int64_t parse_int_text(const char *s, int len)
{
    char buf[64];
    int n = len > 63 ? 63 : len;
    memcpy(buf, s, (size_t)n);
    buf[n] = '\0';
    while (n > 0 && (buf[n-1]=='u'||buf[n-1]=='U'||
                     buf[n-1]=='l'||buf[n-1]=='L'))
        buf[--n] = '\0';
    return strtoll(buf, NULL, 0);
}

/* Pratt, 1973. Top Down Operator Precedence. The prophecy fulfilled. */

static int prefix_bp(int type)
{
    switch (type) {
    case TOK_BANG: case TOK_TILDE:
    case TOK_PLUS: case TOK_MINUS:
    case TOK_STAR: case TOK_AMP:
    case TOK_INC: case TOK_DEC:
        return 150;
    default:
        return -1;
    }
}

static int infix_bp(int type, int *right_bp)
{
    switch (type) {
    case TOK_COMMA:      *right_bp = 11; return 10;
    case TOK_ASSIGN:
    case TOK_PLUS_EQ: case TOK_MINUS_EQ:
    case TOK_STAR_EQ: case TOK_SLASH_EQ: case TOK_PERCENT_EQ:
    case TOK_SHL_EQ: case TOK_SHR_EQ:
    case TOK_AMP_EQ: case TOK_CARET_EQ: case TOK_PIPE_EQ:
                         *right_bp = 20; return 21;
    case TOK_QUESTION:   *right_bp = 30; return 31;
    case TOK_LOR:        *right_bp = 41; return 40;
    case TOK_LAND:       *right_bp = 51; return 50;
    case TOK_PIPE:       *right_bp = 61; return 60;
    case TOK_CARET:      *right_bp = 71; return 70;
    case TOK_AMP:        *right_bp = 81; return 80;
    case TOK_EQ: case TOK_NE:
                         *right_bp = 91; return 90;
    case TOK_LT: case TOK_GT: case TOK_LE: case TOK_GE:
                         *right_bp = 101; return 100;
    case TOK_SHL: case TOK_SHR:
                         *right_bp = 111; return 110;
    case TOK_PLUS: case TOK_MINUS:
                         *right_bp = 121; return 120;
    case TOK_STAR: case TOK_SLASH: case TOK_PERCENT:
                         *right_bp = 131; return 130;
    default:
        return -1;
    }
}

static int postfix_bp(int type)
{
    switch (type) {
    case TOK_INC: case TOK_DEC:
    case TOK_LPAREN: case TOK_LBRACKET:
    case TOK_DOT: case TOK_ARROW:
    case TOK_LAUNCH_OPEN:
        return 160;
    default:
        return -1;
    }
}

static int can_start_unary(int t)
{
    switch (t) {
    case TOK_INT_LIT: case TOK_FLOAT_LIT: case TOK_STRING_LIT:
    case TOK_CHAR_LIT: case TOK_IDENT: case TOK_LPAREN:
    case TOK_BANG: case TOK_TILDE: case TOK_PLUS: case TOK_MINUS:
    case TOK_STAR: case TOK_AMP: case TOK_INC: case TOK_DEC:
    case TOK_TRUE: case TOK_FALSE: case TOK_NULLPTR: case TOK_SIZEOF:
        return 1;
    default:
        return 0;
    }
}

static int looks_like_cast(parser_t *P)
{
    if (cur_type(P) != TOK_LPAREN) return 0;
    int t = peek_type(P, 1);
    if (is_type_keyword(t)) return 1;
    if (t == TOK_CU_DEVICE || t == TOK_CU_HOST) return 1;
    if (t == TOK_IDENT) {
        const token_t *id = &P->tokens[P->pos + 1 < P->num_tokens
                                        ? P->pos + 1 : P->num_tokens - 1];
        int known = is_reg_type(P, id->offset, (uint16_t)id->len);
        if (peek_type(P, 2) == TOK_RPAREN) {
            int after = peek_type(P, 3);
            /* (ident) *expr — the classic C ambiguity. Could be
             * cast+deref OR multiplication. Only treat as cast if
             * we KNOW it's a type. Cost of getting this wrong: one
             * GPU aperture fault and an afternoon of disassembly. */
            if (after == TOK_STAR || after == TOK_AMP) {
                if (known) return 1;
            } else if (can_start_unary(after)) {
                /* (ident)0, (ident)x etc. — no mul/deref ambiguity,
                 * safe to assume cast (covers template params too) */
                return 1;
            }
        }
        /* (ident*) or (ident**) — pointer cast to struct/vector type.
         * Star is INSIDE the parens (part of the type), not ambiguous. */
        if (peek_type(P, 2) == TOK_STAR)
            return 1;
    }
    return 0;
}

static uint32_t parse_primary(parser_t *P)
{
    int t = cur_type(P);

    if (t == TOK_INT_LIT) {
        uint32_t n = alloc_node(P, AST_INT_LIT);
        P->nodes[n].d.text.offset = cur(P)->offset;
        P->nodes[n].d.text.len = cur(P)->len;
        advance(P);
        return n;
    }
    if (t == TOK_FLOAT_LIT) {
        uint32_t n = alloc_node(P, AST_FLOAT_LIT);
        P->nodes[n].d.text.offset = cur(P)->offset;
        P->nodes[n].d.text.len = cur(P)->len;
        advance(P);
        return n;
    }
    if (t == TOK_STRING_LIT) {
        uint32_t n = alloc_node(P, AST_STRING_LIT);
        P->nodes[n].d.text.offset = cur(P)->offset;
        P->nodes[n].d.text.len = cur(P)->len;
        advance(P);
        while (cur_type(P) == TOK_STRING_LIT) {
            P->nodes[n].d.text.len =
                (cur(P)->offset + cur(P)->len) - P->nodes[n].d.text.offset;
            advance(P);
        }
        return n;
    }
    if (t == TOK_CHAR_LIT) {
        uint32_t n = alloc_node(P, AST_CHAR_LIT);
        P->nodes[n].d.text.offset = cur(P)->offset;
        P->nodes[n].d.text.len = cur(P)->len;
        advance(P);
        return n;
    }
    if (t == TOK_TRUE || t == TOK_FALSE) {
        uint32_t n = alloc_node(P, AST_BOOL_LIT);
        P->nodes[n].d.ival = (t == TOK_TRUE) ? 1 : 0;
        advance(P);
        return n;
    }
    if (t == TOK_NULLPTR) {
        uint32_t n = alloc_node(P, AST_NULL_LIT);
        advance(P);
        return n;
    }
    if (t == TOK_IDENT) {
        uint32_t n = alloc_node(P, AST_IDENT);
        P->nodes[n].d.text.offset = cur(P)->offset;
        P->nodes[n].d.text.len = cur(P)->len;
        advance(P);
        if (cur_type(P) == TOK_DCOLON) {
            uint32_t scope = alloc_node(P, AST_SCOPE_RES);
            add_child(P, scope, n);
            advance(P);
            uint32_t rhs = parse_primary(P);
            add_child(P, scope, rhs);
            return scope;
        }
        /* C++ aggregate init: TypeName { expr, expr, ... } */
        if (cur_type(P) == TOK_LBRACE) {
            uint32_t type_node = alloc_node(P, AST_TYPE_SPEC);
            P->nodes[type_node].d.btype.kind = TYPE_NAME;
            add_child(P, type_node, n);
            uint32_t ilist = alloc_node(P, AST_INIT_LIST);
            advance(P);
            while (cur_type(P) != TOK_RBRACE && cur_type(P) != TOK_EOF) {
                uint32_t elem = parse_expr(P, 21);
                add_child(P, ilist, elem);
                if (!match(P, TOK_COMMA)) break;
            }
            expect(P, TOK_RBRACE);
            uint32_t cast = alloc_node(P, AST_CAST);
            add_child(P, cast, type_node);
            add_child(P, cast, ilist);
            P->nodes[cast].d.oper.flags = 0;
            return cast;
        }
        return n;
    }
    if (t == TOK_SIZEOF) {
        uint32_t n = alloc_node(P, AST_SIZEOF);
        advance(P);
        if (match(P, TOK_LPAREN)) {
            if (is_type_keyword(cur_type(P))) {
                uint16_t sq = 0, sc = 0;
                uint32_t inner = parse_type_spec(P, &sq, &sc);
                while (cur_type(P) == TOK_STAR) advance(P);
                add_child(P, n, inner);
            } else {
                uint32_t inner = parse_expr(P, 0);
                add_child(P, n, inner);
            }
            expect(P, TOK_RPAREN);
        } else {
            uint32_t inner = parse_expr(P, 150);
            add_child(P, n, inner);
        }
        return n;
    }
    if (t == TOK_STATIC_CAST || t == TOK_CONST_CAST ||
        t == TOK_DYNAMIC_CAST || t == TOK_REINTERPRET_CAST) {
        advance(P);
        expect(P, TOK_LT);
        uint16_t quals = 0, cuda = 0;
        uint32_t type_node = parse_type_spec(P, &quals, &cuda);
        int ptr_depth = 0;
        while (cur_type(P) == TOK_STAR) { advance(P); ptr_depth++; }
        expect(P, TOK_GT);
        expect(P, TOK_LPAREN);
        uint32_t cast = alloc_node(P, AST_CAST);
        add_child(P, cast, type_node);
        P->nodes[cast].d.oper.flags = ptr_depth;
        uint32_t operand = parse_expr(P, 0);
        add_child(P, cast, operand);
        expect(P, TOK_RPAREN);
        return cast;
    }
    if (looks_like_cast(P)) {
        uint32_t saved = P->pos;
        advance(P);
        uint16_t quals = 0, cuda = 0;
        uint32_t type_node = parse_type_spec(P, &quals, &cuda);
        int ptr_depth = 0;
        while (cur_type(P) == TOK_STAR) { advance(P); ptr_depth++; }
        if (cur_type(P) == TOK_RPAREN && type_node) {
            advance(P);
            uint32_t cast = alloc_node(P, AST_CAST);
            add_child(P, cast, type_node);
            P->nodes[cast].d.oper.flags = ptr_depth;
            uint32_t operand = parse_expr(P, 150);
            add_child(P, cast, operand);
            return cast;
        }
        P->pos = saved;
    }
    if (t == TOK_LPAREN) {
        advance(P);
        uint32_t inner = parse_expr(P, 0);
        expect(P, TOK_RPAREN);
        uint32_t n = alloc_node(P, AST_PAREN);
        add_child(P, n, inner);
        return n;
    }
    if (t == TOK_LBRACE) {
        uint32_t n = alloc_node(P, AST_INIT_LIST);
        advance(P);
        while (cur_type(P) != TOK_RBRACE && cur_type(P) != TOK_EOF) {
            uint32_t elem = parse_expr(P, 21);
            add_child(P, n, elem);
            if (!match(P, TOK_COMMA)) break;
        }
        expect(P, TOK_RBRACE);
        return n;
    }

    int bp = prefix_bp(t);
    if (bp >= 0) {
        uint32_t n = alloc_node(P, AST_UNARY_PREFIX);
        P->nodes[n].d.oper.op = t;
        advance(P);
        uint32_t operand = parse_expr(P, bp);
        add_child(P, n, operand);
        return n;
    }

    parse_error(P, BC_E022);
    advance(P);
    return 0;
}

static uint32_t parse_expr(parser_t *P, int min_prec)
{
    uint32_t lhs = parse_primary(P);
    if (!lhs) return 0;

    for (;;) {
        int t = cur_type(P);
        if (t == TOK_EOF || t == TOK_SEMI || t == TOK_RBRACE) break;

        int pbp = postfix_bp(t);
        if (pbp >= 0 && pbp >= min_prec) {
            if (t == TOK_INC || t == TOK_DEC) {
                uint32_t n = alloc_node(P, AST_UNARY_POSTFIX);
                P->nodes[n].d.oper.op = t;
                advance(P);
                add_child(P, n, lhs);
                lhs = n;
                continue;
            }
            if (t == TOK_LPAREN) {
                uint32_t call = alloc_node(P, AST_CALL);
                add_child(P, call, lhs);
                advance(P);
                while (cur_type(P) != TOK_RPAREN && cur_type(P) != TOK_EOF) {
                    uint32_t arg = parse_expr(P, 21);
                    add_child(P, call, arg);
                    if (!match(P, TOK_COMMA)) break;
                }
                expect(P, TOK_RPAREN);
                lhs = call;
                continue;
            }
            if (t == TOK_LBRACKET) {
                uint32_t sub = alloc_node(P, AST_SUBSCRIPT);
                add_child(P, sub, lhs);
                advance(P);
                uint32_t idx = parse_expr(P, 0);
                add_child(P, sub, idx);
                expect(P, TOK_RBRACKET);
                lhs = sub;
                continue;
            }
            if (t == TOK_DOT || t == TOK_ARROW) {
                uint32_t mem = alloc_node(P, AST_MEMBER);
                P->nodes[mem].d.member.is_arrow = (t == TOK_ARROW);
                add_child(P, mem, lhs);
                advance(P);
                uint32_t field = alloc_node(P, AST_IDENT);
                P->nodes[field].d.text.offset = cur(P)->offset;
                P->nodes[field].d.text.len = cur(P)->len;
                advance(P);
                add_child(P, mem, field);
                lhs = mem;
                continue;
            }
            if (t == TOK_LAUNCH_OPEN) {
                /* <<<grid, block[, smem[, stream]]>>>(args) */
                uint32_t launch = alloc_node(P, AST_LAUNCH);
                add_child(P, launch, lhs);
                advance(P);
                uint32_t grid = parse_expr(P, 21);
                add_child(P, launch, grid);
                expect(P, TOK_COMMA);
                uint32_t block = parse_expr(P, 21);
                add_child(P, launch, block);
                if (match(P, TOK_COMMA)) {
                    uint32_t smem = parse_expr(P, 21);
                    add_child(P, launch, smem);
                    if (match(P, TOK_COMMA)) {
                        uint32_t stream = parse_expr(P, 21);
                        add_child(P, launch, stream);
                    }
                }
                expect(P, TOK_LAUNCH_CLOSE);
                expect(P, TOK_LPAREN);
                while (cur_type(P) != TOK_RPAREN && cur_type(P) != TOK_EOF) {
                    uint32_t arg = parse_expr(P, 21);
                    add_child(P, launch, arg);
                    if (!match(P, TOK_COMMA)) break;
                }
                expect(P, TOK_RPAREN);
                lhs = launch;
                continue;
            }
        }

        int right_bp;
        int left_bp = infix_bp(t, &right_bp);
        if (left_bp < 0 || left_bp < min_prec) break;

        if (t == TOK_QUESTION) {
            uint32_t tern = alloc_node(P, AST_TERNARY);
            advance(P);
            add_child(P, tern, lhs);
            uint32_t then_expr = parse_expr(P, 0);
            add_child(P, tern, then_expr);
            expect(P, TOK_COLON);
            uint32_t else_expr = parse_expr(P, right_bp);
            add_child(P, tern, else_expr);
            lhs = tern;
            continue;
        }

        uint32_t bin = alloc_node(P, AST_BINARY);
        P->nodes[bin].d.oper.op = t;
        advance(P);
        uint32_t rhs = parse_expr(P, right_bp);
        add_child(P, bin, lhs);
        add_child(P, bin, rhs);
        lhs = bin;
    }

    return lhs;
}

/* Where we divine the programmer's intent from a soup of keywords */

static uint32_t parse_type_spec(parser_t *P, uint16_t *quals, uint16_t *cuda)
{
    *quals = 0;
    *cuda = 0;

    for (;;) {
        int ct = cur_type(P);
        if (is_cuda_qualifier(ct)) {
            *cuda |= cuda_flag_for(ct);
            advance(P);
            /* __launch_bounds__(maxThreads[, minBlocks]) — actually read the numbers
               this time, instead of pretending they don't exist */
            if ((ct == TOK_CU_LAUNCH_BOUNDS) && cur_type(P) == TOK_LPAREN) {
                advance(P);  /* ( */
                uint32_t max_expr = parse_expr(P, 21);
                if (max_expr && P->nodes[max_expr].type == AST_INT_LIT)
                    P->lb_max_pending = (uint32_t)parse_int_text(
                        P->src + P->nodes[max_expr].d.text.offset,
                        (int)P->nodes[max_expr].d.text.len);
                if (match(P, TOK_COMMA)) {
                    uint32_t min_expr = parse_expr(P, 21);
                    if (min_expr && P->nodes[min_expr].type == AST_INT_LIT)
                        P->lb_min_pending = (uint32_t)parse_int_text(
                            P->src + P->nodes[min_expr].d.text.offset,
                            (int)P->nodes[min_expr].d.text.len);
                }
                expect(P, TOK_RPAREN);
            }
        } else if (is_storage_class(ct) || ct == TOK_CONST ||
                   ct == TOK_VOLATILE || ct == TOK_CONSTEXPR) {
            *quals |= storage_flag_for(ct);
            advance(P);
            if (ct == TOK_EXTERN && cur_type(P) == TOK_STRING_LIT)
                advance(P);  /* skip "C" / "C++" linkage spec */
        } else {
            break;
        }
    }

    uint32_t node = alloc_node(P, AST_TYPE_SPEC);
    ast_node_t *n = &P->nodes[node];
    n->d.btype.is_unsigned = 0;
    n->d.btype.kind = TYPE_INT;

    int got_type = 0;
    for (;;) {
        int t = cur_type(P);
        if (t == TOK_UNSIGNED) { n->d.btype.is_unsigned = 1; advance(P); got_type = 1; }
        else if (t == TOK_SIGNED) { n->d.btype.is_unsigned = 0; advance(P); got_type = 1; }
        else if (t == TOK_VOID) { n->d.btype.kind = TYPE_VOID; advance(P); got_type = 1; break; }
        else if (t == TOK_BOOL) { n->d.btype.kind = TYPE_BOOL; advance(P); got_type = 1; break; }
        else if (t == TOK_CHAR) { n->d.btype.kind = TYPE_CHAR; advance(P); got_type = 1; break; }
        else if (t == TOK_SHORT) { n->d.btype.kind = TYPE_SHORT; advance(P); got_type = 1; break; }
        else if (t == TOK_INT) { n->d.btype.kind = TYPE_INT; advance(P); got_type = 1; break; }
        else if (t == TOK_LONG) {
            advance(P); got_type = 1;
            if (cur_type(P) == TOK_LONG) { n->d.btype.kind = TYPE_LLONG; advance(P); }
            else if (cur_type(P) == TOK_DOUBLE) { n->d.btype.kind = TYPE_LDOUBLE; advance(P); }
            else { n->d.btype.kind = TYPE_LONG; }
            break;
        }
        else if (t == TOK_FLOAT) { n->d.btype.kind = TYPE_FLOAT; advance(P); got_type = 1; break; }
        else if (t == TOK_DOUBLE) { n->d.btype.kind = TYPE_DOUBLE; advance(P); got_type = 1; break; }
        else if (t == TOK_AUTO) { n->d.btype.kind = TYPE_AUTO; advance(P); got_type = 1; break; }
        else if (t == TOK_STRUCT || t == TOK_UNION || t == TOK_CLASS) {
            n->d.btype.kind = (t == TOK_STRUCT) ? TYPE_STRUCT :
                              (t == TOK_UNION) ? TYPE_UNION : TYPE_CLASS;
            advance(P);
            if (cur_type(P) == TOK_IDENT) {
                uint32_t name = alloc_node(P, AST_IDENT);
                P->nodes[name].d.text.offset = cur(P)->offset;
                P->nodes[name].d.text.len = cur(P)->len;
                advance(P);
                add_child(P, node, name);
            }
            got_type = 1;
            break;
        }
        else if (t == TOK_ENUM) {
            n->d.btype.kind = TYPE_ENUM;
            advance(P);
            if (cur_type(P) == TOK_IDENT) {
                uint32_t name = alloc_node(P, AST_IDENT);
                P->nodes[name].d.text.offset = cur(P)->offset;
                P->nodes[name].d.text.len = cur(P)->len;
                advance(P);
                add_child(P, node, name);
            }
            got_type = 1;
            break;
        }
        else if (t == TOK_IDENT) {
            if (got_type) break;
            n->d.btype.kind = TYPE_NAME;
            uint32_t name = alloc_node(P, AST_IDENT);
            P->nodes[name].d.text.offset = cur(P)->offset;
            P->nodes[name].d.text.len = cur(P)->len;
            advance(P);
            add_child(P, node, name);
            if (cur_type(P) == TOK_DCOLON) {
                advance(P);
                uint32_t rhs = alloc_node(P, AST_IDENT);
                P->nodes[rhs].d.text.offset = cur(P)->offset;
                P->nodes[rhs].d.text.len = cur(P)->len;
                advance(P);
                add_child(P, node, rhs);
            }
            got_type = 1;
            break;
        }
        else if (t == TOK_CONST || t == TOK_VOLATILE) {
            *quals |= storage_flag_for(t);
            advance(P);
        }
        else break;
    }

    if (!got_type) {
        P->num_nodes--;
        return 0;
    }

    while (cur_type(P) == TOK_CONST || cur_type(P) == TOK_VOLATILE) {
        *quals |= storage_flag_for(cur_type(P));
        advance(P);
    }

    n->qualifiers = *quals;
    n->cuda_flags = *cuda;
    return node;
}

static uint32_t parse_param_list(parser_t *P)
{
    uint32_t first = 0, last = 0;

    if (cur_type(P) == TOK_VOID && peek_type(P, 1) == TOK_RPAREN) {
        advance(P);
        return 0;
    }

    while (cur_type(P) != TOK_RPAREN && cur_type(P) != TOK_EOF) {
        if (cur_type(P) == TOK_ELLIPSIS) {
            uint32_t va = alloc_node(P, AST_PARAM);
            P->nodes[va].d.oper.flags = 1;
            advance(P);
            if (!first) first = va; else P->nodes[last].next_sibling = va;
            last = va;
            break;
        }

        uint32_t param = alloc_node(P, AST_PARAM);
        uint16_t quals = 0, cuda = 0;
        uint32_t type = parse_type_spec(P, &quals, &cuda);
        add_child(P, param, type);
        P->nodes[param].qualifiers = quals;
        P->nodes[param].cuda_flags = cuda;

        {
            int ptr_depth = 0;
            while (cur_type(P) == TOK_STAR || cur_type(P) == TOK_AMP ||
                   cur_type(P) == TOK_CONST || cur_type(P) == TOK_CU_RESTRICT) {
                if (cur_type(P) == TOK_STAR) ptr_depth++;
                advance(P);
            }
            P->nodes[param].d.oper.flags = ptr_depth;
        }

        if (cur_type(P) == TOK_IDENT) {
            uint32_t name = alloc_node(P, AST_IDENT);
            P->nodes[name].d.text.offset = cur(P)->offset;
            P->nodes[name].d.text.len = cur(P)->len;
            advance(P);
            add_child(P, param, name);
        }

        while (cur_type(P) == TOK_LBRACKET) {
            advance(P);
            if (cur_type(P) != TOK_RBRACKET)
                parse_expr(P, 0);
            expect(P, TOK_RBRACKET);
        }

        if (cur_type(P) == TOK_ASSIGN) {
            advance(P);
            uint32_t def = parse_expr(P, 21);
            add_child(P, param, def);
        }

        if (!first) first = param; else P->nodes[last].next_sibling = param;
        last = param;
        if (!match(P, TOK_COMMA)) break;
    }
    return first;
}

static int starts_declaration(parser_t *P)
{
    int t = cur_type(P);
    if (is_type_keyword(t) || is_storage_class(t) || is_cuda_qualifier(t))
        return 1;
    if (t == TOK_TEMPLATE) return 1;
    if (t == TOK_TYPEDEF) return 1;
    if (t == TOK_NAMESPACE) return 1;
    if (t == TOK_USING) return 1;
    /* IDENT IDENT or IDENT * IDENT: probably a declaration. Probably. */
    if (t == TOK_IDENT) {
        int next = peek_type(P, 1);
        if (next == TOK_IDENT || next == TOK_STAR || next == TOK_AMP ||
            next == TOK_DCOLON || next == TOK_LT || next == TOK_OPERATOR)
            return 1;
    }
    return 0;
}

static uint32_t parse_declaration(parser_t *P)
{
    uint32_t decl_node;
    uint32_t anon_def = 0;   /* stashed AST_STRUCT_DEF for anonymous inline decls */
    uint16_t quals = 0, cuda = 0;

    if (cur_type(P) == TOK_TEMPLATE) {
        uint32_t tmpl = alloc_node(P, AST_TEMPLATE_DECL);
        advance(P);
        expect(P, TOK_LT);
        while (cur_type(P) != TOK_GT && cur_type(P) != TOK_EOF) {
            uint32_t tp = alloc_node(P, AST_TEMPLATE_PARAM);
            if (cur_type(P) == TOK_TYPENAME || cur_type(P) == TOK_CLASS) {
                P->nodes[tp].d.oper.flags = 0;
                advance(P);
            } else {
                P->nodes[tp].d.oper.flags = 1;
                uint16_t q2, c2;
                uint32_t ptype = parse_type_spec(P, &q2, &c2);
                add_child(P, tp, ptype);
            }
            if (cur_type(P) == TOK_IDENT) {
                uint32_t name = alloc_node(P, AST_IDENT);
                P->nodes[name].d.text.offset = cur(P)->offset;
                P->nodes[name].d.text.len = cur(P)->len;
                advance(P);
                add_child(P, tp, name);
            }
            if (cur_type(P) == TOK_ASSIGN) {
                advance(P);
                uint32_t def = parse_expr(P, 21);
                add_child(P, tp, def);
            }
            add_child(P, tmpl, tp);
            if (!match(P, TOK_COMMA)) break;
        }
        expect(P, TOK_GT);
        uint32_t inner = parse_declaration(P);
        add_child(P, tmpl, inner);
        return tmpl;
    }

    if (cur_type(P) == TOK_NAMESPACE) {
        uint32_t ns = alloc_node(P, AST_NAMESPACE);
        advance(P);
        if (cur_type(P) == TOK_IDENT) {
            uint32_t name = alloc_node(P, AST_IDENT);
            P->nodes[name].d.text.offset = cur(P)->offset;
            P->nodes[name].d.text.len = cur(P)->len;
            advance(P);
            add_child(P, ns, name);
        }
        if (cur_type(P) == TOK_LBRACE) {
            advance(P);
            while (cur_type(P) != TOK_RBRACE && cur_type(P) != TOK_EOF) {
                uint32_t old_pos = P->pos;
                uint32_t inner = parse_decl_or_stmt(P);
                if (inner) add_child(P, ns, inner);
                else if (P->pos == old_pos) {
                    parse_error(P, BC_E023);
                    sync_past_semi(P);
                }
            }
            expect(P, TOK_RBRACE);
        }
        return ns;
    }

    if (cur_type(P) == TOK_USING) {
        uint32_t u = alloc_node(P, AST_USING);
        advance(P);
        if (cur_type(P) == TOK_NAMESPACE) {
            advance(P);
        }
        while (cur_type(P) != TOK_SEMI && cur_type(P) != TOK_EOF) {
            uint32_t name = alloc_node(P, AST_IDENT);
            P->nodes[name].d.text.offset = cur(P)->offset;
            P->nodes[name].d.text.len = cur(P)->len;
            advance(P);
            add_child(P, u, name);
            match(P, TOK_DCOLON);
            if (cur_type(P) == TOK_ASSIGN) {
                advance(P);
                uint16_t q2, c2;
                uint32_t alias_type = parse_type_spec(P, &q2, &c2);
                add_child(P, u, alias_type);
            }
        }
        expect(P, TOK_SEMI);
        return u;
    }

    uint32_t type_node = parse_type_spec(P, &quals, &cuda);
    if (!type_node) {
        parse_error(P, BC_E024);
        advance(P);
        return 0;
    }

    if ((P->nodes[type_node].d.btype.kind == TYPE_STRUCT ||
         P->nodes[type_node].d.btype.kind == TYPE_UNION ||
         P->nodes[type_node].d.btype.kind == TYPE_CLASS ||
         P->nodes[type_node].d.btype.kind == TYPE_ENUM) &&
        cur_type(P) == TOK_LBRACE) {
        int is_enum  = (P->nodes[type_node].d.btype.kind == TYPE_ENUM);
        int is_union = (P->nodes[type_node].d.btype.kind == TYPE_UNION);
        uint32_t def = alloc_node(P, is_enum ? AST_ENUM_DEF : AST_STRUCT_DEF);
        P->nodes[def].qualifiers = quals;
        P->nodes[def].cuda_flags = cuda;
        add_child(P, def, type_node);
        /* Register struct/enum name so (name)*x parses as mul, not cast */
        { uint32_t fc = P->nodes[type_node].first_child;
          if (fc && P->nodes[fc].type == AST_IDENT)
              reg_tname(P, P->nodes[fc].d.text.offset,
                        (uint16_t)P->nodes[fc].d.text.len); }
        advance(P);
        if (is_enum) {
            while (cur_type(P) != TOK_RBRACE && cur_type(P) != TOK_EOF) {
                uint32_t ev = alloc_node(P, AST_ENUMERATOR);
                if (cur_type(P) == TOK_IDENT) {
                    P->nodes[ev].d.text.offset = cur(P)->offset;
                    P->nodes[ev].d.text.len = cur(P)->len;
                    advance(P);
                }
                if (cur_type(P) == TOK_ASSIGN) {
                    advance(P);
                    uint32_t val = parse_expr(P, 21);
                    add_child(P, ev, val);
                }
                add_child(P, def, ev);
                if (!match(P, TOK_COMMA)) break;
            }
        } else {
            while (cur_type(P) != TOK_RBRACE && cur_type(P) != TOK_EOF) {
                if (cur_type(P) == TOK_PUBLIC || cur_type(P) == TOK_PRIVATE ||
                    cur_type(P) == TOK_PROTECTED) {
                    advance(P); match(P, TOK_COLON);
                    continue;
                }
                uint32_t member = parse_declaration(P);
                if (member) add_child(P, def, member);
            }
        }
        expect(P, TOK_RBRACE);

        /* Anonymous struct/union with inline variable declaration?
         *   struct { float f; int i; } cvt;
         * Next token is an identifier (or *) → we need a variable,
         * not just a bare type definition. Give the nameless struct
         * a synthetic name so sema can find it, then build a fresh
         * TYPE_STRUCT node for the var_decl (the original type_node
         * already belongs to def — re-parenting would tangle the
         * sibling chains like Christmas lights in February). */
        if (!is_enum && (cur_type(P) == TOK_IDENT ||
                         cur_type(P) == TOK_STAR)) {
            if (P->anon_len + 16 < sizeof(P->anon_buf)) {
                uint32_t aoff = P->anon_len;
                int n = snprintf(P->anon_buf + aoff,
                                 sizeof(P->anon_buf) - aoff,
                                 "__a%u", P->anon_cnt++);
                P->anon_len += (uint32_t)(n + 1);

                /* Name the nameless struct inside def's type_node */
                uint32_t sn = alloc_node(P, AST_IDENT);
                P->nodes[sn].d.text.offset = BC_ANON_BASE + aoff;
                P->nodes[sn].d.text.len = (uint16_t)n;
                add_child(P, type_node, sn);

                reg_tname(P, BC_ANON_BASE + aoff, (uint16_t)n);

                /* Fresh TYPE_STRUCT referencing the synthetic name.
                 * This one goes into the var_decl; def keeps the
                 * original with all the member definitions. */
                type_node = alloc_node(P, AST_TYPE_SPEC);
                P->nodes[type_node].d.btype.kind =
                    is_union ? TYPE_UNION : TYPE_STRUCT;
                uint32_t ref = alloc_node(P, AST_IDENT);
                P->nodes[ref].d.text.offset = BC_ANON_BASE + aoff;
                P->nodes[ref].d.text.len = (uint16_t)n;
                add_child(P, type_node, ref);

                /* Stash def for injection into the var_decl as
                 * child #3 (after type_node, name). check_var_decl
                 * will spot it and call collect_struct_def before
                 * resolve_typespec needs the struct registered. */
                P->nodes[def].next_sibling = 0;
                anon_def = def;
            }
            /* Fall through to variable declaration code.
             * type_node is now a fresh reference, def rides along. */
        } else {
            match(P, TOK_SEMI);
            return def;
        }
    }

    int ptr_depth = 0;
    while (cur_type(P) == TOK_STAR) { advance(P); ptr_depth++; }
    while (cur_type(P) == TOK_CONST || cur_type(P) == TOK_AMP ||
           cur_type(P) == TOK_CU_RESTRICT) { advance(P); }

    /* __launch_bounds__ in suffix position (tinygrad style) */
    if (cur_type(P) == TOK_CU_LAUNCH_BOUNDS) {
        advance(P);
        if (cur_type(P) == TOK_LPAREN) {
            advance(P);
            uint32_t max_expr = parse_expr(P, 21);
            if (max_expr && P->nodes[max_expr].type == AST_INT_LIT)
                P->lb_max_pending = (uint32_t)parse_int_text(
                    P->src + P->nodes[max_expr].d.text.offset,
                    (int)P->nodes[max_expr].d.text.len);
            if (match(P, TOK_COMMA)) {
                uint32_t min_expr = parse_expr(P, 21);
                if (min_expr && P->nodes[min_expr].type == AST_INT_LIT)
                    P->lb_min_pending = (uint32_t)parse_int_text(
                        P->src + P->nodes[min_expr].d.text.offset,
                        (int)P->nodes[min_expr].d.text.len);
            }
            expect(P, TOK_RPAREN);
        }
    }

    if (cur_type(P) == TOK_IDENT || cur_type(P) == TOK_TILDE
        || cur_type(P) == TOK_OPERATOR) {
        int is_dtor = (cur_type(P) == TOK_TILDE);
        if (is_dtor) advance(P);

        uint32_t name = alloc_node(P, AST_IDENT);

        if (cur_type(P) == TOK_OPERATOR) {
            /* operator+, operator[], operator() etc.
               Use source span from 'operator' keyword through the symbol. */
            uint32_t op_start = cur(P)->offset;
            advance(P); /* consume 'operator' */
            int t = cur_type(P);
            if (t == TOK_LPAREN) {
                advance(P); /* ( */
                /* ) is the end of the operator name */
                P->nodes[name].d.text.offset = op_start;
                P->nodes[name].d.text.len = (cur(P)->offset + cur(P)->len) - op_start;
                advance(P); /* ) */
            } else if (t == TOK_LBRACKET) {
                advance(P); /* [ */
                P->nodes[name].d.text.offset = op_start;
                P->nodes[name].d.text.len = (cur(P)->offset + cur(P)->len) - op_start;
                advance(P); /* ] */
            } else if (t == TOK_PLUS || t == TOK_MINUS || t == TOK_STAR ||
                       t == TOK_SLASH || t == TOK_PERCENT || t == TOK_AMP ||
                       t == TOK_PIPE || t == TOK_CARET || t == TOK_LT ||
                       t == TOK_GT || t == TOK_BANG || t == TOK_TILDE ||
                       t == TOK_EQ || t == TOK_NE || t == TOK_LE ||
                       t == TOK_GE || t == TOK_SHL || t == TOK_SHR ||
                       t == TOK_LAND || t == TOK_LOR || t == TOK_ASSIGN ||
                       t == TOK_PLUS_EQ || t == TOK_MINUS_EQ ||
                       t == TOK_STAR_EQ || t == TOK_SLASH_EQ ||
                       t == TOK_INC || t == TOK_DEC) {
                P->nodes[name].d.text.offset = op_start;
                P->nodes[name].d.text.len = (cur(P)->offset + cur(P)->len) - op_start;
                advance(P);
            } else {
                /* Fallback: just use 'operator' as the name */
                P->nodes[name].d.text.offset = op_start;
                P->nodes[name].d.text.len = 8;
            }
        } else {
            P->nodes[name].d.text.offset = cur(P)->offset;
            P->nodes[name].d.text.len = cur(P)->len;
            advance(P);
        }

        if (cur_type(P) == TOK_DCOLON) {
            advance(P);
            uint32_t method_name = alloc_node(P, AST_IDENT);
            P->nodes[method_name].d.text.offset = cur(P)->offset;
            P->nodes[method_name].d.text.len = cur(P)->len;
            advance(P);
            uint32_t scope = alloc_node(P, AST_SCOPE_RES);
            add_child(P, scope, name);
            add_child(P, scope, method_name);
            name = scope;
        }

        if (cur_type(P) == TOK_LT) {
            advance(P);
            uint32_t targs = alloc_node(P, AST_TEMPLATE_ARGS);
            while (cur_type(P) != TOK_GT && cur_type(P) != TOK_EOF) {
                uint16_t q2, c2;
                uint32_t ta = parse_type_spec(P, &q2, &c2);
                if (ta) add_child(P, targs, ta);
                else {
                    uint32_t ea = parse_expr(P, 21);
                    add_child(P, targs, ea);
                }
                if (!match(P, TOK_COMMA)) break;
            }
            expect(P, TOK_GT);
            add_child(P, name, targs);
        }

        if (cur_type(P) == TOK_LPAREN) {
            int is_def = 0;
            uint32_t func = alloc_node(P, AST_FUNC_DECL);
            P->nodes[func].qualifiers = quals;
            P->nodes[func].cuda_flags = cuda;
            P->nodes[func].d.oper.flags = ptr_depth;
            P->nodes[func].launch_bounds_max = P->lb_max_pending;
            P->nodes[func].launch_bounds_min = P->lb_min_pending;
            P->lb_max_pending = 0;
            P->lb_min_pending = 0;
            add_child(P, func, type_node);
            add_child(P, func, name);

            advance(P);
            uint32_t params = parse_param_list(P);
            if (params) add_child(P, func, params);
            expect(P, TOK_RPAREN);

            while (cur_type(P) == TOK_CONST || cur_type(P) == TOK_NOEXCEPT ||
                   cur_type(P) == TOK_OVERRIDE || cur_type(P) == TOK_FINAL) {
                advance(P);
            }

            if (cur_type(P) == TOK_LBRACE) {
                P->nodes[func].type = AST_FUNC_DEF;
                advance(P);
                uint32_t body = alloc_node(P, AST_BLOCK);
                while (cur_type(P) != TOK_RBRACE && cur_type(P) != TOK_EOF) {
                    uint32_t old_pos = P->pos;
                    uint32_t s = parse_decl_or_stmt(P);
                    if (s) add_child(P, body, s);
                    else if (P->pos == old_pos) {
                        parse_error(P, BC_E025);
                        sync_past_semi(P);
                    }
                }
                expect(P, TOK_RBRACE);
                add_child(P, func, body);
                is_def = 1;
            }

            if (!is_def) match(P, TOK_SEMI);
            return func;
        }

        decl_node = alloc_node(P, AST_VAR_DECL);
        P->nodes[decl_node].qualifiers = quals;
        P->nodes[decl_node].cuda_flags = cuda;
        P->nodes[decl_node].d.oper.flags = ptr_depth;
        add_child(P, decl_node, type_node);
        add_child(P, decl_node, name);
        /* Inline struct/union def rides as child #3 — sema detects
         * and registers it before resolving the type reference. */
        if (anon_def) add_child(P, decl_node, anon_def);
        /* Register typedef name for cast disambiguation */
        if (quals & QUAL_TYPEDEF)
            reg_tname(P, P->nodes[name].d.text.offset,
                      (uint16_t)P->nodes[name].d.text.len);

        { int ndims = 0;
        while (cur_type(P) == TOK_LBRACKET) {
            advance(P);
            if (cur_type(P) != TOK_RBRACKET) {
                uint32_t sz = parse_expr(P, 0);
                add_child(P, decl_node, sz);
                ndims++;
            }
            expect(P, TOK_RBRACKET);
        }
        /* Store array dimension count so lowerer can distinguish
         * `int x[4]` (ndims=1, child=4) from `int x = 4` (ndims=0, child=4).
         * The classic "is the next child an array size or an initializer?"
         * ambiguity that has haunted C compilers since 1972. */
        P->nodes[decl_node].d.oper.op = ndims;
        }

        if (cur_type(P) == TOK_ASSIGN) {
            advance(P);
            uint32_t init = parse_expr(P, 21);
            add_child(P, decl_node, init);
        }

        while (match(P, TOK_COMMA)) {
            uint32_t extra = alloc_node(P, AST_VAR_DECL);
            P->nodes[extra].qualifiers = quals;
            P->nodes[extra].cuda_flags = cuda;
            int ep = 0;
            while (cur_type(P) == TOK_STAR) { advance(P); ep++; }
            P->nodes[extra].d.oper.flags = ep;
            add_child(P, extra, type_node);
            if (cur_type(P) == TOK_IDENT) {
                uint32_t en = alloc_node(P, AST_IDENT);
                P->nodes[en].d.text.offset = cur(P)->offset;
                P->nodes[en].d.text.len = cur(P)->len;
                advance(P);
                add_child(P, extra, en);
            }
            while (cur_type(P) == TOK_LBRACKET) {
                advance(P);
                if (cur_type(P) != TOK_RBRACKET)
                    parse_expr(P, 0);
                expect(P, TOK_RBRACKET);
            }
            if (cur_type(P) == TOK_ASSIGN) {
                advance(P);
                uint32_t init = parse_expr(P, 21);
                add_child(P, extra, init);
            }
            P->nodes[decl_node].next_sibling = extra;
            decl_node = extra;
        }

        if (!expect(P, TOK_SEMI))
            sync_past_semi(P);
        return decl_node; /* returns first in chain */
    }

    match(P, TOK_SEMI);
    return type_node;
}

/* One statement at a time, like a confession */

static uint32_t parse_block(parser_t *P)
{
    uint32_t block = alloc_node(P, AST_BLOCK);
    expect(P, TOK_LBRACE);
    while (cur_type(P) != TOK_RBRACE && cur_type(P) != TOK_EOF) {
        uint32_t old_pos = P->pos;
        uint32_t s = parse_decl_or_stmt(P);
        if (s) add_child(P, block, s);
        else if (P->pos == old_pos) {
            /* Stuck in a block. Skip to the next thing that looks intentional. */
            parse_error(P, BC_E026);
            sync_past_semi(P);
        }
    }
    expect(P, TOK_RBRACE);
    return block;
}

static uint32_t parse_stmt(parser_t *P)
{
    int t = cur_type(P);

    if (t == TOK_LBRACE) return parse_block(P);

    if (t == TOK_IF) {
        uint32_t n = alloc_node(P, AST_IF);
        advance(P);
        expect(P, TOK_LPAREN);
        uint32_t cond = parse_expr(P, 0);
        add_child(P, n, cond);
        expect(P, TOK_RPAREN);
        uint32_t then_s = parse_decl_or_stmt(P);
        add_child(P, n, then_s);
        if (match(P, TOK_ELSE)) {
            uint32_t else_s = parse_decl_or_stmt(P);
            add_child(P, n, else_s);
        }
        return n;
    }

    if (t == TOK_FOR) {
        uint32_t n = alloc_node(P, AST_FOR);
        advance(P);
        expect(P, TOK_LPAREN);
        if (cur_type(P) != TOK_SEMI) {
            uint32_t init = starts_declaration(P) ?
                            parse_declaration(P) : parse_expr(P, 0);
            add_child(P, n, init);
            if (P->nodes[init].type != AST_VAR_DECL)
                match(P, TOK_SEMI);
        } else {
            add_child(P, n, alloc_node(P, AST_NONE));
            advance(P);
        }
        if (cur_type(P) != TOK_SEMI) {
            uint32_t cond = parse_expr(P, 0);
            add_child(P, n, cond);
        } else {
            add_child(P, n, alloc_node(P, AST_NONE));
        }
        expect(P, TOK_SEMI);
        if (cur_type(P) != TOK_RPAREN) {
            uint32_t inc = parse_expr(P, 0);
            add_child(P, n, inc);
        } else {
            add_child(P, n, alloc_node(P, AST_NONE));
        }
        expect(P, TOK_RPAREN);
        uint32_t body = parse_decl_or_stmt(P);
        add_child(P, n, body);
        return n;
    }

    if (t == TOK_WHILE) {
        uint32_t n = alloc_node(P, AST_WHILE);
        advance(P);
        expect(P, TOK_LPAREN);
        uint32_t cond = parse_expr(P, 0);
        add_child(P, n, cond);
        expect(P, TOK_RPAREN);
        uint32_t body = parse_decl_or_stmt(P);
        add_child(P, n, body);
        return n;
    }

    if (t == TOK_DO) {
        uint32_t n = alloc_node(P, AST_DO_WHILE);
        advance(P);
        uint32_t body = parse_decl_or_stmt(P);
        add_child(P, n, body);
        expect(P, TOK_WHILE);
        expect(P, TOK_LPAREN);
        uint32_t cond = parse_expr(P, 0);
        add_child(P, n, cond);
        expect(P, TOK_RPAREN);
        expect(P, TOK_SEMI);
        return n;
    }

    if (t == TOK_SWITCH) {
        uint32_t n = alloc_node(P, AST_SWITCH);
        advance(P);
        expect(P, TOK_LPAREN);
        uint32_t cond = parse_expr(P, 0);
        add_child(P, n, cond);
        expect(P, TOK_RPAREN);
        uint32_t body = parse_block(P);
        add_child(P, n, body);
        return n;
    }

    if (t == TOK_CASE) {
        uint32_t n = alloc_node(P, AST_CASE);
        advance(P);
        uint32_t val = parse_expr(P, 0);
        add_child(P, n, val);
        expect(P, TOK_COLON);
        return n;
    }

    if (t == TOK_DEFAULT) {
        uint32_t n = alloc_node(P, AST_DEFAULT);
        advance(P);
        expect(P, TOK_COLON);
        return n;
    }

    if (t == TOK_RETURN) {
        uint32_t n = alloc_node(P, AST_RETURN);
        advance(P);
        if (cur_type(P) != TOK_SEMI) {
            uint32_t val = parse_expr(P, 0);
            add_child(P, n, val);
        }
        expect(P, TOK_SEMI);
        return n;
    }

    if (t == TOK_BREAK) {
        uint32_t n = alloc_node(P, AST_BREAK);
        advance(P); expect(P, TOK_SEMI);
        return n;
    }
    if (t == TOK_CONTINUE) {
        uint32_t n = alloc_node(P, AST_CONTINUE);
        advance(P); expect(P, TOK_SEMI);
        return n;
    }

    if (t == TOK_GOTO) {
        uint32_t n = alloc_node(P, AST_GOTO);
        advance(P);
        if (cur_type(P) == TOK_IDENT) {
            uint32_t label = alloc_node(P, AST_IDENT);
            P->nodes[label].d.text.offset = cur(P)->offset;
            P->nodes[label].d.text.len = cur(P)->len;
            advance(P);
            add_child(P, n, label);
        }
        expect(P, TOK_SEMI);
        return n;
    }

    if (t == TOK_IDENT && peek_type(P, 1) == TOK_COLON) {
        uint32_t n = alloc_node(P, AST_LABEL);
        P->nodes[n].d.text.offset = cur(P)->offset;
        P->nodes[n].d.text.len = cur(P)->len;
        advance(P); advance(P);
        return n;
    }

    uint32_t expr = parse_expr(P, 0);
    uint32_t n = alloc_node(P, AST_EXPR_STMT);
    add_child(P, n, expr);
    expect(P, TOK_SEMI);
    return n;
}

static uint32_t parse_decl_or_stmt(parser_t *P)
{
    if (cur_type(P) == TOK_PP_LINE) {
        uint32_t pp = alloc_node(P, AST_PP_DIRECTIVE);
        P->nodes[pp].d.text.offset = cur(P)->offset;
        P->nodes[pp].d.text.len = cur(P)->len;
        advance(P);
        return pp;
    }

    if (starts_declaration(P))
        return parse_declaration(P);

    return parse_stmt(P);
}

uint32_t parser_parse(parser_t *P)
{
    uint32_t tu = alloc_node(P, AST_TRANSLATION_UNIT);
    while (cur_type(P) != TOK_EOF) {
        uint32_t old_pos = P->pos;
        uint32_t decl = parse_decl_or_stmt(P);
        if (decl) add_child(P, tu, decl);
        else if (P->pos == old_pos) {
            /* No progress — the parser is going in circles.
               Politely suggest we move on. */
            parse_error(P, BC_E027);
            sync_to_next_decl(P);
        }
    }
    return tu;
}

/* For those who wish to see the tree */

static void dump_node_data(const parser_t *P, const ast_node_t *n)
{
    char text[128];
    switch (n->type) {
    case AST_INT_LIT: case AST_FLOAT_LIT: case AST_STRING_LIT:
    case AST_CHAR_LIT: case AST_IDENT: case AST_PP_DIRECTIVE:
    case AST_LABEL: case AST_ENUMERATOR:
    {
        int len = (int)n->d.text.len;
        if (len > 120) len = 120;
        /* Synthetic anon names live in P->anon_buf, not P->src. Offsets
         * at or above BC_ANON_BASE are sentinels into the anon buffer;
         * blindly reading P->src + huge_offset segfaults, which the
         * typedef-struct dump tripped hard. */
        const char *base = (n->d.text.offset >= BC_ANON_BASE)
            ? P->anon_buf + (n->d.text.offset - BC_ANON_BASE)
            : P->src + n->d.text.offset;
        memcpy(text, base, (size_t)len);
        text[len] = '\0';
        printf(" %s", text);
        break;
    }
    case AST_BINARY:
        printf(" %s", token_type_name(n->d.oper.op));
        break;
    case AST_UNARY_PREFIX: case AST_UNARY_POSTFIX:
        printf(" %s", token_type_name(n->d.oper.op));
        break;
    case AST_BOOL_LIT:
        printf(" %s", n->d.ival ? "true" : "false");
        break;
    case AST_MEMBER:
        printf(" %s", n->d.member.is_arrow ? "->" : ".");
        break;
    case AST_TYPE_SPEC:
    {
        const char *kinds[] = {
            "void","bool","char","short","int","long","llong",
            "float","double","ldouble","unsigned","signed",
            "struct","enum","union","class","auto","name"
        };
        int k = n->d.btype.kind;
        printf(" %s%s", n->d.btype.is_unsigned ? "unsigned " : "",
               (k >= 0 && k <= TYPE_NAME) ? kinds[k] : "?");
        break;
    }
    case AST_FUNC_DEF: case AST_FUNC_DECL:
        if (n->cuda_flags) {
            if (n->cuda_flags & CUDA_GLOBAL) printf(" __global__");
            if (n->cuda_flags & CUDA_DEVICE) printf(" __device__");
            if (n->cuda_flags & CUDA_HOST) printf(" __host__");
        }
        if (n->launch_bounds_max > 0)
            printf(" lb_max=%u", n->launch_bounds_max);
        if (n->launch_bounds_min > 0)
            printf(" lb_min=%u", n->launch_bounds_min);
        break;
    case AST_VAR_DECL:
        if (n->d.oper.flags > 0) printf(" ptr=%d", n->d.oper.flags);
        if (n->cuda_flags) {
            if (n->cuda_flags & CUDA_SHARED) printf(" __shared__");
            if (n->cuda_flags & CUDA_CONSTANT) printf(" __constant__");
            if (n->cuda_flags & CUDA_DEVICE) printf(" __device__");
        }
        break;
    default:
        break;
    }
}

void ast_dump(const parser_t *P, uint32_t idx, int depth)
{
    /* Iterative tree walk. JPL Rule 1: thou shalt not recurse. */
    struct { uint32_t node; int depth; int phase; } stack[BC_MAX_DEPTH];
    int sp = 0;

    if (!idx || idx >= P->num_nodes) return;
    stack[sp].node = idx;
    stack[sp].depth = depth;
    stack[sp].phase = 0;
    sp++;

    while (sp > 0) {
        int top = sp - 1;
        uint32_t ni = stack[top].node;
        int d = stack[top].depth;
        int phase = stack[top].phase;
        if (ni >= P->num_nodes) return;
        const ast_node_t *n = &P->nodes[ni];

        if (phase == 0) {
            for (int i = 0; i < d; i++) printf("  ");
            printf("(%s", ast_type_name(n->type));
            dump_node_data(P, n);

            if (n->first_child) {
                printf("\n");
                stack[top].phase = 1;

                /* Push children in reverse order so first child pops first */
                uint32_t children[BC_MAX_DEPTH];
                int nc = 0;
                uint32_t c = n->first_child;
                uint32_t guard = P->max_nodes;
                while (c && c < P->num_nodes && nc < BC_MAX_DEPTH && --guard) {
                    children[nc++] = c;
                    c = P->nodes[c].next_sibling;
                }
                for (int i = nc - 1; i >= 0 && sp < BC_MAX_DEPTH; i--) {
                    stack[sp].node = children[i];
                    stack[sp].depth = d + 1;
                    stack[sp].phase = 0;
                    sp++;
                }
            } else {
                printf(")\n");
                sp--;
            }
        } else {
            for (int i = 0; i < d; i++) printf("  ");
            printf(")\n");
            sp--;
        }
    }
}
