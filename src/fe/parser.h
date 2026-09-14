#ifndef BARRACUDA_PARSER_H
#define BARRACUDA_PARSER_H

#include "ast.h"
#include "token.h"

typedef struct parser_s {
    const token_t  *tokens;
    uint32_t        num_tokens;
    uint32_t        pos;
    const char     *src;

    ast_node_t     *nodes;
    uint32_t        num_nodes;
    uint32_t        max_nodes;

    /* Launch bounds stash — held between parse_type_spec and func node creation */
    uint32_t    lb_max_pending;
    uint32_t    lb_min_pending;

    /* Type name registry — struct/typedef names for cast disambiguation.
     * Without this, (var) * expr parses as cast+deref instead of mul.
     * The classic C ambiguity that has ruined more weekends than ISO 8601. */
    struct { uint32_t off; uint16_t len; } tnames[BC_MAX_TNAMES];
    int             num_tnames;

    uint8_t         bshad[BC_MAX_BTNAMS];

    /* Synthetic name buffer — anonymous struct/union variable declarations
     * need a name for sema lookup. We can't inject text into the const source
     * buffer, so synthetic names live here. Offsets >= BC_ANON_BASE reference
     * this buffer instead of src. Like a P.O. box for nameless types. */
    char            anon_buf[256];
    uint32_t        anon_len;
    uint32_t        anon_cnt;

    struct { uint32_t off; uint32_t len; } packs[32];
    int             npacks;

    struct { uint32_t off; uint16_t len; } tmpls[BC_MAX_TMPLS];
    int             ntmpls;
    int             gsplit;
    int             cpend;
    int             targ;

    /* Enclosing struct name, so a constructor can be told apart from a
     * declaration that happens to start with a type name. len 0 = not in one. */
    uint32_t        cs_off;
    uint16_t        cs_len;

    char            qrep[8][48];
    int             nqrep;

    bc_error_t      errors[BC_MAX_ERRORS];
    int             num_errors;
} parser_t;

/* Sentinel offset: text offsets >= this come from anon_buf, not src */
#define BC_ANON_BASE  0x80000000u

void parser_init(parser_t *P, const token_t *tokens, uint32_t num_tokens,
                 const char *src, ast_node_t *nodes, uint32_t max_nodes);

uint32_t parser_parse(parser_t *P);

void ast_dump(const parser_t *P, uint32_t node, int depth);

#endif /* BARRACUDA_PARSER_H */
