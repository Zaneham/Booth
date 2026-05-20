/* Triton frontend parser, stub.
 *
 * Subsequent sittings will fill this in with a recursive descent
 * parser that walks the token stream produced by lex.c and builds an
 * AST in the shape the sema and lowering passes want. For this
 * sitting the lexer alone is the deliverable, and the parser is a
 * polite acknowledgement that the slot exists. */

#include "triton.h"

int tn_parse(const tn_lex_t *L)
{
    (void)L;
    return BC_ERR_TRITON;
}
