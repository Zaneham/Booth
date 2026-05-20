/* Triton frontend lowering to BIR, stub.
 *
 * The eventual contents of this file translate the Triton AST into
 * the same BIR our existing backends already know how to consume,
 * with tl.program_id mapping onto BIR_BLOCK_ID, tl.load mapping onto
 * BIR_LOAD, and so on. Lives empty until the lexer, parser, and
 * sema phases above produce something it has any business consuming. */

#include "triton.h"

int tn_lower(bir_module_t *out)
{
    (void)out;
    return BC_ERR_TRITON;
}
