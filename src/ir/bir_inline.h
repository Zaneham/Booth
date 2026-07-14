#ifndef BARRACUDA_BIR_INLINE_H
#define BARRACUDA_BIR_INLINE_H

#include "bir.h"

/* Inline calls to user __device__ functions into their callers.
 *
 * The GPU and vector backends (AMD, NVIDIA, Tensix) have no calling
 * convention wired for device functions, their isel aborts the moment it
 * meets a BIR_CALL. GPUs treat device functions as inline anyway, since a
 * real call costs registers and scratch nobody wants to spend, so the fix
 * is to splice the callee body in at the call site before isel ever runs.
 * The scalar backends (CPU, RV64) emit real calls and never need this, nor
 * does Metal, which hands device functions to its own compiler.
 *
 * Straight-line callees splice in directly; ones with control flow are cloned
 * block by block, with each return turned into a branch to a continuation and
 * multiple returns merged by a phi. Nested device calls are peeled one level
 * per round until none remain. A callee too large for the bounded working set
 * is left in place with a warning rather than compiled wrongly.
 *
 * Returns BC_OK on success, or BC_ERR_VERIFY (with a message on stderr) if the
 * call graph will not converge, which means a __device__ function recurses.
 */
int bir_inline_device(bir_module_t *M);

#endif /* BARRACUDA_BIR_INLINE_H */
