#ifndef BARRACUDA_RV_ISEL_H
#define BARRACUDA_RV_ISEL_H

#include "barracuda.h"
#include "bir.h"
#include "rv_buf.h"
#include "tdf.h"     /* for BC_ERR_TDF */

/*
 * BIR -> RV32IM instruction selection for the Tenstorrent baby cores.
 *
 * This is the bring-up version: stack-spill every BIR value into its own
 * 4-byte slot, no register allocation, no peephole optimisation, and no
 * calling-convention sophistication beyond "the function is leaf and
 * integer-only." The generated code is correct and uniform but slow; the
 * point is to get a kernel running on the n300 first and improve once we
 * have something to measure.
 *
 * Today it handles parameters, i32 loads and stores through a pointer,
 * integer arithmetic with t0/t1 scratch, integer or void returns, and
 * small constants materialised via ADDI. It does not yet handle the
 * harder cases (calls, control flow, GEP, floats, vectors, structs), more
 * than 8 integer parameters, or constants outside the I-type 12-bit
 * signed range that would need LUI+ADDI. Spec sources are cited in the .c
 * file alongside each opcode emission.
 */

/*
 * Select instructions for one BIR function into the given code buffer.
 * Returns BC_OK or BC_ERR_TDF on a construct we don't support yet
 * (which prints a message naming the offending BIR op for triage).
 */
int rv_isel_func(const bir_module_t *M, uint32_t func_idx,
                 rv_buf_t *out);

/*
 * Select every function in the module, emitting them back-to-back
 * into the same code buffer. Used when the kernel calls into other
 * functions (e.g., a soft-float libcall) and those callees need to
 * live in the same ELF. Function 0 is emitted first by convention
 * because that is where the host loader sets the PC; everything
 * else is reachable via JAL with offsets resolved after the whole
 * module is laid out.
 *
 * Inter-function calls work by recording a patch when the JAL is
 * emitted, and filling in the offset once every callee's position
 * is known. Direct recursion is refused because we save the return
 * address only once in the prologue; turning recursion on means
 * teaching the prologue about saving ra to a deeper stack slot,
 * which is a sitting on its own.
 */
int rv_isel_module(const bir_module_t *M, rv_buf_t *out);

#endif /* BARRACUDA_RV_ISEL_H */
