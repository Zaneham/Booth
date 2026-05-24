#ifndef BARRACUDA_RV_ISEL_H
#define BARRACUDA_RV_ISEL_H

#include "barracuda.h"
#include "bir.h"
#include "rv_buf.h"
#include "tdf.h"     /* for BC_ERR_TDF */

/*
 * BIR -> RV32IM instruction selection for the Tenstorrent baby cores.
 *
 * This is the bring-up version: stack-spill every BIR value into its
 * own 4-byte slot, no register allocation, no peephole optimisation,
 * no calling-convention sophistication beyond "the function is leaf
 * and integer-only." That means the generated code is correct and
 * uniform but slow; the point is to get a kernel running on the
 * n300 first and improve once we have something to measure.
 *
 * What it handles today:
 *   BIR_PARAM    parameter i goes from a0+i into its stack slot
 *   BIR_LOAD     i32 loads through a pointer parameter
 *   BIR_STORE    i32 stores through a pointer parameter
 *   BIR_ADD/SUB/MUL  integer arithmetic with t0/t1 scratch
 *   BIR_RET      return integer in a0, or void
 *   BIR_CONST    materialise small constants via ADDI
 *
 * What it does NOT handle (yet):
 *   BIR_CALL, control flow, GEP, floats, vectors, structs.
 *   Calling more than 8 integer parameters.
 *   Constants outside the I-type 12-bit signed range (needs LUI+ADDI).
 *
 * Spec sources cited in the .c file alongside each opcode emission.
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
