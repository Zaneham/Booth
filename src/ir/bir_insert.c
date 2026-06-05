/* bir_insert.c -- safe instruction insertion with operand remap.
 *
 * The shift is the easy part. The danger is the operand remap, and the
 * one thing that makes it safe is knowing, per opcode, which operand
 * slots are value references (slide them) versus block indices, the
 * CALL callee, or constants (leave them). That classification is taken
 * straight from the printer (bir_print.c): every slot it sends to
 * print_block_label is a block and stays put; every print_val slot is a
 * value and moves. Keep the two in agreement and this never corrupts a
 * module. */

#include "bir_insert.h"
#include <string.h>

/* Remap one operand VALUE reference: a non-const, non-none instruction
 * index at or past the insertion point slides up by n. Constants,
 * BIR_VAL_NONE, and indices before the point are untouched. Applied
 * ONLY to slots the per-opcode walk has decided are value references,
 * so a block index or function index never reaches it. */
static uint32_t rv(uint32_t v, uint32_t at, uint32_t n)
{
    uint32_t idx;
    if (v == BIR_VAL_NONE || BIR_VAL_IS_CONST(v))
        return v;
    idx = BIR_VAL_INDEX(v);
    return (idx >= at) ? BIR_MAKE_VAL(idx + n) : v;
}

/* Remap the value references of one instruction. The overflow form
 * keeps operands[0]=extra-base and operands[1]=extra-count; those are
 * indices into extra_operands[], not instruction indices, so they are
 * never touched by insertion and never remapped. */
static void remap_inst(bir_module_t *M, bir_inst_t *I, uint32_t at, uint32_t n)
{
    int      ovf   = (I->num_operands == BIR_OPERANDS_OVERFLOW);
    uint32_t start = I->operands[0];   /* extra base,  when overflow */
    uint32_t count = I->operands[1];   /* extra count, when overflow */
    uint32_t i;
    uint8_t  k, no;

    switch (I->op) {

    case BIR_BR:                       /* ops[0] = block */
        break;

    case BIR_BR_COND:                  /* ops[0]=cond value, [1..3]=blocks */
        I->operands[0] = rv(I->operands[0], at, n);
        break;

    case BIR_SWITCH:
        if (ovf) {                     /* extra: val, default-blk, (const,blk)* */
            if (count >= 1)
                M->extra_operands[start] =
                    rv(M->extra_operands[start], at, n);
            /* default block, case constants, case blocks: all left as-is */
        } else {                       /* ops[0]=val, ops[1]=default block */
            I->operands[0] = rv(I->operands[0], at, n);
        }
        break;

    case BIR_PHI:                      /* (block, value) pairs */
        if (ovf) {
            for (i = 0; i + 1 < count; i += 2)
                M->extra_operands[start + i + 1] =
                    rv(M->extra_operands[start + i + 1], at, n);
        } else {
            for (k = 0; (uint32_t)k + 1 < I->num_operands; k += 2)
                I->operands[k + 1] = rv(I->operands[k + 1], at, n);
        }
        break;

    case BIR_CALL:                     /* callee func index first, then args */
        if (ovf) {                     /* extra[start]=func, rest=args */
            for (i = 1; i < count; i++)
                M->extra_operands[start + i] =
                    rv(M->extra_operands[start + i], at, n);
        } else {                       /* ops[0]=func, ops[1..]=args */
            for (k = 1; k < I->num_operands; k++)
                I->operands[k] = rv(I->operands[k], at, n);
        }
        break;

    default:
        /* Every other opcode carries only value operands inline.
         * (ret, load, store, gep, arithmetic, select, atomics, mfma,
         * inline_asm, ...). None use the overflow form today, but if
         * one ever does, its extra slab is treated as values too. */
        if (!ovf) {
            no = I->num_operands;
            if (no > BIR_OPERANDS_INLINE) no = BIR_OPERANDS_INLINE;
            for (k = 0; k < no; k++)
                I->operands[k] = rv(I->operands[k], at, n);
        } else {
            for (i = 0; i < count; i++)
                M->extra_operands[start + i] =
                    rv(M->extra_operands[start + i], at, n);
        }
        break;
    }
}

uint32_t bir_insert(bir_module_t *M, uint32_t block_idx, uint32_t pos,
                    const bir_inst_t *src, uint32_t n)
{
    bir_block_t *TB;
    uint32_t at, tail, b, f, i;

    if (!M || !src || n == 0)                 return BIR_VAL_NONE;
    if (block_idx >= M->num_blocks)           return BIR_VAL_NONE;
    if (M->num_insts + n > BIR_MAX_INSTS)     return BIR_VAL_NONE;

    TB = &M->blocks[block_idx];
    if (pos > TB->num_insts)                  return BIR_VAL_NONE;

    at   = TB->first_inst + pos;
    tail = M->num_insts - at;          /* instructions to shift up */

    /* 1. Shift the tail of insts[] and inst_lines[] up by n. */
    if (tail) {
        memmove(&M->insts[at + n], &M->insts[at],
                (size_t)tail * sizeof(bir_inst_t));
        memmove(&M->inst_lines[at + n], &M->inst_lines[at],
                (size_t)tail * sizeof(M->inst_lines[0]));
    }

    /* 2. Drop the new instructions into the gap. */
    memcpy(&M->insts[at], src, (size_t)n * sizeof(bir_inst_t));
    for (i = 0; i < n; i++)
        M->inst_lines[at + i] = 0;
    M->num_insts += n;

    /* 3. Block bookkeeping: the target block grows; any block that
     *    started at or after the insertion point slides up. The target
     *    started at or before `at`, so it is excluded from the slide. */
    for (b = 0; b < M->num_blocks; b++) {
        if (b == block_idx)
            M->blocks[b].num_insts += n;
        else if (M->blocks[b].first_inst >= at)
            M->blocks[b].first_inst += n;
    }

    /* 4. The owning function's instruction tally grows by n. */
    for (f = 0; f < M->num_funcs; f++) {
        bir_func_t *F = &M->funcs[f];
        if (block_idx >= F->first_block &&
            block_idx <  F->first_block + F->num_blocks) {
            F->total_insts += n;
            break;
        }
    }

    /* 5. Remap every value reference that pointed past `at`. We walk
     *    ALL instructions: one before the insertion point can still
     *    reference an instruction after it. The freshly-inserted ones
     *    carry no value operands, so walking them is a no-op. */
    for (i = 0; i < M->num_insts; i++)
        remap_inst(M, &M->insts[i], at, n);

    return at;
}
