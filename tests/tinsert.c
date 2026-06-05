/* tinsert.c -- isolation tests for bir_insert.
 *
 * The dangerous part of SROA (and any inserting pass) is the operand
 * remap: when instructions shift, value references must slide and block
 * references must NOT. These tests build tiny modules by hand, insert
 * an alloca, and check exactly that, with the block-reference survival
 * as the headline assertion. Proving the primitive alone, before
 * anything depends on it. */

#include "tharns.h"
#include <stdlib.h>
#include "bir.h"
#include "bir_insert.h"

/* ---- insert: value refs slide, BR block ref stays ---- */

static void insert_basic(void)
{
    bir_module_t *M = malloc(sizeof(*M));
    uint32_t i32, vt, at;
    CHECK(M != NULL);
    bir_module_init(M);
    i32 = bir_type_int(M, 32);
    vt  = bir_type_void(M);

    /* bb0: %0 = thread_id.x
     *      %1 = add %0, %0
     *      br bb1
     * bb1: %3 = add %1, %0
     *      ret %3                                              */
    M->num_insts = 5;
    M->insts[0] = (bir_inst_t){ .op = BIR_THREAD_ID, .num_operands = 0,
                                .subop = 0, .type = i32 };
    M->insts[1] = (bir_inst_t){ .op = BIR_ADD, .num_operands = 2, .type = i32 };
    M->insts[1].operands[0] = BIR_MAKE_VAL(0);
    M->insts[1].operands[1] = BIR_MAKE_VAL(0);
    M->insts[2] = (bir_inst_t){ .op = BIR_BR, .num_operands = 1, .type = vt };
    M->insts[2].operands[0] = 1;                 /* BLOCK index, not a value */
    M->insts[3] = (bir_inst_t){ .op = BIR_ADD, .num_operands = 2, .type = i32 };
    M->insts[3].operands[0] = BIR_MAKE_VAL(1);
    M->insts[3].operands[1] = BIR_MAKE_VAL(0);
    M->insts[4] = (bir_inst_t){ .op = BIR_RET, .num_operands = 1, .type = vt };
    M->insts[4].operands[0] = BIR_MAKE_VAL(3);

    M->num_blocks = 2;
    M->blocks[0] = (bir_block_t){ .name = 0, .first_inst = 0, .num_insts = 3 };
    M->blocks[1] = (bir_block_t){ .name = 0, .first_inst = 3, .num_insts = 2 };

    M->num_funcs = 1;
    M->funcs[0].first_block = 0;
    M->funcs[0].num_blocks  = 2;
    M->funcs[0].total_insts = 5;

    /* Insert one alloca at the very start of bb0. at = 0, all up by 1. */
    {
        bir_inst_t a = { .op = BIR_ALLOCA, .num_operands = 0,
                         .subop = 2, .type = 0 };
        at = bir_insert(M, 0, 0, &a, 1);
    }

    CHECK(at == 0);
    CHECK(M->num_insts == 6);
    CHECK(M->insts[0].op == BIR_ALLOCA);
    CHECK(M->insts[1].op == BIR_THREAD_ID);

    /* %1 = add %0,%0 -> idx2, operands 0 -> 1 */
    CHECK(M->insts[2].op == BIR_ADD);
    CHECK(M->insts[2].operands[0] == BIR_MAKE_VAL(1));
    CHECK(M->insts[2].operands[1] == BIR_MAKE_VAL(1));

    /* br bb1 -> block ref MUST stay 1.  THE headline. */
    CHECK(M->insts[3].op == BIR_BR);
    CHECK(M->insts[3].operands[0] == 1);

    /* %3 = add %1,%0 -> idx4, operands 1->2, 0->1 */
    CHECK(M->insts[4].op == BIR_ADD);
    CHECK(M->insts[4].operands[0] == BIR_MAKE_VAL(2));
    CHECK(M->insts[4].operands[1] == BIR_MAKE_VAL(1));

    /* ret %3 -> 3 -> 4 */
    CHECK(M->insts[5].op == BIR_RET);
    CHECK(M->insts[5].operands[0] == BIR_MAKE_VAL(4));

    CHECK(M->blocks[0].first_inst == 0 && M->blocks[0].num_insts == 4);
    CHECK(M->blocks[1].first_inst == 4 && M->blocks[1].num_insts == 2);
    CHECK(M->funcs[0].total_insts == 6);

    free(M);
    PASS();
}
TH_REG("insert", insert_basic)

/* ---- insert: PHI + BR_COND block halves survive, value halves slide ---- */

static void insert_phi_blockrefs(void)
{
    bir_module_t *M = malloc(sizeof(*M));
    uint32_t i32, vt, at;
    CHECK(M != NULL);
    bir_module_init(M);
    i32 = bir_type_int(M, 32);
    vt  = bir_type_void(M);

    /* bb0: %0 = thread_id.x
     *      br_cond %0, bb1, bb2          (value, block, block)
     * bb1: br bb3
     * bb2: br bb3
     * bb3: %4 = phi [bb1: %0, bb2: %0]   (block,value pairs)
     *      ret %4                                              */
    M->num_insts = 6;
    M->insts[0] = (bir_inst_t){ .op = BIR_THREAD_ID, .num_operands = 0,
                                .type = i32 };
    M->insts[1] = (bir_inst_t){ .op = BIR_BR_COND, .num_operands = 3,
                                .type = vt };
    M->insts[1].operands[0] = BIR_MAKE_VAL(0);   /* value */
    M->insts[1].operands[1] = 1;                 /* block bb1 */
    M->insts[1].operands[2] = 2;                 /* block bb2 */
    M->insts[2] = (bir_inst_t){ .op = BIR_BR, .num_operands = 1, .type = vt };
    M->insts[2].operands[0] = 3;                 /* block bb3 */
    M->insts[3] = (bir_inst_t){ .op = BIR_BR, .num_operands = 1, .type = vt };
    M->insts[3].operands[0] = 3;                 /* block bb3 */
    M->insts[4] = (bir_inst_t){ .op = BIR_PHI, .num_operands = 4, .type = i32 };
    M->insts[4].operands[0] = 1;                 /* block bb1 */
    M->insts[4].operands[1] = BIR_MAKE_VAL(0);   /* value */
    M->insts[4].operands[2] = 2;                 /* block bb2 */
    M->insts[4].operands[3] = BIR_MAKE_VAL(0);   /* value */
    M->insts[5] = (bir_inst_t){ .op = BIR_RET, .num_operands = 1, .type = vt };
    M->insts[5].operands[0] = BIR_MAKE_VAL(4);

    M->num_blocks = 4;
    M->blocks[0] = (bir_block_t){ .first_inst = 0, .num_insts = 2 };
    M->blocks[1] = (bir_block_t){ .first_inst = 2, .num_insts = 1 };
    M->blocks[2] = (bir_block_t){ .first_inst = 3, .num_insts = 1 };
    M->blocks[3] = (bir_block_t){ .first_inst = 4, .num_insts = 2 };
    M->num_funcs = 1;
    M->funcs[0].first_block = 0;
    M->funcs[0].num_blocks  = 4;
    M->funcs[0].total_insts = 6;

    {
        bir_inst_t a = { .op = BIR_ALLOCA, .num_operands = 0,
                         .subop = 2, .type = 0 };
        at = bir_insert(M, 0, 0, &a, 1);
    }
    CHECK(at == 0);

    /* br_cond -> idx2: value 0->1, blocks UNCHANGED */
    CHECK(M->insts[2].op == BIR_BR_COND);
    CHECK(M->insts[2].operands[0] == BIR_MAKE_VAL(1));
    CHECK(M->insts[2].operands[1] == 1);
    CHECK(M->insts[2].operands[2] == 2);

    /* phi -> idx5: block halves stay (1,2), value halves 0->1 */
    CHECK(M->insts[5].op == BIR_PHI);
    CHECK(M->insts[5].operands[0] == 1);
    CHECK(M->insts[5].operands[1] == BIR_MAKE_VAL(1));
    CHECK(M->insts[5].operands[2] == 2);
    CHECK(M->insts[5].operands[3] == BIR_MAKE_VAL(1));

    /* ret -> idx6: %4 -> %5 */
    CHECK(M->insts[6].op == BIR_RET);
    CHECK(M->insts[6].operands[0] == BIR_MAKE_VAL(5));

    CHECK(M->blocks[0].first_inst == 0 && M->blocks[0].num_insts == 3);
    CHECK(M->blocks[1].first_inst == 3);
    CHECK(M->blocks[2].first_inst == 4);
    CHECK(M->blocks[3].first_inst == 5);
    CHECK(M->funcs[0].total_insts == 7);

    free(M);
    PASS();
}
TH_REG("insert", insert_phi_blockrefs)
