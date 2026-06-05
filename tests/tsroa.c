/* tsroa.c -- isolation tests for the SROA pass.
 *
 * Positive: a struct alloca accessed only through constant-index GEPs to
 * scalar fields is split, and every load/store is repointed at the right
 * per-field scalar alloca. Negative: a struct pointer that escapes (a
 * whole-struct load) is left completely untouched. Built in-process so
 * we exercise bir_sroa directly, no binary, no .cu file. */

#include "tharns.h"
#include <stdlib.h>
#include "bir.h"
#include "bir_sroa.h"

/* ---- sroa: a 2-field struct splits, accesses repoint ---- */

static void sroa_splits(void)
{
    bir_module_t *M = malloc(sizeof(*M));
    uint32_t i32, f32, st, pst, pi32, pf32, vt, c0, c1, flds[2];
    CHECK(M != NULL);
    bir_module_init(M);

    i32  = bir_type_int(M, 32);
    f32  = bir_type_float(M, 32);
    vt   = bir_type_void(M);
    flds[0] = i32; flds[1] = f32;
    st   = bir_type_struct(M, flds, 2);
    pst  = bir_type_ptr(M, st,  BIR_AS_PRIVATE);
    pi32 = bir_type_ptr(M, i32, BIR_AS_PRIVATE);
    pf32 = bir_type_ptr(M, f32, BIR_AS_PRIVATE);
    c0   = BIR_MAKE_CONST(bir_const_int(M, i32, 0));
    c1   = BIR_MAKE_CONST(bir_const_int(M, i32, 1));

    /* bb0: %0 = alloca {i32,f32}
     *      %1 = gep %0, 0          ; &field0
     *      %2 = thread_id.x
     *      store %2, %1            ; field0 = tid
     *      %4 = gep %0, 1          ; &field1
     *      %5 = load %4            ; load field1
     *      %6 = gep %0, 0          ; &field0 again
     *      %7 = load %6            ; load field0
     *      ret                                                  */
    M->num_insts = 9;
    M->insts[0] = (bir_inst_t){ .op=BIR_ALLOCA, .num_operands=0,
                                .subop=3, .type=pst };
    M->insts[1] = (bir_inst_t){ .op=BIR_GEP, .num_operands=2, .type=pi32 };
    M->insts[1].operands[0] = BIR_MAKE_VAL(0);
    M->insts[1].operands[1] = c0;
    M->insts[2] = (bir_inst_t){ .op=BIR_THREAD_ID, .num_operands=0, .type=i32 };
    M->insts[3] = (bir_inst_t){ .op=BIR_STORE, .num_operands=2, .type=vt };
    M->insts[3].operands[0] = BIR_MAKE_VAL(2);   /* value */
    M->insts[3].operands[1] = BIR_MAKE_VAL(1);   /* addr = &field0 */
    M->insts[4] = (bir_inst_t){ .op=BIR_GEP, .num_operands=2, .type=pf32 };
    M->insts[4].operands[0] = BIR_MAKE_VAL(0);
    M->insts[4].operands[1] = c1;
    M->insts[5] = (bir_inst_t){ .op=BIR_LOAD, .num_operands=1, .type=f32 };
    M->insts[5].operands[0] = BIR_MAKE_VAL(4);   /* addr = &field1 */
    M->insts[6] = (bir_inst_t){ .op=BIR_GEP, .num_operands=2, .type=pi32 };
    M->insts[6].operands[0] = BIR_MAKE_VAL(0);
    M->insts[6].operands[1] = c0;
    M->insts[7] = (bir_inst_t){ .op=BIR_LOAD, .num_operands=1, .type=i32 };
    M->insts[7].operands[0] = BIR_MAKE_VAL(6);   /* addr = &field0 */
    M->insts[8] = (bir_inst_t){ .op=BIR_RET, .num_operands=0, .type=vt };

    M->num_blocks = 1;
    M->blocks[0] = (bir_block_t){ .first_inst=0, .num_insts=9 };
    M->num_funcs = 1;
    M->funcs[0].first_block=0;
    M->funcs[0].num_blocks=1;
    M->funcs[0].total_insts=9;

    bir_sroa(M);

    /* Two scalar allocas inserted at the front (field0=ptr i32 slot 0,
     * field1=ptr f32 slot 1); everything else shifted up by 2. */
    CHECK(M->num_insts == 11);
    CHECK(M->insts[0].op == BIR_ALLOCA && M->insts[0].type == pi32);
    CHECK(M->insts[1].op == BIR_ALLOCA && M->insts[1].type == pf32);
    CHECK(M->insts[2].op == BIR_ALLOCA && M->insts[2].type == pst); /* orphan */

    /* store (idx5): addr repointed from &field0 GEP to scalar field0 (%0) */
    CHECK(M->insts[5].op == BIR_STORE);
    CHECK(M->insts[5].operands[1] == BIR_MAKE_VAL(0));
    /* the stored VALUE is untouched, just slid: %2 -> %4 */
    CHECK(M->insts[5].operands[0] == BIR_MAKE_VAL(4));

    /* load field1 (idx7): addr -> scalar field1 (%1) */
    CHECK(M->insts[7].op == BIR_LOAD);
    CHECK(M->insts[7].operands[0] == BIR_MAKE_VAL(1));

    /* load field0 (idx9): addr -> scalar field0 (%0) */
    CHECK(M->insts[9].op == BIR_LOAD);
    CHECK(M->insts[9].operands[0] == BIR_MAKE_VAL(0));

    free(M);
    PASS();
}
TH_REG("sroa", sroa_splits)

/* ---- sroa: an escaping struct pointer is left untouched ---- */

static void sroa_escape_bails(void)
{
    bir_module_t *M = malloc(sizeof(*M));
    uint32_t i32, f32, st, pst, pi32, vt, c0, flds[2];
    CHECK(M != NULL);
    bir_module_init(M);

    i32  = bir_type_int(M, 32);
    f32  = bir_type_float(M, 32);
    vt   = bir_type_void(M);
    flds[0] = i32; flds[1] = f32;
    st   = bir_type_struct(M, flds, 2);
    pst  = bir_type_ptr(M, st,  BIR_AS_PRIVATE);
    pi32 = bir_type_ptr(M, i32, BIR_AS_PRIVATE);
    c0   = BIR_MAKE_CONST(bir_const_int(M, i32, 0));

    /* bb0: %0 = alloca {i32,f32}
     *      %1 = gep %0, 0
     *      %2 = load %1            ; legit field load
     *      %3 = load %0            ; WHOLE-STRUCT load = escape -> bail
     *      ret                                                  */
    M->num_insts = 5;
    M->insts[0] = (bir_inst_t){ .op=BIR_ALLOCA, .num_operands=0,
                                .subop=3, .type=pst };
    M->insts[1] = (bir_inst_t){ .op=BIR_GEP, .num_operands=2, .type=pi32 };
    M->insts[1].operands[0] = BIR_MAKE_VAL(0);
    M->insts[1].operands[1] = c0;
    M->insts[2] = (bir_inst_t){ .op=BIR_LOAD, .num_operands=1, .type=i32 };
    M->insts[2].operands[0] = BIR_MAKE_VAL(1);
    M->insts[3] = (bir_inst_t){ .op=BIR_LOAD, .num_operands=1, .type=st };
    M->insts[3].operands[0] = BIR_MAKE_VAL(0);   /* loads the struct ptr itself */
    M->insts[4] = (bir_inst_t){ .op=BIR_RET, .num_operands=0, .type=vt };

    M->num_blocks = 1;
    M->blocks[0] = (bir_block_t){ .first_inst=0, .num_insts=5 };
    M->num_funcs = 1;
    M->funcs[0].first_block=0;
    M->funcs[0].num_blocks=1;
    M->funcs[0].total_insts=5;

    bir_sroa(M);

    /* Nothing changed: no scalars inserted, struct alloca intact. */
    CHECK(M->num_insts == 5);
    CHECK(M->insts[0].op == BIR_ALLOCA && M->insts[0].type == pst);
    CHECK(M->insts[1].op == BIR_GEP);
    CHECK(M->insts[3].op == BIR_LOAD && M->insts[3].operands[0] == BIR_MAKE_VAL(0));

    free(M);
    PASS();
}
TH_REG("sroa", sroa_escape_bails)
