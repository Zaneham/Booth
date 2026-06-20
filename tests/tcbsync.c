/* tcbsync.c -- Tensix circular-buffer synchronisation primitive tests.
 * The signal is a NoC atomic increment of a remote L1 counter; the wait is a
 * local spin. Check the emitted RV32I against the rv_enc encoders so the test
 * documents the exact instruction stream the hardware would fetch. */

#include "tharns.h"
#include "rv_buf.h"
#include "rv_enc.h"
#include "noc.h"
#include "tensix.h"
#include <stdlib.h>
#include <string.h>

/* NIU register file base, high 20 bits (the lui immediate). 0xFFB20000. */
#define NIU_LUI 0xFFB20u

static rv_buf_t B;

/* Find the first word equal to `w`, or -1. */
static int find_word(uint32_t w)
{
    const uint32_t *d = rv_buf_data(&B);
    uint32_t n = rv_buf_n_words(&B);
    for (uint32_t i = 0; i < n; i++) if (d[i] == w) return (int)i;
    return -1;
}

/* ---- the signal programs the NIU for a full-width atomic increment ---- */

static void cb_sem_inc_encodes_atomic(void)
{
    rv_buf_init(&B);
    /* increment counter at remote 0x00009000, coords 0x00001234, by 1. */
    CHEQ(tt_sem_inc(&B, 0x00009000u, 0x00001234u, 1u), BC_OK);

    const uint32_t *d = rv_buf_data(&B);
    uint32_t n = rv_buf_n_words(&B);

    /* base register set up first. */
    CHEQ(d[0], rv_lui(RV_T0, NIU_LUI));

    /* NOC_AT_LEN_BE (offset 0x20) is loaded with 0x107C: op=1 in [15:12]
     * (lui t1,1 ; addi t1,t1,0x7C) then stored. */
    int s = find_word(rv_sw(RV_T1, RV_T0, 0x20));
    CHECK(s >= 2);
    CHEQ(d[s - 2], rv_lui (RV_T1, 1u));
    CHEQ(d[s - 1], rv_addi(RV_T1, RV_T1, 0x7C));

    /* request type NOC_CMD_AT (1) stored to NOC_CTRL (0x1C). */
    CHECK(find_word(rv_sw(RV_T1, RV_T0, 0x1C)) >= 0);

    /* last thing done is fire: write 1 to NOC_CMD_CTRL (0x28). */
    CHEQ(d[n - 2], rv_addi(RV_T1, RV_X0, 1));
    CHEQ(d[n - 1], rv_sw  (RV_T1, RV_T0, 0x28));
    PASS();
}
TH_REG("tensix", cb_sem_inc_encodes_atomic);

/* ---- the wait is a two-instruction spin that branches back on itself ---- */

static void cb_wait_is_spin_loop(void)
{
    rv_buf_init(&B);
    CHEQ(tt_sem_wait_ge(&B, 0x00009000u, 4u), BC_OK);

    const uint32_t *d = rv_buf_data(&B);
    uint32_t n = rv_buf_n_words(&B);

    /* loop body: reload the counter, branch back to the load if it is
     * still below the threshold. The branch displacement is -4 (one word). */
    CHEQ(d[n - 2], rv_lw  (RV_T1, RV_T0, 0));
    CHEQ(d[n - 1], rv_bltu(RV_T1, RV_T2, -4));
    PASS();
}
TH_REG("tensix", cb_wait_is_spin_loop);

/* ---- the CB ops are the wait/signal pair under pipeline names ---- */

static void cb_push_is_sem_inc(void)
{
    static rv_buf_t A;
    rv_buf_init(&A);
    rv_buf_init(&B);
    tt_cb_push_back(&A, 0x9000u, 0x1234u, 2u);
    tt_sem_inc     (&B, 0x9000u, 0x1234u, 2u);
    uint32_t n = rv_buf_n_words(&A);
    CHEQ(n, rv_buf_n_words(&B));
    CHECK(memcmp(rv_buf_data(&A), rv_buf_data(&B), n * 4u) == 0);
    PASS();
}
TH_REG("tensix", cb_push_is_sem_inc);

static void cb_wait_front_is_acquire(void)
{
    static rv_buf_t A;
    rv_buf_init(&A);
    rv_buf_init(&B);
    /* wait_front acquires a credit: spin until >= n, then atomic -= n. */
    tt_cb_wait_front(&A, 0x9000u, 3u);
    tt_sem_acquire  (&B, 0x9000u, 3u);
    uint32_t n = rv_buf_n_words(&A);
    CHEQ(n, rv_buf_n_words(&B));
    CHECK(memcmp(rv_buf_data(&A), rv_buf_data(&B), n * 4u) == 0);
    PASS();
}
TH_REG("tensix", cb_wait_front_is_acquire);

/* ---- acquire = wait_ge followed by an atomic decrement ---- */

static void acquire_waits_then_decrements(void)
{
    static rv_buf_t A;
    rv_buf_init(&A);
    rv_buf_init(&B);
    tt_sem_acquire(&A, 0x9000u, 1u);
    tt_sem_wait_ge(&B, 0x9000u, 1u);
    tt_sem_inc    (&B, 0x9000u, 0u, 0xFFFFFFFFu);   /* atomic -1 */
    uint32_t n = rv_buf_n_words(&A);
    CHEQ(n, rv_buf_n_words(&B));
    CHECK(memcmp(rv_buf_data(&A), rv_buf_data(&B), n * 4u) == 0);
    PASS();
}
TH_REG("tensix", acquire_waits_then_decrements);

/* ---- compute weave brackets the issue stream with the CB handshake ---- */

static void compute_weave_brackets_issue(void)
{
    tt_module_t *m = (tt_module_t *)malloc(sizeof *m);
    tt_compute_sync_t s;
    uint32_t n, last;

    CHECK(m != NULL);
    memset(m, 0, sizeof *m);
    /* two real Tensix ops as the body. */
    m->minsts[0].op = TT_SFPADD; m->minsts[0].fmt = TT_FMT_A;
    m->minsts[1].op = TT_SFPMUL; m->minsts[1].fmt = TT_FMT_A;
    m->num_minsts = 2u;

    memset(&s, 0, sizeof s);
    s.ntiles_addr   = 0x8030u;
    s.in_recv_addr  = 0xA100u;
    s.in_free_lo    = 0x9000u;
    s.out_free_addr = 0xB100u;
    s.out_recv_lo   = 0x9100u;
    s.out_depth     = 2u;

    rv_buf_init(&B);
    CHEQ(tensix_emit_compute_rv(m, &B, &s), BC_OK);

    n = rv_buf_n_words(&B);
    CHECK(n > 8u);
    /* a CB wait is a spin: lw then bltu back by one word. */
    CHECK(find_word(rv_bltu(RV_T1, RV_T2, -4)) >= 0);
    /* a CB signal is an atomic increment: NOC_AT_LEN_BE = 0x107C materialised. */
    CHECK(find_word(rv_addi(RV_T1, RV_T1, 0x7C)) >= 0);
    /* the loop has an exit branch: beq <reg>, zero, done. */
    {
        int found_exit = 0;
        uint32_t k;
        for (k = 0; k < n; k++) {
            uint32_t w = rv_buf_data(&B)[k];
            if ((w & 0x7fu) == 0x63u             /* BRANCH        */
             && ((w >> 12) & 7u) == 0u           /* funct3 = beq  */
             && ((w >> 20) & 0x1fu) == 0u) {     /* rs2 = zero    */
                found_exit = 1; break;
            }
        }
        CHECK(found_exit);
    }
    /* and the loop closes with a back jump. */
    last = rv_buf_data(&B)[n - 1u];
    CHEQ(last & 0x7fu, 0x6fu);                         /* JAL  */
    CHEQ((last >> 7) & 0x1fu, 0u);                     /* rd == zero */
    free(m);
    PASS();
}
TH_REG("tensix", compute_weave_brackets_issue);
