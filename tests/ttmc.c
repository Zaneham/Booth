/* ttmc.c -- Tensix machine-code emit.
 *
 * tensix_emit_binary writes raw 32-bit Tensix words and tensix_emit_ttinsn
 * writes the same words rol-2 encoded for the baby RISC-V push. Neither had a
 * byte-level test; they were only ever checked by hand through ttas and Kahu,
 * so a codegen change could shift an encoding while every unit test stayed
 * green. These build a small module, emit it to a temp file, read it back
 * little-endian and lock what those tools confirmed, needing neither of them so
 * it runs anywhere. emit_stream wraps the words in a sync bracket that emit.c
 * calls provisional, so the checks stay bracket-agnostic. */

#include "tharns.h"
#include "tensix.h"

/* ~20 MB of minst arena and code buffer, so it lives in BSS rather than on
 * MinGW's lazy stack. Shared: each test sets num_minsts and the slots it uses. */
static tt_module_t TT;

#define MAXW 64
#define TMP_BIN "ttmc_tmp.bin"
#define TMP_TTI "ttmc_tmp.ttinsn"

/* Zero a slot and set its opcode. With operands NONE and num_uses 0 encode_inst
 * returns exactly hw_opcode << 24 for any format, which the goldens rely on. */
static void set_op(uint32_t i, uint16_t op)
{
    memset(&TT.minsts[i], 0, sizeof(tt_minst_t));
    TT.minsts[i].op = op;
}

static void set_imm(uint32_t i, int slot, int32_t v)
{
    TT.minsts[i].operands[slot].kind = TT_MOP_IMM;
    TT.minsts[i].operands[slot].imm  = v;
}

/* Read a little-endian 32-bit word stream back off disk. */
static int read_words(const char *path, uint32_t *out, int max)
{
    FILE *fp = fopen(path, "rb");
    if (!fp) return -1;
    int n = 0;
    uint8_t b[4];
    while (n < max && fread(b, 1, 4, fp) == 4) {
        out[n++] = (uint32_t)b[0]        | ((uint32_t)b[1] << 8)
                 | ((uint32_t)b[2] << 16) | ((uint32_t)b[3] << 24);
    }
    fclose(fp);
    return n;
}

static uint32_t rol2(uint32_t w) { return (w << 2) | (w >> 30); }

static int has_word(const uint32_t *w, int n, uint32_t t)
{
    for (int i = 0; i < n; i++) if (w[i] == t) return 1;
    return 0;
}

/* Count real instructions, everything below the sync opcode range where the
 * bracket lives, so we can prove the pseudo-op went without counting the bracket. */
static int n_compute(const uint32_t *w, int n)
{
    int c = 0;
    for (int i = 0; i < n; i++) if ((w[i] >> 24) < 0xA0u) c++;
    return c;
}

/* One instruction per encoding format with operands zero, so each word is just
 * its opcode byte, and a pseudo-op in the middle that must be dropped. */
static void ttmc_bin_opcodes(void)
{
    uint32_t k = 0;
    set_op(k++, TT_SFPNOP);      /* 0x02, FMT_C */
    set_op(k++, TT_SFPMAD);      /* 0x84, FMT_A */
    set_op(k++, TT_SFPSETCC);    /* 0x7B, FMT_B */
    set_op(k++, TT_PSEUDO_COPY); /* skipped: no hardware opcode */
    set_op(k++, TT_SFPWNOP);     /* 0x8F, FMT_C */
    TT.num_minsts = k;

    CHEQ(tensix_emit_binary(&TT, TMP_BIN), BC_OK);
    uint32_t w[MAXW];
    int n = read_words(TMP_BIN, w, MAXW);
    remove(TMP_BIN);
    CHECK(n > 0);

    /* four real compute words, the pseudo-op dropped */
    CHEQ(n_compute(w, n), 4);
    CHECK(has_word(w, n, 0x02000000u));  /* sfpnop   */
    CHECK(has_word(w, n, 0x84000000u));  /* sfpmad   */
    CHECK(has_word(w, n, 0x7B000000u));  /* sfpsetcc */
    CHECK(has_word(w, n, 0x8F000000u));  /* sfpwnop  */
    PASS();
}
TH_REG("ttmc", ttmc_bin_opcodes);

/* The baby core pushes each Tensix word through a custom instruction encoded as
 * the word rol-2'd, so both emitters must agree word-for-word under rol2. */
static void ttmc_ttinsn_is_rol2(void)
{
    uint32_t k = 0;
    set_op(k++, TT_SFPNOP);
    set_op(k++, TT_SFPMOV);
    set_op(k++, TT_SFPADD);
    set_op(k++, TT_SFPMUL);
    set_op(k++, TT_SFPLOADI);
    TT.num_minsts = k;

    CHEQ(tensix_emit_binary(&TT, TMP_BIN), BC_OK);
    CHEQ(tensix_emit_ttinsn(&TT, TMP_TTI), BC_OK);
    uint32_t bin[MAXW], tti[MAXW];
    int nb = read_words(TMP_BIN, bin, MAXW);
    int nt = read_words(TMP_TTI, tti, MAXW);
    remove(TMP_BIN);
    remove(TMP_TTI);

    CHECK(nb > 0);
    CHEQ(nt, nb);
    for (int i = 0; i < nb; i++)
        CHEQX(tti[i], rol2(bin[i]));
    PASS();
}
TH_REG("ttmc", ttmc_ttinsn_is_rol2);

/* rol-2 only round-trips because every real opcode is below 0xC0000000, leaving
 * the top two bits free. Check the ceiling holds and ror-2 restores each word. */
static void ttmc_ttinsn_reversible(void)
{
    uint32_t k = 0;
    set_op(k++, TT_SFPLOAD);
    set_op(k++, TT_SFPSTORE);
    set_op(k++, TT_SFPMAD);
    set_op(k++, TT_SFPXOR);   /* 0x8D, one of the higher opcodes */
    TT.num_minsts = k;

    CHEQ(tensix_emit_binary(&TT, TMP_BIN), BC_OK);
    CHEQ(tensix_emit_ttinsn(&TT, TMP_TTI), BC_OK);
    uint32_t bin[MAXW], tti[MAXW];
    int nb = read_words(TMP_BIN, bin, MAXW);
    int nt = read_words(TMP_TTI, tti, MAXW);
    remove(TMP_BIN);
    remove(TMP_TTI);

    CHECK(nb > 0);
    CHEQ(nt, nb);
    for (int i = 0; i < nb; i++) {
        CHECK(bin[i] < 0xC0000000u);               /* ceiling holds    */
        uint32_t ror2 = (tti[i] >> 2) | (tti[i] << 30);
        CHEQX(ror2, bin[i]);                        /* round-trips back */
    }
    PASS();
}
TH_REG("ttmc", ttmc_ttinsn_reversible);

/* Pin the Sync Unit field packing, not just the opcode byte, using values
 * distinct from the bracket's own sync words so each match is unambiguous. The
 * bit layouts these goldens encode are the field definitions in emit.c. */
static void ttmc_sync_fields(void)
{
    set_op(0, TT_SEMINIT);                          /* sem_sel 3, init 1, max 2 */
    set_imm(0, 0, 3); set_imm(0, 1, 1); set_imm(0, 2, 2);
    set_op(1, TT_SEMWAIT);                          /* sem_sel 4, cond 1, stall 2 */
    set_imm(1, 0, 4); set_imm(1, 1, 1); set_imm(1, 2, 2);
    set_op(2, TT_SEMPOST);                          /* sem_sel 5 */
    set_imm(2, 0, 5);
    TT.num_minsts = 3;

    CHEQ(tensix_emit_binary(&TT, TMP_BIN), BC_OK);
    uint32_t w[MAXW];
    int n = read_words(TMP_BIN, w, MAXW);
    remove(TMP_BIN);
    CHECK(n > 0);

    /* seminit: 0xA3 | (max 2 << 20) | (init 1 << 16) | (sem_sel 3 << 2) */
    CHECK(has_word(w, n, 0xA3000000u | (2u << 20) | (1u << 16) | (3u << 2)));
    /* semwait: 0xA6 | (stall_res 2 << 15) | (sem_sel 4 << 2) | (cond 1) */
    CHECK(has_word(w, n, 0xA6000000u | (2u << 15) | (4u << 2) | 1u));
    /* sempost: 0xA4 | (sem_sel 5 << 2) */
    CHECK(has_word(w, n, 0xA4000000u | (5u << 2)));
    PASS();
}
TH_REG("ttmc", ttmc_sync_fields);
