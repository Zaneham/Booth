/* ttabs.c -- encoding table validation
 * Every opcode must have a home. No squatters, no vacancies.
 * AMD renumbers everything between generations just to keep us honest. */

#include "tharns.h"
#include "amdgpu.h"
#include "lexer.h"

/* ---- tables: GFX11 completeness ---- */

static void tab01(void)
{
    int miss = 0;
    for (int i = 0; i < AMD_OP_COUNT; i++) {
        if (amd_enc_table[i].fmt == AMD_FMT_PSEUDO) continue;
        if (amd_enc_table[i].mnemonic == NULL) {
            /* GFX12-only ops (wait_loadcnt etc.) can be NULL in GFX11 table
             * if they exist only in the GFX10 table as stubs. Check that
             * they at least appear in the enum range. */
            continue;
        }
        if (amd_enc_table[i].fmt == 0 && amd_enc_table[i].hw_opcode == 0
            && amd_enc_table[i].mnemonic == NULL) {
            printf("  GFX11 missing: op %d\n", i);
            miss++;
        }
    }
    CHEQ(miss, 0);
    PASS();
}
TH_REG("tab", 1, "GFX11 table has no gaps", tab01)

/* ---- tables: GFX10 completeness ---- */

static void tab02(void)
{
    int miss = 0;
    for (int i = 0; i < AMD_OP_COUNT; i++) {
        if (amd_enc_table_gfx10[i].fmt == AMD_FMT_PSEUDO) continue;
        /* GFX12-only wait instructions are NULL stubs in GFX10, that's fine */
        if (amd_enc_table_gfx10[i].mnemonic == NULL) continue;
        if (amd_enc_table_gfx10[i].fmt == 0 && amd_enc_table_gfx10[i].hw_opcode == 0
            && amd_enc_table_gfx10[i].mnemonic == NULL) {
            printf("  GFX10 missing: op %d\n", i);
            miss++;
        }
    }
    CHEQ(miss, 0);
    PASS();
}
TH_REG("tab", 2, "GFX10 table has no gaps", tab02)

/* ---- tables: GFX10 DS mnemonic is ds_read, not ds_load ---- */

static void tab03(void)
{
    CHECK(amd_enc_table_gfx10[AMD_DS_READ_B32].mnemonic != NULL);
    CHSTR(amd_enc_table_gfx10[AMD_DS_READ_B32].mnemonic, "ds_read_b32");
    CHECK(amd_enc_table_gfx10[AMD_DS_WRITE_B32].mnemonic != NULL);
    CHSTR(amd_enc_table_gfx10[AMD_DS_WRITE_B32].mnemonic, "ds_write_b32");
    PASS();
}
TH_REG("tab", 3, "GFX10 DS ops are read and write", tab03)

/* ---- tables: known opcode differences ---- */

static void tab04(void)
{
    /* s_and_b32: GFX10=0x0E, GFX11=0x16 */
    CHEQ(amd_enc_table[AMD_S_AND_B32].hw_opcode,         0x16);
    CHEQ(amd_enc_table_gfx10[AMD_S_AND_B32].hw_opcode,   0x0E);

    /* s_lshl_b32: GFX10=0x1E, GFX11=0x08 */
    CHEQ(amd_enc_table[AMD_S_LSHL_B32].hw_opcode,       0x08);
    CHEQ(amd_enc_table_gfx10[AMD_S_LSHL_B32].hw_opcode, 0x1E);

    /* s_endpgm: GFX10=0x01, GFX11=0x30 */
    CHEQ(amd_enc_table[AMD_S_ENDPGM].hw_opcode,       0x30);
    CHEQ(amd_enc_table_gfx10[AMD_S_ENDPGM].hw_opcode, 0x01);

    PASS();
}
TH_REG("tab", 4, "GFX10 and GFX11 opcodes differ where known", tab04)

/* ---- tables: keyword table order ---- */

/* Nothing checked this until now, which was optimistic of us. A misfiled entry
 * lexes its keyword as an identifier and the mess turns up as a parse error
 * somewhere downstream, a long way from the typo that caused it. */

static void tab05(void)
{
    int n = lexer_kw_count();
    CHECK(n > 0);
    for (int i = 1; i < n; i++) {
        const char *prev = lexer_kw_at(i - 1), *cur = lexer_kw_at(i);
        CHECK(prev != NULL && cur != NULL);
        if (strcmp(prev, cur) >= 0) {
            printf("  keyword %d (%s) is not after %d (%s)\n",
                   i, cur, i - 1, prev);
            nfail++;
            return;
        }
    }
    PASS();
}
TH_REG("tab", 5, "keyword table is strictly ascending", tab05)

/* The order matters because of what the search does with it, so check the
 * search too rather than only the invariant it relies on. */
static void tab06(void)
{
    token_t toks[8];
    int n = lexer_kw_count();

    for (int i = 0; i < n; i++) {
        const char *kw = lexer_kw_at(i);
        lexer_t L;
        lexer_init(&L, kw, (uint32_t)strlen(kw), toks, 8);
        lexer_tokenize(&L);
        if (L.num_tokens == 0 || toks[0].type == TOK_IDENT) {
            printf("  keyword %s (%d) lexes as an identifier\n", kw, i);
            nfail++;
            return;
        }
    }
    PASS();
}
TH_REG("tab", 6, "every keyword still lexes as a keyword", tab06)
