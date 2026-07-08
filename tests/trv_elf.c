/* trv_elf.c -- RV32IM ELF emitter tests.
 * Write a tiny ELF, read it back, verify header fields and code
 * bytes match what the spec says they should be. */

#include "tharns.h"
#include "rv_buf.h"
#include "rv_enc.h"
#include "rv_elf.h"

#define ELF_OUT  "tdf_test_kernel.elf"

static rv_buf_t B;
static uint8_t  rd[RV_BUF_MAX_WORDS * 4u + 256u];

/* Read whole file into rd, return size or -1 on error. */
static long slurp(const char *path)
{
    FILE *fp = fopen(path, "rb");
    if (!fp) return -1;
    long n = (long)fread(rd, 1, sizeof(rd), fp);
    fclose(fp);
    return n;
}

static uint16_t rd16(uint32_t off)
{
    return (uint16_t)(rd[off] | (rd[off + 1] << 8));
}

static uint32_t rd32(uint32_t off)
{
    return  (uint32_t)rd[off]
         | ((uint32_t)rd[off + 1] << 8)
         | ((uint32_t)rd[off + 2] << 16)
         | ((uint32_t)rd[off + 3] << 24);
}

/* Build a trivial three-instruction kernel (addi a0, zero, 42; addi a1,
 * zero, 7; add a0, a0, a1), with no store, just enough body for the ELF
 * emitter to write. */

static void build_kernel(void)
{
    rv_buf_init(&B);
    rv_buf_emit(&B, rv_addi(RV_A0, RV_ZERO, 42));
    rv_buf_emit(&B, rv_addi(RV_A1, RV_ZERO, 7));
    rv_buf_emit(&B, rv_add (RV_A0, RV_A0, RV_A1));
}

/* ---- magic bytes are present ---- */

static void rv_elf_magic(void)
{
    build_kernel();
    CHEQ(rv_elf_write(&B, ELF_OUT), BC_OK);
    long n = slurp(ELF_OUT);
    CHECK(n > 52);
    CHEQ(rd[0], 0x7Fu);
    CHEQ(rd[1], (uint8_t)'E');
    CHEQ(rd[2], (uint8_t)'L');
    CHEQ(rd[3], (uint8_t)'F');
    PASS();
}
TH_REG("rv_enc", rv_elf_magic);

/* ---- e_ident says 32-bit LE, EV_CURRENT, no OSABI ---- */

static void rv_elf_ident(void)
{
    build_kernel();
    rv_elf_write(&B, ELF_OUT);
    slurp(ELF_OUT);
    CHEQ(rd[4], 1u);   /* ELFCLASS32 */
    CHEQ(rd[5], 1u);   /* ELFDATA2LSB */
    CHEQ(rd[6], 1u);   /* EV_CURRENT */
    CHEQ(rd[7], 0u);   /* ELFOSABI_NONE */
    PASS();
}
TH_REG("rv_enc", rv_elf_ident);

/* ---- e_type, e_machine, e_version ---- */

static void rv_elf_type_machine(void)
{
    build_kernel();
    rv_elf_write(&B, ELF_OUT);
    slurp(ELF_OUT);
    CHEQ(rd16(16), 2u);     /* ET_EXEC */
    CHEQ(rd16(18), 243u);   /* EM_RISCV */
    CHEQ(rd32(20), 1u);     /* e_version */
    PASS();
}
TH_REG("rv_enc", rv_elf_type_machine);

/* ---- e_flags is zero (soft-float ABI, no compressed) ---- */

static void rv_elf_flags_soft(void)
{
    build_kernel();
    rv_elf_write(&B, ELF_OUT);
    slurp(ELF_OUT);
    CHEQ(rd32(36), 0u);
    PASS();
}
TH_REG("rv_enc", rv_elf_flags_soft);

/* ---- e_entry matches the documented load address ---- */

static void rv_elf_entry_addr(void)
{
    build_kernel();
    rv_elf_write(&B, ELF_OUT);
    slurp(ELF_OUT);
    CHEQ(rd32(24), RV_ELF_LOAD_ADDR);
    PASS();
}
TH_REG("rv_enc", rv_elf_entry_addr);

/* ---- PT_LOAD program header points at the code bytes ---- */

static void rv_elf_pt_load(void)
{
    build_kernel();
    rv_elf_write(&B, ELF_OUT);
    slurp(ELF_OUT);
    /* e_phoff is at offset 28, expected to be 52 (right after ehdr). */
    uint32_t e_phoff = rd32(28);
    CHEQ(e_phoff, 52u);
    /* First PT_LOAD: p_type at e_phoff. */
    CHEQ(rd32(e_phoff + 0),  1u);              /* PT_LOAD */
    /* p_offset: where code starts in the file, right after the
     * 32-byte program header at 84. */
    CHEQ(rd32(e_phoff + 4),  52u + 32u);
    /* p_vaddr equals the documented load address. */
    CHEQ(rd32(e_phoff + 8),  RV_ELF_LOAD_ADDR);
    /* p_filesz == p_memsz == 12 bytes (3 instructions). */
    CHEQ(rd32(e_phoff + 16), 12u);
    CHEQ(rd32(e_phoff + 20), 12u);
    /* p_flags = PF_R | PF_X = 5. */
    CHEQ(rd32(e_phoff + 24), 5u);
    PASS();
}
TH_REG("rv_enc", rv_elf_pt_load);

/* ---- code bytes appear at the offset the program header claims ---- */

static void rv_elf_code_bytes(void)
{
    build_kernel();
    rv_elf_write(&B, ELF_OUT);
    slurp(ELF_OUT);
    uint32_t code_off = rd32(rd32(28) + 4);    /* p_offset */
    uint32_t w0 = rd32(code_off + 0);
    uint32_t w1 = rd32(code_off + 4);
    uint32_t w2 = rd32(code_off + 8);
    CHEQ(w0, rv_addi(RV_A0, RV_ZERO, 42));
    CHEQ(w1, rv_addi(RV_A1, RV_ZERO, 7));
    CHEQ(w2, rv_add (RV_A0, RV_A0, RV_A1));
    PASS();
}
TH_REG("rv_enc", rv_elf_code_bytes);

/* ---- section header string table contains the section names ---- */

static void rv_elf_shstrtab(void)
{
    build_kernel();
    rv_elf_write(&B, ELF_OUT);
    long n = slurp(ELF_OUT);
    CHECK(n > 0);
    /* The string table is near the end of the file. Scan rd for
     * the substrings; both should be present once. */
    int found_text = 0, found_shstr = 0;
    for (long i = 0; i + 5 < n; i++) {
        if (memcmp(&rd[i], ".text", 5) == 0) found_text = 1;
        if (i + 9 < n && memcmp(&rd[i], ".shstrtab", 9) == 0)
            found_shstr = 1;
    }
    CHECK(found_text);
    CHECK(found_shstr);
    PASS();
}
TH_REG("rv_enc", rv_elf_shstrtab);

/* ---- empty buffer refused ---- */

static void rv_elf_empty(void)
{
    rv_buf_init(&B);
    CHEQ(rv_elf_write(&B, ELF_OUT), BC_ERR_IO);
    PASS();
}
TH_REG("rv_enc", rv_elf_empty);
