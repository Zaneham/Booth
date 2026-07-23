/* trv_elf.c -- RV32IM ELF emitter tests.
 * Write a tiny ELF, read it back, verify header fields and code
 * bytes match what the spec says they should be. */

#include "tharns.h"
#include "rv_buf.h"
#include "rv_enc.h"
#include "rv_elf.h"
#include "tdf.h"

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
    /* p_offset: 84 rounded up to the 16-byte segment alignment tt-metal
     * links with (-Wl,-z,max-page-size=16). */
    CHEQ(rd32(e_phoff + 4),  96u);
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

/*
 * The tests below encode tt-metal's loader contract from
 * tt_metal/llrt/tt_elffile.cpp. Each one stands for a specific throw in
 * ReadImage or XIPify.
 */

/* File offset of section header i. */
static uint32_t shdr(uint32_t i)
{
    return rd32(32) + i * 40u;
}

/* Index of the section named n, or 0xFFFFFFFF. */
static uint32_t sfind(const char *n)
{
    uint32_t nsec  = rd16(48);
    uint32_t strsh = shdr(rd16(50));
    uint32_t strb  = rd32(strsh + 16);
    for (uint32_t i = 0; i < nsec; i++) {
        const char *s = (const char *)&rd[strb + rd32(shdr(i))];
        if (strcmp(s, n) == 0) return i;
    }
    return 0xFFFFFFFFu;
}

/* ReadImage:498 -- "first loadable segment is not text". */
static void rv_elf_entry_is_first_seg(void)
{
    build_kernel();
    rv_elf_write(&B, ELF_OUT);
    slurp(ELF_OUT);
    CHEQ(rd32(24), rd32(rd32(28) + 8));
    PASS();
}
TH_REG("rv_enc", rv_elf_entry_is_first_seg);

/* ReadImage:472 -- p_offset, p_vaddr and p_paddr share 4-byte alignment. */
static void rv_elf_phdr_aligned(void)
{
    build_kernel();
    rv_elf_write(&B, ELF_OUT);
    slurp(ELF_OUT);
    uint32_t ph = rd32(28);
    CHEQ((rd32(ph + 4) | rd32(ph + 8) | rd32(ph + 12)) & 3u, 0u);
    PASS();
}
TH_REG("rv_enc", rv_elf_phdr_aligned);

/* ReadImage:435 -- sections and a valid nonzero shstrndx are mandatory. */
static void rv_elf_has_sections(void)
{
    build_kernel();
    rv_elf_write(&B, ELF_OUT);
    slurp(ELF_OUT);
    CHECK(rd32(32) != 0u);
    CHEQ(rd16(46), 40u);
    CHECK(rd16(50) != 0u);
    CHECK(rd16(50) < rd16(48));
    PASS();
}
TH_REG("rv_enc", rv_elf_has_sections);

/* XIPify:1039 -- "there are no relocation sections". The section must be
 * SHT_RELA, and sh_info must name an alloc section that lies inside a
 * segment, else the loader skips it and the count stays zero. */
static void rv_elf_rela_present(void)
{
    build_kernel();
    rv_elf_write(&B, ELF_OUT);
    slurp(ELF_OUT);
    uint32_t r = sfind(".rela.text");
    CHECK(r != 0xFFFFFFFFu);
    CHEQ(rd32(shdr(r) + 4), 4u);                 /* SHT_RELA */
    uint32_t tgt = rd32(shdr(r) + 28);           /* sh_info */
    CHEQ(tgt, sfind(".text"));
    CHEQ(rd32(shdr(tgt) + 8) & 2u, 2u);          /* target is SHF_ALLOC */
    CHEQ(rd32(shdr(r) + 24), sfind(".symtab")); /* sh_link */
    PASS();
}
TH_REG("rv_enc", rv_elf_rela_present);

/* TrimSegments:545 -- matched by name, SHT_PROGBITS and NOT SHF_ALLOC,
 * holding one (vma, trim_bound, size_limit) triple per segment. */
static void rv_elf_segments_meta(void)
{
    build_kernel();
    rv_elf_write(&B, ELF_OUT);
    slurp(ELF_OUT);
    uint32_t s = sfind(".segments");
    CHECK(s != 0xFFFFFFFFu);
    CHEQ(rd32(shdr(s) + 4), 1u);                 /* SHT_PROGBITS */
    CHEQ(rd32(shdr(s) + 8) & 2u, 0u);            /* not SHF_ALLOC */
    CHEQ(rd32(shdr(s) + 20), 12u);
    uint32_t off = rd32(shdr(s) + 16);
    CHEQ(rd32(off + 0), RV_ELF_LOAD_ADDR);       /* vma */
    CHEQ(rd32(off + 4), RV_ELF_LOAD_ADDR);       /* trim_bound, so no trim */
    CHEQ(rd32(off + 8), td_txtmax(td_chip()));
    PASS();
}
TH_REG("rv_enc", rv_elf_segments_meta);

/* XIPify:779 resolves relocation symbols through the symtab, so it must
 * exist and be non-alloc (WeakenDataSymbols skips alloc symtabs). */
static void rv_elf_symtab(void)
{
    build_kernel();
    rv_elf_write(&B, ELF_OUT);
    slurp(ELF_OUT);
    uint32_t y = sfind(".symtab");
    CHECK(y != 0xFFFFFFFFu);
    CHEQ(rd32(shdr(y) + 4), 2u);                 /* SHT_SYMTAB */
    CHEQ(rd32(shdr(y) + 8) & 2u, 0u);            /* not SHF_ALLOC */
    CHEQ(rd32(shdr(y) + 36), 16u);               /* sh_entsize */
    CHEQ(rd32(shdr(y) + 24), sfind(".strtab"));  /* sh_link */
    /* Entry 0 is the reserved null symbol. */
    uint32_t off = rd32(shdr(y) + 16);
    CHEQ(rd32(off + 0), 0u);
    CHEQ(rd32(off + 4), 0u);
    /* Entry 1 covers the text, and sh_info says it is the first global. */
    CHEQ(rd32(shdr(y) + 28), 1u);
    CHEQ(rd32(off + 16 + 4), RV_ELF_LOAD_ADDR);
    CHEQ(rd32(off + 16 + 8), 12u);
    CHEQ(rd[off + 16 + 12], 0x12u);              /* STB_GLOBAL | STT_FUNC */
    CHEQ(rd16(off + 16 + 14), sfind(".text"));
    PASS();
}
TH_REG("rv_enc", rv_elf_symtab);

/* ReadImage:520 -- every SHF_ALLOC section must fall inside a segment. */
static void rv_elf_alloc_in_seg(void)
{
    build_kernel();
    rv_elf_write(&B, ELF_OUT);
    slurp(ELF_OUT);
    uint32_t ph = rd32(28);
    uint32_t lo = rd32(ph + 8);
    uint32_t hi = lo + rd32(ph + 20);
    uint32_t nsec = rd16(48);
    for (uint32_t i = 1; i < nsec; i++) {
        if (!(rd32(shdr(i) + 8) & 2u)) continue;
        uint32_t a = rd32(shdr(i) + 12);
        CHECK(a >= lo && a + rd32(shdr(i) + 20) <= hi);
    }
    PASS();
}
TH_REG("rv_enc", rv_elf_alloc_in_seg);

/*
 * Structural invariants. The field-by-field tests above check that each
 * value is what we meant; these check the layout is self-consistent, which
 * is what actually breaks when the planner and the writer disagree.
 */

/* Every section body lies inside the file. SHT_NULL and SHT_NOBITS occupy
 * no file space and are exempt. */
static void rv_elf_sections_in_file(void)
{
    build_kernel();
    rv_elf_write(&B, ELF_OUT);
    long n = slurp(ELF_OUT);
    uint32_t nsec = rd16(48);
    for (uint32_t i = 1; i < nsec; i++) {
        if (rd32(shdr(i) + 4) == 0u || rd32(shdr(i) + 4) == 8u) continue;
        uint32_t off = rd32(shdr(i) + 16);
        uint32_t sz  = rd32(shdr(i) + 20);
        CHECK((long)(off + sz) <= n);
    }
    PASS();
}
TH_REG("rv_enc", rv_elf_sections_in_file);

/* The section header table itself lies inside the file. */
static void rv_elf_shdrs_in_file(void)
{
    build_kernel();
    rv_elf_write(&B, ELF_OUT);
    long n = slurp(ELF_OUT);
    CHECK((long)(rd32(32) + rd16(48) * 40u) <= n);
    PASS();
}
TH_REG("rv_enc", rv_elf_shdrs_in_file);

/* No two non-empty section bodies overlap on disk. A planner that
 * miscomputes one offset usually shows up here first. */
static void rv_elf_no_overlap(void)
{
    build_kernel();
    rv_elf_write(&B, ELF_OUT);
    slurp(ELF_OUT);
    uint32_t nsec = rd16(48);
    for (uint32_t i = 1; i < nsec; i++) {
        uint32_t ao = rd32(shdr(i) + 16), az = rd32(shdr(i) + 20);
        if (az == 0u || rd32(shdr(i) + 4) == 8u) continue;
        for (uint32_t j = i + 1u; j < nsec; j++) {
            uint32_t bo = rd32(shdr(j) + 16), bz = rd32(shdr(j) + 20);
            if (bz == 0u || rd32(shdr(j) + 4) == 8u) continue;
            CHECK(ao + az <= bo || bo + bz <= ao);
        }
    }
    PASS();
}
TH_REG("rv_enc", rv_elf_no_overlap);

/* ReadImage:509 -- alloc, rela and symtab sections need sh_offset and
 * sh_addr 4-byte aligned. */
static void rv_elf_section_align(void)
{
    build_kernel();
    rv_elf_write(&B, ELF_OUT);
    slurp(ELF_OUT);
    uint32_t nsec = rd16(48);
    for (uint32_t i = 1; i < nsec; i++) {
        uint32_t ty = rd32(shdr(i) + 4);
        int needs = (rd32(shdr(i) + 8) & 2u) || ty == 4u || ty == 2u;
        if (!needs) continue;
        CHEQ((rd32(shdr(i) + 16) | rd32(shdr(i) + 12)) & 3u, 0u);
    }
    PASS();
}
TH_REG("rv_enc", rv_elf_section_align);

/* Every section name resolves inside .shstrtab and is NUL-terminated. */
static void rv_elf_names_resolve(void)
{
    build_kernel();
    rv_elf_write(&B, ELF_OUT);
    slurp(ELF_OUT);
    uint32_t strsh = shdr(rd16(50));
    uint32_t base = rd32(strsh + 16), size = rd32(strsh + 20);
    CHEQ(rd[base + size - 1u], 0u);
    uint32_t nsec = rd16(48);
    for (uint32_t i = 0; i < nsec; i++) {
        CHECK(rd32(shdr(i)) < size);
    }
    PASS();
}
TH_REG("rv_enc", rv_elf_names_resolve);

/* .text's section header and its program header must describe the same
 * bytes, since the loader validates coverage across the two. */
static void rv_elf_text_matches_phdr(void)
{
    build_kernel();
    rv_elf_write(&B, ELF_OUT);
    slurp(ELF_OUT);
    uint32_t t = sfind(".text"), ph = rd32(28);
    CHEQ(rd32(shdr(t) + 12), rd32(ph + 8));      /* sh_addr  == p_vaddr  */
    CHEQ(rd32(shdr(t) + 16), rd32(ph + 4));      /* sh_offset == p_offset */
    CHEQ(rd32(shdr(t) + 20), rd32(ph + 16));     /* sh_size  == p_filesz */
    PASS();
}
TH_REG("rv_enc", rv_elf_text_matches_phdr);

/* The load address is named in three places and the loader checks each
 * against a different thing, so they have to agree. TrimSegments runs
 * during ReadImage and matches the link-time address, before XIPify
 * rezeros it. */
static void rv_elf_load_addr_agrees(void)
{
    build_kernel();
    rv_elf_write(&B, ELF_OUT);
    slurp(ELF_OUT);
    uint32_t segs = rd32(shdr(sfind(".segments")) + 16);
    CHEQ(rd32(24), rd32(rd32(28) + 8));          /* e_entry == p_vaddr */
    CHEQ(rd32(24), rd32(segs));                  /* e_entry == segments vma */
    PASS();
}
TH_REG("rv_enc", rv_elf_load_addr_agrees);

/* .symtab size must be a whole number of entries, and sh_info must name a
 * real one. */
static void rv_elf_symtab_wellformed(void)
{
    build_kernel();
    rv_elf_write(&B, ELF_OUT);
    slurp(ELF_OUT);
    uint32_t y = sfind(".symtab");
    uint32_t sz = rd32(shdr(y) + 20), es = rd32(shdr(y) + 36);
    CHECK(es != 0u);
    CHEQ(sz % es, 0u);
    CHECK(rd32(shdr(y) + 28) <= sz / es);
    PASS();
}
TH_REG("rv_enc", rv_elf_symtab_wellformed);

/* A one-instruction kernel still has to satisfy every alignment rule; the
 * smallest input is where padding bugs hide. */
static void rv_elf_minimal_kernel(void)
{
    rv_buf_init(&B);
    rv_buf_emit(&B, rv_nop());
    CHEQ(rv_elf_write(&B, ELF_OUT), BC_OK);
    long n = slurp(ELF_OUT);
    uint32_t ph = rd32(28);
    CHEQ(rd32(ph + 16), 4u);
    CHEQ((rd32(ph + 4) | rd32(ph + 8) | rd32(ph + 12)) & 3u, 0u);
    CHECK((long)(rd32(32) + rd16(48) * 40u) <= n);
    PASS();
}
TH_REG("rv_enc", rv_elf_minimal_kernel);

/* Text size tracks the buffer rather than a fixed guess. */
static void rv_elf_size_tracks_code(void)
{
    rv_buf_init(&B);
    for (int i = 0; i < 64; i++) rv_buf_emit(&B, rv_nop());
    CHEQ(rv_elf_write(&B, ELF_OUT), BC_OK);
    slurp(ELF_OUT);
    CHEQ(rd32(rd32(28) + 16), 256u);
    CHEQ(rd32(shdr(sfind(".text")) + 20), 256u);
    PASS();
}
TH_REG("rv_enc", rv_elf_size_tracks_code);

/*
 * End-to-end: drive the real compiler binary over a real .cu and validate
 * what lands on disk. Everything above builds its input by hand, so none of
 * it covers main.c's wiring, which is where the worst bug in this backend
 * lived. Compiling in-process would not have caught it either.
 */

/* system() goes through cmd.exe on Windows, which does not search the cwd,
 * so the path has to be explicit and backslashed. */
#ifdef _WIN32
#define KATH ".\\kath"
#define QUIET "2>nul"
#else
#define KATH "./kath"
#define QUIET "2>/dev/null"
#endif

#define E2E_OUT "e2e_rv.elf"

/* Compile a fixture with --rv-elf. Returns the binary's exit status. */
static int compile_cu(const char *cu)
{
    char cmd[512];
    snprintf(cmd, sizeof(cmd), "%s --rv-elf -o %s tests/%s %s",
             KATH, E2E_OUT, cu, QUIET);
    return system(cmd);
}

/* File offset and word count of .text in the slurped image. */
static void text_span(uint32_t *off, uint32_t *nwords)
{
    uint32_t t = sfind(".text");
    *off = rd32(shdr(t) + 16);
    *nwords = rd32(shdr(t) + 20) / 4u;
}

/* A kernel with __device__ helpers compiles, and the result satisfies the
 * same loader contract the synthetic tests check. */
static void e2e_rv_device_calls(void)
{
    CHEQ(compile_cu("device_calls.cu"), 0);
    long n = slurp(E2E_OUT);
    CHECK(n > 52);
    CHEQ(rd[0], 0x7Fu);
    CHEQ(rd16(18), 243u);                        /* EM_RISCV */
    CHECK(sfind(".segments") != 0xFFFFFFFFu);
    CHECK(sfind(".rela.text") != 0xFFFFFFFFu);
    CHECK(sfind(".symtab") != 0xFFFFFFFFu);
    CHEQ(rd32(24), rd32(rd32(28) + 8));          /* e_entry == p_vaddr */
    PASS();
}
TH_REG("rv_enc", e2e_rv_device_calls);

/* An unpatched call placeholder is a zero word, which is an illegal RV32
 * instruction. main.c used to call rv_isel_func, which records call
 * patches but never resolves them. */
static void e2e_rv_calls_patched(void)
{
    CHEQ(compile_cu("device_calls.cu"), 0);
    slurp(E2E_OUT);
    uint32_t off, nw;
    text_span(&off, &nw);
    CHECK(nw > 0);
    int jal = 0;
    for (uint32_t i = 0; i < nw; i++) {
        uint32_t w = rd32(off + i * 4u);
        CHECK(w != 0u);
        if ((w & 0x7Fu) == 0x6Fu) jal++;
    }
    CHECK(jal > 0);
    PASS();
}
TH_REG("rv_enc", e2e_rv_calls_patched);

/*
 * The entry point must be the __global__ kernel. Source order puts the
 * __device__ helpers first, so emitting functions in index order made the
 * ELF enter sq() instead of the kernel: a valid ELF running the wrong
 * program, with no diagnostic.
 *
 * device_calls.cu's kernel calls helpers and every helper is a leaf, so
 * "the first function contains a JAL" distinguishes them.
 */
static void e2e_rv_kernel_is_entry(void)
{
    CHEQ(compile_cu("device_calls.cu"), 0);
    slurp(E2E_OUT);
    uint32_t off, nw;
    text_span(&off, &nw);
    int jal = 0;
    uint32_t i = 0;
    for (; i < nw; i++) {
        uint32_t w = rd32(off + i * 4u);
        if ((w & 0x7Fu) == 0x6Fu) jal++;
        if (w == 0x00008067u) break;             /* jalr zero, ra, 0 */
    }
    CHECK(i < nw);                               /* entry function returns */
    CHECK(jal > 0);                              /* and it is not a leaf */
    PASS();
}
TH_REG("rv_enc", e2e_rv_kernel_is_entry);

/* Text is a whole number of 4-byte instructions and fits a baby core. */
static void e2e_rv_text_sane(void)
{
    CHEQ(compile_cu("device_calls.cu"), 0);
    slurp(E2E_OUT);
    uint32_t t = sfind(".text");
    uint32_t sz = rd32(shdr(t) + 20);
    CHEQ(sz % 4u, 0u);
    CHECK(sz > 0u && sz <= td_txtmax(td_chip()));
    PASS();
}
TH_REG("rv_enc", e2e_rv_text_sane);

/* Switch lowers to a compare chain, one BEQ per case, falling through to an
 * unconditional jump to the default block. */
static void e2e_rv_switch_chain(void)
{
    CHEQ(compile_cu("rv_switch.cu"), 0);
    slurp(E2E_OUT);
    uint32_t off, nw;
    text_span(&off, &nw);
    int beq = 0, jal = 0;
    for (uint32_t i = 0; i < nw; i++) {
        uint32_t w = rd32(off + i * 4u);
        CHECK(w != 0u);
        if ((w & 0x7Fu) == 0x63u && ((w >> 12) & 7u) == 0u) beq++;
        if ((w & 0x7Fu) == 0x6Fu) jal++;
    }
    CHEQ(beq, 3);
    CHECK(jal > 0);
    PASS();
}
TH_REG("rv_enc", e2e_rv_switch_chain);

/* __shared__ resolves to an absolute address in the L1 slab at the top of
 * memory, materialised with a LUI whose upper immediate is the slab base. */
static void e2e_rv_shared_addr(void)
{
    CHEQ(compile_cu("rv_shared.cu"), 0);
    slurp(E2E_OUT);
    uint32_t off, nw;
    text_span(&off, &nw);
    int found = 0;
    for (uint32_t i = 0; i < nw; i++) {
        uint32_t w = rd32(off + i * 4u);
        if ((w & 0x7Fu) == 0x37u && ((w >> 12) << 12) == td_shbase(td_chip()))
            found = 1;
    }
    CHECK(found);
    PASS();
}
TH_REG("rv_enc", e2e_rv_shared_addr);

/* The shared slab and the circular buffers must not overlap, which is only
 * true while the placer's ceiling is the slab base. */
static void rv_l1_shared_below_end(void)
{
    CHECK(td_shbase(td_chip()) + TD_L1_SHARED_SIZE == td_l1end(td_chip()));
    CHECK(TD_L1_CB_BASE < td_shbase(td_chip()));
    PASS();
}
TH_REG("rv_enc", rv_l1_shared_below_end);
