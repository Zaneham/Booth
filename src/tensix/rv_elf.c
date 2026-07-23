#include "rv_elf.h"
#include <stdio.h>
#include <string.h>

/*
 * ELF32 little-endian writer. Constants are inlined rather than pulled from
 * elf.h because MinGW has no portable one. All fields are written
 * little-endian; a big-endian host would need byte-swizzling.
 */

/* ---- Constants ---- */

#define ELF_MAGIC0     0x7Fu
#define ELF_MAGIC1     'E'
#define ELF_MAGIC2     'L'
#define ELF_MAGIC3     'F'

#define ELFCLASS32     1u
#define ELFDATA2LSB    1u
#define EV_CURRENT     1u
#define ELFOSABI_NONE  0u

#define ET_EXEC        2u
#define EM_RISCV       243u

#define PT_LOAD        1u
#define PF_X           1u
#define PF_R           4u

#define SHT_NULL       0u
#define SHT_PROGBITS   1u
#define SHT_SYMTAB     2u
#define SHT_STRTAB     3u
#define SHT_RELA       4u

#define SHF_ALLOC      2u
#define SHF_EXECINSTR  4u

#define STB_GLOBAL     1u
#define STT_FUNC       2u

#define EHDR_SIZE      52u
#define PHDR_SIZE      32u
#define SHDR_SIZE      40u
#define SYM_SIZE       16u
#define RELA_SIZE      12u

/* Section header indices, in the order we emit them. */
#define SEC_NULL       0u
#define SEC_TEXT       1u
#define SEC_RELA       2u
#define SEC_SYMTAB     3u
#define SEC_STRTAB     4u
#define SEC_SEGS       5u
#define SEC_SHSTR      6u
#define SEC_COUNT      7u

/* ---- String tables ---- */

static const char k_shstr[] =
    "\0.text\0.rela.text\0.symtab\0.strtab\0.segments\0.shstrtab\0";
#define SHSTR_SIZE      54u
#define SHSTR_TEXT      1u
#define SHSTR_RELA      7u
#define SHSTR_SYMTAB    18u
#define SHSTR_STRTAB    26u
#define SHSTR_SEGS      34u
#define SHSTR_SHSTR     44u

static const char k_str[] = "\0_start\0";
#define STR_SIZE        8u
#define STR_START       1u

/* ---- Little-endian writers ---- */

static void w8(uint8_t **p, uint8_t v) { **p = v; (*p)++; }

static void w16(uint8_t **p, uint16_t v)
{
    (*p)[0] = (uint8_t)(v & 0xFFu);
    (*p)[1] = (uint8_t)((v >> 8) & 0xFFu);
    *p += 2;
}

static void w32(uint8_t **p, uint32_t v)
{
    (*p)[0] = (uint8_t)(v         & 0xFFu);
    (*p)[1] = (uint8_t)((v >> 8)  & 0xFFu);
    (*p)[2] = (uint8_t)((v >> 16) & 0xFFu);
    (*p)[3] = (uint8_t)((v >> 24) & 0xFFu);
    *p += 4;
}

static uint32_t alignup(uint32_t x, uint32_t a)
{
    return (x + (a - 1u)) & ~(a - 1u);
}

/* ---- Header builders ---- */

static void w_ehdr(uint8_t **p, uint32_t shoff)
{
    w8(p, ELF_MAGIC0);
    w8(p, ELF_MAGIC1);
    w8(p, ELF_MAGIC2);
    w8(p, ELF_MAGIC3);
    w8(p, ELFCLASS32);
    w8(p, ELFDATA2LSB);
    w8(p, EV_CURRENT);
    w8(p, ELFOSABI_NONE);
    for (int i = 0; i < 8; i++) w8(p, 0);

    w16(p, (uint16_t)ET_EXEC);
    w16(p, (uint16_t)EM_RISCV);
    w32(p, EV_CURRENT);
    /* Must equal the first PT_LOAD's p_vaddr or ReadImage rejects the image. */
    w32(p, RV_ELF_ENTRY);
    w32(p, EHDR_SIZE);
    w32(p, shoff);
    w32(p, 0);                        /* e_flags; the loader never reads it */
    w16(p, (uint16_t)EHDR_SIZE);
    w16(p, (uint16_t)PHDR_SIZE);
    w16(p, 1u);
    w16(p, (uint16_t)SHDR_SIZE);
    w16(p, (uint16_t)SEC_COUNT);
    w16(p, (uint16_t)SEC_SHSTR);
}

static void w_phdr(uint8_t **p, uint32_t off, uint32_t sz)
{
    w32(p, PT_LOAD);
    w32(p, off);
    w32(p, RV_ELF_LOAD_ADDR);
    w32(p, RV_ELF_LOAD_ADDR);         /* LMA equals VMA; no MMU */
    w32(p, sz);
    w32(p, sz);
    w32(p, PF_R | PF_X);
    w32(p, RV_ELF_SEG_ALIGN);
}

static void w_shdr(uint8_t **p, uint32_t name, uint32_t type, uint32_t flags,
                   uint32_t addr, uint32_t off, uint32_t sz, uint32_t link,
                   uint32_t info, uint32_t align, uint32_t entsz)
{
    w32(p, name);
    w32(p, type);
    w32(p, flags);
    w32(p, addr);
    w32(p, off);
    w32(p, sz);
    w32(p, link);
    w32(p, info);
    w32(p, align);
    w32(p, entsz);
}

static void w_sym(uint8_t **p, uint32_t name, uint32_t val, uint32_t sz,
                  uint8_t info, uint16_t shndx)
{
    w32(p, name);
    w32(p, val);
    w32(p, sz);
    w8(p, info);
    w8(p, 0);                         /* st_other */
    w16(p, shndx);
}

/* ---- Layout planner ---- */

typedef struct {
    uint32_t text_off;
    uint32_t text_sz;
    uint32_t segs_off;
    uint32_t rela_off;
    uint32_t sym_off;
    uint32_t str_off;
    uint32_t shstr_off;
    uint32_t shdr_off;
    uint32_t total;
} lay_t;

/* The loader demands p_offset, p_vaddr and p_paddr share 4-byte alignment, and
 * that every alloc, rela and symtab section is 4-byte aligned on disk. */
static void plan(uint32_t code_bytes, lay_t *l)
{
    l->text_off  = alignup(EHDR_SIZE + PHDR_SIZE, RV_ELF_SEG_ALIGN);
    l->text_sz   = code_bytes;
    l->segs_off  = alignup(l->text_off + l->text_sz, 4u);
    l->rela_off  = l->segs_off + 12u;
    l->sym_off   = l->rela_off;       /* .rela.text is empty, so it occupies nothing */
    l->str_off   = l->sym_off + 2u * SYM_SIZE;
    l->shstr_off = l->str_off + STR_SIZE;
    l->shdr_off  = alignup(l->shstr_off + SHSTR_SIZE, 4u);
    l->total     = l->shdr_off + SEC_COUNT * SHDR_SIZE;
}

/* ---- Public entry point ---- */

int rv_elf_write(const rv_buf_t *code, const char *path)
{
    if (!code || !path) {
        fprintf(stderr, "rv_elf: NULL argument\n");
        return BC_ERR_IO;
    }

    uint32_t code_bytes = rv_buf_nbytes(code);
    if (code_bytes == 0u) {
        fprintf(stderr, "rv_elf: refusing to write an empty kernel\n");
        return BC_ERR_IO;
    }
    uint32_t txtmax = td_txtmax(td_chip());
    if (code_bytes > txtmax) {
        fprintf(stderr, "rv_elf: text %u bytes exceeds the %s limit of %u\n",
                code_bytes, td_cname(td_chip()), txtmax);
        return BC_ERR_IO;
    }

    lay_t lay;
    plan(code_bytes, &lay);

    static uint8_t out[RV_BUF_MAX_WORDS * 4u + 512u];
    if (lay.total > sizeof(out)) {
        fprintf(stderr, "rv_elf: image %u bytes exceeds buffer %lu\n",
                lay.total, (unsigned long)sizeof(out));
        return BC_ERR_IO;
    }

    memset(out, 0, lay.total);
    uint8_t *p = out;

    w_ehdr(&p, lay.shdr_off);
    w_phdr(&p, lay.text_off, code_bytes);

    memcpy(out + lay.text_off, rv_buf_data(code), code_bytes);

    /* .segments: one (vma, trim_bound, size_limit) triple per loadable segment.
     * trim_bound equal to the vma makes the loader's head-trim a no-op. */
    p = out + lay.segs_off;
    w32(&p, RV_ELF_LOAD_ADDR);
    w32(&p, RV_ELF_LOAD_ADDR);
    w32(&p, txtmax);

    /* XIPify resolves relocation symbols through .symtab, so it must exist even
     * though we emit no entries. Index 0 is the reserved null symbol. */
    p = out + lay.sym_off;
    w_sym(&p, 0, 0, 0, 0, 0);
    w_sym(&p, STR_START, RV_ELF_LOAD_ADDR, code_bytes,
          (uint8_t)((STB_GLOBAL << 4) | STT_FUNC), (uint16_t)SEC_TEXT);

    memcpy(out + lay.str_off, k_str, STR_SIZE);
    memcpy(out + lay.shstr_off, k_shstr, SHSTR_SIZE);

    p = out + lay.shdr_off;
    w_shdr(&p, 0, SHT_NULL, 0, 0, 0, 0, 0, 0, 0, 0);
    w_shdr(&p, SHSTR_TEXT, SHT_PROGBITS, SHF_ALLOC | SHF_EXECINSTR,
           RV_ELF_LOAD_ADDR, lay.text_off, code_bytes, 0, 0, 4u, 0);
    /* Empty, but its presence is what stops the loader throwing "there are no
     * relocation sections". sh_info must name an alloc section inside a
     * segment or the loader skips it and the count stays zero. */
    w_shdr(&p, SHSTR_RELA, SHT_RELA, 0, 0, lay.rela_off, 0,
           SEC_SYMTAB, SEC_TEXT, 4u, RELA_SIZE);
    /* sh_info is the index of the first non-local symbol. */
    w_shdr(&p, SHSTR_SYMTAB, SHT_SYMTAB, 0, 0, lay.sym_off, 2u * SYM_SIZE,
           SEC_STRTAB, 1u, 4u, SYM_SIZE);
    w_shdr(&p, SHSTR_STRTAB, SHT_STRTAB, 0, 0, lay.str_off, STR_SIZE,
           0, 0, 1u, 0);
    /* Non-alloc and SHT_PROGBITS, which is how the loader identifies it. */
    w_shdr(&p, SHSTR_SEGS, SHT_PROGBITS, 0, 0, lay.segs_off, 12u,
           0, 0, 4u, 0);
    w_shdr(&p, SHSTR_SHSTR, SHT_STRTAB, 0, 0, lay.shstr_off, SHSTR_SIZE,
           0, 0, 1u, 0);

    FILE *fp = fopen(path, "wb");
    if (!fp) {
        fprintf(stderr, "rv_elf: cannot open %s for writing\n", path);
        return BC_ERR_IO;
    }
    size_t wrote = fwrite(out, 1, lay.total, fp);
    fclose(fp);
    if (wrote != lay.total) {
        fprintf(stderr, "rv_elf: short write to %s (%lu of %u)\n",
                path, (unsigned long)wrote, lay.total);
        return BC_ERR_IO;
    }
    return BC_OK;
}
