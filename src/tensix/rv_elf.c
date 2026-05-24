#include "rv_elf.h"
#include <stdio.h>
#include <string.h>

/*
 * ELF32 little-endian header writer. Constants are inlined as
 * literals rather than included from elf.h because there is no
 * portable elf.h on Windows/MinGW and inlining keeps the build
 * dependency-free, which is the whole point of this compiler.
 *
 * All multibyte fields are written little-endian, which matches
 * both the ELFDATA2LSB declaration and the host x86-64 byte order
 * the compiler runs on. The native-write below assumes that;
 * porting to a big-endian compiler host would need explicit
 * byte-swizzling, which is a problem we can have when we have it.
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
#define PF_W           2u
#define PF_R           4u

#define SHT_NULL       0u
#define SHT_PROGBITS   1u
#define SHT_STRTAB     3u

#define SHF_ALLOC      2u
#define SHF_EXECINSTR  4u

#define EHDR_SIZE      52u
#define PHDR_SIZE      32u
#define SHDR_SIZE      40u

/* ---- Little-endian writers ---- */

static void w8 (uint8_t  **p, uint8_t  v) { **p = v;       (*p)++; }
static void w16(uint8_t  **p, uint16_t v)
{
    (*p)[0] = (uint8_t)(v & 0xFFu);
    (*p)[1] = (uint8_t)((v >> 8) & 0xFFu);
    *p += 2;
}
static void w32(uint8_t  **p, uint32_t v)
{
    (*p)[0] = (uint8_t)(v        & 0xFFu);
    (*p)[1] = (uint8_t)((v >> 8) & 0xFFu);
    (*p)[2] = (uint8_t)((v >> 16) & 0xFFu);
    (*p)[3] = (uint8_t)((v >> 24) & 0xFFu);
    *p += 4;
}

/* ---- Header builders ---- */

/* Section names live in a tiny string table. Indices into the table:
 *   0: empty string (every strtab starts with NUL)
 *   1: ".text"      (length 5 + NUL = 6)
 *   7: ".shstrtab"  (length 9 + NUL = 10)
 * Total: 17 bytes. */
static const char k_shstrtab[] = "\0.text\0.shstrtab\0";
#define SHSTRTAB_SIZE   17u
#define SHSTR_OFF_TEXT  1u
#define SHSTR_OFF_SHSTR 7u

static void write_ehdr(uint8_t **p, uint32_t e_phoff, uint32_t e_shoff,
                       uint32_t e_phnum, uint32_t e_shnum,
                       uint32_t e_shstrndx)
{
    /* e_ident: 16 bytes */
    w8(p, ELF_MAGIC0);
    w8(p, ELF_MAGIC1);
    w8(p, ELF_MAGIC2);
    w8(p, ELF_MAGIC3);
    w8(p, ELFCLASS32);
    w8(p, ELFDATA2LSB);
    w8(p, EV_CURRENT);
    w8(p, ELFOSABI_NONE);
    for (int i = 0; i < 8; i++) w8(p, 0);   /* padding */

    w16(p, (uint16_t)ET_EXEC);                /* e_type */
    w16(p, (uint16_t)EM_RISCV);               /* e_machine */
    w32(p, EV_CURRENT);                       /* e_version */
    w32(p, RV_ELF_ENTRY);                     /* e_entry */
    w32(p, e_phoff);                          /* e_phoff */
    w32(p, e_shoff);                          /* e_shoff */
    w32(p, 0);                                /* e_flags, soft-float ABI */
    w16(p, (uint16_t)EHDR_SIZE);              /* e_ehsize */
    w16(p, (uint16_t)PHDR_SIZE);              /* e_phentsize */
    w16(p, (uint16_t)e_phnum);                /* e_phnum */
    w16(p, (uint16_t)SHDR_SIZE);              /* e_shentsize */
    w16(p, (uint16_t)e_shnum);                /* e_shnum */
    w16(p, (uint16_t)e_shstrndx);             /* e_shstrndx */
}

static void write_phdr(uint8_t **p, uint32_t p_offset, uint32_t p_vaddr,
                       uint32_t p_filesz, uint32_t p_memsz,
                       uint32_t p_flags, uint32_t p_align)
{
    w32(p, PT_LOAD);    /* p_type */
    w32(p, p_offset);   /* p_offset */
    w32(p, p_vaddr);    /* p_vaddr */
    w32(p, p_vaddr);    /* p_paddr, same as vaddr on Tensix (no MMU) */
    w32(p, p_filesz);
    w32(p, p_memsz);
    w32(p, p_flags);
    w32(p, p_align);
}

static void write_shdr(uint8_t **p, uint32_t name, uint32_t type,
                       uint32_t flags, uint32_t addr, uint32_t offset,
                       uint32_t size, uint32_t link, uint32_t info,
                       uint32_t addralign, uint32_t entsize)
{
    w32(p, name);        /* sh_name (index into .shstrtab) */
    w32(p, type);        /* sh_type */
    w32(p, flags);
    w32(p, addr);
    w32(p, offset);
    w32(p, size);
    w32(p, link);
    w32(p, info);
    w32(p, addralign);
    w32(p, entsize);
}

/* ---- Layout planner ---- */

typedef struct {
    uint32_t code_off;
    uint32_t code_sz;
    uint32_t shstr_off;
    uint32_t shdr_off;
    uint32_t total_sz;
} rv_elf_lay_t;

static void plan_layout(uint32_t code_bytes, rv_elf_lay_t *l)
{
    l->code_off  = EHDR_SIZE + PHDR_SIZE;
    l->code_sz   = code_bytes;
    l->shstr_off = l->code_off + l->code_sz;
    l->shdr_off  = l->shstr_off + SHSTRTAB_SIZE;
    /* Three section headers: NULL, .text, .shstrtab. */
    l->total_sz  = l->shdr_off + 3u * SHDR_SIZE;
}

/* ---- Public entry point ---- */

int rv_elf_write(const rv_buf_t *code, const char *path)
{
    if (!code || !path) {
        fprintf(stderr, "rv_elf: NULL argument\n");
        return BC_ERR_IO;
    }

    uint32_t code_bytes = rv_buf_pos_bytes(code);
    if (code_bytes == 0u) {
        fprintf(stderr, "rv_elf: refusing to write an empty kernel\n");
        return BC_ERR_IO;
    }

    rv_elf_lay_t lay;
    plan_layout(code_bytes, &lay);

    /* Fixed maximum size: 16 KiB code + 256 bytes of header/section
     * metadata is well under what we want to hold in a stack frame
     * but the file write below uses the buffer as a single blob,
     * so we cap it here and refuse anything bigger. */
    static uint8_t out[RV_BUF_MAX_WORDS * 4u + 256u];
    if (lay.total_sz > sizeof(out)) {
        fprintf(stderr,
                "rv_elf: kernel image %u bytes exceeds buffer %lu\n",
                lay.total_sz, (unsigned long)sizeof(out));
        return BC_ERR_IO;
    }

    memset(out, 0, lay.total_sz);
    uint8_t *p = out;

    /* ELF header */
    write_ehdr(&p,
               EHDR_SIZE,                     /* e_phoff: right after ehdr */
               lay.shdr_off,                  /* e_shoff */
               1u,                            /* e_phnum: one PT_LOAD */
               3u,                            /* e_shnum: NULL/.text/.shstrtab */
               2u);                           /* e_shstrndx: .shstrtab at idx 2 */

    /* Program header */
    write_phdr(&p,
               lay.code_off,
               RV_ELF_LOAD_ADDR,
               code_bytes,
               code_bytes,
               PF_R | PF_X,
               4u);

    /* Code bytes */
    memcpy(out + lay.code_off, rv_buf_data(code), code_bytes);

    /* .shstrtab content */
    memcpy(out + lay.shstr_off, k_shstrtab, SHSTRTAB_SIZE);

    /* Section header table */
    p = out + lay.shdr_off;
    write_shdr(&p, 0, SHT_NULL,     0, 0, 0, 0, 0, 0, 0, 0);
    write_shdr(&p, SHSTR_OFF_TEXT,  SHT_PROGBITS,
               SHF_ALLOC | SHF_EXECINSTR,
               RV_ELF_LOAD_ADDR, lay.code_off, code_bytes,
               0, 0, 4, 0);
    write_shdr(&p, SHSTR_OFF_SHSTR, SHT_STRTAB,
               0, 0, lay.shstr_off, SHSTRTAB_SIZE,
               0, 0, 1, 0);

    FILE *fp = fopen(path, "wb");
    if (!fp) {
        fprintf(stderr, "rv_elf: cannot open %s for writing\n", path);
        return BC_ERR_IO;
    }
    size_t wrote = fwrite(out, 1, lay.total_sz, fp);
    fclose(fp);
    if (wrote != lay.total_sz) {
        fprintf(stderr, "rv_elf: short write to %s (%lu of %u)\n",
                path, (unsigned long)wrote, lay.total_sz);
        return BC_ERR_IO;
    }
    return BC_OK;
}
