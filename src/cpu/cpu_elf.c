/* cpu_elf.c -- ELF64 relocatable object. The kernel is exported as a global
 * function symbol, and any calls it makes out to libm turn up as undefined
 * globals with matching .rela.text entries, so the linker can chase them
 * down and fill in the call sites. Six sections (NULL, .text, .rela.text,
 * .shstrtab, .symtab, .strtab), byte-at-a-time, no library, no nonsense. */

#include "cpu.h"
#include <stdio.h>
#include <string.h>

static void p32(FILE*f,uint32_t v){fputc((int)(v&0xff),f);fputc((int)((v>>8)&0xff),f);fputc((int)((v>>16)&0xff),f);fputc((int)((v>>24)&0xff),f);}
static void p64(FILE*f,uint64_t v){p32(f,(uint32_t)v);p32(f,(uint32_t)(v>>32));}
static void p16(FILE*f,unsigned v){fputc((int)(v&0xff),f);fputc((int)((v>>8)&0xff),f);}

int cpu_elf(const cpu_mod_t *X, const char *path)
{
    FILE *f = fopen(path,"wb"); if(!f) return -1;
    const char *kn = (X->M->num_funcs && X->M->funcs[0].name < X->M->string_len)
                     ? &X->M->strings[X->M->funcs[0].name] : "kernel";

    /* ---- section-header string table ---- */
    char shstr[64]={0}; uint32_t sh=1;
    uint32_t o_text=sh; memcpy(shstr+sh,".text",6);      sh+=6;
    uint32_t o_rela=sh; memcpy(shstr+sh,".rela.text",11);sh+=11;
    uint32_t o_sst =sh; memcpy(shstr+sh,".shstrtab",10); sh+=10;
    uint32_t o_sym =sh; memcpy(shstr+sh,".symtab",8);    sh+=8;
    uint32_t o_str =sh; memcpy(shstr+sh,".strtab",8);    sh+=8;

    /* ---- symbol string table: kernel name, then each external ---- */
    char strt[CPU_EXTSYM_MAX*CPU_EXTSYM_LEN + 64]={0}; uint32_t st=1;
    uint32_t s_k=st; size_t kl=strlen(kn); memcpy(strt+st,kn,kl+1); st+=(uint32_t)kl+1;
    uint32_t s_ext[CPU_EXTSYM_MAX];
    for (int i=0;i<X->n_extsym;i++){
        s_ext[i]=st; size_t l=strlen(X->extsym[i]);
        memcpy(strt+st,X->extsym[i],l+1); st+=(uint32_t)l+1;
    }

    uint32_t nsym  = 2 + (uint32_t)X->n_extsym;       /* NULL + kernel + externals */
    uint64_t relasz= (uint64_t)X->n_reloc*24;
    uint64_t symsz = (uint64_t)nsym*24;

    uint64_t eh=64;
    uint64_t text = eh;
    uint64_t rela = text + X->codelen;
    uint64_t ss   = rela + relasz;
    uint64_t sy   = ss + sh;
    uint64_t str2 = sy + symsz;
    uint64_t shoff= str2 + st;

    /* ---- ELF header: REL, x86-64, 6 sections, shstrtab at index 3 ---- */
    p32(f,0x464c457f);fputc(2,f);fputc(1,f);fputc(1,f);fputc(0,f);for(int i=0;i<8;i++)fputc(0,f);
    p16(f,1);p16(f,62);p32(f,1);p64(f,0);p64(f,0);p64(f,shoff);p32(f,0);p16(f,64);p16(f,0);p16(f,0);p16(f,64);p16(f,6);p16(f,3);

    /* ---- .text ---- */
    fwrite(X->code,1,X->codelen,f);

    /* ---- .rela.text: every external call site, R_X86_64_PLT32, addend -4 ---- */
    for (int i=0;i<X->n_reloc;i++){
        p64(f,X->reloc[i].off);
        p64(f, ((uint64_t)(2+X->reloc[i].sym)<<32) | 4u);
        p64(f,(uint64_t)(-4));
    }

    /* ---- .shstrtab ---- */
    fwrite(shstr,1,(size_t)sh,f);

    /* ---- .symtab: NULL, kernel (global func, defined), then undef globals ---- */
    p32(f,0);p32(f,0);p64(f,0);p64(f,0);
    p32(f,s_k);fputc(0x12,f);fputc(0,f);p16(f,1);p64(f,0);p64(f,X->codelen);
    for (int i=0;i<X->n_extsym;i++){ p32(f,s_ext[i]);fputc(0x10,f);fputc(0,f);p16(f,0);p64(f,0);p64(f,0); }

    /* ---- .strtab ---- */
    fwrite(strt,1,(size_t)st,f);

    /* ---- section headers ---- */
    p32(f,0);p32(f,0);p64(f,0);p64(f,0);p64(f,0);p64(f,0);p32(f,0);p32(f,0);p64(f,0);p64(f,0);                         /* [0] NULL */
    p32(f,o_text);p32(f,1);p64(f,6);p64(f,0);p64(f,text);p64(f,X->codelen);p32(f,0);p32(f,0);p64(f,16);p64(f,0);       /* [1] .text */
    p32(f,o_rela);p32(f,4);p64(f,0);p64(f,0);p64(f,rela);p64(f,relasz);p32(f,4);p32(f,1);p64(f,8);p64(f,24);           /* [2] .rela.text -> symtab(4), applies to .text(1) */
    p32(f,o_sst );p32(f,3);p64(f,0);p64(f,0);p64(f,ss);p64(f,(uint64_t)sh);p32(f,0);p32(f,0);p64(f,1);p64(f,0);        /* [3] .shstrtab */
    p32(f,o_sym );p32(f,2);p64(f,0);p64(f,0);p64(f,sy);p64(f,symsz);p32(f,5);p32(f,1);p64(f,8);p64(f,24);             /* [4] .symtab -> strtab(5), first global = 1 */
    p32(f,o_str );p32(f,3);p64(f,0);p64(f,0);p64(f,str2);p64(f,(uint64_t)st);p32(f,0);p32(f,0);p64(f,1);p64(f,0);      /* [5] .strtab */

    fclose(f); return 0;
}
