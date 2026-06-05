/* rv64_elf.c -- ELF64 relocatable object for RV64. Same shape as cpu_elf.c:
 * the kernel as a global function symbol, libm calls as undefined globals
 * with .rela.text entries. Only the machine (EM_RISCV = 243), the ABI flag,
 * and the relocation type (R_RISCV_CALL_PLT = 19, the auipc/jalr pair) differ
 * from the x86 cousin. */

#include "rv64.h"
#include <stdio.h>
#include <string.h>

static void p32(FILE*f,uint32_t v){fputc((int)(v&0xff),f);fputc((int)((v>>8)&0xff),f);fputc((int)((v>>16)&0xff),f);fputc((int)((v>>24)&0xff),f);}
static void p64(FILE*f,uint64_t v){p32(f,(uint32_t)v);p32(f,(uint32_t)(v>>32));}
static void p16(FILE*f,unsigned v){fputc((int)(v&0xff),f);fputc((int)((v>>8)&0xff),f);}

int rv64_elf(const rv64_mod_t *V, const char *path)
{
    FILE *f = fopen(path,"wb"); if(!f) return -1;
    const char *kn = (V->M->num_funcs && V->M->funcs[0].name < V->M->string_len)
                     ? &V->M->strings[V->M->funcs[0].name] : "kernel";

    char shstr[64]={0}; uint32_t sh=1;
    uint32_t o_text=sh; memcpy(shstr+sh,".text",6);      sh+=6;
    uint32_t o_rela=sh; memcpy(shstr+sh,".rela.text",11);sh+=11;
    uint32_t o_sst =sh; memcpy(shstr+sh,".shstrtab",10); sh+=10;
    uint32_t o_sym =sh; memcpy(shstr+sh,".symtab",8);    sh+=8;
    uint32_t o_str =sh; memcpy(shstr+sh,".strtab",8);    sh+=8;

    char strt[RV_EXTSYM_MAX*RV_EXTSYM_LEN + 64]={0}; uint32_t st=1;
    uint32_t s_k=st; size_t kl=strlen(kn); memcpy(strt+st,kn,kl+1); st+=(uint32_t)kl+1;
    uint32_t s_ext[RV_EXTSYM_MAX];
    for (int i=0;i<V->n_extsym;i++){
        s_ext[i]=st; size_t l=strlen(V->extsym[i]);
        memcpy(strt+st,V->extsym[i],l+1); st+=(uint32_t)l+1;
    }

    uint32_t nsym  = 2 + (uint32_t)V->n_extsym;
    uint64_t relasz= (uint64_t)V->n_reloc*24;
    uint64_t symsz = (uint64_t)nsym*24;

    uint64_t eh=64;
    uint64_t text = eh;
    uint64_t rela = text + V->codelen;
    uint64_t ss   = rela + relasz;
    uint64_t sy   = ss + sh;
    uint64_t str2 = sy + symsz;
    uint64_t shoff= str2 + st;

    /* ELF header: REL, RISC-V (243), hard-float ABI (e_flags=4), 6 sections */
    p32(f,0x464c457f);fputc(2,f);fputc(1,f);fputc(1,f);fputc(0,f);for(int i=0;i<8;i++)fputc(0,f);
    p16(f,1);p16(f,243);p32(f,1);p64(f,0);p64(f,0);p64(f,shoff);p32(f,4);p16(f,64);p16(f,0);p16(f,0);p16(f,64);p16(f,6);p16(f,3);

    fwrite(V->code,1,V->codelen,f);

    /* .rela.text: R_RISCV_CALL_PLT on each auipc, addend 0 */
    for (int i=0;i<V->n_reloc;i++){
        p64(f,V->reloc[i].off);
        p64(f, ((uint64_t)(2+V->reloc[i].sym)<<32) | 19u);
        p64(f,0);
    }

    fwrite(shstr,1,(size_t)sh,f);

    p32(f,0);p32(f,0);p64(f,0);p64(f,0);
    p32(f,s_k);fputc(0x12,f);fputc(0,f);p16(f,1);p64(f,0);p64(f,V->codelen);
    for (int i=0;i<V->n_extsym;i++){ p32(f,s_ext[i]);fputc(0x10,f);fputc(0,f);p16(f,0);p64(f,0);p64(f,0); }

    fwrite(strt,1,(size_t)st,f);

    p32(f,0);p32(f,0);p64(f,0);p64(f,0);p64(f,0);p64(f,0);p32(f,0);p32(f,0);p64(f,0);p64(f,0);                       /* [0] NULL */
    p32(f,o_text);p32(f,1);p64(f,6);p64(f,0);p64(f,text);p64(f,V->codelen);p32(f,0);p32(f,0);p64(f,4);p64(f,0);      /* [1] .text */
    p32(f,o_rela);p32(f,4);p64(f,0);p64(f,0);p64(f,rela);p64(f,relasz);p32(f,4);p32(f,1);p64(f,8);p64(f,24);         /* [2] .rela.text */
    p32(f,o_sst );p32(f,3);p64(f,0);p64(f,0);p64(f,ss);p64(f,(uint64_t)sh);p32(f,0);p32(f,0);p64(f,1);p64(f,0);      /* [3] .shstrtab */
    p32(f,o_sym );p32(f,2);p64(f,0);p64(f,0);p64(f,sy);p64(f,symsz);p32(f,5);p32(f,1);p64(f,8);p64(f,24);           /* [4] .symtab */
    p32(f,o_str );p32(f,3);p64(f,0);p64(f,0);p64(f,str2);p64(f,(uint64_t)st);p32(f,0);p32(f,0);p64(f,1);p64(f,0);    /* [5] .strtab */

    fclose(f); return 0;
}
