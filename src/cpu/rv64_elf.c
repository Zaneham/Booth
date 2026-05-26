/* rv64_elf.c -- ELF64 relocatable object for RV64, kernel exported as a
 * global function symbol. Same shape as cpu_elf.c; only the machine type
 * changes (EM_RISCV = 243). Byte-at-a-time, no library. */

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
    char shstr[40]={0}; uint32_t sh=1, o_text=sh; memcpy(shstr+sh,".text",6);sh+=6;
    uint32_t o_sst=sh; memcpy(shstr+sh,".shstrtab",10);sh+=10; uint32_t o_sym=sh; memcpy(shstr+sh,".symtab",8);sh+=8;
    uint32_t o_str=sh; memcpy(shstr+sh,".strtab",8);sh+=8;
    char strt[64]={0}; uint32_t st=1, s_k=st; size_t kl=strlen(kn); memcpy(strt+st,kn,kl+1); st+=(uint32_t)kl+1;
    uint64_t eh=64, text=eh, ss=text+V->codelen, sy=ss+(uint32_t)sh, st2=sy+48, shoff=st2+(uint32_t)st;
    p32(f,0x464c457f);fputc(2,f);fputc(1,f);fputc(1,f);fputc(0,f);for(int i=0;i<8;i++)fputc(0,f);
    /* e_flags = 0x4 (EF_RISCV_FLOAT_ABI_DOUBLE): we pass floats in fa regs,
     * so the object declares the hard-float (lp64d) ABI to link with one. */
    p16(f,1);p16(f,243);p32(f,1);p64(f,0);p64(f,0);p64(f,shoff);p32(f,4);p16(f,64);p16(f,0);p16(f,0);p16(f,64);p16(f,5);p16(f,2);
    fwrite(V->code,1,V->codelen,f); fwrite(shstr,1,(size_t)sh,f);
    p32(f,0);p32(f,0);p64(f,0);p64(f,0); p32(f,s_k);fputc(0x12,f);fputc(0,f);p16(f,1);p64(f,0);p64(f,V->codelen); /* sym1: kernel */
    fwrite(strt,1,(size_t)st,f);
    p32(f,0);p32(f,0);p64(f,0);p64(f,0);p64(f,0);p64(f,0);p32(f,0);p32(f,0);p64(f,0);p64(f,0); /* sh0 null */
    p32(f,o_text);p32(f,1);p64(f,6);p64(f,0);p64(f,text);p64(f,V->codelen);p32(f,0);p32(f,0);p64(f,4);p64(f,0); /* .text */
    p32(f,o_sst);p32(f,3);p64(f,0);p64(f,0);p64(f,ss);p64(f,(uint64_t)sh);p32(f,0);p32(f,0);p64(f,1);p64(f,0); /* .shstrtab */
    p32(f,o_sym);p32(f,2);p64(f,0);p64(f,0);p64(f,sy);p64(f,48);p32(f,4);p32(f,1);p64(f,8);p64(f,24); /* .symtab */
    p32(f,o_str);p32(f,3);p64(f,0);p64(f,0);p64(f,st2);p64(f,(uint64_t)st);p32(f,0);p32(f,0);p64(f,1);p64(f,0); /* .strtab */
    fclose(f); return 0;
}
