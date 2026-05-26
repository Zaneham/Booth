/* rv64_emit.c -- BIR to RV64IMFD, stack-everything, the RISC-V cousin of
 * cpu_emit.c. Encoder ripped from my Karearea RV64 backend; the slot model,
 * frame, and SIMT-as-a-loop are straight off the x86 version.
 *
 * First cut is the integer path: enough to run int kernels (vadd,
 * reductions) under qemu-riscv64. Floats, the full branch/phi set, and the
 * register allocator are follow-on increments. */

#include "rv64.h"
#include <string.h>

/* ---- raw 32-bit instruction word ---- */
static void ew(rv64_mod_t *V, uint32_t w){
    if (V->codelen+4<=RV_CODE_MAX){
        V->code[V->codelen++]=(uint8_t)w; V->code[V->codelen++]=(uint8_t)(w>>8);
        V->code[V->codelen++]=(uint8_t)(w>>16); V->code[V->codelen++]=(uint8_t)(w>>24);
    }
}

/* ---- format builders (from Karearea, bit-packing kept in unsigned so
 * the high-bit shifts don't trip -Wsign-conversion) ---- */
#define U(x) ((uint32_t)(x))
static uint32_t mk_R(int op,int rd,int f3,int rs1,int rs2,int f7){
    return U(op&0x7F)|(U(rd&0x1F)<<7)|(U(f3&7)<<12)|(U(rs1&0x1F)<<15)|(U(rs2&0x1F)<<20)|(U(f7&0x7F)<<25); }
static uint32_t mk_I(int op,int rd,int f3,int rs1,int imm12){
    return U(op&0x7F)|(U(rd&0x1F)<<7)|(U(f3&7)<<12)|(U(rs1&0x1F)<<15)|(U(imm12&0xFFF)<<20); }
static uint32_t mk_S(int op,int f3,int rs1,int rs2,int imm12){
    return U(op&0x7F)|(U(imm12&0x1F)<<7)|(U(f3&7)<<12)|(U(rs1&0x1F)<<15)|(U(rs2&0x1F)<<20)|(U((imm12>>5)&0x7F)<<25); }
static uint32_t mk_B(int op,int f3,int rs1,int rs2,int imm13){
    return U(op&0x7F)|(U((imm13>>11)&1)<<7)|(U((imm13>>1)&0xF)<<8)|(U(f3&7)<<12)|(U(rs1&0x1F)<<15)|(U(rs2&0x1F)<<20)|(U((imm13>>5)&0x3F)<<25)|(U((imm13>>12)&1)<<31); }
static uint32_t mk_U(int op,int rd,int imm20){
    return U(op&0x7F)|(U(rd&0x1F)<<7)|(U(imm20&0xFFFFF)<<12); }
static uint32_t mk_J(int op,int rd,int imm21){
    return U(op&0x7F)|(U(rd&0x1F)<<7)|(U((imm21>>12)&0xFF)<<12)|(U((imm21>>11)&1)<<20)|(U((imm21>>1)&0x3FF)<<21)|(U((imm21>>20)&1)<<31); }

/* ---- convenience emitters ---- */
static void e_add (rv64_mod_t*V,int d,int a,int b){ ew(V,mk_R(0x33,d,0,a,b,0x00)); }
static void e_sub (rv64_mod_t*V,int d,int a,int b){ ew(V,mk_R(0x33,d,0,a,b,0x20)); }
static void e_mul (rv64_mod_t*V,int d,int a,int b){ ew(V,mk_R(0x33,d,0,a,b,0x01)); }
static void e_addi(rv64_mod_t*V,int d,int a,int i){ ew(V,mk_I(0x13,d,0,a,i&0xFFF)); }
static void e_ld  (rv64_mod_t*V,int d,int a,int o){ ew(V,mk_I(0x03,d,3,a,o&0xFFF)); }   /* 64-bit */
static void e_lw  (rv64_mod_t*V,int d,int a,int o){ ew(V,mk_I(0x03,d,2,a,o&0xFFF)); }   /* 32-bit, sign-ext */
static void e_lh  (rv64_mod_t*V,int d,int a,int o){ ew(V,mk_I(0x03,d,1,a,o&0xFFF)); }
static void e_lb  (rv64_mod_t*V,int d,int a,int o){ ew(V,mk_I(0x03,d,0,a,o&0xFFF)); }
static void e_sd  (rv64_mod_t*V,int a,int s,int o){ ew(V,mk_S(0x23,3,a,s,o&0xFFF)); }
static void e_sw  (rv64_mod_t*V,int a,int s,int o){ ew(V,mk_S(0x23,2,a,s,o&0xFFF)); }
static void e_sh  (rv64_mod_t*V,int a,int s,int o){ ew(V,mk_S(0x23,1,a,s,o&0xFFF)); }
static void e_sb  (rv64_mod_t*V,int a,int s,int o){ ew(V,mk_S(0x23,0,a,s,o&0xFFF)); }
static void e_lui (rv64_mod_t*V,int d,int i){ ew(V,mk_U(0x37,d,i)); }
static void e_jal (rv64_mod_t*V,int d,int i){ ew(V,mk_J(0x6F,d,i)); }
static void e_jalr(rv64_mod_t*V,int d,int a,int i){ ew(V,mk_I(0x67,d,0,a,i&0xFFF)); }
static void e_bge (rv64_mod_t*V,int a,int b,int o){ ew(V,mk_B(0x63,5,a,b,o)); }          /* signed >= */

/* load a 32-bit signed immediate */
static void e_li(rv64_mod_t*V,int d,int32_t v){
    if (v>=-2048 && v<2048){ e_addi(V,d,V_ZERO,v); return; }
    int32_t hi=(v+0x800)>>12, lo=v-(hi<<12);
    e_lui(V,d,hi&0xFFFFF);
    if (lo) e_addi(V,d,d,lo);
}

static int32_t slot(rv64_mod_t*V,uint32_t i){ return V->slots[i]; }
static void ld_slot(rv64_mod_t*V,int r,int32_t o){ e_ld(V,r,V_S0,o); }
static void st_slot(rv64_mod_t*V,int r,int32_t o){ e_sd(V,V_S0,r,o); }

/* value into reg: const -> li, else load its 64-bit slot */
static void load_val(rv64_mod_t*V,int r,uint32_t v){
    if (BIR_VAL_IS_CONST(v)){ uint32_t c=BIR_VAL_INDEX(v);
        e_li(V,r,(int32_t)(c<V->M->num_consts?V->M->consts[c].d.ival:0)); }
    else ld_slot(V,r,slot(V,BIR_VAL_INDEX(v)));
}

/* element size in bytes of a pointer's pointee (drives GEP stride) */
static int pointee_sz(rv64_mod_t*V,uint32_t ty){
    if (ty<V->M->num_types && V->M->types[ty].kind==BIR_TYPE_PTR){
        uint32_t in=V->M->types[ty].inner;
        if (in<V->M->num_types){
            if (V->M->types[in].kind==BIR_TYPE_PTR) return 8;
            uint32_t w=V->M->types[in].width; if (w>=8) return (int)(w/8);
        }
    }
    return 4;
}

static void rv64_func(rv64_mod_t *V,const bir_func_t *F){
    int is_kernel=(F->cuda_flags&CUDA_GLOBAL)!=0;
    /* frame: [s0-8]=ra, [s0-16]=saved s0, slots below that */
    int32_t off=-16;
    for (uint16_t p=0;p<F->num_params;p++){ off-=8; V->slots[p]=off; }
    for (uint16_t b=0;b<F->num_blocks;b++){
        const bir_block_t*B=&V->M->blocks[F->first_block+b];
        for(uint32_t i=0;i<B->num_insts;i++){ off-=8; V->slots[B->first_inst+i]=off; }
    }
    off-=8; int32_t tid_off=off;
    off-=8; int32_t ntid_off=off;
    int32_t frame=(-off+15)&~15;

    /* prologue */
    e_addi(V,V_SP,V_SP,-frame);
    e_sd(V,V_SP,V_RA,frame-8);
    e_sd(V,V_SP,V_S0,frame-16);
    e_addi(V,V_S0,V_SP,frame);

    /* params: integer/pointer args in a0..a7 (first cut: int/ptr only) */
    for (uint16_t p=0;p<F->num_params && p<8;p++) st_slot(V,V_A0+p,V->slots[p]);

    uint32_t loop_head=0, bge_off=0;
    if (is_kernel){
        if (F->num_params<8) st_slot(V,V_A0+F->num_params,ntid_off);  /* hidden nthreads */
        else { e_li(V,V_T0,1); st_slot(V,V_T0,ntid_off); }
        e_li(V,V_T0,0); st_slot(V,V_T0,tid_off);                       /* tid = 0 */
        loop_head=V->codelen;
        ld_slot(V,V_T0,tid_off); ld_slot(V,V_T1,ntid_off);
        bge_off=V->codelen; e_bge(V,V_T0,V_T1,0);                      /* bge tid,ntid,END (patched) */
    }

    uint32_t retfix[256]; int n_ret=0;
    for (uint16_t b=0;b<F->num_blocks;b++){
        const bir_block_t*B=&V->M->blocks[F->first_block+b]; V->blk_off[F->first_block+b]=V->codelen;
        for (uint32_t i=0;i<B->num_insts;i++){ uint32_t ix=B->first_inst+i; const bir_inst_t*I=&V->M->insts[ix]; int32_t s=slot(V,ix);
        switch(I->op){
        case BIR_PARAM: break;
        case BIR_THREAD_ID: if(is_kernel){ ld_slot(V,V_T0,tid_off); } else e_li(V,V_T0,0); st_slot(V,V_T0,s); break;
        case BIR_BLOCK_ID:  e_li(V,V_T0,0); st_slot(V,V_T0,s); break;
        case BIR_BLOCK_DIM: if(is_kernel){ ld_slot(V,V_T0,ntid_off); } else e_li(V,V_T0,1); st_slot(V,V_T0,s); break;
        case BIR_GRID_DIM:  e_li(V,V_T0,1); st_slot(V,V_T0,s); break;
        case BIR_ADD: load_val(V,V_T0,I->operands[0]);load_val(V,V_T1,I->operands[1]);e_add(V,V_T0,V_T0,V_T1);st_slot(V,V_T0,s); break;
        case BIR_SUB: load_val(V,V_T0,I->operands[0]);load_val(V,V_T1,I->operands[1]);e_sub(V,V_T0,V_T0,V_T1);st_slot(V,V_T0,s); break;
        case BIR_MUL: load_val(V,V_T0,I->operands[0]);load_val(V,V_T1,I->operands[1]);e_mul(V,V_T0,V_T0,V_T1);st_slot(V,V_T0,s); break;
        case BIR_GEP: { int sz=pointee_sz(V,I->type); load_val(V,V_T0,I->operands[1]); e_li(V,V_T1,sz); e_mul(V,V_T0,V_T0,V_T1); load_val(V,V_T1,I->operands[0]); e_add(V,V_T0,V_T0,V_T1); st_slot(V,V_T0,s); break; }
        case BIR_LOAD: { load_val(V,V_T0,I->operands[0]);
            const bir_type_t*t=(I->type<V->M->num_types)?&V->M->types[I->type]:0;
            int w=t?(int)t->width:32;
            if (w==64) e_ld(V,V_T0,V_T0,0); else if (w==16) e_lh(V,V_T0,V_T0,0); else if (w==8||w==1) e_lb(V,V_T0,V_T0,0); else e_lw(V,V_T0,V_T0,0);
            st_slot(V,V_T0,s); break; }
        case BIR_STORE: { load_val(V,V_T0,I->operands[0]); load_val(V,V_T1,I->operands[1]);
            int w=pointee_sz(V,(BIR_VAL_IS_CONST(I->operands[1])?0:V->M->insts[BIR_VAL_INDEX(I->operands[1])].type))*8;
            if (w==64) e_sd(V,V_T1,V_T0,0); else if (w==16) e_sh(V,V_T1,V_T0,0); else if (w==8) e_sb(V,V_T1,V_T0,0); else e_sw(V,V_T1,V_T0,0);
            break; }
        case BIR_RET:
            if (is_kernel){ if(n_ret<256) retfix[n_ret++]=V->codelen; e_jal(V,V_ZERO,0); }   /* -> loop_cont */
            else { e_ld(V,V_RA,V_SP,frame-8); e_ld(V,V_S0,V_SP,frame-16); e_addi(V,V_SP,V_SP,frame); e_jalr(V,V_ZERO,V_RA,0); }
            break;
        default: e_li(V,V_T0,0); st_slot(V,V_T0,s); break;
        }}
    }

    if (is_kernel){
        uint32_t loop_cont=V->codelen;
        ld_slot(V,V_T0,tid_off); e_addi(V,V_T0,V_T0,1); st_slot(V,V_T0,tid_off);
        e_jal(V,V_ZERO,(int32_t)loop_head-(int32_t)V->codelen);                 /* jal back to head */
        uint32_t loop_end=V->codelen;
        /* patch bge tid,ntid -> loop_end */
        { uint32_t w=mk_B(0x63,5,V_T0,V_T1,(int32_t)loop_end-(int32_t)bge_off);
          V->code[bge_off]=(uint8_t)w;V->code[bge_off+1]=(uint8_t)(w>>8);V->code[bge_off+2]=(uint8_t)(w>>16);V->code[bge_off+3]=(uint8_t)(w>>24); }
        /* patch each RET's jal -> loop_cont */
        for (int k=0;k<n_ret;k++){ uint32_t o=retfix[k]; uint32_t w=mk_J(0x6F,V_ZERO,(int32_t)loop_cont-(int32_t)o);
          V->code[o]=(uint8_t)w;V->code[o+1]=(uint8_t)(w>>8);V->code[o+2]=(uint8_t)(w>>16);V->code[o+3]=(uint8_t)(w>>24); }
    }
    /* epilogue (fallthrough / non-kernel safety net) */
    e_ld(V,V_RA,V_SP,frame-8); e_ld(V,V_S0,V_SP,frame-16); e_addi(V,V_SP,V_SP,frame); e_jalr(V,V_ZERO,V_RA,0);
}

void rv64_init(rv64_mod_t *V,const bir_module_t *M){ memset(V,0,sizeof(*V)); V->M=M; }
int rv64_emit(rv64_mod_t *V){ for(uint32_t f=0;f<V->M->num_funcs;f++) rv64_func(V,&V->M->funcs[f]); return 0; }
