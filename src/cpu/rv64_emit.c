/* rv64_emit.c -- BIR to RV64IMFD, stack-everything.
 *
 * RISC-V: the instruction set that took one look at x86's forty years of
 * barnacles and politely declined the lot. Thirty-two registers, fixed
 * 32-bit instructions, no prefixes, no REX, no fever dream. We honour that
 * elegance by ignoring it almost entirely and shoving every value onto the
 * stack, because correctness now beats cleverness later, and the stack,
 * like the Ankh-Morpork river, will hold absolutely anything you put in it.
 *
 * The encoder is lifted from the RV64 backend I wrote for Karearea; the
 * slot model and the SIMT-as-a-thread-loop trick are straight off the x86
 * cousin in cpu_emit.c. A kernel runs its body once per thread_id, with
 * nthreads a hidden arg on the end. */

#include "rv64.h"
#include <string.h>

static void ew(rv64_mod_t *V, uint32_t w){
    if (V->codelen+4<=RV_CODE_MAX){
        V->code[V->codelen++]=(uint8_t)w; V->code[V->codelen++]=(uint8_t)(w>>8);
        V->code[V->codelen++]=(uint8_t)(w>>16); V->code[V->codelen++]=(uint8_t)(w>>24);
    }
}

/* ---- format builders. Six shapes, and RISC-V scatters the immediate bits
 * across the word like a wizard who has lost confidence halfway through the
 * spell. unsigned throughout so -Wconversion stays calm. ---- */
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

/* ---- integer ops ---- */
static void e_add (rv64_mod_t*V,int d,int a,int b){ ew(V,mk_R(0x33,d,0,a,b,0x00)); }
static void e_sub (rv64_mod_t*V,int d,int a,int b){ ew(V,mk_R(0x33,d,0,a,b,0x20)); }
static void e_mul (rv64_mod_t*V,int d,int a,int b){ ew(V,mk_R(0x33,d,0,a,b,0x01)); }
static void e_slt (rv64_mod_t*V,int d,int a,int b){ ew(V,mk_R(0x33,d,2,a,b,0x00)); }
static void e_sltu(rv64_mod_t*V,int d,int a,int b){ ew(V,mk_R(0x33,d,3,a,b,0x00)); }
static void e_addi(rv64_mod_t*V,int d,int a,int i){ ew(V,mk_I(0x13,d,0,a,i&0xFFF)); }
static void e_xori(rv64_mod_t*V,int d,int a,int i){ ew(V,mk_I(0x13,d,4,a,i&0xFFF)); }
static void e_sltiu(rv64_mod_t*V,int d,int a,int i){ ew(V,mk_I(0x13,d,3,a,i&0xFFF)); }
static void e_ld  (rv64_mod_t*V,int d,int a,int o){ ew(V,mk_I(0x03,d,3,a,o&0xFFF)); }
static void e_lw  (rv64_mod_t*V,int d,int a,int o){ ew(V,mk_I(0x03,d,2,a,o&0xFFF)); }
static void e_lh  (rv64_mod_t*V,int d,int a,int o){ ew(V,mk_I(0x03,d,1,a,o&0xFFF)); }
static void e_lb  (rv64_mod_t*V,int d,int a,int o){ ew(V,mk_I(0x03,d,0,a,o&0xFFF)); }
static void e_sd  (rv64_mod_t*V,int a,int s,int o){ ew(V,mk_S(0x23,3,a,s,o&0xFFF)); }
static void e_sw  (rv64_mod_t*V,int a,int s,int o){ ew(V,mk_S(0x23,2,a,s,o&0xFFF)); }
static void e_sh  (rv64_mod_t*V,int a,int s,int o){ ew(V,mk_S(0x23,1,a,s,o&0xFFF)); }
static void e_sb  (rv64_mod_t*V,int a,int s,int o){ ew(V,mk_S(0x23,0,a,s,o&0xFFF)); }
static void e_lui (rv64_mod_t*V,int d,int i){ ew(V,mk_U(0x37,d,i)); }
static void e_jal (rv64_mod_t*V,int d,int i){ ew(V,mk_J(0x6F,d,i)); }
static void e_jalr(rv64_mod_t*V,int d,int a,int i){ ew(V,mk_I(0x67,d,0,a,i&0xFFF)); }
static void e_beq (rv64_mod_t*V,int a,int b,int o){ ew(V,mk_B(0x63,0,a,b,o)); }
static void e_blt (rv64_mod_t*V,int a,int b,int o){ ew(V,mk_B(0x63,4,a,b,o)); }

/* ---- single-precision float ops (the F extension; D is there too but BIR
 * floats are f32, so we live in .S land) ---- */
static void e_flw  (rv64_mod_t*V,int fd,int a,int o){ ew(V,mk_I(0x07,fd,2,a,o&0xFFF)); }
static void e_fsw  (rv64_mod_t*V,int a,int fs,int o){ ew(V,mk_S(0x27,2,a,fs,o&0xFFF)); }
static void e_fadds(rv64_mod_t*V,int d,int a,int b){ ew(V,mk_R(0x53,d,0,a,b,0x00)); }
static void e_fsubs(rv64_mod_t*V,int d,int a,int b){ ew(V,mk_R(0x53,d,0,a,b,0x04)); }
static void e_fmuls(rv64_mod_t*V,int d,int a,int b){ ew(V,mk_R(0x53,d,0,a,b,0x08)); }
static void e_fmvwx(rv64_mod_t*V,int fd,int rs){ ew(V,mk_R(0x53,fd,0,rs,0,0x78)); }   /* GPR bits -> f32 reg */

/* load a 32-bit signed immediate. Small ones fit in an addi; larger ones
 * need the lui/addi two-step, because asking for a 32-bit constant in one
 * 32-bit instruction was apparently a bridge too far. */
static void e_li(rv64_mod_t*V,int d,int32_t v){
    if (v>=-2048 && v<2048){ e_addi(V,d,V_ZERO,v); return; }
    int32_t hi=(v+0x800)>>12, lo=v-(hi<<12);
    e_lui(V,d,hi&0xFFFFF);
    if (lo) e_addi(V,d,d,lo);
}

static int32_t slot(rv64_mod_t*V,uint32_t i){ return V->slots[i]; }

/* Slots hang off s0. The catch: ld/sd carry a signed 12-bit offset, so a
 * frame fatter than ~2KB puts the deeper slots out of reach. When that
 * happens we compute the address the long way through t2, which is slower
 * but, crucially, not wrong. */
static int fits12(int32_t o){ return o>=-2048 && o<2048; }
static void ld_slot(rv64_mod_t*V,int r,int32_t o){
    if (fits12(o)){ e_ld(V,r,V_S0,o); }
    else { e_li(V,V_T2,o); e_add(V,V_T2,V_S0,V_T2); e_ld(V,r,V_T2,0); }
}
static void st_slot(rv64_mod_t*V,int r,int32_t o){
    if (fits12(o)){ e_sd(V,V_S0,r,o); }
    else { e_li(V,V_T2,o); e_add(V,V_T2,V_S0,V_T2); e_sd(V,V_T2,r,0); }
}

/* a value into an integer reg: a constant becomes a literal, otherwise we
 * fetch its 64-bit slot. */
static void load_val(rv64_mod_t*V,int r,uint32_t v){
    if (BIR_VAL_IS_CONST(v)){ uint32_t c=BIR_VAL_INDEX(v);
        e_li(V,r,(int32_t)(c<V->M->num_consts?V->M->consts[c].d.ival:0)); }
    else ld_slot(V,r,slot(V,BIR_VAL_INDEX(v)));
}

/* a value into a float reg. A float constant gets its bits loaded into a
 * GPR and shuffled across with fmv.w.x; a real value is just an flw from
 * its slot (the slot holds the bits either way). */
static void load_fval(rv64_mod_t*V,int fr,uint32_t v){
    if (BIR_VAL_IS_CONST(v)){
        uint32_t c=BIR_VAL_INDEX(v); uint32_t bits=0;
        if (c<V->M->num_consts){ float fv=(float)V->M->consts[c].d.fval; memcpy(&bits,&fv,4); }
        e_li(V,V_T0,(int32_t)bits); e_fmvwx(V,fr,V_T0);
    } else {
        int32_t o=slot(V,BIR_VAL_INDEX(v));
        if (fits12(o)){ e_flw(V,fr,V_S0,o); }
        else { e_li(V,V_T2,o); e_add(V,V_T2,V_S0,V_T2); e_flw(V,fr,V_T2,0); }
    }
}
static void fst_slot(rv64_mod_t*V,int fr,int32_t o){
    if (fits12(o)){ e_fsw(V,V_S0,fr,o); }
    else { e_li(V,V_T2,o); e_add(V,V_T2,V_S0,V_T2); e_fsw(V,V_T2,fr,0); }
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
static uint32_t val_type(const rv64_mod_t*V,uint32_t v){
    uint32_t i=BIR_VAL_INDEX(v);
    if (BIR_VAL_IS_CONST(v)) return i<V->M->num_consts?V->M->consts[i].type:0;
    return i<V->M->num_insts?V->M->insts[i].type:0;
}
static int param_is_float(const rv64_mod_t*V,const bir_func_t*F,uint16_t p){
    if (F->type>=V->M->num_types) return 0;
    const bir_type_t*t=&V->M->types[F->type];
    if (t->kind!=BIR_TYPE_FUNC || p>=t->num_fields) return 0;
    uint32_t pt=V->M->type_fields[t->count+p];
    if (pt>=V->M->num_types) return 0;
    int k=V->M->types[pt].kind; return k==BIR_TYPE_FLOAT||k==BIR_TYPE_BFLOAT;
}

/* ---- phi: the incoming value for a given predecessor block ---- */
static uint32_t phi_incoming(const rv64_mod_t*V,const bir_inst_t*P,uint32_t pred){
    if (P->num_operands==BIR_OPERANDS_OVERFLOW){
        uint32_t st=P->operands[0],cnt=P->operands[1];
        for(uint32_t j=0;j+1<cnt;j+=2) if(V->M->extra_operands[st+j]==pred) return V->M->extra_operands[st+j+1];
    } else {
        for(uint32_t j=0;j+1<P->num_operands;j+=2) if(P->operands[j]==pred) return P->operands[j+1];
    }
    return BIR_VAL_NONE;
}
/* on the edge pred->succ, drop each of succ's phi values into its slot
 * before we jump. Phis cluster at the head of a block, so we stop at the
 * first one that isn't. */
static void emit_phi_copies(rv64_mod_t*V,uint32_t succ,uint32_t pred){
    if (succ>=V->M->num_blocks) return;
    const bir_block_t*B=&V->M->blocks[succ];
    for(uint32_t i=0;i<B->num_insts;i++){ uint32_t ix=B->first_inst+i; const bir_inst_t*P=&V->M->insts[ix];
        if (P->op!=BIR_PHI) break;
        uint32_t v=phi_incoming(V,P,pred); if(v==BIR_VAL_NONE) continue;
        load_val(V,V_T0,v); st_slot(V,V_T0,slot(V,ix));
    }
}

/* record a jal at the current spot that wants to land at block `blk`;
 * patched once every block has an address. */
static void jal_block(rv64_mod_t*V,uint32_t blk){
    if (V->n_fix<RV_FIX_MAX){ V->fix[V->n_fix].off=V->codelen; V->fix[V->n_fix].blk=blk; V->fix[V->n_fix].kind=0; V->n_fix++; }
    e_jal(V,V_ZERO,0);
}

#define RV_RET_MAX 256

static void rv64_func(rv64_mod_t *V,const bir_func_t *F){
    int is_kernel=(F->cuda_flags&CUDA_GLOBAL)!=0;

    /* Frame, decrement-then-assign so nothing overlaps. ra and saved s0 go
     * at the very bottom (sp+0, sp+8) where the offsets are always tiny;
     * the slots fill the space above. Allocas grab a backing region sized
     * from their pointee, same as the x86 side. */
    int32_t off=0;
    for (uint16_t p=0;p<F->num_params;p++){ off-=8; V->slots[p]=off; }
    int na=0;
    for (uint16_t b=0;b<F->num_blocks;b++){
        const bir_block_t*B=&V->M->blocks[F->first_block+b];
        for(uint32_t i=0;i<B->num_insts;i++){ uint32_t ix=B->first_inst+i; off-=8; V->slots[ix]=off;
            const bir_inst_t*I=&V->M->insts[ix];
            if ((I->op==BIR_ALLOCA||I->op==BIR_SHARED_ALLOC) && na<RV_ALLOCA_MAX){
                uint32_t pte=(I->type<V->M->num_types)?V->M->types[I->type].inner:0;
                int sz=8; if (pte<V->M->num_types){ const bir_type_t*t=&V->M->types[pte];
                    if (t->kind==BIR_TYPE_ARRAY) sz=(int)t->count*( (t->inner<V->M->num_types&&V->M->types[t->inner].width)?(int)(V->M->types[t->inner].width/8):4 );
                    else if (t->width) sz=(int)(t->width/8); }
                sz=(sz+7)&~7; if(sz<8)sz=8; off-=sz; V->alloca_off[na++]=off;
            }
        }
    }
    int na_emit=0;
    off-=8; int32_t tid_off=off;
    off-=8; int32_t ntid_off=off;
    int32_t frame=(((-off)+15)&~15)+16;   /* +16 for ra/s0 parked at the bottom */

    /* prologue */
    e_addi(V,V_SP,V_SP,-frame);
    e_sd(V,V_SP,V_RA,0);
    e_sd(V,V_SP,V_S0,8);
    e_addi(V,V_S0,V_SP,frame);

    /* params: int/pointer in a0..a7, float in fa0..fa7, each on its own
     * tally, stack overflow above s0. The hidden nthreads tags along as an
     * integer. (RV's calling convention, like a sensible queue, keeps the
     * two kinds of customer in separate lines.) */
    {
        int gi=0, fi=0; int32_t stk=0;
        uint16_t total=(uint16_t)(F->num_params + (is_kernel?1:0));
        for (uint16_t p=0;p<total;p++){
            int isf=(p<F->num_params) && param_is_float(V,F,p);
            int32_t dest=(p<F->num_params)?V->slots[p]:ntid_off;
            if (isf){ if(fi<8) fst_slot(V,V_FA0+fi++,dest); else { ld_slot(V,V_T0,0); st_slot(V,V_T0,dest); stk+=8; } }
            else { if(gi<8) st_slot(V,V_A0+gi++,dest); else { e_li(V,V_T2,stk); e_add(V,V_T2,V_S0,V_T2); e_ld(V,V_T0,V_T2,0); st_slot(V,V_T0,dest); stk+=8; } }
        }
    }

    uint32_t loop_head=0;
    if (is_kernel){
        e_li(V,V_T0,0); st_slot(V,V_T0,tid_off);                    /* tid = 0 */
        loop_head=V->codelen;
        ld_slot(V,V_T0,tid_off); ld_slot(V,V_T1,ntid_off);
        e_blt(V,V_T0,V_T1,8);                                       /* tid<ntid? skip the bail-out jal */
        jal_block(V,F->first_block+F->num_blocks);                  /* sentinel: "loop end", patched below */
    }

    uint32_t retfix[RV_RET_MAX]; int n_ret=0;
    for (uint16_t b=0;b<F->num_blocks;b++){
        const bir_block_t*B=&V->M->blocks[F->first_block+b]; V->blk_off[F->first_block+b]=V->codelen;
        for (uint32_t i=0;i<B->num_insts;i++){ uint32_t ix=B->first_inst+i; const bir_inst_t*I=&V->M->insts[ix]; int32_t s=slot(V,ix);
        switch(I->op){
        case BIR_PARAM: break;
        case BIR_ALLOCA: case BIR_SHARED_ALLOC:
            if (na_emit<RV_ALLOCA_MAX){ int32_t o=V->alloca_off[na_emit++]; e_li(V,V_T0,o); e_add(V,V_T0,V_S0,V_T0); st_slot(V,V_T0,s); }
            break;
        case BIR_THREAD_ID: if(is_kernel){ ld_slot(V,V_T0,tid_off); } else e_li(V,V_T0,0); st_slot(V,V_T0,s); break;
        case BIR_BLOCK_ID:  e_li(V,V_T0,0); st_slot(V,V_T0,s); break;
        case BIR_BLOCK_DIM: if(is_kernel){ ld_slot(V,V_T0,ntid_off); } else e_li(V,V_T0,1); st_slot(V,V_T0,s); break;
        case BIR_GRID_DIM:  e_li(V,V_T0,1); st_slot(V,V_T0,s); break;
        case BIR_ADD: load_val(V,V_T0,I->operands[0]);load_val(V,V_T1,I->operands[1]);e_add(V,V_T0,V_T0,V_T1);st_slot(V,V_T0,s); break;
        case BIR_SUB: load_val(V,V_T0,I->operands[0]);load_val(V,V_T1,I->operands[1]);e_sub(V,V_T0,V_T0,V_T1);st_slot(V,V_T0,s); break;
        case BIR_MUL: load_val(V,V_T0,I->operands[0]);load_val(V,V_T1,I->operands[1]);e_mul(V,V_T0,V_T0,V_T1);st_slot(V,V_T0,s); break;
        case BIR_FADD: load_fval(V,V_FT0,I->operands[0]);load_fval(V,V_FT1,I->operands[1]);e_fadds(V,V_FT0,V_FT0,V_FT1);fst_slot(V,V_FT0,s); break;
        case BIR_FSUB: load_fval(V,V_FT0,I->operands[0]);load_fval(V,V_FT1,I->operands[1]);e_fsubs(V,V_FT0,V_FT0,V_FT1);fst_slot(V,V_FT0,s); break;
        case BIR_FMUL: load_fval(V,V_FT0,I->operands[0]);load_fval(V,V_FT1,I->operands[1]);e_fmuls(V,V_FT0,V_FT0,V_FT1);fst_slot(V,V_FT0,s); break;
        case BIR_GEP: { int sz=pointee_sz(V,I->type); load_val(V,V_T0,I->operands[1]); e_li(V,V_T1,sz); e_mul(V,V_T0,V_T0,V_T1); load_val(V,V_T1,I->operands[0]); e_add(V,V_T0,V_T0,V_T1); st_slot(V,V_T0,s); break; }
        case BIR_LOAD: { load_val(V,V_T0,I->operands[0]);
            const bir_type_t*t=(I->type<V->M->num_types)?&V->M->types[I->type]:0; int w=t?(int)t->width:32;
            if (w==64) e_ld(V,V_T0,V_T0,0); else if (w==16) e_lh(V,V_T0,V_T0,0); else if (w==8||w==1) e_lb(V,V_T0,V_T0,0); else e_lw(V,V_T0,V_T0,0);
            st_slot(V,V_T0,s); break; }
        case BIR_STORE: { load_val(V,V_T0,I->operands[0]); load_val(V,V_T1,I->operands[1]);
            int w=pointee_sz(V,val_type(V,I->operands[1]))*8;
            if (w==64) e_sd(V,V_T1,V_T0,0); else if (w==16) e_sh(V,V_T1,V_T0,0); else if (w==8) e_sb(V,V_T1,V_T0,0); else e_sw(V,V_T1,V_T0,0);
            break; }
        case BIR_ICMP: { load_val(V,V_T0,I->operands[0]); load_val(V,V_T1,I->operands[1]);
            switch(I->subop){
            case BIR_ICMP_SLT: e_slt(V,V_T0,V_T0,V_T1); break;
            case BIR_ICMP_SGT: e_slt(V,V_T0,V_T1,V_T0); break;
            case BIR_ICMP_SLE: e_slt(V,V_T0,V_T1,V_T0); e_xori(V,V_T0,V_T0,1); break;
            case BIR_ICMP_SGE: e_slt(V,V_T0,V_T0,V_T1); e_xori(V,V_T0,V_T0,1); break;
            case BIR_ICMP_ULT: e_sltu(V,V_T0,V_T0,V_T1); break;
            case BIR_ICMP_UGT: e_sltu(V,V_T0,V_T1,V_T0); break;
            case BIR_ICMP_ULE: e_sltu(V,V_T0,V_T1,V_T0); e_xori(V,V_T0,V_T0,1); break;
            case BIR_ICMP_UGE: e_sltu(V,V_T0,V_T0,V_T1); e_xori(V,V_T0,V_T0,1); break;
            case BIR_ICMP_EQ:  e_sub(V,V_T0,V_T0,V_T1); e_sltiu(V,V_T0,V_T0,1); break;
            default:           e_sub(V,V_T0,V_T0,V_T1); e_sltu(V,V_T0,V_ZERO,V_T0); break; /* NE */
            } st_slot(V,V_T0,s); break; }
        case BIR_BR: { uint32_t cur=F->first_block+b; emit_phi_copies(V,I->operands[0],cur); jal_block(V,I->operands[0]); break; }
        case BIR_BR_COND: { uint32_t cur=F->first_block+b;
            load_val(V,V_T0,I->operands[0]);
            uint32_t beq_off=V->codelen; e_beq(V,V_T0,V_ZERO,0);    /* cond==0 -> false_pad (local, patched) */
            emit_phi_copies(V,I->operands[1],cur); jal_block(V,I->operands[1]);
            { uint32_t w=mk_B(0x63,0,V_T0,V_ZERO,(int32_t)V->codelen-(int32_t)beq_off);
              V->code[beq_off]=(uint8_t)w;V->code[beq_off+1]=(uint8_t)(w>>8);V->code[beq_off+2]=(uint8_t)(w>>16);V->code[beq_off+3]=(uint8_t)(w>>24); }
            emit_phi_copies(V,I->operands[2],cur); jal_block(V,I->operands[2]); break; }
        case BIR_PHI: break; /* the work happens on the edges, not here */
        case BIR_RET:
            if (is_kernel){ if(n_ret<RV_RET_MAX) retfix[n_ret++]=V->codelen; e_jal(V,V_ZERO,0); } /* -> loop_cont */
            else { e_ld(V,V_RA,V_SP,0); e_ld(V,V_S0,V_SP,8); e_addi(V,V_SP,V_SP,frame); e_jalr(V,V_ZERO,V_RA,0); }
            break;
        default: e_li(V,V_T0,0); st_slot(V,V_T0,s); break;
        }}
    }

    uint32_t loop_end_blk = F->first_block+F->num_blocks;   /* sentinel index for "past the last block" */
    if (is_kernel){
        uint32_t loop_cont=V->codelen;
        ld_slot(V,V_T0,tid_off); e_addi(V,V_T0,V_T0,1); st_slot(V,V_T0,tid_off);
        e_jal(V,V_ZERO,(int32_t)loop_head-(int32_t)V->codelen);
        V->blk_off[loop_end_blk]=V->codelen;                 /* loop end lands here */
        for (int k=0;k<n_ret;k++){ uint32_t o=retfix[k]; uint32_t w=mk_J(0x6F,V_ZERO,(int32_t)loop_cont-(int32_t)o);
          V->code[o]=(uint8_t)w;V->code[o+1]=(uint8_t)(w>>8);V->code[o+2]=(uint8_t)(w>>16);V->code[o+3]=(uint8_t)(w>>24); }
    }
    /* epilogue */
    e_ld(V,V_RA,V_SP,0); e_ld(V,V_S0,V_SP,8); e_addi(V,V_SP,V_SP,frame); e_jalr(V,V_ZERO,V_RA,0);

    /* resolve every jal-to-block now that the blocks have addresses */
    for (int k=0;k<V->n_fix;k++){ uint32_t bo=V->blk_off[V->fix[k].blk]; uint32_t o=V->fix[k].off;
        uint32_t w=mk_J(0x6F,V_ZERO,(int32_t)bo-(int32_t)o);
        V->code[o]=(uint8_t)w;V->code[o+1]=(uint8_t)(w>>8);V->code[o+2]=(uint8_t)(w>>16);V->code[o+3]=(uint8_t)(w>>24); }
    V->n_fix=0;
}

void rv64_init(rv64_mod_t *V,const bir_module_t *M){ memset(V,0,sizeof(*V)); V->M=M; }
int rv64_emit(rv64_mod_t *V){ for(uint32_t f=0;f<V->M->num_funcs;f++) rv64_func(V,&V->M->funcs[f]); return 0; }
