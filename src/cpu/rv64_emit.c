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
#include "backend.h"
#include "barracuda.h"
#include <string.h>

static void rv_cap(rv64_mod_t *V,const char *what,unsigned lim){
    V->n_errs++;
    if(V->capr) return;
    V->capr=1;
    (void)be_fail(BC_E545,"cpu-rv64",what,lim);
}
static void ew(rv64_mod_t *V, uint32_t w){
    if (V->codelen+4<=RV_CODE_MAX){
        V->code[V->codelen++]=(uint8_t)w; V->code[V->codelen++]=(uint8_t)(w>>8);
        V->code[V->codelen++]=(uint8_t)(w>>16); V->code[V->codelen++]=(uint8_t)(w>>24);
    }
    else rv_cap(V,"the code buffer",(unsigned)RV_CODE_MAX);
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
static void e_mulhu(rv64_mod_t*V,int d,int a,int b){ ew(V,mk_R(0x33,d,3,a,b,0x01)); }
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
static void e_bne (rv64_mod_t*V,int a,int b,int o){ ew(V,mk_B(0x63,1,a,b,o)); }
static void e_blt (rv64_mod_t*V,int a,int b,int o){ ew(V,mk_B(0x63,4,a,b,o)); }
/* atomic memory op (A extension): rd = *rs1; *rs1 = rd OP rs2. funct7 is
 * the op's funct5 shifted up, with aq/rl left at zero. */
static void e_amo(rv64_mod_t*V,int f5,int w64,int rd,int addr,int val){ ew(V,mk_R(0x2F,rd,w64?3:2,addr,val,f5<<2)); }

/* ---- Integer ALU. RISC-V keeps it tidy: bitwise and the full-width
 * shifts on opcode 0x33, and the narrow *W cousins on 0x3B, which chew on
 * the low 32 bits and sign-extend the answer back to 64, exactly the canon
 * a sign-extended slot wants. Divide and remainder ride the M extension
 * (funct7 = 0x01). ---- */
static void e_and (rv64_mod_t*V,int d,int a,int b){ ew(V,mk_R(0x33,d,7,a,b,0x00)); }
static void e_or  (rv64_mod_t*V,int d,int a,int b){ ew(V,mk_R(0x33,d,6,a,b,0x00)); }
static void e_xor (rv64_mod_t*V,int d,int a,int b){ ew(V,mk_R(0x33,d,4,a,b,0x00)); }
static void e_sll (rv64_mod_t*V,int d,int a,int b){ ew(V,mk_R(0x33,d,1,a,b,0x00)); }
static void e_srl (rv64_mod_t*V,int d,int a,int b){ ew(V,mk_R(0x33,d,5,a,b,0x00)); }
static void e_sra (rv64_mod_t*V,int d,int a,int b){ ew(V,mk_R(0x33,d,5,a,b,0x20)); }
static void e_sllw(rv64_mod_t*V,int d,int a,int b){ ew(V,mk_R(0x3B,d,1,a,b,0x00)); }
static void e_srlw(rv64_mod_t*V,int d,int a,int b){ ew(V,mk_R(0x3B,d,5,a,b,0x00)); }
static void e_sraw(rv64_mod_t*V,int d,int a,int b){ ew(V,mk_R(0x3B,d,5,a,b,0x20)); }
static void e_div (rv64_mod_t*V,int d,int a,int b){ ew(V,mk_R(0x33,d,4,a,b,0x01)); }
static void e_divu(rv64_mod_t*V,int d,int a,int b){ ew(V,mk_R(0x33,d,5,a,b,0x01)); }
static void e_rem (rv64_mod_t*V,int d,int a,int b){ ew(V,mk_R(0x33,d,6,a,b,0x01)); }
static void e_remu(rv64_mod_t*V,int d,int a,int b){ ew(V,mk_R(0x33,d,7,a,b,0x01)); }
static void e_divw (rv64_mod_t*V,int d,int a,int b){ ew(V,mk_R(0x3B,d,4,a,b,0x01)); }
static void e_divuw(rv64_mod_t*V,int d,int a,int b){ ew(V,mk_R(0x3B,d,5,a,b,0x01)); }
static void e_remw (rv64_mod_t*V,int d,int a,int b){ ew(V,mk_R(0x3B,d,6,a,b,0x01)); }
static void e_remuw(rv64_mod_t*V,int d,int a,int b){ ew(V,mk_R(0x3B,d,7,a,b,0x01)); }

/* ---- single-precision float ops (the F extension; D is there too but BIR
 * floats are f32, so we live in .S land) ---- */
static void e_flw  (rv64_mod_t*V,int fd,int a,int o){ ew(V,mk_I(0x07,fd,2,a,o&0xFFF)); }
static void e_fsw  (rv64_mod_t*V,int a,int fs,int o){ ew(V,mk_S(0x27,2,a,fs,o&0xFFF)); }
static void e_fadds(rv64_mod_t*V,int d,int a,int b){ ew(V,mk_R(0x53,d,0,a,b,0x00)); }
static void e_fsubs(rv64_mod_t*V,int d,int a,int b){ ew(V,mk_R(0x53,d,0,a,b,0x04)); }
static void e_fmuls(rv64_mod_t*V,int d,int a,int b){ ew(V,mk_R(0x53,d,0,a,b,0x08)); }
static void e_fmvwx(rv64_mod_t*V,int fd,int rs){ ew(V,mk_R(0x53,fd,0,rs,0,0x78)); }   /* GPR bits -> f32 reg */
/* Float: divide, min/max, the three compares (result lands in a GPR,
 * 0 or 1), and the two fcvts the remainder and the casts both lean on. */
static void e_fdivs(rv64_mod_t*V,int d,int a,int b){ ew(V,mk_R(0x53,d,0,a,b,0x0C)); }
static void e_fmins(rv64_mod_t*V,int d,int a,int b){ ew(V,mk_R(0x53,d,0,a,b,0x14)); }
static void e_fmaxs(rv64_mod_t*V,int d,int a,int b){ ew(V,mk_R(0x53,d,1,a,b,0x14)); }
static void e_feqs (rv64_mod_t*V,int rd,int a,int b){ ew(V,mk_R(0x53,rd,2,a,b,0x50)); }
static void e_flts (rv64_mod_t*V,int rd,int a,int b){ ew(V,mk_R(0x53,rd,1,a,b,0x50)); }
static void e_fles (rv64_mod_t*V,int rd,int a,int b){ ew(V,mk_R(0x53,rd,0,a,b,0x50)); }
static void e_fcvtws(rv64_mod_t*V,int rd,int fs){ ew(V,mk_R(0x53,rd,1,fs,0,0x60)); }  /* float -> i32, round-to-zero */
static void e_fcvtsw(rv64_mod_t*V,int fd,int rs){ ew(V,mk_R(0x53,fd,0,rs,0,0x68)); }  /* i32 -> float */
/* Conversion kit: 64-bit fcvts (the slot holds the canonical
 * sign-extended value, so the .l forms take any int width), the bit-move,
 * and the shift/mask primitives the width changes are built from. */
static void e_fcvtsl (rv64_mod_t*V,int fd,int rs){ ew(V,mk_R(0x53,fd,0,rs,2,0x68)); } /* i64 -> float */
static void e_fcvtls (rv64_mod_t*V,int rd,int fs){ ew(V,mk_R(0x53,rd,1,fs,2,0x60)); } /* float -> i64, RTZ */
static void e_fcvtslu(rv64_mod_t*V,int fd,int rs){ ew(V,mk_R(0x53,fd,0,rs,3,0x68)); } /* u64 -> float */
static void e_fcvtlus(rv64_mod_t*V,int rd,int fs){ ew(V,mk_R(0x53,rd,1,fs,3,0x60)); } /* float -> u64, RTZ */
/* Math the F extension does outright: sqrt, and abs as fsgnjx of a value
 * with itself (sign XOR sign = 0). Rounding has no instruction of its own,
 * so it goes out to an int and back at the requested rounding mode. */
static void e_fsqrts(rv64_mod_t*V,int d,int s){ ew(V,mk_R(0x53,d,0,s,0,0x2C)); }
static void e_fabss (rv64_mod_t*V,int d,int s){ ew(V,mk_R(0x53,d,2,s,s,0x10)); }
static void e_fcvtws_rm(rv64_mod_t*V,int rd,int fs,int rm){ ew(V,mk_R(0x53,rd,rm,fs,0,0x60)); }
static void e_slli (rv64_mod_t*V,int d,int a,int sh){ ew(V,mk_I(0x13,d,1,a,sh&0x3F)); }
static void e_srli (rv64_mod_t*V,int d,int a,int sh){ ew(V,mk_I(0x13,d,5,a,sh&0x3F)); }
static void e_srai (rv64_mod_t*V,int d,int a,int sh){ ew(V,mk_I(0x13,d,5,a,(sh&0x3F)|0x400)); }
static void e_sextw(rv64_mod_t*V,int d,int a){ ew(V,mk_I(0x1B,d,0,a,0)); }            /* addiw d,a,0 */
static void e_andi (rv64_mod_t*V,int d,int a,int i){ ew(V,mk_I(0x13,d,7,a,i&0xFFF)); }

/* Canonicalise reg `r` by sign- or zero-extending up from a w-bit value. */
static void rv_sext_to(rv64_mod_t*V,int r,int w){
    if (w>=64) return;
    if (w==1){ e_andi(V,r,r,1); return; }
    if (w==32){ e_sextw(V,r,r); return; }
    int sh=64-w; e_slli(V,r,r,sh); e_srai(V,r,r,sh);
}
static void rv_zext_to(rv64_mod_t*V,int r,int w){
    if (w>=64) return;
    if (w==1){ e_andi(V,r,r,1); return; }
    if (w==8){ e_andi(V,r,r,0xFF); return; }
    int sh=64-w; e_slli(V,r,r,sh); e_srli(V,r,r,sh);
}

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

/* ---- Bit counting, with no Zbb to lean on ---- */

/* SWAR popcount over the low 32 bits of t0. No Zbb, but the multiplier the
 * SWAR tail needs is there. t0 in and out, t1 and t2 scratch. */
static void rv_popc32(rv64_mod_t *V)
{
    e_srli(V,V_T1,V_T0,1);
    e_li(V,V_T2,0x55555555); e_and(V,V_T1,V_T1,V_T2);
    e_sub(V,V_T0,V_T0,V_T1);
    e_srli(V,V_T1,V_T0,2);
    e_li(V,V_T2,0x33333333);
    e_and(V,V_T0,V_T0,V_T2); e_and(V,V_T1,V_T1,V_T2);
    e_add(V,V_T0,V_T0,V_T1);
    e_srli(V,V_T1,V_T0,4); e_add(V,V_T0,V_T0,V_T1);
    e_li(V,V_T2,0x0F0F0F0F); e_and(V,V_T0,V_T0,V_T2);
    e_li(V,V_T2,0x01010101); e_mul(V,V_T0,V_T0,V_T2);
    e_srli(V,V_T0,V_T0,24); e_andi(V,V_T0,V_T0,0xFF);
}

/* Five rounds of the same swap, halving the block each time. */
static void rv_brev32(rv64_mod_t *V)
{
    static const struct { int sh; int32_t m; } r[5] = {
        { 1, 0x55555555 }, { 2, 0x33333333 }, { 4, 0x0F0F0F0F },
        { 8, 0x00FF00FF }, { 16, 0x0000FFFF }
    };
    for (int i = 0; i < 5; i++) {
        e_srli(V,V_T1,V_T0,r[i].sh);
        e_li(V,V_T2,r[i].m);
        e_and(V,V_T1,V_T1,V_T2); e_and(V,V_T0,V_T0,V_T2);
        e_slli(V,V_T0,V_T0,r[i].sh);
        e_or(V,V_T0,V_T0,V_T1);
    }
}

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
        if (c<V->M->num_consts && V->M->consts[c].kind==BIR_CONST_FLOAT){
            /* float const's union holds a double; narrow to f32 and take
             * those bits, the shape a store or bit-shuffle expects. */
            uint32_t bits=0; float fv=(float)V->M->consts[c].d.fval; memcpy(&bits,&fv,4);
            e_li(V,r,(int32_t)bits);
        } else {
            e_li(V,r,(int32_t)(c<V->M->num_consts?V->M->consts[c].d.ival:0));
        }
    }
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

/* The raw 32 bits of a float value, in a GPR (zero-extended via lwu), and
 * the way back. SELECT blends floats by their bits so it never has to touch
 * a float register. */
static void ld_fbits(rv64_mod_t*V,int r,uint32_t v){
    if (BIR_VAL_IS_CONST(v)){
        uint32_t c=BIR_VAL_INDEX(v); uint32_t bits=0;
        if (c<V->M->num_consts){ float fv=(float)V->M->consts[c].d.fval; memcpy(&bits,&fv,4); }
        e_li(V,r,(int32_t)bits);
    } else {
        int32_t o=slot(V,BIR_VAL_INDEX(v));
        if (fits12(o)){ ew(V,mk_I(0x03,r,6,V_S0,o&0xFFF)); }                 /* lwu r, o(s0) */
        else { e_li(V,V_T2,o); e_add(V,V_T2,V_S0,V_T2); ew(V,mk_I(0x03,r,6,V_T2,0)); }
    }
}
static void st_fbits(rv64_mod_t*V,int r,int32_t o){
    if (fits12(o)){ e_sw(V,V_S0,r,o); }
    else { e_li(V,V_T2,o); e_add(V,V_T2,V_S0,V_T2); e_sw(V,V_T2,r,0); }
}

/* Intern an external symbol name, de-duplicated. */
static uint32_t rv_extsym(rv64_mod_t*V,const char *name){
    for (int i=0;i<V->n_extsym;i++) if(!strcmp(V->extsym[i],name)) return (uint32_t)i;
    if (V->n_extsym>=RV_EXTSYM_MAX){
        rv_cap(V,"external symbols",(unsigned)RV_EXTSYM_MAX);
        return 0;
    }
    int idx=V->n_extsym++;
    size_t n=strlen(name); if(n>=RV_EXTSYM_LEN)n=RV_EXTSYM_LEN-1;
    memcpy(V->extsym[idx],name,n); V->extsym[idx][n]='\0';
    return (uint32_t)idx;
}

/* Call an external symbol with auipc + jalr, the pair the linker patches
 * from a single R_RISCV_CALL_PLT on the auipc. The frame is already
 * 16-aligned and ra is restored at the epilogue, so the call clobbering ra
 * and t1 mid-body bothers nobody. Arg in fa0, result back in fa0. */
static void emit_call_ext(rv64_mod_t*V,const char *name){
    uint32_t sym=rv_extsym(V,name);
    if (V->n_reloc<RV_RELOC_MAX){ V->reloc[V->n_reloc].off=V->codelen; V->reloc[V->n_reloc].sym=sym; V->n_reloc++; }
    else rv_cap(V,"relocations",(unsigned)RV_RELOC_MAX);
    ew(V,mk_U(0x17,V_T1,0));        /* auipc t1, 0  (the reloc target) */
    e_jalr(V,V_RA,V_T1,0);          /* jalr ra, t1, 0 */
}

/* A plain one-argument libm call: float in fa0, float out fa0. */
static void call_libm1(rv64_mod_t*V,const char *name,uint32_t op0,int32_t s){
    load_fval(V,V_FA0,op0);
    emit_call_ext(V,name);
    fst_slot(V,V_FA0,s);
}

static const char *tystr(const rv64_mod_t *V,uint32_t ty){
    static char buf[128];
    if (bir_type_str(V->M,ty,buf,(int)sizeof buf)<=0) buf[0]=0;
    return buf;
}

static int type_size(const rv64_mod_t*V,uint32_t ty){
    return (int)bir_bsz(V->M,ty,8);
}

static int pointee_sz(const rv64_mod_t*V,uint32_t ty){
    return (int)bir_gstr(V->M,ty,8);
}
/* 64 for an i64, 32 for anything narrower; the shifts and the divider use
 * it to pick the *W word ops so a 32-bit value shifts like a 32-bit value. */
static int int_w(const rv64_mod_t*V,uint32_t ty){
    if (ty<V->M->num_types && V->M->types[ty].kind==BIR_TYPE_INT)
        return V->M->types[ty].width>=64?64:32;
    return 64;
}
/* exact bit width of a type, for the conversions */
static int tw(const rv64_mod_t*V,uint32_t ty){
    if (ty<V->M->num_types){ uint16_t w=V->M->types[ty].width; if(w)return(int)w;
        if(V->M->types[ty].kind==BIR_TYPE_PTR)return 64; }
    return 32;
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

static uint32_t ptype(const rv64_mod_t*V,const bir_func_t*F,uint16_t p){
    if (F->type>=V->M->num_types) return 0;
    const bir_type_t*t=&V->M->types[F->type];
    if (t->kind!=BIR_TYPE_FUNC || p>=t->num_fields) return 0;
    return V->M->type_fields[t->count+p];
}
static int isflt(const rv64_mod_t*V,uint32_t ty){
    if (ty>=V->M->num_types) return 0;
    int k=V->M->types[ty].kind; return k==BIR_TYPE_FLOAT||k==BIR_TYPE_BFLOAT;
}

static int argok(const rv64_mod_t*V,uint32_t ty){
    if (ty>=V->M->num_types) return 0;
    switch (V->M->types[ty].kind){
    case BIR_TYPE_PTR: return 1;
    case BIR_TYPE_INT: case BIR_TYPE_FLOAT: case BIR_TYPE_BFLOAT:
        return V->M->types[ty].width<=64;
    default: return 0;
    }
}

#define RV_ONSTK (-1)
static int arloc(int isf,int *gi,int *fi){
    if (isf && *fi<8) return 32+V_FA0+(*fi)++;
    if (*gi<8) return V_A0+(*gi)++;
    return RV_ONSTK;
}

static int32_t pslot(const rv64_mod_t*V,const bir_func_t*F,uint16_t p){
    if (F->num_blocks==0) return 0;
    const bir_block_t*B=&V->M->blocks[F->first_block];
    for (uint32_t i=0;i<B->num_insts;i++){
        uint32_t ix=B->first_inst+i;
        if (ix>=V->M->num_insts || ix>=BIR_MAX_INSTS) break;
        if (V->M->insts[ix].op==BIR_PARAM && V->M->insts[ix].subop==p) return V->slots[ix];
    }
    return 0;
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

static void pat32(rv64_mod_t*V,uint32_t off,uint32_t w){
    if (off+4>RV_CODE_MAX) return;
    V->code[off]=(uint8_t)w; V->code[off+1]=(uint8_t)(w>>8);
    V->code[off+2]=(uint8_t)(w>>16); V->code[off+3]=(uint8_t)(w>>24);
}

static int jfits(int32_t o){ return o>=-1048576 && o<=1048574 && !(o&1); }
static void pjal(rv64_mod_t*V,uint32_t off,int rd,int32_t rel,const char*what){
    if (!jfits(rel)){ (void)be_fail(BC_E580,"cpu-rv64",what,(int)rel); V->n_errs++; return; }
    pat32(V,off,mk_J(0x6F,rd,rel));
}

/* record a jal at the current spot that wants to land at block `blk`;
 * patched once every block has an address. */
static void jal_block(rv64_mod_t*V,uint32_t blk){
    if (V->n_fix<RV_FIX_MAX){ V->fix[V->n_fix].off=V->codelen; V->fix[V->n_fix].blk=blk; V->fix[V->n_fix].kind=0; V->n_fix++; }
    else rv_cap(V,"branch fixups",(unsigned)RV_FIX_MAX);
    e_jal(V,V_ZERO,0);
}

#define RV_RET_MAX 256

static void rv64_func(rv64_mod_t *V,const bir_func_t *F){
    int is_kernel=(F->cuda_flags&CUDA_GLOBAL)!=0;
    { uint32_t fx=(uint32_t)(F-V->M->funcs); if (fx<BIR_MAX_FUNCS) V->func_off[fx]=V->codelen; }

    /* Frame, decrement-then-assign so nothing overlaps. ra and saved s0 go
     * at the very bottom (sp+0, sp+8) where the offsets are always tiny;
     * the slots fill the space above. Allocas grab a backing region sized
     * from their pointee, same as the x86 side. */
    int32_t off=0;
    int na=0;
    for (uint16_t b=0;b<F->num_blocks;b++){
        const bir_block_t*B=&V->M->blocks[F->first_block+b];
        for(uint32_t i=0;i<B->num_insts;i++){ uint32_t ix=B->first_inst+i; off-=8; V->slots[ix]=off;
            const bir_inst_t*I=&V->M->insts[ix];
            if ((I->op==BIR_ALLOCA||I->op==BIR_SHARED_ALLOC) && na>=RV_ALLOCA_MAX)
                rv_cap(V,"stack allocations",(unsigned)RV_ALLOCA_MAX);
            if ((I->op==BIR_ALLOCA||I->op==BIR_SHARED_ALLOC) && na<RV_ALLOCA_MAX){
                uint32_t pte=(I->type<V->M->num_types)?V->M->types[I->type].inner:0;
                int sz=type_size(V,pte);
                if(!sz){ (void)be_fail(BC_E540,"cpu-rv64","alloca",tystr(V,pte)); V->n_errs++; }
                int aln=1<<(I->subop&31); if(aln<8)aln=8;
                sz=(sz+7)&~7; if(sz<8)sz=8; off-=sz; off&=-aln;
                V->alloca_off[na++]=off;
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
            int32_t dest=(p<F->num_params)?pslot(V,F,p):ntid_off;
            int r=arloc(isf,&gi,&fi);
            if (r==RV_ONSTK){
                if (dest){ e_li(V,V_T2,stk); e_add(V,V_T2,V_S0,V_T2);
                           e_ld(V,V_T0,V_T2,0); st_slot(V,V_T0,dest); }
                stk+=8;
            }
            else if (dest){ if (r>=32) fst_slot(V,r-32,dest); else st_slot(V,r,dest); }
        }
    }

    uint32_t loop_head=0, bail_off=0;
    if (is_kernel){
        e_li(V,V_T0,0); st_slot(V,V_T0,tid_off);                    /* tid = 0 */
        loop_head=V->codelen;
        ld_slot(V,V_T0,tid_off); ld_slot(V,V_T1,ntid_off);
        e_blt(V,V_T0,V_T1,8);                                       /* tid<ntid? skip the next jal */
        bail_off=V->codelen; e_jal(V,V_ZERO,0);                     /* tid>=ntid: bail to loop end (patched) */
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
        case BIR_UMULHI: load_val(V,V_T0,I->operands[0]);load_val(V,V_T1,I->operands[1]);e_mulhu(V,V_T0,V_T0,V_T1);st_slot(V,V_T0,s); break;

        /* ---- bit counting ---- */
        case BIR_POPCOUNT: case BIR_CTZ: case BIR_CLZ: case BIR_BREV: {
            if (int_w(V,val_type(V,I->operands[0]))!=32){
                (void)be_fail(BC_E541,"cpu-rv64",
                              "bit counting at a width other than 32");
                V->n_errs++; break;
            }
            load_val(V,V_T0,I->operands[0]); rv_zext_to(V,V_T0,32);
            if (I->op==BIR_POPCOUNT) rv_popc32(V);
            else if (I->op==BIR_BREV) rv_brev32(V);
            else if (I->op==BIR_CTZ){
                /* popcount(~x & (x-1)) is ctz, and zero walks to 32 unaided */
                e_addi(V,V_T1,V_T0,-1); e_xori(V,V_T0,V_T0,-1);
                e_and(V,V_T0,V_T0,V_T1); rv_zext_to(V,V_T0,32);
                rv_popc32(V);
            } else {
                /* Smear the top set bit down; what stays uncounted is clz */
                for (int sh=1; sh<32; sh<<=1){
                    e_srli(V,V_T1,V_T0,sh); e_or(V,V_T0,V_T0,V_T1);
                }
                rv_popc32(V);
                e_li(V,V_T1,32); e_sub(V,V_T0,V_T1,V_T0);
            }
            st_slot(V,V_T0,s); break;
        }

        /* ---- integer bitwise + width-aware shift/divide ---- */
        case BIR_AND: load_val(V,V_T0,I->operands[0]);load_val(V,V_T1,I->operands[1]);e_and(V,V_T0,V_T0,V_T1);st_slot(V,V_T0,s); break;
        case BIR_OR:  load_val(V,V_T0,I->operands[0]);load_val(V,V_T1,I->operands[1]);e_or (V,V_T0,V_T0,V_T1);st_slot(V,V_T0,s); break;
        case BIR_XOR: load_val(V,V_T0,I->operands[0]);load_val(V,V_T1,I->operands[1]);e_xor(V,V_T0,V_T0,V_T1);st_slot(V,V_T0,s); break;
        case BIR_SHL: load_val(V,V_T0,I->operands[0]);load_val(V,V_T1,I->operands[1]); if(int_w(V,I->type)==64)e_sll(V,V_T0,V_T0,V_T1);else e_sllw(V,V_T0,V_T0,V_T1); st_slot(V,V_T0,s); break;
        case BIR_LSHR:load_val(V,V_T0,I->operands[0]);load_val(V,V_T1,I->operands[1]); if(int_w(V,I->type)==64)e_srl(V,V_T0,V_T0,V_T1);else e_srlw(V,V_T0,V_T0,V_T1); st_slot(V,V_T0,s); break;
        case BIR_ASHR:load_val(V,V_T0,I->operands[0]);load_val(V,V_T1,I->operands[1]); if(int_w(V,I->type)==64)e_sra(V,V_T0,V_T0,V_T1);else e_sraw(V,V_T0,V_T0,V_T1); st_slot(V,V_T0,s); break;
        case BIR_SDIV:load_val(V,V_T0,I->operands[0]);load_val(V,V_T1,I->operands[1]); if(int_w(V,I->type)==64)e_div (V,V_T0,V_T0,V_T1);else e_divw (V,V_T0,V_T0,V_T1); st_slot(V,V_T0,s); break;
        case BIR_UDIV:load_val(V,V_T0,I->operands[0]);load_val(V,V_T1,I->operands[1]); if(int_w(V,I->type)==64)e_divu(V,V_T0,V_T0,V_T1);else e_divuw(V,V_T0,V_T0,V_T1); st_slot(V,V_T0,s); break;
        case BIR_SREM:load_val(V,V_T0,I->operands[0]);load_val(V,V_T1,I->operands[1]); if(int_w(V,I->type)==64)e_rem (V,V_T0,V_T0,V_T1);else e_remw (V,V_T0,V_T0,V_T1); st_slot(V,V_T0,s); break;
        case BIR_UREM:load_val(V,V_T0,I->operands[0]);load_val(V,V_T1,I->operands[1]); if(int_w(V,I->type)==64)e_remu(V,V_T0,V_T0,V_T1);else e_remuw(V,V_T0,V_T0,V_T1); st_slot(V,V_T0,s); break;
        case BIR_FADD: load_fval(V,V_FT0,I->operands[0]);load_fval(V,V_FT1,I->operands[1]);e_fadds(V,V_FT0,V_FT0,V_FT1);fst_slot(V,V_FT0,s); break;
        case BIR_FSUB: load_fval(V,V_FT0,I->operands[0]);load_fval(V,V_FT1,I->operands[1]);e_fsubs(V,V_FT0,V_FT0,V_FT1);fst_slot(V,V_FT0,s); break;
        case BIR_FMUL: load_fval(V,V_FT0,I->operands[0]);load_fval(V,V_FT1,I->operands[1]);e_fmuls(V,V_FT0,V_FT0,V_FT1);fst_slot(V,V_FT0,s); break;

        /* ---- float divide, min/max ---- */
        case BIR_FDIV: load_fval(V,V_FT0,I->operands[0]);load_fval(V,V_FT1,I->operands[1]);e_fdivs(V,V_FT0,V_FT0,V_FT1);fst_slot(V,V_FT0,s); break;
        case BIR_FMAX: load_fval(V,V_FT0,I->operands[0]);load_fval(V,V_FT1,I->operands[1]);e_fmaxs(V,V_FT0,V_FT0,V_FT1);fst_slot(V,V_FT0,s); break;
        case BIR_FMIN: load_fval(V,V_FT0,I->operands[0]);load_fval(V,V_FT1,I->operands[1]);e_fmins(V,V_FT0,V_FT0,V_FT1);fst_slot(V,V_FT0,s); break;

        /* ---- float remainder, a - trunc(a/b)*b through the fcvts ---- */
        case BIR_FREM:
            load_fval(V,V_FT0,I->operands[0]); load_fval(V,V_FT1,I->operands[1]);
            e_fdivs(V,V_FT2,V_FT0,V_FT1); e_fcvtws(V,V_T0,V_FT2); e_fcvtsw(V,V_FT2,V_T0);
            e_fmuls(V,V_FT2,V_FT2,V_FT1); e_fsubs(V,V_FT0,V_FT0,V_FT2);
            fst_slot(V,V_FT0,s); break;

        /* ---- float compare -> i1 (0/1 in a GPR). flt/fle take their
         * operands swapped to reach the greater-than pair; ne is eq xor 1. ---- */
        case BIR_FCMP:
            load_fval(V,V_FT0,I->operands[0]); load_fval(V,V_FT1,I->operands[1]);
            switch(I->subop){
            case BIR_FCMP_OEQ: case BIR_FCMP_UEQ: e_feqs(V,V_T0,V_FT0,V_FT1); break;
            case BIR_FCMP_ONE: case BIR_FCMP_UNE: e_feqs(V,V_T0,V_FT0,V_FT1); e_xori(V,V_T0,V_T0,1); break;
            case BIR_FCMP_OLT: case BIR_FCMP_ULT: e_flts(V,V_T0,V_FT0,V_FT1); break;
            case BIR_FCMP_OLE: case BIR_FCMP_ULE: e_fles(V,V_T0,V_FT0,V_FT1); break;
            case BIR_FCMP_OGT: case BIR_FCMP_UGT: e_flts(V,V_T0,V_FT1,V_FT0); break;
            case BIR_FCMP_OGE: case BIR_FCMP_UGE: e_fles(V,V_T0,V_FT1,V_FT0); break;
            default: e_feqs(V,V_T0,V_FT0,V_FT1); break;
            }
            st_slot(V,V_T0,s); break;

        /* ---- select. No cmov on RISC-V, so blend with a mask:
         * result = b ^ ((a ^ b) & -cond). Works for an integer in full width
         * or a float by its 32 bits. ---- */
        case BIR_SELECT: {
            int isf=(I->type<V->M->num_types)
                    && (V->M->types[I->type].kind==BIR_TYPE_FLOAT
                        || V->M->types[I->type].kind==BIR_TYPE_BFLOAT);
            load_val(V,V_T0,I->operands[0]); e_sub(V,V_T0,V_ZERO,V_T0);    /* mask = -cond */
            if (isf){
                ld_fbits(V,V_T1,I->operands[1]); ld_fbits(V,V_T2,I->operands[2]);
                e_xor(V,V_T1,V_T1,V_T2); e_and(V,V_T1,V_T1,V_T0); e_xor(V,V_T1,V_T1,V_T2);
                st_fbits(V,V_T1,s);
            } else {
                load_val(V,V_T1,I->operands[1]); load_val(V,V_T2,I->operands[2]);
                e_xor(V,V_T1,V_T1,V_T2); e_and(V,V_T1,V_T1,V_T0); e_xor(V,V_T1,V_T1,V_T2);
                st_slot(V,V_T1,s);
            }
            break; }

        /* ---- integer width / sign ---- */
        case BIR_TRUNC: load_val(V,V_T0,I->operands[0]); rv_sext_to(V,V_T0,tw(V,I->type)); st_slot(V,V_T0,s); break;
        case BIR_SEXT:  load_val(V,V_T0,I->operands[0]); rv_sext_to(V,V_T0,tw(V,val_type(V,I->operands[0]))); st_slot(V,V_T0,s); break;
        case BIR_ZEXT:  load_val(V,V_T0,I->operands[0]); rv_zext_to(V,V_T0,tw(V,val_type(V,I->operands[0]))); st_slot(V,V_T0,s); break;

        /* ---- int <-> float (the .l fcvts take the canonical 64-bit) ---- */
        case BIR_SITOFP: load_val(V,V_T0,I->operands[0]); e_fcvtsl(V,V_FT0,V_T0); fst_slot(V,V_FT0,s); break;
        case BIR_UITOFP: load_val(V,V_T0,I->operands[0]); rv_zext_to(V,V_T0,tw(V,val_type(V,I->operands[0]))); e_fcvtslu(V,V_FT0,V_T0); fst_slot(V,V_FT0,s); break;
        case BIR_FPTOSI: load_fval(V,V_FT0,I->operands[0]); e_fcvtls(V,V_T0,V_FT0); rv_sext_to(V,V_T0,tw(V,I->type)); st_slot(V,V_T0,s); break;
        case BIR_FPTOUI: load_fval(V,V_FT0,I->operands[0]); e_fcvtlus(V,V_T0,V_FT0); rv_zext_to(V,V_T0,tw(V,I->type)); st_slot(V,V_T0,s); break;

        /* ---- float width is a copy; f64/f16 wait their turn ---- */
        case BIR_FPEXT: case BIR_FPTRUNC: load_fval(V,V_FT0,I->operands[0]); fst_slot(V,V_FT0,s); break;

        /* ---- reinterpret bits + pointer<->int ---- */
        case BIR_BITCAST: {
            uint32_t srct=val_type(V,I->operands[0]);
            int srcf=(srct<V->M->num_types)&&(V->M->types[srct].kind==BIR_TYPE_FLOAT||V->M->types[srct].kind==BIR_TYPE_BFLOAT);
            int dstf=(I->type<V->M->num_types)&&(V->M->types[I->type].kind==BIR_TYPE_FLOAT||V->M->types[I->type].kind==BIR_TYPE_BFLOAT);
            if (srcf) ld_fbits(V,V_T0,I->operands[0]); else load_val(V,V_T0,I->operands[0]);
            if (dstf) st_fbits(V,V_T0,s); else { rv_sext_to(V,V_T0,tw(V,I->type)); st_slot(V,V_T0,s); }
            break; }
        case BIR_PTRTOINT: load_val(V,V_T0,I->operands[0]); if(tw(V,I->type)<64) rv_sext_to(V,V_T0,tw(V,I->type)); st_slot(V,V_T0,s); break;
        case BIR_INTTOPTR: load_val(V,V_T0,I->operands[0]); st_slot(V,V_T0,s); break;

        /* ---- Math the F extension already knows. sqrt and abs outright;
         * rcp and rsqrt as an honest 1.0 / x; rounding out to an int at the
         * right mode and back (which clamps past 2^31, fine for the energies
         * and positions these kernels actually carry). ---- */
        case BIR_SQRT: load_fval(V,V_FT0,I->operands[0]); e_fsqrts(V,V_FT0,V_FT0); fst_slot(V,V_FT0,s); break;
        case BIR_FABS: load_fval(V,V_FT0,I->operands[0]); e_fabss(V,V_FT0,V_FT0); fst_slot(V,V_FT0,s); break;
        case BIR_RCP:
            e_li(V,V_T0,0x3F800000); e_fmvwx(V,V_FT0,V_T0);
            load_fval(V,V_FT1,I->operands[0]); e_fdivs(V,V_FT0,V_FT0,V_FT1); fst_slot(V,V_FT0,s); break;
        case BIR_RSQ:
            load_fval(V,V_FT1,I->operands[0]); e_fsqrts(V,V_FT1,V_FT1);
            e_li(V,V_T0,0x3F800000); e_fmvwx(V,V_FT0,V_T0);
            e_fdivs(V,V_FT0,V_FT0,V_FT1); fst_slot(V,V_FT0,s); break;
        case BIR_FLOOR: case BIR_CEIL: case BIR_FTRUNC: case BIR_RNDNE: {
            int rm=(I->op==BIR_FLOOR)?2:(I->op==BIR_CEIL)?3:(I->op==BIR_FTRUNC)?1:0;
            load_fval(V,V_FT0,I->operands[0]);
            e_fcvtws_rm(V,V_T0,V_FT0,rm); e_fcvtsw(V,V_FT0,V_T0);
            fst_slot(V,V_FT0,s); break; }

        /* ---- Transcendentals out to libm. sin/cos arrive in turns (the
         * frontend's 1/2pi prescale for the GPU), so wind 2pi back in to get
         * the radians libm wants; exp2/log2 go straight out. ---- */
        case BIR_SIN: case BIR_COS: {
            const char *fn=(I->op==BIR_SIN)?"sinf":"cosf";
            load_fval(V,V_FA0,I->operands[0]);
            e_li(V,V_T0,0x40C90FDB); e_fmvwx(V,V_FT1,V_T0);  /* ft1 = 2pi */
            e_fmuls(V,V_FA0,V_FA0,V_FT1);                    /* fa0 = turns*2pi */
            emit_call_ext(V,fn);
            fst_slot(V,V_FA0,s); break; }
        case BIR_EXP2: call_libm1(V,"exp2f",I->operands[0],s); break;
        case BIR_LOG2: call_libm1(V,"log2f",I->operands[0],s); break;

        /* ---- Atomics. The A extension does the whole load-op-store and
         * returns the old value in one instruction, which is exactly the
         * shape CUDA's atomics want. Subtract has no amo of its own, so it
         * adds a negated value. ---- */
        case BIR_ATOMIC_ADD: case BIR_ATOMIC_SUB: case BIR_ATOMIC_AND:
        case BIR_ATOMIC_OR:  case BIR_ATOMIC_XOR: case BIR_ATOMIC_MIN:
        case BIR_ATOMIC_MAX: case BIR_ATOMIC_XCHG: {
            int w64=(int_w(V,I->type)==64);
            load_val(V,V_T1,I->operands[0]); load_val(V,V_T2,I->operands[1]);
            if (I->op==BIR_ATOMIC_SUB) e_sub(V,V_T2,V_ZERO,V_T2);
            int f5=(I->op==BIR_ATOMIC_ADD||I->op==BIR_ATOMIC_SUB)?0x00:
                   (I->op==BIR_ATOMIC_XCHG)?0x01:(I->op==BIR_ATOMIC_AND)?0x0C:
                   (I->op==BIR_ATOMIC_OR)?0x08:(I->op==BIR_ATOMIC_XOR)?0x04:
                   (I->op==BIR_ATOMIC_MIN)?0x10:0x14;
            e_amo(V,f5,w64,V_T0,V_T1,V_T2);
            st_slot(V,V_T0,s); break; }
        case BIR_ATOMIC_CAS: {
            int w64=(int_w(V,I->type)==64);
            load_val(V,V_T1,I->operands[0]); load_val(V,V_T0,I->operands[1]); load_val(V,V_T2,I->operands[2]);
            if (w64) e_ld(V,V_A0,V_T1,0); else e_lw(V,V_A0,V_T1,0);   /* old */
            e_bne(V,V_A0,V_T0,8);                                     /* old != expected -> skip store */
            if (w64) e_sd(V,V_T1,V_T2,0); else e_sw(V,V_T1,V_T2,0);
            st_slot(V,V_A0,s); break; }
        case BIR_ATOMIC_LOAD: {
            int w64=(int_w(V,I->type)==64);
            load_val(V,V_T1,I->operands[0]);
            if (w64) e_ld(V,V_T0,V_T1,0); else e_lw(V,V_T0,V_T1,0);
            st_slot(V,V_T0,s); break; }
        case BIR_ATOMIC_STORE: {
            int w64=(int_w(V,val_type(V,I->operands[1]))==64);
            load_val(V,V_T1,I->operands[0]); load_val(V,V_T2,I->operands[1]);
            if (w64) e_sd(V,V_T1,V_T2,0); else e_sw(V,V_T1,V_T2,0); break; }

        /* ---- Barriers: one thread, nothing to wait for. ---- */
        case BIR_BARRIER: case BIR_BARRIER_GROUP: break;

        /* ---- Warp primitives, single-lane degenerate forms. ---- */
        case BIR_SHFL: case BIR_SHFL_UP: case BIR_SHFL_DOWN: case BIR_SHFL_XOR:
        case BIR_BALLOT: case BIR_VOTE_ANY: case BIR_VOTE_ALL:
            load_val(V,V_T0,I->operands[1]); st_slot(V,V_T0,s); break;
        case BIR_GEP: { int sz=pointee_sz(V,I->type); uint32_t fo=0;
            if (bir_fgep(V->M,I,8,&fo)) { e_li(V,V_T0,(int32_t)fo); }
            else {
                if(!sz){ (void)be_fail(BC_E540,"cpu-rv64","gep",tystr(V,I->type)); V->n_errs++; break; }
                load_val(V,V_T0,I->operands[1]); e_li(V,V_T1,sz); e_mul(V,V_T0,V_T0,V_T1);
            }
            load_val(V,V_T1,I->operands[0]); e_add(V,V_T0,V_T0,V_T1); st_slot(V,V_T0,s); break; }
        case BIR_LOAD: { load_val(V,V_T0,I->operands[0]);
            int aw=pointee_sz(V,val_type(V,I->operands[0]));
            if (aw==8) e_ld(V,V_T0,V_T0,0); else if (aw==4) e_lw(V,V_T0,V_T0,0);
            else if (aw==2) e_lh(V,V_T0,V_T0,0); else if (aw==1) e_lb(V,V_T0,V_T0,0);
            else { (void)be_fail(BC_E547,"cpu-rv64",(unsigned)aw,"load"); V->n_errs++; break; }
            st_slot(V,V_T0,s); break; }
        case BIR_STORE: { load_val(V,V_T0,I->operands[0]); load_val(V,V_T1,I->operands[1]);
            int aw=pointee_sz(V,val_type(V,I->operands[1]));
            if (aw==8) e_sd(V,V_T1,V_T0,0); else if (aw==4) e_sw(V,V_T1,V_T0,0);
            else if (aw==2) e_sh(V,V_T1,V_T0,0); else if (aw==1) e_sb(V,V_T1,V_T0,0);
            else { (void)be_fail(BC_E547,"cpu-rv64",(unsigned)aw,"store"); V->n_errs++; }
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
            e_bne(V,V_T0,V_ZERO,8);                                 /* cond!=0 -> step past the jal */
            uint32_t fj=V->codelen; e_jal(V,V_ZERO,0);              /* cond==0 -> false_pad (patched) */
            emit_phi_copies(V,I->operands[1],cur); jal_block(V,I->operands[1]);
            pjal(V,fj,V_ZERO,(int32_t)V->codelen-(int32_t)fj,"a conditional branch");
            emit_phi_copies(V,I->operands[2],cur); jal_block(V,I->operands[2]); break; }
        case BIR_PHI: break; /* the work happens on the edges, not here */
        case BIR_RET:
            if (is_kernel){ if(n_ret<RV_RET_MAX) retfix[n_ret++]=V->codelen;
                            else rv_cap(V,"kernel return sites",(unsigned)RV_RET_MAX);
                            e_jal(V,V_ZERO,0); } /* -> loop_cont */
            else { if (I->num_operands){
                       uint32_t rt=val_type(V,I->operands[0]);
                       if (isflt(V,rt)) load_fval(V,V_FA0,I->operands[0]);
                       else load_val(V,V_A0,I->operands[0]); }
                   e_ld(V,V_RA,V_SP,0); e_ld(V,V_S0,V_SP,8); e_addi(V,V_SP,V_SP,frame); e_jalr(V,V_ZERO,V_RA,0); }
            break;
        case BIR_CALL: {
            uint32_t nop=I->num_operands; const uint32_t *ops=I->operands;
            if (nop==BIR_OPERANDS_OVERFLOW){
                uint32_t st=I->operands[0]; nop=I->operands[1];
                if (st>=V->M->num_extra_ops || nop>V->M->num_extra_ops-st){
                    (void)be_fail(BC_E584,"cpu-rv64"); V->n_errs++; break; }
                ops=&V->M->extra_operands[st];
            }
            if (nop<1){ (void)be_fail(BC_E584,"cpu-rv64"); V->n_errs++; break; }
            uint32_t callee=ops[0];
            if (callee>=V->M->num_funcs){
                (void)be_fail(BC_E581,"cpu-rv64",(unsigned)callee); V->n_errs++; break; }
            const bir_func_t *cF=&V->M->funcs[callee];

            int gi=0,fi=0,ns=0,bad=0;
            for (uint32_t a=1;a<nop;a++){
                uint32_t pt=ptype(V,cF,(uint16_t)(a-1));
                uint32_t at=pt?pt:val_type(V,ops[a]);
                if (!argok(V,at)){ (void)be_fail(BC_E582,"cpu-rv64",tystr(V,at)); V->n_errs++; bad=1; }
                if (arloc(pt?isflt(V,pt):0,&gi,&fi)==RV_ONSTK) ns++;
            }
            if (bad) break;
            if (ns>RV_ARG_MAX){ rv_cap(V,"arguments passed on the stack",(unsigned)RV_ARG_MAX); break; }

            int32_t sarea=(int32_t)(((ns*8)+15)&~15);
            if (sarea) e_addi(V,V_SP,V_SP,-sarea);
            gi=0;fi=0; { int32_t so=0;
            for (uint32_t a=1;a<nop;a++){
                uint32_t pt=ptype(V,cF,(uint16_t)(a-1));
                int isf=pt?isflt(V,pt):0, r=arloc(isf,&gi,&fi);
                if (r==RV_ONSTK){
                    if (isf) ld_fbits(V,V_T0,ops[a]); else load_val(V,V_T0,ops[a]);
                    e_sd(V,V_SP,V_T0,so); so+=8;
                }
                else if (r>=32) load_fval(V,r-32,ops[a]);
                else if (isf) ld_fbits(V,r,ops[a]);
                else load_val(V,r,ops[a]);
            } }

            if (V->n_cfix<RV_CALL_MAX){
                V->cfix[V->n_cfix].off=V->codelen; V->cfix[V->n_cfix].fn=callee; V->n_cfix++; }
            else rv_cap(V,"call fixups",(unsigned)RV_CALL_MAX);
            e_jal(V,V_RA,0);
            if (sarea) e_addi(V,V_SP,V_SP,sarea);

            if (!(I->type<V->M->num_types && V->M->types[I->type].kind==BIR_TYPE_VOID)){
                if (!argok(V,I->type)){ (void)be_fail(BC_E583,"cpu-rv64",tystr(V,I->type)); V->n_errs++; }
                else if (isflt(V,I->type)) fst_slot(V,V_FA0,s);
                else st_slot(V,V_A0,s);
            }
            break; }
        case BIR_INLINE_ASM:
            (void)be_fail(BC_E541,"cpu-rv64",
                          "inline asm, which Booth binds for PTX only");
            V->n_errs++; break;
        case BIR_TRAP: ew(V,0xC0001073u); break;
        case BIR_FNREF:
            (void)be_fail(BC_E541,"cpu-rv64",
                          "taking the address of a device function");
            V->n_errs++; break;
        case BIR_PRINTF:
            (void)be_fail(BC_E541,"cpu-rv64",
                          "device printf, which needs a data section to hold "
                          "the format string");
            V->n_errs++; break;
        case BIR_MMA: case BIR_MFRG:
            (void)be_fail(BC_E541,"cpu-rv64","warp-collective mma");
            V->n_errs++; break;
        default:
            (void)be_fail(BC_E542,"cpu-rv64",bir_op_name(I->op));
            V->n_errs++; break;
        }}
    }

    if (is_kernel){
        /* loop continue: bump tid and go again. Every RET landed here, and
         * so does the bail branch once tid runs out. */
        uint32_t loop_cont=V->codelen;
        ld_slot(V,V_T0,tid_off); e_addi(V,V_T0,V_T0,1); st_slot(V,V_T0,tid_off);
        { uint32_t o=V->codelen; e_jal(V,V_ZERO,0);
          pjal(V,o,V_ZERO,(int32_t)loop_head-(int32_t)o,"the thread loop back edge"); }
        uint32_t loop_end=V->codelen;
        pjal(V,bail_off,V_ZERO,(int32_t)loop_end-(int32_t)bail_off,"the thread loop exit");
        for (int k=0;k<n_ret;k++)
            pjal(V,retfix[k],V_ZERO,(int32_t)loop_cont-(int32_t)retfix[k],"a kernel return");
    }
    /* epilogue */
    e_ld(V,V_RA,V_SP,0); e_ld(V,V_S0,V_SP,8); e_addi(V,V_SP,V_SP,frame); e_jalr(V,V_ZERO,V_RA,0);

    /* resolve every jal-to-block now that the blocks have addresses */
    for (int k=0;k<V->n_fix;k++){
        if (V->fix[k].blk>=V->M->num_blocks) continue;
        pjal(V,V->fix[k].off,V_ZERO,
             (int32_t)V->blk_off[V->fix[k].blk]-(int32_t)V->fix[k].off,"a branch to a basic block"); }
    V->n_fix=0;
}

void rv64_init(rv64_mod_t *V,const bir_module_t *M){ memset(V,0,sizeof(*V)); V->M=M; }
int rv64_emit(rv64_mod_t *V){
    for(uint32_t f=0;f<V->M->num_funcs;f++) rv64_func(V,&V->M->funcs[f]);
    for(int k=0;k<V->n_cfix;k++){
        uint32_t fn=V->cfix[k].fn; if (fn>=BIR_MAX_FUNCS) continue;
        pjal(V,V->cfix[k].off,V_RA,
             (int32_t)V->func_off[fn]-(int32_t)V->cfix[k].off,"a call to a device function");
    }
    V->n_cfix=0;
    return 0;
}
