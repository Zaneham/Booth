/* cpu_emit.c -- BIR to x86-64, the stack-everything way.
 *
 * No register allocator, and that's deliberate. Every value gets its
 * own slot on the stack; every op loads its operands into RAX/RCX, does
 * the work, writes the answer back. Not fast. Correct, though, and
 * correct is the only thing worth being first at. Speed can wait.
 *
 * Calling convention is plain SysV: the kernel's params ride in
 * RDI/RSI/RDX/RCX/R8/R9, and the seventh onward spill to the caller's
 * stack at [rbp+16+...].
 *
 * The cheeky part is doing SIMT on a chip with no warps. A __global__
 * kernel has no hardware threads to fan out across, so we wrap its body
 * in a loop and run the threads one after the other. How many is
 * nthreads, a hidden arg the host tacks on after the real params, so
 * one call covers a whole block (block_id=0, grid_dim=1,
 * block_dim=nthreads). For a 1-D launch, set nthreads to the element
 * count and you're sorted. A `return` only finishes its own thread, so
 * it hops to the loop tail rather than leaving the function. Ordinary
 * non-__global__ functions skip the whole circus and return normally.
 *
 * Still on the list: more than one block per call, and obeying tl.load
 * masks. Until then, nthreads == element count and there's no ragged
 * tail to worry about. */

#include "cpu.h"
#include <string.h>

static void eb(cpu_mod_t *X, uint8_t b){ if(X->codelen<CPU_CODE_MAX) X->code[X->codelen++]=b; }
static void ei32(cpu_mod_t *X, int32_t v){ eb(X,(uint8_t)v);eb(X,(uint8_t)(v>>8));eb(X,(uint8_t)(v>>16));eb(X,(uint8_t)(v>>24)); }
static void rexw(cpu_mod_t *X,int r,int rm){ uint8_t x=0x48; if(r>=8)x|=4; if(rm>=8)x|=1; eb(X,x); }
static void modrm(cpu_mod_t *X,int mod,int reg,int rm){ eb(X,(uint8_t)((mod<<6)|((reg&7)<<3)|(rm&7))); }
static void ld_slot(cpu_mod_t *X,int r,int32_t o){ rexw(X,r,X_RBP);eb(X,0x8B);modrm(X,2,r,X_RBP);ei32(X,o); }
static void st_slot(cpu_mod_t *X,int r,int32_t o){ rexw(X,r,X_RBP);eb(X,0x89);modrm(X,2,r,X_RBP);ei32(X,o); }
static void mov_imm(cpu_mod_t *X,int r,int64_t v){ rexw(X,0,r);eb(X,0xC7);modrm(X,3,0,r);ei32(X,(int32_t)v); }
static int32_t slot(cpu_mod_t *X,uint32_t i){ return X->slots[i]; }

/* Width of an integer result, in the only two sizes that matter to the
 * shifters and the divider: 64 for an i64, 32 for everything narrower.
 * Pointers and the rest fall through to 64, which is what they are. The
 * shift and divide ops are the ones that actually care, because a 64-bit
 * `>>` on a sign-extended i32 drags the sign bits down into the answer and
 * the xorwow RNG starts handing out the same number forever. */
static int int_w(const cpu_mod_t *X, uint32_t ty){
    if (ty<X->M->num_types && X->M->types[ty].kind==BIR_TYPE_INT)
        return X->M->types[ty].width>=64 ? 64 : 32;
    return 64;
}

static void load_val(cpu_mod_t *X,int reg,uint32_t v){
    if (BIR_VAL_IS_CONST(v)){
        uint32_t c=BIR_VAL_INDEX(v);
        if (c<X->M->num_consts && X->M->consts[c].kind==BIR_CONST_FLOAT){
            /* a float const's union holds a DOUBLE; grabbing its low 32 bits
             * as an int hands you garbage. Narrow to f32 and take those bits,
             * which is what a store or a bit-pattern shuffle actually wants. */
            uint32_t bits=0; float fv=(float)X->M->consts[c].d.fval; memcpy(&bits,&fv,4);
            mov_imm(X,reg,(int32_t)bits);
        } else {
            mov_imm(X,reg,c<X->M->num_consts?X->M->consts[c].d.ival:0);
        }
    }
    else ld_slot(X,reg,slot(X,BIR_VAL_INDEX(v)));
}

/* element size in bytes of a pointer's pointee, default 4 (i32/f32).
 * Drives GEP stride, so it must be width-accurate: i8->1, i16->2,
 * i32/f32->4, i64/f64->8, ptr-to-ptr->8. */
static int type_size(const cpu_mod_t *X,uint32_t ty);

static int pointee_sz(cpu_mod_t *X,uint32_t ty){
    if (ty<X->M->num_types && X->M->types[ty].kind==BIR_TYPE_PTR){
        uint32_t in=X->M->types[ty].inner;
        if (in<X->M->num_types){
            uint8_t k=X->M->types[in].kind;
            if (k==BIR_TYPE_PTR) return 8;
            /* an array of structs strides by the whole struct, not by 4: a
             * struct pointee has no width field, so size it properly or every
             * index past the first lands in the wrong element. */
            if (k==BIR_TYPE_STRUCT || k==BIR_TYPE_ARRAY || k==BIR_TYPE_VECTOR) return type_size(X,in);
            uint32_t w=X->M->types[in].width;
            if (w>=8) return (int)(w/8);
        }
    }
    return 4;
}

/* size in bytes of a type. Aggregate layout is naive (no padding),
 * which is fine here: the only aggregates we size are tiles, and a tile
 * is a run of one uniform scalar, so the plain sum lands exactly right. */
static int type_size(const cpu_mod_t *X,uint32_t ty){
    if (ty>=X->M->num_types) return 8;
    const bir_type_t *t=&X->M->types[ty];
    switch (t->kind){
    case BIR_TYPE_INT: case BIR_TYPE_FLOAT: case BIR_TYPE_BFLOAT: return t->width?(int)(t->width/8):4;
    case BIR_TYPE_PTR: return 8;
    case BIR_TYPE_ARRAY: return (int)t->count*type_size(X,t->inner);
    case BIR_TYPE_VECTOR: return (int)t->width*type_size(X,t->inner);
    case BIR_TYPE_STRUCT: { int s=0; for(uint16_t i=0;i<t->num_fields;i++) s+=type_size(X,X->M->type_fields[t->count+i]); return s; }
    default: return 8;
    }
}

/* type index of a value (const or inst result); 0 if unknown. */
static uint32_t val_type(const cpu_mod_t *X,uint32_t v){
    uint32_t i=BIR_VAL_INDEX(v);
    if (BIR_VAL_IS_CONST(v)) return i<X->M->num_consts?X->M->consts[i].type:0;
    return i<X->M->num_insts?X->M->insts[i].type:0;
}

/* pointee bit width of a pointer-typed value, default 32. A pointer to a
 * pointer stores 8 bytes, not 4: the inner PTR carries no width field, so
 * spot it by kind or the high half of the address quietly falls off and the
 * next dereference reads litter. */
static int pointee_bits(const cpu_mod_t *X,uint32_t v){
    uint32_t ty=val_type(X,v);
    if (ty<X->M->num_types && X->M->types[ty].kind==BIR_TYPE_PTR){
        uint32_t in=X->M->types[ty].inner;
        if (in<X->M->num_types && X->M->types[in].kind==BIR_TYPE_PTR) return 64;
        if (in<X->M->num_types && X->M->types[in].width) return (int)X->M->types[in].width;
    }
    return 32;
}

/* SysV hands float/double scalars in XMM0..7 and everything else (ints,
 * pointers) in RDI.. , each class counted on its own. So we have to know
 * which class a param is, or a float scalar like saxpy's alpha reads out
 * of the wrong register and the kernel quietly does nothing. */
static int param_is_float(const cpu_mod_t *X,const bir_func_t *F,uint16_t p){
    if (F->type>=X->M->num_types) return 0;
    const bir_type_t *t=&X->M->types[F->type];
    if (t->kind!=BIR_TYPE_FUNC || p>=t->num_fields) return 0;
    uint32_t pt=X->M->type_fields[t->count+p];
    if (pt>=X->M->num_types) return 0;
    int k=X->M->types[pt].kind;
    return k==BIR_TYPE_FLOAT || k==BIR_TYPE_BFLOAT;
}
static int param_is_f64(const cpu_mod_t *X,const bir_func_t *F,uint16_t p){
    if (F->type>=X->M->num_types) return 0;
    const bir_type_t *t=&X->M->types[F->type];
    if (t->kind!=BIR_TYPE_FUNC || p>=t->num_fields) return 0;
    uint32_t pt=X->M->type_fields[t->count+p];
    return pt<X->M->num_types && X->M->types[pt].width==64;
}
/* Width of an integer parameter, 0 if the param is not an integer. SysV
 * leaves the high half of a narrow int arg undefined, so the prologue uses
 * this to know which params need sign-extending into their slots. */
static int param_int_w(const cpu_mod_t *X,const bir_func_t *F,uint16_t p){
    if (F->type>=X->M->num_types) return 0;
    const bir_type_t *t=&X->M->types[F->type];
    if (t->kind!=BIR_TYPE_FUNC || p>=t->num_fields) return 0;
    uint32_t pt=X->M->type_fields[t->count+p];
    if (pt<X->M->num_types && X->M->types[pt].kind==BIR_TYPE_INT) return (int)X->M->types[pt].width;
    return 0;
}
/* store an incoming XMM arg straight into its frame slot (movss/movsd). */
static void st_xmm_slot(cpu_mod_t *X,int n,int32_t o,int is64){
    eb(X,is64?0xF2:0xF3);eb(X,0x0F);eb(X,0x11);modrm(X,2,n,X_RBP);ei32(X,o);
}

/* Load a float operand into xmm `xn`. A constant gets its bits parked in eax
 * and shuffled across with movd; a real value is a plain movss from its slot.
 * The old float ops read a constant operand straight out of slot 0, which is
 * to say out of whatever happened to be living there, so this is also the
 * quiet end of a bug. */
static void ld_fval(cpu_mod_t *X,int xn,uint32_t v){
    if (BIR_VAL_IS_CONST(v)){
        uint32_t c=BIR_VAL_INDEX(v); uint32_t bits=0;
        if (c<X->M->num_consts){ float fv=(float)X->M->consts[c].d.fval; memcpy(&bits,&fv,4); }
        mov_imm(X,X_RAX,(int32_t)bits);
        eb(X,0x66);eb(X,0x0F);eb(X,0x6E);modrm(X,3,xn,X_RAX);          /* movd xn, eax */
    } else {
        eb(X,0xF3);eb(X,0x0F);eb(X,0x10);modrm(X,2,xn,X_RBP);ei32(X,slot(X,BIR_VAL_INDEX(v)));
    }
}

/* The raw 32 bits of a float operand in a GPR, zero-extended. Used by SELECT,
 * which blends two floats by their bit patterns and never has to know it was
 * floats it was choosing between. */
static void ld_fbits(cpu_mod_t *X,int r,uint32_t v){
    if (BIR_VAL_IS_CONST(v)){
        uint32_t c=BIR_VAL_INDEX(v); uint32_t bits=0;
        if (c<X->M->num_consts){ float fv=(float)X->M->consts[c].d.fval; memcpy(&bits,&fv,4); }
        eb(X,0xC7);modrm(X,3,0,r);ei32(X,(int32_t)bits);              /* mov r32, imm32 */
    } else {
        eb(X,0x8B);modrm(X,2,r,X_RBP);ei32(X,slot(X,BIR_VAL_INDEX(v))); /* mov r32, [rbp+slot] */
    }
}

/* Is this type a float kind? SELECT and a few others branch on it. */
static int is_float_ty(const cpu_mod_t *X,uint32_t ty){
    if (ty>=X->M->num_types) return 0;
    uint8_t k=X->M->types[ty].kind;
    return k==BIR_TYPE_FLOAT || k==BIR_TYPE_BFLOAT;
}

/* A float compare predicate, turned into the unsigned setcc condition that
 * matches ucomiss. NaN is assumed not to show up; the day it does, an isnan
 * guard goes here. */
static int fcmp_cc(uint8_t p){
    switch(p){
    case BIR_FCMP_OEQ: case BIR_FCMP_UEQ: return XCC_E;
    case BIR_FCMP_ONE: case BIR_FCMP_UNE: return XCC_NE;
    case BIR_FCMP_OLT: case BIR_FCMP_ULT: return XCC_B;
    case BIR_FCMP_OLE: case BIR_FCMP_ULE: return XCC_BE;
    case BIR_FCMP_OGT: case BIR_FCMP_UGT: return XCC_A;
    case BIR_FCMP_OGE: case BIR_FCMP_UGE: return XCC_AE;
    default: return XCC_NE;
    }
}

/* The type a value carries, and that type's bit width. The conversions need
 * to know both the size they are coming from and the size they are going to. */
static uint32_t val_type_x(const cpu_mod_t *X,uint32_t v){
    uint32_t i=BIR_VAL_INDEX(v);
    if (BIR_VAL_IS_CONST(v)) return i<X->M->num_consts?X->M->consts[i].type:0;
    return i<X->M->num_insts?X->M->insts[i].type:0;
}
static int ty_w(const cpu_mod_t *X,uint32_t ty){
    if (ty<X->M->num_types){ uint16_t w=X->M->types[ty].width; if(w) return (int)w;
        if (X->M->types[ty].kind==BIR_TYPE_PTR) return 64; }
    return 32;
}

/* Canonicalise rax by sign-extending up from a `w`-bit value, and the
 * zero-extending twin. The slot convention is sign-extended, so a freshly
 * narrowed or widened integer goes through one of these to stop the high
 * bits telling stories. */
static void sext_to(cpu_mod_t *X,int w){
    if (w>=64) return;
    if (w==32){ rexw(X,X_RAX,X_RAX);eb(X,0x63);modrm(X,3,X_RAX,X_RAX); }              /* movsxd rax,eax */
    else if (w==16){ rexw(X,X_RAX,X_RAX);eb(X,0x0F);eb(X,0xBF);modrm(X,3,X_RAX,X_RAX); } /* movsx rax,ax */
    else if (w==8){ rexw(X,X_RAX,X_RAX);eb(X,0x0F);eb(X,0xBE);modrm(X,3,X_RAX,X_RAX); }  /* movsx rax,al */
    else { rexw(X,0,X_RAX);eb(X,0x83);modrm(X,3,4,X_RAX);eb(X,1); }                    /* and rax,1 */
}
static void zext_to(cpu_mod_t *X,int w){
    if (w>=64) return;
    if (w==32){ eb(X,0x89);modrm(X,3,X_RAX,X_RAX); }                                  /* mov eax,eax */
    else if (w==16){ rexw(X,X_RAX,X_RAX);eb(X,0x0F);eb(X,0xB7);modrm(X,3,X_RAX,X_RAX); } /* movzx rax,ax */
    else if (w==8){ rexw(X,X_RAX,X_RAX);eb(X,0x0F);eb(X,0xB6);modrm(X,3,X_RAX,X_RAX); }  /* movzx rax,al */
    else { rexw(X,0,X_RAX);eb(X,0x83);modrm(X,3,4,X_RAX);eb(X,1); }                    /* and rax,1 */
}

/* Intern an external symbol name (sinf and the like), de-duplicated, and
 * hand back its index. The ELF writer reads these out as undefined globals
 * for the linker to chase down in libm. */
static uint32_t cpu_extsym(cpu_mod_t *X,const char *name){
    for (int i=0;i<X->n_extsym;i++) if(!strcmp(X->extsym[i],name)) return (uint32_t)i;
    if (X->n_extsym>=CPU_EXTSYM_MAX) return 0;
    int idx=X->n_extsym++;
    size_t n=strlen(name); if(n>=CPU_EXTSYM_LEN) n=CPU_EXTSYM_LEN-1;
    memcpy(X->extsym[idx],name,n); X->extsym[idx][n]='\0';
    return (uint32_t)idx;
}

/* Align the stack to 16 and call an external symbol, leaving the rel32 for
 * the linker and a note in the reloc table. SysV insists on the alignment,
 * and a misaligned call into a function that uses an aligned move is a
 * segfault biding its time. Whatever the call wants in xmm0 must already be
 * there; whatever it returns comes back the same way. */
static void emit_call_ext(cpu_mod_t *X,const char *name){
    rexw(X,0,X_RSP);eb(X,0x83);modrm(X,3,4,X_RSP);eb(X,0xF0); /* and rsp,-16 */
    uint32_t sym=cpu_extsym(X,name);
    eb(X,0xE8);
    if (X->n_reloc<CPU_RELOC_MAX){ X->reloc[X->n_reloc].off=X->codelen; X->reloc[X->n_reloc].sym=sym; X->n_reloc++; }
    ei32(X,0);
}

/* A plain one-argument libm call: float operand in, float result out. */
static void call_libm1(cpu_mod_t *X,const char *name,uint32_t op0,int32_t s){
    ld_fval(X,X_XMM0,op0);
    emit_call_ext(X,name);
    st_xmm_slot(X,X_XMM0,s,0);
}

/* incoming value of a phi for a given predecessor block (abs index). */
static uint32_t phi_incoming(const cpu_mod_t *X,const bir_inst_t *P,uint32_t pred){
    if (P->num_operands==BIR_OPERANDS_OVERFLOW){
        uint32_t st=P->operands[0],cnt=P->operands[1];
        for(uint32_t j=0;j+1<cnt;j+=2) if(X->M->extra_operands[st+j]==pred) return X->M->extra_operands[st+j+1];
    } else {
        for(uint32_t j=0;j+1<P->num_operands;j+=2) if(P->operands[j]==pred) return P->operands[j+1];
    }
    return BIR_VAL_NONE;
}

/* Lower phis on the edge pred->succ: copy each phi's incoming value into
 * its slot before the branch. Phis sit at the head of a block (mem2reg
 * step7), so stop at the first non-phi. Sequential RAX copies; a phi whose
 * source is another phi in the same block (swap idiom) would need a temp,
 * but loop/if merges never produce that. */
static void emit_phi_copies(cpu_mod_t *X,uint32_t succ,uint32_t pred){
    if (succ>=X->M->num_blocks) return;
    const bir_block_t *B=&X->M->blocks[succ];
    for(uint32_t i=0;i<B->num_insts;i++){
        uint32_t ix=B->first_inst+i; const bir_inst_t *P=&X->M->insts[ix];
        if(P->op!=BIR_PHI) break;
        uint32_t v=phi_incoming(X,P,pred);
        if(v==BIR_VAL_NONE) continue;
        load_val(X,X_RAX,v); st_slot(X,X_RAX,slot(X,ix));
    }
}

/* RET sites that must branch to the thread loop's continue point rather
 * than leave the function. A return inside a kernel ends one thread. */
#define CPU_RET_MAX 256

static void cpu_func(cpu_mod_t *X,const bir_func_t *F){
    /* One slot per BIR value index, spanning params and every block
     * inst. Params are signature insts 0..num_params-1; block insts
     * follow. Single pass so nothing aliases. */
    /* Frame allocator: decrement-then-assign, so every allocation claims
     * [new_off, old_off) and nothing overlaps. An 8-byte value slot per
     * inst; alloca/shared-alloc also reserve a backing region sized from
     * the pointee type. The body loop walks insts in this same order, so a
     * running counter pairs each alloca with its region offset. */
    int32_t off=0;
    for (uint16_t p=0;p<F->num_params;p++){ off-=8; X->slots[p]=off; }
    int na=0;
    for (uint16_t b=0;b<F->num_blocks;b++){
        const bir_block_t*B=&X->M->blocks[F->first_block+b];
        for(uint32_t i=0;i<B->num_insts;i++){
            uint32_t ix=B->first_inst+i; off-=8; X->slots[ix]=off;
            const bir_inst_t*I=&X->M->insts[ix];
            if ((I->op==BIR_ALLOCA||I->op==BIR_SHARED_ALLOC) && na<CPU_ALLOCA_MAX){
                uint32_t pte=(I->type<X->M->num_types)?X->M->types[I->type].inner:0;
                int sz=(type_size(X,pte)+7)&~7; if(sz<8)sz=8;
                off-=sz; X->alloca_off[na++]=off;
            }
        }
    }
    int na_emit=0;
    /* The thread loop lives here. A kernel runs its body once per
     * thread_id, counting up to ntid (the hidden trailing arg). block_id
     * and grid_dim stay 0 and 1 because one call is one block for now.
     * tid and ntid get their own frame slots, tucked below the values. */
    int is_kernel=(F->cuda_flags&CUDA_GLOBAL)!=0;
    off-=8; int32_t tid_off=off;
    off-=8; int32_t ntid_off=off;
    uint32_t retfix[CPU_RET_MAX]; int n_ret=0;
    eb(X,0x55); rexw(X,X_RBP,X_RSP);eb(X,0x89);modrm(X,3,X_RSP,X_RBP);
    int32_t frame=((-off+15)&~15)+8;
    rexw(X,0,X_RSP);eb(X,0x81);modrm(X,3,5,X_RSP);ei32(X,frame);
    /* Walk the args in order, splitting by SysV class: ints/pointers come
     * out of the GPRs, floats out of XMM, each on its own counter, and
     * anything past the registers spills to the caller's stack in arg
     * order. The hidden nthreads (kernels only) rides along as one more
     * integer arg on the end. */
    {
        static const int areg[6]={X_RDI,X_RSI,X_RDX,X_RCX,8,9};
        int gi=0, xi=0; int32_t stk=16;
        uint16_t total=(uint16_t)(F->num_params + (is_kernel?1:0));
        for (uint16_t p=0;p<total;p++){
            int isf=(p<F->num_params) && param_is_float(X,F,p);
            int32_t dest=(p<F->num_params)?X->slots[p]:ntid_off;
            if (isf){
                if (xi<8) st_xmm_slot(X,xi++,dest,param_is_f64(X,F,p));
                else { rexw(X,X_RAX,X_RBP);eb(X,0x8B);modrm(X,2,X_RAX,X_RBP);ei32(X,stk); st_slot(X,X_RAX,dest); stk+=8; }
            } else {
                int pw=(p<F->num_params)?param_int_w(X,F,p):0;
                int narrow=(pw>0 && pw<64);
                if (gi<6){
                    if (narrow){
                        rexw(X,areg[gi],X_RAX);eb(X,0x89);modrm(X,3,areg[gi],X_RAX); /* mov rax, arg */
                        sext_to(X,pw); st_slot(X,X_RAX,dest); gi++;
                    } else st_slot(X,areg[gi++],dest);
                } else {
                    rexw(X,X_RAX,X_RBP);eb(X,0x8B);modrm(X,2,X_RAX,X_RBP);ei32(X,stk); /* mov rax, [rbp+stk] */
                    if (narrow) sext_to(X,pw);
                    st_slot(X,X_RAX,dest); stk+=8; gi++;
                }
            }
        }
    }
    uint32_t loop_head=0,jge_fix=0;
    if (is_kernel){
        mov_imm(X,X_RAX,0); st_slot(X,X_RAX,tid_off);                 /* tid = 0 */
        loop_head=X->codelen;
        ld_slot(X,X_RAX,tid_off); rexw(X,X_RAX,X_RBP);eb(X,0x3B);modrm(X,2,X_RAX,X_RBP);ei32(X,ntid_off); /* cmp tid,[ntid] */
        eb(X,0x0F);eb(X,0x8D); jge_fix=X->codelen; ei32(X,0);        /* jge loop_end (patched) */
    }
    for (uint16_t b=0;b<F->num_blocks;b++){
        const bir_block_t*B=&X->M->blocks[F->first_block+b]; X->blk_off[F->first_block+b]=X->codelen;
        for (uint32_t i=0;i<B->num_insts;i++){ uint32_t ix=B->first_inst+i; const bir_inst_t*I=&X->M->insts[ix]; int32_t s=slot(X,ix);
        switch(I->op){
        case BIR_PARAM: break; /* materialized once in the prologue (ld_arg), outside the thread loop */
        case BIR_ALLOCA: case BIR_SHARED_ALLOC: /* value = pointer to the reserved frame region */
            if (na_emit<CPU_ALLOCA_MAX){ rexw(X,X_RAX,X_RBP);eb(X,0x8D);modrm(X,2,X_RAX,X_RBP);ei32(X,X->alloca_off[na_emit++]); st_slot(X,X_RAX,s); }
            break;
        case BIR_THREAD_ID: if(is_kernel){ ld_slot(X,X_RAX,tid_off); } else mov_imm(X,X_RAX,0); st_slot(X,X_RAX,s); break;
        case BIR_BLOCK_ID: mov_imm(X,X_RAX,0); st_slot(X,X_RAX,s); break;
        case BIR_BLOCK_DIM: if(is_kernel){ ld_slot(X,X_RAX,ntid_off); } else mov_imm(X,X_RAX,1); st_slot(X,X_RAX,s); break;
        case BIR_GRID_DIM: mov_imm(X,X_RAX,1); st_slot(X,X_RAX,s); break;
        case BIR_ADD: load_val(X,X_RAX,I->operands[0]);load_val(X,X_RCX,I->operands[1]);rexw(X,X_RCX,X_RAX);eb(X,0x01);modrm(X,3,X_RCX,X_RAX);st_slot(X,X_RAX,s);break;
        case BIR_SUB: load_val(X,X_RAX,I->operands[0]);load_val(X,X_RCX,I->operands[1]);rexw(X,X_RCX,X_RAX);eb(X,0x29);modrm(X,3,X_RCX,X_RAX);st_slot(X,X_RAX,s);break;
        case BIR_MUL: load_val(X,X_RAX,I->operands[0]);load_val(X,X_RCX,I->operands[1]);eb(X,0x48);eb(X,0x0F);eb(X,0xAF);modrm(X,3,X_RAX,X_RCX);st_slot(X,X_RAX,s);break;

        /* ---- integer bitwise (width-agnostic, plain 64-bit) ---- */
        case BIR_AND: load_val(X,X_RAX,I->operands[0]);load_val(X,X_RCX,I->operands[1]);rexw(X,X_RCX,X_RAX);eb(X,0x21);modrm(X,3,X_RCX,X_RAX);st_slot(X,X_RAX,s);break;
        case BIR_OR:  load_val(X,X_RAX,I->operands[0]);load_val(X,X_RCX,I->operands[1]);rexw(X,X_RCX,X_RAX);eb(X,0x09);modrm(X,3,X_RCX,X_RAX);st_slot(X,X_RAX,s);break;
        case BIR_XOR: load_val(X,X_RAX,I->operands[0]);load_val(X,X_RCX,I->operands[1]);rexw(X,X_RCX,X_RAX);eb(X,0x31);modrm(X,3,X_RCX,X_RAX);st_slot(X,X_RAX,s);break;

        /* ---- shifts (count in cl). Narrow shifts run 32-bit and
         * sign-extend the answer back to canonical 64, so the high bits stop
         * lying. ext picks shl(4)/shr(5)/sar(7). ---- */
        case BIR_SHL: case BIR_LSHR: case BIR_ASHR: {
            int ext=(I->op==BIR_SHL)?4:(I->op==BIR_LSHR)?5:7;
            load_val(X,X_RAX,I->operands[0]); load_val(X,X_RCX,I->operands[1]);
            if (int_w(X,I->type)==64){ rexw(X,0,X_RAX);eb(X,0xD3);modrm(X,3,ext,X_RAX); }
            else { eb(X,0xD3);modrm(X,3,ext,X_RAX); rexw(X,X_RAX,X_RAX);eb(X,0x63);modrm(X,3,X_RAX,X_RAX); }
            st_slot(X,X_RAX,s); break; }

        /* ---- signed divide/remainder. cqo/cdq sign-fills the high
         * half, idiv leaves quotient in rax and remainder in rdx. A narrow
         * result gets sign-extended back. (Divide by zero traps, same as the
         * hardware; the kernels are expected to guard their divisors.) ---- */
        case BIR_SDIV: case BIR_SREM: {
            int w64=(int_w(X,I->type)==64);
            load_val(X,X_RAX,I->operands[0]); load_val(X,X_RCX,I->operands[1]);
            if (w64){ eb(X,0x48);eb(X,0x99); rexw(X,0,X_RCX);eb(X,0xF7);modrm(X,3,7,X_RCX); }
            else { eb(X,0x99); eb(X,0xF7);modrm(X,3,7,X_RCX); }
            if (I->op==BIR_SREM){ rexw(X,X_RDX,X_RAX);eb(X,0x89);modrm(X,3,X_RDX,X_RAX); }
            if (!w64){ rexw(X,X_RAX,X_RAX);eb(X,0x63);modrm(X,3,X_RAX,X_RAX); }
            st_slot(X,X_RAX,s); break; }

        /* ---- unsigned divide/remainder. rdx cleared, div leaves
         * quotient in rax, remainder in rdx; the narrow result is naturally
         * zero-extended, which is exactly what an unsigned 32-bit wants. ---- */
        case BIR_UDIV: case BIR_UREM: {
            int w64=(int_w(X,I->type)==64);
            load_val(X,X_RAX,I->operands[0]); load_val(X,X_RCX,I->operands[1]);
            if (w64){ eb(X,0x48);eb(X,0x31);modrm(X,3,X_RDX,X_RDX); rexw(X,0,X_RCX);eb(X,0xF7);modrm(X,3,6,X_RCX); }
            else { eb(X,0x31);modrm(X,3,X_RDX,X_RDX); eb(X,0xF7);modrm(X,3,6,X_RCX); }
            if (I->op==BIR_UREM){ rexw(X,X_RDX,X_RAX);eb(X,0x89);modrm(X,3,X_RDX,X_RAX); }
            st_slot(X,X_RAX,s); break; }
        case BIR_GEP: { int sz=pointee_sz(X,I->type); load_val(X,X_RCX,I->operands[1]); mov_imm(X,X_RAX,sz); eb(X,0x48);eb(X,0x0F);eb(X,0xAF);modrm(X,3,X_RCX,X_RAX); load_val(X,X_RAX,I->operands[0]); rexw(X,X_RCX,X_RAX);eb(X,0x01);modrm(X,3,X_RCX,X_RAX); st_slot(X,X_RAX,s); break; }
        case BIR_LOAD: { load_val(X,X_RAX,I->operands[0]); /* addr in rax */
            const bir_type_t *t=(I->type<X->M->num_types)?&X->M->types[I->type]:0;
            int isflt=t&&(t->kind==BIR_TYPE_FLOAT||t->kind==BIR_TYPE_BFLOAT);
            int w=t?(int)t->width:32;
            if (t && t->kind==BIR_TYPE_PTR) w=64;  /* a loaded pointer is 8 bytes, not a sign-extended 4 */
            /* C integers are signed, so sign-extend to 64 bits. Skip this
             * and a negative int reads back as a giant positive one, and
             * every 64-bit compare downstream quietly lies. Floats stay a
             * 32-bit zero-extend: the bits only flow on to movss, which
             * reads the low 32 and ignores the rest. */
            if (isflt)        { eb(X,0x8B);modrm(X,0,X_RAX,X_RAX); }                       /* mov eax,[rax] */
            else if (w==64)   { rexw(X,X_RAX,X_RAX);eb(X,0x8B);modrm(X,0,X_RAX,X_RAX); }   /* mov rax,[rax] */
            else if (w==16)   { rexw(X,X_RAX,X_RAX);eb(X,0x0F);eb(X,0xBF);modrm(X,0,X_RAX,X_RAX); } /* movsx rax,word[rax] */
            else if (w==8||w==1){ rexw(X,X_RAX,X_RAX);eb(X,0x0F);eb(X,0xBE);modrm(X,0,X_RAX,X_RAX); } /* movsx rax,byte[rax] */
            else              { rexw(X,X_RAX,X_RAX);eb(X,0x63);modrm(X,0,X_RAX,X_RAX); }   /* movsxd rax,dword[rax] */
            st_slot(X,X_RAX,s); break; }
        case BIR_STORE: { load_val(X,X_RAX,I->operands[0]); load_val(X,X_RCX,I->operands[1]); /* val in rax, addr in rcx */
            int w=pointee_bits(X,I->operands[1]); /* store width = pointee of the address */
            if (w==64)      { rexw(X,X_RAX,X_RCX);eb(X,0x89);modrm(X,0,X_RAX,X_RCX); }   /* mov [rcx],rax */
            else if (w==16) { eb(X,0x66);eb(X,0x89);modrm(X,0,X_RAX,X_RCX); }            /* mov [rcx],ax  */
            else if (w==8||w==1){ eb(X,0x88);modrm(X,0,X_RAX,X_RCX); }                   /* mov [rcx],al  */
            else            { eb(X,0x89);modrm(X,0,X_RAX,X_RCX); }                        /* mov [rcx],eax (32-bit, incl f32) */
            break; }
        /* ---- float arithmetic. ops[0] -> xmm0, ops[1] -> xmm1, and
         * the op leaves the answer in xmm0. addss/subss/mulss/divss. ---- */
        case BIR_FADD: case BIR_FSUB: case BIR_FMUL: case BIR_FDIV: {
            uint8_t op=(I->op==BIR_FADD)?0x58:(I->op==BIR_FSUB)?0x5C:(I->op==BIR_FMUL)?0x59:0x5E;
            ld_fval(X,X_XMM0,I->operands[0]); ld_fval(X,X_XMM1,I->operands[1]);
            eb(X,0xF3);eb(X,0x0F);eb(X,op);modrm(X,3,X_XMM0,X_XMM1);
            st_xmm_slot(X,X_XMM0,s,0); break; }

        /* ---- float min/max. maxss/minss; NaN handling is the
         * hardware's (returns the second operand), good enough sans NaN. ---- */
        case BIR_FMAX: case BIR_FMIN: {
            uint8_t op=(I->op==BIR_FMAX)?0x5F:0x5D;
            ld_fval(X,X_XMM0,I->operands[0]); ld_fval(X,X_XMM1,I->operands[1]);
            eb(X,0xF3);eb(X,0x0F);eb(X,op);modrm(X,3,X_XMM0,X_XMM1);
            st_xmm_slot(X,X_XMM0,s,0); break; }

        /* ---- float remainder the scenic route, a - trunc(a/b)*b, since
         * x86 never did grow a frem instruction and was not about to start
         * on our account. ---- */
        case BIR_FREM: {
            ld_fval(X,X_XMM0,I->operands[0]); ld_fval(X,X_XMM1,I->operands[1]);
            eb(X,0x0F);eb(X,0x28);modrm(X,3,X_XMM2,X_XMM0);                 /* movaps xmm2,xmm0 (a) */
            eb(X,0xF3);eb(X,0x0F);eb(X,0x5E);modrm(X,3,X_XMM0,X_XMM1);      /* divss xmm0,xmm1 (a/b) */
            eb(X,0xF3);eb(X,0x0F);eb(X,0x2C);modrm(X,3,X_RAX,X_XMM0);       /* cvttss2si eax,xmm0 (trunc) */
            eb(X,0xF3);eb(X,0x0F);eb(X,0x2A);modrm(X,3,X_XMM0,X_RAX);       /* cvtsi2ss xmm0,eax */
            eb(X,0xF3);eb(X,0x0F);eb(X,0x59);modrm(X,3,X_XMM0,X_XMM1);      /* mulss xmm0,xmm1 (*b) */
            eb(X,0x0F);eb(X,0x28);modrm(X,3,X_XMM1,X_XMM2);                 /* movaps xmm1,xmm2 (a) */
            eb(X,0xF3);eb(X,0x0F);eb(X,0x5C);modrm(X,3,X_XMM1,X_XMM0);      /* subss xmm1,xmm0 (a - ...) */
            st_xmm_slot(X,X_XMM1,s,0); break; }

        /* ---- float compare -> i1. ucomiss sets the unsigned flags,
         * setcc reads them, movzx widens the bool. ---- */
        case BIR_FCMP: {
            ld_fval(X,X_XMM0,I->operands[0]); ld_fval(X,X_XMM1,I->operands[1]);
            eb(X,0x0F);eb(X,0x2E);modrm(X,3,X_XMM0,X_XMM1);                 /* ucomiss xmm0,xmm1 */
            int cc=fcmp_cc(I->subop);
            eb(X,0x0F);eb(X,(uint8_t)(0x90+cc));modrm(X,3,0,X_RAX);         /* setcc al */
            rexw(X,0,X_RAX);eb(X,0x0F);eb(X,0xB6);modrm(X,3,X_RAX,X_RAX);   /* movzx rax,al */
            st_slot(X,X_RAX,s); break; }

        /* ---- select. cond ? a : b. Integers ride cmov; floats get
         * blended by their bits, so cmov works on those too without xmm ever
         * hearing about the choice. ---- */
        case BIR_SELECT: {
            if (is_float_ty(X,I->type)){
                ld_fbits(X,X_RAX,I->operands[2]); ld_fbits(X,X_RCX,I->operands[1]);
                load_val(X,X_RDX,I->operands[0]);
                rexw(X,X_RDX,X_RDX);eb(X,0x85);modrm(X,3,X_RDX,X_RDX);      /* test rdx,rdx */
                eb(X,0x0F);eb(X,0x45);modrm(X,3,X_RAX,X_RCX);              /* cmovnz eax,ecx */
                eb(X,0x89);modrm(X,2,X_RAX,X_RBP);ei32(X,s);              /* mov [rbp+s],eax */
            } else {
                load_val(X,X_RAX,I->operands[2]); load_val(X,X_RCX,I->operands[1]);
                load_val(X,X_RDX,I->operands[0]);
                rexw(X,X_RDX,X_RDX);eb(X,0x85);modrm(X,3,X_RDX,X_RDX);      /* test rdx,rdx */
                rexw(X,X_RAX,X_RCX);eb(X,0x0F);eb(X,0x45);modrm(X,3,X_RAX,X_RCX); /* cmovnz rax,rcx */
                st_slot(X,X_RAX,s);
            }
            break; }

        /* ---- integer width / sign changes. Truncate canonicalises
         * to the result width; sext/zext widen from the source width. ---- */
        case BIR_TRUNC: load_val(X,X_RAX,I->operands[0]); sext_to(X,ty_w(X,I->type)); st_slot(X,X_RAX,s); break;
        case BIR_SEXT:  load_val(X,X_RAX,I->operands[0]); sext_to(X,ty_w(X,val_type_x(X,I->operands[0]))); st_slot(X,X_RAX,s); break;
        case BIR_ZEXT:  load_val(X,X_RAX,I->operands[0]); zext_to(X,ty_w(X,val_type_x(X,I->operands[0]))); st_slot(X,X_RAX,s); break;

        /* ---- int <-> float. cvtsi2ss takes the sign-extended 64-bit
         * value (unsigned zero-extends first); cvttss2si truncates toward
         * zero, the way CUDA's (int)f does. ---- */
        case BIR_SITOFP: load_val(X,X_RAX,I->operands[0]);
            /* read the source at its own width: a 32-bit int arrives with its
             * high half undefined (SysV does not sign-extend args), so a
             * 64-bit cvtsi2ss would read it as a giant unsigned. cvtsi2ss off
             * eax takes the i32 honestly. */
            eb(X,0xF3); if(ty_w(X,val_type_x(X,I->operands[0]))>32) rexw(X,X_XMM0,X_RAX);
            eb(X,0x0F);eb(X,0x2A);modrm(X,3,X_XMM0,X_RAX);
            st_xmm_slot(X,X_XMM0,s,0); break;
        case BIR_UITOFP: load_val(X,X_RAX,I->operands[0]); zext_to(X,ty_w(X,val_type_x(X,I->operands[0])));
            eb(X,0xF3);rexw(X,X_XMM0,X_RAX);eb(X,0x0F);eb(X,0x2A);modrm(X,3,X_XMM0,X_RAX);
            st_xmm_slot(X,X_XMM0,s,0); break;
        case BIR_FPTOSI: ld_fval(X,X_XMM0,I->operands[0]);
            eb(X,0xF3);rexw(X,X_RAX,X_XMM0);eb(X,0x0F);eb(X,0x2C);modrm(X,3,X_RAX,X_XMM0);
            sext_to(X,ty_w(X,I->type)); st_slot(X,X_RAX,s); break;
        case BIR_FPTOUI: ld_fval(X,X_XMM0,I->operands[0]);
            eb(X,0xF3);rexw(X,X_RAX,X_XMM0);eb(X,0x0F);eb(X,0x2C);modrm(X,3,X_RAX,X_XMM0);
            zext_to(X,ty_w(X,I->type)); st_slot(X,X_RAX,s); break;

        /* ---- float width. BIR floats are f32, so f32<->f32 is a
         * copy; a real f64 or f16 is a later sitting. ---- */
        case BIR_FPEXT: case BIR_FPTRUNC: ld_fval(X,X_XMM0,I->operands[0]); st_xmm_slot(X,X_XMM0,s,0); break;

        /* ---- reinterpret the bits (32-bit), and pointer<->int. ---- */
        case BIR_BITCAST: {
            uint32_t srct=val_type_x(X,I->operands[0]);
            if (is_float_ty(X,srct)) ld_fbits(X,X_RAX,I->operands[0]); else load_val(X,X_RAX,I->operands[0]);
            if (is_float_ty(X,I->type)){ eb(X,0x89);modrm(X,2,X_RAX,X_RBP);ei32(X,s); }
            else { sext_to(X,ty_w(X,I->type)); st_slot(X,X_RAX,s); }
            break; }
        case BIR_PTRTOINT: load_val(X,X_RAX,I->operands[0]); if(ty_w(X,I->type)<64) sext_to(X,ty_w(X,I->type)); st_slot(X,X_RAX,s); break;
        case BIR_INTTOPTR: load_val(X,X_RAX,I->operands[0]); st_slot(X,X_RAX,s); break;

        /* ---- Math the chip already knows. sqrt and round are single
         * instructions; abs just wipes the sign bit out of the bits; rcp and
         * rsqrt are an honest 1.0 / x rather than the fast-and-loose
         * approximations, because a reactor is a poor place to be roughly
         * right. ---- */
        case BIR_SQRT: ld_fval(X,X_XMM0,I->operands[0]);
            eb(X,0xF3);eb(X,0x0F);eb(X,0x51);modrm(X,3,X_XMM0,X_XMM0);      /* sqrtss xmm0,xmm0 */
            st_xmm_slot(X,X_XMM0,s,0); break;
        case BIR_FABS: ld_fbits(X,X_RAX,I->operands[0]);
            eb(X,0x25);ei32(X,0x7FFFFFFF);                                 /* and eax,0x7FFFFFFF */
            eb(X,0x89);modrm(X,2,X_RAX,X_RBP);ei32(X,s); break;            /* mov [rbp+s],eax */
        case BIR_RCP:
            mov_imm(X,X_RAX,0x3F800000);eb(X,0x66);eb(X,0x0F);eb(X,0x6E);modrm(X,3,X_XMM0,X_RAX); /* xmm0=1.0f */
            ld_fval(X,X_XMM1,I->operands[0]);
            eb(X,0xF3);eb(X,0x0F);eb(X,0x5E);modrm(X,3,X_XMM0,X_XMM1);      /* divss xmm0,xmm1 */
            st_xmm_slot(X,X_XMM0,s,0); break;
        case BIR_RSQ:
            ld_fval(X,X_XMM1,I->operands[0]);
            eb(X,0xF3);eb(X,0x0F);eb(X,0x51);modrm(X,3,X_XMM1,X_XMM1);      /* sqrtss xmm1,xmm1 */
            mov_imm(X,X_RAX,0x3F800000);eb(X,0x66);eb(X,0x0F);eb(X,0x6E);modrm(X,3,X_XMM0,X_RAX);
            eb(X,0xF3);eb(X,0x0F);eb(X,0x5E);modrm(X,3,X_XMM0,X_XMM1);      /* divss xmm0,xmm1 */
            st_xmm_slot(X,X_XMM0,s,0); break;
        case BIR_FLOOR: case BIR_CEIL: case BIR_FTRUNC: case BIR_RNDNE: {
            uint8_t mode=(I->op==BIR_FLOOR)?1:(I->op==BIR_CEIL)?2:(I->op==BIR_FTRUNC)?3:0;
            ld_fval(X,X_XMM0,I->operands[0]);
            eb(X,0x66);eb(X,0x0F);eb(X,0x3A);eb(X,0x0A);modrm(X,3,X_XMM0,X_XMM0);eb(X,mode); /* roundss */
            st_xmm_slot(X,X_XMM0,s,0); break; }

        /* ---- Transcendentals the chip does not do in one instruction, so
         * they go out to libm and the linker earns its keep. sin and cos
         * arrive in turns (the frontend pre-scales by 1/2pi for the GPU's
         * hardware sin), and libm wants radians, so wind the 2pi back in
         * first. exp2 and log2 are already in the units libm uses. ---- */
        case BIR_SIN: case BIR_COS: {
            const char *fn=(I->op==BIR_SIN)?"sinf":"cosf";
            ld_fval(X,X_XMM0,I->operands[0]);
            mov_imm(X,X_RAX,0x40C90FDB);                                /* 2pi as f32 */
            eb(X,0x66);eb(X,0x0F);eb(X,0x6E);modrm(X,3,X_XMM1,X_RAX);   /* movd xmm1,2pi */
            eb(X,0xF3);eb(X,0x0F);eb(X,0x59);modrm(X,3,X_XMM0,X_XMM1);  /* mulss xmm0,xmm1 */
            emit_call_ext(X,fn);
            st_xmm_slot(X,X_XMM0,s,0); break; }
        case BIR_EXP2: call_libm1(X,"exp2f", I->operands[0], s); break;
        case BIR_LOG2: call_libm1(X,"log2f", I->operands[0], s); break;

        /* ---- Atomics. The SIMT loop runs one thread at a time, so an
         * atomic is a load, an op, and a store, handing back the old value
         * the way CUDA promises. No lock prefix: there is nobody else in the
         * building to race. ---- */
        case BIR_ATOMIC_ADD: case BIR_ATOMIC_SUB: case BIR_ATOMIC_AND:
        case BIR_ATOMIC_OR:  case BIR_ATOMIC_XOR: case BIR_ATOMIC_MIN:
        case BIR_ATOMIC_MAX: case BIR_ATOMIC_XCHG: {
            int w64=(int_w(X,I->type)==64);
            load_val(X,X_RCX,I->operands[0]);
            load_val(X,X_RDX,I->operands[1]);
            if (w64) rexw(X,X_RAX,X_RCX);
            eb(X,0x8B);modrm(X,0,X_RAX,X_RCX);                 /* old = *addr */
            if (!w64) sext_to(X,32);
            st_slot(X,X_RAX,s);                                /* result = old */
            uint8_t aop=(I->op==BIR_ATOMIC_ADD)?0x01:(I->op==BIR_ATOMIC_SUB)?0x29:
                        (I->op==BIR_ATOMIC_AND)?0x21:(I->op==BIR_ATOMIC_OR)?0x09:
                        (I->op==BIR_ATOMIC_XOR)?0x31:0x00;
            if (aop){ if(w64)rexw(X,X_RDX,X_RAX); eb(X,aop);modrm(X,3,X_RDX,X_RAX); }
            else if (I->op==BIR_ATOMIC_XCHG){ if(w64)rexw(X,X_RDX,X_RAX); eb(X,0x89);modrm(X,3,X_RDX,X_RAX); }
            else { uint8_t cc=(I->op==BIR_ATOMIC_MIN)?0x4F:0x4C;       /* cmovg / cmovl */
                   if(w64){rexw(X,X_RDX,X_RAX);} eb(X,0x39);modrm(X,3,X_RDX,X_RAX);
                   if(w64){rexw(X,X_RAX,X_RDX);} eb(X,0x0F);eb(X,cc);modrm(X,3,X_RAX,X_RDX); }
            if (w64) rexw(X,X_RAX,X_RCX);
            eb(X,0x89);modrm(X,0,X_RAX,X_RCX);                 /* *addr = new */
            break; }
        case BIR_ATOMIC_CAS: {
            int w64=(int_w(X,I->type)==64);
            load_val(X,X_RCX,I->operands[0]);
            load_val(X,X_RAX,I->operands[1]);                  /* expected */
            load_val(X,X_RDX,I->operands[2]);                  /* new */
            if (w64) rexw(X,X_RDX,X_RCX);
            eb(X,0x0F);eb(X,0xB1);modrm(X,0,X_RDX,X_RCX);      /* cmpxchg [rcx],edx -> eax=old */
            if (!w64) sext_to(X,32);
            st_slot(X,X_RAX,s); break; }
        case BIR_ATOMIC_LOAD: {
            int w64=(int_w(X,I->type)==64);
            load_val(X,X_RCX,I->operands[0]);
            if (w64) rexw(X,X_RAX,X_RCX);
            eb(X,0x8B);modrm(X,0,X_RAX,X_RCX);
            if (!w64) sext_to(X,32);
            st_slot(X,X_RAX,s); break; }
        case BIR_ATOMIC_STORE: {
            int w64=(int_w(X,val_type_x(X,I->operands[1]))==64);
            load_val(X,X_RCX,I->operands[0]);
            load_val(X,X_RAX,I->operands[1]);
            if (w64) rexw(X,X_RAX,X_RCX);
            eb(X,0x89);modrm(X,0,X_RAX,X_RCX); break; }

        /* ---- Barriers: one thread, nothing to wait for. ---- */
        case BIR_BARRIER: case BIR_BARRIER_GROUP: break;

        /* ---- Warp primitives, degenerate at a single lane: a shuffle hands
         * back the lane's own value, a ballot is just this lane's bit, a vote
         * is the predicate itself. ---- */
        case BIR_SHFL: case BIR_SHFL_UP: case BIR_SHFL_DOWN: case BIR_SHFL_XOR:
            load_val(X,X_RAX,I->operands[0]); st_slot(X,X_RAX,s); break;
        case BIR_BALLOT: mov_imm(X,X_RAX,1); st_slot(X,X_RAX,s); break;
        case BIR_VOTE_ANY: case BIR_VOTE_ALL:
            load_val(X,X_RAX,I->operands[0]); st_slot(X,X_RAX,s); break;
        case BIR_ICMP: { load_val(X,X_RAX,I->operands[0]);load_val(X,X_RCX,I->operands[1]); rexw(X,X_RCX,X_RAX);eb(X,0x39);modrm(X,3,X_RCX,X_RAX); int cc; switch(I->subop){ case BIR_ICMP_EQ:cc=XCC_E;break; case BIR_ICMP_NE:cc=XCC_NE;break; case BIR_ICMP_SLT:cc=XCC_L;break; case BIR_ICMP_SLE:cc=XCC_LE;break; case BIR_ICMP_SGT:cc=XCC_G;break; case BIR_ICMP_SGE:cc=XCC_GE;break; case BIR_ICMP_ULT:cc=XCC_B;break; case BIR_ICMP_ULE:cc=XCC_BE;break; case BIR_ICMP_UGT:cc=XCC_A;break; case BIR_ICMP_UGE:cc=XCC_AE;break; default:cc=XCC_NE;break; } eb(X,0x0F);eb(X,(uint8_t)(0x90+cc));modrm(X,3,0,X_RAX); rexw(X,0,X_RAX);eb(X,0x0F);eb(X,0xB6);modrm(X,3,X_RAX,X_RAX); st_slot(X,X_RAX,s); break; }
        case BIR_BR: { uint32_t cur=F->first_block+b; emit_phi_copies(X,I->operands[0],cur); eb(X,0xE9); X->fix[X->n_fix].off=X->codelen; X->fix[X->n_fix++].blk=I->operands[0]; ei32(X,0); break; }
        case BIR_BR_COND: { uint32_t cur=F->first_block+b;
            load_val(X,X_RAX,I->operands[0]); rexw(X,0,X_RAX);eb(X,0x85);modrm(X,3,X_RAX,X_RAX); /* test cond */
            eb(X,0x0F);eb(X,0x84); uint32_t jefix=X->codelen; ei32(X,0); /* je -> false_pad (patched below) */
            emit_phi_copies(X,I->operands[1],cur); /* true edge */
            eb(X,0xE9); X->fix[X->n_fix].off=X->codelen; X->fix[X->n_fix++].blk=I->operands[1]; ei32(X,0);
            { int32_t r=(int32_t)X->codelen-(int32_t)(jefix+4); X->code[jefix]=(uint8_t)r;X->code[jefix+1]=(uint8_t)(r>>8);X->code[jefix+2]=(uint8_t)(r>>16);X->code[jefix+3]=(uint8_t)(r>>24); }
            emit_phi_copies(X,I->operands[2],cur); /* false edge */
            eb(X,0xE9); X->fix[X->n_fix].off=X->codelen; X->fix[X->n_fix++].blk=I->operands[2]; ei32(X,0); break; }
        case BIR_PHI: break; /* resolved by predecessor edge-copies */
        case BIR_RET:
            if (is_kernel){ /* return = this thread is done; jump to loop continue */
                eb(X,0xE9); if(n_ret<CPU_RET_MAX) retfix[n_ret++]=X->codelen; ei32(X,0);
            } else { /* plain function: inline epilogue (a RET block need not be last) */
                if(I->num_operands) load_val(X,X_RAX,I->operands[0]);
                rexw(X,X_RSP,X_RBP);eb(X,0x89);modrm(X,3,X_RBP,X_RSP);eb(X,0x5D);eb(X,0xC3);
            }
            break;
        default: mov_imm(X,X_RAX,0); st_slot(X,X_RAX,s); break;
        }}
    }
    if (is_kernel){
        uint32_t loop_cont=X->codelen;
        ld_slot(X,X_RAX,tid_off); rexw(X,0,X_RAX);eb(X,0x83);modrm(X,3,0,X_RAX);eb(X,1); st_slot(X,X_RAX,tid_off); /* tid++ */
        eb(X,0xE9); { int32_t r=(int32_t)loop_head-(int32_t)(X->codelen+4); ei32(X,r); }                            /* jmp loop_head */
        uint32_t loop_end=X->codelen;
        { int32_t r=(int32_t)loop_end-(int32_t)(jge_fix+4); X->code[jge_fix]=(uint8_t)r;X->code[jge_fix+1]=(uint8_t)(r>>8);X->code[jge_fix+2]=(uint8_t)(r>>16);X->code[jge_fix+3]=(uint8_t)(r>>24); }
        for(int k=0;k<n_ret;k++){ int32_t r=(int32_t)loop_cont-(int32_t)(retfix[k]+4); uint32_t o=retfix[k]; X->code[o]=(uint8_t)r;X->code[o+1]=(uint8_t)(r>>8);X->code[o+2]=(uint8_t)(r>>16);X->code[o+3]=(uint8_t)(r>>24); }
    }
    rexw(X,X_RSP,X_RBP);eb(X,0x89);modrm(X,3,X_RBP,X_RSP);eb(X,0x5D);eb(X,0xC3);
    /* patch branch rel32: target block offset - (fixup_end) */
    for(int k=0;k<X->n_fix;k++){ uint32_t bo=X->blk_off[X->fix[k].blk]; int32_t rel=(int32_t)bo-(int32_t)(X->fix[k].off+4); uint32_t o=X->fix[k].off; X->code[o]=(uint8_t)rel;X->code[o+1]=(uint8_t)(rel>>8);X->code[o+2]=(uint8_t)(rel>>16);X->code[o+3]=(uint8_t)(rel>>24); } X->n_fix=0;
}

void cpu_init(cpu_mod_t *X,const bir_module_t *M){ memset(X,0,sizeof(*X)); X->M=M; }
int cpu_emit(cpu_mod_t *X){ for(uint32_t f=0;f<X->M->num_funcs;f++) cpu_func(X,&X->M->funcs[f]); return 0; }
