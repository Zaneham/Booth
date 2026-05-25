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

static void load_val(cpu_mod_t *X,int reg,uint32_t v){
    if (BIR_VAL_IS_CONST(v)){ uint32_t c=BIR_VAL_INDEX(v); mov_imm(X,reg,c<X->M->num_consts?X->M->consts[c].d.ival:0); }
    else ld_slot(X,reg,slot(X,BIR_VAL_INDEX(v)));
}

/* element size in bytes of a pointer's pointee, default 4 (i32/f32).
 * Drives GEP stride, so it must be width-accurate: i8->1, i16->2,
 * i32/f32->4, i64/f64->8, ptr-to-ptr->8. */
static int pointee_sz(cpu_mod_t *X,uint32_t ty){
    if (ty<X->M->num_types && X->M->types[ty].kind==BIR_TYPE_PTR){
        uint32_t in=X->M->types[ty].inner;
        if (in<X->M->num_types){
            if (X->M->types[in].kind==BIR_TYPE_PTR) return 8;
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

/* pointee bit width of a pointer-typed value, default 32. */
static int pointee_bits(const cpu_mod_t *X,uint32_t v){
    uint32_t ty=val_type(X,v);
    if (ty<X->M->num_types && X->M->types[ty].kind==BIR_TYPE_PTR){
        uint32_t in=X->M->types[ty].inner;
        if (in<X->M->num_types && X->M->types[in].width) return (int)X->M->types[in].width;
    }
    return 32;
}

/* materialize incoming SysV arg p into reg: first six in registers, the
 * rest on the caller's stack at [rbp+16 + 8*(p-6)]. */
static void ld_arg(cpu_mod_t *X,int reg,uint16_t p){
    static const int areg[6]={X_RDI,X_RSI,X_RDX,X_RCX,8,9};
    if (p<6){ rexw(X,areg[p],reg);eb(X,0x89);modrm(X,3,areg[p],reg); }      /* mov reg, areg[p] */
    else { int32_t o=16+(int32_t)(p-6)*8; rexw(X,reg,X_RBP);eb(X,0x8B);modrm(X,2,reg,X_RBP);ei32(X,o); } /* mov reg,[rbp+o] */
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
    for (uint16_t p=0;p<F->num_params;p++){ ld_arg(X,X_RAX,p); st_slot(X,X_RAX,X->slots[p]); }
    uint32_t loop_head=0,jge_fix=0;
    if (is_kernel){
        ld_arg(X,X_RAX,F->num_params); st_slot(X,X_RAX,ntid_off);     /* hidden ntid arg (after user params) */
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
        case BIR_GEP: { int sz=pointee_sz(X,I->type); load_val(X,X_RCX,I->operands[1]); mov_imm(X,X_RAX,sz); eb(X,0x48);eb(X,0x0F);eb(X,0xAF);modrm(X,3,X_RCX,X_RAX); load_val(X,X_RAX,I->operands[0]); rexw(X,X_RCX,X_RAX);eb(X,0x01);modrm(X,3,X_RCX,X_RAX); st_slot(X,X_RAX,s); break; }
        case BIR_LOAD: { load_val(X,X_RAX,I->operands[0]); /* addr in rax */
            const bir_type_t *t=(I->type<X->M->num_types)?&X->M->types[I->type]:0;
            int isflt=t&&(t->kind==BIR_TYPE_FLOAT||t->kind==BIR_TYPE_BFLOAT);
            int w=t?(int)t->width:32;
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
        case BIR_FADD: case BIR_FSUB: case BIR_FMUL: { uint8_t op=(I->op==BIR_FADD)?0x58:(I->op==BIR_FSUB)?0x5C:0x59; int32_t sa=slot(X,(BIR_VAL_IS_CONST(I->operands[0]))?0:BIR_VAL_INDEX(I->operands[0])); int32_t sb=slot(X,(BIR_VAL_IS_CONST(I->operands[1]))?0:BIR_VAL_INDEX(I->operands[1])); eb(X,0xF3);eb(X,0x0F);eb(X,0x10);modrm(X,2,X_XMM0,X_RBP);ei32(X,sa); eb(X,0xF3);eb(X,0x0F);eb(X,0x10);modrm(X,2,X_XMM1,X_RBP);ei32(X,sb); eb(X,0xF3);eb(X,0x0F);eb(X,op);modrm(X,3,X_XMM0,X_XMM1); eb(X,0xF3);eb(X,0x0F);eb(X,0x11);modrm(X,2,X_XMM0,X_RBP);ei32(X,s); break; }
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
