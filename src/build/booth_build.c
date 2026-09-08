/* booth_build.c -- see booth_build.h. */

#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include "bir.h"
#include "backend.h"
#include "backend_cfg.h"
#include "booth_build.h"

/* Bound only; a runaway caller is refused rather than scribbling past M. */
#define BB_MAX_BLK 4096

struct bb {
    bir_module_t *M;
    uint32_t      func;   /* index into M->funcs, or BB_NONE when none is open */
    uint32_t      cur;    /* block being filled, or BB_NONE */
    uint32_t      line;   /* source line stamped on each instruction emitted */
    uint32_t      nblk;   /* blocks reserved in the open function */
    uint32_t      finst;  /* first instruction of the open function */
};

/* ---- Lifecycle ---- */

bb_t *bb_new(void)
{
    bb_t *B = calloc(1, sizeof *B);
    if (B == NULL) return NULL;
    B->M = malloc(sizeof *B->M);
    if (B->M == NULL) { free(B); return NULL; }
    bir_module_init(B->M);
    B->func = BB_NONE;
    B->cur  = BB_NONE;
    return B;
}

void bb_free(bb_t *B)
{
    if (B == NULL) return;
    free(B->M);
    free(B);
}

int bb_full(const bb_t *B) { return B != NULL && B->M->pool_full != 0; }

struct bir_module *bb_module(bb_t *B)
{
    return (B != NULL) ? (struct bir_module *)B->M : NULL;
}

/* ---- Types ---- */

uint32_t bb_void(bb_t *B)                     { return bir_type_void(B->M); }
uint32_t bb_int (bb_t *B, int bits)           { return bir_type_int(B->M, bits); }
uint32_t bb_flt (bb_t *B, int bits)           { return bir_type_float(B->M, bits); }
uint32_t bb_ptr (bb_t *B, uint32_t p, int as) { return bir_type_ptr(B->M, p, as); }
uint32_t bb_arr (bb_t *B, uint32_t e, uint32_t n) { return bir_type_array(B->M, e, n); }
uint32_t bb_vec (bb_t *B, uint32_t e, uint32_t n) { return bir_type_vector(B->M, e, n); }

/* ---- Emission ---- */

static bb_val emit(bb_t *B, uint16_t op, uint8_t subop, uint32_t ty,
                   const uint32_t *ops, uint8_t n)
{
    bir_module_t *M = B->M;
    if (M->num_insts >= BIR_MAX_INSTS) {
        bir_pfull(M, BIR_P_INSTS);
        return BB_NONE;
    }
    uint32_t i = M->num_insts++;
    bir_inst_t *I = &M->insts[i];
    memset(I, 0, sizeof *I);
    I->op = op;
    I->subop = subop;
    I->type = ty;
    I->num_operands = n;
    for (uint8_t k = 0; k < n && k < BIR_OPERANDS_INLINE; k++)
        I->operands[k] = ops[k];
    M->inst_lines[i] = B->line;
    return i;
}

void bb_line(bb_t *B, uint32_t line) { B->line = line; }

/* ---- Immediates ---- */

bb_val bb_ci(bb_t *B, uint32_t ty, int64_t v)
{
    return BIR_MAKE_CONST(bir_const_int(B->M, ty, v));
}

bb_val bb_cf(bb_t *B, uint32_t ty, double v)
{
    return BIR_MAKE_CONST(bir_const_float(B->M, ty, v));
}

int bb_isflt(const bb_t *B, uint32_t ty)
{
    if (B == NULL || ty >= B->M->num_types) return 0;
    uint8_t k = B->M->types[ty].kind;
    return k == BIR_TYPE_FLOAT || k == BIR_TYPE_BFLOAT;
}

/* ---- Functions ---- */

int bb_func(bb_t *B, const char *name, uint32_t ret,
            const uint32_t *params, int nparams, int kernel)
{
    bir_module_t *M = B->M;
    if (B->func != BB_NONE) return -1;              /* already open */
    if (M->num_funcs >= BIR_MAX_FUNCS) { bir_pfull(M, BIR_P_FUNCS); return -1; }

    uint32_t t = bir_type_func(M, ret, params, nparams);
    uint32_t f = M->num_funcs++;
    bir_func_t *F = &M->funcs[f];
    memset(F, 0, sizeof *F);
    F->name        = bir_add_string(M, name, (uint32_t)strlen(name));
    F->type        = t;
    F->first_block = M->num_blocks;
    F->num_params  = (uint16_t)nparams;
    F->cuda_flags  = kernel ? CUDA_GLOBAL : CUDA_DEVICE;

    B->func  = f;
    B->nblk  = 0;
    B->finst = M->num_insts;
    return 0;
}

void bb_fend(bb_t *B)
{
    if (B->func == BB_NONE) return;
    bir_module_t *M = B->M;
    bir_func_t *F = &M->funcs[B->func];
    F->num_blocks  = (uint16_t)(M->num_blocks - F->first_block);
    /* Per function, as bir.h says and as mem2reg's compaction assumes. The
     * module-wide count here moved the wrong number of instructions and
     * spliced one function's body into the next. */
    F->total_insts = M->num_insts - B->finst;
    B->func = BB_NONE;
}

/* ---- Blocks ---- */

uint32_t bb_blk(bb_t *B, const char *name)
{
    bir_module_t *M = B->M;
    if (B->nblk >= BB_MAX_BLK || M->num_blocks >= BIR_MAX_BLOCKS) {
        bir_pfull(M, BIR_P_BLOCKS);
        return BB_NONE;
    }
    /* Index is claimed now so a branch can name it; bb_close fills the body. */
    B->nblk++;
    uint32_t b = M->num_blocks++;
    M->blocks[b].name       = bir_add_string(M, name, (uint32_t)strlen(name));
    M->blocks[b].first_inst = 0;
    M->blocks[b].num_insts  = 0;
    return b;   /* absolute index, which is what a branch operand holds */
}

void bb_open(bb_t *B, uint32_t blk)
{
    B->cur = blk;
    B->M->blocks[blk].first_inst = B->M->num_insts;
}

void bb_close(bb_t *B)
{
    if (B->cur == BB_NONE) return;
    bir_block_t *bl = &B->M->blocks[B->cur];
    bl->num_insts = B->M->num_insts - bl->first_inst;
    B->cur = BB_NONE;
}

/* ---- Instructions ---- */

bb_val bb_param(bb_t *B, int idx, uint32_t ty)
{ return emit(B, BIR_PARAM, (uint8_t)idx, ty, NULL, 0); }

bb_val bb_tid (bb_t *B, int d) { return emit(B, BIR_THREAD_ID, (uint8_t)d, bb_int(B,32), NULL, 0); }
bb_val bb_bid (bb_t *B, int d) { return emit(B, BIR_BLOCK_ID,  (uint8_t)d, bb_int(B,32), NULL, 0); }
bb_val bb_bdim(bb_t *B, int d) { return emit(B, BIR_BLOCK_DIM, (uint8_t)d, bb_int(B,32), NULL, 0); }
bb_val bb_gdim(bb_t *B, int d) { return emit(B, BIR_GRID_DIM,  (uint8_t)d, bb_int(B,32), NULL, 0); }

static bb_val bin(bb_t *B, uint16_t op, uint32_t ty, bb_val a, bb_val b)
{
    uint32_t o[2] = { BIR_MAKE_VAL(a), BIR_MAKE_VAL(b) };
    return emit(B, op, 0, ty, o, 2);
}

bb_val bb_add (bb_t *B, uint32_t t, bb_val a, bb_val b) { return bin(B, BIR_ADD,  t, a, b); }
bb_val bb_sub (bb_t *B, uint32_t t, bb_val a, bb_val b) { return bin(B, BIR_SUB,  t, a, b); }
bb_val bb_mul (bb_t *B, uint32_t t, bb_val a, bb_val b) { return bin(B, BIR_MUL,  t, a, b); }
bb_val bb_fadd(bb_t *B, uint32_t t, bb_val a, bb_val b) { return bin(B, BIR_FADD, t, a, b); }
bb_val bb_fsub(bb_t *B, uint32_t t, bb_val a, bb_val b) { return bin(B, BIR_FSUB, t, a, b); }
bb_val bb_fmul(bb_t *B, uint32_t t, bb_val a, bb_val b) { return bin(B, BIR_FMUL, t, a, b); }

bb_val bb_icmp(bb_t *B, int pred, bb_val a, bb_val b)
{
    uint32_t o[2] = { BIR_MAKE_VAL(a), BIR_MAKE_VAL(b) };
    return emit(B, BIR_ICMP, (uint8_t)pred, bb_int(B, 1), o, 2);
}

bb_val bb_fcmp(bb_t *B, int pred, bb_val a, bb_val b)
{
    uint32_t o[2] = { BIR_MAKE_VAL(a), BIR_MAKE_VAL(b) };
    return emit(B, BIR_FCMP, (uint8_t)pred, bb_int(B, 1), o, 2);
}

/* Indexed by the enums in booth_build.h, so a gap here is a compile error
 * rather than a silently wrong opcode. */
static const uint16_t op2_tab[] = {
    BIR_ADD, BIR_SUB, BIR_MUL, BIR_SDIV, BIR_UDIV, BIR_SREM, BIR_UREM,
    BIR_FADD, BIR_FSUB, BIR_FMUL, BIR_FDIV, BIR_FREM,
    BIR_AND, BIR_OR, BIR_XOR, BIR_SHL, BIR_LSHR, BIR_ASHR
};

static const uint16_t fn_tab[] = {
    BIR_SQRT, BIR_RSQ, BIR_RCP, BIR_EXP2, BIR_LOG2, BIR_SIN, BIR_COS,
    BIR_FABS, BIR_FLOOR, BIR_CEIL, BIR_FTRUNC, BIR_RNDNE,
    BIR_FMAX, BIR_FMIN
};

static const uint16_t cvt_tab[] = {
    BIR_TRUNC, BIR_ZEXT, BIR_SEXT, BIR_FPTRUNC, BIR_FPEXT,
    BIR_FPTOSI, BIR_FPTOUI, BIR_SITOFP, BIR_UITOFP, BIR_BITCAST
};

#define NELEM(a) ((int)(sizeof (a) / sizeof (a)[0]))

bb_val bb_op(bb_t *B, int op, uint32_t ty, bb_val a, bb_val b)
{
    if (op < 0 || op >= NELEM(op2_tab)) return BB_NONE;
    return bin(B, op2_tab[op], ty, a, b);
}

bb_val bb_fn1(bb_t *B, int fn, uint32_t ty, bb_val a)
{
    if (fn < 0 || fn >= NELEM(fn_tab)) return BB_NONE;
    uint32_t o[1] = { BIR_MAKE_VAL(a) };
    return emit(B, fn_tab[fn], 0, ty, o, 1);
}

bb_val bb_fn2(bb_t *B, int fn, uint32_t ty, bb_val a, bb_val b)
{
    if (fn < 0 || fn >= NELEM(fn_tab)) return BB_NONE;
    return bin(B, fn_tab[fn], ty, a, b);
}

bb_val bb_cvt(bb_t *B, int c, uint32_t dst, bb_val a)
{
    if (c < 0 || c >= NELEM(cvt_tab)) return BB_NONE;
    uint32_t o[1] = { BIR_MAKE_VAL(a) };
    return emit(B, cvt_tab[c], 0, dst, o, 1);
}

bb_val bb_sel(bb_t *B, uint32_t ty, bb_val c, bb_val t, bb_val f)
{
    uint32_t o[3] = { BIR_MAKE_VAL(c), BIR_MAKE_VAL(t), BIR_MAKE_VAL(f) };
    return emit(B, BIR_SELECT, 0, ty, o, 3);
}

uint32_t bb_tyof(const bb_t *B, bb_val v)
{
    if (B == NULL || v == BB_NONE) return 0;
    if (BIR_VAL_IS_CONST(v)) {
        uint32_t i = v & ~BIR_VAL_CONST_BIT;
        return (i < B->M->num_consts) ? B->M->consts[i].type : 0;
    }
    return (v < B->M->num_insts) ? B->M->insts[v].type : 0;
}

bb_val bb_alca(bb_t *B, uint32_t ty)
{
    return emit(B, BIR_ALLOCA, 0, ty, NULL, 0);
}

bb_val bb_shal(bb_t *B, uint32_t ty)
{
    return emit(B, BIR_SHARED_ALLOC, 0, ty, NULL, 0);
}

/* No min or max here on purpose: BIR carries one opcode for each and the
 * backends disagree on its signedness, so the operation has no single
 * meaning to expose yet. */
static const uint16_t atom_tab[] = {
    BIR_ATOMIC_ADD, BIR_ATOMIC_SUB, BIR_ATOMIC_AND,
    BIR_ATOMIC_OR,  BIR_ATOMIC_XOR, BIR_ATOMIC_XCHG
};

bb_val bb_atom(bb_t *B, int op, int order, uint32_t ty, bb_val p, bb_val v)
{
    if (op < 0 || op >= (int)(sizeof atom_tab / sizeof atom_tab[0]))
        return BB_NONE;
    uint32_t o[2] = { BIR_MAKE_VAL(p), BIR_MAKE_VAL(v) };
    return emit(B, atom_tab[op], (uint8_t)order, ty, o, 2);
}

void bb_barr(bb_t *B)
{
    emit(B, BIR_BARRIER, 0, bb_void(B), NULL, 0);
}

bb_val bb_gep(bb_t *B, uint32_t ty, bb_val base, bb_val idx)
{
    uint32_t o[2] = { BIR_MAKE_VAL(base), BIR_MAKE_VAL(idx) };
    return emit(B, BIR_GEP, 0, ty, o, 2);
}

bb_val bb_load(bb_t *B, uint32_t ty, bb_val addr)
{
    uint32_t o[1] = { BIR_MAKE_VAL(addr) };
    return emit(B, BIR_LOAD, 0, ty, o, 1);
}

void bb_store(bb_t *B, bb_val v, bb_val addr)
{
    uint32_t o[2] = { BIR_MAKE_VAL(v), BIR_MAKE_VAL(addr) };
    emit(B, BIR_STORE, 0, bb_void(B), o, 2);
}

void bb_br(bb_t *B, uint32_t blk)
{
    uint32_t o[1] = { blk };
    emit(B, BIR_BR, 0, bb_void(B), o, 1);
}

void bb_brif(bb_t *B, bb_val c, uint32_t t, uint32_t f, uint32_t merge)
{
    uint32_t o[4] = { BIR_MAKE_VAL(c), t, f, merge };
    emit(B, BIR_BR_COND, 0, bb_void(B), o, 4);
}

void bb_ret(bb_t *B) { emit(B, BIR_RET, 0, bb_void(B), NULL, 0); }

/* The printed type comes from the operand, so the instruction itself is void
 * and ty is only what an immediate operand would be typed as. */
void bb_retv(bb_t *B, uint32_t ty, bb_val v)
{
    uint32_t o[1] = { BIR_MAKE_VAL(v) };

    if (B != NULL && B->func != BB_NONE && B->func < B->M->num_funcs) {
        uint32_t ft = B->M->funcs[B->func].type;
        if (ft < B->M->num_types
            && B->M->types[ft].kind == BIR_TYPE_FUNC
            && B->M->types[ft].inner < B->M->num_types
            && B->M->types[B->M->types[ft].inner].kind == BIR_TYPE_VOID)
            B->M->types[ft].inner = ty;
    }
    emit(B, BIR_RET, 0, bb_void(B), o, 1);
}

int bb_ffind(const bb_t *B, const char *name)
{
    if (B == NULL || name == NULL) return -1;
    const bir_module_t *M = B->M;
    for (uint32_t i = 0; i < M->num_funcs; i++) {
        if (M->funcs[i].name < M->string_len &&
            strcmp(&M->strings[M->funcs[i].name], name) == 0)
            return (int)i;
    }
    return -1;
}

bb_val bb_call(bb_t *B, uint32_t ty, int fidx, const bb_val *args, int nargs)
{
    if (fidx < 0 || nargs < 0 || nargs > BB_MAX_ARGS) return BB_NONE;
    uint32_t o[BB_MAX_ARGS + 1];
    o[0] = (uint32_t)fidx;
    for (int i = 0; i < nargs; i++) o[i + 1] = BIR_MAKE_VAL(args[i]);
    return emit(B, BIR_CALL, 0, ty, o, (uint8_t)(nargs + 1));
}

/* ---- Output ---- */

int bb_print(bb_t *B, const char *path)
{
    FILE *f = (path == NULL) ? stdout : fopen(path, "w");
    if (f == NULL) return -1;
    bir_print_module(B->M, f);
    fprintf(f, "\n; %u functions, %u globals, %u instructions\n",
            B->M->num_funcs, B->M->num_globals, B->M->num_insts);
    if (f != stdout) fclose(f);
    return 0;
}

/* Until the registry takes a name, selection goes through its flag. These are
   the flags that set `on`; the others each backend declares only pick a target
   variant, so passing one selects nothing and be_run then does nothing. */
static const char *flag_for(const char *target)
{
    if (strcmp(target, "cpu")     == 0) return "--cpu";
    if (strcmp(target, "rv64")    == 0) return "--rv64";
    if (strcmp(target, "amdgpu")  == 0) return "--amdgpu";
    if (strcmp(target, "nvptx")   == 0) return "--nvidia-ptx";
    if (strcmp(target, "tensix")  == 0) return "--tensix";
    if (strcmp(target, "metal")   == 0) return "--metal";
    if (strcmp(target, "intel")   == 0) return "--intel-spirv";
    return NULL;
}

int bb_emit(bb_t *B, const char *target, const char *path)
{
    if (bir_pchk(B->M, "booth_build") != 0) return -1;

    const char *flag = flag_for(target);
    if (flag == NULL) {
        fprintf(stderr, "booth_build: unknown target '%s'\n", target);
        return -1;
    }

    int used = 0;
    if (be_parse_flag(flag, NULL, &used) != 1) {
        fprintf(stderr, "booth_build: no backend claimed '%s'\n", flag);
        return -1;
    }

    be_cfg_t cfg;
    memset(&cfg, 0, sizeof cfg);
    cfg.output_file = path;

    return be_run((const struct bir_module *)B->M,
                  (const struct be_cfg *)&cfg);
}
