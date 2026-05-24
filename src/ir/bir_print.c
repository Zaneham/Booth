#include "bir.h"
#include <stdio.h>
#include <string.h>

/* Where the IR learns to speak. */

/* ---- Type Strings ---- */

static int print_simple_type(const bir_module_t *M, uint32_t tidx,
                             char *buf, int size)
{
    if (tidx >= M->num_types)
        return snprintf(buf, (size_t)size, "???");
    const bir_type_t *t = &M->types[tidx];
    switch (t->kind) {
    case BIR_TYPE_VOID:  return snprintf(buf, (size_t)size, "void");
    case BIR_TYPE_INT:   return snprintf(buf, (size_t)size, "i%u", t->width);
    case BIR_TYPE_FLOAT:
        if (t->width == 16) return snprintf(buf, (size_t)size, "f16");
        if (t->width == 32) return snprintf(buf, (size_t)size, "f32");
        return snprintf(buf, (size_t)size, "f64");
    case BIR_TYPE_BFLOAT:
        return snprintf(buf, (size_t)size, "bf16");
    case BIR_TYPE_PTR:
        return snprintf(buf, (size_t)size, "ptr<%s>",
                        bir_addrspace_name(t->addrspace));
    default:
        return snprintf(buf, (size_t)size, "type_%u", tidx);
    }
}

int bir_type_str(const bir_module_t *M, uint32_t tidx, char *buf, int size)
{
    if (tidx >= M->num_types)
        return snprintf(buf, (size_t)size, "???");
    const bir_type_t *t = &M->types[tidx];
    char inner[64];
    int pos = 0;

    switch (t->kind) {
    case BIR_TYPE_VOID:
    case BIR_TYPE_INT:
    case BIR_TYPE_FLOAT:
    case BIR_TYPE_BFLOAT:
        return print_simple_type(M, tidx, buf, size);

    case BIR_TYPE_PTR:
        print_simple_type(M, t->inner, inner, sizeof(inner));
        return snprintf(buf, (size_t)size, "ptr<%s, %s>",
                        bir_addrspace_name(t->addrspace), inner);

    case BIR_TYPE_ARRAY:
        print_simple_type(M, t->inner, inner, sizeof(inner));
        return snprintf(buf, (size_t)size, "[%u x %s]", t->count, inner);

    case BIR_TYPE_VECTOR:
        print_simple_type(M, t->inner, inner, sizeof(inner));
        return snprintf(buf, (size_t)size, "<%u x %s>", t->width, inner);

    case BIR_TYPE_STRUCT:
        pos = snprintf(buf, (size_t)size, "{");
        for (uint16_t i = 0; i < t->num_fields && pos < size - 1; i++) {
            if (i > 0) pos += snprintf(buf+pos, (size_t)(size-pos), ", ");
            print_simple_type(M, M->type_fields[t->count + i],
                              inner, sizeof(inner));
            pos += snprintf(buf+pos, (size_t)(size-pos), "%s", inner);
        }
        pos += snprintf(buf+pos, (size_t)(size-pos), "}");
        return pos;

    case BIR_TYPE_FUNC:
        print_simple_type(M, t->inner, inner, sizeof(inner));
        pos = snprintf(buf, (size_t)size, "(");
        for (uint16_t i = 0; i < t->num_fields && pos < size - 1; i++) {
            if (i > 0) pos += snprintf(buf+pos, (size_t)(size-pos), ", ");
            print_simple_type(M, M->type_fields[t->count + i],
                              inner, sizeof(inner));
            pos += snprintf(buf+pos, (size_t)(size-pos), "%s", inner);
        }
        pos += snprintf(buf+pos, (size_t)(size-pos), ") -> %s", inner);
        return pos;

    default:
        return snprintf(buf, (size_t)size, "type_%u", tidx);
    }
}

/* ---- Value References ---- */

static void print_val(const bir_module_t *M, uint32_t ref,
                      uint32_t base_inst, FILE *out)
{
    if (ref == BIR_VAL_NONE) {
        fprintf(out, "none");
        return;
    }
    if (BIR_VAL_IS_CONST(ref)) {
        uint32_t ci = BIR_VAL_INDEX(ref);
        if (ci >= M->num_consts) { fprintf(out, "const_???"); return; }
        const bir_const_t *c = &M->consts[ci];
        switch (c->kind) {
        case BIR_CONST_INT:   fprintf(out, "%lld", (long long)c->d.ival); break;
        case BIR_CONST_FLOAT: fprintf(out, "%g", c->d.fval); break;
        case BIR_CONST_BYTES: fprintf(out, "bytes[%u@%u]",
                                       c->d.bytes.len, c->d.bytes.off); break;
        case BIR_CONST_NULL:  fprintf(out, "null"); break;
        case BIR_CONST_UNDEF: fprintf(out, "undef"); break;
        case BIR_CONST_ZERO:  fprintf(out, "zeroinit"); break;
        default:              fprintf(out, "const_%u", ci); break;
        }
        return;
    }
    fprintf(out, "%%%u", BIR_VAL_INDEX(ref) - base_inst);
}

static uint32_t val_type(const bir_module_t *M, uint32_t ref)
{
    if (ref == BIR_VAL_NONE) return 0;
    if (BIR_VAL_IS_CONST(ref)) {
        uint32_t ci = BIR_VAL_INDEX(ref);
        return ci < M->num_consts ? M->consts[ci].type : 0;
    }
    uint32_t ii = BIR_VAL_INDEX(ref);
    return ii < M->num_insts ? M->insts[ii].type : 0;
}

/* ---- Block Labels ---- */

static void print_block_label(const bir_module_t *M, uint32_t bidx,
                              uint32_t base_block, FILE *out)
{
    if (bidx < M->num_blocks) {
        const bir_block_t *b = &M->blocks[bidx];
        if (b->name < M->string_len && M->strings[b->name] != '\0') {
            fprintf(out, "%s", &M->strings[b->name]);
            return;
        }
    }
    fprintf(out, "bb%u", bidx - base_block);
}

/* ---- CUDA Flags ---- */

static void print_cuda_flags(uint16_t flags, FILE *out)
{
    if (flags & CUDA_GLOBAL)      fprintf(out, " __global__");
    if (flags & CUDA_DEVICE)      fprintf(out, " __device__");
    if (flags & CUDA_HOST)        fprintf(out, " __host__");
    if (flags & CUDA_SHARED)      fprintf(out, " __shared__");
    if (flags & CUDA_CONSTANT)    fprintf(out, " __constant__");
    if (flags & CUDA_MANAGED)     fprintf(out, " __managed__");
    if (flags & CUDA_FORCEINLINE) fprintf(out, " __forceinline__");
    if (flags & CUDA_NOINLINE)    fprintf(out, " __noinline__");
}

/* ---- Instructions ---- */

static const char *dim_str(int d)
{
    if (d == 0) return ".x";
    if (d == 1) return ".y";
    if (d == 2) return ".z";
    return "";
}

static void print_inst(const bir_module_t *M, const bir_inst_t *I,
                       uint32_t global_idx, uint32_t local_idx,
                       uint32_t base_inst, uint32_t base_block, FILE *out)
{
    char tbuf[128];
    int is_void = (I->type < M->num_types
                   && M->types[I->type].kind == BIR_TYPE_VOID);

    fprintf(out, "    ");
    if (!is_void)
        fprintf(out, "%%%u = ", local_idx);
    fprintf(out, "%s", bir_op_name(I->op));

    switch (I->op) {

    /* Binary arithmetic/bitwise: OP TYPE %a, %b */
    case BIR_ADD: case BIR_SUB: case BIR_MUL:
    case BIR_SDIV: case BIR_UDIV: case BIR_SREM: case BIR_UREM:
    case BIR_FADD: case BIR_FSUB: case BIR_FMUL: case BIR_FDIV: case BIR_FREM:
    case BIR_AND: case BIR_OR: case BIR_XOR:
    case BIR_SHL: case BIR_LSHR: case BIR_ASHR:
        bir_type_str(M, I->type, tbuf, sizeof(tbuf));
        fprintf(out, " %s ", tbuf);
        print_val(M, I->operands[0], base_inst, out);
        fprintf(out, ", ");
        print_val(M, I->operands[1], base_inst, out);
        break;

    /* Comparison: icmp/fcmp PRED OPERAND_TYPE %a, %b */
    case BIR_ICMP:
    case BIR_FCMP: {
        uint32_t op_t = val_type(M, I->operands[0]);
        bir_type_str(M, op_t, tbuf, sizeof(tbuf));
        fprintf(out, " %s %s ", bir_cmp_name(I->subop), tbuf);
        print_val(M, I->operands[0], base_inst, out);
        fprintf(out, ", ");
        print_val(M, I->operands[1], base_inst, out);
        break;
    }

    /* Conversion: CONV SRC_TYPE %a to DST_TYPE */
    case BIR_TRUNC: case BIR_ZEXT: case BIR_SEXT:
    case BIR_FPTRUNC: case BIR_FPEXT:
    case BIR_FPTOSI: case BIR_FPTOUI: case BIR_SITOFP: case BIR_UITOFP:
    case BIR_PTRTOINT: case BIR_INTTOPTR: case BIR_BITCAST: {
        char src[64];
        bir_type_str(M, val_type(M, I->operands[0]), src, sizeof(src));
        bir_type_str(M, I->type, tbuf, sizeof(tbuf));
        fprintf(out, " %s ", src);
        print_val(M, I->operands[0], base_inst, out);
        fprintf(out, " to %s", tbuf);
        break;
    }

    /* Alloca / shared_alloc: TYPE [, align N] */
    case BIR_ALLOCA:
    case BIR_SHARED_ALLOC:
        bir_type_str(M, I->type, tbuf, sizeof(tbuf));
        fprintf(out, " %s", tbuf);
        if (I->subop > 0)
            fprintf(out, ", align %u", 1u << I->subop);
        break;

    /* global_ref: @name (subop = global index) */
    case BIR_GLOBAL_REF: {
        uint32_t gi = I->subop;
        bir_type_str(M, I->type, tbuf, sizeof(tbuf));
        if (gi < M->num_globals)
            fprintf(out, " %s, @%s", tbuf, &M->strings[M->globals[gi].name]);
        else
            fprintf(out, " %s, @?", tbuf);
        break;
    }

    /* Load: [volatile] TYPE, %ptr */
    case BIR_LOAD:
        if (I->subop) fprintf(out, " volatile");
        bir_type_str(M, I->type, tbuf, sizeof(tbuf));
        fprintf(out, " %s, ", tbuf);
        print_val(M, I->operands[0], base_inst, out);
        break;

    /* Store: [volatile] TYPE %val, %ptr */
    case BIR_STORE: {
        if (I->subop) fprintf(out, " volatile");
        uint32_t vt = val_type(M, I->operands[0]);
        bir_type_str(M, vt, tbuf, sizeof(tbuf));
        fprintf(out, " %s ", tbuf);
        print_val(M, I->operands[0], base_inst, out);
        fprintf(out, ", ");
        print_val(M, I->operands[1], base_inst, out);
        break;
    }

    /* GEP: RESULT_TYPE, %base, %idx... */
    case BIR_GEP:
        bir_type_str(M, I->type, tbuf, sizeof(tbuf));
        fprintf(out, " %s", tbuf);
        for (uint8_t i = 0; i < I->num_operands; i++) {
            fprintf(out, ", ");
            print_val(M, I->operands[i], base_inst, out);
        }
        break;

    /* Branch: br BLOCK */
    case BIR_BR:
        fprintf(out, " ");
        print_block_label(M, I->operands[0], base_block, out);
        break;

    /* Conditional branch: br_cond %cond, TRUE, FALSE */
    case BIR_BR_COND:
        fprintf(out, " ");
        print_val(M, I->operands[0], base_inst, out);
        fprintf(out, ", ");
        print_block_label(M, I->operands[1], base_block, out);
        fprintf(out, ", ");
        print_block_label(M, I->operands[2], base_block, out);
        if (I->num_operands > 3) {
            fprintf(out, " merge ");
            print_block_label(M, I->operands[3], base_block, out);
        }
        break;

    /* Switch: switch %val, default BLOCK, [const: BLOCK], ... */
    case BIR_SWITCH:
        if (I->num_operands == BIR_OPERANDS_OVERFLOW) {
            uint32_t start = I->operands[0];
            uint32_t count = I->operands[1];
            if (count >= 2) {
                fprintf(out, " ");
                print_val(M, M->extra_operands[start], base_inst, out);
                fprintf(out, ", default ");
                print_block_label(M, M->extra_operands[start + 1],
                                  base_block, out);
                for (uint32_t i = 2; i + 1 < count; i += 2) {
                    fprintf(out, ", [");
                    print_val(M, M->extra_operands[start + i],
                              base_inst, out);
                    fprintf(out, ": ");
                    print_block_label(M, M->extra_operands[start + i + 1],
                                      base_block, out);
                    fprintf(out, "]");
                }
            }
        } else {
            fprintf(out, " ");
            print_val(M, I->operands[0], base_inst, out);
            fprintf(out, ", default ");
            print_block_label(M, I->operands[1], base_block, out);
        }
        break;

    /* Return: ret TYPE %val | ret void */
    case BIR_RET:
        if (I->num_operands > 0) {
            uint32_t rt = val_type(M, I->operands[0]);
            bir_type_str(M, rt, tbuf, sizeof(tbuf));
            fprintf(out, " %s ", tbuf);
            print_val(M, I->operands[0], base_inst, out);
        } else {
            fprintf(out, " void");
        }
        break;

    case BIR_UNREACHABLE:
        break;

    /* Phi: phi TYPE [block: %val, block: %val, ...] */
    case BIR_PHI: {
        bir_type_str(M, I->type, tbuf, sizeof(tbuf));
        fprintf(out, " %s ", tbuf);
        if (I->num_operands == BIR_OPERANDS_OVERFLOW) {
            uint32_t start = I->operands[0];
            uint32_t count = I->operands[1];
            for (uint32_t i = 0; i + 1 < count; i += 2) {
                if (i > 0) fprintf(out, ", ");
                fprintf(out, "[");
                print_block_label(M, M->extra_operands[start + i],
                                  base_block, out);
                fprintf(out, ": ");
                print_val(M, M->extra_operands[start + i + 1],
                          base_inst, out);
                fprintf(out, "]");
            }
        } else {
            for (uint8_t i = 0; i + 1 < I->num_operands; i += 2) {
                if (i > 0) fprintf(out, ", ");
                fprintf(out, "[");
                print_block_label(M, I->operands[i], base_block, out);
                fprintf(out, ": ");
                print_val(M, I->operands[i + 1], base_inst, out);
                fprintf(out, "]");
            }
        }
        break;
    }

    /* Param: param TYPE */
    case BIR_PARAM:
        bir_type_str(M, I->type, tbuf, sizeof(tbuf));
        fprintf(out, " %s", tbuf);
        break;

    /* Thread model: thread_id.DIM */
    case BIR_THREAD_ID: case BIR_BLOCK_ID:
    case BIR_BLOCK_DIM: case BIR_GRID_DIM:
        fprintf(out, "%s", dim_str(I->subop));
        break;

    /* Barriers */
    case BIR_BARRIER:
    case BIR_BARRIER_GROUP:
        break;

    /* Atomics: atomic_OP ORDER TYPE %ptr, %val */
    case BIR_ATOMIC_ADD: case BIR_ATOMIC_SUB:
    case BIR_ATOMIC_AND: case BIR_ATOMIC_OR: case BIR_ATOMIC_XOR:
    case BIR_ATOMIC_MIN: case BIR_ATOMIC_MAX:
    case BIR_ATOMIC_XCHG:
        bir_type_str(M, I->type, tbuf, sizeof(tbuf));
        fprintf(out, " %s %s ", bir_order_name(I->subop), tbuf);
        print_val(M, I->operands[0], base_inst, out);
        fprintf(out, ", ");
        print_val(M, I->operands[1], base_inst, out);
        break;

    /* CAS: atomic_cas ORDER TYPE %ptr, %cmp, %new */
    case BIR_ATOMIC_CAS:
        bir_type_str(M, I->type, tbuf, sizeof(tbuf));
        fprintf(out, " %s %s ", bir_order_name(I->subop), tbuf);
        print_val(M, I->operands[0], base_inst, out);
        fprintf(out, ", ");
        print_val(M, I->operands[1], base_inst, out);
        fprintf(out, ", ");
        print_val(M, I->operands[2], base_inst, out);
        break;

    /* Atomic load/store */
    case BIR_ATOMIC_LOAD:
        bir_type_str(M, I->type, tbuf, sizeof(tbuf));
        fprintf(out, " %s %s, ", bir_order_name(I->subop), tbuf);
        print_val(M, I->operands[0], base_inst, out);
        break;
    case BIR_ATOMIC_STORE: {
        uint32_t vt = val_type(M, I->operands[0]);
        bir_type_str(M, vt, tbuf, sizeof(tbuf));
        fprintf(out, " %s %s ", bir_order_name(I->subop), tbuf);
        print_val(M, I->operands[0], base_inst, out);
        fprintf(out, ", ");
        print_val(M, I->operands[1], base_inst, out);
        break;
    }

    /* Warp ops */
    case BIR_SHFL: case BIR_SHFL_UP: case BIR_SHFL_DOWN: case BIR_SHFL_XOR:
    case BIR_BALLOT: case BIR_VOTE_ANY: case BIR_VOTE_ALL:
        bir_type_str(M, I->type, tbuf, sizeof(tbuf));
        fprintf(out, " %s ", tbuf);
        for (uint8_t i = 0; i < I->num_operands; i++) {
            if (i > 0) fprintf(out, ", ");
            print_val(M, I->operands[i], base_inst, out);
        }
        break;

    /* Call: call TYPE @func(%a, %b, ...) */
    case BIR_CALL: {
        bir_type_str(M, I->type, tbuf, sizeof(tbuf));
        fprintf(out, " %s ", tbuf);
        if (I->num_operands == BIR_OPERANDS_OVERFLOW) {
            uint32_t start = I->operands[0];
            uint32_t count = I->operands[1];
            uint32_t fi = count > 0 ? M->extra_operands[start] : 0;
            if (fi < M->num_funcs && M->funcs[fi].name < M->string_len)
                fprintf(out, "@%s(", &M->strings[M->funcs[fi].name]);
            else
                fprintf(out, "@func_%u(", fi);
            for (uint32_t i = 1; i < count; i++) {
                if (i > 1) fprintf(out, ", ");
                print_val(M, M->extra_operands[start + i], base_inst, out);
            }
            fprintf(out, ")");
        } else {
            uint32_t fi = I->operands[0];
            if (fi < M->num_funcs && M->funcs[fi].name < M->string_len)
                fprintf(out, "@%s(", &M->strings[M->funcs[fi].name]);
            else
                fprintf(out, "@func_%u(", fi);
            for (uint8_t i = 1; i < I->num_operands; i++) {
                if (i > 1) fprintf(out, ", ");
                print_val(M, I->operands[i], base_inst, out);
            }
            fprintf(out, ")");
        }
        break;
    }

    /* Select: select TYPE %cond, %true, %false */
    case BIR_SELECT:
        bir_type_str(M, I->type, tbuf, sizeof(tbuf));
        fprintf(out, " %s ", tbuf);
        print_val(M, I->operands[0], base_inst, out);
        fprintf(out, ", ");
        print_val(M, I->operands[1], base_inst, out);
        fprintf(out, ", ");
        print_val(M, I->operands[2], base_inst, out);
        break;

    case BIR_INLINE_ASM:
        fprintf(out, " \"...\"");
        break;

    default:
        for (uint8_t i = 0; i < I->num_operands && i < BIR_OPERANDS_INLINE; i++) {
            fprintf(out, " ");
            print_val(M, I->operands[i], base_inst, out);
        }
        break;
    }

    /* Source location — because "it broke somewhere" is not a diagnosis */
    if (global_idx < BIR_MAX_INSTS && M->inst_lines[global_idx] > 0)
        fprintf(out, "  ; line %u", M->inst_lines[global_idx]);

    fprintf(out, "\n");
}

/* ---- Module Printer ---- */

void bir_print_module(const bir_module_t *M, FILE *out)
{
    char tbuf[128];
    uint32_t i;

    fprintf(out, "; BarraCUDA IR\n");

    /* Globals */
    for (i = 0; i < M->num_globals; i++) {
        const bir_global_t *g = &M->globals[i];
        fprintf(out, "\n");
        if (g->name < M->string_len && M->strings[g->name] != '\0')
            fprintf(out, "@%s = ", &M->strings[g->name]);
        else
            fprintf(out, "@global_%u = ", i);
        fprintf(out, "%s ", g->is_const ? "constant" : "global");
        bir_type_str(M, g->type, tbuf, sizeof(tbuf));
        fprintf(out, "%s", tbuf);
        if (g->initializer != BIR_VAL_NONE) {
            uint32_t ci = BIR_VAL_INDEX(g->initializer);
            if (ci < M->num_consts) {
                fprintf(out, " ");
                const bir_const_t *c = &M->consts[ci];
                switch (c->kind) {
                case BIR_CONST_INT:   fprintf(out, "%lld", (long long)c->d.ival); break;
                case BIR_CONST_FLOAT: fprintf(out, "%g", c->d.fval); break;
                case BIR_CONST_NULL:  fprintf(out, "null"); break;
                case BIR_CONST_ZERO:  fprintf(out, "zeroinit"); break;
                case BIR_CONST_BYTES: {
                    /* Print as a quoted string with C escapes; the
                     * lex/parse roundtrip should produce identical
                     * bytes. */
                    fprintf(out, "\"");
                    const char *b = &M->strings[c->d.bytes.off];
                    uint32_t bn = c->d.bytes.len;
                    /* Drop the trailing null we appended so the
                     * printed form matches the source. */
                    if (bn > 0 && b[bn - 1] == '\0') bn--;
                    for (uint32_t j = 0; j < bn; j++) {
                        unsigned char ch = (unsigned char)b[j];
                        if (ch == '\\')      fprintf(out, "\\\\");
                        else if (ch == '"')  fprintf(out, "\\\"");
                        else if (ch == '\n') fprintf(out, "\\n");
                        else if (ch == '\t') fprintf(out, "\\t");
                        else if (ch == '\r') fprintf(out, "\\r");
                        else if (ch >= 32 && ch < 127) fprintf(out, "%c", ch);
                        else fprintf(out, "\\x%02x", ch);
                    }
                    fprintf(out, "\"");
                    break;
                }
                default:              fprintf(out, "const_%u", ci); break;
                }
            }
        }
        print_cuda_flags(g->cuda_flags, out);
        fprintf(out, "\n");
    }

    /* Functions */
    for (i = 0; i < M->num_funcs; i++) {
        const bir_func_t *F = &M->funcs[i];
        uint32_t base_inst = 0;
        uint32_t base_block = F->first_block;
        uint16_t b;

        if (F->num_blocks > 0)
            base_inst = M->blocks[F->first_block].first_inst;

        fprintf(out, "\nfunc ");
        if (F->name < M->string_len && M->strings[F->name] != '\0')
            fprintf(out, "@%s", &M->strings[F->name]);
        else
            fprintf(out, "@func_%u", i);

        /* Parameters from PARAM instructions */
        fprintf(out, "(");
        for (uint16_t p = 0; p < F->num_params; p++) {
            if (p > 0) fprintf(out, ", ");
            uint32_t pi = base_inst + p;
            if (pi < M->num_insts) {
                bir_type_str(M, M->insts[pi].type, tbuf, sizeof(tbuf));
                fprintf(out, "%s %%%u", tbuf, p);
            }
        }
        fprintf(out, ")");
        print_cuda_flags(F->cuda_flags, out);
        if (F->launch_bounds_max > 0) {
            fprintf(out, " __launch_bounds__(%u", F->launch_bounds_max);
            if (F->launch_bounds_min > 0)
                fprintf(out, ", %u", F->launch_bounds_min);
            fprintf(out, ")");
        }

        if (F->num_blocks == 0) {
            fprintf(out, "\n");
            continue;
        }
        fprintf(out, " {\n");

        /* Blocks */
        for (b = 0; b < F->num_blocks; b++) {
            uint32_t bidx = F->first_block + b;
            const bir_block_t *B = &M->blocks[bidx];
            uint32_t j;

            /* Block label */
            if (B->name < M->string_len && M->strings[B->name] != '\0')
                fprintf(out, "%s:\n", &M->strings[B->name]);
            else
                fprintf(out, "bb%u:\n", b);

            /* Instructions — skip params (shown in signature) */
            for (j = 0; j < B->num_insts; j++) {
                uint32_t iidx = B->first_inst + j;
                if (M->insts[iidx].op == BIR_PARAM)
                    continue;
                uint32_t local = iidx - base_inst;
                print_inst(M, &M->insts[iidx], iidx, local, base_inst,
                           base_block, out);
            }

            if (b + 1 < F->num_blocks)
                fprintf(out, "\n");
        }
        fprintf(out, "}\n");
    }
}
