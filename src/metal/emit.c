#include "metal.h"
#include "barracuda.h"

#include <stdio.h>
#include <string.h>
#include <stdarg.h>

/* Apple Metal backend: BIR to Metal Shading Language text.
 *
 * This file is the staking of a flag, in the Eddie Izzard sense, on the
 * particular patch of the source tree where Apple GPU support is going
 * to live. The point of the flag is not that we have done the work, the
 * point is that the work belongs here now and that any subsequent claim
 * to this territory needs to come and have a word with whoever was here
 * first, namely the people writing comments like the one you are
 * currently reading.
 *
 * The plan, when fully populated, is to lower BIR into MSL text and to
 * hand that text to Apple's own Metal compiler at load time, much as we
 * hand PTX text to the NVIDIA driver while trying to look casual about
 * how little of the work belongs to us. The first sitting, of which this
 * file is the product, gets the kernel signature and the surrounding
 * scaffolding correct, so that an MSL compiler asked politely will at
 * least agree that the document it is reading is in fact MSL, even if
 * the function body inside is at present a comment confessing that the
 * body lowering has not been written yet.
 *
 * The CUDA to MSL mappings the emitter cares about, expressed as a
 * vocabulary list rather than as code we have actually written:
 *   threadIdx        -> thread_position_in_threadgroup (uint3 attribute)
 *   blockIdx         -> threadgroup_position_in_grid   (uint3 attribute)
 *   blockDim         -> threads_per_threadgroup        (uint3 attribute)
 *   gridDim          -> threadgroups_per_grid          (uint3 attribute)
 *   __syncthreads    -> threadgroup_barrier(mem_flags::mem_threadgroup)
 *   __shared__       -> threadgroup storage qualifier
 *   device global    -> device storage qualifier
 *   __constant__     -> constant storage qualifier
 * The rest of the translation table arrives in subsequent sittings as
 * we extend the emitter from signature only to full function body. */

/* ---- Output Buffer Helpers ----
 * mt_wput appends a single byte, mt_wstr appends a C string, mt_wfmt
 * appends a formatted string. All three respect the buffer bound and
 * return zero on overflow rather than risk an off-by-one into the next
 * struct member, which would be a less than dignified way for an open
 * source compiler to end its day. */

static int mt_wput(metal_module_t *mm, char c)
{
    if (mm->out_len + 1 >= MTL_MAX_OUT) return 0;
    mm->out_buf[mm->out_len++] = c;
    return 1;
}

static int mt_wstr(metal_module_t *mm, const char *s)
{
    while (*s) {
        if (!mt_wput(mm, *s++)) return 0;
    }
    return 1;
}

static int mt_wfmt(metal_module_t *mm, const char *fmt, ...)
{
    char tmp[512];
    va_list ap;
    va_start(ap, fmt);
    int n = vsnprintf(tmp, sizeof(tmp), fmt, ap);
    va_end(ap);
    if (n < 0) return 0;
    if ((size_t)n >= sizeof(tmp)) n = (int)sizeof(tmp) - 1;
    tmp[n] = '\0';
    return mt_wstr(mm, tmp);
}

/* Resolve a string-table offset to a null-terminated string, or return
 * a stable fallback when the offset is out of range or points at the
 * empty string. Offset zero is a legal name in BIR's string table,
 * because bir_add_string hands out offsets starting from string_len
 * and the first string added gets offset zero, so the test that
 * actually matters is whether the byte at the offset is non-null. */

static const char *mt_name(const bir_module_t *M, uint32_t off, const char *fb)
{
    if (off >= M->string_len) return fb;
    if (M->strings[off] == '\0') return fb;
    return &M->strings[off];
}

/* Emit the MSL address-space qualifier that goes in front of a pointer
 * parameter type. MSL refuses to accept pointers with no address space
 * declared, on the entirely reasonable grounds that it cannot tell
 * where in the GPU's many addressable regions the pointee is meant to
 * live without being told. */

static const char *mt_addr(uint8_t as)
{
    switch (as) {
    case BIR_AS_GLOBAL:    return "device";
    case BIR_AS_SHARED:    return "threadgroup";
    case BIR_AS_CONSTANT:  return "constant";
    case BIR_AS_PRIVATE:   return "thread";
    case BIR_AS_GENERIC:   /* MSL has no generic, fall through to device */
    default:               return "device";
    }
}

/* ---- Type Emission ----
 * Translate a BIR type into the closest MSL type name. The MSL primitive
 * set is small enough that this is mostly a switch statement, plus the
 * trick that bool maps to bool not int1 because the MSL specification
 * happens to spell it the same way C does. */

static int mt_etype(metal_module_t *mm, uint32_t ti);

static int mt_etype(metal_module_t *mm, uint32_t ti)
{
    const bir_type_t *T = &mm->bir->types[ti];
    switch (T->kind) {
    case BIR_TYPE_VOID:
        return mt_wstr(mm, "void");
    case BIR_TYPE_INT:
        switch (T->width) {
        case 1:  return mt_wstr(mm, "bool");
        case 8:  return mt_wstr(mm, "char");
        case 16: return mt_wstr(mm, "short");
        case 32: return mt_wstr(mm, "int");
        case 64: return mt_wstr(mm, "long");
        default: return mt_wfmt(mm, "int%u", (unsigned)T->width);
        }
    case BIR_TYPE_FLOAT:
        switch (T->width) {
        case 16: return mt_wstr(mm, "half");
        case 32: return mt_wstr(mm, "float");
        case 64: /* MSL does not have double on most Apple GPUs */
                 return mt_wstr(mm, "float");
        default: return mt_wstr(mm, "float");
        }
    case BIR_TYPE_BFLOAT:
        /* bf16 is Metal 3.1+ only, which is M3 and later. For earlier
         * targets the caller will eventually need to widen to float. */
        return mt_wstr(mm, "bfloat");
    case BIR_TYPE_PTR:
        if (!mt_wstr(mm, mt_addr(T->addrspace))) return 0;
        if (!mt_wput(mm, ' ')) return 0;
        if (!mt_etype(mm, T->inner)) return 0;
        return mt_wstr(mm, " *");
    case BIR_TYPE_VECTOR: {
        /* MSL spells vectors as type-name followed by lane count: float2,
         * float3, float4, int2 and so on. Anything other than those four
         * sizes is not legal MSL and will need a different lowering. */
        const bir_type_t *EL = &mm->bir->types[T->inner];
        const char *base = "int";
        if (EL->kind == BIR_TYPE_FLOAT) base = (EL->width == 16) ? "half" : "float";
        else if (EL->kind == BIR_TYPE_INT && EL->width == 8) base = "char";
        else if (EL->kind == BIR_TYPE_INT && EL->width == 16) base = "short";
        else if (EL->kind == BIR_TYPE_INT && EL->width == 64) base = "long";
        return mt_wfmt(mm, "%s%u", base, (unsigned)T->width);
    }
    case BIR_TYPE_STRUCT:
    case BIR_TYPE_ARRAY:
    case BIR_TYPE_FUNC:
    case BIR_TYPE_KIND_COUNT:
    default:
        /* Aggregate types and anything we have not yet taught MSL about
         * fall back to a placeholder so the output stays parseable while
         * the proper translation waits its turn in a later sitting. */
        return mt_wstr(mm, "/* TODO: aggregate type */ int");
    }
}

/* ---- Kernel Signature ----
 * Write the kernel header, including each parameter with its address
 * space and binding index, followed by the synthesised builtin
 * parameters carrying their MSL attributes. The order is significant:
 * MSL allows ordinary parameters and attributed builtins to be mixed,
 * but a single uniform convention of "real parameters first, builtins
 * after" keeps the output readable. */

static int mt_ksig(metal_module_t *mm, const mtl_kern_t *K)
{
    const bir_module_t *M = mm->bir;
    const char *kw = K->is_kern ? "kernel " : "";
    const bir_func_t *F = &M->funcs[K->bir_func];
    const bir_type_t *FT = &M->types[F->type];

    if (!mt_wstr(mm, kw)) return 0;
    if (!mt_etype(mm, FT->inner)) return 0;
    if (!mt_wfmt(mm, " %s(", mt_name(M, K->name, "unnamed"))) return 0;

    if (K->num_params == 0 && K->builtins == 0) {
        return mt_wstr(mm, "void)");
    }

    /* Real parameters first. For kernels, MSL is particular about how
     * arguments arrive. Pointer parameters get their address-space
     * qualifier followed by the pointee type. Scalar parameters get
     * passed by const reference out of the constant address space,
     * because MSL kernel arguments are not registers and a bare scalar
     * with a [[buffer(N)]] binding is rejected as nonsense by the
     * Metal compiler. For __device__ functions none of this applies
     * and we emit plain C++-style parameters. */
    int first = 1;
    for (uint16_t pi = 0; pi < K->num_params; pi++) {
        const mtl_param_t *P = &K->params[pi];
        const bir_type_t  *PT = &M->types[P->type];
        if (!first && !mt_wstr(mm, ",\n    ")) return 0;
        if (first && !mt_wstr(mm, "\n    ")) return 0;
        first = 0;

        if (K->is_kern && PT->kind != BIR_TYPE_PTR) {
            /* Scalar kernel argument, must arrive as constant T &p. */
            if (!mt_wstr(mm, "constant ")) return 0;
            if (!mt_etype(mm, P->type)) return 0;
            if (!mt_wfmt(mm, " &p%u [[buffer(%u)]]",
                         (unsigned)pi, (unsigned)pi)) return 0;
        } else if (K->is_kern) {
            /* Pointer kernel argument, mt_etype already emits the
             * address-space qualifier as part of the pointer type. */
            if (!mt_etype(mm, P->type)) return 0;
            if (!mt_wfmt(mm, " p%u [[buffer(%u)]]",
                         (unsigned)pi, (unsigned)pi)) return 0;
        } else {
            /* __device__ function, plain parameter, no binding index. */
            if (!mt_etype(mm, P->type)) return 0;
            if (!mt_wfmt(mm, " p%u", (unsigned)pi)) return 0;
        }
    }

    /* Synthetic builtin parameters, attached to the slots that come
     * after the buffer bindings. The MSL compiler does not number these
     * the way it numbers buffers, it identifies them entirely by
     * attribute, which means we are at liberty to add as many of them
     * as the kernel will actually reference. */
    if (K->is_kern) {
        if (K->builtins & MTL_BI_TID) {
            if (!first && !mt_wstr(mm, ",\n    ")) return 0;
            first = 0;
            if (!mt_wstr(mm, "uint3 tid [[thread_position_in_threadgroup]]")) return 0;
        }
        if (K->builtins & MTL_BI_BID) {
            if (!first && !mt_wstr(mm, ",\n    ")) return 0;
            first = 0;
            if (!mt_wstr(mm, "uint3 bid [[threadgroup_position_in_grid]]")) return 0;
        }
        if (K->builtins & MTL_BI_BDIM) {
            if (!first && !mt_wstr(mm, ",\n    ")) return 0;
            first = 0;
            if (!mt_wstr(mm, "uint3 bdim [[threads_per_threadgroup]]")) return 0;
        }
        if (K->builtins & MTL_BI_GDIM) {
            if (!first && !mt_wstr(mm, ",\n    ")) return 0;
            first = 0;
            if (!mt_wstr(mm, "uint3 gdim [[threadgroups_per_grid]]")) return 0;
        }
    }

    return mt_wstr(mm, ")");
}

/* Placeholder body. Subsequent sittings replace this with an actual
 * statement walker that translates each BIR instruction into its MSL
 * equivalent. For now we emit the comment a confessor would expect to
 * find when looking inside a kernel that has been declared but not yet
 * lowered, followed by a default return value if the function expects
 * one, because MSL refuses to compile a non-void function whose body
 * does not eventually produce a value to return. The C++ value-init
 * syntax (the empty braces) covers scalars, vectors, and structs in
 * one stroke and is legal MSL on every version we care about. */

static int mt_kbody(metal_module_t *mm, const mtl_kern_t *K)
{
    const bir_module_t *M = mm->bir;
    const bir_func_t *F = &M->funcs[K->bir_func];
    const bir_type_t *FT = &M->types[F->type];
    const bir_type_t *RT = &M->types[FT->inner];

    if (!mt_wstr(mm, "\n{\n")) return 0;
    if (!mt_wstr(mm, "    /* TODO: BIR-to-MSL instruction lowering "
                     "lives in a future sitting. */\n")) return 0;

    if (RT->kind != BIR_TYPE_VOID) {
        if (!mt_wstr(mm, "    return ")) return 0;
        if (RT->kind == BIR_TYPE_PTR) {
            if (!mt_wstr(mm, "nullptr")) return 0;
        } else {
            /* Value-initialise whatever the return type happens to be. */
            if (!mt_wstr(mm, "{}")) return 0;
        }
        if (!mt_wstr(mm, ";\n")) return 0;
    }

    return mt_wstr(mm, "}\n\n");
}

/* ---- Builtin Scan ----
 * Walk every instruction in every block of a function and set the bit
 * in K->builtins for each thread-model opcode we see. This tells the
 * signature emitter which attributed parameters to synthesise. */

static void mt_scan(metal_module_t *mm, mtl_kern_t *K)
{
    const bir_module_t *M = mm->bir;
    const bir_func_t *F = &M->funcs[K->bir_func];
    uint32_t bg = 0;
    for (uint16_t bi = 0; bi < F->num_blocks && bg < 65536; bi++, bg++) {
        uint32_t bidx = F->first_block + bi;
        if (bidx >= M->num_blocks) break;
        const bir_block_t *B = &M->blocks[bidx];
        for (uint32_t ii = 0; ii < B->num_insts; ii++) {
            const bir_inst_t *I = &M->insts[B->first_inst + ii];
            switch (I->op) {
            case BIR_THREAD_ID:  K->builtins |= MTL_BI_TID;  break;
            case BIR_BLOCK_ID:   K->builtins |= MTL_BI_BID;  break;
            case BIR_BLOCK_DIM:  K->builtins |= MTL_BI_BDIM; break;
            case BIR_GRID_DIM:   K->builtins |= MTL_BI_GDIM; break;
            default: break;
            }
        }
    }
}

/* ---- Parameter Extraction ----
 * Walk the entry block looking for BIR_PARAM instructions, in the same
 * pattern the NVIDIA backend uses, and copy each one's type into the
 * kernel descriptor. For pointer types we additionally lift the BIR
 * address space onto the parameter so the emitter can place the
 * correct MSL qualifier in front of it. */

static void mt_params(metal_module_t *mm, mtl_kern_t *K)
{
    const bir_module_t *M = mm->bir;
    const bir_func_t *F = &M->funcs[K->bir_func];
    if (F->first_block >= M->num_blocks) {
        K->num_params = 0;
        return;
    }
    const bir_block_t *entry = &M->blocks[F->first_block];
    uint16_t pi = 0;
    uint32_t pg = 64;
    for (uint32_t ii = 0; ii < entry->num_insts && pg > 0; ii++, pg--) {
        const bir_inst_t *I = &M->insts[entry->first_inst + ii];
        if (I->op != BIR_PARAM) continue;
        if (pi >= MTL_MAX_PARAMS) break;
        K->params[pi].name = 0;
        K->params[pi].type = I->type;
        const bir_type_t *PT = &M->types[I->type];
        if (PT->kind == BIR_TYPE_PTR) {
            K->params[pi].addrspace = PT->addrspace;
        } else {
            K->params[pi].addrspace = BIR_AS_PRIVATE;
        }
        K->params[pi].is_const = 0;
        pi++;
    }
    K->num_params = pi;
}

/* ---- metal_compile ----
 * Walk every function in the module, pick out the ones the user has
 * marked __global__ or __device__, and populate a mtl_kern_t for each.
 * Everything else, including host functions, is none of our business
 * and is skipped without comment. */

int metal_compile(const bir_module_t *bir, metal_module_t *mm)
{
    memset(mm, 0, sizeof(*mm));
    mm->bir = bir;

    for (uint32_t fi = 0; fi < bir->num_funcs; fi++) {
        const bir_func_t *F = &bir->funcs[fi];
        if (!(F->cuda_flags & (CUDA_GLOBAL | CUDA_DEVICE))) continue;
        if (mm->num_kerns >= MTL_MAX_KERNS) return BC_ERR_METAL;

        mtl_kern_t *K = &mm->kerns[mm->num_kerns++];
        K->name              = F->name;
        K->bir_func          = fi;
        K->is_kern           = (F->cuda_flags & CUDA_GLOBAL) ? 1 : 0;
        K->num_params        = 0;
        K->builtins          = 0;
        K->launch_bounds_max = F->launch_bounds_max;
        K->launch_bounds_min = F->launch_bounds_min;
        memset(K->params, 0, sizeof(K->params));

        mt_params(mm, K);
        mt_scan(mm, K);
    }

    return BC_OK;
}

/* ---- metal_emit_msl ----
 * Lay down the MSL header, then for each kernel write its signature and
 * a placeholder body. The body content is intentionally minimal in this
 * first sitting, the point being to produce something the Metal
 * compiler will accept as syntactically valid while we go off and
 * write the actual instruction lowering. */

int metal_emit_msl(metal_module_t *mm, const char *path)
{
    mm->out_len = 0;

    if (!mt_wstr(mm,
        "/* Generated by BarraCUDA: Apple Metal backend.\n"
        " * Hand this MSL to Apple's Metal compiler via xcrun metal,\n"
        " * load the resulting .metallib at runtime, and dispatch.\n"
        " */\n\n"
        "#include <metal_stdlib>\n"
        "using namespace metal;\n\n")) return BC_ERR_METAL;

    for (uint32_t ki = 0; ki < mm->num_kerns; ki++) {
        const mtl_kern_t *K = &mm->kerns[ki];
        if (!mt_ksig(mm, K)) return BC_ERR_METAL;
        if (!mt_kbody(mm, K)) return BC_ERR_METAL;
    }

    FILE *fp = fopen(path, "wb");
    if (!fp) {
        fprintf(stderr, "metal: cannot open '%s' for writing\n", path);
        return BC_ERR_IO;
    }
    size_t w = fwrite(mm->out_buf, 1, mm->out_len, fp);
    fclose(fp);
    if (w != mm->out_len) {
        fprintf(stderr, "metal: short write to '%s'\n", path);
        return BC_ERR_IO;
    }

    fprintf(stderr, "wrote %s (%u bytes, %u kernels)\n",
            path, mm->out_len, mm->num_kerns);
    return BC_OK;
}
