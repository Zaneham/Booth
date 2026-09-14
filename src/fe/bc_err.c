/* bc_err.c — Error localisation for Booth
 *
 * Compiled-in English defaults + optional external translation file.
 * Fixed 32KB buffer for translations. No malloc. Bounded loops.
 * If your error message doesn't fit in 32KB, you have bigger problems
 * than localisation. */

#include "bc_err.h"
#include <stdio.h>
#include <string.h>
#include <ctype.h>

/* ---- Compiled-in English defaults ---- */

static const char *bc_dflt[BC_EID_MAX] = {
    /* E000 */ "internal compiler error",
    /* ---- Lexer ---- */
    /* E001 */ "token buffer overflow",
    /* E002 */ "unterminated block comment",
    /* E003 */ "newline in string literal",
    /* E004 */ "unterminated string literal",
    /* E005 */ "unexpected character",
    NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,
    /* ---- Parser ---- */
    /* E020 */ "expected '%s', got '%s'",
    /* E021 */ "AST node limit exceeded",
    /* E022 */ "expected expression",
    /* E023 */ "unexpected token in namespace",
    /* E024 */ "expected declaration",
    /* E025 */ "unexpected token in function body",
    /* E026 */ "unexpected token in block",
    /* E027 */ "unexpected token at top level",
    /* E028 */ "parameter pack must be the last template parameter here",
    /* E029 */ "default argument on a template parameter pack",
    /* E030 */ "unsupported: %s",
    /* E031 */ "pack expansion has no unexpanded parameter pack",
    NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,
    /* ---- Preprocessor ---- */
    /* E040 */ "macro string pool exhausted",
    /* E041 */ "too many macros (max %d)",
    /* E042 */ "#if nesting too deep (max %d)",
    /* E043 */ "#else without matching #if",
    /* E044 */ "#elif without matching #if",
    /* E045 */ "#endif without matching #if",
    /* E046 */ "#define: expected macro name",
    /* E047 */ "#error %s",
    /* E048 */ "#include: expected \"file\" or <file>",
    /* E049 */ "#include: nesting too deep (max %d)",
    /* E050 */ "#include: cannot read '%s'",
    /* E051 */ "unknown directive: #%s",
    /* E052 */ "unterminated #if/#ifdef (missing %d #endif)",
    /* E053 */ "preprocessor output overflow (max %d bytes)",
    NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,
    /* ---- Sema ---- */
    /* E070 */ "arrow on non-pointer",
    /* E071 */ "no field '%s' in struct '%s'",
    /* E072 */ "invalid vector field '%s'",
    /* E073 */ "'%s' expects %d args, got %d",
    /* E074 */ "'%s' expects 1 arg, got %d",
    /* E075 */ "'%s' expects 3 args, got %d",
    /* E076 */ "condition must be scalar type",
    /* E077 */ "for-condition must be scalar type",
    /* E078 */ "while-condition must be scalar type",
    /* E079 */ "do-while condition must be scalar type",
    /* E080 */ "switch expression must be integer type",
    /* E081 */ "__global__ function must return void",
    /* E082 */ "'%s' passes more than %d arguments",
    /* E083 */ "parameter pack '%s' must be expanded",
    NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,
    /* ---- Lowering ---- */
    /* E100 */ "too many labels (max 256)",
    /* E101 */ "undefined variable",
    /* E102 */ "unsupported binary op",
    /* E103 */ "unsupported unary prefix",
    /* E104 */ "unsupported postfix op",
    /* E105 */ "unknown function '%s' in call",
    /* E106 */ "unsupported expression node",
    /* E107 */ "undefined lvalue",
    /* E108 */ "parameter not addressable",
    /* E109 */ "not an lvalue (prefix)",
    /* E110 */ "unknown field in lvalue",
    /* E111 */ "not an lvalue",
    /* ---- Triton frontend (E112-E125): lexer + parser, fixed messages ---- */
    /* E112 */ "indentation too deeply nested",
    /* E113 */ "inconsistent dedent",
    /* E114 */ "unterminated string literal",
    /* E115 */ "unexpected character '!'",
    /* E116 */ "unexpected character",
    /* E117 */ "lexer made no progress (internal bug)",
    /* E118 */ "unterminated triple-quoted string",
    /* E119 */ "too many AST children",
    /* E120 */ "expected identifier",
    /* E121 */ "expected identifier after '.'",
    /* E122 */ "expected parameter name",
    /* E123 */ "expected identifier after 'import'",
    /* E124 */ "expected attribute name after '.'",
    /* E125 */ "unrecognised expression",
    /* E126 */ "'%s' is defined in more than one translation unit",
    /* E127 */ "too many globals to reference (max %d)",
    /* E128 */ "constexpr '%s' has an initialiser Booth cannot evaluate",
    NULL,
    /* E130 */ "too many constexpr declarations (max %d)",
    /* E131 */ "array bound in '%s' is not a constant Booth can fold",
    /* E132 */ "too many template names (max %d)",
    /* E133 */ "'%s' is a class template Booth cannot instantiate",
    /* E134 */ "std::%s is not provided by Booth",
    /* E135 */ "range-based 'for' is not supported",
    /* E136 */ "parenthesised initialiser for '%s' is not supported",
    /* E137 */ "too many type names (max %d)",
    /* E138 */ "too many struct definitions (max %d)",
    /* E139 */ "too many enum constants (max %d)",
    /* E140 */ "too many typedefs (max %d)",
    /* E141 */ "too many function templates (max %d)",
    /* E142 */ "'%s' needs %d-byte alignment that Booth cannot express here",
    /* E143 */ "device intrinsic '%s' is not implemented",
    /* E144 */ "struct '%s' has more members than Booth can record",
    /* E145 */ "function '%s' returns a reference, which Booth cannot lower",
    /* E146 */ "'%s' is an rvalue reference parameter, which Booth cannot lower",
    /* E147 */ "'%s' is a reference parameter of a __global__ kernel",
    /* E148 */ "'%s' binds a non-const reference to something that is not an lvalue",
    /* E149 */ "'%s' binds a non-const reference to a different type",
    /* E150 */ "'%s' is a reference member, which Booth cannot lay out in a struct",
    /* E151 */ "'%s' is a reference in a parameter pack, which Booth cannot lower",
    /* E152 */ "'%s' is a reference declared without an initialiser",
    /* E153 */ "'%s' is a reference to an array, which Booth cannot lower",
    /* E154 */ "'%s' names a reference type, which Booth cannot alias",
    /* E155 */ "unknown type name '%s'",
    /* E156 */ "call to '%s' is ambiguous: Booth cannot choose an overload",
    /* E157 */ "'%s' launches a kernel through a forwarding wrapper Booth cannot expand here",
    /* E158 */ "the condition of 'if constexpr' is not a constant expression: '%s'",
    NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,
    /* E170 */ "%s cannot be lowered to device code",
    NULL,NULL,
    /* E171-E172 */
    /* E173 */ "decltype is not supported",
    /* E174 */ "macro '%s' is given more arguments than Booth can expand (max %d)",
    /* E175 */ "explicit template instantiation is not supported",
    NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,
    NULL,NULL,NULL,NULL,
    /* E176-E219 */
    /* E220 */ "%d is not an alignment: it must be a positive power of two",
    /* E221 */ "inline asm is refused: %s",
    NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,
    /* E240 */ "'%s' sits in an anonymous union in '%s', which Booth's IR has no word for",
    NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,
    /* E260 */ "'%s' is a generic lambda parameter ('auto'), which Booth cannot instantiate",
    /* E261 */ "capture of '%s' is a form Booth cannot lower: %s",
    NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,
    /* E300 */ "macro '%s' declares more parameters than Booth can hold (max %d)",
    NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,
    /* E320 */ "'%s' fits more than one partial specialisation and Booth will not guess",
    NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,
    /* E340 */ "'%s' needs more room than Booth reserves here (max %d)",
    NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,
    /* E360 */ "decltype cannot type '%s'",
    NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,
    /* E400 */ "'%s' has no in-class initialiser, so it has no value here",
    /* E401 */ "'%s' is a mutable static member and has no constant value",
    /* E402 */ "'%s' has a static initialiser Booth cannot fold",
    /* E403 */ "'%s' is reached through a base class Booth cannot name",
    NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,
    /* E420 */ "'%s' expands a pack whose elements Booth cannot type alike",
    /* E421 */ "'%s' is a form of forwarding Booth cannot collapse",
    NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,
    /* E440 */ "no conversion for this cast: %s",
    NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,
    NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,
    /* E500 */ "the Metal backend cannot express '%s'",
    /* E501 */ "the Metal backend cannot name %s",
    /* E502 */ "the Metal backend cannot express %s as a constant",
    /* E503 */ "__shared__ needs a kernel and '%s' is a __device__ function",
    NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,
    /* E520 */ "%s has no Tensix SFPU instruction; atomics live in L1 and belong on the baby cores",
    /* E521 */ "%s has no Tensix SFPU equivalent; the 32 lanes are one vector, not a warp",
    /* E522 */ "the Tensix SFPU cannot divide: %s needs a reciprocal Booth does not synthesise",
    /* E523 */ "the Tensix SFPU has no integer divider, so %s cannot be selected",
    /* E524 */ "__shared__ reaches the Tensix compute core only through a circular buffer, which Booth does not bind",
    /* E525 */ "switch is not lowered on the Tensix SFPU path, where control flow is predication",
    /* E526 */ "a %s in address space %u cannot be reached from the Tensix compute core",
    /* E527 */ "arithmetic right shift has no Tensix SFPU instruction",
    /* E528 */ "bit counting is not supported on the Tensix SFPU path; --rv-elf is where the baby cores are",
    /* E529 */ "__trap has no SFPU instruction; the baby cores are where a fault can be raised",
    /* E530 */ "the address of a device function is not supported on the Tensix backend",
    /* E531 */ "device printf is not supported on the Tensix backend; use the SYSPRINT buffer",
    /* E532 */ "warp-collective mma is not supported on the Tensix backend",
    /* E533 */ "inline asm is bound for PTX only and has no Tensix SFPU equivalent",
    /* E534 */ "BIR op %u has no pattern in the Tensix baby-core isel",
    /* E535 */ "%s at %u has a pointee type with no storage size",
    /* E536 */ "__shared__ needs %u bytes and the baby-core L1 slab holds %u",
    /* E537 */ "the Tensix instruction arena is full at %u instructions and the rest of this kernel would be dropped",
    NULL,NULL,
    /* E540 */ "%s: %s of %s has no storage size",
    /* E541 */ "%s: this backend cannot express %s",
    /* E542 */ "%s: no lowering for BIR op %s",
    /* E543 */ "%s: no lowering for %s variant %u",
    /* E544 */ "%s: the emitter cannot print machine op %u",
    /* E545 */ "%s: %s exceeds the %u this backend holds",
    /* E546 */ "%s: a module-scope global has no name or no initialiser the emitter can lay down",
    /* E547 */ "%s: a %u-byte %s has no single instruction on this backend",
    NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,
    /* E560 */ "%s: %s reaches this backend with no signedness, so neither the signed nor the unsigned PTX form can be chosen",
    /* E561 */ "%s: an atomic in address space %u has no PTX form; atom reaches global, shared and generic only",
    /* E562 */ "%s: a %u-byte atomic has no PTX form; atom is 32-bit or 64-bit",
    NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,
    /* E580 */ "%s: %s needs a jump of %d bytes, past the 1 MiB reach of jal",
    /* E581 */ "%s: a call names function %u, which is not in the module",
    /* E582 */ "%s: a call argument of type %s needs a register pair or a reference, which Booth does not lay down",
    /* E583 */ "%s: a call returning %s needs a register pair or a reference, which Booth does not lay down",
    /* E584 */ "%s: a call carries an operand list the emitter cannot read",
    NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,
    /* E600 */ "%s: a cubin holds one kernel and this module has %u",
    /* E601 */ "%s: %s has no SASS form the backend can encode",
    /* E602 */ "%s: %s needs %u basic blocks and the SASS control-flow pass holds %u",
    /* E603 */ "%s: %s needs more than the %u registers the SASS backend allocates",
    /* E604 */ "%s: %s needs more than the %u predicates the SASS backend allocates",
    /* E605 */ "%s: %s has a divergent region the SASS backend cannot reconverge (check %u at block %u)",
    /* E606 */ "%s: %s has more virtual registers than the %u the SASS backend maps",
    /* E607 */ "%s: %s outgrew the %u byte SASS code buffer",
    /* E608 */ "%s: the cubin for %s did not assemble",
    /* E609 */ "%s: %s takes an operand the SASS backend has no form for",
    /* E610 */ "%s: special register %u has no SASS form",
    /* E611 */ "%s: %s has an address or width the SASS backend has no form for",
    NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,
    /* E640 */ "'%s' needs a PTX form no Booth backend emits",
    NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,
    /* E660 */ "'%s' has more initialisers than it has elements",
    /* E661 */ "Booth cannot lay this initialiser down: %s",
    NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,
    /* E681 */ "'%s' takes template arguments and Booth has no declaration for it, so it cannot be named here",
    NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,
    /* E700 */ "a copy of %s runs past the %d fields or %d levels Booth lays down in one go",
    /* E701 */ "%s has no address, so Booth cannot hand it over by reference",
    /* E702 */ "%s is declared to return %s and a return in it carries %s",
    /* E703 */ "a call to %s reads a result of type %s where %s returns %s",
    /* E704 */ "a call to %s passes %d arguments where it takes %d",
    NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,
    /* E720 */ "'%s' is not a constant Booth can fold as a template argument",
    /* E721 */ "'%s' names a template Booth cannot resolve to a function here",
    NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,
    /* E760 */ "a printf argument of type %s, which varargs cannot carry",
    NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,
    /* E780 */ "inlining leaves %s with %u blocks and Booth carries at most %u in a function",
    /* E781 */ "a call to %s passes more arguments than the %d Booth binds while inlining",
    /* E782 */ "the body of %s runs past the %d instructions Booth maps for one splice",
    NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,
    /* E800 */ "linkage specification %s is not one Booth implements",
    /* E801 */ "unexpected token in linkage specification",
    /* E802 */ "'struct %s' is declared but never defined, so it has no field to reach",
    /* E803 */ "more struct definitions than Booth records (max %d)",
    NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,
    /* E820 */ "'%s' has more fields than Booth folds as a constexpr struct",
    /* E821 */ "Booth ran out of room folding the constexpr struct '%s'",
    NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,
    /* E840 */ "in %s, %s operand %u wants a %s register but got a %s one",
    /* E841 */ "in %s, PTX machine op %u has no verifier signature",
    /* E842 */ "in %s, the return value is %s but the function returns %s",
    /* E843 */ "in %s, a call result is %s but the callee returns %s",
    /* E844 */ "in %s, a value in a %s register cannot be converted to %s",
    /* E845 */ "in %s, %s operand %u must be a register, not a literal",
    NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,
    /* E860 */ "a %s in %s names a block that belongs to %s",
    /* E861 */ "an operand in %s names a value defined in %s",
    /* E862 */ "%s and %s claim the same blocks",
    /* E863 */ "%s and %s claim the same instructions",
    /* E864 */ "a phi in block %s of %s names a predecessor that does not branch there",
    /* E865 */ "a phi in block %s of %s names %d predecessors where %d branch there",
    /* E866 */ "%s has no Tensix SFPU form; the vector unit reaches LReg and Dst, never memory another core can observe",
    /* E867 */ "%s has no pattern in the Tensix SFPU isel",
    NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,
    /* E900 */ "a WMMA fragment dimension %s that is not a plain positive number",
    /* E901 */ "a WMMA fragment element type %s the PTX ISA does not list",
    /* E902 */ "a WMMA fragment use %s that is not matrix_a, matrix_b or accumulator",
    /* E903 */ "a WMMA fragment layout %s that is not row_major or col_major",
    /* E904 */ "a WMMA fragment shape and element type %s the PTX ISA does not pair",
    /* E905 */ "wmma::%s given a fragment argument Booth cannot name",
    /* E906 */ "wmma::%s given the wrong number of arguments",
    /* E907 */ "a WMMA multiply over %s that the PTX ISA does not list",
    /* E908 */ "a WMMA accumulator layout %s that is not mem_row_major or mem_col_major",
    /* E909 */ "a %s WMMA fragment needs matrix_a row-major and matrix_b column-major",
    /* E910 */ "a WMMA fragment %s reached through a member Booth does not expose",
    /* E911 */ "wmma::bmma_sync with a bit operation %s Booth does not know",
    /* E912 */ "a WMMA accumulator fragment given a layout %s it does not take",
    NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,
    /* E940 */ "%s given an operand that is not a half2 or bfloat162",
    NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,
    /* E960 */ "Booth cannot tell which specialisation '%s' names here",
    NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,
    /* E980 */ "%s names global %u, and the module holds %u",
    NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,
    NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,
    /* E1060 */ "%s: the byte-array global '%s' lives in the %s address space, which this backend cannot lay down in .rodata",
    /* E1061 */ "%s: the byte-array global '%s' is writable, and this backend only lays read-only data in .rodata",
    /* E1062 */ "%s: the byte initialiser of '%s' names a slice outside the module string table",
};

/* ---- ABEND compiled-in defaults ----
 * Keyed by hex code (0x001..0x0FF). Sparse, most slots NULL. */

static const char *ab_dflt[AB_AID_MAX];
static int ab_dflt_init;

static void ab_dset(void)
{
    if (ab_dflt_init) return;
    ab_dflt_init = 1;
    ab_dflt[0x0C1] = "illegal GPU instruction";
    ab_dflt[0x0C4] = "memory access violation (page not mapped)";
    ab_dflt[0x0C5] = "addressing exception (unmapped address)";
    ab_dflt[0x0C7] = "data exception (alignment or arithmetic fault)";
    ab_dflt[0x0CB] = "GPU hardware error (machine check)";
    ab_dflt[0x001] = "kernel dispatch failure";
    ab_dflt[0x002] = "kernel timeout exceeded";
    ab_dflt[0x003] = "GPU memory exhaustion";
    ab_dflt[0x0FF] = "unknown GPU fault";
}

/* ---- Translation overlay ---- */

#define XLAT_BUF_SZ  32768
#define XLAT_MAX_LN  512

static char        xlat_buf[XLAT_BUF_SZ];
static const char *bc_xlat[BC_EID_MAX];
static const char *ab_xlat[AB_AID_MAX];

/* ---- Lookup ---- */

const char *bc_efmt(bc_eid_t eid)
{
    int id = (int)eid;
    if (id < 0 || id >= BC_EID_MAX) return "unknown error";
    if (bc_xlat[id]) return bc_xlat[id];
    if (bc_dflt[id]) return bc_dflt[id];
    return "unknown error";
}

const char *ab_afmt(uint16_t code)
{
    ab_dset();
    if (code >= AB_AID_MAX) return "unknown GPU fault";
    if (ab_xlat[code]) return ab_xlat[code];
    if (ab_dflt[code]) return ab_dflt[code];
    return "unknown GPU fault";
}

/* ---- Translation file loader ----
 * Format: "ENNN=message text" per line. # comments. Blank lines ignored.
 * Bounded: max XLAT_MAX_LN lines, XLAT_BUF_SZ total bytes. */

int bc_eload(const char *path)
{
    if (!path) return 0;

    FILE *fp = fopen(path, "r");
    if (!fp) {
        fprintf(stderr, "warning: cannot open lang file '%s'\n", path);
        return -1;
    }

    char line[1024];
    int  nlines = 0;
    unsigned wpos = 0;

    memset(bc_xlat, 0, sizeof(bc_xlat));
    memset(ab_xlat, 0, sizeof(ab_xlat));
    ab_dset();

    while (fgets(line, (int)sizeof(line), fp) && nlines < XLAT_MAX_LN) {
        nlines++;

        /* Strip trailing newline */
        size_t ln = strlen(line);
        while (ln > 0 && (line[ln-1] == '\n' || line[ln-1] == '\r'))
            line[--ln] = '\0';

        /* Skip blank lines and comments */
        if (ln == 0 || line[0] == '#') continue;

        if (line[0] == 'A') {
            /* Parse ANNN=text (hex ID, e.g. A0C4=...) */
            int aid = 0, i = 1, ndig = 0;
            for (; i < 4 && ndig < 3; i++) {
                char c = line[i];
                int v = -1;
                if (c >= '0' && c <= '9') v = c - '0';
                else if (c >= 'A' && c <= 'F') v = c - 'A' + 10;
                else if (c >= 'a' && c <= 'f') v = c - 'a' + 10;
                else break;
                aid = aid * 16 + v;
                ndig++;
            }
            if (ndig == 0 || line[i] != '=' || aid >= AB_AID_MAX) continue;
            i++;

            size_t mlen = ln - (size_t)i;
            if (wpos + mlen + 1 > XLAT_BUF_SZ) break;
            memcpy(xlat_buf + wpos, line + i, mlen);
            xlat_buf[wpos + mlen] = '\0';
            ab_xlat[aid] = xlat_buf + wpos;
            wpos += (unsigned)(mlen + 1);
            continue;
        }

        /* Parse ENNN=text */
        if (line[0] != 'E') continue;
        int eid = 0;
        int i = 1;
        int ndig = 0;
        while (i < 4 && isdigit((unsigned char)line[i])) {
            eid = eid * 10 + (line[i] - '0');
            i++;
            ndig++;
        }
        if (ndig == 0 || line[i] != '=' || eid >= BC_EID_MAX) continue;
        i++; /* skip '=' */

        /* Copy message text into xlat_buf */
        size_t mlen = ln - (size_t)i;
        if (wpos + mlen + 1 > XLAT_BUF_SZ) break; /* buffer full */

        memcpy(xlat_buf + wpos, line + i, mlen);
        xlat_buf[wpos + mlen] = '\0';
        bc_xlat[eid] = xlat_buf + wpos;
        wpos += (unsigned)(mlen + 1);
    }

    fclose(fp);
    return 0;
}
