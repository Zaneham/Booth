/* bc_err.h — Error localisation for Booth
 *
 * Every diagnostic gets a numeric ID (E001..E158). English is compiled in;
 * external translation files loaded via --lang override individual messages.
 * A Japanese dev can Google "Booth E020" regardless of locale.
 * No malloc. No nonsense. */

#ifndef BARRACUDA_BC_ERR_H
#define BARRACUDA_BC_ERR_H

#include <stdint.h>

/* ---- Error ID Ranges ----
 * E001-E019  Lexer
 * E020-E039  Parser
 * E040-E069  Preprocessor
 * E070-E099  Sema
 * E100-E158  Lowering */

#define BC_EID_MAX 1000







typedef enum {
    BC_E000 = 0,   /* internal compiler error — the "this shouldn't happen" */

    /* ---- Lexer (E001-E019) ---- */
    BC_E001 = 1,   /* token buffer overflow */
    BC_E002 = 2,   /* unterminated block comment */
    BC_E003 = 3,   /* newline in string literal */
    BC_E004 = 4,   /* unterminated string literal */
    BC_E005 = 5,   /* unexpected character */

    /* ---- Parser (E020-E039) ---- */
    BC_E020 = 20,  /* expected '%s', got '%s' */
    BC_E021 = 21,  /* AST node limit exceeded */
    BC_E022 = 22,  /* expected expression */
    BC_E023 = 23,  /* unexpected token in namespace */
    BC_E024 = 24,  /* expected declaration */
    BC_E025 = 25,  /* unexpected token in function body */
    BC_E026 = 26,  /* unexpected token in block */
    BC_E027 = 27,  /* unexpected token at top level */
    BC_E028 = 28,  /* parameter pack must be the last template parameter here */
    BC_E029 = 29,  /* default argument on a template parameter pack */
    BC_E030 = 30,  /* unsupported: %s -- also raised by sema and lowering */
    BC_E031 = 31,  /* pack expansion has no unexpanded parameter pack */

    /* ---- Preprocessor (E040-E069) ---- */
    BC_E040 = 40,  /* macro string pool exhausted */
    BC_E041 = 41,  /* too many macros (max %d) */
    BC_E042 = 42,  /* #if nesting too deep (max %d) */
    BC_E043 = 43,  /* #else without matching #if */
    BC_E044 = 44,  /* #elif without matching #if */
    BC_E045 = 45,  /* #endif without matching #if */
    BC_E046 = 46,  /* #define: expected macro name */
    BC_E047 = 47,  /* #error %s */
    BC_E048 = 48,  /* #include: expected "file" or <file> */
    BC_E049 = 49,  /* #include: nesting too deep (max %d) */
    BC_E050 = 50,  /* #include: cannot read '%s' */
    BC_E051 = 51,  /* unknown directive: #%s */
    BC_E052 = 52,  /* unterminated #if/#ifdef (missing %d #endif) */
    BC_E053 = 53,  /* preprocessor output overflow (max %d bytes) */

    /* ---- Sema (E070-E099) ---- */
    BC_E070 = 70,  /* arrow on non-pointer */
    BC_E071 = 71,  /* no field '%s' in struct '%s' */
    BC_E072 = 72,  /* invalid vector field '%s' */
    BC_E073 = 73,  /* '%s' expects %d args, got %d */
    BC_E074 = 74,  /* '%s' expects 1 arg, got %d */
    BC_E075 = 75,  /* '%s' expects 3 args, got %d */
    BC_E076 = 76,  /* condition must be scalar type */
    BC_E077 = 77,  /* for-condition must be scalar type */
    BC_E078 = 78,  /* while-condition must be scalar type */
    BC_E079 = 79,  /* do-while condition must be scalar type */
    BC_E080 = 80,  /* switch expression must be integer type */
    BC_E081 = 81,  /* __global__ function must return void */
    BC_E082 = 82,  /* '%s' passes more than %d arguments */
    BC_E083 = 83,  /* parameter pack '%s' must be expanded */

    /* ---- Lowering (E100-E158) ---- */
    BC_E100 = 100, /* too many labels (max 256) */
    BC_E101 = 101, /* undefined variable */
    BC_E102 = 102, /* unsupported binary op */
    BC_E103 = 103, /* unsupported unary prefix */
    BC_E104 = 104, /* unsupported postfix op */
    BC_E105 = 105, /* unknown function '%s' in call */
    BC_E106 = 106, /* unsupported expression node */
    BC_E107 = 107, /* undefined lvalue */
    BC_E108 = 108, /* parameter not addressable */
    BC_E109 = 109, /* not an lvalue (prefix) */
    BC_E110 = 110, /* unknown field in lvalue */
    BC_E111 = 111, /* not an lvalue */

    BC_E112 = 112, /* indentation too deeply nested */
    BC_E113 = 113, /* inconsistent dedent */
    BC_E114 = 114, /* unterminated string literal (triton) */
    BC_E115 = 115, /* unexpected character '!' */
    BC_E116 = 116, /* unexpected character (triton) */
    BC_E117 = 117, /* lexer made no progress */
    BC_E118 = 118, /* unterminated triple-quoted string */
    BC_E119 = 119, /* too many AST children */
    BC_E120 = 120, /* expected identifier */
    BC_E121 = 121, /* expected identifier after '.' */
    BC_E122 = 122, /* expected parameter name */
    BC_E123 = 123, /* expected identifier after 'import' */
    BC_E124 = 124, /* expected attribute name after '.' */
    BC_E125 = 125, /* unrecognised expression */

    /* ---- Linking translation units (E126-E127) ---- */
    BC_E126 = 126, /* symbol defined in more than one translation unit */
    BC_E127 = 127, /* module holds more globals than Booth can carry */

    /* ---- Constant expressions (E128-E131) ---- */
    BC_E128 = 128, /* constexpr initialiser is not a constant expression */
    BC_E129 = 129, /* retired: if constexpr now lowers */
    BC_E130 = 130, /* too many constexpr declarations */
    BC_E131 = 131, /* array bound is not a constant expression */

    BC_E132 = 132, /* too many template names */
    BC_E133 = 133, /* class template Booth cannot instantiate */
    BC_E134 = 134, /* std facility Booth does not provide */
    BC_E135 = 135, /* range-based for is not supported */
    BC_E136 = 136, /* parenthesised initialiser is not supported */
    BC_E137 = 137, /* too many type names */

    BC_E138 = 138, /* too many struct definitions */
    BC_E139 = 139, /* too many enum constants */
    BC_E140 = 140, /* too many typedefs */
    BC_E141 = 141, /* too many function templates */

    BC_E142 = 142, /* vector member needs an alignment BIR cannot carry */
    BC_E143 = 143, /* device intrinsic not implemented */
    BC_E144 = 144, /* struct has more members than Booth can record */

    BC_E145 = 145, /* function returns a reference */
    BC_E146 = 146, /* rvalue reference parameter */
    BC_E147 = 147, /* reference parameter of a __global__ kernel */
    BC_E148 = 148, /* non-const reference bound to a non-lvalue */
    BC_E149 = 149, /* non-const reference bound to a different type */
    BC_E150 = 150, /* reference member in a struct */
    BC_E151 = 151, /* reference in a parameter pack */
    BC_E152 = 152, /* reference declared without an initialiser */
    BC_E153 = 153, /* reference to an array */
    BC_E154 = 154, /* typedef naming a reference type */
    BC_E155 = 155, /* unknown type name */
    BC_E156 = 156, /* call is ambiguous between overloads */
    BC_E157 = 157, /* kernel launched through a wrapper we cannot expand */
    BC_E158 = 158, /* if constexpr condition is not constant */

    BC_E170 = 170, /* host-only expression reached device code */
    BC_E173 = 173, /* decltype */
    BC_E174 = 174, /* macro argument list longer than the expander holds */
    BC_E175 = 175, /* explicit template instantiation */

    BC_E220 = 220, /* alignment is not a positive power of two */
    BC_E221 = 221, /* inline asm operand Booth cannot bind */
    BC_E240 = 240, /* member of an anonymous union */

    BC_E260 = 260, /* generic lambda parameter */
    BC_E261 = 261, /* lambda capture Booth cannot lower */
    BC_E300 = 300, /* macro definition has more parameters than we hold */
    BC_E320 = 320, /* ambiguous class template specialisation */
    BC_E340 = 340, /* construct needs more room than Booth reserves */
    BC_E360 = 360, /* decltype operand has no type Booth can name */

    BC_E400 = 400, /* static data member with no in-class initialiser */
    BC_E401 = 401, /* static data member that is not const */
    BC_E402 = 402, /* static data member initialiser will not fold */
    BC_E403 = 403, /* lookup crosses a base class Booth cannot name */

    BC_E420 = 420, /* pack elements in an argument list do not type alike */
    BC_E421 = 421, /* std::forward shape Booth cannot collapse */

    BC_E440 = 440, /* cast whose conversion does not exist */

    /* ---- Metal backend (E500-E519) ---- */
    BC_E500 = 500, /* the Metal backend cannot express '%s' */
    BC_E501 = 501, /* the Metal backend cannot name %s */
    BC_E502 = 502, /* the Metal backend cannot express %s as a constant */
    BC_E503 = 503, /* __shared__ needs a kernel and '%s' is a __device__ function */

    BC_E520 = 520, /* atomic has no SFPU instruction */
    BC_E521 = 521, /* warp-collective has no SFPU instruction */
    BC_E522 = 522, /* floating-point division on the SFPU */
    BC_E523 = 523, /* integer division or remainder on the SFPU */
    BC_E524 = 524, /* __shared__ on the Tensix compute path */
    BC_E525 = 525, /* switch on the Tensix SFPU path */
    BC_E526 = 526, /* address space the compute core cannot reach */
    BC_E527 = 527, /* arithmetic right shift on the SFPU */
    BC_E528 = 528, /* bit counting on the SFPU */
    BC_E529 = 529, /* __trap on the SFPU */
    BC_E530 = 530, /* address of a device function on Tensix */
    BC_E531 = 531, /* device printf on Tensix */
    BC_E532 = 532, /* warp-collective mma on Tensix */
    BC_E533 = 533, /* inline asm on Tensix */
    BC_E534 = 534, /* BIR op the baby-core isel has no pattern for */
    BC_E535 = 535, /* Tensix allocation whose type has no storage size */
    BC_E536 = 536, /* __shared__ larger than the baby-core L1 slab */
    BC_E537 = 537, /* Tensix instruction arena full */

    BC_E540 = 540, /* backend: operand type has no storage size */
    BC_E541 = 541, /* backend cannot express this operation */
    BC_E542 = 542, /* backend has no lowering for a BIR op */
    BC_E543 = 543, /* backend has no lowering for a variant of one */
    BC_E544 = 544, /* emitter met a machine op it cannot print */
    BC_E545 = 545, /* a fixed capacity in the backend is exceeded */
    BC_E546 = 546, /* module-scope global the emitter cannot lay down */
    BC_E547 = 547, /* access width with no single instruction */

    BC_E560 = 560, /* atomic min/max with no signedness in the IR */
    BC_E561 = 561, /* atomic in an address space PTX atom cannot reach */
    BC_E562 = 562, /* atomic of a width PTX atom has no form for */

    BC_E580 = 580, /* a jump is past the reach of jal */
    BC_E581 = 581, /* a call names a function not in the module */
    BC_E582 = 582, /* a call argument the RV64 ABI cannot pass */
    BC_E583 = 583, /* a call return the RV64 ABI cannot carry */
    BC_E584 = 584, /* a call operand list the emitter cannot read */
    BC_E600 = 600, /* a cubin holds one kernel and the module has more */
    BC_E601 = 601, /* machine op with no SASS form */
    BC_E602 = 602, /* more basic blocks than the SASS CF pass holds */
    BC_E603 = 603, /* SASS register pressure beyond the file */
    BC_E604 = 604, /* SASS predicate pressure beyond the file */
    BC_E605 = 605, /* divergent region the SASS backend cannot reconverge */
    BC_E606 = 606, /* more virtual registers than the SASS backend maps */
    BC_E607 = 607, /* SASS code buffer full */
    BC_E608 = 608, /* the cubin writer refused the kernel */
    BC_E609 = 609, /* operand kind with no SASS form */
    BC_E610 = 610, /* special register with no SASS form */
    BC_E611 = 611, /* access address or width with no SASS form */
    BC_E640 = 640, /* CUDA intrinsic with no PTX form any backend emits */

    BC_E660 = 660,
    BC_E661 = 661,
    BC_E681 = 681, /* qualified template-id Booth cannot name */

    BC_E700 = 700, /* an aggregate copy wider than Booth lays down in one go */
    BC_E701 = 701, /* an aggregate rvalue with no address to pass or return */
    BC_E702 = 702, /* a ret disagrees with the declared return type */
    BC_E703 = 703, /* a call result disagrees with the callee return type */
    BC_E704 = 704, /* a call passes the wrong number of arguments */
    BC_E720 = 720, /* template argument Booth cannot fold, named */
    BC_E721 = 721, /* template-id used as a value Booth cannot resolve */
    BC_E760 = 760, /* a printf argument varargs cannot carry */
    BC_E780 = 780,
    BC_E781 = 781,
    BC_E782 = 782,
    BC_E800 = 800, /* linkage specification Booth does not implement */
    BC_E801 = 801, /* unexpected token inside a linkage specification */
    BC_E802 = 802, /* field reached through a tag that is never defined */
    BC_E803 = 803, /* more struct definitions than sema records */
    BC_E820 = 820, /* constexpr struct wider than the evaluator folds */
    BC_E821 = 821, /* constexpr struct values outrun the evaluator pool */
    BC_E840 = 840, /* a PTX operand register file the type suffix forbids */
    BC_E841 = 841, /* a PTX machine op with no verifier signature */
    BC_E842 = 842, /* a PTX return store disagrees with the return type */
    BC_E843 = 843, /* a PTX call result disagrees with the callee type */
    BC_E844 = 844, /* a register file change PTX has no conversion for */
    BC_E845 = 845, /* a PTX operand that must be a register but is not */
    BC_E860 = 860, /* a branch target that belongs to another function */
    BC_E861 = 861, /* a value operand defined in another function */
    BC_E862 = 862, /* two functions claiming the same blocks */
    BC_E863 = 863, /* two functions claiming the same instructions */
    BC_E864 = 864, /* a phi naming a block that does not branch to it */
    BC_E865 = 865, /* a phi whose predecessor count is not the block's */
    BC_E866 = 866, /* a memory fence with no Tensix SFPU form */
    BC_E867 = 867, /* an opcode with no pattern in the Tensix SFPU isel */

    BC_E900 = 900,
    BC_E901 = 901,
    BC_E902 = 902,
    BC_E903 = 903,
    BC_E904 = 904,
    BC_E905 = 905,
    BC_E906 = 906,
    BC_E907 = 907,
    BC_E908 = 908,
    BC_E909 = 909,
    BC_E910 = 910,
    BC_E911 = 911,
    BC_E912 = 912,
    BC_E940 = 940,
    BC_E960 = 960,

    BC_E980 = 980
} bc_eid_t;

/* Returns format string for eid -- loaded translation or compiled-in English */
const char *bc_efmt(bc_eid_t eid);

/* Load external translation file. Returns 0 on success. No-op if path NULL.
 * Parses both ENNN= (compiler errors) and ANNN= (ABEND messages, hex IDs). */
int bc_eload(const char *path);

/* ---- ABEND Messages ----
 * Keyed by uint16_t ABEND code (0x001..0x0FF).
 * Lang files use hex: "A0C4=memory access violation" etc. */

#define AB_AID_MAX 256

const char *ab_afmt(uint16_t code);

#endif /* BARRACUDA_BC_ERR_H */
