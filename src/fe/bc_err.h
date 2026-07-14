/* bc_err.h — Error localisation for Booth
 *
 * Every diagnostic gets a numeric ID (E001..E129). English is compiled in;
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
 * E100-E129  Lowering
 */

#define BC_EID_MAX 130

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

    /* ---- Lowering (E100-E129) ---- */
    BC_E100 = 100, /* too many labels (max 256) */
    BC_E101 = 101, /* undefined variable */
    BC_E102 = 102, /* unsupported binary op */
    BC_E103 = 103, /* unsupported unary prefix */
    BC_E104 = 104, /* unsupported postfix op */
    BC_E105 = 105, /* unknown function in call */
    BC_E106 = 106, /* unsupported expression node */
    BC_E107 = 107, /* undefined lvalue */
    BC_E108 = 108, /* parameter not addressable */
    BC_E109 = 109, /* not an lvalue (prefix) */
    BC_E110 = 110, /* unknown field in lvalue */
    BC_E111 = 111  /* not an lvalue */
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
