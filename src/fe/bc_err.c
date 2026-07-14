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
    /* E006-E019 */ NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,

    /* ---- Parser ---- */
    /* E020 */ "expected '%s', got '%s'",
    /* E021 */ "AST node limit exceeded",
    /* E022 */ "expected expression",
    /* E023 */ "unexpected token in namespace",
    /* E024 */ "expected declaration",
    /* E025 */ "unexpected token in function body",
    /* E026 */ "unexpected token in block",
    /* E027 */ "unexpected token at top level",
    /* E028-E039 */ NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,

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
    /* E053-E069 */ NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,

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
    /* E082-E099 */ NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,

    /* ---- Lowering ---- */
    /* E100 */ "too many labels (max 256)",
    /* E101 */ "undefined variable",
    /* E102 */ "unsupported binary op",
    /* E103 */ "unsupported unary prefix",
    /* E104 */ "unsupported postfix op",
    /* E105 */ "unknown function in call",
    /* E106 */ "unsupported expression node",
    /* E107 */ "undefined lvalue",
    /* E108 */ "parameter not addressable",
    /* E109 */ "not an lvalue (prefix)",
    /* E110 */ "unknown field in lvalue",
    /* E111 */ "not an lvalue",
    /* E112-E129 */
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
