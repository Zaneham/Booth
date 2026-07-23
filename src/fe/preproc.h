#ifndef BARRACUDA_PREPROC_H
#define BARRACUDA_PREPROC_H

#include "barracuda.h"

/*
 * CUDA preprocessor for Booth.
 * Runs before the lexer on raw source text.
 * Handles: #include, #define/#undef, macro expansion,
 *          #ifdef/#ifndef/#if/#elif/#else/#endif, #pragma, #error.
 * Output: single expanded source buffer ready for lexer.
 */

#define BC_ERR_PREPROC      -5

#define PP_MAX_MACROS       2048
#define PP_MAX_PARAMS       16
#define PP_MAX_INCLUDE_PATHS 32
#define PP_MAX_COND_DEPTH   64
#define PP_MAX_FILE_DEPTH   16
#define PP_MAX_EXPAND_DEPTH 32
#define PP_POOL_SIZE        (256 * 1024)   /* 256KB for macro names + bodies */
#define PP_LINE_BUF         (64 * 1024)    /* 64KB max logical line */
#define PP_EXPAND_BUF       (128 * 1024)   /* 128KB expansion workspace */

typedef struct {
    uint32_t name_off;
    uint16_t name_len;
    uint32_t body_off;
    uint16_t body_len;
    int16_t  num_params;   /* -1 = object-like, 0+ = function-like */
    uint16_t param_off[PP_MAX_PARAMS];
    uint8_t  param_len[PP_MAX_PARAMS];
} pp_macro_t;   /* ~56 bytes */

typedef struct {
    int  active;        /* currently emitting code? */
    int  seen_true;     /* has any branch been true? */
    int  parent_active; /* was parent block active? */
} pp_cond_t;

typedef struct {
    const char *saved_src;
    uint32_t    saved_src_len;
    uint32_t    saved_pos;
    uint32_t    saved_line;
    char        saved_filename[BC_MAX_PATH];
    char       *buf;    /* malloc'd include buffer (freed on pop) */
} pp_file_entry_t;

typedef struct {
    /* Input */
    const char *src;
    uint32_t    src_len;
    uint32_t    pos;
    uint32_t    line;

    /* Output */
    char       *out;
    uint32_t    out_len;
    uint32_t    out_max;

    /* Current filename */
    char        filename[BC_MAX_PATH];

    /* Macro table + string pool */
    pp_macro_t  macros[PP_MAX_MACROS];
    uint32_t    num_macros;
    char        pool[PP_POOL_SIZE];
    uint32_t    pool_len;

    /* Include search paths */
    char        include_paths[PP_MAX_INCLUDE_PATHS][BC_MAX_PATH];
    uint32_t    num_include_paths;

    /* Conditional compilation stack */
    pp_cond_t   cond_stack[PP_MAX_COND_DEPTH];
    int         cond_depth;

    /* File inclusion stack */
    pp_file_entry_t file_stack[PP_MAX_FILE_DEPTH];
    int         file_depth;

    /* Expansion workspace */
    char        line_buf[PP_LINE_BUF];
    char        exp_buf[PP_EXPAND_BUF];

    /* Block comment still open at end of line. Expansion runs per line, so
     * without this a macro named in comment prose gets substituted. */
    int         in_bcmt;

    /* Errors */
    bc_error_t  errors[BC_MAX_ERRORS];
    int         num_errors;
} preproc_t;

void pp_init(preproc_t *pp, const char *src, uint32_t len,
             char *out_buf, uint32_t out_max, const char *filename);

int  pp_add_include_path(preproc_t *pp, const char *path);
int  pp_define(preproc_t *pp, const char *name, const char *value);
int  pp_process(preproc_t *pp);

#endif /* BARRACUDA_PREPROC_H */
