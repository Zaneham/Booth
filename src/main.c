#include "preproc.h"
#include "lexer.h"
#include "parser.h"
#include "sema.h"
#include "bir_lower.h"
#include "bir_mem2reg.h"
#include "bir_cfold.h"
#include "bir_dce.h"
#include "amdgpu.h"
#include "sched.h"
#include "verify.h"
#include "tensix.h"
#include "nvidia.h"
#include "metal.h"
#include "intel.h"
#include "triton.h"
#include <stdlib.h>

static char       source_buf[BC_MAX_SOURCE];
static char       pp_out_buf[BC_MAX_SOURCE];  /* preprocessor output */
static token_t    token_buf[BC_MAX_TOKENS];
static ast_node_t node_buf[BC_MAX_NODES];
static bir_module_t *bir_module; /* heap-allocated (~11 MB) */

static int read_file(const char *path, char *buf, uint32_t max, uint32_t *out_len)
{
    FILE *fp = fopen(path, "rb");
    if (!fp) {
        fprintf(stderr, "error: cannot open '%s'\n", path);
        return BC_ERR_IO;
    }
    fseek(fp, 0, SEEK_END);
    long sz = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    if (sz < 0 || (uint32_t)sz >= max) {
        fprintf(stderr, "error: file too large (%ld bytes, max %u)\n",
                sz, max);
        fclose(fp);
        return BC_ERR_IO;
    }
    *out_len = (uint32_t)fread(buf, 1, (size_t)sz, fp);
    buf[*out_len] = '\0';
    fclose(fp);
    return BC_OK;
}

static void dump_tokens(const lexer_t *L)
{
    char text[256];
    for (uint32_t i = 0; i < L->num_tokens; i++) {
        const token_t *t = &L->tokens[i];
        lexer_token_text(L, t, text, sizeof(text));

        printf("%4u:%-3u  %-20s  %s\n",
               t->line, t->col,
               token_type_name(t->type),
               text);
    }
}

static void usage(const char *prog)
{
    fprintf(stderr,
        "BarraCUDA - CUDA Compiler\n"
        "Usage: %s [options] <file.cu>\n"
        "\n"
        "Options:\n"
        "  --lex         Tokenize and dump token stream\n"
        "  --parse       Parse and dump AST\n"
        "  --ir          Lower to BIR and print IR\n"
        "  --no-mem2reg  Skip mem2reg optimization pass\n"
        "  --no-cfold    Skip constant folding\n"
        "  --no-dce      Skip dead code elimination\n"
        "  --no-sched    Skip instruction scheduling\n"
        "  --sema        Run semantic analysis and dump types\n"
        "  --pp          Preprocess only and print result\n"
        "  --no-pp       Skip preprocessor\n"
        "  -I <dir>      Add include search path\n"
        "  -D <name[=val]> Define a preprocessor macro\n"
        "  --amdgpu      Compile to AMDGCN assembly (default: gfx1100)\n"
        "  --amdgpu-bin  Compile to AMDGPU ELF code object (.hsaco)\n"
        "  --gfx90a      Target CDNA 2 (gfx90a, MI250)\n"
        "  --gfx942      Target CDNA 3 (gfx942, MI300X)\n"
        "  --gfx1030     Target RDNA 2 (gfx1030)\n"
        "  --gfx1200     Target RDNA 4 (gfx1200)\n"
        "  --no-graphcolor  Force linear scan register allocation\n"
        "  --ssa-ra         Divergence-aware SSA register allocation\n"
        "  --max-vgprs N    Cap VGPR count for regalloc (forces spills)\n"
        "  --tensix      Compile to TT-Metalium C++ (Tensix SFPU)\n"
        "  --nvidia-ptx  Compile to NVIDIA PTX (sm_89)\n"
        "  --hip         HIP frontend mode (predefines __HIPCC__ and platform macros;\n"
        "                auto-on for .hip files; combine with --amdgpu-bin or --nvidia-ptx)\n"
        "  --triton      Triton frontend mode (parses Python source); use --lex to dump\n"
        "                tokens or --parse to dump the AST; sema/lower still to come\n"
        "  --metal       Compile to Apple Metal Shading Language (stub)\n"
        "  --intel-spirv Compile to SPIR-V for Intel Arc Xe (stub)\n"
        "  --xe-lpg      Target Xe-LPG (Arc / integrated)\n"
        "  --xe-hpg      Target Xe-HPG (Alchemist, Battlemage) [default]\n"
        "  --xe-hpc      Target Xe-HPC (Ponte Vecchio)\n"
        "  --xe2         Target Xe2 (Lunar Lake, next-gen Arc)\n"
        "  -o <file>     Output file (for --amdgpu-bin, --tensix, --nvidia-ptx, --metal, --intel-spirv)\n"
        "  --lang <file> Load translated error messages\n"
        "  --help        Show this message\n"
        "\n", prog);
}

int main(int argc, char *argv[])
{
    const char *file = NULL;
    const char *output_file = NULL;
    const char *lang_file = NULL;
    int mode_pp = 0;
    int mode_lex = 0;
    int mode_parse = 0;
    int mode_sema = 0;
    int mode_ir = 0;
    int mode_amdgpu = 0;
    int mode_amdgpu_bin = 0;
    int mode_tensix = 0;
    int mode_nvidia = 0;
    int mode_metal = 0;
    int mode_intel = 0;
    int mode_hip = 0;           /* HIP frontend: see HIP NOTES below */
    int mode_triton = 0;        /* Triton frontend: see TRITON NOTES below */
    intel_target_t intel_target = INTEL_TARGET_XE_HPG;
    int nv_bkhit = 0;
    int no_mem2reg = 0;
    int no_cfold = 0;
    int no_dce = 0;
    int no_sched = 0;
    int no_pp = 0;
    int snap_mode = 0;
    amd_target_t amd_target = AMD_TARGET_GFX1100;
    uint32_t     amd_elfm  = 0x41;       /* EF_AMDGPU_MACH for exact chip */
    const char  *amd_chip  = "gfx1100";  /* chip string for ELF metadata */

    /* Collect -I and -D options for preprocessor */
    const char *include_paths[PP_MAX_INCLUDE_PATHS];
    int num_include_paths = 0;
    const char *defines[128];
    int num_defines = 0;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--lex") == 0)
            mode_lex = 1;
        else if (strcmp(argv[i], "--parse") == 0)
            mode_parse = 1;
        else if (strcmp(argv[i], "--sema") == 0)
            mode_sema = 1;
        else if (strcmp(argv[i], "--ir") == 0)
            mode_ir = 1;
        else if (strcmp(argv[i], "--pp") == 0)
            mode_pp = 1;
        else if (strcmp(argv[i], "--no-pp") == 0)
            no_pp = 1;
        else if (strcmp(argv[i], "--amdgpu") == 0)
            mode_amdgpu = 1;
        else if (strcmp(argv[i], "--amdgpu-bin") == 0)
            mode_amdgpu_bin = 1;
        /* CDNA 2 (GFX9) */
        else if (strcmp(argv[i], "--gfx90a") == 0)
            { amd_target = AMD_TARGET_GFX90A; amd_elfm = 0x3F; amd_chip = "gfx90a"; }
        /* CDNA 3 (GFX9.4.2) */
        else if (strcmp(argv[i], "--gfx942") == 0)
            { amd_target = AMD_TARGET_GFX942; amd_elfm = 0x54C; amd_chip = "gfx942"; } /* xnack=off sramecc=off */
        /* RDNA 2 (GFX10.3) */
        else if (strcmp(argv[i], "--gfx1030") == 0)
            { amd_target = AMD_TARGET_GFX1030; amd_elfm = 0x36; amd_chip = "gfx1030"; }
        else if (strcmp(argv[i], "--gfx1031") == 0)
            { amd_target = AMD_TARGET_GFX1030; amd_elfm = 0x37; amd_chip = "gfx1031"; }
        else if (strcmp(argv[i], "--gfx1032") == 0)
            { amd_target = AMD_TARGET_GFX1030; amd_elfm = 0x38; amd_chip = "gfx1032"; }
        else if (strcmp(argv[i], "--gfx1033") == 0)
            { amd_target = AMD_TARGET_GFX1030; amd_elfm = 0x39; amd_chip = "gfx1033"; }
        else if (strcmp(argv[i], "--gfx1034") == 0)
            { amd_target = AMD_TARGET_GFX1030; amd_elfm = 0x3e; amd_chip = "gfx1034"; }
        else if (strcmp(argv[i], "--gfx1035") == 0)
            { amd_target = AMD_TARGET_GFX1030; amd_elfm = 0x3d; amd_chip = "gfx1035"; }
        else if (strcmp(argv[i], "--gfx1036") == 0)
            { amd_target = AMD_TARGET_GFX1030; amd_elfm = 0x45; amd_chip = "gfx1036"; }
        /* RDNA 3 (GFX11) */
        else if (strcmp(argv[i], "--gfx1100") == 0)
            { amd_target = AMD_TARGET_GFX1100; amd_elfm = 0x41; amd_chip = "gfx1100"; }
        else if (strcmp(argv[i], "--gfx1101") == 0)
            { amd_target = AMD_TARGET_GFX1100; amd_elfm = 0x46; amd_chip = "gfx1101"; }
        else if (strcmp(argv[i], "--gfx1102") == 0)
            { amd_target = AMD_TARGET_GFX1100; amd_elfm = 0x47; amd_chip = "gfx1102"; }
        else if (strcmp(argv[i], "--gfx1103") == 0)
            { amd_target = AMD_TARGET_GFX1100; amd_elfm = 0x44; amd_chip = "gfx1103"; }
        /* RDNA 3.5 (GFX11.5) */
        else if (strcmp(argv[i], "--gfx1150") == 0)
            { amd_target = AMD_TARGET_GFX1100; amd_elfm = 0x43; amd_chip = "gfx1150"; }
        else if (strcmp(argv[i], "--gfx1151") == 0)
            { amd_target = AMD_TARGET_GFX1100; amd_elfm = 0x4a; amd_chip = "gfx1151"; }
        else if (strcmp(argv[i], "--gfx1152") == 0)
            { amd_target = AMD_TARGET_GFX1100; amd_elfm = 0x55; amd_chip = "gfx1152"; }
        else if (strcmp(argv[i], "--gfx1153") == 0)
            { amd_target = AMD_TARGET_GFX1100; amd_elfm = 0x58; amd_chip = "gfx1153"; }
        /* RDNA 4 (GFX12) */
        else if (strcmp(argv[i], "--gfx1200") == 0)
            { amd_target = AMD_TARGET_GFX1200; amd_elfm = 0x48; amd_chip = "gfx1200"; }
        else if (strcmp(argv[i], "--gfx1201") == 0)
            { amd_target = AMD_TARGET_GFX1200; amd_elfm = 0x4e; amd_chip = "gfx1201"; }
        else if (strcmp(argv[i], "--tensix") == 0)
            mode_tensix = 1;
        else if (strcmp(argv[i], "--nvidia-ptx") == 0)
            mode_nvidia = 1;
        else if (strcmp(argv[i], "--metal") == 0)
            mode_metal = 1;
        else if (strcmp(argv[i], "--intel-spirv") == 0)
            mode_intel = 1;
        else if (strcmp(argv[i], "--xe-lpg") == 0)
            intel_target = INTEL_TARGET_XE_LPG;
        else if (strcmp(argv[i], "--xe-hpg") == 0)
            intel_target = INTEL_TARGET_XE_HPG;
        else if (strcmp(argv[i], "--xe-hpc") == 0)
            intel_target = INTEL_TARGET_XE_HPC;
        else if (strcmp(argv[i], "--xe2") == 0)
            intel_target = INTEL_TARGET_XE2;
        else if (strcmp(argv[i], "--hip") == 0)
            mode_hip = 1;
        else if (strcmp(argv[i], "--triton") == 0)
            mode_triton = 1;
        else if (strcmp(argv[i], "--bkhit") == 0)
            nv_bkhit = 1;
        else if (strcmp(argv[i], "--lang") == 0 && i + 1 < argc)
            lang_file = argv[++i];
        else if (strcmp(argv[i], "-o") == 0 && i + 1 < argc)
            output_file = argv[++i];
        else if (strcmp(argv[i], "-I") == 0 && i + 1 < argc) {
            if (num_include_paths < PP_MAX_INCLUDE_PATHS)
                include_paths[num_include_paths++] = argv[++i];
        } else if (strncmp(argv[i], "-I", 2) == 0 && argv[i][2]) {
            if (num_include_paths < PP_MAX_INCLUDE_PATHS)
                include_paths[num_include_paths++] = argv[i] + 2;
        } else if (strcmp(argv[i], "-D") == 0 && i + 1 < argc) {
            if (num_defines < 128)
                defines[num_defines++] = argv[++i];
        } else if (strncmp(argv[i], "-D", 2) == 0 && argv[i][2]) {
            if (num_defines < 128)
                defines[num_defines++] = argv[i] + 2;
        } else if (strcmp(argv[i], "--no-mem2reg") == 0)
            no_mem2reg = 1;
        else if (strcmp(argv[i], "--no-cfold") == 0)
            no_cfold = 1;
        else if (strcmp(argv[i], "--no-dce") == 0)
            no_dce = 1;
        else if (strcmp(argv[i], "--no-sched") == 0)
            no_sched = 1;
        else if (strcmp(argv[i], "--no-graphcolor") == 0)
            amd_ra_lin = 1;
        else if (strcmp(argv[i], "--ssa-ra") == 0)
            amd_ra_ssa = 1;
        else if (strcmp(argv[i], "--max-vgprs") == 0 && i + 1 < argc)
            amd_max_vgpr = atoi(argv[++i]);
        else if (strcmp(argv[i], "--snap") == 0)
            snap_mode = 1;
        else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            usage(argv[0]);
            return 0;
        } else if (argv[i][0] != '-')
            file = argv[i];
        else {
            fprintf(stderr, "unknown option: %s\n", argv[i]);
            usage(argv[0]);
            return 1;
        }
    }

    if (!file) {
        usage(argv[0]);
        return 1;
    }

    if (!mode_pp && !mode_lex && !mode_parse && !mode_sema && !mode_ir &&
        !mode_amdgpu && !mode_amdgpu_bin && !mode_tensix && !mode_nvidia &&
        !mode_metal && !mode_intel)
        mode_parse = 1;

    /* ---- HIP NOTES (1 of 2) -------------------------------------------
     * HIP is a frontend-only mode, not a separate parser. The HIP source
     * language is, in practice, a syntactic superset of CUDA: the same
     * __global__ / __device__ / __shared__ qualifiers, the same
     * threadIdx / blockIdx / blockDim builtins, and the same arithmetic
     * happens with the same syntax. What changes when you flip --hip on
     * is the set of preprocessor macros we predefine, so that any
     * #if defined(__HIPCC__) or __HIP_PLATFORM_AMD__ guards in the source
     * pick the HIP branch instead of falling through to whatever the
     * source thought the default platform was.
     *
     * Auto-detection: if the filename ends in ".hip", we assume HIP mode
     * without making the user say so on the command line, since the
     * extension is a strong enough signal for anyone using a HIP build
     * pipeline to drop their files into BarraCUDA unchanged. */
    if (file) {
        size_t flen = strlen(file);
        if (flen >= 4 && strcmp(file + flen - 4, ".hip") == 0)
            mode_hip = 1;
    }

    /* Load translation file before any diagnostics fire */
    if (lang_file) bc_eload(lang_file);

    uint32_t src_len = 0;
    if (read_file(file, source_buf, BC_MAX_SOURCE, &src_len) != BC_OK)
        return 1;

    /* ---- TRITON NOTES -------------------------------------------------
     * The Triton frontend is a parallel input path that does not share
     * the C99 preprocessor or lexer. When --triton is on, we route the
     * source through src/triton/ instead of through src/fe/. For now
     * the Triton frontend stops at the lexer; --lex dumps the token
     * stream and the program exits without going any further down the
     * pipeline, because the parser, sema, and lowering passes are
     * still stubs. The downstream backends do not need to know any of
     * this is happening: once tn_lower starts producing BIR, the same
     * BIR consumers we use for CUDA and HIP will accept it without
     * comment. */
    if (mode_triton) {
        static tn_lex_t   tn_lex_state;
        static tn_tok_t   tn_tok_buf[TN_MAX_TOKENS];
        tn_lex_init(&tn_lex_state, source_buf, src_len,
                    tn_tok_buf, TN_MAX_TOKENS);
        int trc = tn_tokenize(&tn_lex_state);
        if (tn_lex_state.num_errors > 0) {
            for (int i = 0; i < tn_lex_state.num_errors; i++) {
                fprintf(stderr, "%s:%u:%u: E%03u: %s\n",
                        file,
                        tn_lex_state.errors[i].loc.line,
                        tn_lex_state.errors[i].loc.col,
                        tn_lex_state.errors[i].eid,
                        tn_lex_state.errors[i].msg);
            }
        }
        if (mode_lex) {
            char text[256];
            for (uint32_t i = 0; i < tn_lex_state.num_tokens; i++) {
                const tn_tok_t *t = &tn_tok_buf[i];
                tn_tok_text(&tn_lex_state, t, text, sizeof(text));
                printf("%4u:%-3u  %-12s  %s\n",
                       t->line, (unsigned)t->col,
                       tn_tok_name(t->kind),
                       text);
            }
            printf("\n%u tokens, %d error(s)\n",
                   tn_lex_state.num_tokens, tn_lex_state.num_errors);
            return trc != BC_OK ? 1 : 0;
        }
        if (mode_parse) {
            tn_parse_t *tnp = (tn_parse_t *)malloc(sizeof(tn_parse_t));
            if (!tnp) {
                fprintf(stderr, "error: failed to allocate Triton parser\n");
                return 1;
            }
            tn_parse_init(tnp, &tn_lex_state);
            int prc = tn_parse(tnp);
            for (int i = 0; i < tnp->num_errors; i++) {
                fprintf(stderr, "%s:%u:%u: E%03u: %s\n",
                        file,
                        tnp->errors[i].loc.line,
                        tnp->errors[i].loc.col,
                        tnp->errors[i].eid,
                        tnp->errors[i].msg);
            }
            tn_ast_dump(tnp, stdout);
            free(tnp);
            return prc != BC_OK ? 1 : 0;
        }
        /* Anything beyond --lex and --parse is not implemented yet
         * for Triton. Sema and BIR lowering arrive in future sittings. */
        fprintf(stderr,
            "triton: lexer and parser only in this sitting. Sema and\n"
            "        BIR lowering arrive in future sittings. Use --lex\n"
            "        to dump tokens or --parse to dump the AST.\n");
        return trc != BC_OK ? 1 : 0;
    }

    /* Preprocessing */
    const char *lex_src = source_buf;
    uint32_t    lex_len = src_len;

    if (!no_pp) {
        preproc_t *pp = (preproc_t *)malloc(sizeof(preproc_t));
        if (!pp) {
            fprintf(stderr, "error: failed to allocate preprocessor\n");
            return 1;
        }
        pp_init(pp, source_buf, src_len, pp_out_buf, BC_MAX_SOURCE, file);

        /* ---- HIP NOTES (2 of 2) ---------------------------------------
         * This is the only spot in the pipeline that knows or cares
         * whether we are compiling CUDA or HIP. pp_init has just defined
         * the CUDA defaults (__BARRACUDA__, __CUDA_ARCH__, __CUDACC__)
         * unconditionally, which is correct for CUDA and harmless for
         * HIP because real HIP source files distinguish platforms with
         * __HIP_PLATFORM_AMD__ versus __HIP_PLATFORM_NVIDIA__ rather
         * than by the presence or absence of __CUDACC__.
         *
         * When --hip is on, we additively define the HIP-specific
         * macros so that the preprocessor takes the HIP branch wherever
         * the source asks for it:
         *   __HIPCC__               compiler identity, "we are a HIP compiler"
         *   __HIP_DEVICE_COMPILE__  we are compiling device code (always true here)
         *   __HIP_PLATFORM_AMD__    target is AMD silicon (the common case)
         *   __HIP_PLATFORM_NVIDIA__ target is NVIDIA via the HIP-on-CUDA path
         *
         * Beyond these macros, nothing in the parser, sema, IR, or
         * backends needs to know about HIP. The pipeline downstream of
         * here is identical to a CUDA compile. */
        if (mode_hip) {
            pp_define(pp, "__HIPCC__", "1");
            pp_define(pp, "__HIP_DEVICE_COMPILE__", "1");
            if (mode_nvidia)
                pp_define(pp, "__HIP_PLATFORM_NVIDIA__", "1");
            else
                pp_define(pp, "__HIP_PLATFORM_AMD__", "1");
        }

        for (int i = 0; i < num_include_paths; i++)
            pp_add_include_path(pp, include_paths[i]);
        for (int i = 0; i < num_defines; i++) {
            char dname[BC_MAX_IDENT];
            const char *eq = strchr(defines[i], '=');
            if (eq) {
                uint32_t nlen = (uint32_t)(eq - defines[i]);
                if (nlen >= BC_MAX_IDENT) nlen = BC_MAX_IDENT - 1;
                memcpy(dname, defines[i], nlen);
                dname[nlen] = '\0';
                pp_define(pp, dname, eq + 1);
            } else {
                pp_define(pp, defines[i], "1");
            }
        }

        int prc = pp_process(pp);

        if (pp->num_errors > 0) {
            for (int i = 0; i < pp->num_errors; i++) {
                fprintf(stderr, "%s:%u: E%03u: %s\n",
                        file, pp->errors[i].loc.line,
                        pp->errors[i].eid, pp->errors[i].msg);
            }
        }

        if (mode_pp) {
            fwrite(pp_out_buf, 1, pp->out_len, stdout);
            free(pp);
            return prc != BC_OK ? 1 : 0;
        }

        lex_src = pp_out_buf;
        lex_len = pp->out_len;
        free(pp);
    }

    lexer_t L;
    lexer_init(&L, lex_src, lex_len, token_buf, BC_MAX_TOKENS);
    int rc = lexer_tokenize(&L);

    if (L.num_errors > 0) {
        for (int i = 0; i < L.num_errors; i++) {
            fprintf(stderr, "%s:%u:%u: E%03u: %s\n",
                    file, L.errors[i].loc.line, L.errors[i].loc.col,
                    L.errors[i].eid, L.errors[i].msg);
        }
    }

    if (mode_lex) {
        dump_tokens(&L);
        printf("\n%u tokens, %d error(s)\n", L.num_tokens, L.num_errors);
    }

    if (mode_parse || mode_sema || mode_ir || mode_amdgpu || mode_amdgpu_bin ||
        mode_tensix || mode_nvidia || mode_metal || mode_intel) {
        parser_t P;
        parser_init(&P, token_buf, L.num_tokens, lex_src,
                    node_buf, BC_MAX_NODES);
        uint32_t root = parser_parse(&P);

        if (P.num_errors > 0) {
            for (int i = 0; i < P.num_errors; i++) {
                fprintf(stderr, "%s:%u:%u: E%03u: %s\n",
                        file, P.errors[i].loc.line, P.errors[i].loc.col,
                        P.errors[i].eid, P.errors[i].msg);
            }
        }

        if (mode_parse) {
            ast_dump(&P, root, 0);
            printf("\n%u nodes, %d parse error(s)\n",
                   P.num_nodes, P.num_errors);
        }

        /* Semantic analysis */
        sema_ctx_t *sema_ctx = NULL;
        if ((mode_sema || mode_ir || mode_amdgpu || mode_amdgpu_bin ||
             mode_tensix || mode_nvidia || mode_metal || mode_intel) &&
            P.num_errors == 0)
        {
            sema_ctx = (sema_ctx_t *)malloc(sizeof(sema_ctx_t));
            if (!sema_ctx) {
                fprintf(stderr, "error: failed to allocate sema context\n");
                return 1;
            }
            sema_init(sema_ctx, &P, root);
            sema_check(sema_ctx, root);

            if (sema_ctx->num_errors > 0) {
                for (int i = 0; i < sema_ctx->num_errors; i++) {
                    fprintf(stderr, "%s:%u:%u: E%03u: %s\n",
                            file, sema_ctx->errors[i].loc.line,
                            sema_ctx->errors[i].loc.col,
                            sema_ctx->errors[i].eid,
                            sema_ctx->errors[i].msg);
                }
            }

            if (mode_sema) {
                sema_dump(sema_ctx, root);
                int sema_rc = sema_ctx->num_errors > 0 ? 1 : 0;
                free(sema_ctx);
                return sema_rc;
            }
        }

        if ((mode_ir || mode_amdgpu || mode_amdgpu_bin || mode_tensix ||
             mode_nvidia || mode_metal || mode_intel) &&
            P.num_errors == 0) {
            bc_error_t lower_errs[BC_MAX_ERRORS];
            int num_lower_errs = 0;
            bir_module = (bir_module_t *)malloc(sizeof(bir_module_t));
            if (!bir_module) {
                fprintf(stderr, "error: failed to allocate BIR module\n");
                return 1;
            }
            int lrc = bir_lower(&P, root, bir_module, sema_ctx,
                                lower_errs, &num_lower_errs);
            if (num_lower_errs > 0) {
                for (int i = 0; i < num_lower_errs; i++) {
                    fprintf(stderr, "%s:%u:%u: E%03u: %s\n",
                            file, lower_errs[i].loc.line,
                            lower_errs[i].loc.col,
                            lower_errs[i].eid, lower_errs[i].msg);
                }
            }
            if (lrc == BC_OK) {
                if (!no_mem2reg)
                    bir_mem2reg(bir_module);
                if (!no_cfold)
                    bir_cfold(bir_module);
                if (!no_dce)
                    bir_dce(bir_module);

                if (mode_ir) {
                    bir_print_module(bir_module, stdout);
                    printf("\n; %u functions, %u globals, %u instructions\n",
                           bir_module->num_funcs, bir_module->num_globals,
                           bir_module->num_insts);
                }

                if (mode_amdgpu || mode_amdgpu_bin) {
                    amd_module_t *amd = (amd_module_t *)malloc(sizeof(amd_module_t));
                    if (!amd) {
                        fprintf(stderr, "error: failed to allocate AMD module\n");
                        free(bir_module);
                        return 1;
                    }
                    amd->target = amd_target;
                    amd->elf_mach = amd_elfm;
                    amd->snap_mode = (uint8_t)snap_mode;
                    snprintf(amd->chip_name, sizeof(amd->chip_name),
                             "%s", amd_chip);
                    int arc = amdgpu_compile(bir_module, amd);
                    if (arc == BC_OK) {
                        vfy_res_t v1 = bc_vfy(amd, VFY_ISEL);
                        if (v1.errs) {
                            fprintf(stderr, "verify: %u error(s) after isel\n",
                                    v1.errs);
                            arc = BC_ERR_VERIFY;
                        }
                    }
                    if (arc == BC_OK) {
                        if (!no_sched)
                            amdgpu_sched(amd);
                        amdgpu_regalloc(amd);
                        vfy_res_t v2 = bc_vfy(amd, VFY_RA);
                        if (v2.errs) {
                            fprintf(stderr, "verify: %u error(s) after regalloc\n",
                                    v2.errs);
                            arc = BC_ERR_VERIFY;
                        }
                    }
                    if (arc == BC_OK) {
                        if (mode_amdgpu_bin)
                            amdgpu_emit_elf(amd,
                                output_file ? output_file : "a.hsaco");
                        else
                            amdgpu_emit_asm(amd, stdout);
                    } else {
                        if (arc != BC_ERR_VERIFY)
                            fprintf(stderr, "error: AMDGPU compilation failed\n");
                        rc = arc;
                    }
                    free(amd);
                }

                if (mode_tensix) {
                    tt_module_t *ttm = (tt_module_t *)malloc(sizeof(tt_module_t));
                    if (!ttm) {
                        fprintf(stderr, "error: failed to allocate Tensix module\n");
                        free(bir_module);
                        return 1;
                    }
                    int trc = tensix_compile(bir_module, ttm);
                    if (trc == BC_OK) {
                        tensix_coarsen(ttm);
                        tensix_regalloc(ttm);

                        const char *compute_path =
                            output_file ? output_file : "a_compute.cpp";

                        tensix_analyze_datamov(bir_module, ttm, &ttm->dmov);
                        tensix_emit_metalium(ttm, compute_path);

                        char host_path[BC_MAX_PATH];
                        char reader_path[BC_MAX_PATH];
                        char writer_path[BC_MAX_PATH];
                        const char *stem = strstr(compute_path, "_compute");
                        int pfx;
                        if (stem) {
                            pfx = (int)(stem - compute_path);
                        } else {
                            const char *dot = strrchr(compute_path, '.');
                            pfx = dot ? (int)(dot - compute_path)
                                      : (int)strlen(compute_path);
                        }
                        snprintf(host_path,   sizeof(host_path),
                                 "%.*s_host.cpp",   pfx, compute_path);
                        snprintf(reader_path, sizeof(reader_path),
                                 "%.*s_reader.cpp", pfx, compute_path);
                        snprintf(writer_path, sizeof(writer_path),
                                 "%.*s_writer.cpp", pfx, compute_path);

                        tensix_emit_reader(ttm, &ttm->dmov, reader_path);
                        tensix_emit_writer(ttm, &ttm->dmov, writer_path);
                        tensix_emit_host_full(ttm, &ttm->dmov, host_path,
                                              reader_path, compute_path,
                                              writer_path);
                    } else {
                        fprintf(stderr, "error: Tensix compilation failed\n");
                        rc = trc;
                    }
                    free(ttm);
                }

                if (mode_nvidia) {
                    nv_module_t *nvm = (nv_module_t *)malloc(sizeof(nv_module_t));
                    if (!nvm) {
                        fprintf(stderr, "error: failed to allocate NVIDIA module\n");
                        free(bir_module);
                        return 1;
                    }
                    int nrc = nv_compile(bir_module, nvm);
                    if (nrc == BC_OK) {
                        nvm->bkhit = (uint8_t)nv_bkhit;
                        nv_emit_ptx(nvm,
                            output_file ? output_file : "a.ptx");
                    } else {
                        fprintf(stderr, "error: NVIDIA PTX compilation failed\n");
                        rc = nrc;
                    }
                    free(nvm);
                }

                if (mode_metal) {
                    metal_module_t *mm = (metal_module_t *)malloc(sizeof(metal_module_t));
                    if (!mm) {
                        fprintf(stderr, "error: failed to allocate Metal module\n");
                        free(bir_module);
                        return 1;
                    }
                    int mrc = metal_compile(bir_module, mm);
                    if (mrc == BC_OK) {
                        metal_emit_msl(mm,
                            output_file ? output_file : "a.metal");
                    } else {
                        fprintf(stderr,
                            "error: Metal backend compilation failed\n");
                        rc = mrc;
                    }
                    free(mm);
                }

                if (mode_intel) {
                    intel_module_t *im = (intel_module_t *)malloc(sizeof(intel_module_t));
                    if (!im) {
                        fprintf(stderr, "error: failed to allocate Intel module\n");
                        free(bir_module);
                        return 1;
                    }
                    int irc = intel_compile(bir_module, im, intel_target);
                    if (irc == BC_OK) {
                        intel_emit_spirv(im,
                            output_file ? output_file : "a.spv");
                    } else {
                        fprintf(stderr,
                            "error: Intel SPIR-V backend is also staked-out "
                            "territory, also not yet a working compiler. "
                            "Same disclaimer, different vendor.\n");
                        rc = irc;
                    }
                    free(im);
                }
            }
            free(bir_module);
            if (lrc != BC_OK) rc = lrc;
        }

        if (sema_ctx) free(sema_ctx);
        if (P.num_errors > 0) rc = BC_ERR_PARSE;
    }

    return rc != BC_OK ? 1 : 0;
}
