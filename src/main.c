#include "preproc.h"
#include "lexer.h"
#include "parser.h"
#include "sema.h"
#include "bc_render.h"
#include "bir_lower.h"
#include "bir_sroa.h"
#include "bir_inline.h"
#include "bir_mem2reg.h"
#include "bir_cfold.h"
#include "bir_dce.h"
#include "bir_softfp.h"
#include "amdgpu.h"
#include "sched.h"
#include "verify.h"
#include "tensix.h"
#include "nvidia.h"
#include "metal.h"
#include "intel.h"
#include "triton.h"
#include "tdf.h"
#include "rv_buf.h"
#include "rv_elf.h"
#include "rv_isel.h"
#include "cpu.h"
#include "rv64.h"
#include "backend.h"
#include "backend_cfg.h"
#include <stdlib.h>

static char       source_buf[BC_MAX_SOURCE];
static char       pp_out_buf[BC_MAX_SOURCE];  /* preprocessor output */
static token_t    token_buf[BC_MAX_TOKENS];
static ast_node_t node_buf[BC_MAX_NODES];
static bir_module_t *bir_module; /* heap-allocated (~11 MB) */

/* ---- Shared Backend Dispatcher ----
 * After a frontend has filled bir_module, the optimisation passes
 * and the per-backend code are identical regardless of which
 * frontend was used. Packaged here so the C99 path and the Triton
 * path can both call it without duplicating two hundred lines of
 * backend wiring. */

/* be_cfg_t definition lives in src/backend/backend_cfg.h so every
 * backend descriptor sees the same field layout. Alias kept for the
 * few local uses below that predate the move; new code should use
 * be_cfg_t directly. */
typedef be_cfg_t backend_cfg_t;

/* TDF module and lowering scratch live in BSS, not on the stack.
 * The struct is ~20 KB and trips -Wstack-usage hard if you put it
 * in run_bir_backends, plus the rest of that function is already
 * doing four large allocations in shared scope. */
static td_mod_t  g_tdf_mod;
static td_lout_t g_tdf_out;

static int target_for_cfg(const backend_cfg_t *cfg)
{
    /* The first mode that wins picks the TDF target. Lowering is a
     * passthrough on all targets today, so the choice only matters
     * for the dump label and for when Tensix fission starts doing
     * something interesting on its own branch. */
    if (cfg->mode_tdf_fission) return TD_TGT_TENSIX;
    const be_desc_t *b = be_active();
    if (b != NULL && strcmp(b->name, "tensix") == 0) return TD_TGT_TENSIX;
    if (b != NULL && strcmp(b->name, "nvptx")  == 0) return TD_TGT_NVIDIA;
    return TD_TGT_AMD;
}

static int run_bir_backends(bir_module_t *bir, const backend_cfg_t *cfg)
{
    int rc = BC_OK;

    /* String literal globals (BIR_CONST_BYTES initializer) require
     * backend support that is still being wired in. Phase 1 of the
     * string-literal work landed the BIR shape and the frontend
     * lowering; Phase 2 per backend (AMD .rodata, NVIDIA .const,
     * Tensix C++ static const, Metal/Intel) is open as a set of
     * GitHub issues. Until those land, refuse cleanly rather than
     * emit silent wrong code that reads from address zero. */
    for (uint32_t gi = 0; gi < bir->num_globals; gi++) {
        if (!bir_global_is_bytes(bir, gi)) continue;
        const char *gname = (bir->globals[gi].name < bir->string_len)
                            ? &bir->strings[bir->globals[gi].name]
                            : "<anon>";
        fprintf(stderr,
            "E110: string literal global '%s' requires backend "
            "codegen support that is not yet wired (see issues "
            "#93 AMD, #94 NVIDIA, #95 Tensix). String literals "
            "in device code will not compile until those land.\n",
            gname);
        return BC_ERR_VERIFY;
    }

    /* Device-call inlining. The GPU and vector backends have no calling
     * convention for __device__ functions, their isel aborts on a BIR_CALL,
     * so splice the callee bodies in before anything else runs. mem2reg then
     * cleans up the inlined parameter stores as if they were always local.
     * CPU, RV64 and Metal emit real calls and are left untouched. */
    {
        const be_desc_t *b = be_active();
        if (b != NULL && (b->feats & BE_F_NOCALL)) {
            int irc = bir_inline_device(bir);
            if (irc != BC_OK) return irc;
        }
    }

    /* Optimisation passes: same shape regardless of frontend. */
    if (!cfg->no_sroa)    bir_sroa(bir);
    if (!cfg->no_mem2reg) bir_mem2reg(bir);
    if (!cfg->no_cfold)   bir_cfold(bir);
    if (!cfg->no_dce)     bir_dce(bir);

    /* After cfold, so constant float folds instead of becoming a call. */
    if (cfg->mode_rv_elf) {
        int src = bir_softfp(bir);
        if (src != BC_OK) return src;
    }

    /* Wrap the BIR in a TDF module and lower it. For AMD and NVIDIA
     * this is a degenerate passthrough, the lowering hands the same
     * BIR pointer straight back, and the cost is one memset plus
     * three field assignments. The reason we go through the dance
     * at all is so the layer stays exercised and the day Tensix
     * fission lands here is the day it works rather than the day
     * we go looking for which backend forgot to call it. */
    int tdrc = td_build_solo_from_bir(&g_tdf_mod, target_for_cfg(cfg), bir);
    if (tdrc != BC_OK) return tdrc;

    /* Fission preview: run the analysis pass, dump the resulting
     * three-region graph, and stop. The lowering can't materialise
     * three baby-core BIR bodies yet, so going past dump would just
     * trip td_lower into refusing. This is the flag that lets us
     * stare at what fission produces while the BIR-splitting half
     * of the pass is still being written. */
    if (cfg->mode_tdf_fission) {
        int frc = td_fission_tensix(&g_tdf_mod);
        if (frc != BC_OK) return frc;
        /* Place channels into L1 so the dump shows real offsets
         * rather than zeros. Placement after fission and before
         * dump is the order multi-region lowering will want too:
         * the eventual RV32IM emitter reads l1_off from each
         * channel when materialising the CB descriptor pointers. */
        int prc = td_place_l1(&g_tdf_mod);
        if (prc != BC_OK) return prc;
        /* NoC orchestration fills in noc_id and length on every
         * RD and WR arc. Runs after placement because the length
         * computation reads from the channel tag the placer also
         * uses, and ordering the passes the same way means the
         * eventual RV32IM emitter sees them in the order it will
         * consume them. */
        int nrc = td_noc_orchestrate(&g_tdf_mod);
        if (nrc != BC_OK) return nrc;
        td_dump(&g_tdf_mod, stdout);
        return BC_OK;
    }

    if (cfg->mode_tdf) {
        td_dump(&g_tdf_mod, stdout);
    }
    tdrc = td_lower(&g_tdf_mod, &g_tdf_out);
    if (tdrc != BC_OK) return tdrc;
    bir = g_tdf_out.mods[0];   /* same pointer today, future-proof tomorrow */

    if (cfg->mode_ir) {
        bir_print_module(bir, stdout);
        printf("\n; %u functions, %u globals, %u instructions\n",
               bir->num_funcs, bir->num_globals, bir->num_insts);
    }

    int brc = be_run((struct bir_module *)bir, (struct be_cfg *)cfg);
    if (brc != BE_OK && rc == BC_OK) rc = BC_ERR_VERIFY;

    return rc;
}

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
        "Booth - CUDA Compiler\n"
        "Usage: %s [options] <file.cu>\n"
        "\n"
        "Options:\n"
        "  --lex         Tokenize and dump token stream\n"
        "  --parse       Parse and dump AST\n"
        "  --ir          Lower to BIR and print IR\n"
        "  --tdf         Dump the Tile DataFlow graph just before lowering\n"
        "  --tdf-fission Run the Tensix fission analysis on the BIR and dump\n"
        "                the three-region (RDR/CMP/WRT) graph it produces.\n"
        "                Preview mode: does not invoke any backend.\n"
        "  --rv-elf      Native RV32IM emission for the Tensix baby cores.\n"
        "                Writes an ELF that tt-metal can load. Integer kernels\n"
        "                only for now; soft-float runtime not yet linked.\n"
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
        "  --tt-chip C   Tenstorrent part: wormhole or blackhole "
        "(default blackhole)\n"
        "  AMD targets (default --gfx1100):\n"
        "    CDNA 2   --gfx90a (MI250)\n"
        "    CDNA 3   --gfx942 (MI300X)\n"
        "    RDNA 2   --gfx1030 --gfx1031 --gfx1032 --gfx1033 --gfx1034\n"
        "             --gfx1035 --gfx1036\n"
        "    RDNA 3   --gfx1100 --gfx1101 --gfx1102 --gfx1103\n"
        "    RDNA 3.5 --gfx1150 --gfx1151 --gfx1152 --gfx1153\n"
        "    RDNA 4   --gfx1200 --gfx1201\n"
        "  --no-graphcolor  Force linear scan register allocation\n"
        "  --ssa-ra         Divergence-aware SSA register allocation\n"
        "  --max-vgprs N    Cap VGPR count for regalloc (forces spills)\n"
        "  --tensix      Compile to TT-Metalium C++ (Tensix SFPU)\n"
        "  --nvidia-ptx  Compile to NVIDIA PTX (sm_89)\n"
        "  --hip         HIP frontend mode (predefines __HIPCC__ and platform macros;\n"
        "                auto-on for .hip files; combine with --amdgpu-bin or --nvidia-ptx)\n"
        "  --triton      Triton frontend mode (parses Python source). Pair with a target\n"
        "                backend (--cpu, --amdgpu-bin, --nvidia-ptx). tl.dot matmul runs.\n"
        "  --cpu         x86-64 host backend; emits a normal object you can link and run\n"
        "  --rv64        RV64IMFD backend; emits a Linux ELF object (run under qemu-riscv64)\n"
        "  --metal       Compile to Apple Metal Shading Language (stub)\n"
        "  --intel-spirv Compile to SPIR-V for Intel Arc Xe (stub)\n"
        "  --xe-lpg      Target Xe-LPG (Arc / integrated)\n"
        "  --xe-hpg      Target Xe-HPG (Alchemist, Battlemage) [default]\n"
        "  --xe-hpc      Target Xe-HPC (Ponte Vecchio)\n"
        "  --xe2         Target Xe2 (Lunar Lake, next-gen Arc)\n"
        "  -o <file>     Output file (for --amdgpu, --amdgpu-bin, --tensix, --nvidia-ptx,\n"
        "                --metal, --intel-spirv). --amdgpu writes to stdout without it.\n"
        "  --snap        AMD: write each kernel parameter's register value into a\n"
        "                host-visible buffer on entry, for the ABEND dump to read back\n"
        "  --bkhit       NVIDIA: add a __bkhit counter parameter each block atomically\n"
        "                increments, so you can see which blocks actually ran\n"
        "  --lang <file> Load translated error messages\n"
        "  --version     Print version and exit\n"
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
    int mode_tdf = 0;
    int mode_tdf_fission = 0;
    int mode_hip = 0;           /* HIP frontend: see HIP NOTES below */
    int mode_triton = 0;        /* Triton frontend: see TRITON NOTES below */
    int no_mem2reg = 0;
    int no_cfold = 0;
    int no_dce = 0;
    int no_sroa = 0;
    int no_sched = 0;
    int no_pp = 0;
    td_chip_t tt_chip = TD_CHIP_BH;

    /* Collect -I and -D options for preprocessor */
    const char *include_paths[PP_MAX_INCLUDE_PATHS];
    int num_include_paths = 0;
    const char *defines[128];
    int num_defines = 0;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--version") == 0) {
            printf("Booth %s\n", BC_VERSION_STRING);
            printf("From-scratch CUDA/HIP/Triton compiler.\n");
            return 0;
        }
        else if (strcmp(argv[i], "--lex") == 0)
            mode_lex = 1;
        else if (strcmp(argv[i], "--parse") == 0)
            mode_parse = 1;
        else if (strcmp(argv[i], "--sema") == 0)
            mode_sema = 1;
        else if (strcmp(argv[i], "--ir") == 0)
            mode_ir = 1;
        else if (strcmp(argv[i], "--tdf") == 0)
            mode_tdf = 1;
        else if (strcmp(argv[i], "--tdf-fission") == 0)
            mode_tdf_fission = 1;
        else if (strcmp(argv[i], "--pp") == 0)
            mode_pp = 1;
        else if (strcmp(argv[i], "--no-pp") == 0)
            no_pp = 1;
        /* CDNA 2 (GFX9) */
        /* CDNA 3 (GFX9.4.2) */
        /* RDNA 2 (GFX10.3) */
        /* RDNA 3 (GFX11) */
        /* RDNA 3.5 (GFX11.5) */
        /* RDNA 4 (GFX12) */
        else if (strcmp(argv[i], "--hip") == 0)
            mode_hip = 1;
        else if (strcmp(argv[i], "--triton") == 0)
            mode_triton = 1;
        else if (strcmp(argv[i], "--tt-chip") == 0 && i + 1 < argc) {
            if (td_pchip(argv[++i], &tt_chip) != BC_OK) {
                fprintf(stderr, "unknown Tenstorrent chip: %s "
                                "(want wormhole or blackhole)\n", argv[i]);
                return 1;
            }
        }
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
        else if (strcmp(argv[i], "--no-sroa") == 0)
            no_sroa = 1;
        else if (strcmp(argv[i], "--no-sched") == 0)
            no_sched = 1;
        else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            usage(argv[0]);
            return 0;
        } else if (argv[i][0] != '-')
            file = argv[i];
        else {
            /* Not one of the driver's, so offer it round the backend
             * registry before calling it unknown. Target selection and
             * every target-specific knob lives in the backend that owns
             * it, which is why main no longer has a --gfx list. */
            int used_next = 0;
            int taken = be_parse_flag(argv[i],
                                      (i + 1 < argc) ? argv[i + 1] : NULL,
                                      &used_next);
            if (taken < 0) return 1;
            if (taken > 0) { i += used_next; continue; }

            fprintf(stderr, "unknown option: %s\n", argv[i]);
            usage(argv[0]);
            return 1;
        }
    }

    if (!file) {
        usage(argv[0]);
        return 1;
    }

    td_setchip(tt_chip);

    /* One cascade for the C99 pipeline, each level meaning "this stage, or
     * anything downstream of it". Four independent lists is how --cpu and then
     * --tdf reached some gates and not others. Triton keeps its own list below;
     * it gates a different frontend and does not accept --rv-elf. */
    int want_bir  = mode_ir || mode_tdf || mode_tdf_fission ||
                    be_num_on() > 0u;
    int want_sema = mode_sema || want_bir;

    if (!mode_pp && !mode_lex && !mode_parse && !want_sema)
        mode_parse = 1;

    int want_ast = mode_parse || want_sema;

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
     * pipeline to drop their files into Booth unchanged. */
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

    /* One translation unit: BIR_CALL resolves a callee by index into funcs[].
       Only when the kernel wants float, or every integer kernel pays the text. */
    if (mode_rv_elf && strstr(source_buf, "float") != NULL) {
        /* A quoted include resolves against the user's kernel now, not us. */
        if (num_include_paths + 2 <= PP_MAX_INCLUDE_PATHS) {
            include_paths[num_include_paths++] = "runtime";
            include_paths[num_include_paths++] = "../runtime";
        }
        static const char *const rt[] = {
            "runtime/soft_fp.c",
            "../runtime/soft_fp.c",
        };
        uint32_t rt_len = 0;
        uint32_t i = 0;
        for (; i < sizeof(rt) / sizeof(rt[0]); i++) {
            if (read_file(rt[i], source_buf + src_len,
                          BC_MAX_SOURCE - src_len, &rt_len) == BC_OK) break;
        }
        if (i == sizeof(rt) / sizeof(rt[0])) {
            fprintf(stderr,
                    "error: --rv-elf needs runtime/soft_fp.c and could not "
                    "find it; run from the repo root\n");
            return 1;
        }
        src_len += rt_len;
    }

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
        bc_diag(file, source_buf, tn_lex_state.errors, tn_lex_state.num_errors);
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
        /* One list, used by both the gate and the sema/lower decision below.
         * It was two, they disagreed, and --cpu fell down the gap. */
        int want_backend = be_num_on() > 0u;
        if (mode_parse || mode_sema || mode_ir || want_backend) {
            tn_parse_t *tnp = (tn_parse_t *)malloc(sizeof(tn_parse_t));
            if (!tnp) {
                fprintf(stderr, "error: failed to allocate Triton parser\n");
                return 1;
            }
            tn_parse_init(tnp, &tn_lex_state);
            int prc = tn_parse(tnp);
            bc_diag(file, source_buf, tnp->errors, tnp->num_errors);
            if (mode_sema || mode_ir || want_backend) {
                tn_sema_t *tns = (tn_sema_t *)malloc(sizeof(tn_sema_t));
                if (!tns) {
                    fprintf(stderr, "error: failed to allocate Triton sema\n");
                    free(tnp);
                    return 1;
                }
                tn_sema_init(tns, tnp);
                int src_code = tn_sema(tns);
                bc_diag(file, source_buf, tns->errors, tns->num_errors);
                if (mode_ir || want_backend) {
                    tn_lower_t *tnl = (tn_lower_t *)malloc(sizeof(tn_lower_t));
                    if (!tnl) {
                        fprintf(stderr, "error: failed to allocate Triton lower\n");
                        free(tns); free(tnp);
                        return 1;
                    }
                    bir_module = (bir_module_t *)malloc(sizeof(bir_module_t));
                    if (!bir_module) {
                        fprintf(stderr, "error: failed to allocate BIR module\n");
                        free(tnl); free(tns); free(tnp);
                        return 1;
                    }
                    tn_lower_init(tnl, tnp, tns, bir_module);
                    int lrc = tn_lower(tnl);
                    bc_diag(file, source_buf, tnl->errors, tnl->num_errors);

                    int brc = BC_OK;
                    if (lrc == BC_OK && (mode_ir || want_backend)) {
                        backend_cfg_t cfg = {0};
                        cfg.no_mem2reg = no_mem2reg;
                        cfg.no_cfold   = no_cfold;
                        cfg.no_dce     = no_dce;
                        cfg.no_sched   = no_sched;
                        cfg.no_sroa    = no_sroa;
                        cfg.mode_ir    = mode_ir;
                        cfg.mode_tdf   = mode_tdf;
                        cfg.mode_tdf_fission = mode_tdf_fission;
                        cfg.output_file     = output_file;
                        brc = run_bir_backends(bir_module, &cfg);
                    }

                    free(tnl); free(tns); free(tnp); free(bir_module);
                    return (prc != BC_OK || src_code != BC_OK ||
                            lrc != BC_OK || brc != BC_OK) ? 1 : 0;
                }
                tn_sema_dump(tns, stdout);
                free(tns);
                free(tnp);
                return (prc != BC_OK || src_code != BC_OK) ? 1 : 0;
            }
            tn_ast_dump(tnp, stdout);
            free(tnp);
            return prc != BC_OK ? 1 : 0;
        }
        /* Anything beyond the supported modes falls through here. Reaching
         * this point means we did not do what was asked, so never report 0
         * however well the lexer did. */
        fprintf(stderr,
            "triton: use --lex / --parse / --sema / --ir, or pair\n"
            "        --triton with a backend (--amdgpu-bin / --nvidia-ptx /\n"
            "        --tensix / --metal / --intel-spirv / --cpu / --rv64).\n");
        return 1;
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
            /* Which platform HIP thinks it is compiling for follows the
             * selected target, so ask the registry rather than keep a
             * copy of the mode flag here. */
            const be_desc_t *hb = be_active();
            if (hb != NULL && strcmp(hb->name, "nvptx") == 0)
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

        bc_diag(file, source_buf, pp->errors, pp->num_errors);

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

    bc_diag(file, lex_src, L.errors, L.num_errors);

    if (mode_lex) {
        dump_tokens(&L);
        printf("\n%u tokens, %d error(s)\n", L.num_tokens, L.num_errors);
    }

    if (want_ast) {
        parser_t P;
        parser_init(&P, token_buf, L.num_tokens, lex_src,
                    node_buf, BC_MAX_NODES);
        uint32_t root = parser_parse(&P);

        bc_diag(file, lex_src, P.errors, P.num_errors);

        if (mode_parse) {
            ast_dump(&P, root, 0);
            printf("\n%u nodes, %d parse error(s)\n",
                   P.num_nodes, P.num_errors);
        }

        /* Semantic analysis */
        sema_ctx_t *sema_ctx = NULL;
        if (want_sema && P.num_errors == 0)
        {
            sema_ctx = (sema_ctx_t *)malloc(sizeof(sema_ctx_t));
            if (!sema_ctx) {
                fprintf(stderr, "error: failed to allocate sema context\n");
                return 1;
            }
            sema_init(sema_ctx, &P, root, (int)be_warp_size());
            sema_check(sema_ctx, root);

            bc_diag(file, lex_src, sema_ctx->errors, sema_ctx->num_errors);

            if (mode_sema) {
                sema_dump(sema_ctx, root);
                int sema_rc = sema_ctx->num_errors > 0 ? 1 : 0;
                free(sema_ctx);
                return sema_rc;
            }

            /* Only --sema used to act on these. Every other mode printed the
             * errors, carried on into codegen, wrote an output file and exited
             * zero, so a build system saw a clean compile and a kernel that
             * had been lowered from source we had already rejected. */
            if (sema_ctx->num_errors > 0) {
                free(sema_ctx);
                return 1;
            }
        }

        if (want_bir && P.num_errors == 0) {
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
                backend_cfg_t cfg = {0};
                cfg.no_mem2reg = no_mem2reg;
                cfg.no_cfold   = no_cfold;
                cfg.no_dce     = no_dce;
                cfg.no_sched   = no_sched;
                cfg.no_sroa    = no_sroa;
                cfg.mode_ir    = mode_ir;
                cfg.mode_tdf   = mode_tdf;
                cfg.mode_tdf_fission = mode_tdf_fission;
                cfg.output_file     = output_file;
                int brc = run_bir_backends(bir_module, &cfg);
                if (brc != BC_OK) rc = brc;
            }
            free(bir_module);
            if (lrc != BC_OK) rc = lrc;
        }

        if (sema_ctx) free(sema_ctx);
        if (P.num_errors > 0) rc = BC_ERR_PARSE;
    }

    return rc != BC_OK ? 1 : 0;
}
