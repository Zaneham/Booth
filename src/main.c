#include "preproc.h"
#include "lexer.h"
#include "parser.h"
#include "sema.h"
#include "bir_lower.h"
#include "bir_sroa.h"
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
#include "tdf.h"
#include "rv_buf.h"
#include "rv_elf.h"
#include "rv_isel.h"
#include "cpu.h"
#include "rv64.h"
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

typedef struct {
    int             no_mem2reg, no_cfold, no_dce, no_sched, no_sroa;
    int             mode_ir, mode_tdf, mode_tdf_fission;
    int             mode_amdgpu, mode_amdgpu_bin;
    int             mode_tensix, mode_nvidia, nv_bkhit;
    int             mode_metal, mode_intel, mode_rv_elf, mode_cpu, mode_rv64;
    amd_target_t    amd_target;
    uint32_t        amd_elfm;
    const char     *amd_chip;
    int             snap_mode;
    intel_target_t  intel_target;
    const char     *output_file;
} backend_cfg_t;

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
    if (cfg->mode_tensix) return TD_TGT_TENSIX;
    if (cfg->mode_nvidia) return TD_TGT_NVIDIA;
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

    /* Optimisation passes: same shape regardless of frontend. */
    if (!cfg->no_sroa)    bir_sroa(bir);
    if (!cfg->no_mem2reg) bir_mem2reg(bir);
    if (!cfg->no_cfold)   bir_cfold(bir);
    if (!cfg->no_dce)     bir_dce(bir);

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

    if (cfg->mode_amdgpu || cfg->mode_amdgpu_bin) {
        amd_module_t *amd = (amd_module_t *)malloc(sizeof(amd_module_t));
        if (!amd) {
            fprintf(stderr, "error: failed to allocate AMD module\n");
            return BC_ERR_IO;
        }
        amd->target = cfg->amd_target;
        amd->elf_mach = cfg->amd_elfm;
        amd->snap_mode = (uint8_t)cfg->snap_mode;
        snprintf(amd->chip_name, sizeof(amd->chip_name), "%s", cfg->amd_chip);
        int arc = amdgpu_compile(bir, amd);
        if (arc == BC_OK) {
            vfy_res_t v1 = bc_vfy(amd, VFY_ISEL);
            if (v1.errs) {
                fprintf(stderr, "verify: %u error(s) after isel\n", v1.errs);
                arc = BC_ERR_VERIFY;
            }
        }
        if (arc == BC_OK) {
            if (!cfg->no_sched) amdgpu_sched(amd);
            amdgpu_regalloc(amd);
            vfy_res_t v2 = bc_vfy(amd, VFY_RA);
            if (v2.errs) {
                fprintf(stderr, "verify: %u error(s) after regalloc\n", v2.errs);
                arc = BC_ERR_VERIFY;
            }
        }
        if (arc == BC_OK) {
            if (cfg->mode_amdgpu_bin) {
                amdgpu_emit_elf(amd,
                    cfg->output_file ? cfg->output_file : "a.hsaco");
            } else {
                FILE *out = stdout;
                if (cfg->output_file) {
                    out = fopen(cfg->output_file, "w");
                    if (!out) {
                        fprintf(stderr, "error: could not open output file %s\n", cfg->output_file);
                        out = stdout;
                    }
                }
                amdgpu_emit_asm(amd, out);
                if (out != stdout) {
                    fclose(out);
                }
            }
        } else {
            if (arc != BC_ERR_VERIFY)
                fprintf(stderr, "error: AMDGPU compilation failed\n");
            rc = arc;
        }
        free(amd);
    }

    if (cfg->mode_nvidia) {
        nv_module_t *nvm = (nv_module_t *)malloc(sizeof(nv_module_t));
        if (!nvm) {
            fprintf(stderr, "error: failed to allocate NVIDIA module\n");
            return BC_ERR_IO;
        }
        int nrc = nv_compile(bir, nvm);
        if (nrc == BC_OK) {
            nvm->bkhit = (uint8_t)cfg->nv_bkhit;
            nv_emit_ptx(nvm, cfg->output_file ? cfg->output_file : "a.ptx");
        } else {
            fprintf(stderr, "error: NVIDIA PTX compilation failed\n");
            rc = nrc;
        }
        free(nvm);
    }

    if (cfg->mode_metal) {
        metal_module_t *mm = (metal_module_t *)malloc(sizeof(metal_module_t));
        if (!mm) {
            fprintf(stderr, "error: failed to allocate Metal module\n");
            return BC_ERR_IO;
        }
        int mrc = metal_compile(bir, mm);
        if (mrc == BC_OK) {
            metal_emit_msl(mm, cfg->output_file ? cfg->output_file : "a.metal");
        } else {
            fprintf(stderr, "error: Metal backend compilation failed\n");
            rc = mrc;
        }
        free(mm);
    }

    if (cfg->mode_intel) {
        intel_module_t *im = (intel_module_t *)malloc(sizeof(intel_module_t));
        if (!im) {
            fprintf(stderr, "error: failed to allocate Intel module\n");
            return BC_ERR_IO;
        }
        int irc = intel_compile(bir, im, cfg->intel_target);
        if (irc == BC_OK) {
            intel_emit_spirv(im, cfg->output_file ? cfg->output_file : "a.spv");
        } else {
            fprintf(stderr,
                "error: Intel SPIR-V backend not yet a working compiler\n");
            rc = irc;
        }
        free(im);
    }

    /* Native RV32IM emission for the baby cores. Picks the first
     * function in the BIR module (which is the first CUDA kernel
     * defined in the source) and runs it through the bring-up isel
     * into an ELF that the tt-metal host loader can drop onto a
     * baby core. Soft-float not yet linked in; integer kernels
     * only for now. */
    if (cfg->mode_cpu) {
        static cpu_mod_t cm;
        if (bir->num_funcs == 0u) { fprintf(stderr,"error: no functions\n"); return BC_ERR_TDF; }
        cpu_init(&cm, bir);
        cpu_emit(&cm);
        const char *path = cfg->output_file ? cfg->output_file : "a.o";
        if (cpu_elf(&cm, path) != 0) { fprintf(stderr,"cpu: elf write failed\n"); return BC_ERR_IO; }
        fprintf(stderr, "wrote %s (%u bytes x86-64)\n", path, cm.codelen);
    }

    if (cfg->mode_rv64) {
        static rv64_mod_t vm;
        if (bir->num_funcs == 0u) { fprintf(stderr,"error: no functions\n"); return BC_ERR_TDF; }
        rv64_init(&vm, bir);
        rv64_emit(&vm);
        const char *path = cfg->output_file ? cfg->output_file : "a.o";
        if (rv64_elf(&vm, path) != 0) { fprintf(stderr,"rv64: elf write failed\n"); return BC_ERR_IO; }
        fprintf(stderr, "wrote %s (%u bytes RV64)\n", path, vm.codelen);
    }

    if (cfg->mode_rv_elf) {
        static rv_buf_t rv_code;
        rv_buf_init(&rv_code);
        if (bir->num_funcs == 0u) {
            fprintf(stderr, "error: BIR module has no functions\n");
            return BC_ERR_TDF;
        }
        int irc = rv_isel_func(bir, 0u, &rv_code);
        if (irc != BC_OK) return irc;
        const char *path = cfg->output_file ? cfg->output_file : "a.elf";
        int erc = rv_elf_write(&rv_code, path);
        if (erc != BC_OK) return erc;
        fprintf(stderr, "wrote %s (%u bytes code, %u instructions)\n",
                path, rv_buf_pos_bytes(&rv_code),
                rv_buf_n_words(&rv_code));
    }

    /* Tensix needs additional reader/writer/host emission besides
     * the compute kernel, which is why it lives slightly off the
     * shared shape. */
    if (cfg->mode_tensix) {
        tt_module_t *ttm = (tt_module_t *)malloc(sizeof(tt_module_t));
        if (!ttm) {
            fprintf(stderr, "error: failed to allocate Tensix module\n");
            return BC_ERR_IO;
        }
        int trc = tensix_compile(bir, ttm);
        if (trc == BC_OK) {
            tensix_coarsen(ttm);
            tensix_regalloc(ttm);
            const char *compute_path =
                cfg->output_file ? cfg->output_file : "a_compute.cpp";
            tensix_analyze_datamov(bir, ttm, &ttm->dmov);
            tensix_emit_metalium(ttm, compute_path);
            char host_path[BC_MAX_PATH];
            char reader_path[BC_MAX_PATH];
            char writer_path[BC_MAX_PATH];
            const char *stem = strstr(compute_path, "_compute");
            int pfx;
            if (stem) pfx = (int)(stem - compute_path);
            else {
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
                                  reader_path, compute_path, writer_path);
        } else {
            fprintf(stderr, "error: Tensix compilation failed\n");
            rc = trc;
        }
        free(ttm);
    }

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
        "BarraCUDA - CUDA Compiler\n"
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
        "  -o <file>     Output file (for --amdgpu-bin, --tensix, --nvidia-ptx, --metal, --intel-spirv)\n"
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
    int mode_amdgpu = 0;
    int mode_amdgpu_bin = 0;
    int mode_cpu = 0;
    int mode_rv64 = 0;
    int mode_tensix = 0;
    int mode_nvidia = 0;
    int mode_metal = 0;
    int mode_intel = 0;
    int mode_rv_elf = 0;
    int mode_hip = 0;           /* HIP frontend: see HIP NOTES below */
    int mode_triton = 0;        /* Triton frontend: see TRITON NOTES below */
    intel_target_t intel_target = INTEL_TARGET_XE_HPG;
    int nv_bkhit = 0;
    int no_mem2reg = 0;
    int no_cfold = 0;
    int no_dce = 0;
    int no_sroa = 0;
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
        if (strcmp(argv[i], "--version") == 0) {
            printf("BarraCUDA %s\n", BC_VERSION_STRING);
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
        else if (strcmp(argv[i], "--rv-elf") == 0)
            mode_rv_elf = 1;
        else if (strcmp(argv[i], "--cpu") == 0)
            mode_cpu = 1;
        else if (strcmp(argv[i], "--rv64") == 0)
            mode_rv64 = 1;
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
        else if (strcmp(argv[i], "--no-sroa") == 0)
            no_sroa = 1;
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
        !mode_metal && !mode_intel && !mode_rv_elf && !mode_cpu && !mode_rv64)
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
        if (mode_parse || mode_sema || mode_ir ||
            mode_amdgpu || mode_amdgpu_bin || mode_tensix ||
            mode_nvidia || mode_metal || mode_intel) {
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
            int want_backend = mode_amdgpu || mode_amdgpu_bin ||
                               mode_tensix || mode_nvidia ||
                               mode_metal || mode_intel || mode_cpu || mode_rv64;
            if (mode_sema || mode_ir || want_backend) {
                tn_sema_t *tns = (tn_sema_t *)malloc(sizeof(tn_sema_t));
                if (!tns) {
                    fprintf(stderr, "error: failed to allocate Triton sema\n");
                    free(tnp);
                    return 1;
                }
                tn_sema_init(tns, tnp);
                int src_code = tn_sema(tns);
                for (int i = 0; i < tns->num_errors; i++) {
                    fprintf(stderr, "%s:%u:%u: E%03u: %s\n",
                            file,
                            tns->errors[i].loc.line,
                            tns->errors[i].loc.col,
                            tns->errors[i].eid,
                            tns->errors[i].msg);
                }
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
                    for (int i = 0; i < tnl->num_errors; i++) {
                        fprintf(stderr, "%s:%u:%u: E%03u: %s\n",
                                file,
                                tnl->errors[i].loc.line,
                                tnl->errors[i].loc.col,
                                tnl->errors[i].eid,
                                tnl->errors[i].msg);
                    }

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
                        cfg.mode_rv_elf      = mode_rv_elf; cfg.mode_cpu = mode_cpu; cfg.mode_rv64 = mode_rv64;
                        cfg.mode_amdgpu     = mode_amdgpu;
                        cfg.mode_amdgpu_bin = mode_amdgpu_bin;
                        cfg.mode_tensix     = mode_tensix;
                        cfg.mode_nvidia     = mode_nvidia;
                        cfg.nv_bkhit        = nv_bkhit;
                        cfg.mode_metal      = mode_metal;
                        cfg.mode_intel      = mode_intel;
                        cfg.amd_target      = amd_target;
                        cfg.amd_elfm        = amd_elfm;
                        cfg.amd_chip        = amd_chip;
                        cfg.snap_mode       = snap_mode;
                        cfg.intel_target    = intel_target;
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
        /* Anything beyond the supported modes falls through here. */
        fprintf(stderr,
            "triton: use --lex / --parse / --sema / --ir, or pair\n"
            "        --triton with a backend (--amdgpu-bin /\n"
            "        --nvidia-ptx / --tensix / --metal / --intel-spirv).\n");
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
        mode_tensix || mode_nvidia || mode_metal || mode_intel || mode_rv_elf || mode_cpu || mode_rv64) {
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
             mode_tensix || mode_nvidia || mode_metal || mode_intel ||
             mode_cpu || mode_rv64 || mode_rv_elf) &&
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
             mode_nvidia || mode_metal || mode_intel || mode_rv_elf || mode_cpu || mode_rv64) &&
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
                backend_cfg_t cfg = {0};
                cfg.no_mem2reg = no_mem2reg;
                cfg.no_cfold   = no_cfold;
                cfg.no_dce     = no_dce;
                cfg.no_sched   = no_sched;
                cfg.no_sroa    = no_sroa;
                cfg.mode_ir    = mode_ir;
                cfg.mode_tdf   = mode_tdf;
                cfg.mode_tdf_fission = mode_tdf_fission;
                cfg.mode_rv_elf = mode_rv_elf; cfg.mode_cpu = mode_cpu; cfg.mode_rv64 = mode_rv64;
                cfg.mode_amdgpu     = mode_amdgpu;
                cfg.mode_amdgpu_bin = mode_amdgpu_bin;
                cfg.mode_tensix     = mode_tensix;
                cfg.mode_nvidia     = mode_nvidia;
                cfg.nv_bkhit        = nv_bkhit;
                cfg.mode_metal      = mode_metal;
                cfg.mode_intel      = mode_intel;
                cfg.amd_target      = amd_target;
                cfg.amd_elfm        = amd_elfm;
                cfg.amd_chip        = amd_chip;
                cfg.snap_mode       = snap_mode;
                cfg.intel_target    = intel_target;
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
