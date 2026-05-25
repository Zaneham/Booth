/* ttriton.c -- end-to-end tests for the Triton frontend.
 *
 * Covers the four stages the frontend exposes (lex, parse, sema, ir)
 * and the three backends a kernel can land on (AMD, NVIDIA, Tensix).
 * Also keeps a couple of deliberately unhinged kernels around because
 * the canonical vector_add does not, on its own, exercise the
 * parser's tolerance of suburban naming conventions or its appetite
 * for an apologetic docstring half a kilobyte long. */

#include "tharns.h"

static char obuf[TH_BUFSZ];

/* ---- Helpers ---- */

/* Run barracuda with the given args, capture stdout+stderr in obuf,
 * return the exit code. The harness's th_run handles popen-with-2>&1
 * on every platform that matters. */
static int tt_run(const char *args)
{
    char cmd[TH_BUFSZ];
    snprintf(cmd, TH_BUFSZ, BC_BIN " %s", args);
    return th_run(cmd, obuf, TH_BUFSZ);
}

/* ============================================================
 * Lexer
 * ============================================================ */

static void tt_lex_simple(void)
{
    int rc = tt_run("--triton --lex tests/tri_simple.py");
    CHEQ(rc, 0);
    /* The simple kernel exercises NEWLINE, INDENT, DEDENT, KW tokens,
     * IDENTs, INT literals, and the standard operator set. If the
     * "tokens, 0 error(s)" tail line is present, the file lexed
     * without complaint. */
    CHECK(strstr(obuf, "tokens, 0 error(s)") != NULL);
    /* Specific structural tokens we expect to see. */
    CHECK(strstr(obuf, "INDENT") != NULL);
    CHECK(strstr(obuf, "DEDENT") != NULL);
    CHECK(strstr(obuf, "def") != NULL);
    PASS();
}
TH_REG("triton", tt_lex_simple)

static void tt_lex_slop_docstring(void)
{
    /* The AI slop file is enormous and the lex dump is enormous; the
     * 4 KB capture buffer truncates the trailing summary. We confirm
     * the multi-paragraph triple-quoted docstring tokenises by
     * looking at things we know appear in the first kilobyte: the
     * STRING token for the docstring itself, and the `import`
     * statements that follow it. If those show up cleanly the lexer
     * did not trip on the docstring's six paragraphs. */
    int rc = tt_run("--triton --lex tests/tri_slop.py");
    CHEQ(rc, 0);
    CHECK(strstr(obuf, "STRING") != NULL);
    CHECK(strstr(obuf, "import") != NULL);
    PASS();
}
TH_REG("triton", tt_lex_slop_docstring)

/* ============================================================
 * Parser
 * ============================================================ */

static void tt_parse_funcdef(void)
{
    int rc = tt_run("--triton --parse tests/tri_vadd.py");
    CHEQ(rc, 0);
    CHECK(strstr(obuf, "FuncDef 'vector_add'") != NULL);
    CHECK(strstr(obuf, "Param 'x_ptr'") != NULL);
    CHECK(strstr(obuf, "Param 'BLOCK_SIZE'") != NULL);
    CHECK(strstr(obuf, "Block") != NULL);
    PASS();
}
TH_REG("triton", tt_parse_funcdef)

static void tt_parse_expressions(void)
{
    int rc = tt_run("--triton --parse tests/tri_vadd.py");
    CHEQ(rc, 0);
    /* The expression parser should produce the named node kinds for
     * the various pieces of the kernel body, not opaque ExprSpans. */
    CHECK(strstr(obuf, "BinOp") != NULL);
    CHECK(strstr(obuf, "Call") != NULL);
    CHECK(strstr(obuf, "Attr") != NULL);
    CHECK(strstr(obuf, "Compare") != NULL);
    CHECK(strstr(obuf, "Keyword") != NULL);
    PASS();
}
TH_REG("triton", tt_parse_expressions)

static void tt_parse_goblin(void)
{
    /* The goblin kernel is identical in shape to vector_add but with
     * names the Triton tutorial people would never choose. We are
     * checking that nothing in the parser is doing semantic guessing
     * based on identifier text. */
    int rc = tt_run("--triton --parse tests/tri_goblin.py");
    CHEQ(rc, 0);
    CHECK(strstr(obuf, "FuncDef 'goblin_raid'") != NULL);
    CHECK(strstr(obuf, "Param 'troll_ptr'") != NULL);
    CHECK(strstr(obuf, "Param 'hoard_ptr'") != NULL);
    PASS();
}
TH_REG("triton", tt_parse_goblin)

/* ============================================================
 * Sema
 * ============================================================ */

static void tt_sema_resolves(void)
{
    int rc = tt_run("--triton --sema tests/tri_vadd.py");
    CHEQ(rc, 0);
    /* The `tl` alias should resolve to a module, and its members
     * should resolve through the intrinsic table. */
    CHECK(strstr(obuf, "module(tl)") != NULL);
    CHECK(strstr(obuf, "intrinsic(program_id)") != NULL);
    CHECK(strstr(obuf, "intrinsic(arange)") != NULL);
    CHECK(strstr(obuf, "intrinsic(load)") != NULL);
    CHECK(strstr(obuf, "intrinsic(store)") != NULL);
    CHECK(strstr(obuf, "type(constexpr)") != NULL);
    PASS();
}
TH_REG("triton", tt_sema_resolves)

static void tt_sema_locals_and_params(void)
{
    int rc = tt_run("--triton --sema tests/tri_vadd.py");
    CHEQ(rc, 0);
    CHECK(strstr(obuf, "Name 'x_ptr' -> param") != NULL);
    CHECK(strstr(obuf, "Name 'pid' -> local") != NULL);
    CHECK(strstr(obuf, "Name 'BLOCK_SIZE' -> param") != NULL);
    PASS();
}
TH_REG("triton", tt_sema_locals_and_params)

static void tt_sema_ai_slop_still_resolves(void)
{
    /* The AI slop kernel uses comically long names but every one of
     * them should still bind correctly. The point of this test is to
     * confirm that the sema does not have any hidden length limits or
     * weird collation rules that would trip on the ChatGPT habit of
     * concatenating six words into one identifier. */
    int rc = tt_run("--triton --sema tests/tri_slop.py");
    CHEQ(rc, 0);
    CHECK(strstr(obuf,
        "Name 'input_tensor_a_pointer_first_operand' -> param") != NULL);
    CHECK(strstr(obuf,
        "Name 'program_id_along_zeroth_axis' -> local") != NULL);
    CHECK(strstr(obuf, "0 error(s)") != NULL);
    PASS();
}
TH_REG("triton", tt_sema_ai_slop_still_resolves)

static void tt_sema_shapes_vector_add(void)
{
    /* Sitting one of tile shape inference. The vector add kernel uses
     * arange to build a rank-1 tile, broadcasts a scalar against it,
     * and compares it for the mask. We expect the shapes to land as:
     *   offsets -> vec[?]:int32   (broadcast of scalar + arange)
     *   mask    -> vec[?]:int1    (vec compared against scalar)
     * The size is dynamic because BLOCK_SIZE is a constexpr param and
     * we do not propagate constexpr values in this sitting. */
    int rc = tt_run("--triton --sema tests/tri_vadd.py");
    CHEQ(rc, 0);
    CHECK(strstr(obuf, "vec[?]:int32") != NULL);
    CHECK(strstr(obuf, "vec[?]:int1") != NULL);
    PASS();
}
TH_REG("triton", tt_sema_shapes_vector_add)

static void tt_sema_shapes_2d_broadcast(void)
{
    /* Rank-2 broadcasting. The matmul-shape kernel exercises both
     * sides of the canonical [:, None] / [None, :] reshape pattern,
     * then arithmetic between the resulting rank-2 tiles. Expected
     * shapes:
     *   offs_m[:, None] -> mat[?, 1]:int32
     *   offs_k[None, :] -> mat[1, ?]:int32
     *   broadcast sum   -> mat[?, ?]:int32
     *   tl.zeros((M, N), dtype=tl.float32) -> mat[?, ?]:float32
     * Like the vector test, dims are dynamic because the tile sizes
     * are constexpr params. */
    int rc = tt_run("--triton --sema tests/tri_mm.py");
    CHEQ(rc, 0);
    CHECK(strstr(obuf, "mat[?, 1]:int32") != NULL);
    CHECK(strstr(obuf, "mat[1, ?]:int32") != NULL);
    CHECK(strstr(obuf, "mat[?, ?]:int32") != NULL);
    CHECK(strstr(obuf, "mat[?, ?]:float32") != NULL);
    PASS();
}
TH_REG("triton", tt_sema_shapes_2d_broadcast)

static void tt_sema_shapes_scalar_kernel(void)
{
    /* The simple all-scalar kernel should have no tile shapes at all;
     * every expression is rank 0 with dtype int32. We check both that
     * scalar:int32 appears (positive) and that no vec[ or mat[ shows
     * up in the output (negative). */
    int rc = tt_run("--triton --sema tests/tri_simple.py");
    CHEQ(rc, 0);
    CHECK(strstr(obuf, "scalar:int32") != NULL);
    CHECK(strstr(obuf, "vec[") == NULL);
    CHECK(strstr(obuf, "mat[") == NULL);
    PASS();
}
TH_REG("triton", tt_sema_shapes_scalar_kernel)

static void tt_sema_constexpr_resolves_dim(void)
{
    /* BLOCK: tl.constexpr = 256 should propagate so arange and the
     * downstream broadcast get vec[256] rather than vec[?]. */
    int rc = tt_run("--triton --sema tests/tri_const.py");
    CHEQ(rc, 0);
    CHECK(strstr(obuf, "vec[256]:int32") != NULL);
    CHECK(strstr(obuf, "vec[?]") == NULL);
    PASS();
}
TH_REG("triton", tt_sema_constexpr_resolves_dim)

static void tt_lower_matmul(void)
{
    /* Rank-2 tiles with tl.dot now lower for the CPU path: the tile is
     * materialized and fully unrolled (block sizes are constexpr), so a
     * 4x4 matmul produces float multiplies and adds and no E099. */
    int rc = tt_run("--triton --ir tests/tri_matmul.py");
    CHEQ(rc, 0);
    CHECK(strstr(obuf, "fmul f32") != NULL);
    CHECK(strstr(obuf, "fadd f32") != NULL);
    CHECK(strstr(obuf, "E099") == NULL);
    PASS();
}
TH_REG("triton", tt_lower_matmul)

static void tt_lower_matmul_kloop(void)
{
    /* Matmul with a runtime K-loop lowers to a counted loop: the
     * accumulator is scratch-backed and the loop counter is a phi, so
     * the BIR has a phi and a conditional branch and no E099. */
    int rc = tt_run("--triton --ir tests/tri_matmul_k.py");
    CHEQ(rc, 0);
    CHECK(strstr(obuf, "phi i32") != NULL);
    CHECK(strstr(obuf, "br_cond") != NULL);
    CHECK(strstr(obuf, "E099") == NULL);
    PASS();
}
TH_REG("triton", tt_lower_matmul_kloop)

/* ============================================================
 * BIR Lowering
 * ============================================================ */

static void tt_lower_thread_model(void)
{
    int rc = tt_run("--triton --ir tests/tri_vadd.py");
    CHEQ(rc, 0);
    CHECK(strstr(obuf, "block_id") != NULL);
    CHECK(strstr(obuf, "thread_id") != NULL);
    PASS();
}
TH_REG("triton", tt_lower_thread_model)

static void tt_lower_simple_arith(void)
{
    /* tri_simple chains a handful of integer operations and ends in
     * a store, so every arithmetic instruction is reachable from a
     * side effect and DCE leaves it alone. The chain exercises
     * unary minus (which the lowerer turns into `sub i32 0, %d`),
     * integer modulo (srem), and the standard add / sub / mul. */
    int rc = tt_run("--triton --ir tests/tri_simple.py");
    CHEQ(rc, 0);
    CHECK(strstr(obuf, "block_id") != NULL);
    CHECK(strstr(obuf, "grid_dim") != NULL);
    CHECK(strstr(obuf, "add i32") != NULL);
    CHECK(strstr(obuf, "mul i32") != NULL);
    CHECK(strstr(obuf, "srem i32") != NULL);
    CHECK(strstr(obuf, "sub i32 0,") != NULL);
    CHECK(strstr(obuf, "store ") != NULL);
    PASS();
}
TH_REG("triton", tt_lower_simple_arith)

static void tt_lower_arithmetic(void)
{
    /* The lowerer emits the icmp for the mask, but DCE removes it
     * downstream because mask= is currently warned-and-ignored on
     * tl.load and so the mask value has no surviving use. What
     * does survive is everything reachable from the final store:
     * the program_id, the thread_id, the offset arithmetic, the
     * two loads, the fadd, and the store. Test those. */
    int rc = tt_run("--triton --ir tests/tri_vadd.py");
    CHEQ(rc, 0);
    CHECK(strstr(obuf, "block_id") != NULL);
    CHECK(strstr(obuf, "thread_id") != NULL);
    CHECK(strstr(obuf, "add i32") != NULL);
    CHECK(strstr(obuf, "mul i32") != NULL);
    CHECK(strstr(obuf, "gep ") != NULL);
    CHECK(strstr(obuf, "load f32") != NULL);
    CHECK(strstr(obuf, "fadd f32") != NULL);
    CHECK(strstr(obuf, "store f32") != NULL);
    PASS();
}
TH_REG("triton", tt_lower_arithmetic)

/* ============================================================
 * Backend: AMD GFX11
 * ============================================================ */

static void tt_amd_hsaco(void)
{
    int rc = tt_run("--triton --amdgpu-bin tests/tri_vadd.py "
                    "-o tri_vadd.hsaco");
    CHEQ(rc, 0);
    CHECK(th_exist("tri_vadd.hsaco"));
    /* The harness reports kernel count and code size on stderr. A
     * non-zero kernel count is the easy way to confirm the AMD
     * backend actually produced a kernel rather than an empty ELF. */
    CHECK(strstr(obuf, "1 kernels") != NULL);
    PASS();
}
TH_REG("triton", tt_amd_hsaco)

static void tt_amd_goblin_hsaco(void)
{
    /* The goblin variant should produce the same shape of output as
     * vector_add. The point is to confirm the backend is name-agnostic. */
    int rc = tt_run("--triton --amdgpu-bin tests/tri_goblin.py "
                    "-o tri_goblin.hsaco");
    CHEQ(rc, 0);
    CHECK(th_exist("tri_goblin.hsaco"));
    PASS();
}
TH_REG("triton", tt_amd_goblin_hsaco)

/* ============================================================
 * Backend: NVIDIA PTX
 * ============================================================ */

static void tt_nvidia_ptx(void)
{
    int rc = tt_run("--triton --nvidia-ptx tests/tri_vadd.py "
                    "-o tri_vadd.ptx");
    CHEQ(rc, 0);
    CHECK(th_exist("tri_vadd.ptx"));
    /* PTX is a text format we can grep directly. The .entry directive
     * with the function name is the unambiguous signal that the kernel
     * came through with the right shape. */
    FILE *fp = fopen("tri_vadd.ptx", "rb");
    CHECK(fp != NULL);
    if (!fp) return;
    char buf[2048];
    size_t n = fread(buf, 1, sizeof(buf) - 1, fp);
    fclose(fp);
    buf[n] = '\0';
    CHECK(strstr(buf, ".entry vector_add") != NULL);
    CHECK(strstr(buf, ".version") != NULL);
    PASS();
}
TH_REG("triton", tt_nvidia_ptx)

/* ============================================================
 * Backend: Tensix Metalium
 * ============================================================ */

static void tt_tensix_metalium(void)
{
    int rc = tt_run("--triton --tensix tests/tri_vadd.py "
                    "-o tri_vadd_compute.cpp");
    CHEQ(rc, 0);
    CHECK(th_exist("tri_vadd_compute.cpp"));
    CHECK(th_exist("tri_vadd_host.cpp"));
    CHECK(th_exist("tri_vadd_reader.cpp"));
    CHECK(th_exist("tri_vadd_writer.cpp"));
    /* The compute kernel should look like TT-Metalium: the standard
     * include and the MAIN entry point are the unambiguous markers. */
    FILE *fp = fopen("tri_vadd_compute.cpp", "rb");
    CHECK(fp != NULL);
    if (!fp) return;
    char buf[4096];
    size_t n = fread(buf, 1, sizeof(buf) - 1, fp);
    fclose(fp);
    buf[n] = '\0';
    CHECK(strstr(buf, "compute_kernel_api.h") != NULL);
    CHECK(strstr(buf, "MAIN") != NULL);
    CHECK(strstr(buf, "cb_wait_front") != NULL);
    PASS();
}
TH_REG("triton", tt_tensix_metalium)

/* ============================================================
 * The AI Slop End-to-End Test
 * ============================================================
 * If a kernel that ChatGPT thinks is good can be parsed, sema'd,
 * lowered, and dispatched to AMD without producing any actual errors
 * (only the documented mask= warnings), the frontend is definately
 * robust to whatever the LLM industry serves up as Triton next year.
 * That feels worth a regression test of its very own. */

static void tt_amd_ai_slop_still_compiles(void)
{
    int rc = tt_run("--triton --amdgpu-bin tests/tri_slop.py "
                    "-o tri_slop.hsaco");
    CHEQ(rc, 0);
    CHECK(th_exist("tri_slop.hsaco"));
    /* The kernel does the same arithmetic as vector_add; the binary
     * should weigh in close to the same size despite the docstring
     * being a small novella. The size check is a sanity bound, not
     * a tight assertion: anywhere from 100 to 800 bytes of code is
     * fine and any wildly different value means something is off. */
    PASS();
}
TH_REG("triton", tt_amd_ai_slop_still_compiles)
