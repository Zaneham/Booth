/* tinline.c -- __device__ call inlining (issue #101).
 * The GPU backends have no calling convention for device functions, so the
 * inliner must splice every call away before isel. The scalar CPU backend
 * emits real calls and must be left alone. */

#include "tharns.h"

static char obuf[TH_BUFSZ];

/* On a GPU target every device call is inlined, so no call survives in the IR
 * (the standalone device bodies are inlined into their callers too). */
static void inl_gpu_no_calls(void)
{
    int rc = th_run(BC_BIN " --amdgpu --ir tests/device_calls.cu",
                    obuf, TH_BUFSZ);
    CHECK(rc == 0);
    CHECK(strstr(obuf, "= call") == NULL);
    PASS();
}

/* And the kernel compiles the whole way to a .hsaco with the calls gone. */
static void inl_gpu_binary(void)
{
    const char *out = "test_inline.hsaco";
    int rc = th_run(BC_BIN " --amdgpu-bin tests/device_calls.cu "
                    "-o test_inline.hsaco", obuf, TH_BUFSZ);
    CHECK(rc == 0);
    CHECK(th_exist(out));
    remove(out);
    PASS();
}

/* NVIDIA and Tensix isel cannot emit a call either, so they inline too. */
static void inl_other_gpu_no_calls(void)
{
    int rc = th_run(BC_BIN " --nvidia-ptx --ir tests/device_calls.cu",
                    obuf, TH_BUFSZ);
    CHECK(rc == 0);
    CHECK(strstr(obuf, "= call") == NULL);

    rc = th_run(BC_BIN " --tensix --ir tests/device_calls.cu",
                obuf, TH_BUFSZ);
    CHECK(rc == 0);
    CHECK(strstr(obuf, "= call") == NULL);
    PASS();
}

/* The CPU backend has a real SysV call ABI, so the inliner must not touch it:
 * the device calls stay as calls. */
static void inl_cpu_keeps_calls(void)
{
    int rc = th_run(BC_BIN " --cpu --ir tests/device_calls.cu",
                    obuf, TH_BUFSZ);
    CHECK(rc == 0);
    CHECK(strstr(obuf, "= call") != NULL);
    PASS();
}

TH_REG("inline", inl_gpu_no_calls);
TH_REG("inline", inl_gpu_binary);
TH_REG("inline", inl_other_gpu_no_calls);
TH_REG("inline", inl_cpu_keeps_calls);
