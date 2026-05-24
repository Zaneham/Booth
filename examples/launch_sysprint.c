/* launch_sysprint.c -- run the SYSPRINT demo on real AMD hardware.
 *
 * Build:  gcc -std=c99 -O2 -I src/runtime -I runtime -I examples \
 *             examples/launch_sysprint.c \
 *             src/runtime/bc_runtime.c \
 *             runtime/sysprint.c \
 *             -ldl -lm -o launch_sysprint
 * Run:    ./barracuda --amdgpu-bin -I runtime -I examples \
 *             -o sp_demo.hsaco examples/sysprint_kernel.cu
 *         ./launch_sysprint sp_demo.hsaco
 *
 * Currently the AMD codegen for the payload-copy loop is hitting
 * a regalloc bug (s_add_u32: VGPR in scalar source). The NVIDIA
 * PTX and Tensix Metalium backends compile the kernel cleanly;
 * AMD coverage will catch up once the codegen issue is fixed.
 *
 * The launcher itself is the canonical pattern: allocate the
 * SYSPRINT buffer in host-coherent memory, hand its pointer to
 * the kernel as part of the kernarg block, dispatch, drain on
 * return. */

#include "bc_runtime.h"
#include "sysprint.h"
#include "sysprint_classes.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define N           16
#define BLOCK       16
#define SP_BUF_SZ   (64 * 1024)

/* Sink for the single result summary the demo emits. */
static void result_sink(uint32_t cid, const char *cname,
                        const void *p, uint32_t n, void *u)
{
    (void)cid; (void)u;
    printf("%s: %.*s\n", cname, (int)n, (const char *)p);
}

/* Kernarg layout the kernel expects. Must match the C order of
 * the kernel's parameters: (sp, c, a, b, n). The kernel signature
 * is `sp_demo(bc_sp_buf_t *sp, float *c, const float *a,
 * const float *b, int n)`. */
typedef struct {
    void *sp_buf;
    void *c;
    void *a;
    void *b;
    int   n;
    int   _pad;
} kernarg_t;

int main(int argc, char *argv[])
{
    const char *hsaco = (argc > 1) ? argv[1] : "sp_demo.hsaco";

    printf("BarraCUDA SYSPRINT launcher\n");
    printf("  .hsaco:  %s\n", hsaco);

    /* Intern demo classes and register a sink for the result. */
    BC_SP_INIT_DEMO_CLASSES();
    bc_sp_register_sink("DEMO.RESULT", result_sink, NULL);

    bc_device_t dev;
    if (bc_device_init(&dev) != BC_RT_OK) {
        fprintf(stderr, "Device init failed\n");
        return 1;
    }

    bc_kernel_t kern;
    if (bc_load_kernel(&dev, hsaco, "sp_demo", &kern) != BC_RT_OK) {
        fprintf(stderr, "Kernel load failed\n");
        bc_device_shutdown(&dev);
        return 1;
    }

    /* The SYSPRINT buffer lives in device-allocated host-coherent
     * memory so the kernel can write to it and the host can read
     * it back without an explicit copy. The bc_sp_buf_t header
     * sits at offset 0 of the allocation; the record area
     * follows it. */
    size_t sp_alloc = sizeof(bc_sp_buf_t) + SP_BUF_SZ;
    void *sp_dev = bc_alloc(&dev, sp_alloc);
    if (!sp_dev) {
        fprintf(stderr, "SYSPRINT alloc failed\n");
        return 1;
    }
    bc_sp_buf_t hdr;
    hdr.data = (uint8_t *)sp_dev + sizeof(bc_sp_buf_t);
    hdr.size = SP_BUF_SZ;
    hdr.head = 0;
    hdr.dropped = 0;
    bc_copy_h2d(&dev, sp_dev, &hdr, sizeof(hdr));

    /* The kernel's arithmetic inputs. */
    void *d_a = bc_alloc(&dev, N * sizeof(float));
    void *d_b = bc_alloc(&dev, N * sizeof(float));
    void *d_c = bc_alloc(&dev, N * sizeof(float));
    float h_a[N], h_b[N];
    for (int i = 0; i < N; i++) {
        h_a[i] = (float)i;
        h_b[i] = (float)(i * 2);
    }
    bc_copy_h2d(&dev, d_a, h_a, N * sizeof(float));
    bc_copy_h2d(&dev, d_b, h_b, N * sizeof(float));

    /* Dispatch. */
    kernarg_t args = {0};
    args.sp_buf = sp_dev;
    args.c = d_c;
    args.a = d_a;
    args.b = d_b;
    args.n = N;
    int rc = bc_dispatch(&dev, &kern,
                         1, 1, 1,
                         BLOCK, 1, 1,
                         &args, sizeof(args));
    if (rc != BC_RT_OK) {
        fprintf(stderr, "Dispatch failed (%d)\n", rc);
        return 1;
    }

    /* Read the SYSPRINT header back and drain whatever the kernel
     * wrote. The data pointer in our local copy still points at
     * the device-side bytes, which bc_sp_drain will read through
     * after we update head. */
    bc_copy_d2h(&dev, &hdr, sp_dev, sizeof(hdr));
    bc_sp_buf_t local;
    uint8_t local_data[SP_BUF_SZ];
    local.data = local_data;
    local.size = SP_BUF_SZ;
    local.head = hdr.head;
    local.dropped = hdr.dropped;
    bc_copy_d2h(&dev, local_data,
                (uint8_t *)sp_dev + sizeof(bc_sp_buf_t), hdr.head);

    printf("---- SYSPRINT drain ----\n");
    bc_sp_drain(&local);
    printf("---- end (dropped: %u) ----\n", local.dropped);

    bc_free(&dev, d_a);
    bc_free(&dev, d_b);
    bc_free(&dev, d_c);
    bc_free(&dev, sp_dev);
    bc_unload_kernel(&dev, &kern);
    bc_device_shutdown(&dev);
    return 0;
}
