/* bc_runtime.c — Booth HSA Runtime Launcher
 * dlopen's libhsa-runtime64.so. No compile-time ROCm dependency. Linux only. */

#ifdef __linux__

#include "bc_runtime.h"
#include "bc_abend.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <dlfcn.h>

/* ---- HSA Type Definitions ---- */

typedef uint32_t hsa_status_t;

#define HSA_STATUS_SUCCESS   0x0
#define HSA_STATUS_INFO_BREAK 0x1

typedef struct { uint64_t handle; } hsa_agent_t;
typedef struct { uint64_t handle; } hsa_signal_t;
typedef struct { uint64_t handle; } hsa_region_t;
typedef struct { uint64_t handle; } hsa_executable_t;
typedef struct { uint64_t handle; } hsa_executable_symbol_t;
typedef struct { uint64_t handle; } hsa_code_object_reader_t;
typedef struct { uint64_t handle; } hsa_loaded_code_object_t;
typedef struct { uint64_t handle; } hsa_amd_memory_pool_t;
typedef int64_t hsa_signal_value_t;

/* AMD memory pool constants */
#define HSA_AMD_SEGMENT_GLOBAL                    0
#define HSA_AMD_MEMORY_POOL_INFO_SEGMENT          0
#define HSA_AMD_MEMORY_POOL_INFO_GLOBAL_FLAGS     1
#define HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_KERNARG   8
#define HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_FINE      2

/* HSA queue — layout must match ABI exactly */
typedef struct {
    uint32_t    type;
    uint32_t    features;
    void       *base_address;
    hsa_signal_t doorbell_signal;
    uint32_t    size;
    uint32_t    reserved1;
    uint64_t    id;
} hsa_queue_t;

/* AQL dispatch packet — 64 bytes */
typedef struct {
    uint16_t    header;
    uint16_t    setup;
    uint16_t    workgroup_size_x;
    uint16_t    workgroup_size_y;
    uint16_t    workgroup_size_z;
    uint16_t    reserved0;
    uint32_t    grid_size_x;
    uint32_t    grid_size_y;
    uint32_t    grid_size_z;
    uint32_t    private_segment_size;
    uint32_t    group_segment_size;
    uint64_t    kernel_object;
    void       *kernarg_address;
    uint64_t    reserved2;
    hsa_signal_t completion_signal;
} hsa_kernel_dispatch_packet_t;

/* ---- HSA Constants ---- */

#define HSA_DEVICE_TYPE_GPU             1
#define HSA_AGENT_INFO_DEVICE           17
#define HSA_AGENT_INFO_QUEUE_MAX_SIZE   14
#define HSA_QUEUE_TYPE_SINGLE           1
#define HSA_PROFILE_FULL                1
#define HSA_FLOAT_ROUNDING_DEFAULT      0

/* Packet header fields */
#define HSA_PKT_TYPE_DISPATCH           2
#define HSA_PKT_HEADER_TYPE             0
#define HSA_PKT_HEADER_BARRIER          8
#define HSA_PKT_HEADER_SCACQUIRE        9
#define HSA_PKT_HEADER_SCRELEASE       11
#define HSA_FENCE_SCOPE_SYSTEM          2
#define HSA_DISPATCH_SETUP_DIMS         0

/* Region info */
#define HSA_REGION_INFO_SEGMENT         0
#define HSA_REGION_INFO_GLOBAL_FLAGS    1
#define HSA_REGION_SEGMENT_GLOBAL       0
#define HSA_REGION_FLAG_KERNARG         1
#define HSA_REGION_FLAG_COARSE          4

/* Signal wait */
#define HSA_SIGNAL_CONDITION_LT         2
#define HSA_WAIT_STATE_BLOCKED          0

/* Executable symbol info */
#define HSA_SYM_INFO_KERNEL_OBJECT      22
#define HSA_SYM_INFO_KERNARG_SIZE       11
#define HSA_SYM_INFO_GROUP_SIZE         13
#define HSA_SYM_INFO_PRIVATE_SIZE       14

/* ---- Function Pointer Types ---- */

typedef hsa_status_t (*pfn_init_t)(void);
typedef hsa_status_t (*pfn_shut_down_t)(void);
typedef hsa_status_t (*pfn_iterate_agents_t)(
    hsa_status_t(*)(hsa_agent_t, void*), void*);
typedef hsa_status_t (*pfn_agent_get_info_t)(
    hsa_agent_t, uint32_t, void*);
typedef hsa_status_t (*pfn_queue_create_t)(
    hsa_agent_t, uint32_t, uint32_t,
    void(*)(hsa_status_t, hsa_queue_t*, void*), void*,
    uint32_t, uint32_t, hsa_queue_t**);
typedef hsa_status_t (*pfn_queue_destroy_t)(hsa_queue_t*);
typedef hsa_status_t (*pfn_signal_create_t)(
    hsa_signal_value_t, uint32_t, const hsa_agent_t*, hsa_signal_t*);
typedef hsa_status_t (*pfn_signal_destroy_t)(hsa_signal_t);
typedef void (*pfn_signal_store_relaxed_t)(
    hsa_signal_t, hsa_signal_value_t);
typedef hsa_signal_value_t (*pfn_signal_wait_scacquire_t)(
    hsa_signal_t, uint32_t, hsa_signal_value_t, uint64_t, uint32_t);
typedef hsa_status_t (*pfn_mem_alloc_t)(hsa_region_t, size_t, void**);
typedef hsa_status_t (*pfn_mem_free_t)(void*);
typedef hsa_status_t (*pfn_mem_copy_t)(void*, const void*, size_t);
typedef hsa_status_t (*pfn_iterate_regions_t)(
    hsa_agent_t, hsa_status_t(*)(hsa_region_t, void*), void*);
typedef hsa_status_t (*pfn_region_get_info_t)(
    hsa_region_t, uint32_t, void*);
typedef hsa_status_t (*pfn_reader_create_t)(
    const void*, size_t, hsa_code_object_reader_t*);
typedef hsa_status_t (*pfn_reader_destroy_t)(
    hsa_code_object_reader_t);
typedef hsa_status_t (*pfn_exec_create_alt_t)(
    uint32_t, uint32_t, const char*, hsa_executable_t*);
typedef hsa_status_t (*pfn_exec_load_agent_t)(
    hsa_executable_t, hsa_agent_t, hsa_code_object_reader_t,
    const char*, hsa_loaded_code_object_t*);
typedef hsa_status_t (*pfn_exec_freeze_t)(
    hsa_executable_t, const char*);
typedef hsa_status_t (*pfn_exec_get_symbol_t)(
    hsa_executable_t, const char*, const hsa_agent_t*,
    hsa_executable_symbol_t*);
typedef hsa_status_t (*pfn_symbol_get_info_t)(
    hsa_executable_symbol_t, uint32_t, void*);
typedef hsa_status_t (*pfn_exec_destroy_t)(hsa_executable_t);
typedef uint64_t (*pfn_queue_load_write_idx_t)(const hsa_queue_t*);
typedef void (*pfn_queue_store_write_idx_t)(
    const hsa_queue_t*, uint64_t);

/* AMD memory pool extensions */
typedef hsa_status_t (*pfn_amd_pool_alloc_t)(
    hsa_amd_memory_pool_t, size_t, uint32_t, void**);
typedef hsa_status_t (*pfn_amd_pool_free_t)(void*);
typedef hsa_status_t (*pfn_amd_iterate_pools_t)(
    hsa_agent_t, hsa_status_t(*)(hsa_amd_memory_pool_t, void*), void*);
typedef hsa_status_t (*pfn_amd_pool_get_info_t)(
    hsa_amd_memory_pool_t, uint32_t, void*);
typedef hsa_status_t (*pfn_amd_allow_access_t)(
    uint32_t, const hsa_agent_t*, const uint32_t*, const void*);

/* ---- Internal Device Structure ---- */

typedef struct {
    void *hsa_lib;

    /* HSA function pointers */
    pfn_init_t                  hsa_init;
    pfn_shut_down_t             hsa_shut_down;
    pfn_iterate_agents_t        iterate_agents;
    pfn_agent_get_info_t        agent_get_info;
    pfn_queue_create_t          queue_create;
    pfn_queue_destroy_t         queue_destroy;
    pfn_signal_create_t         signal_create;
    pfn_signal_destroy_t        signal_destroy;
    pfn_signal_store_relaxed_t  signal_store_relaxed;
    pfn_signal_wait_scacquire_t signal_wait;
    pfn_mem_alloc_t             mem_alloc;
    pfn_mem_free_t              mem_free;
    pfn_mem_copy_t              mem_copy;
    pfn_iterate_regions_t       iterate_regions;
    pfn_region_get_info_t       region_get_info;
    pfn_reader_create_t         reader_create;
    pfn_reader_destroy_t        reader_destroy;
    pfn_exec_create_alt_t       exec_create;
    pfn_exec_load_agent_t       exec_load;
    pfn_exec_freeze_t           exec_freeze;
    pfn_exec_get_symbol_t       exec_get_symbol;
    pfn_symbol_get_info_t       symbol_get_info;
    pfn_exec_destroy_t          exec_destroy;
    pfn_queue_load_write_idx_t  queue_load_write_idx;
    pfn_queue_store_write_idx_t queue_store_write_idx;

    /* AMD memory pool extensions — because hsa_memory_allocate
     * gives memory the scalar unit can't read on MI300X. Cheers AMD. */
    pfn_amd_pool_alloc_t    amd_palloc;
    pfn_amd_pool_free_t     amd_pfree;
    pfn_amd_iterate_pools_t amd_ipools;
    pfn_amd_pool_get_info_t amd_pinfo;
    pfn_amd_allow_access_t  amd_allow;

    /* Device state */
    hsa_agent_t  gpu_agent;
    hsa_queue_t *queue;
    hsa_region_t kernarg_region;
    hsa_region_t device_region;
    hsa_amd_memory_pool_t kernarg_pool;  /* AMD pool: the one that works */
    int          initialized;

    /* ABEND fault diagnostics — heap-allocated (~37KB) */
    ab_ctx_t    *abend;
} bc_device_impl_t;

_Static_assert(sizeof(bc_device_impl_t) <= BC_DEVICE_OPAQUE_SIZE,
               "BC_DEVICE_OPAQUE_SIZE too small — increase it");

static bc_device_impl_t *D_of(bc_device_t *dev)
{
    return (bc_device_impl_t *)dev->_data;
}

/* ---- dlsym Helper ---- */

#define LOAD(field, name) do { \
    *(void **)(&D->field) = dlsym(D->hsa_lib, name); \
    if (!D->field) { \
        fprintf(stderr, "bc_runtime: missing %s\n", name); \
        dlclose(D->hsa_lib); D->hsa_lib = NULL; \
        return BC_RT_ERR_DLSYM; \
    } \
} while(0)

/* ---- HSA Iteration Callbacks ---- */

static hsa_status_t find_gpu_cb(hsa_agent_t agent, void *data)
{
    bc_device_impl_t *D = (bc_device_impl_t *)data;
    uint32_t dev_type = 0;
    D->agent_get_info(agent, HSA_AGENT_INFO_DEVICE, &dev_type);
    if (dev_type == HSA_DEVICE_TYPE_GPU) {
        D->gpu_agent = agent;
        return HSA_STATUS_INFO_BREAK;
    }
    return HSA_STATUS_SUCCESS;
}

static hsa_status_t find_kernarg_cb(hsa_region_t region, void *data)
{
    bc_device_impl_t *D = (bc_device_impl_t *)data;
    uint32_t segment = 0;
    D->region_get_info(region, HSA_REGION_INFO_SEGMENT, &segment);
    if (segment != HSA_REGION_SEGMENT_GLOBAL) return HSA_STATUS_SUCCESS;
    uint32_t flags = 0;
    D->region_get_info(region, HSA_REGION_INFO_GLOBAL_FLAGS, &flags);
    if (flags & HSA_REGION_FLAG_KERNARG) {
        D->kernarg_region = region;
        return HSA_STATUS_INFO_BREAK;
    }
    return HSA_STATUS_SUCCESS;
}

static hsa_status_t find_device_cb(hsa_region_t region, void *data)
{
    bc_device_impl_t *D = (bc_device_impl_t *)data;
    uint32_t segment = 0;
    D->region_get_info(region, HSA_REGION_INFO_SEGMENT, &segment);
    if (segment != HSA_REGION_SEGMENT_GLOBAL) return HSA_STATUS_SUCCESS;
    uint32_t flags = 0;
    D->region_get_info(region, HSA_REGION_INFO_GLOBAL_FLAGS, &flags);
    if (flags & HSA_REGION_FLAG_COARSE) {
        D->device_region = region;
        return HSA_STATUS_INFO_BREAK;
    }
    return HSA_STATUS_SUCCESS;
}

/* ---- AMD Pool Discovery ---- */

/* Find the kernarg pool the way HIP does it, not the way the HSA
 * spec says to. The spec is a lovely work of fiction. */
static hsa_status_t find_kpool_cb(hsa_amd_memory_pool_t pool, void *data)
{
    bc_device_impl_t *D = (bc_device_impl_t *)data;
    uint32_t segment = 0;
    D->amd_pinfo(pool, HSA_AMD_MEMORY_POOL_INFO_SEGMENT, &segment);
    if (segment != HSA_AMD_SEGMENT_GLOBAL) return HSA_STATUS_SUCCESS;
    uint32_t flags = 0;
    D->amd_pinfo(pool, HSA_AMD_MEMORY_POOL_INFO_GLOBAL_FLAGS, &flags);
    if (flags & HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_FINE) {
        D->kernarg_pool = pool;
        return HSA_STATUS_INFO_BREAK;
    }
    return HSA_STATUS_SUCCESS;
}

/* ---- Queue Error Callback ---- */

/* Called by HSA when the queue faults — aperture violations, invalid
 * packets, the usual GPU misdemeanours. Maps HSA status to an ABEND
 * code and fires the dump. This is where the magic happens. */
static void bc_qerr(hsa_status_t status, hsa_queue_t *source, void *data)
{
    (void)source;
    ab_ctx_t *A = (ab_ctx_t *)data;
    if (!A) return;

    /* Map HSA queue error to ABEND code.
     * HSA_STATUS_ERROR_MEMORY_APERTURE_VIOLATION = 41 (0x29)
     * HSA_STATUS_ERROR_MEMORY_FAULT = 43 (0x2B) */
    uint16_t code = AB_G0FF;
    if (status == 41)       /* MEMORY_APERTURE_VIOLATION */
        code = AB_G0C5;
    else if (status == 43)  /* MEMORY_FAULT */
        code = AB_G0C4;
    else if (status == 42)  /* ILLEGAL_INSTRUCTION */
        code = AB_G0C1;

    A->code = code;
    A->reason = (uint32_t)status;
    A->faulted = 1;

    ab_dump(A, stderr);
}

/* ---- Init / Shutdown ---- */

int bc_device_init(bc_device_t *dev)
{
    memset(dev, 0, sizeof(*dev));
    bc_device_impl_t *D = D_of(dev);

    D->hsa_lib = dlopen("libhsa-runtime64.so", RTLD_NOW);
    if (!D->hsa_lib)
        D->hsa_lib = dlopen("/opt/rocm/lib/libhsa-runtime64.so", RTLD_NOW);
    if (!D->hsa_lib) {
        fprintf(stderr, "bc_runtime: cannot load libhsa-runtime64.so: %s\n",
                dlerror());
        return BC_RT_ERR_DLOPEN;
    }

    LOAD(hsa_init,           "hsa_init");
    LOAD(hsa_shut_down,      "hsa_shut_down");
    LOAD(iterate_agents,     "hsa_iterate_agents");
    LOAD(agent_get_info,     "hsa_agent_get_info");
    LOAD(queue_create,       "hsa_queue_create");
    LOAD(queue_destroy,      "hsa_queue_destroy");
    LOAD(signal_create,      "hsa_signal_create");
    LOAD(signal_destroy,     "hsa_signal_destroy");
    LOAD(signal_store_relaxed, "hsa_signal_store_relaxed");
    LOAD(signal_wait,        "hsa_signal_wait_scacquire");
    LOAD(mem_alloc,          "hsa_memory_allocate");
    LOAD(mem_free,           "hsa_memory_free");
    LOAD(mem_copy,           "hsa_memory_copy");
    LOAD(iterate_regions,    "hsa_agent_iterate_regions");
    LOAD(region_get_info,    "hsa_region_get_info");
    LOAD(reader_create,      "hsa_code_object_reader_create_from_memory");
    LOAD(reader_destroy,     "hsa_code_object_reader_destroy");
    LOAD(exec_create,        "hsa_executable_create_alt");
    LOAD(exec_load,          "hsa_executable_load_agent_code_object");
    LOAD(exec_freeze,        "hsa_executable_freeze");
    LOAD(exec_get_symbol,    "hsa_executable_get_symbol_by_name");
    LOAD(symbol_get_info,    "hsa_executable_symbol_get_info");
    LOAD(exec_destroy,       "hsa_executable_destroy");
    LOAD(queue_load_write_idx,  "hsa_queue_load_write_index_relaxed");
    LOAD(queue_store_write_idx, "hsa_queue_store_write_index_relaxed");
    LOAD(amd_palloc,  "hsa_amd_memory_pool_allocate");
    LOAD(amd_pfree,   "hsa_amd_memory_pool_free");
    LOAD(amd_ipools,  "hsa_amd_agent_iterate_memory_pools");
    LOAD(amd_pinfo,   "hsa_amd_memory_pool_get_info");
    LOAD(amd_allow,   "hsa_amd_agents_allow_access");

    hsa_status_t st = D->hsa_init();
    if (st != HSA_STATUS_SUCCESS) {
        fprintf(stderr, "bc_runtime: hsa_init failed (0x%x)\n", st);
        dlclose(D->hsa_lib);
        return BC_RT_ERR_HSA;
    }

    D->gpu_agent.handle = 0;
    D->iterate_agents(find_gpu_cb, D);
    if (D->gpu_agent.handle == 0) {
        fprintf(stderr, "bc_runtime: no AMD GPU found\n");
        D->hsa_shut_down();
        dlclose(D->hsa_lib);
        return BC_RT_ERR_NO_GPU;
    }

    D->kernarg_region.handle = 0;
    D->iterate_regions(D->gpu_agent, find_kernarg_cb, D);
    if (D->kernarg_region.handle == 0) {
        fprintf(stderr, "bc_runtime: kernarg memory region not found\n");
        D->hsa_shut_down();
        dlclose(D->hsa_lib);
        return BC_RT_ERR_NO_MEM;
    }

    D->device_region.handle = 0;
    D->iterate_regions(D->gpu_agent, find_device_cb, D);
    if (D->device_region.handle == 0) {
        fprintf(stderr, "bc_runtime: device memory region not found\n");
        D->hsa_shut_down();
        dlclose(D->hsa_lib);
        return BC_RT_ERR_NO_MEM;
    }

    /* AMD pool discovery — find kernarg pool the way HIP does it */
    D->kernarg_pool.handle = 0;
    D->amd_ipools(D->gpu_agent, find_kpool_cb, D);
    if (D->kernarg_pool.handle == 0) {
        fprintf(stderr, "bc_runtime: AMD kernarg pool not found\n");
        D->hsa_shut_down();
        dlclose(D->hsa_lib);
        return BC_RT_ERR_NO_MEM;
    }
    fprintf(stderr, "bc_runtime: AMD kernarg pool found (handle=0x%lx)\n",
            (unsigned long)D->kernarg_pool.handle);

    /* ABEND diagnostics — init before queue so we can hook the
     * queue error callback. Aperture violations are queue errors,
     * not VM faults, so the system event handler alone won't catch them. */
    D->abend = (ab_ctx_t *)calloc(1, sizeof(ab_ctx_t));
    if (D->abend)
        ab_init(D->abend, D->hsa_lib);

    uint32_t queue_size = 0;
    D->agent_get_info(D->gpu_agent, HSA_AGENT_INFO_QUEUE_MAX_SIZE,
                      &queue_size);
    if (queue_size == 0) queue_size = 128;

    st = D->queue_create(D->gpu_agent, queue_size, HSA_QUEUE_TYPE_SINGLE,
                         bc_qerr, D->abend, UINT32_MAX, UINT32_MAX,
                         &D->queue);
    if (st != HSA_STATUS_SUCCESS) {
        fprintf(stderr, "bc_runtime: queue_create failed (0x%x)\n", st);
        D->hsa_shut_down();
        dlclose(D->hsa_lib);
        return BC_RT_ERR_HSA;
    }

    D->initialized = 1;

    fprintf(stderr, "bc_runtime: GPU ready, queue size %u\n", queue_size);
    return BC_RT_OK;
}

void bc_device_shutdown(bc_device_t *dev)
{
    bc_device_impl_t *D = D_of(dev);
    if (!D->initialized) return;

    if (D->abend) { ab_shut(D->abend); free(D->abend); }
    if (D->queue)    D->queue_destroy(D->queue);
    D->hsa_shut_down();
    if (D->hsa_lib)  dlclose(D->hsa_lib);

    memset(dev, 0, sizeof(*dev));
}

/* ---- Code Object Loading ---- */

#define BC_MAX_HSACO (1u * 1024 * 1024)
static uint8_t hsaco_buf[BC_MAX_HSACO];

int bc_load_kernel(bc_device_t *dev, const char *hsaco_path,
                   const char *kernel_name, bc_kernel_t *out)
{
    bc_device_impl_t *D = D_of(dev);
    memset(out, 0, sizeof(*out));

    FILE *fp = fopen(hsaco_path, "rb");
    if (!fp) {
        fprintf(stderr, "bc_runtime: cannot open '%s'\n", hsaco_path);
        return BC_RT_ERR_IO;
    }
    fseek(fp, 0, SEEK_END);
    long sz = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    if (sz <= 0 || (size_t)sz > BC_MAX_HSACO) {
        fprintf(stderr, "bc_runtime: '%s' too large (%ld bytes, max %u)\n",
                hsaco_path, sz, BC_MAX_HSACO);
        fclose(fp);
        return BC_RT_ERR_IO;
    }
    size_t n = fread(hsaco_buf, 1, (size_t)sz, fp);
    fclose(fp);

    hsa_code_object_reader_t reader;
    hsa_status_t st = D->reader_create(hsaco_buf, n, &reader);
    if (st != HSA_STATUS_SUCCESS) {
        fprintf(stderr, "bc_runtime: reader_create failed (0x%x)\n", st);
        return BC_RT_ERR_HSA;
    }

    hsa_executable_t exec;
    st = D->exec_create(HSA_PROFILE_FULL, HSA_FLOAT_ROUNDING_DEFAULT,
                        NULL, &exec);
    if (st != HSA_STATUS_SUCCESS) {
        fprintf(stderr, "bc_runtime: executable_create failed (0x%x)\n", st);
        D->reader_destroy(reader);
        return BC_RT_ERR_HSA;
    }

    st = D->exec_load(exec, D->gpu_agent, reader, NULL, NULL);
    if (st != HSA_STATUS_SUCCESS) {
        fprintf(stderr, "bc_runtime: executable_load failed (0x%x)\n", st);
        D->exec_destroy(exec);
        D->reader_destroy(reader);
        return BC_RT_ERR_HSA;
    }

    st = D->exec_freeze(exec, NULL);
    if (st != HSA_STATUS_SUCCESS) {
        fprintf(stderr, "bc_runtime: executable_freeze failed (0x%x)\n", st);
        D->exec_destroy(exec);
        D->reader_destroy(reader);
        return BC_RT_ERR_HSA;
    }

    /* V3 code objects use "name.kd" suffix */
    char sym_name[256];
    snprintf(sym_name, sizeof(sym_name), "%s.kd", kernel_name);

    hsa_executable_symbol_t symbol;
    st = D->exec_get_symbol(exec, sym_name, &D->gpu_agent, &symbol);
    if (st != HSA_STATUS_SUCCESS) {
        fprintf(stderr, "bc_runtime: kernel '%s' not found (tried '%s', "
                "status 0x%x)\n", kernel_name, sym_name, st);
        D->exec_destroy(exec);
        D->reader_destroy(reader);
        return BC_RT_ERR_SYMBOL;
    }

    D->symbol_get_info(symbol, HSA_SYM_INFO_KERNEL_OBJECT,
                       &out->kernel_object);
    D->symbol_get_info(symbol, HSA_SYM_INFO_KERNARG_SIZE,
                       &out->kernarg_size);
    D->symbol_get_info(symbol, HSA_SYM_INFO_GROUP_SIZE,
                       &out->group_size);
    D->symbol_get_info(symbol, HSA_SYM_INFO_PRIVATE_SIZE,
                       &out->private_size);

    out->_exec = exec.handle;
    out->_reader = reader.handle;

    /* Load source map and stash kernel name for ABEND dumps */
    if (D->abend) {
        ab_slod(D->abend, hsaco_buf, (uint32_t)n);
        /* Pre-fill dispatch context with kernel info so the dump
         * has something useful even if ab_snag hasn't been called */
        ab_snag(D->abend, out, kernel_name, "gfx942",
                0, 0, 0, 0, 0, 0, NULL, 0);
    }

    fprintf(stderr, "bc_runtime: loaded '%s' (kernarg=%u, lds=%u, "
            "scratch=%u)\n", kernel_name, out->kernarg_size,
            out->group_size, out->private_size);
    return BC_RT_OK;
}

void bc_unload_kernel(bc_device_t *dev, bc_kernel_t *kern)
{
    bc_device_impl_t *D = D_of(dev);
    if (kern->_exec) {
        hsa_executable_t exec = { kern->_exec };
        D->exec_destroy(exec);
    }
    if (kern->_reader) {
        hsa_code_object_reader_t reader = { kern->_reader };
        D->reader_destroy(reader);
    }
    memset(kern, 0, sizeof(*kern));
}

/* ---- Memory Management ---- */

void *bc_alloc(bc_device_t *dev, size_t size)
{
    bc_device_impl_t *D = D_of(dev);
    void *ptr = NULL;
    hsa_status_t st = D->mem_alloc(D->device_region, size, &ptr);
    if (st != HSA_STATUS_SUCCESS) {
        fprintf(stderr, "bc_runtime: alloc %zu bytes failed (0x%x)\n",
                size, st);
        return NULL;
    }
    return ptr;
}

void bc_free(bc_device_t *dev, void *ptr)
{
    if (!ptr) return;
    bc_device_impl_t *D = D_of(dev);
    D->mem_free(ptr);
}

void bc_trak(bc_device_t *dev, void *ptr, size_t size,
             const char *label, int flags)
{
    bc_device_impl_t *D = D_of(dev);
    if (D->abend)
        ab_trak(D->abend, (uint64_t)(uintptr_t)ptr,
                (uint64_t)size, label, (uint8_t)flags);
}

int bc_copy_h2d(bc_device_t *dev, void *dst, const void *src, size_t size)
{
    bc_device_impl_t *D = D_of(dev);
    hsa_status_t st = D->mem_copy(dst, src, size);
    if (st != HSA_STATUS_SUCCESS) {
        fprintf(stderr, "bc_runtime: h2d copy failed (0x%x)\n", st);
        return BC_RT_ERR_HSA;
    }
    return BC_RT_OK;
}

int bc_copy_d2h(bc_device_t *dev, void *dst, const void *src, size_t size)
{
    bc_device_impl_t *D = D_of(dev);
    hsa_status_t st = D->mem_copy(dst, src, size);
    if (st != HSA_STATUS_SUCCESS) {
        fprintf(stderr, "bc_runtime: d2h copy failed (0x%x)\n", st);
        return BC_RT_ERR_HSA;
    }
    return BC_RT_OK;
}

/* ---- Kernel Dispatch ---- */

int bc_dispatch(bc_device_t *dev, const bc_kernel_t *kern,
                uint32_t grid_x, uint32_t grid_y, uint32_t grid_z,
                uint32_t block_x, uint32_t block_y, uint32_t block_z,
                const void *args, uint32_t args_size)
{
    bc_device_impl_t *D = D_of(dev);
    hsa_status_t st;

    if (grid_x == 0)  grid_x = 1;
    if (grid_y == 0)  grid_y = 1;
    if (grid_z == 0)  grid_z = 1;
    if (block_x == 0) block_x = 1;
    if (block_y == 0) block_y = 1;
    if (block_z == 0) block_z = 1;

    void *kernarg_buf = NULL;
    uint32_t alloc_sz = kern->kernarg_size;
    if (args_size > alloc_sz) alloc_sz = args_size;
    st = D->amd_palloc(D->kernarg_pool, alloc_sz, 0, &kernarg_buf);
    if (st != HSA_STATUS_SUCCESS || !kernarg_buf) {
        fprintf(stderr, "bc_runtime: kernarg pool alloc failed (0x%x)\n", st);
        return BC_RT_ERR_HSA;
    }

    /* Grant GPU access to the kernarg buffer. Without this the IOMMU
     * doesn't map the pages and s_load_dword returns zeros silently.
     * Undocumented, of course. Thanks AMD. */
    D->amd_allow(1, &D->gpu_agent, NULL, kernarg_buf);

    memset(kernarg_buf, 0, alloc_sz);
    memcpy(kernarg_buf, args, args_size);

    /* Update dispatch dims + kernarg snapshot for ABEND dump */
    if (D->abend) {
        D->abend->dctx.grid[0] = grid_x;
        D->abend->dctx.grid[1] = grid_y;
        D->abend->dctx.grid[2] = grid_z;
        D->abend->dctx.block[0] = block_x;
        D->abend->dctx.block[1] = block_y;
        D->abend->dctx.block[2] = block_z;
        if (args && args_size > 0) {
            uint32_t snap = args_size;
            if (snap > sizeof(D->abend->args_snap))
                snap = (uint32_t)sizeof(D->abend->args_snap);
            memcpy(D->abend->args_snap, args, snap);
            D->abend->dctx.args_sz = snap;
        }
    }

    hsa_signal_t signal;
    st = D->signal_create(1, 0, NULL, &signal);
    if (st != HSA_STATUS_SUCCESS) {
        fprintf(stderr, "bc_runtime: signal_create failed (0x%x)\n", st);
        D->amd_pfree(kernarg_buf);
        return BC_RT_ERR_HSA;
    }

    uint64_t idx = D->queue_load_write_idx(D->queue);
    uint32_t mask = D->queue->size - 1;
    hsa_kernel_dispatch_packet_t *pkt =
        &((hsa_kernel_dispatch_packet_t *)D->queue->base_address)[idx & mask];

    uint32_t dims = 1;
    if (grid_y > 1 || block_y > 1) dims = 2;
    if (grid_z > 1 || block_z > 1) dims = 3;

    /* Fill fields before header — packet processor may execute on sight */
    pkt->setup             = (uint16_t)(dims << HSA_DISPATCH_SETUP_DIMS);
    pkt->workgroup_size_x  = (uint16_t)block_x;
    pkt->workgroup_size_y  = (uint16_t)block_y;
    pkt->workgroup_size_z  = (uint16_t)block_z;
    pkt->reserved0         = 0;
    pkt->grid_size_x       = grid_x * block_x;
    pkt->grid_size_y       = grid_y * block_y;
    pkt->grid_size_z       = grid_z * block_z;
    pkt->private_segment_size = kern->private_size;
    pkt->group_segment_size   = kern->group_size;
    pkt->kernel_object     = kern->kernel_object;
    pkt->kernarg_address   = kernarg_buf;
    pkt->reserved2         = 0;
    pkt->completion_signal = signal;

    uint16_t header = (uint16_t)(
        (HSA_PKT_TYPE_DISPATCH  << HSA_PKT_HEADER_TYPE)     |
        (1                      << HSA_PKT_HEADER_BARRIER)   |
        (HSA_FENCE_SCOPE_SYSTEM << HSA_PKT_HEADER_SCACQUIRE) |
        (HSA_FENCE_SCOPE_SYSTEM << HSA_PKT_HEADER_SCRELEASE)
    );

    __atomic_store_n(&pkt->header, header, __ATOMIC_RELEASE);

    D->queue_store_write_idx(D->queue, idx + 1);
    D->signal_store_relaxed(D->queue->doorbell_signal,
                            (hsa_signal_value_t)idx);

    D->signal_wait(signal, HSA_SIGNAL_CONDITION_LT, 1,
                   UINT64_MAX, HSA_WAIT_STATE_BLOCKED);

    /* If a GPU fault occurred, the ABEND callback (ab_fcb in bc_abend.c)
     * has already fired and printed the dump to stderr. The signal_wait
     * returns normally even after a fault -- the CP still signals
     * completion, it's just that the results are garbage. Callers who
     * set up an ab_ctx_t via ab_init/ab_snag can check A->faulted. */

    D->signal_destroy(signal);
    D->amd_pfree(kernarg_buf);

    return BC_RT_OK;
}

#else /* !__linux__ */

/* Non-Linux stubs */

#include "bc_runtime.h"
#include <stdio.h>
#include <string.h>

int bc_device_init(bc_device_t *dev)
{
    (void)dev;
    fprintf(stderr, "bc_runtime: HSA runtime requires Linux with ROCm\n");
    return BC_RT_ERR_DLOPEN;
}

void bc_device_shutdown(bc_device_t *dev) { (void)dev; }

int bc_load_kernel(bc_device_t *dev, const char *hsaco_path,
                   const char *kernel_name, bc_kernel_t *out)
{
    (void)dev; (void)hsaco_path; (void)kernel_name; (void)out;
    return BC_RT_ERR_DLOPEN;
}

void bc_unload_kernel(bc_device_t *dev, bc_kernel_t *kern)
{ (void)dev; (void)kern; }

void *bc_alloc(bc_device_t *dev, size_t size)
{ (void)dev; (void)size; return NULL; }

void bc_free(bc_device_t *dev, void *ptr)
{ (void)dev; (void)ptr; }

void bc_trak(bc_device_t *dev, void *ptr, size_t size,
             const char *label, int flags)
{ (void)dev; (void)ptr; (void)size; (void)label; (void)flags; }

int bc_copy_h2d(bc_device_t *dev, void *dst, const void *src, size_t size)
{ (void)dev; (void)dst; (void)src; (void)size; return BC_RT_ERR_DLOPEN; }

int bc_copy_d2h(bc_device_t *dev, void *dst, const void *src, size_t size)
{ (void)dev; (void)dst; (void)src; (void)size; return BC_RT_ERR_DLOPEN; }

int bc_dispatch(bc_device_t *dev, const bc_kernel_t *kern,
                uint32_t grid_x, uint32_t grid_y, uint32_t grid_z,
                uint32_t block_x, uint32_t block_y, uint32_t block_z,
                const void *args, uint32_t args_size)
{
    (void)dev; (void)kern; (void)grid_x; (void)grid_y; (void)grid_z;
    (void)block_x; (void)block_y; (void)block_z;
    (void)args; (void)args_size;
    return BC_RT_ERR_DLOPEN;
}

#endif /* __linux__ */
