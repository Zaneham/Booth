/* nv_exec.c -- nv_rt behind rt_desc_t, so kath run can drive the card */

#include "exec.h"
#include "booth/nv_rt.h"
#include <string.h>

static int nv_map(int rc)
{
    switch (rc) {
    case NV_RT_OK:         return RT_OK;
    case NV_RT_ERR_DL:     return RT_ENODRV;
    case NV_RT_ERR_SYM:    return RT_ESYM;
    case NV_RT_ERR_NO_GPU: return RT_ENODEV;
    case NV_RT_ERR_IO:     return RT_EIO;
    case NV_RT_ERR_KERN:   return RT_EKERN;
    default:               return RT_EDRV;
    }
}

static int nvx_open(void *dev) { return nv_map(nv_rt_init((nv_dev_t *)dev)); }
static void nvx_shut(void *dev) { nv_rt_shut((nv_dev_t *)dev); }

static int nvx_load(void *dev, const char *path, const char *kern, void *out)
{
    return nv_map(nv_rt_load((nv_dev_t *)dev, path, kern, (nv_kern_t *)out));
}
static void nvx_unload(void *dev, void *kern)
{
    nv_rt_unload((nv_dev_t *)dev, (nv_kern_t *)kern);
}

static rt_ptr nvx_alloc(void *dev, size_t n)
{
    return (rt_ptr)nv_rt_alloc((nv_dev_t *)dev, n);
}
static void nvx_dfre(void *dev, rt_ptr p)
{
    nv_rt_free((nv_dev_t *)dev, (CUdevptr)p);
}
static int nvx_h2d(void *dev, rt_ptr dst, const void *src, size_t n)
{
    return nv_map(nv_rt_h2d((nv_dev_t *)dev, (CUdevptr)dst, src, n));
}
static int nvx_d2h(void *dev, void *dst, rt_ptr src, size_t n)
{
    return nv_map(nv_rt_d2h((nv_dev_t *)dev, dst, (CUdevptr)src, n));
}
static int nvx_sync(void *dev) { return nv_map(nv_rt_sync((nv_dev_t *)dev)); }

static int nvx_run(void *dev, const void *kern, const rt_dim_t *dim,
                   const rt_arg_t *args, uint32_t nargs)
{
    void *params[RT_MAX * 2];
    if (nargs > sizeof params / sizeof params[0]) return RT_EINPUT;
    for (uint32_t i = 0; i < nargs; i++) {
        void *p; memcpy(&p, &args[i].p, sizeof p);
        params[i] = p;
    }
    return nv_map(nv_rt_launch((nv_dev_t *)dev, (nv_kern_t *)kern,
                               dim->grid[0], dim->grid[1], dim->grid[2],
                               dim->block[0], dim->block[1], dim->block[2],
                               dim->shmem, params));
}

const rt_desc_t nv_rt_desc = {
    .name      = "nvptx",
    .artefact  = "ptx",
    .dev_size  = (uint32_t)sizeof(nv_dev_t),
    .kern_size = (uint32_t)sizeof(nv_kern_t),
    .open   = nvx_open,
    .shut   = nvx_shut,
    .load   = nvx_load,
    .unload = nvx_unload,
    .alloc  = nvx_alloc,
    .dfre   = nvx_dfre,
    .h2d    = nvx_h2d,
    .d2h    = nvx_d2h,
    .run    = nvx_run,
    .sync   = nvx_sync,
    .hmap   = NULL,
    .hfre   = NULL,
    .trak   = NULL,
    .errs   = NULL
};
