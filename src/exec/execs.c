/* execs.c -- registration list and dispatch for the run side.
 * Adding a target is one line in rt_list and one <name>_exec.c file. */

#include "exec.h"
#include <string.h>

/* cpu is always here; nvptx dlopens the driver and fails open() cleanly on a
 * machine without one. bc_runtime (AMD/HSA) is still on its own API and not
 * yet behind the contract. */
extern const rt_desc_t cpu_rt_desc;
extern const rt_desc_t nv_rt_desc;

const rt_desc_t * const rt_list[] = {
    &cpu_rt_desc,
    &nv_rt_desc,
    NULL
};

const rt_desc_t *rt_find(const char *name)
{
    if (name == NULL) return NULL;
    for (uint32_t i = 0; i < RT_MAX && rt_list[i] != NULL; i++)
        if (rt_list[i]->name != NULL && strcmp(rt_list[i]->name, name) == 0)
            return rt_list[i];
    return NULL;
}

/* A descriptor wanting more room than the handle holds would scribble over
 * its neighbour, and one missing a required op is a build bug, not a runtime
 * condition. Both are refused here so no call site has to check. */
static int usable(const rt_desc_t *d)
{
    if (d == NULL) return 0;
    if (d->dev_size > RT_DEV_MAX || d->kern_size > RT_KERN_MAX) return 0;
    return d->open && d->shut && d->load && d->unload && d->alloc && d->dfre
        && d->h2d && d->d2h && d->run && d->sync;
}

static int try_open(rt_dev_t *dev, const rt_desc_t *d)
{
    if (!usable(d)) return RT_EINPUT;
    memset(dev->raw, 0, sizeof dev->raw);
    int rc = d->open(dev->raw);
    if (rc != RT_OK) return rc;
    dev->d = d;
    return RT_OK;
}

int rt_open(rt_dev_t *dev, const char *name)
{
    if (dev == NULL) return RT_EINPUT;
    dev->d = NULL;

    if (name != NULL) {
        const rt_desc_t *d = rt_find(name);
        if (d == NULL) return RT_EINPUT;
        return try_open(dev, d);
    }

    /* Probing takes the first target with a device on this machine. */
    int last = RT_ENODEV;
    for (uint32_t i = 0; i < RT_MAX && rt_list[i] != NULL; i++) {
        int rc = try_open(dev, rt_list[i]);
        if (rc == RT_OK) return RT_OK;
        last = rc;
    }
    return last;
}

void rt_shut(rt_dev_t *dev)
{
    if (dev == NULL || dev->d == NULL) return;
    dev->d->shut(dev->raw);
    dev->d = NULL;
}

int rt_load(rt_dev_t *dev, const char *path, const char *kern, rt_kern_t *out)
{
    if (dev == NULL || dev->d == NULL || out == NULL) return RT_EINPUT;
    memset(out->raw, 0, sizeof out->raw);
    return dev->d->load(dev->raw, path, kern, out->raw);
}

void rt_unload(rt_dev_t *dev, rt_kern_t *k)
{
    if (dev == NULL || dev->d == NULL || k == NULL) return;
    dev->d->unload(dev->raw, k->raw);
}

rt_ptr rt_alloc(rt_dev_t *dev, size_t n)
{
    if (dev == NULL || dev->d == NULL) return 0;
    return dev->d->alloc(dev->raw, n);
}

void rt_free(rt_dev_t *dev, rt_ptr p)
{
    if (dev == NULL || dev->d == NULL || p == 0) return;
    dev->d->dfre(dev->raw, p);
}

int rt_h2d(rt_dev_t *dev, rt_ptr dst, const void *src, size_t n)
{
    if (dev == NULL || dev->d == NULL) return RT_EINPUT;
    return dev->d->h2d(dev->raw, dst, src, n);
}

int rt_d2h(rt_dev_t *dev, void *dst, rt_ptr src, size_t n)
{
    if (dev == NULL || dev->d == NULL) return RT_EINPUT;
    return dev->d->d2h(dev->raw, dst, src, n);
}

int rt_run(rt_dev_t *dev, const rt_kern_t *k, const rt_dim_t *dim,
           const rt_arg_t *args, uint32_t nargs)
{
    if (dev == NULL || dev->d == NULL || k == NULL || dim == NULL)
        return RT_EINPUT;
    if (nargs > 0 && args == NULL) return RT_EINPUT;
    return dev->d->run(dev->raw, k->raw, dim, args, nargs);
}

int rt_sync(rt_dev_t *dev)
{
    if (dev == NULL || dev->d == NULL) return RT_EINPUT;
    return dev->d->sync(dev->raw);
}

int rt_hmap(rt_dev_t *dev, void **hp, rt_ptr *dp, size_t n)
{
    if (dev == NULL || dev->d == NULL || hp == NULL || dp == NULL)
        return RT_EINPUT;
    if (dev->d->hmap == NULL) return RT_EUNSUP;
    return dev->d->hmap(dev->raw, hp, dp, n);
}

void rt_hfre(rt_dev_t *dev, void *hp)
{
    if (dev == NULL || dev->d == NULL || dev->d->hfre == NULL) return;
    dev->d->hfre(dev->raw, hp);
}

void rt_trak(rt_dev_t *dev, rt_ptr p, size_t n, const char *lbl, int fl)
{
    if (dev == NULL || dev->d == NULL || dev->d->trak == NULL) return;
    dev->d->trak(dev->raw, p, n, lbl, fl);
}

const char *rt_errs(const rt_dev_t *dev, int rc)
{
    if (dev != NULL && dev->d != NULL && dev->d->errs != NULL) {
        const char *s = dev->d->errs(rc);
        if (s != NULL) return s;
    }
    switch (rc) {
    case RT_OK:      return "ok";
    case RT_ENODRV:  return "the vendor driver library is not installed";
    case RT_ESYM:    return "the driver library is missing a symbol";
    case RT_EDRV:    return "a driver call failed";
    case RT_ENODEV:  return "no device to run on";
    case RT_EIO:     return "the artefact could not be read";
    case RT_EKERN:   return "no kernel by that name in the artefact";
    case RT_EUNSUP:  return "this target cannot do that";
    case RT_EINPUT:  return "bad argument";
    default:         return "unknown error";
    }
}
