/* exec.h -- running a compiled artefact on the machine.
 * One descriptor per target, the same shape be_desc_t gives the backends. */

#ifndef BOOTH_EXEC_H
#define BOOTH_EXEC_H

#include <stdint.h>
#include <stddef.h>

/* ---- Status ---- */

#define RT_OK        0
#define RT_ENODRV   -1   /* the vendor driver library is not installed */
#define RT_ESYM     -2   /* driver library is missing a symbol we need */
#define RT_EDRV     -3   /* a driver call failed */
#define RT_ENODEV   -4   /* driver loaded, no device to run on */
#define RT_EIO      -5   /* artefact could not be read */
#define RT_EKERN    -6   /* artefact holds no kernel by that name */
#define RT_EUNSUP   -7   /* this target cannot do what was asked */
#define RT_EINPUT   -8   /* caller passed something impossible */

/* Every target we run on addresses with 64 bits. Zero is the null address. */
typedef uint64_t rt_ptr;

/* ---- Arguments ---- */

/* CUDA wants an array of pointers to the arguments, HSA wants them packed into
 * one buffer, and neither form can be built from the other without the sizes.
 * klass says which register bank a by-value scalar rides on a target that
 * calls the kernel directly (CPU); the device launchers never read it. */
#define RT_K_INT  0u
#define RT_K_FLT  1u

typedef struct {
    const void *p;
    uint32_t    size;
    uint8_t     klass;
} rt_arg_t;

/* A target that packs aligns each argument to min(size, 8) before writing it,
 * which is the layout a C struct of the same fields would have had. Get this
 * wrong and the kernel reads shifted garbage rather than failing. */
#define RT_ALIGN(sz) ((sz) < 8u ? (sz) : 8u)

/* ---- Handles ---- */

/* Opaque and caller-held, so a launcher needs no allocator. A descriptor
 * declaring more than these is refused rather than trusted, as with be_opts. */
#define RT_DEV_MAX   1024
#define RT_KERN_MAX   192

struct rt_desc;

typedef struct {
    const struct rt_desc *d;         /* NULL until rt_open succeeds */
    uint8_t raw[RT_DEV_MAX];
} rt_dev_t;

typedef struct { uint8_t raw[RT_KERN_MAX]; } rt_kern_t;

/* Shared memory is per launch. A target that fixes it when the kernel is built
 * returns RT_EUNSUP for a non-zero value rather than quietly dropping it. */
typedef struct {
    uint32_t grid[3];
    uint32_t block[3];
    uint32_t shmem;
} rt_dim_t;

/* ---- Descriptor ---- */

/* No capability bits. A caller that wants host-visible memory calls hmap and
 * either gets it or gets RT_EUNSUP, rather than testing a flag and growing a
 * second code path for hardware it does not own. */

typedef struct rt_desc {
    const char *name;        /* the backend name it runs, "nvptx", "amdgpu" */
    const char *artefact;    /* extension it loads, "ptx", "hsaco" */
    uint32_t    dev_size;    /* bytes of rt_dev_t.raw this target uses */
    uint32_t    kern_size;

    /* Required. A descriptor missing any of these is a build-time bug. */
    int    (*open)  (void *dev);
    void   (*shut)  (void *dev);
    int    (*load)  (void *dev, const char *path, const char *kern, void *out);
    void   (*unload)(void *dev, void *kern);
    rt_ptr (*alloc) (void *dev, size_t n);
    void   (*dfre)  (void *dev, rt_ptr p);
    int    (*h2d)   (void *dev, rt_ptr dst, const void *src, size_t n);
    int    (*d2h)   (void *dev, void *dst, rt_ptr src, size_t n);
    int    (*run)   (void *dev, const void *kern, const rt_dim_t *dim,
                     const rt_arg_t *args, uint32_t nargs);
    int    (*sync)  (void *dev);

    /* Optional, NULL where the target has no answer. */
    int    (*hmap)  (void *dev, void **hp, rt_ptr *dp, size_t n);
    void   (*hfre)  (void *dev, void *hp);
    void   (*trak)  (void *dev, rt_ptr p, size_t n, const char *lbl, int fl);
    const char *(*errs)(int rc);
} rt_desc_t;

/* NULL-terminated, and bounded so a missing terminator cannot walk off it.
 * Every loop over rt_list uses this, including callers. */
#define RT_MAX 32

extern const rt_desc_t * const rt_list[];

const rt_desc_t *rt_find(const char *name);

/* ---- Calls ---- */

/* run may return before the kernel finishes; sync is what waits. A target that
 * dispatches synchronously still has to accept the sync call. */

int    rt_open  (rt_dev_t *dev, const char *name);  /* NULL name probes */
void   rt_shut  (rt_dev_t *dev);
int    rt_load  (rt_dev_t *dev, const char *path, const char *kern,
                 rt_kern_t *out);
void   rt_unload(rt_dev_t *dev, rt_kern_t *k);
rt_ptr rt_alloc (rt_dev_t *dev, size_t n);
void   rt_free  (rt_dev_t *dev, rt_ptr p);
int    rt_h2d   (rt_dev_t *dev, rt_ptr dst, const void *src, size_t n);
int    rt_d2h   (rt_dev_t *dev, void *dst, rt_ptr src, size_t n);
int    rt_run   (rt_dev_t *dev, const rt_kern_t *k, const rt_dim_t *dim,
                 const rt_arg_t *args, uint32_t nargs);
int    rt_sync  (rt_dev_t *dev);

/* Memory both sides can reach. RT_EUNSUP where the target cannot, which is the
 * only honest answer; the caller writes one path either way. */
int    rt_hmap  (rt_dev_t *dev, void **hp, rt_ptr *dp, size_t n);
void   rt_hfre  (rt_dev_t *dev, void *hp);
void   rt_trak  (rt_dev_t *dev, rt_ptr p, size_t n, const char *lbl, int fl);

const char *rt_errs(const rt_dev_t *dev, int rc);

#endif /* BOOTH_EXEC_H */
