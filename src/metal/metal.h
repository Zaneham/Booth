#ifndef BARRACUDA_METAL_H
#define BARRACUDA_METAL_H

#include "bir.h"

/* Apple Metal backend, which is to say the part of BarraCUDA that lowers
 * your perfectly serviceable CUDA source into the Metal Shading Language
 * so that Apple's own driver can compile it at load time, in exactly the
 * way we already convince the NVIDIA driver to compile PTX while we
 * stand a respectful distance away and pretend we did all the difficult
 * parts. We do not descend any deeper than MSL into AIR or AGX bytecode,
 * because Apple did not invite anyone in there and the Asahi Linux folk
 * have already spent literal years doing the rest of that homework on
 * everyone else's behalf. */

#define BC_ERR_METAL    -8

typedef struct metal_module_t {
    int placeholder;
} metal_module_t;

/* Lower BIR to a Metal module representation. Returns BC_OK or an error. */
int  metal_compile(const bir_module_t *bir, metal_module_t *mm);

/* Emit MSL source to `path`. Caller pairs with Metal compiler / runtime. */
int  metal_emit_msl(const metal_module_t *mm, const char *path);

#endif /* BARRACUDA_METAL_H */
