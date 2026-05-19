#ifndef BARRACUDA_INTEL_H
#define BARRACUDA_INTEL_H

#include "bir.h"

/* Intel Arc Xe backend, which rounds out the small but growing club of
 * "let somebody else's driver finish the job" arrangements that we have
 * struck with the three GPU vendors who actually let us in the door.
 * We emit SPIR-V, which is the closest thing the graphics industry has
 * ever produced to a sensible intermediate representation, in that it
 * was designed by a committee and yet has somehow contrived to be both
 * structured and publicly documented at the same time. Level Zero and
 * OpenCL will accept it without complaint, and the Intel driver will
 * then JIT it down to whichever Xe variant the owner of the machine
 * has actually paid Intel for. Going below SPIR-V into the native Xe
 * bytecode is the sort of thing one undertakes after several months of
 * staring directly into the source of IGC and reading conference slides
 * that were not strictly meant to leak, and would buy us essentially
 * nothing that the driver was not already going to do on our behalf. */

#define BC_ERR_INTEL    -9

typedef enum {
    INTEL_TARGET_XE_LPG = 0,   /* Arc / integrated (Meteor Lake etc) */
    INTEL_TARGET_XE_HPG,       /* Alchemist, Battlemage */
    INTEL_TARGET_XE_HPC,       /* Ponte Vecchio */
    INTEL_TARGET_XE2           /* Lunar Lake, next-gen Arc */
} intel_target_t;

typedef struct intel_module_t {
    intel_target_t target;
    int placeholder;
} intel_module_t;

int  intel_compile(const bir_module_t *bir, intel_module_t *im,
                   intel_target_t target);

int  intel_emit_spirv(const intel_module_t *im, const char *path);

#endif /* BARRACUDA_INTEL_H */
