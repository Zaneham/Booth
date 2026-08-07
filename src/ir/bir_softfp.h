#ifndef BARRACUDA_BIR_SOFTFP_H
#define BARRACUDA_BIR_SOFTFP_H

#include "bir.h"

/* fp32 to libgcc-named soft-float calls, for targets with no FPU. Refuses if
   the runtime is not already in the module. */
int bir_softfp(bir_module_t *M);

#endif /* BARRACUDA_BIR_SOFTFP_H */
