/* bir_sroa.h — Scalar Replacement of Aggregates */
#ifndef BIR_SROA_H
#define BIR_SROA_H

#include "bir.h"

/* Break struct allocas into per-field scalar allocas.
 * Run BEFORE mem2reg so it can promote the scalars to SSA. */
void bir_sroa(bir_module_t *M);

#endif
