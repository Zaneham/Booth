/* booth_run.h -- the argument-staging and launch core kath's run verb drives. */

#ifndef BOOTH_RUN_H
#define BOOTH_RUN_H

#include "exec.h"

int  brun_dims(const char *s, uint32_t out[3]);
void brun_reset(void);
int  brun_addarg(const char *spec);
int  brun_launch(const char *artefact, const char *kernel,
                 const char *target, const rt_dim_t *dim);

#endif /* BOOTH_RUN_H */
