/* front.h -- the verbs a user drives kath through, and the hook the run
 * verb uses to learn the kernel name out of a module it just built. */

#ifndef BOOTH_FRONT_H
#define BOOTH_FRONT_H

#define KATH_KERN_MAX 128

typedef struct kath_out {
    char kernel[KATH_KERN_MAX];
    int  have;
} kath_out_t;

/* The old main. A flag argv in, an artefact out, and the __global__ name
 * written into out when one is asked for. */
int kath_compile(int argc, char *argv[], kath_out_t *out);

int booth_is_verb(const char *s);
int booth_verb(int argc, char *argv[]);

#endif /* BOOTH_FRONT_H */
