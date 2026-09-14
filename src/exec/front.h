/* front.h -- kath's verbs */

#ifndef BOOTH_FRONT_H
#define BOOTH_FRONT_H

#define KATH_KERN_MAX 128

typedef struct kath_out {
    char kernel[KATH_KERN_MAX];
    int  have;
} kath_out_t;

/* The old main, writing the __global__ name into out when asked */
int kath_compile(int argc, char *argv[], kath_out_t *out);

int booth_is_verb(const char *s);
int booth_verb(int argc, char *argv[]);

#endif /* BOOTH_FRONT_H */
