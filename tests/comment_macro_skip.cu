/* comment_macro_skip.cu -- preprocessor comments stay opaque */

#define COMMENT_MACRO 23 /* trailing body comment */

/* COMMENT_MACRO should remain literal inside this block comment. */
__global__ void comment_macro_skip(int *out)
{
    out[0] = COMMENT_MACRO;
}
