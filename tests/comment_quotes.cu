/* comment_quotes.cu -- block comments may contain quote characters */

/* The kernel's first instruction is not here: ' " ` */
__global__ void comment_quotes(int *out)
{
    out[0] = 1;
}
