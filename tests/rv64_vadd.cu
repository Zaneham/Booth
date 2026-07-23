/* rv64_vadd.cu -- kernel for the RV64 QEMU job. Paired with rv64_host.c,
 * which links against it and checks the arithmetic actually came out right. */

__global__ void vadd(int *out, int *a, int *b)
{
    int i = threadIdx.x;
    out[i] = a[i] + b[i];
}
