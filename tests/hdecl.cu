#include <stdio.h>
#include <stdint.h>
#include <cuda_runtime.h>

__device__ float dbl(float x)
{
    return x + x;
}

__global__ void vset(float *o, int64_t /*stride*/, int n, float v = 0.5f)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) o[i] = v;
}

__global__ void vdbl(float *o, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) o[i] = dbl(o[i]);
}

int main(void)
{
    enum { N = 64 };
    float  hf[N];
    float *d;
    int    bad = 0;

    cudaMalloc((void **)&d, N * sizeof(float));

    vset<<<(N + 63) / 64, 64>>>(d, (int64_t)1, N, 1.25f);
    vdbl<<<(N + 63) / 64, 64>>>(d, N);

    cudaDeviceSynchronize();
    cudaMemcpy(hf, d, N * sizeof(float), cudaMemcpyDeviceToHost);

    for (int i = 0; i < N; i++)
        if (hf[i] != 2.5f) bad++;

    cudaFree(d);

    if (bad != 0) {
        printf("hdecl: %d of %d lanes disagree\n", bad, (int)N);
        return 1;
    }
    printf("hdecl: %d lanes, o[0]=%g, all agree\n", (int)N, (double)hf[0]);
    return 0;
}
