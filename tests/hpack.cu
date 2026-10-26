#include <stdio.h>
#include <cuda_runtime.h>

template <typename T, typename... Ps>
__global__ void vfan(T *o, int n, Ps... ps)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) o[i] = (T)sizeof...(Ps);
}

int main(void)
{
    enum { N = 64 };
    float  hf[N];
    float *da, *db, *dc;
    int    bad = 0;

    cudaMalloc((void **)&da, N * sizeof(float));
    cudaMalloc((void **)&db, N * sizeof(float));
    cudaMalloc((void **)&dc, N * sizeof(float));

    vfan<float, const float *, const float *><<<(N + 63) / 64, 64>>>(dc, N, da, db);

    cudaDeviceSynchronize();
    cudaMemcpy(hf, dc, N * sizeof(float), cudaMemcpyDeviceToHost);

    for (int i = 0; i < N; i++)
        if (hf[i] != 2.0f) bad++;

    cudaFree(da);
    cudaFree(db);
    cudaFree(dc);

    if (bad != 0) {
        printf("hpack: %d of %d lanes disagree\n", bad, (int)N);
        return 1;
    }
    printf("hpack: %d lanes, o[0]=%g, all agree\n", (int)N, (double)hf[0]);
    return 0;
}
