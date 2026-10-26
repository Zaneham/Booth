#include <stdio.h>
#include <cuda_runtime.h>

__device__ float twice(float x)
{
    return x + x;
}

__global__ void vadd(const float *a, const float *b, float *c, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) c[i] = a[i] + b[i];
}

__global__ void vdbl(float *c, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) c[i] = twice(c[i]);
}

template <typename T>
__global__ void vfill(T *o, T v, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) o[i] = v;
}

int main(void)
{
    enum { N = 128 };
    float hf[N];
    int   hi[N];
    float *da, *db, *dc;
    int   *di;
    int    bad = 0;

    cudaMalloc((void **)&da, N * sizeof(float));
    cudaMalloc((void **)&db, N * sizeof(float));
    cudaMalloc((void **)&dc, N * sizeof(float));
    cudaMalloc((void **)&di, N * sizeof(int));

    vfill<float><<<(N + 63) / 64, 64>>>(da, 1.5f, N);
    vfill<float><<<(N + 63) / 64, 64>>>(db, 2.25f, N);
    vfill<int><<<(N + 63) / 64, 64>>>(di, 7, N);

    vadd<<<(N + 63) / 64, 64>>>(da, db, dc, N);
    vdbl<<<(N + 63) / 64, 64>>>(dc, N);

    cudaDeviceSynchronize();
    cudaMemcpy(hf, dc, N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(hi, di, N * sizeof(int), cudaMemcpyDeviceToHost);

    for (int i = 0; i < N; i++) {
        if (hf[i] != 7.5f) bad++;
        if (hi[i] != 7) bad++;
    }

    cudaFree(da);
    cudaFree(db);
    cudaFree(dc);
    cudaFree(di);

    if (bad != 0) {
        printf("hmulti: %d disagreements over %d lanes\n", bad, (int)N);
        return 1;
    }
    printf("hmulti: %d lanes, f=%g i=%d, all agree\n",
           (int)N, (double)hf[0], hi[0]);
    return 0;
}
