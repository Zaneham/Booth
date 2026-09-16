#include <stdio.h>
#include <cuda_runtime.h>

typedef void (*cpy_kernel_t)(const char *cx, char *cdst);

template <typename src_t, typename dst_t>
static __device__ void cpy1(const char *cxi, char *cdsti)
{
    *(dst_t *)cdsti = (dst_t)(*(const src_t *)cxi);
}

template <cpy_kernel_t cpy_1>
__global__ void cpyk(const char *cx, char *cdst, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) cpy_1(cx + i * 4, cdst + i * 4);
}

template <typename src_t, typename dst_t>
static void run(const char *cx, char *cdst, int n)
{
    cpyk<cpy1<src_t, dst_t>><<<(n + 63) / 64, 64>>>(cx, cdst, n);
}

int main(void)
{
    enum { N = 64 };
    float  hs[N], hd[N];
    char  *ds, *dd;
    int    bad = 0;

    for (int i = 0; i < N; i++) {
        hs[i] = 1.5f * (float)i;
        hd[i] = 0.0f;
    }

    cudaMalloc((void **)&ds, N * sizeof(float));
    cudaMalloc((void **)&dd, N * sizeof(float));
    cudaMemcpy(ds, hs, N * sizeof(float), cudaMemcpyHostToDevice);

    run<float, float>(ds, dd, N);

    cudaDeviceSynchronize();
    cudaMemcpy(hd, dd, N * sizeof(float), cudaMemcpyDeviceToHost);

    for (int i = 0; i < N; i++)
        if (hd[i] != hs[i]) bad++;

    cudaFree(ds);
    cudaFree(dd);

    if (bad != 0) {
        printf("htid: %d of %d lanes disagree\n", bad, (int)N);
        return 1;
    }
    printf("htid: %d lanes, d[1]=%g d[63]=%g, all agree\n",
           (int)N, (double)hd[1], (double)hd[63]);
    return 0;
}
