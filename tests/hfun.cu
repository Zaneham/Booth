#include <stdio.h>
#include <cuda_runtime.h>

template <int tag>
__global__ void kern(float *o, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) o[i] = (float)tag;
}

template <int tag>
static void fire(float *o, int n)
{
    kern<tag><<<(n + 63) / 64, 64>>>(o, n);
}

template <int tag>
struct maker {
    template <typename T>
    void operator()(T *o, int n) const
    {
        fire<tag>(o, n);
    }
};

template <typename M>
static void thru(float *o, int n)
{
    M()(o, n);
}

int main(void)
{
    enum { N = 64 };
    float  h[N];
    float *d;
    int    bad = 0;

    cudaMalloc((void **)&d, N * sizeof(float));
    fire<3>(d, N);
    cudaDeviceSynchronize();
    thru<maker<5> >(d, N);
    cudaDeviceSynchronize();
    cudaMemcpy(h, d, N * sizeof(float), cudaMemcpyDeviceToHost);
    for (int i = 0; i < N; i++)
        if (h[i] != 5.0f) bad++;
    cudaFree(d);

    if (bad != 0) {
        printf("hfun: %d disagreements over %d lanes\n", bad, (int)N);
        return 1;
    }
    printf("hfun: %d lanes, v=%g, all agree\n", (int)N, (double)h[0]);
    return 0;
}
