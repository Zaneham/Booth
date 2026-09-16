#include <stdio.h>
#include <cuda_runtime.h>

typedef void (*kfn_t)(float *, int);

template <int S, bool B>
__global__ void scale(float *o, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= n) return;
    o[i] = B ? (float)(S * 2) : (float)S;
}

template <int S>
static void fire(float *o, int n, kfn_t k)
{
    k<<<(n + 63) / 64, 64>>>(o, n);
}

template <int S>
static void run(float *o, int n, int dbl)
{
    kfn_t k;

    if (dbl) k = scale<S, true>;
    else     k = scale<S, false>;
    fire<S>(o, n, k);
}

int main(void)
{
    enum { N = 64 };
    float  h[N];
    float *d;
    int    bad = 0;

    cudaMalloc((void **)&d, N * sizeof(float));
    run<3>(d, N, 1);
    cudaDeviceSynchronize();
    cudaMemcpy(h, d, N * sizeof(float), cudaMemcpyDeviceToHost);
    for (int i = 0; i < N; i++)
        if (h[i] != 6.0f) bad++;
    cudaFree(d);

    if (bad != 0) {
        printf("hfnb: %d disagreements over %d lanes\n", bad, (int)N);
        return 1;
    }
    printf("hfnb: %d lanes, v=%g, all agree\n", (int)N, (double)h[0]);
    return 0;
}
