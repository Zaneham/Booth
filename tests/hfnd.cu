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

template <typename K, typename... A>
static void fire(K k, int g, int b, A &&... a)
{
    k<<<g, b>>>(std::forward<A>(a)...);
}

template <int S>
static void mid(float *o, int n, kfn_t k)
{
    fire(k, (n + 63) / 64, 64, o, n);
}

template <int S>
static void run(float *o, int n, int dbl)
{
    kfn_t k;

    if (dbl) {
        constexpr bool B = true;
        k = scale<S, B>;
    } else {
        constexpr bool B = false;
        k = scale<S, B>;
    }
    mid<S>(o, n, k);
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
        printf("hfnd: %d disagreements over %d lanes\n", bad, (int)N);
        return 1;
    }
    printf("hfnd: %d lanes, v=%g, all agree\n", (int)N, (double)h[0]);
    return 0;
}
