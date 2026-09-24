#include <stdio.h>
#include <cuda_runtime.h>

#define LAUNCH(k, n, o, v) k<<<((n) + 63) / 64, 64>>>(o, v, n)

struct vec2 { float a; float b; };

template <typename T>
__global__ void fill2(T *o, T v, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) o[i] = v;
}

template <typename K, typename T>
static void fwd(K k, T *o, T v, int n)
{
    k<<<(n + 63) / 64, 64>>>(o, v, n);
}

int main(void)
{
    enum { N = 64 };
    vec2   hp[N];
    float  hf[N];
    vec2  *dp;
    float *df;
    vec2   one;
    int    bad = 0;

    one.a = 1.0f;
    one.b = 2.0f;
    cudaMalloc((void **)&dp, N * sizeof(vec2));
    cudaMalloc((void **)&df, N * sizeof(float));

    LAUNCH(fill2<vec2>, N, dp, one);
    fwd(fill2<float>, df, 0.5f, N);

    cudaDeviceSynchronize();
    cudaMemcpy(hp, dp, N * sizeof(vec2), cudaMemcpyDeviceToHost);
    cudaMemcpy(hf, df, N * sizeof(float), cudaMemcpyDeviceToHost);

    for (int i = 0; i < N; i++) {
        if (hp[i].a != 1.0f || hp[i].b != 2.0f) bad++;
        if (hf[i] != 0.5f) bad++;
    }

    cudaFree(dp);
    cudaFree(df);

    if (bad != 0) {
        printf("hmac: %d disagreements over %d lanes\n", bad, (int)N);
        return 1;
    }
    printf("hmac: %d lanes, p=(%g,%g) f=%g, all agree\n",
           (int)N, (double)hp[0].a, (double)hp[0].b, (double)hf[0]);
    return 0;
}
