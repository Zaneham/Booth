#include <stdio.h>
#include <cuda_runtime.h>

enum mode { M_ADD = 0, M_MUL = 1 };

struct vec2 { float a; float b; };

template <typename T, mode M>
__global__ void apply(T *o, vec2 p, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= n) return;
    o[i] = (M == M_MUL) ? (T)(p.a * p.b) : (T)(p.a + p.b);
}

template <typename T>
static void run(T *o, vec2 p, int n, int mul)
{
    if (mul) apply<T, M_MUL><<<(n + 63) / 64, 64>>>(o, p, n);
    else     apply<T, M_ADD><<<(n + 63) / 64, 64>>>(o, p, n);
}

int main(void)
{
    enum { N = 64 };
    float  hf[N];
    int    hi[N];
    float *df;
    int   *di;
    vec2   p;
    int    bad = 0;

    p.a = 3.0f;
    p.b = 4.0f;
    cudaMalloc((void **)&df, N * sizeof(float));
    cudaMalloc((void **)&di, N * sizeof(int));

    run<float>(df, p, N, 1);
    run<int>(di, p, N, 0);

    cudaDeviceSynchronize();
    cudaMemcpy(hf, df, N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(hi, di, N * sizeof(int), cudaMemcpyDeviceToHost);

    for (int i = 0; i < N; i++) {
        if (hf[i] != 12.0f) bad++;
        if (hi[i] != 7) bad++;
    }

    cudaFree(df);
    cudaFree(di);

    if (bad != 0) {
        printf("htfn: %d disagreements over %d lanes\n", bad, (int)N);
        return 1;
    }
    printf("htfn: %d lanes, f=%g i=%d, all agree\n",
           (int)N, (double)hf[0], hi[0]);
    return 0;
}
