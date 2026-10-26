#include <stdio.h>
#include <cuda_runtime.h>

__global__ void vadd(const float *a, const float *b, float *c, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) c[i] = a[i] + b[i];
}

int main(void)
{
    enum { N = 256 };
    float ha[N], hb[N], hc[N];
    float *da, *db, *dc;
    int    bad = 0;

    for (int i = 0; i < N; i++) {
        ha[i] = (float)i;
        hb[i] = 0.25f * (float)i;
        hc[i] = 0.0f;
    }

    cudaMalloc((void **)&da, N * sizeof(float));
    cudaMalloc((void **)&db, N * sizeof(float));
    cudaMalloc((void **)&dc, N * sizeof(float));
    cudaMemcpy(da, ha, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(db, hb, N * sizeof(float), cudaMemcpyHostToDevice);

    vadd<<<(N + 63) / 64, 64>>>(da, db, dc, N);

    cudaDeviceSynchronize();
    cudaMemcpy(hc, dc, N * sizeof(float), cudaMemcpyDeviceToHost);

    for (int i = 0; i < N; i++)
        if (hc[i] != ha[i] + hb[i]) bad++;

    cudaFree(da);
    cudaFree(db);
    cudaFree(dc);

    if (bad != 0) {
        printf("hvadd: %d of %d lanes disagree\n", bad, (int)N);
        return 1;
    }
    printf("hvadd: %d lanes, c[1]=%g c[255]=%g, all agree\n",
           (int)N, (double)hc[1], (double)hc[255]);
    return 0;
}
