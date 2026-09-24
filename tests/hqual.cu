#include <cuda_runtime.h>

struct qop {
    int v;

    __device__ qop(int a) : v(a) {}
    __device__ int operator()(int x) const { return x + v; }
    static __device__ int dbl(int x) { return 2 * x; }
    __device__ __forceinline__ int trip(int x) const { return 3 * x; }
    __host__ __device__ int keep(int x) const { return x; }
};

__global__ void qk(int *o, int n)
{
    int i = threadIdx.x;

    if (i < n) o[i] = i;
}

void qgo(int *o, int n)
{
    qk<<<1, 64>>>(o, n);
}
