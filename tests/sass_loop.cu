__global__ void loopk(const int *a, int *o, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        int s = 0;
        for (int k = 0; k <= (i & 15); k++) s += a[k] * (k + 1);
        o[i] = s;
    }
}
