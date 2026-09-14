__global__ void shrk(const int *a, int *o, int n)
{
    __shared__ int s[256];
    int t = threadIdx.x;
    int i = blockIdx.x * blockDim.x + t;
    s[t] = (i < n) ? a[i] : 0;
    __syncthreads();
    int v = s[t] + s[(t + 1) & 255];
    if (i < n) o[i] = v;
}
