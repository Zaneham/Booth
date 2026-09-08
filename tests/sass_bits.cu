__global__ void bitk(const int *a, int *o, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        unsigned x = (unsigned)a[i];
        unsigned y = (x << 3) ^ (x >> 2);
        y = (y & 0xF0F0F0F0u) | (~y & 0x0F0F0F0Fu);
        long long w = (long long)a[i] * 1103515245LL + 12345LL;
        int s = (a[i] < 0) ? -1 : 1;
        o[i] = (int)(y + (unsigned)(w >> 7)) * s - (a[i] >> 3);
    }
}
