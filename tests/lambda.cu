__global__ void lamval(const float *src, float *dst, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int base = n * 7;
    auto load = [=](int off) -> float { return src[base + off]; };
    if (i < n) dst[i] = load(1) + load(2);
}

__global__ void lamref(float *dst, int n)
{
    int acc = 0;
    auto bump = [&](int by) { acc += by; };
    bump(3);
    bump(4);
    if (threadIdx.x < (unsigned)n) dst[threadIdx.x] = (float)acc;
}

__global__ void lamnone(float *dst)
{
    auto sq = [](float x) -> float { return x * x; };
    dst[threadIdx.x] = sq(3.0f);
}

__global__ void lamnow(float *dst)
{
    dst[threadIdx.x] = [](float a, float b) -> float { return a - b; }(9.0f, 4.0f);
}

__global__ void lammix(float *dst, int a, int b)
{
    int t = a * b;
    auto f = [=, &t](int k) -> int { t = t + k + a; return t - b; };
    dst[threadIdx.x] = (float)(f(1) + f(2) + t);
}

__global__ void lamnest(float *dst, int n)
{
    int a = n * 3;
    auto outer = [=](int k) -> int {
        auto inner = [=](int j) -> int { return a + j; };
        return inner(k) * 2;
    };
    dst[threadIdx.x] = (float)outer(4);
}
