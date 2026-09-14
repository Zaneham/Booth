template <int I_, typename T>
struct tile { static constexpr int I = I_; };

template <int I_>
struct tile<I_, double> { static constexpr int I = I_ * 100; };

template <typename T>
__device__ int fold(int base) {
    typedef tile<3, T> tq;
    auto g = [&](int b) -> int {
        int s = b;
        for (int i = 0; i < tq::I; ++i) s += i;
        return s;
    };
    return g(base);
}

__global__ void tsmall(int *o, int base) { o[0] = fold<float>(base); }
__global__ void tbig(int *o, int base)   { o[0] = fold<double>(base); }
