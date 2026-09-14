__global__ void hpack(const float *f, const half2 *h, half2 *o, float2 *g)
{
    int i = threadIdx.x;
    const float2 *f2 = (const float2 *) f;

    o[i*10+0] = __float22half2_rn(f2[i]);
    o[i*10+1] = __floats2half2_rn(f[2*i], f[2*i+1]);
    o[i*10+2] = __float2half2_rn(f[2*i]);
    o[i*10+3] = __halves2half2(__float2half(f[2*i]), __float2half(f[2*i+1]));
    o[i*10+4] = __lows2half2(h[2*i], h[2*i+1]);
    o[i*10+5] = __highs2half2(h[2*i], h[2*i+1]);
    o[i*10+6] = __low2half2(h[2*i]);
    o[i*10+7] = __high2half2(h[2*i]);
    o[i*10+8] = __lowhigh2highlow(h[2*i]);
    o[i*10+9] = __halves2half2(__low2half(h[2*i]), __high2half(h[2*i+1]));
    g[i] = __half22float2(h[2*i]);
}

__global__ void bpack(const float *f, const __nv_bfloat162 *b,
                      __nv_bfloat162 *o, float2 *g, __nv_bfloat16 *s)
{
    int i = threadIdx.x;
    const float2 *f2 = (const float2 *) f;

    o[i*9+0] = __float22bfloat162_rn(f2[i]);
    o[i*9+1] = __floats2bfloat162_rn(f[2*i], f[2*i+1]);
    o[i*9+2] = __float2bfloat162_rn(f[2*i]);
    o[i*9+3] = __halves2bfloat162(__float2bfloat16(f[2*i]),
                                  __float2bfloat16(f[2*i+1]));
    o[i*9+4] = __bfloat162bfloat162(__float2bfloat16(f[2*i]));
    o[i*9+5] = __lows2bfloat162(b[2*i], b[2*i+1]);
    o[i*9+6] = __highs2bfloat162(b[2*i], b[2*i+1]);
    o[i*9+7] = __low2bfloat162(b[2*i]);
    o[i*9+8] = __lowhigh2highlow(b[2*i]);
    g[i] = __bfloat1622float2(b[2*i]);
    s[2*i]   = __low2bfloat16(b[2*i]);
    s[2*i+1] = __high2bfloat16(b[2*i]);
}
