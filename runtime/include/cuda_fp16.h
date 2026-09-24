#ifndef CUDA_FP16_H
#define CUDA_FP16_H

#if !defined(__CUDACC__)

#include <string.h>

typedef struct __half {
    unsigned short __x;
#if defined(__cplusplus)
    __half() = default;
    __half(float f);
#endif
} __half;
typedef struct __half2 { __half x, y; } __half2;

typedef __half  half;
typedef __half2 half2;

static unsigned __hbits(float f)
{
    unsigned u;

    memcpy(&u, &f, sizeof u);
    return u;
}

static float __hfrombits(unsigned u)
{
    float f;

    memcpy(&f, &u, sizeof f);
    return f;
}

static __half __float2half(float f)
{
    unsigned u = __hbits(f);
    unsigned s = (u >> 16) & 0x8000u;
    int      e = (int)((u >> 23) & 0xFFu) - 112;
    unsigned m = u & 0x7FFFFFu;
    __half   h;

    if (((u >> 23) & 0xFFu) == 0xFFu) {
        h.__x = (unsigned short)(s | 0x7C00u | (m != 0u ? 0x200u : 0u));
        return h;
    }
    if (e >= 0x1F) { h.__x = (unsigned short)(s | 0x7C00u); return h; }
    if (e <= 0) {
        if (e < -10) { h.__x = (unsigned short)s; return h; }
        m |= 0x800000u;
        {
            unsigned sh = (unsigned)(14 - e);
            unsigned r  = m & ((1u << sh) - 1u);
            unsigned v  = m >> sh;

            if (r > (1u << (sh - 1u))
                || (r == (1u << (sh - 1u)) && (v & 1u) != 0u)) v++;
            h.__x = (unsigned short)(s | v);
        }
        return h;
    }
    {
        unsigned v = ((unsigned)e << 10) | (m >> 13);
        unsigned r = m & 0x1FFFu;

        if (r > 0x1000u || (r == 0x1000u && (v & 1u) != 0u)) v++;
        h.__x = (unsigned short)(s | v);
    }
    return h;
}

static float __half2float(__half h)
{
    unsigned s = (unsigned)(h.__x & 0x8000u) << 16;
    unsigned e = (unsigned)(h.__x >> 10) & 0x1Fu;
    unsigned m = (unsigned)h.__x & 0x3FFu;

    if (e == 0u) {
        if (m == 0u) return __hfrombits(s);
        while ((m & 0x400u) == 0u) { m <<= 1; e--; }
        e++;
        m &= 0x3FFu;
    } else if (e == 0x1Fu) {
        return __hfrombits(s | 0x7F800000u | (m << 13));
    }
    return __hfrombits(s | ((e + 112u) << 23) | (m << 13));
}

#if defined(__cplusplus)
inline __half::__half(float f) { __x = __float2half(f).__x; }
#endif

static __half2 __halves2half2(__half a, __half b)
{
    __half2 r;

    r.x = a;
    r.y = b;
    return r;
}

static __half2 make_half2(__half a, __half b)
{
    return __halves2half2(a, b);
}

static __half2 __float2half2_rn(float f)
{
    return __halves2half2(__float2half(f), __float2half(f));
}

static __half __low2half(__half2 a)  { return a.x; }
static __half __high2half(__half2 a) { return a.y; }

#if defined(__cplusplus)

static inline __half operator+(__half a, __half b)
{ return __float2half(__half2float(a) + __half2float(b)); }
static inline __half operator-(__half a, __half b)
{ return __float2half(__half2float(a) - __half2float(b)); }
static inline __half operator*(__half a, __half b)
{ return __float2half(__half2float(a) * __half2float(b)); }
static inline __half operator/(__half a, __half b)
{ return __float2half(__half2float(a) / __half2float(b)); }

static inline bool operator==(__half a, __half b)
{ return __half2float(a) == __half2float(b); }
static inline bool operator!=(__half a, __half b)
{ return __half2float(a) != __half2float(b); }
static inline bool operator<(__half a, __half b)
{ return __half2float(a) < __half2float(b); }
static inline bool operator>(__half a, __half b)
{ return __half2float(a) > __half2float(b); }
static inline bool operator<=(__half a, __half b)
{ return __half2float(a) <= __half2float(b); }
static inline bool operator>=(__half a, __half b)
{ return __half2float(a) >= __half2float(b); }

#endif

#endif

#endif
