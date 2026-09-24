#ifndef CUDA_BF16_H
#define CUDA_BF16_H

#if !defined(__CUDACC__)

#include <string.h>

typedef struct __nv_bfloat16 { unsigned short __x; } __nv_bfloat16;
typedef struct __nv_bfloat162 { __nv_bfloat16 x, y; } __nv_bfloat162;

typedef __nv_bfloat16  nv_bfloat16;
typedef __nv_bfloat162 nv_bfloat162;

static unsigned __bbits(float f)
{
    unsigned u;

    memcpy(&u, &f, sizeof u);
    return u;
}

static float __bfrombits(unsigned u)
{
    float f;

    memcpy(&f, &u, sizeof f);
    return f;
}

static __nv_bfloat16 __float2bfloat16(float f)
{
    unsigned      u = __bbits(f);
    __nv_bfloat16 b;

    if (((u >> 23) & 0xFFu) == 0xFFu) {
        b.__x = (unsigned short)(u >> 16);
        return b;
    }
    b.__x = (unsigned short)((u + 0x7FFFu + ((u >> 16) & 1u)) >> 16);
    return b;
}

static float __bfloat162float(__nv_bfloat16 b)
{
    return __bfrombits((unsigned)b.__x << 16);
}

static __nv_bfloat162 __halves2bfloat162(__nv_bfloat16 a, __nv_bfloat16 b)
{
    __nv_bfloat162 r;

    r.x = a;
    r.y = b;
    return r;
}

static __nv_bfloat162 __float2bfloat162_rn(float f)
{
    return __halves2bfloat162(__float2bfloat16(f), __float2bfloat16(f));
}

static __nv_bfloat16 __low2bfloat16(__nv_bfloat162 a)  { return a.x; }
static __nv_bfloat16 __high2bfloat16(__nv_bfloat162 a) { return a.y; }

#if defined(__cplusplus)

static inline __nv_bfloat16 operator+(__nv_bfloat16 a, __nv_bfloat16 b)
{ return __float2bfloat16(__bfloat162float(a) + __bfloat162float(b)); }
static inline __nv_bfloat16 operator-(__nv_bfloat16 a, __nv_bfloat16 b)
{ return __float2bfloat16(__bfloat162float(a) - __bfloat162float(b)); }
static inline __nv_bfloat16 operator*(__nv_bfloat16 a, __nv_bfloat16 b)
{ return __float2bfloat16(__bfloat162float(a) * __bfloat162float(b)); }
static inline __nv_bfloat16 operator/(__nv_bfloat16 a, __nv_bfloat16 b)
{ return __float2bfloat16(__bfloat162float(a) / __bfloat162float(b)); }

static inline bool operator==(__nv_bfloat16 a, __nv_bfloat16 b)
{ return __bfloat162float(a) == __bfloat162float(b); }
static inline bool operator!=(__nv_bfloat16 a, __nv_bfloat16 b)
{ return __bfloat162float(a) != __bfloat162float(b); }
static inline bool operator<(__nv_bfloat16 a, __nv_bfloat16 b)
{ return __bfloat162float(a) < __bfloat162float(b); }
static inline bool operator>(__nv_bfloat16 a, __nv_bfloat16 b)
{ return __bfloat162float(a) > __bfloat162float(b); }
static inline bool operator<=(__nv_bfloat16 a, __nv_bfloat16 b)
{ return __bfloat162float(a) <= __bfloat162float(b); }
static inline bool operator>=(__nv_bfloat16 a, __nv_bfloat16 b)
{ return __bfloat162float(a) >= __bfloat162float(b); }

#endif

#endif

#endif
