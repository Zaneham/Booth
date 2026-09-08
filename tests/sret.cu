/* sret.cu -- every shape of struct a __device__ function can hand back.
 *
 * One kernel, one slot per shape, so a wrong return is a wrong number rather
 * than a wrong instruction nobody reads. tests/tnv_sret.c runs it on a card. */

struct Pair  { int a; int b; };
struct Inner { int x; int y; };
struct Outer { Inner i; int z; };
struct Empty { };
struct Wide  { int v[24]; };
struct Mixed { float2 f; int n; };
union  U     { int i; float f; };
struct WithU { U u; int t; };

__device__ Pair  mkp(int x) { Pair p; p.a = x; p.b = x + 1; return p; }
__device__ Outer mko(int n) { Outer o; o.i.x = n; o.i.y = n + 1; o.z = n + 2; return o; }
__device__ Empty mke(void)  { Empty e; return e; }
__device__ Wide  mkw(int n) { Wide w; for (int i = 0; i < 24; i = i + 1) w.v[i] = n + i; return w; }
__device__ Mixed mkm(int n) { Mixed m; m.f.x = (float)n; m.f.y = (float)(n + 1); m.n = n + 2; return m; }
__device__ Pair  fwd(int x) { return mkp(x + 1); }
__device__ WithU mku(int n) { WithU w; w.u.i = n; w.t = n + 1; return w; }
__device__ int   sum2(Pair p) { return p.a + p.b; }

__global__ void sret_probe(int *o, const int *in)
{
    int n = in[0];

    o[0] = mkp(n).b;
    Pair p = mkp(n);
    o[1] = p.a + p.b;
    Outer ou = mko(n);
    o[2] = ou.i.y;
    o[3] = ou.z;
    Empty e = mke();
    o[4] = n * 3;
    Wide w = mkw(n);
    o[5] = w.v[23];
    o[6] = w.v[0];
    Mixed m = mkm(n);
    o[7] = (int)m.f.y;
    o[8] = m.n;
    Pair q = fwd(n);
    o[9] = q.b;
    WithU u = mku(n);
    o[10] = u.u.i;
    o[11] = u.t;
    o[12] = sum2(mkp(n));
    Pair r = p;
    r.a = 100;
    o[13] = p.a + r.a;
}
