/* device_calls.cu -- fixture for the __device__ call inliner (issue #101).
 * Covers a straight-line helper, one with control flow and several returns,
 * a device function that calls another device function, and a call sitting
 * inside a loop. The GPU backends must inline every one of these away; the
 * scalar CPU backend keeps them as real calls. */

__device__ int sq(int n) { return n * n; }

__device__ int clampv(int x, int lo, int hi) {
    if (x < lo) return lo;
    if (x > hi) return hi;
    return x;
}

__device__ int norm(int x) {
    int c = clampv(x, 0, 100);
    if (c > 50) return c - 50;
    return c;
}

__global__ void k(int *o) {
    int t = threadIdx.x;
    int acc = 0;
    for (int i = 0; i < 4; i++)
        acc += clampv(sq(o[i]) + norm(i), 1, 9);
    o[t] = acc;
}
