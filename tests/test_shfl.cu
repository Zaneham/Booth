/* test_shfl.cu — warp shuffle and vote intrinsics (sync and bare HIP forms) */
extern "C" __device__ int __ockl_get_local_id(int);

extern "C" __global__ void test_shfl(float *out, float *in) {
    int lid = __ockl_get_local_id(0);
    float val = in[lid];
    float s0 = __shfl_down_sync(0xFFFFFFFFu, val, 1, 64);
    float s1 = __shfl(val, 0);
    float s2 = __shfl_up(val, 1);
    float s3 = __shfl_down(val, 1);
    float s4 = __shfl_xor(val, 1);
    unsigned int b = __ballot(val > 0.0f);
    int a0 = __any(val > 0.0f);
    int a1 = __all(val > 0.0f);
    out[lid] = s0 + s1 + s2 + s3 + s4 + (float)b + (float)a0 + (float)a1;
}
