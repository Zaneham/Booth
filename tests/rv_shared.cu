/* rv_shared.cu -- fixture for __shared__ lowering. The tile is placed in the
 * L1 slab carved off the top, so the kernel must materialise an absolute
 * address at TD_L1_SHARED_BASE rather than a stack offset. */

__global__ void k(int *o) {
    __shared__ int tile[64];
    int t = threadIdx.x;
    tile[t] = o[t] * 2;
    o[t] = tile[t];
}
