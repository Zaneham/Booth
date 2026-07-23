/* rv_switch.cu -- fixture for the RV32 switch lowering. Three cases plus a
 * default, so the isel must emit a three-deep compare chain and fall through
 * to the default block. */

__global__ void k(int *o) {
    int t = threadIdx.x;
    int r;
    switch (t) {
        case 0:  r = 10; break;
        case 1:  r = 20; break;
        case 7:  r = 30; break;
        default: r = 99; break;
    }
    o[t] = r;
}
