__global__ void cf_deadarm(int *out, int x) {
    if ((128 <= 1024) && (128 % 32) == 0) out[0] = x;
    else                                  out[0] = 99;
}
