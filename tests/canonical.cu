/* canonical.cu — Classic CUDA patterns from NVIDIA samples and GPU Gems,
 * adapted for Booth's current parser capabilities.
 *
 * Original sources: NVIDIA cuda-samples, GPU Gems 3 Ch.39, various
 * NVIDIA blog posts. Adapted by removing compound assignment (+=),
 * const qualifiers, __inline__, 2D shared arrays, and other syntax
 * we haven't bothered to implement yet. The algorithms are intact;
 * only the syntactic sugar has been scraped off. */

#define TILE 16

/* ---- Tiled SGEMM (non-square, bounds-checked) ---- */
/* The proper version with M/N/K dimensions and bounds checks.
 * Unlike the notgpt.cu version which only works for square matrices
 * that happen to be multiples of 16, this one handles the real world
 * where matrices are whatever size the user fancies. */

__global__ void matmul_general(float *C, float *A, float *B,
                               int M, int N, int K)
{
    __shared__ float As[256];
    __shared__ float Bs[256];

    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;
    float sum = 0.0f;

    int numTiles = (K + TILE - 1) / TILE;
    for (int t = 0; t < numTiles; t = t + 1) {
        int aCol = t * TILE + threadIdx.x;
        int bRow = t * TILE + threadIdx.y;

        /* Bounds-checked load: zero if out of range.
         * The ternary here is load-bearing. Remove it and
         * enjoy your segfault. */
        float a_val = (row < M && aCol < K) ? A[row * K + aCol] : 0.0f;
        float b_val = (bRow < K && col < N) ? B[bRow * N + col] : 0.0f;
        As[threadIdx.y * TILE + threadIdx.x] = a_val;
        Bs[threadIdx.y * TILE + threadIdx.x] = b_val;
        __syncthreads();

        for (int i = 0; i < TILE; i = i + 1)
            sum = sum + As[threadIdx.y * TILE + i] * Bs[i * TILE + threadIdx.x];

        __syncthreads();
    }

    if (row < M && col < N)
        C[row * N + col] = sum;
}


/* ---- Block Reduce with Grid-Stride Loop ---- */
/* The "proper" two-phase reduction: grid-stride loop to accumulate
 * per-thread sums, then warp shuffle to reduce within warps, then
 * shared memory to reduce across warps. This is what production
 * code actually looks like, minus the templates. */

__device__ float warp_reduce(float v)
{
    int mask = 0xFFFFFFFF;
    float v1 = v  + __shfl_down_sync(mask, v,  16);
    float v2 = v1 + __shfl_down_sync(mask, v1, 8);
    float v3 = v2 + __shfl_down_sync(mask, v2, 4);
    float v4 = v3 + __shfl_down_sync(mask, v3, 2);
    float v5 = v4 + __shfl_down_sync(mask, v4, 1);
    return v5;
}

__device__ float block_reduce(float v)
{
    __shared__ float warp_vals[32];
    int lane = threadIdx.x & 31;
    int wid = threadIdx.x >> 5;

    float reduced = warp_reduce(v);

    if (lane == 0)
        warp_vals[wid] = reduced;
    __syncthreads();

    /* First warp loads per-warp sums and reduces them.
     * Everyone else contemplates the void. */
    int num_warps = (blockDim.x + 31) / 32;
    float loaded = (threadIdx.x < num_warps) ? warp_vals[lane] : 0.0f;
    float result = loaded;
    if (wid == 0)
        result = warp_reduce(loaded);

    return result;
}

__global__ void reduce_sum(float *output, float *input, int n)
{
    float sum = 0.0f;

    /* Grid-stride loop: the civilised way to handle arrays
     * larger than your grid. Each thread accumulates multiple
     * elements before the reduction begins. */
    int stride = blockDim.x * gridDim.x;
    int start = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = start; i < n; i = i + stride)
        sum = sum + input[i];

    sum = block_reduce(sum);
    if (threadIdx.x == 0)
        atomicAdd(output, sum);
}


/* ---- Histogram (cooperative init, grid-stride) ---- */
/* The textbook shared-memory privatisation pattern. Each block
 * maintains its own histogram in shared memory, then merges to
 * global. The cooperative zeroing loop handles arbitrary block
 * sizes — no more "hope blockDim.x == 256" nonsense. */

#define NUM_BINS 256

__global__ void histogram_proper(int *hist, int *data, int n)
{
    __shared__ int smem[NUM_BINS];

    /* Cooperative zeroing: works for any blockDim.x. */
    for (int i = threadIdx.x; i < NUM_BINS; i = i + blockDim.x)
        smem[i] = 0;
    __syncthreads();

    /* Grid-stride accumulation */
    int stride = blockDim.x * gridDim.x;
    int start = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = start; i < n; i = i + stride) {
        int bin = data[i] & 255;
        atomicAdd(&smem[bin], 1);
    }

    __syncthreads();

    /* Merge to global: also cooperative. */
    for (int i = threadIdx.x; i < NUM_BINS; i = i + blockDim.x)
        atomicAdd(&hist[i], smem[i]);
}


/* ---- Dot Product (warp shuffle + shared) ---- */
/* Because sometimes you just need a dot product and don't want
 * to think about it too hard. Uses the same block_reduce pattern
 * as above, proving that code reuse works even in CUDA. */

__global__ void dot_product(float *result, float *a, float *b, int n)
{
    float sum = 0.0f;
    int stride = blockDim.x * gridDim.x;
    int start = blockIdx.x * blockDim.x + threadIdx.x;

    for (int i = start; i < n; i = i + stride)
        sum = sum + a[i] * b[i];

    sum = block_reduce(sum);
    if (threadIdx.x == 0)
        atomicAdd(result, sum);
}


/* ---- SAXPY (the "hello world" nobody bothers to test) ---- */
/* So simple it's almost insulting, yet it exercises the basic
 * kernel launch pattern that everything else depends on. */

__global__ void saxpy(float *y, float *x, float a, int n)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n)
        y[idx] = a * x[idx] + y[idx];
}


/* ---- Transpose (shared memory, coalesced writes) ---- */
/* The shared-memory transpose trick: coalesced reads into shared,
 * coalesced writes from shared. The +1 padding to avoid bank
 * conflicts is omitted because we flatten to 1D anyway. */

#define TRANS_TILE 16

__global__ void transpose(float *out, float *in, int width, int height)
{
    __shared__ float tile[256];

    int xIdx = blockIdx.x * TRANS_TILE + threadIdx.x;
    int yIdx = blockIdx.y * TRANS_TILE + threadIdx.y;

    if (xIdx < width && yIdx < height)
        tile[threadIdx.y * TRANS_TILE + threadIdx.x] = in[yIdx * width + xIdx];

    __syncthreads();

    /* Swap x and y for the write. The transposition happens here,
     * in this single index swap. All that shared memory business
     * above was just to make the memory access pattern not terrible. */
    int newX = blockIdx.y * TRANS_TILE + threadIdx.x;
    int newY = blockIdx.x * TRANS_TILE + threadIdx.y;

    if (newX < height && newY < width)
        out[newY * height + newX] = tile[threadIdx.x * TRANS_TILE + threadIdx.y];
}


/* ---- Moving Average (sliding window in registers) ---- */
/* A 1D moving average with a window of 5. Each thread processes
 * one output element by reading 5 input elements. Not the most
 * efficient approach but it exercises the register pressure. */

__global__ void moving_avg(float *out, float *in, int n)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= n) return;

    float sum = 0.0f;
    int count = 0;

    for (int d = -2; d <= 2; d = d + 1) {
        int j = idx + d;
        if (j >= 0 && j < n) {
            sum = sum + in[j];
            count = count + 1;
        }
    }

    out[idx] = sum / (float)count;
}


/* ---- Exclusive Scan (simplified Blelloch) ---- */
/* A cleaned-up Blelloch scan for 256-element blocks. Uses power-of-two
 * stride doubling and halving. The variable naming is slightly less
 * cryptic than the GPU Gems version, which is not saying much. */

__launch_bounds__(128)
__global__ void exclusive_scan(int *out, int *in, int n)
{
    __shared__ int buf[256];
    int tid = threadIdx.x;
    int base = blockIdx.x * 256;

    /* Each thread loads two elements. */
    buf[tid] = (base + tid < n) ? in[base + tid] : 0;
    buf[tid + 128] = (base + tid + 128 < n) ? in[base + tid + 128] : 0;
    __syncthreads();

    /* Up-sweep: accumulate from leaves to root. */
    int stride = 1;
    for (int half = 128; half > 0; half = half / 2) {
        __syncthreads();
        if (tid < half) {
            int left = stride * (2 * tid + 1) - 1;
            int right = stride * (2 * tid + 2) - 1;
            buf[right] = buf[right] + buf[left];
        }
        stride = stride * 2;
    }

    /* Clear the last element for exclusive scan. */
    if (tid == 0)
        buf[255] = 0;

    /* Down-sweep: distribute partial sums back down. */
    for (int half = 1; half < 256; half = half * 2) {
        stride = stride / 2;
        __syncthreads();
        if (tid < half) {
            int left = stride * (2 * tid + 1) - 1;
            int right = stride * (2 * tid + 2) - 1;
            int tmp = buf[left];
            buf[left] = buf[right];
            buf[right] = buf[right] + tmp;
        }
    }

    __syncthreads();
    if (base + tid < n) out[base + tid] = buf[tid];
    if (base + tid + 128 < n) out[base + tid + 128] = buf[tid + 128];
}
