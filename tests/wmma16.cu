/* Every WMMA shape and type Booth lowers, one kernel each. */
extern "C" __global__ void wrc(const half *a, const half *b, float *d,
                               int lda, int ldb, int ldd)
{
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> af;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> bf;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> cf;

    wmma::load_matrix_sync(af, a, lda);
    wmma::load_matrix_sync(bf, b, ldb);
    wmma::load_matrix_sync(cf, d, ldd, wmma::mem_row_major);
    wmma::mma_sync(cf, af, bf, cf);
    wmma::store_matrix_sync(d, cf, ldd, wmma::mem_row_major);
}

extern "C" __global__ void wrr(const half *a, const half *b, float *d,
                               int lda, int ldb, int ldd)
{
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> af;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> bf;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> cf;

    wmma::load_matrix_sync(af, a, lda);
    wmma::load_matrix_sync(bf, b, ldb);
    wmma::load_matrix_sync(cf, d, ldd, wmma::mem_row_major);
    wmma::mma_sync(cf, af, bf, cf);
    wmma::store_matrix_sync(d, cf, ldd, wmma::mem_row_major);
}

extern "C" __global__ void wcr(const half *a, const half *b, float *d,
                               int lda, int ldb, int ldd)
{
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::col_major> af;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> bf;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> cf;

    wmma::load_matrix_sync(af, a, lda);
    wmma::load_matrix_sync(bf, b, ldb);
    wmma::load_matrix_sync(cf, d, ldd, wmma::mem_row_major);
    wmma::mma_sync(cf, af, bf, cf);
    wmma::store_matrix_sync(d, cf, ldd, wmma::mem_row_major);
}

extern "C" __global__ void wzero(const half *a, const half *b, float *d,
                                 int lda, int ldb, int ldd)
{
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> af;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> bf;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> cf;

    wmma::fill_fragment(cf, 0.0f);
    wmma::load_matrix_sync(af, a, lda);
    wmma::load_matrix_sync(bf, b, ldb);
    wmma::mma_sync(cf, af, bf, cf);
    wmma::store_matrix_sync(d, cf, ldd, wmma::mem_row_major);
}

extern "C" __global__ void wbf(const __nv_bfloat16 *a, const __nv_bfloat16 *b,
                               float *d, int lda, int ldb, int ldd)
{
    wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16,
                   wmma::row_major> af;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16,
                   wmma::col_major> bf;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> cf;

    wmma::load_matrix_sync(af, a, lda);
    wmma::load_matrix_sync(bf, b, ldb);
    wmma::load_matrix_sync(cf, d, ldd, wmma::mem_row_major);
    wmma::mma_sync(cf, af, bf, cf);
    wmma::store_matrix_sync(d, cf, ldd, wmma::mem_row_major);
}

extern "C" __global__ void w8n32(const half *a, const half *b, float *d,
                                 int lda, int ldb, int ldd)
{
    wmma::fragment<wmma::matrix_a, 8, 32, 16, half, wmma::row_major> af;
    wmma::fragment<wmma::matrix_b, 8, 32, 16, half, wmma::col_major> bf;
    wmma::fragment<wmma::accumulator, 8, 32, 16, float> cf;

    wmma::load_matrix_sync(af, a, lda);
    wmma::load_matrix_sync(bf, b, ldb);
    wmma::load_matrix_sync(cf, d, ldd, wmma::mem_row_major);
    wmma::mma_sync(cf, af, bf, cf);
    wmma::store_matrix_sync(d, cf, ldd, wmma::mem_row_major);
}

extern "C" __global__ void w32n8(const half *a, const half *b, float *d,
                                 int lda, int ldb, int ldd)
{
    wmma::fragment<wmma::matrix_a, 32, 8, 16, half, wmma::row_major> af;
    wmma::fragment<wmma::matrix_b, 32, 8, 16, half, wmma::col_major> bf;
    wmma::fragment<wmma::accumulator, 32, 8, 16, float> cf;

    wmma::load_matrix_sync(af, a, lda);
    wmma::load_matrix_sync(bf, b, ldb);
    wmma::load_matrix_sync(cf, d, ldd, wmma::mem_row_major);
    wmma::mma_sync(cf, af, bf, cf);
    wmma::store_matrix_sync(d, cf, ldd, wmma::mem_row_major);
}

extern "C" __global__ void ws8(const signed char *a, const signed char *b,
                               int *d, int lda, int ldb, int ldd)
{
    wmma::fragment<wmma::matrix_a, 16, 16, 16, signed char,
                   wmma::row_major> af;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, signed char,
                   wmma::col_major> bf;
    wmma::fragment<wmma::accumulator, 16, 16, 16, int> cf;

    wmma::load_matrix_sync(af, a, lda);
    wmma::load_matrix_sync(bf, b, ldb);
    wmma::load_matrix_sync(cf, d, ldd, wmma::mem_row_major);
    wmma::mma_sync(cf, af, bf, cf);
    wmma::store_matrix_sync(d, cf, ldd, wmma::mem_row_major);
}

extern "C" __global__ void wtf(const float *a, const float *b, float *d,
                               int lda, int ldb, int ldd)
{
    wmma::fragment<wmma::matrix_a, 16, 16, 8, wmma::precision::tf32,
                   wmma::row_major> af;
    wmma::fragment<wmma::matrix_b, 16, 16, 8, wmma::precision::tf32,
                   wmma::col_major> bf;
    wmma::fragment<wmma::accumulator, 16, 16, 8, float> cf;

    wmma::load_matrix_sync(af, a, lda);
    wmma::load_matrix_sync(bf, b, ldb);
    wmma::load_matrix_sync(cf, d, ldd, wmma::mem_row_major);
    wmma::mma_sync(cf, af, bf, cf);
    wmma::store_matrix_sync(d, cf, ldd, wmma::mem_row_major);
}

extern "C" __global__ void wf64(const double *a, const double *b, double *d,
                                int lda, int ldb, int ldd)
{
    wmma::fragment<wmma::matrix_a, 8, 8, 4, double, wmma::row_major> af;
    wmma::fragment<wmma::matrix_b, 8, 8, 4, double, wmma::col_major> bf;
    wmma::fragment<wmma::accumulator, 8, 8, 4, double> cf;

    wmma::load_matrix_sync(af, a, lda);
    wmma::load_matrix_sync(bf, b, ldb);
    wmma::load_matrix_sync(cf, d, ldd, wmma::mem_row_major);
    wmma::mma_sync(cf, af, bf, cf);
    wmma::store_matrix_sync(d, cf, ldd, wmma::mem_row_major);
}
