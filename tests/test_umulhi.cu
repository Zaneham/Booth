/* Differential test for __umul64hi -> BIR_UMULHI.
 *
 * High 64 bits of an unsigned 64x64 product. The keystone for 256-bit
 * field arithmetic (Montgomery multiply = mul-lo + mul-hi + carry), and
 * a ruthless correctness check: one wrong bit and a ZK proof fails.
 *
 * Host oracle:  (unsigned long long)(((__uint128_t)a * b) >> 64)
 *
 * Emit an object, then link a driver and diff against the oracle. The kernel
 * uses the SIMT thread-loop ABI: it compiles to k(out, a, b, n) where the
 * trailing n is the thread/iteration count, so launch with n=1:
 *     kath --cpu  -o k_cpu.o tests/test_umulhi.cu   # then gcc + run
 *     kath --rv64 -o k_rv.o  tests/test_umulhi.cu   # then link + qemu-riscv64
 *     driver: extern void k(u64*,u64,u64,long); k(&out, a, b, 1);
 * Verified ALL OK on --cpu (x86-64) and --rv64 (qemu) against the oracle.
 *
 * Suggested vectors (a, b -> expected high 64):
 *     0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF -> 0xFFFFFFFFFFFFFFFE
 *     0xFFFFFFFFFFFFFFFF, 0x0000000000000002 -> 0x0000000000000001
 *     0x0000000100000000, 0x0000000100000000 -> 0x0000000000000001
 *     0xDEADBEEFCAFEBABE, 0x1234567890ABCDEF -> 0x0FD5BDEEE268600E
 *     0x0000000000000000, 0xFFFFFFFFFFFFFFFF -> 0x0000000000000000
 */

__global__ void k(unsigned long long *out,
                  unsigned long long a, unsigned long long b)
{
    out[0] = __umul64hi(a, b);
}
