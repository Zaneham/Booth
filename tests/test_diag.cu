/* Fixture for the diagnostic renderer test (tests/terrs.c).
 * The missing ';' after '5' makes the parser choke on 'p' on the next
 * line, which exercises E020 ("expected ';', got 'p'"), the source-context
 * rendering, and the caret. */
__global__ void k(int *p) {
    int x = 5
    p[0] = x;
}
