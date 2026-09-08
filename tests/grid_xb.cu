namespace cooperative_groups { }
namespace cg = cooperative_groups;

__global__ void gridxb(int *d, int *s, int n, int rnd)
{
    cg::grid_group g = cg::this_grid();
    int r = g.thread_rank();
    int p = (r + n / 2) % n;
    int v = r + 1;

    for (int i = 0; i < rnd; i++) {
        d[r] = v;
        g.sync();
        v = d[p] + 1;
        g.sync();
    }
    s[r] = v;
}
