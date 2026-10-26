#ifndef COOPERATIVE_GROUPS_H
#define COOPERATIVE_GROUPS_H

#include <cuda_runtime.h>

#if defined(__cplusplus)

namespace cooperative_groups {

class thread_group {
public:
    void               sync() const;
    unsigned long long size() const;
    unsigned long long num_threads() const;
    unsigned long long thread_rank() const;
    bool               is_valid() const;
};

class grid_group : public thread_group {
public:
    dim3 group_dim() const;
    dim3 group_index() const;
    dim3 thread_index() const;
    unsigned long long num_blocks() const;
    unsigned long long block_rank() const;
};

class thread_block : public thread_group {
public:
    dim3 group_index() const;
    dim3 thread_index() const;
    dim3 dim_threads() const;
};

class coalesced_group : public thread_group {
public:
    unsigned int ballot(int pred) const;
    int          any(int pred) const;
    int          all(int pred) const;
    unsigned int match_any(int val) const;
    unsigned int match_all(int val, int &pred) const;
};

grid_group      this_grid();
thread_block    this_thread_block();
coalesced_group coalesced_threads();

void sync(const thread_group &g);

}

#endif

#endif
