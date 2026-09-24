#ifndef COOPERATIVE_GROUPS_REDUCE_H
#define COOPERATIVE_GROUPS_REDUCE_H

#include <cooperative_groups.h>

#if defined(__cplusplus)

namespace cooperative_groups {

template <class T> struct plus    { T operator()(T a, T b) const; };
template <class T> struct less    { T operator()(T a, T b) const; };
template <class T> struct greater { T operator()(T a, T b) const; };
template <class T> struct bit_and { T operator()(T a, T b) const; };
template <class T> struct bit_or  { T operator()(T a, T b) const; };
template <class T> struct bit_xor { T operator()(T a, T b) const; };

template <class G, class T, class Op>
T reduce(const G &g, T val, Op op);

}

#endif

#endif
