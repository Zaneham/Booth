# A project consuming an installed Booth. Build it against a prefix you
# installed to:
#
#     make install PREFIX=/tmp/booth
#     cmake -S examples/cmake -B build -DCMAKE_PREFIX_PATH=/tmp/booth
#     cmake --build build
#
# LANGUAGES NONE because nothing here needs a host compiler; every target is
# a kernel Booth produces. See docs/cmake.md for linking the cpu and rv64
# objects into a real executable.

cmake_minimum_required(VERSION 3.10)
project(booth_cmake_example LANGUAGES NONE)

find_package(Booth 0.6 REQUIRED)
message(STATUS "Booth ${Booth_VERSION} at ${Booth_EXECUTABLE}")

booth_add_kernel(vadd_ptx    SOURCE vadd.cu BACKEND nvidia-ptx)
booth_add_kernel(vadd_gfx942 SOURCE vadd.cu BACKEND amdgpu-bin ARCH gfx942)
booth_add_kernel(vadd_asm    SOURCE vadd.cu BACKEND amdgpu     ARCH gfx1100)
booth_add_kernel(vadd_tt     SOURCE vadd.cu BACKEND tensix     TT_CHIP blackhole)
booth_add_kernel(vadd_host   SOURCE vadd.cu BACKEND cpu)

# Every kernel reports where it landed, as a target property and as a
# variable in this scope.
get_target_property(_ptx vadd_ptx BOOTH_KERNEL_OUTPUT)
message(STATUS "PTX at ${_ptx}")
message(STATUS "host object at ${vadd_host_OUTPUT}")
