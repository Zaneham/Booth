# Using Booth from CMake

Booth builds with its own Makefile; there is no CMake build of the compiler
itself. What this gives you is the other direction, a CMake project consuming
an installed Booth to compile kernels as part of its build.

## Installing

```sh
make
sudo make install                 # /usr/local by default
make install PREFIX=$HOME/.local  # or wherever
```

That puts `kath` in `<prefix>/bin`, the `--lang` catalogues in
`<prefix>/share/booth/lang`, and the package config in
`<prefix>/lib/cmake/Booth`. `DESTDIR` is honoured for staged installs.

## Finding it

```cmake
find_package(Booth 0.6 REQUIRED)
```

If the prefix is not one CMake already searches, point it there with
`-DCMAKE_PREFIX_PATH=<prefix>`.

While Booth is on 0.x the minor version has to match, so asking for 0.5 is
satisfied by 0.5.2 but not by 0.6. That is deliberate: 0.x is where things
are still allowed to move, and a package config that quietly accepted 0.6
would be promising something Booth is not ready to promise. From 1.0 the
usual same-major rule takes over.

You get `Booth_VERSION`, `Booth_EXECUTABLE`, `Booth_LANG_DIR`, and a
`Booth::kath` imported target.

## Compiling a kernel

```cmake
booth_add_kernel(vadd_amd
    SOURCE  vadd.cu
    BACKEND amdgpu-bin
    ARCH    gfx942)
```

That adds a target `vadd_amd` that runs `kath` at build time. The output path
comes back two ways, as a `BOOTH_KERNEL_OUTPUT` target property and as a
`vadd_amd_OUTPUT` variable in the calling scope:

```cmake
add_custom_command(TARGET app POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy ${vadd_amd_OUTPUT} $<TARGET_FILE_DIR:app>)
```

The `cpu` and `rv64` backends emit ordinary relocatable objects, so those can
go straight into a target:

```cmake
booth_add_kernel(vadd_host SOURCE vadd.cu BACKEND cpu)
add_executable(app main.c)
add_dependencies(app vadd_host)
target_link_libraries(app PRIVATE ${vadd_host_OUTPUT})
```

### Arguments

| | |
|---|---|
| `SOURCE` | The one input file. `kath` compiles a single file per run, so a kernel is one source. |
| `BACKEND` | `amdgpu`, `amdgpu-bin`, `nvidia-ptx`, `tensix`, `rv-elf`, `cpu`, `rv64`, `metal`, `intel-spirv`. |
| `LANGUAGE` | `CUDA` (default), `HIP`, or `TRITON`. `.hip` files pick HIP up on their own. |
| `ARCH` | `gfx90a`, `gfx942`, `gfx1030`…`gfx1201`, or `xe-lpg`/`xe-hpg`/`xe-hpc`/`xe2`. |
| `TT_CHIP` | `blackhole` or `wormhole`. |
| `OUTPUT` | Output path. Defaults to `<name>` in the build dir with a suffix picked from the backend. |
| `INCLUDE_DIRECTORIES` | Passed through as `-I`. |
| `COMPILE_DEFINITIONS` | Passed through as `-D`. |
| `OPTIONS` | Any other flags, handed to `kath` untouched. |
| `DEPENDS` | Extra files to rebuild on. |
| `EXCLUDE_FROM_ALL` | Build only when something asks for it. |

### Two things to know

`kath` writes no depfile, so editing a header the kernel includes will not
retrigger a build on its own. List those headers in `DEPENDS` if you need it.

The `tensix` backend emits a whole Metalium program rather than one file: the
compute kernel at `OUTPUT`, plus reader, writer and host sources and their
ELFs alongside it. Those are declared as byproducts, so `clean` reaches them,
but `BOOTH_KERNEL_OUTPUT` names only the compute kernel.
