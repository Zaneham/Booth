# Differential testing

One BIR, many targets. Run the same kernel through two backends and diff
the output buffers. If they disagree, one backend's codegen is wrong, and
since the CPU backend is the simplest target it makes a reliable oracle.
This is the fastest way to answer the question that keeps coming up: when a
kernel gives a wrong answer, is it the maths or the codegen?

## Run it

```bash
bash tests/diff/run_diff.sh
```

It runs on Linux or WSL, not MinGW (see the ABI note below). `$KATH`
points at the compiler (default `./kath`). Each case runs twice, clean
and `--inject`, because a harness that can't fail isn't testing anything.

Two kinds of test:

- **cpu vs host** always runs. It checks the `--cpu` backend against a plain
  host reference, which catches frontend, lowering and CPU codegen bugs.
  Needs only `gcc`.
- **cpu vs rv64** is the real cross-backend one: the same source through
  `--cpu` and `--rv64`, diffed against each other and the host. Needs
  `riscv64-unknown-elf-gcc` and `qemu-riscv64`; it skips cleanly without
  them.

## How it fits together

`bc_diff.h` is the compare core: tolerance-based float diff with a report.
It knows nothing about who produced the bytes, on purpose. Float is not
exact across backends, so it is a tolerance check, not bit-for-bit, and you
pick the tolerance to suit the kernel.

Each backend has one job, produce its answer. `diff_vadd.c` runs the `--cpu`
kernel in-process and `diff_vadd_rv.c` runs the `--rv64` kernel under qemu
and writes the raw output bytes to stdout. `diff_xbackend.c` reads both and
diffs everything against the host oracle. Adding a GPU runner later is the
same shape: produce a buffer, hand it to `bc_diff_f32`. The compare logic
never changes.

## RISC-V gotchas (the ones that cost time)

- **Run on Linux / WSL, not MinGW.** The CPU and rv64 backends emit System V
  ELF objects with the SysV calling convention. MinGW gcc is Windows ABI, so
  it will happily link the object and then segfault on the first call. Build
  the host and run under a native Linux gcc.
- **The rv64 host is freestanding.** `qemu-riscv64` user-mode does not give
  newlib its syscalls and does not propagate the guest exit code, so the
  runner skips libc: its own `_start`, raw `write`/`exit` ecalls, and the
  verdict comes from the bytes it writes, never from `$?`.
- **Set the global pointer.** A freestanding `_start` must set `gp` to
  `__global_pointer$` first thing (`.option norelax` around the `la`). The
  linker relaxes data loads to gp-relative, and without a valid `gp` they
  land in the weeds. The symptom is a segfault that looks like nothing.
- **Match the float ABI.** The rv64 objects declare hard float (lp64d), so
  build the runner with `-march=rv64imfd -mabi=lp64d` or the link is a
  mismatch.
