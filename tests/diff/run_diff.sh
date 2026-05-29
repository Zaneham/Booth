#!/usr/bin/env bash
# run_diff.sh -- build and run the BarraCUDA differential tests.
#
# The idea: one BIR, many targets, so run the same kernel through two of
# them and diff. If they disagree, one backend's codegen is lying, and the
# CPU backend (simplest target) makes a good oracle. Runs on Linux or WSL,
# because the CPU/rv64 backends emit System V ELF objects that link and run
# with a native gcc, not under MinGW.
#
# Two flavours of test:
#   * cpu vs host    -- always runs. Catches frontend/lowering/cpu-codegen
#                       bugs against a plain host reference. Needs gcc only.
#   * cpu vs rv64    -- genuine cross-backend. Same source through --cpu and
#                       --rv64, diffed. Needs riscv64-unknown-elf-gcc and
#                       qemu-riscv64; skipped cleanly if they are missing.
#
# Every case runs twice, clean (must PASS) and --inject (must FAIL), so a
# green result actually means something. The compiler is found via
# $BARRACUDA (default ./barracuda); point it at a Linux-built one.
set -u

BARRACUDA="${BARRACUDA:-./barracuda}"
DIFFDIR="tests/diff"
CC="${CC:-gcc}"
RVCC="${RVCC:-riscv64-unknown-elf-gcc}"
QEMU="${QEMU:-qemu-riscv64}"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

fail=0

expect_pass() { if "$@" >/dev/null; then echo "  clean: PASS"; else echo "  clean: unexpected FAIL"; fail=1; fi; }
expect_fail() { if "$@" >/dev/null; then echo "  inject: no FAIL, harness has no teeth"; fail=1; else echo "  inject: FAIL as expected"; fi; }

# ---- cpu vs host: the portable smoke test ----
cpu_vs_host() {
    local src="$1" driver="$2" obj="$WORK/h.o" bin="$WORK/h"
    echo "== cpu vs host: $src =="
    "$BARRACUDA" --triton --cpu -o "$obj" "$src" 2>/dev/null \
        && "$CC" -no-pie -O2 -I"$DIFFDIR" "$DIFFDIR/$driver" "$obj" -o "$bin" -lm 2>/dev/null \
        || { echo "  build failed"; fail=1; return; }
    "$bin"; expect_pass "$bin"; expect_fail "$bin" --inject
}

# ---- cpu vs rv64: the real cross-backend test ----
cpu_vs_rv64() {
    local src="$1" rvrunner="$2" cmp="$3"
    echo "== cpu vs rv64: $src =="
    if ! command -v "$RVCC" >/dev/null || ! command -v "$QEMU" >/dev/null; then
        echo "  SKIP (need $RVCC and $QEMU)"; return
    fi
    local cpuobj="$WORK/x.o" rvobj="$WORK/r.o" runner="$WORK/rvrun" out="$WORK/rv.bin" bin="$WORK/xb"
    "$BARRACUDA" --triton --cpu  -o "$cpuobj" "$src" 2>/dev/null \
        && "$BARRACUDA" --triton --rv64 -o "$rvobj" "$src" 2>/dev/null \
        && "$RVCC" -nostdlib -static -march=rv64imfd -mabi=lp64d -I"$DIFFDIR" "$DIFFDIR/$rvrunner" "$rvobj" -o "$runner" 2>/dev/null \
        && "$QEMU" "$runner" > "$out" \
        && "$CC" -no-pie -O2 -I"$DIFFDIR" "$DIFFDIR/$cmp" "$cpuobj" -o "$bin" -lm 2>/dev/null \
        || { echo "  build/run failed"; fail=1; return; }
    "$bin" "$out"; expect_pass "$bin" "$out"; expect_fail "$bin" --inject "$out"
}

cpu_vs_host "tests/tri_vadd.py" "diff_vadd.c"
cpu_vs_rv64 "tests/tri_vadd.py" "diff_vadd_rv.c" "diff_xbackend.c"

if [ "$fail" = "0" ]; then echo "all differential tests OK"; else echo "differential tests had failures"; fi
exit "$fail"
