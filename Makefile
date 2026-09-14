CC      = gcc
CF_PROT := $(shell echo 'int main(void){return 0;}' | $(CC) -x c - -fcf-protection -c -o /dev/null >/dev/null 2>&1 && echo -fcf-protection)

# Apple's clang rejects GCC-only warning flags (-Wstack-usage, -Wredundant-decls)
# as a hard error under -Werror, where GCC accepts them. On clang we therefore
# drop those flags and -Werror itself until the tree is proven clang-clean. GCC
# builds (Linux, Windows/MinGW) keep the full strict set and are unchanged.
CLANG   := $(shell $(CC) --version 2>/dev/null | grep -i clang)
ifeq ($(CLANG),)
  GCC_ONLY = -Werror -Wstack-usage=4096 -Wno-error=stack-usage= -Wredundant-decls
else
  GCC_ONLY =
endif

# Empty for a normal build. `make coverage` re-invokes make with this set, and
# it lands at the end of CFLAGS/TCFLAGS so its -O0 beats the -O2 above.
COVFLAGS =

CFLAGS  = -std=c99 -MMD -MP -Wall -Wextra -pedantic -O2 \
          -Wshadow -Wstrict-prototypes -Wmissing-prototypes \
          -Wformat=2 -Wundef -Wcast-align -Wnull-dereference \
          -Wconversion -Wold-style-definition \
          -Wdouble-promotion -Wswitch-enum -Wwrite-strings \
          -D_FORTIFY_SOURCE=2 -fstack-protector-strong -fPIE $(CF_PROT) \
          $(GCC_ONLY) \
          -Isrc -Isrc/fe -Isrc/ir -Isrc/tdf -Isrc/backend -Isrc/amdgpu -Isrc/tensix -Isrc/nvidia -Isrc/nvidia/vendor -Isrc/metal -Isrc/intel -Isrc/triton -Isrc/cpu -Isrc/build -Isrc/exec -Iruntime/include \
          $(COVFLAGS)
LDFLAGS = -pie
LIBS    = -lm
# Linux/ELF only: -Wl,-z,relro,-z,now -Wl,-z,noexecstack

# Host-side launchers. They dlopen a vendor driver and link into trunner only,
# never into kath. nv_rt is portable (LoadLibraryA on Windows, dlopen elsewhere);
# bc_runtime is Linux, bare dlfcn.h and libhsa. -ldl names Linux rather than
# "not Windows" because macOS keeps dlopen in libSystem and ships no libdl.
# lf_gpu implements LFortran's GPU offload ABI on top of nv_rt, so an
# LFortran-compiled Fortran program can launch kernels kath produced. It
# links into the Fortran executable, not into kath; it sits in HOSTRT so
# the strict build keeps checking it.
UNAME_S := $(shell uname -s 2>/dev/null)

# Objects carry their host's format, so a Git Bash build and a WSL build in the
# same checkout used to overwrite each other and hand the linker a mix of COFF
# and ELF. That surfaces as undefined glibc symbols, which reads like a broken
# toolchain rather than a stale tree. One object dir per host, no collision.
OBJDIR  := build/$(UNAME_S)

HOSTRT   = $(OBJDIR)/runtime/host/cuda/nv_rt.o \
           $(OBJDIR)/runtime/host/cuda/lf_gpu.o
DL_LIB   =
ifeq ($(UNAME_S),Linux)
  HOSTRT += $(OBJDIR)/runtime/host/hsa/bc_runtime.o
  DL_LIB  = -ldl
endif

# lf_gpu_hsa is the AMD sibling of lf_gpu: the same ABI over bc_runtime. It
# defines the same symbols, so it can't sit in HOSTRT next to lf_gpu; we just
# compile it (Linux only) to keep it under the strict flags. The Fortran build
# links one runtime or the other depending on the target.
ALT_RT =
ifeq ($(UNAME_S),Linux)
  ALT_RT = $(OBJDIR)/runtime/host/hsa/lf_gpu_hsa.o
endif

SOURCES = src/main.c src/kauri_impl.c \
          src/fe/bc_err.c src/fe/bc_render.c src/fe/preproc.c src/fe/lexer.c src/fe/parser.c src/fe/sema.c \
          src/ir/bir.c src/ir/bir_print.c src/ir/bir_lower.c src/ir/bir_mem2reg.c src/ir/bir_cfold.c src/ir/bir_dce.c src/ir/bir_struct.c src/ir/bir_insert.c src/ir/bir_sroa.c src/ir/bir_inline.c \
          src/tdf/tdf.c src/tdf/tdf_lower.c src/tdf/tdf_fission.c src/tdf/tdf_place.c src/tdf/tdf_noc.c \
          src/backend/backends.c \
          src/build/booth_build.c src/build/bir_parse.c \
          src/amdgpu/amd_rplan.c src/amdgpu/isel.c src/amdgpu/emit.c src/amdgpu/ra_ssa.c src/amdgpu/encode.c src/amdgpu/enc_tab.c src/amdgpu/sched.c src/amdgpu/verify.c src/amdgpu/amd_be.c \
          src/tensix/isel.c src/tensix/emit.c src/tensix/coarsen.c src/tensix/datamov.c src/tensix/noc.c \
          src/tensix/rv_enc.c src/tensix/rv_buf.c src/tensix/rv_elf.c src/tensix/rv_isel.c src/tensix/tensix_be.c src/cpu/cpu_emit.c src/cpu/cpu_elf.c src/cpu/rv64_emit.c src/cpu/rv64_elf.c src/cpu/cpu_be.c \
          src/nvidia/isel.c src/nvidia/emit.c src/nvidia/nv_be.c src/nvidia/emit_sass.c src/nvidia/verify.c \
          src/metal/emit.c src/metal/metal_be.c \
          src/intel/emit.c src/intel/intel_be.c \
          src/triton/lex.c src/triton/parse.c src/triton/sema.c src/triton/lower.c \
          src/mlir/mlir_fe.c src/mlir/lower.c \
          src/exec/execs.c src/exec/cpu_exec.c src/exec/booth_run.c src/exec/verbs.c \
          runtime/host/cuda/nv_rt.c runtime/host/cuda/nv_exec.c

# Certik's pure-C MLIR reader, vendored under src/mlir/vendor. It carries his
# corec base library and a syscall shim per host, so only one of the three
# platform files is ever built.
#
# c2x rather than c99 because corec's format.h dispatches on _Generic and
# needs __VA_OPT__, and the ~100 format() call sites through it are not worth
# rewriting. -Wno-switch-enum because these switch over a 140-value op enum
# with a default label and upstream keeps adding ops. Nothing else in the
# warning set is relaxed. PLATFORM_SKIP_ENTRY leaves main to Booth,
# COREC_STDLIB_PROVIDES_MEM stops corec defining memcpy and memset when a real
# libc is already doing it.
VDIR = src/mlir/vendor
VPLAT = platform_windows.c
ifeq ($(UNAME_S),Linux)
  VPLAT = platform_linux.c
endif
ifeq ($(UNAME_S),Darwin)
  VPLAT = platform_macos.c
endif
VSOURCES = $(VDIR)/tokenizer.c $(VDIR)/mlir_parser.c $(VDIR)/op_parsers.c \
           $(VDIR)/mlir_api_impl.c $(VDIR)/mlir_op_names.c \
           $(VDIR)/mlir_classic_printer.c $(VDIR)/mlir_lift_cf_to_scf.c \
           $(VDIR)/base/io.c $(VDIR)/base/buddy.c $(VDIR)/base/arena.c \
           $(VDIR)/base/scratch.c $(VDIR)/base/format.c $(VDIR)/base/math.c \
           $(VDIR)/base/string.c $(VDIR)/base/strbuf.c $(VDIR)/base/mem.c \
           $(VDIR)/base/numconv.c $(VDIR)/base/assert.c $(VDIR)/base/exit.c \
           $(VDIR)/platform/$(VPLAT)
# Simply expanded, so the target-specific assignment below is a plain string
# rather than something that re-expands CFLAGS into itself.
VCFLAGS := $(subst -std=c99,-std=c2x,$(CFLAGS)) -Wno-switch-enum \
           -DPLATFORM_SKIP_ENTRY -DCOREC_STDLIB_PROVIDES_MEM -I$(VDIR)

# Zane's SASS back end, vendored across from Latimer under src/nvidia/vendor.
# The instruction encoder, the cubin writer and the note tables came over whole;
# lt_nvcf.c is the only file that changed, and only so the reconvergence pass
# takes a control-flow graph the caller has already built rather than deriving
# one from Latimer's Wave IR. It meets the strict flags unaltered, so unlike
# the MLIR reader below it needs no relaxed warning set of its own.
NVDIR = src/nvidia/vendor
NVSOURCES = $(NVDIR)/lt_nvi.c $(NVDIR)/lt_nv.c $(NVDIR)/lt_nvcf.c \
            $(NVDIR)/nv_note.c

OBJECTS = $(SOURCES:%.c=$(OBJDIR)/%.o) $(VSOURCES:%.c=$(OBJDIR)/%.o) \
          $(NVSOURCES:%.c=$(OBJDIR)/%.o)

# Everything under src/mlir compiles on VCFLAGS, vendored or not. mlir_fe.c and
# lower.c are ours but they speak corec types, so they want the same flags.
# A target-specific variable rather than a pattern rule, because two patterns
# match these objects and make 3.81, which is what macOS ships, does not
# resolve that the way make 4 does. It took the generic rule and the build lost
# its include path.
MLOBJECTS = $(VSOURCES:%.c=$(OBJDIR)/%.o) \
            $(OBJDIR)/src/mlir/mlir_fe.o $(OBJDIR)/src/mlir/lower.o
$(MLOBJECTS): CFLAGS := $(VCFLAGS)
TARGET  = kath

# Host programs that link a vendor driver at run time: the examples and the
# NVIDIA harness. Compiled but never linked here, so they keep meeting the
# strict flags without libhsa or libcuda having to be present. Neither is a
# trunner test; both carry their own main and want real hardware.
EXAMPLE_SRC = $(wildcard examples/*.c)
HOSTCHK     = $(OBJDIR)/tests/tnv_rt.o $(OBJDIR)/tests/tnv_sass.o $(OBJDIR)/tests/tnv_i1.o $(OBJDIR)/tests/tnv_bstr.o $(OBJDIR)/tests/tnv_sret.o $(OBJDIR)/tests/tnv_call.o $(OBJDIR)/tests/tnv_grid.o $(OBJDIR)/tests/gpu_tid.o $(OBJDIR)/tests/gpu_mma.o $(OBJDIR)/tests/gpu_wmma.o $(OBJDIR)/tests/gpu_hcvt.o $(patsubst examples/%.c,$(OBJDIR)/examples/%.o,$(EXAMPLE_SRC))
all: $(TARGET) $(ALT_RT) $(HOSTCHK)

$(TARGET): $(OBJECTS)
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ $^ $(LIBS) $(DL_LIB)

$(OBJDIR)/%.o: %.c
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -c $< -o $@

# ---- Test Suite ----
TCFLAGS = -std=c99 -MMD -MP -D_POSIX_C_SOURCE=200809L -Wall -Wextra -O0 -g \
          -Isrc -Isrc/fe -Isrc/ir -Isrc/tdf -Isrc/backend -Isrc/amdgpu -Isrc/tensix -Isrc/nvidia -Isrc/nvidia/vendor -Isrc/metal -Isrc/intel -Isrc/triton -Isrc/cpu -Isrc/build -Isrc/exec \
          -Isrc/mlir -Iruntime/include $(COVFLAGS)
TSRC    = tests/tmain.c tests/tsmoke.c tests/tcomp.c tests/tenc.c \
          tests/tasy.c \
          tests/ttabs.c tests/ttypes.c tests/terrs.c tests/tphase.c \
          tests/tdce.c \
          tests/tcfold.c \
          tests/tstruct.c \
          tests/tinsert.c \
          tests/tsroa.c \
          tests/tinline.c \
          tests/tsched.c \
          tests/twarpsize.c \
          tests/tabend.c \
          tests/tregalloc.c \
          tests/tra_ssa.c \
          tests/tguard.c \
          tests/ttriton.c \
          tests/tmma.c \
          tests/ttdf.c \
          tests/ttmc.c \
          tests/trv_enc.c tests/trv_buf.c tests/trv_elf.c tests/trv_isel.c \
          tests/tcbsync.c \
          tests/tsoft_fp.c \
          tests/tsysprint.c \
          tests/tbackend.c \
          tests/tordr.c \
          tests/trpi.c \
          tests/tmlir.c \
          tests/tbir.c \
          tests/tocm.c \
          tests/tpack.c \
          tests/tmtu.c \
          tests/tcxp.c \
          tests/ttpl.c \
          tests/tvec.c \
          tests/tref.c tests/ttyc.c tests/tcpp.c \
          tests/tmof.c tests/tasm.c

TOBJS   = $(TSRC:%.c=$(OBJDIR)/%.o)
COBJS   = $(OBJDIR)/src/kauri_impl.o $(OBJDIR)/src/ir/bir.o $(OBJDIR)/src/ir/bir_print.o $(OBJDIR)/src/ir/bir_lower.o $(OBJDIR)/src/ir/bir_mem2reg.o $(OBJDIR)/src/ir/bir_cfold.o $(OBJDIR)/src/ir/bir_dce.o $(OBJDIR)/src/ir/bir_struct.o $(OBJDIR)/src/ir/bir_insert.o $(OBJDIR)/src/ir/bir_sroa.o $(OBJDIR)/src/ir/bir_inline.o \
          $(OBJDIR)/src/tdf/tdf.o $(OBJDIR)/src/tdf/tdf_lower.o $(OBJDIR)/src/tdf/tdf_fission.o $(OBJDIR)/src/tdf/tdf_place.o $(OBJDIR)/src/tdf/tdf_noc.o \
          $(OBJDIR)/src/tensix/rv_enc.o $(OBJDIR)/src/tensix/rv_buf.o $(OBJDIR)/src/tensix/rv_elf.o $(OBJDIR)/src/tensix/rv_isel.o $(OBJDIR)/src/tensix/noc.o $(OBJDIR)/src/tensix/emit.o \
          $(OBJDIR)/runtime/device/soft_fp.o $(OBJDIR)/runtime/host/sysprint.o \
          $(OBJDIR)/src/amdgpu/amd_rplan.o $(OBJDIR)/src/amdgpu/encode.o $(OBJDIR)/src/amdgpu/enc_tab.o $(OBJDIR)/src/amdgpu/isel.o $(OBJDIR)/src/amdgpu/emit.o $(OBJDIR)/src/amdgpu/ra_ssa.o $(OBJDIR)/src/amdgpu/sched.o $(OBJDIR)/src/amdgpu/verify.o \
          $(OBJDIR)/src/fe/bc_err.o $(OBJDIR)/src/fe/lexer.o $(OBJDIR)/src/fe/parser.o $(OBJDIR)/src/fe/preproc.o $(OBJDIR)/src/fe/sema.o \
          $(OBJDIR)/runtime/host/bc_abend.o $(HOSTRT) \
          $(OBJDIR)/src/backend/backends.o \
          $(OBJDIR)/src/amdgpu/amd_be.o $(OBJDIR)/src/nvidia/nv_be.o \
          $(OBJDIR)/src/tensix/tensix_be.o $(OBJDIR)/src/cpu/cpu_be.o \
          $(OBJDIR)/src/metal/metal_be.o $(OBJDIR)/src/intel/intel_be.o \
          $(OBJDIR)/src/nvidia/isel.o $(OBJDIR)/src/nvidia/emit.o $(OBJDIR)/src/nvidia/emit_sass.o $(OBJDIR)/src/nvidia/verify.o \
          $(OBJDIR)/src/cpu/cpu_emit.o $(OBJDIR)/src/cpu/cpu_elf.o \
          $(OBJDIR)/src/cpu/rv64_emit.o $(OBJDIR)/src/cpu/rv64_elf.o \
          $(OBJDIR)/src/tensix/isel.o $(OBJDIR)/src/tensix/coarsen.o $(OBJDIR)/src/tensix/datamov.o \
          $(OBJDIR)/src/metal/emit.o $(OBJDIR)/src/intel/emit.o \
          $(OBJDIR)/src/mlir/mlir_fe.o $(OBJDIR)/src/mlir/lower.o $(VSOURCES:%.c=$(OBJDIR)/%.o) $(NVSOURCES:%.c=$(OBJDIR)/%.o)

rules:
	@sh tests/rules.sh

test: $(TARGET) trunner
	@sh tests/rules.sh
	./trunner --all

# bir.h claims a deterministic layout. This makes that a property rather
# than an intention, and it only stays cheap if it runs from now on.
# Bends one line at a time in a scratch copy and checks the suite notices.
# Never touches the working tree. See tests/mutants.tbl.
mutate: $(TARGET) trunner
	sh tests/mutate.sh

mutate-discover: $(TARGET) trunner
	sh tests/mutate.sh --discover

repro: $(TARGET)
	tests/reprocheck.sh

trunner: $(TOBJS) $(COBJS)
	$(CC) $(TCFLAGS) -o $@ $^ $(LIBS) $(DL_LIB)

$(OBJDIR)/tests/%.o: tests/%.c
	@mkdir -p $(dir $@)
	$(CC) $(TCFLAGS) -c $< -o $@

$(OBJDIR)/runtime/host/%.o: runtime/host/%.c
	@mkdir -p $(dir $@)
	$(CC) $(TCFLAGS) -c $< -o $@

hostchk: $(HOSTCHK)

$(OBJDIR)/examples/%.o: examples/%.c
	@mkdir -p $(dir $@)
	$(CC) $(TCFLAGS) -c $< -o $@

# Target-side runtime (soft-float, etc). Built with host gcc here
# so we can host-test the IEEE math; Booth will compile the
# same .c files separately when generating kernel ELFs.
$(OBJDIR)/runtime/device/%.o: runtime/device/%.c
	@mkdir -p $(dir $@)
	$(CC) $(TCFLAGS) -c $< -o $@

# ---- Install ----
# Booth is consumed as a compiler rather than a library, so an install is the
# binary, the message catalogues --lang reads, and a CMake package config so
# downstream CMake projects can find_package(Booth) and run kath as a build
# step. Version comes out of the header so there is one place to bump it.
PREFIX  ?= /usr/local
BINDIR   = $(DESTDIR)$(PREFIX)/bin
SHAREDIR = $(DESTDIR)$(PREFIX)/share/booth
CMAKEDIR = $(DESTDIR)$(PREFIX)/lib/cmake/Booth

# Matching on the macro name rather than the whole "#define" line: make 3.81,
# which is what macOS ships, takes a # inside $(shell) as the start of a
# comment and swallows the rest of the call. Quoting does not save it, and
# neither does awk over sed, so the # simply has to go.
VER_MAJOR := $(shell awk '$$2 == "BC_VERSION_MAJOR" {print $$3}' src/barracuda.h)
VER_MINOR := $(shell awk '$$2 == "BC_VERSION_MINOR" {print $$3}' src/barracuda.h)
VER_PATCH := $(shell awk '$$2 == "BC_VERSION_PATCH" {print $$3}' src/barracuda.h)
VERSION   := $(VER_MAJOR).$(VER_MINOR).$(VER_PATCH)

# MinGW gcc appends .exe when -o names no suffix, so the built file is not
# always $(TARGET); clean has always known this, install needs to as well.
EXE :=
ifneq (,$(findstring MINGW,$(UNAME_S)))
  EXE := .exe
endif

install: $(TARGET)
	install -d $(BINDIR) $(SHAREDIR)/lang $(CMAKEDIR)
	install -m 755 $(TARGET)$(EXE) $(BINDIR)/$(TARGET)$(EXE)
	install -m 644 lang/en.txt lang/mi.txt $(SHAREDIR)/lang/
	sed -e 's/@BOOTH_VERSION@/$(VERSION)/g' -e 's/@BOOTH_VERSION_MAJOR@/$(VER_MAJOR)/g' \
	    -e 's/@BOOTH_VERSION_MINOR@/$(VER_MINOR)/g' \
	    cmake/BoothConfig.cmake.in > $(CMAKEDIR)/BoothConfig.cmake
	sed -e 's/@BOOTH_VERSION@/$(VERSION)/g' -e 's/@BOOTH_VERSION_MAJOR@/$(VER_MAJOR)/g' \
	    -e 's/@BOOTH_VERSION_MINOR@/$(VER_MINOR)/g' \
	    cmake/BoothConfigVersion.cmake.in > $(CMAKEDIR)/BoothConfigVersion.cmake

uninstall:
	rm -f $(BINDIR)/$(TARGET)$(EXE)
	rm -rf $(SHAREDIR) $(CMAKEDIR)

# ---- Coverage ----
# Instrumented objects live in their own tree. Sharing one with the normal build
# means a later `make test` relinks against gcov objects and dies on missing
# __gcov symbols, which reads like a broken toolchain rather than a stale tree.
COVDIR  := build/cov-$(UNAME_S)

# kath is instrumented too, not just trunner: trv_elf, ttdf and ttriton shell out
# to ./kath, so a lot of the backend is only reached through the real binary.
# -U_FORTIFY_SOURCE because glibc #warnings at -O0 when it's set, and -Werror
# turns that into a build failure.
# -w because -O0 drops the value-range info that lets -Wformat-truncation prove
# its bounds at -O2, so the strict set fires on code that is fine. The normal
# build is the warnings gate; this one only counts lines.
COV_CF   = --coverage -O0 -U_FORTIFY_SOURCE -w

# Both binaries land in the repo root whichever tree built them, so clear them
# first to force an instrumented link, and again at the end so the next plain
# `make` doesn't quietly keep running the instrumented one.
coverage:
	rm -f $(TARGET) $(TARGET).exe trunner trunner.exe
	find $(COVDIR) -name '*.gcda' -delete 2>/dev/null || true
	$(MAKE) OBJDIR=$(COVDIR) COVFLAGS="$(COV_CF)" $(TARGET) trunner
	-./trunner --all
	@command -v gcovr >/dev/null 2>&1 || { echo "gcovr not found. pip install gcovr"; exit 1; }
	@mkdir -p coverage-html
	@# gcovr snuffles through the object dir like a skaven after warp tokens, so vendored data has to be gone rather than filtered.
	find $(COVDIR)/src/mlir \( -name '*.gcda' -o -name '*.gcno' \) -delete 2>/dev/null || true
	gcovr --root . --object-directory $(COVDIR) \
	      --filter 'src/' --filter 'runtime/' \
	      --exclude 'src/mlir/vendor/' \
	      --exclude-unreachable-branches \
	      --print-summary --txt coverage.txt --html-details coverage-html/index.html
	rm -f $(TARGET) $(TARGET).exe trunner trunner.exe
	@echo "report: coverage.txt and coverage-html/index.html"

clean:
	rm -rf $(OBJDIR) $(COVDIR)
	rm -f $(TARGET) $(TARGET).exe trunner trunner.exe tnv_rt tnv_rt.exe tnv_sass tnv_sass.exe
	rm -rf coverage.txt coverage-html
	rm -f *.hsaco *.ptx *.spv *.metal *.elf *.bin *.ttinsn *.o
	rm -f *_host.cpp *_reader.cpp *_writer.cpp *_compute.cpp
	rm -rf tests/ocean/out

# Header deps from -MMD. Without these a header edit leaves stale objects
# linked in and the build silently disagrees with the source.
-include $(OBJECTS:.o=.d) $(TOBJS:.o=.d) $(HOSTRT:.o=.d)

.PHONY: all clean test rules repro mutate mutate-discover install uninstall coverage hostchk
