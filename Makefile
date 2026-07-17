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

CFLAGS  = -std=c99 -Wall -Wextra -pedantic -O2 \
          -Wshadow -Wstrict-prototypes -Wmissing-prototypes \
          -Wformat=2 -Wundef -Wcast-align -Wnull-dereference \
          -Wconversion -Wold-style-definition \
          -Wdouble-promotion -Wswitch-enum -Wwrite-strings \
          -D_FORTIFY_SOURCE=2 -fstack-protector-strong -fPIE $(CF_PROT) \
          $(GCC_ONLY) \
          -Isrc -Isrc/fe -Isrc/ir -Isrc/tdf -Isrc/amdgpu -Isrc/tensix -Isrc/nvidia -Isrc/metal -Isrc/intel -Isrc/triton -Isrc/cpu -Isrc/runtime
LDFLAGS = -pie
LIBS    = -lm
# Linux/ELF only: -Wl,-z,relro,-z,now -Wl,-z,noexecstack

# Host-side launchers. They dlopen a vendor driver and link into trunner only,
# never into kath. nv_rt is portable (LoadLibraryA on Windows, dlopen elsewhere);
# bc_runtime is Linux, bare dlfcn.h and libhsa. -ldl names Linux rather than
# "not Windows" because macOS keeps dlopen in libSystem and ships no libdl.
UNAME_S := $(shell uname -s 2>/dev/null)
HOSTRT   = src/nvidia/nv_rt.o
DL_LIB   =
ifeq ($(UNAME_S),Linux)
  HOSTRT += src/runtime/bc_runtime.o
  DL_LIB  = -ldl
endif

SOURCES = src/main.c \
          src/fe/bc_err.c src/fe/bc_render.c src/fe/preproc.c src/fe/lexer.c src/fe/parser.c src/fe/sema.c \
          src/ir/bir.c src/ir/bir_print.c src/ir/bir_lower.c src/ir/bir_mem2reg.c src/ir/bir_cfold.c src/ir/bir_dce.c src/ir/bir_struct.c src/ir/bir_insert.c src/ir/bir_sroa.c src/ir/bir_inline.c \
          src/tdf/tdf.c src/tdf/tdf_lower.c src/tdf/tdf_fission.c src/tdf/tdf_place.c src/tdf/tdf_noc.c \
          src/amdgpu/amd_rplan.c src/amdgpu/isel.c src/amdgpu/emit.c src/amdgpu/ra_ssa.c src/amdgpu/encode.c src/amdgpu/enc_tab.c src/amdgpu/sched.c src/amdgpu/verify.c \
          src/tensix/isel.c src/tensix/emit.c src/tensix/coarsen.c src/tensix/datamov.c src/tensix/noc.c \
          src/tensix/rv_enc.c src/tensix/rv_buf.c src/tensix/rv_elf.c src/tensix/rv_isel.c src/cpu/cpu_emit.c src/cpu/cpu_elf.c src/cpu/rv64_emit.c src/cpu/rv64_elf.c \
          src/nvidia/isel.c src/nvidia/emit.c \
          src/metal/emit.c \
          src/intel/emit.c \
          src/triton/lex.c src/triton/parse.c src/triton/sema.c src/triton/lower.c
OBJECTS = $(SOURCES:.c=.o)
TARGET  = kath

all: $(TARGET)

$(TARGET): $(OBJECTS)
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ $^ $(LIBS)

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

# ---- Test Suite ----
TCFLAGS = -std=c99 -D_POSIX_C_SOURCE=200809L -Wall -Wextra -O0 -g \
          -Isrc -Isrc/fe -Isrc/ir -Isrc/tdf -Isrc/amdgpu -Isrc/tensix -Isrc/nvidia -Isrc/metal -Isrc/intel -Isrc/triton -Isrc/cpu -Isrc/runtime \
          -Iruntime
TSRC    = tests/tmain.c tests/tsmoke.c tests/tcomp.c tests/tenc.c \
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
          tests/ttriton.c \
          tests/ttdf.c \
          tests/ttmc.c \
          tests/trv_enc.c tests/trv_buf.c tests/trv_elf.c tests/trv_isel.c \
          tests/tcbsync.c \
          tests/tsoft_fp.c \
          tests/tsysprint.c

TOBJS   = $(TSRC:.c=.o)
COBJS   = src/ir/bir.o src/ir/bir_print.o src/ir/bir_lower.o src/ir/bir_mem2reg.o src/ir/bir_cfold.o src/ir/bir_dce.o src/ir/bir_struct.o src/ir/bir_insert.o src/ir/bir_sroa.o src/ir/bir_inline.o \
          src/tdf/tdf.o src/tdf/tdf_lower.o src/tdf/tdf_fission.o src/tdf/tdf_place.o src/tdf/tdf_noc.o \
          src/tensix/rv_enc.o src/tensix/rv_buf.o src/tensix/rv_elf.o src/tensix/rv_isel.o src/tensix/noc.o src/tensix/emit.o \
          runtime/soft_fp.o runtime/sysprint.o \
          src/amdgpu/amd_rplan.o src/amdgpu/encode.o src/amdgpu/enc_tab.o src/amdgpu/isel.o src/amdgpu/emit.o src/amdgpu/ra_ssa.o src/amdgpu/sched.o src/amdgpu/verify.o \
          src/fe/bc_err.o src/fe/lexer.o src/fe/parser.o src/fe/preproc.o src/fe/sema.o \
          src/runtime/bc_abend.o $(HOSTRT)

test: $(TARGET) trunner
	./trunner --all

trunner: $(TOBJS) $(COBJS)
	$(CC) $(TCFLAGS) -o $@ $^ $(LIBS) $(DL_LIB)

tests/%.o: tests/%.c
	$(CC) $(TCFLAGS) -c $< -o $@

src/runtime/%.o: src/runtime/%.c
	$(CC) $(TCFLAGS) -c $< -o $@

# Explicit, so it beats the generic %.o rule: nv_rt needs the POSIX visibility
# TCFLAGS carries, and its neighbours in src/nvidia are compiler files that don't.
src/nvidia/nv_rt.o: src/nvidia/nv_rt.c
	$(CC) $(TCFLAGS) -c $< -o $@

# Target-side runtime (soft-float, etc). Built with host gcc here
# so we can host-test the IEEE math; Booth will compile the
# same .c files separately when generating kernel ELFs.
runtime/%.o: runtime/%.c
	$(CC) $(TCFLAGS) -c $< -o $@

clean:
	rm -f $(OBJECTS) $(TARGET) $(TARGET).exe trunner trunner.exe $(TOBJS) $(HOSTRT) src/runtime/*.o runtime/*.o

.PHONY: all clean test
