CC      = gcc
CFLAGS  = -std=c99 -Wall -Wextra -pedantic -Werror -O2 \
          -Wshadow -Wstrict-prototypes -Wmissing-prototypes \
          -Wformat=2 -Wundef -Wcast-align -Wnull-dereference \
          -Wstack-usage=4096 -Wno-error=stack-usage= \
          -Wconversion -Wold-style-definition \
          -Wdouble-promotion -Wswitch-enum -Wredundant-decls -Wwrite-strings \
          -D_FORTIFY_SOURCE=2 -fstack-protector-strong -fPIE -fcf-protection \
          -Isrc -Isrc/fe -Isrc/ir -Isrc/amdgpu -Isrc/tensix -Isrc/nvidia -Isrc/metal -Isrc/intel -Isrc/triton -Isrc/runtime
LDFLAGS = -pie
LIBS    = -lm
# Linux/ELF only: -Wl,-z,relro,-z,now -Wl,-z,noexecstack

SOURCES = src/main.c \
          src/fe/bc_err.c src/fe/preproc.c src/fe/lexer.c src/fe/parser.c src/fe/sema.c \
          src/ir/bir.c src/ir/bir_print.c src/ir/bir_lower.c src/ir/bir_mem2reg.c src/ir/bir_cfold.c src/ir/bir_dce.c \
          src/amdgpu/amd_rplan.c src/amdgpu/isel.c src/amdgpu/emit.c src/amdgpu/ra_ssa.c src/amdgpu/encode.c src/amdgpu/enc_tab.c src/amdgpu/sched.c src/amdgpu/verify.c \
          src/tensix/isel.c src/tensix/emit.c src/tensix/coarsen.c src/tensix/datamov.c \
          src/nvidia/isel.c src/nvidia/emit.c \
          src/metal/emit.c \
          src/intel/emit.c \
          src/triton/lex.c src/triton/parse.c src/triton/sema.c src/triton/lower.c
OBJECTS = $(SOURCES:.c=.o)
TARGET  = barracuda

all: $(TARGET)

$(TARGET): $(OBJECTS)
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ $^ $(LIBS)

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

# ---- Test Suite ----
TCFLAGS = -std=c99 -D_POSIX_C_SOURCE=200809L -Wall -Wextra -O0 -g \
          -Isrc -Isrc/fe -Isrc/ir -Isrc/amdgpu -Isrc/tensix -Isrc/nvidia -Isrc/metal -Isrc/intel -Isrc/triton -Isrc/runtime
TSRC    = tests/tmain.c tests/tsmoke.c tests/tcomp.c tests/tenc.c \
          tests/ttabs.c tests/ttypes.c tests/terrs.c tests/tphase.c \
          tests/tdce.c \
          tests/tcfold.c \
          tests/tsched.c \
          tests/tabend.c \
          tests/tregalloc.c
TOBJS   = $(TSRC:.c=.o)
COBJS   = src/ir/bir.o src/ir/bir_print.o src/ir/bir_lower.o src/ir/bir_mem2reg.o src/ir/bir_cfold.o src/ir/bir_dce.o \
          src/amdgpu/amd_rplan.o src/amdgpu/encode.o src/amdgpu/enc_tab.o src/amdgpu/isel.o src/amdgpu/emit.o src/amdgpu/ra_ssa.o src/amdgpu/sched.o src/amdgpu/verify.o \
          src/fe/bc_err.o src/fe/lexer.o src/fe/parser.o src/fe/preproc.o src/fe/sema.o \
          src/runtime/bc_abend.o

test: $(TARGET) trunner
	./trunner --all

trunner: $(TOBJS) $(COBJS)
	$(CC) $(TCFLAGS) -o $@ $^ $(LIBS)

tests/%.o: tests/%.c
	$(CC) $(TCFLAGS) -c $< -o $@

src/runtime/%.o: src/runtime/%.c
	$(CC) $(TCFLAGS) -c $< -o $@

clean:
	rm -f $(OBJECTS) $(TARGET) $(TARGET).exe trunner trunner.exe $(TOBJS) src/runtime/*.o

.PHONY: all clean test
