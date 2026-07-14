# BIR: Booth Intermediate Representation

BIR is the target-independent SSA intermediate representation at the core of the Booth compiler. It sits between semantic analysis and backend code generation, providing a typed, hardware-agnostic instruction set that captures GPU compute semantics without committing to any particular ISA.

```
Source (.cu) -> Lexer -> Parser -> AST -> Semantic Analysis
    -> BIR (SSA, typed) -> mem2reg -> Instruction Selection -> Machine Code
```

## Design Principles

- **No malloc.** The entire module is a single statically-sized struct. All arrays have compile-time upper bounds. No unbounded allocation means no memory leaks.
- **No recursion.** All tree walks use explicit stacks with bounded depth.
- **All loops bounded.** Guard counters prevent infinite iteration.
- **Index-based references.** No pointers into the module; everything is referenced by 32-bit indices.
- **SSA form.** After `mem2reg`, the IR is in SSA form with phi nodes at dominance frontiers.
- **Target-independent.** The BIR makes no hardware assumptions. Adding a new backend means writing a new `isel` + `emit` pair.
- **Cache-friendly layout.** Instructions are 32 bytes (two per cache line), types and constants are 16 bytes each.

---

## Table of Contents

1. [Module Structure](#module-structure)
2. [Type System](#type-system)
3. [Constants](#constants)
4. [Value References](#value-references)
5. [Instruction Format](#instruction-format)
6. [Address Spaces](#address-spaces)
7. [Instruction Reference](#instruction-reference)
   - [Arithmetic](#arithmetic)
   - [Bitwise](#bitwise)
   - [Comparison](#comparison)
   - [Conversion](#conversion)
   - [Memory](#memory)
   - [Control Flow](#control-flow)
   - [SSA](#ssa)
   - [Thread Model](#thread-model)
   - [Barriers](#barriers)
   - [Atomics](#atomics)
   - [Warp-Level Primitives](#warp-level-primitives)
   - [Miscellaneous](#miscellaneous)
8. [Comparison Predicates](#comparison-predicates)
9. [Memory Ordering](#memory-ordering)
10. [Operand Overflow](#operand-overflow)
11. [IR Construction](#ir-construction)
12. [Optimization Passes](#optimization-passes)
13. [Textual IR Format](#textual-ir-format)
14. [Limits](#limits)

---

## Module Structure

A `bir_module_t` contains the entire program in a single flat struct:

| Field | Type | Capacity | Description |
|-------|------|----------|-------------|
| `types[]` | `bir_type_t` | 16,384 | Interned type descriptors |
| `type_fields[]` | `uint32_t` | 65,536 | Field/parameter type indices for struct and function types |
| `consts[]` | `bir_const_t` | 65,536 | Compile-time constant values |
| `insts[]` | `bir_inst_t` | 262,144 | All instructions across all functions |
| `inst_lines[]` | `uint32_t` | 262,144 | Source line number per instruction |
| `extra_operands[]` | `uint32_t` | 65,536 | Overflow operand storage for phi nodes and large calls |
| `blocks[]` | `bir_block_t` | 65,536 | Basic block descriptors |
| `funcs[]` | `bir_func_t` | 4,096 | Function descriptors |
| `globals[]` | `bir_global_t` | 4,096 | Global variable descriptors |
| `strings[]` | `char` | 1,048,576 | Null-terminated string table (1 MB) |

Functions own a contiguous range of blocks. Blocks own a contiguous range of instructions. This flat layout means the entire IR is serializable and walkable with simple index arithmetic.

### Functions (`bir_func_t`)

```
| name | type | first_block | total_insts | num_blocks | num_params | cuda_flags | launch_bounds |
```

- `name` -- Offset into the string table.
- `type` -- Index of a `BIR_TYPE_FUNC` type describing the signature.
- `first_block` -- Index of the entry block in `blocks[]`.
- `total_insts` -- Total instruction count across all blocks.
- `num_params` -- Number of parameters (corresponding `BIR_PARAM` instructions at the start of the entry block).
- `cuda_flags` -- Bitmask: `CUDA_GLOBAL`, `CUDA_DEVICE`, `CUDA_HOST`.
- `launch_bounds_max` / `launch_bounds_min` -- From `__launch_bounds__` annotations.

### Basic Blocks (`bir_block_t`)

```
| name | first_inst | num_insts |
```

- `name` -- String table offset for the block label.
- `first_inst` -- Index of the first instruction in `insts[]`.
- `num_insts` -- Number of instructions in this block.

Every block must end with a terminator instruction (`br`, `br_cond`, `switch`, `ret`, or `unreachable`).

### Global Variables (`bir_global_t`)

```
| name | type | initializer | cuda_flags | addrspace | is_const |
```

- `initializer` -- A constant reference (`BIR_MAKE_CONST(idx)`), or `BIR_VAL_NONE` if uninitialized.
- `addrspace` -- The address space the global resides in (`global`, `constant`, `shared`).
- `is_const` -- Whether the global is read-only (`__constant__`).

---

## Type System

All types are interned: creating the same type twice returns the same index. Types are described by `bir_type_t` (16 bytes):

```c
typedef struct {
    uint8_t     kind;       /* bir_type_kind_t */
    uint8_t     addrspace;  /* PTR only: address space */
    uint16_t    width;      /* INT/FLOAT: bit width. VECTOR: lane count */
    uint32_t    inner;      /* PTR: pointee. ARRAY/VECTOR: element. FUNC: return type */
    uint32_t    count;      /* ARRAY: element count. STRUCT/FUNC: fields_start index */
    uint16_t    num_fields; /* STRUCT/FUNC: field/param count */
    uint16_t    pad;
} bir_type_t;
```

### Type Kinds

| Kind | Description | Key Fields |
|------|-------------|------------|
| `BIR_TYPE_VOID` | Void type | None |
| `BIR_TYPE_INT` | Integer | `width`: 1, 8, 16, 32, or 64 bits |
| `BIR_TYPE_FLOAT` | Floating-point | `width`: 16 (half), 32 (float), or 64 (double) |
| `BIR_TYPE_PTR` | Pointer | `addrspace`: address space, `inner`: pointee type index |
| `BIR_TYPE_VECTOR` | SIMD vector | `width`: lane count, `inner`: element type index |
| `BIR_TYPE_STRUCT` | Aggregate | `count`: start index in `type_fields[]`, `num_fields`: field count |
| `BIR_TYPE_ARRAY` | Fixed-size array | `inner`: element type index, `count`: element count |
| `BIR_TYPE_FUNC` | Function signature | `inner`: return type, `count`: start in `type_fields[]`, `num_fields`: param count |

### Textual Type Notation

| Type | Notation | Examples |
|------|----------|---------|
| Void | `void` | `void` |
| Integer | `iN` | `i1`, `i8`, `i16`, `i32`, `i64` |
| Float | `fN` | `f16`, `f32`, `f64` |
| Pointer | `ptr<addrspace, pointee>` | `ptr<global, f32>`, `ptr<shared, i32>` |
| Array | `[N x elem]` | `[256 x f32]` |
| Vector | `<N x elem>` | `<4 x f32>`, `<2 x i32>` |
| Struct | `{field, field, ...}` | `{f32, f32, f32}` |
| Function | `(params) -> ret` | `(ptr<global, f32>, i32) -> void` |

### Type Construction API

```c
uint32_t bir_type_void(bir_module_t *M);
uint32_t bir_type_int(bir_module_t *M, int width_bits);
uint32_t bir_type_float(bir_module_t *M, int width_bits);
uint32_t bir_type_ptr(bir_module_t *M, uint32_t pointee, int addrspace);
uint32_t bir_type_array(bir_module_t *M, uint32_t elem, uint32_t count);
uint32_t bir_type_vector(bir_module_t *M, uint32_t elem, uint32_t count);
uint32_t bir_type_struct(bir_module_t *M, const uint32_t *fields, int nfields);
uint32_t bir_type_func(bir_module_t *M, uint32_t ret, const uint32_t *params, int nparams);
```

All functions return the index of an existing or newly created type. Type index 0 is always `void` (reserved during `bir_module_init`).

---

## Constants

Constants are compile-time values stored in `bir_const_t` (16 bytes):

```c
typedef struct {
    uint8_t     kind;       /* bir_const_kind_t */
    uint8_t     pad[3];
    uint32_t    type;       /* type index */
    union {
        int64_t     ival;
        double      fval;
    } d;
} bir_const_t;
```

### Constant Kinds

| Kind | Description | Data Field |
|------|-------------|------------|
| `BIR_CONST_INT` | Integer literal | `d.ival` |
| `BIR_CONST_FLOAT` | Floating-point literal | `d.fval` |
| `BIR_CONST_NULL` | Null pointer | None (value is zero) |
| `BIR_CONST_UNDEF` | Undefined value | None |
| `BIR_CONST_ZERO` | Zero-initialized aggregate | None |

Constants are interned: creating the same constant twice returns the same index.

### Constant Construction API

```c
uint32_t bir_const_int(bir_module_t *M, uint32_t type, int64_t val);
uint32_t bir_const_float(bir_module_t *M, uint32_t type, double val);
uint32_t bir_const_null(bir_module_t *M, uint32_t type);
```

Each returns an index into `consts[]`. To use a constant as an instruction operand, wrap it with `BIR_MAKE_CONST(index)`.

---

## Value References

Operand slots in instructions are 32-bit tagged references. Bit 31 distinguishes constants from instruction-produced values:

| Macro | Description |
|-------|-------------|
| `BIR_MAKE_VAL(i)` | Reference to the value produced by instruction at global index `i` |
| `BIR_MAKE_CONST(i)` | Reference to the constant at index `i` in `consts[]` |
| `BIR_VAL_IS_CONST(v)` | True if `v` references a constant (bit 31 set) |
| `BIR_VAL_INDEX(v)` | Extract the index (strip tag bit) |
| `BIR_VAL_NONE` | Sentinel value `0xFFFFFFFF`, meaning "no value" |
| `BIR_VAL_CONST_BIT` | The tag bit: `0x80000000` |

Block references in branch targets are plain block indices, identified by operand position (the opcode defines which slots are block references).

In the textual IR, instruction values are printed as `%N` (local index within the function), constants are printed as their literal value (`42`, `3.14`, `null`, `undef`, `zeroinit`), and block references are printed as `bb0`, `bb1`, or by name if one was assigned.

---

## Instruction Format

Every instruction is a `bir_inst_t` (32 bytes, two per cache line):

```c
typedef struct {
    uint16_t    op;             /* bir_op_t */
    uint8_t     num_operands;   /* 0-6 inline, or 0xFF for overflow */
    uint8_t     subop;          /* predicate, dimension, ordering, volatile flag, alignment */
    uint32_t    type;           /* result type index */
    uint32_t    operands[6];    /* inline operand slots */
} bir_inst_t;
```

- `op` -- The opcode (see [Instruction Reference](#instruction-reference)).
- `num_operands` -- Number of inline operands (0-6). If `0xFF` (`BIR_OPERANDS_OVERFLOW`), the operands are stored in `extra_operands[]` (see [Operand Overflow](#operand-overflow)).
- `subop` -- Overloaded field whose meaning depends on the opcode:
  - Comparison instructions: comparison predicate (`bir_cmp_pred_t`)
  - Thread model instructions: dimension (0=x, 1=y, 2=z)
  - Atomic instructions: memory ordering (`bir_mem_order_t`)
  - `alloca`: log2 of alignment
  - `load`/`store`: 0 = normal, 1 = volatile
  - `global_ref`: global variable index
  - `param`: parameter index
- `type` -- Index of the result type. For void instructions (`store`, `br`, `ret void`, `barrier`), this is the void type index.
- `operands[]` -- Six inline operand slots. Most instructions use 0-3; the remaining slots are unused.

---

## Address Spaces

BIR distinguishes five address spaces for pointer types:

| Address Space | Enum | Description |
|---------------|------|-------------|
| Private | `BIR_AS_PRIVATE` | Per-thread stack memory. Used by `alloca`. |
| Shared | `BIR_AS_SHARED` | Per-block scratchpad (LDS on AMD). Used by `__shared__` variables. |
| Global | `BIR_AS_GLOBAL` | Device DRAM. Kernel pointer arguments and `__device__` globals. |
| Constant | `BIR_AS_CONSTANT` | Read-only, hardware-cached memory. `__constant__` globals. |
| Generic | `BIR_AS_GENERIC` | Unresolved address space. Backend decides at instruction selection. |

The address space is encoded in the pointer type's `addrspace` field. The backend uses this to select the appropriate load/store instruction family (e.g., SMEM for constant, DS for shared, GLOBAL for global, SCRATCH for private on AMDGPU).

---

## Instruction Reference

BIR defines 64 implemented opcodes plus 16 reserved texture/surface slots (total enum range: 144).

### Arithmetic

Integer and floating-point arithmetic. All binary operations take two operands of the same type and produce a result of the same type.

| Opcode | Mnemonic | Semantics | Operands |
|--------|----------|-----------|----------|
| `BIR_ADD` | `add` | Integer addition | `ops[0] + ops[1]` |
| `BIR_SUB` | `sub` | Integer subtraction | `ops[0] - ops[1]` |
| `BIR_MUL` | `mul` | Integer multiplication | `ops[0] * ops[1]` |
| `BIR_SDIV` | `sdiv` | Signed integer division | `ops[0] / ops[1]` (signed) |
| `BIR_UDIV` | `udiv` | Unsigned integer division | `ops[0] / ops[1]` (unsigned) |
| `BIR_SREM` | `srem` | Signed integer remainder | `ops[0] % ops[1]` (signed) |
| `BIR_UREM` | `urem` | Unsigned integer remainder | `ops[0] % ops[1]` (unsigned) |
| `BIR_FADD` | `fadd` | Floating-point addition | `ops[0] + ops[1]` |
| `BIR_FSUB` | `fsub` | Floating-point subtraction | `ops[0] - ops[1]` |
| `BIR_FMUL` | `fmul` | Floating-point multiplication | `ops[0] * ops[1]` |
| `BIR_FDIV` | `fdiv` | Floating-point division | `ops[0] / ops[1]` |
| `BIR_FREM` | `frem` | Floating-point remainder | `ops[0] % ops[1]` |

**Textual format:**
```
%2 = add i32 %0, %1
%5 = fmul f32 %3, %4
```

### Bitwise

Bitwise logical and shift operations on integers.

| Opcode | Mnemonic | Semantics | Operands |
|--------|----------|-----------|----------|
| `BIR_AND` | `and` | Bitwise AND | `ops[0] & ops[1]` |
| `BIR_OR` | `or` | Bitwise OR | `ops[0] \| ops[1]` |
| `BIR_XOR` | `xor` | Bitwise XOR | `ops[0] ^ ops[1]` |
| `BIR_SHL` | `shl` | Shift left | `ops[0] << ops[1]` |
| `BIR_LSHR` | `lshr` | Logical shift right | `ops[0] >> ops[1]` (zero-fill) |
| `BIR_ASHR` | `ashr` | Arithmetic shift right | `ops[0] >> ops[1]` (sign-extend) |

**Textual format:**
```
%3 = and i32 %1, %2
%5 = shl i32 %0, 2
```

### Comparison

Comparison instructions produce an `i1` (1-bit integer) result. The comparison predicate is encoded in the `subop` field. See [Comparison Predicates](#comparison-predicates) for the full list.

| Opcode | Mnemonic | Semantics |
|--------|----------|-----------|
| `BIR_ICMP` | `icmp` | Integer comparison. `subop` = integer predicate. |
| `BIR_FCMP` | `fcmp` | Floating-point comparison. `subop` = float predicate. |

**Textual format** (prints predicate name and operand type, not result type):
```
%3 = icmp slt i32 %1, %2
%6 = fcmp oge f32 %4, %5
```

### Conversion

Type conversion instructions. Each takes a single operand and converts it to the instruction's result type.

| Opcode | Mnemonic | Semantics |
|--------|----------|-----------|
| `BIR_TRUNC` | `trunc` | Truncate integer to narrower width |
| `BIR_ZEXT` | `zext` | Zero-extend integer to wider width |
| `BIR_SEXT` | `sext` | Sign-extend integer to wider width |
| `BIR_FPTRUNC` | `fptrunc` | Truncate float to narrower width (e.g., `f64` -> `f32`, `f32` -> `f16`) |
| `BIR_FPEXT` | `fpext` | Extend float to wider width (e.g., `f16` -> `f32`, `f32` -> `f64`) |
| `BIR_FPTOSI` | `fptosi` | Float to signed integer |
| `BIR_FPTOUI` | `fptoui` | Float to unsigned integer |
| `BIR_SITOFP` | `sitofp` | Signed integer to float |
| `BIR_UITOFP` | `uitofp` | Unsigned integer to float |
| `BIR_PTRTOINT` | `ptrtoint` | Pointer to integer (reinterpret bits) |
| `BIR_INTTOPTR` | `inttoptr` | Integer to pointer (reinterpret bits) |
| `BIR_BITCAST` | `bitcast` | Bitwise reinterpret (operand and result must have same bit width) |

**Textual format:**
```
%2 = zext i16 %1 to i32
%4 = fptosi f32 %3 to i32
%6 = bitcast i32 %5 to f32
```

### Memory

Instructions for allocating and accessing memory.

#### `alloca`

```
%0 = alloca ptr<private, T> [, align N]
```

Allocates space for one value of type `T` on the per-thread stack. Returns a pointer in the `private` address space. The `subop` field encodes `log2(alignment)`. After `mem2reg`, scalar allocas whose only uses are plain loads and stores are promoted to SSA registers.

- **Operands:** None
- **Result type:** `ptr<private, T>`
- **subop:** `log2(alignment)` (0 if unspecified)

#### `shared_alloc`

```
%0 = shared_alloc ptr<shared, T>
```

Allocates per-block shared memory (CUDA `__shared__`). Not stack, not heap -- its own address space. The lowerer accumulates a cumulative LDS offset per function.

- **Operands:** None
- **Result type:** `ptr<shared, T>`

#### `global_ref`

```
%0 = global_ref ptr<AS, T>, @name
```

References a `__device__` or `__constant__` global variable. Yields a pointer to the global.

- **Operands:** None
- **Result type:** Pointer to the global's type
- **subop:** Global variable index in `globals[]`

#### `load`

```
%1 = load [volatile] T, %ptr
```

Loads a value of type `T` from the address in `%ptr`. The backend selects the appropriate load instruction family based on the address space of the pointer type.

- **Operands:** `ops[0]` = address
- **Result type:** The loaded value type
- **subop:** 0 = normal, 1 = volatile

#### `store`

```
store [volatile] T %val, %ptr
```

Stores `%val` to the address in `%ptr`. This is a void instruction (no result value).

- **Operands:** `ops[0]` = value, `ops[1]` = address
- **Result type:** `void`
- **subop:** 0 = normal, 1 = volatile

#### `gep` (Get Element Pointer)

```
%3 = gep ptr<AS, T>, %base, %idx0 [, %idx1, ...]
```

Computes a derived pointer from a base pointer plus typed indices. Equivalent to `base + sum(index[i] * element_size)`. Simpler than LLVM's GEP.

- **Operands:** `ops[0]` = base pointer, `ops[1..N]` = indices
- **Result type:** Pointer (same address space as base)

**Textual format:**
```
%3 = alloca ptr<private, i32>
%5 = load i32, %3
store i32 %0, %3
%8 = gep ptr<global, f32>, %1, %7
```

### Control Flow

Terminator instructions that end basic blocks.

#### `br`

```
br TARGET
```

Unconditional branch to a target basic block.

- **Operands:** `ops[0]` = target block index

#### `br_cond`

```
br_cond %cond, TRUE_BLOCK, FALSE_BLOCK
```

Conditional branch. If `%cond` (type `i1`) is true, transfers control to `TRUE_BLOCK`; otherwise to `FALSE_BLOCK`. On the AMDGPU backend, divergent conditional branches are lowered using EXEC mask save/restore.

- **Operands:** `ops[0]` = condition (`i1`), `ops[1]` = true block, `ops[2]` = false block

#### `switch`

```
switch %val, default BLOCK [, [CONST: BLOCK], ...]
```

Multi-way branch. Compares `%val` against a set of constant values, branching to the matching block or to the default block if no match. Case pairs are stored in `extra_operands[]` via the overflow mechanism.

- **Operands:** `ops[0]` = selector value, `ops[1]` = default block. Additional `(value, block)` pairs in overflow.

#### `ret`

```
ret T %val
ret void
```

Returns from the current function. For non-void functions, `ops[0]` is the return value. For void functions, `num_operands` is 0.

- **Operands:** `ops[0]` = return value (optional)

#### `unreachable`

```
unreachable
```

Marks a point in the program that should never be reached. Lowered to `s_endpgm` on AMDGPU.

- **Operands:** None

**Textual format:**
```
    br bb1
    br_cond %5, bb2, bb3
    switch %0, default bb4, [0: bb1], [1: bb2], [2: bb3]
    ret i32 %7
    ret void
    unreachable
```

### SSA

Instructions for SSA (Static Single Assignment) form.

#### `phi`

```
%5 = phi T [BLOCK_A: %val_a], [BLOCK_B: %val_b], ...
```

SSA phi node. Selects a value based on which predecessor block control flow arrived from. Operands are `(block, value)` pairs. Phi nodes are inserted by the `mem2reg` pass at iterated dominance frontiers. Must appear at the beginning of a basic block (before any non-phi instructions, except `param`).

- **Operands:** Alternating `(block_index, value_ref)` pairs. Uses overflow mode when there are more than 3 predecessors.
- **Result type:** The merged value type

#### `param`

```
%0 = param T
```

Declares a function parameter. Appears at the start of the entry block, one per parameter. The `subop` field encodes the parameter index. For kernel functions, parameters are loaded from the kernarg segment via scalar memory operations; for device functions, parameters arrive in VGPRs.

- **Operands:** None
- **Result type:** The parameter type
- **subop:** Parameter index (0, 1, 2, ...)

**Textual format:**
```
entry:
    %0 = param ptr<global, f32>
    %1 = param ptr<global, f32>
    %2 = param i32
    ...
merge:
    %10 = phi i32 [bb1: %5], [bb2: %8]
```

### Thread Model

Hardware-agnostic thread identification. These map directly to CUDA's `threadIdx`, `blockIdx`, `blockDim`, and `gridDim` builtins. The `subop` field encodes the dimension: 0=x, 1=y, 2=z.

| Opcode | Mnemonic | CUDA Equivalent | Divergence |
|--------|----------|-----------------|------------|
| `BIR_THREAD_ID` | `thread_id` | `threadIdx.{x,y,z}` | Always divergent |
| `BIR_BLOCK_ID` | `block_id` | `blockIdx.{x,y,z}` | Uniform (same for all threads in a block) |
| `BIR_BLOCK_DIM` | `block_dim` | `blockDim.{x,y,z}` | Uniform |
| `BIR_GRID_DIM` | `grid_dim` | `gridDim.{x,y,z}` | Uniform |

- **Operands:** None
- **Result type:** `i32`
- **subop:** Dimension (0=x, 1=y, 2=z)

**Textual format:**
```
%3 = thread_id.x
%4 = block_id.x
%5 = block_dim.x
```

### Barriers

Synchronization instructions. The IR says "synchronize here"; backends determine the actual mechanism.

| Opcode | Mnemonic | CUDA Equivalent | Backend (AMDGPU) |
|--------|----------|-----------------|-------------------|
| `BIR_BARRIER` | `barrier` | `__syncthreads()` | `s_waitcnt` + `s_barrier` |
| `BIR_BARRIER_GROUP` | `barrier_group` | Cooperative groups `.sync()` | `s_waitcnt` + `s_barrier` |

- **Operands:** None
- **Result type:** `void`

**Textual format:**
```
    barrier
    barrier_group
```

### Atomics

Atomic read-modify-write operations. The `subop` field encodes the memory ordering (see [Memory Ordering](#memory-ordering)). All atomic RMW operations take a pointer and a value, perform the operation atomically, and return the **old** value at the memory location.

| Opcode | Mnemonic | Operation | Operands |
|--------|----------|-----------|----------|
| `BIR_ATOMIC_ADD` | `atomic_add` | `*ptr += val` | `ops[0]` = ptr, `ops[1]` = val |
| `BIR_ATOMIC_SUB` | `atomic_sub` | `*ptr -= val` | `ops[0]` = ptr, `ops[1]` = val |
| `BIR_ATOMIC_AND` | `atomic_and` | `*ptr &= val` | `ops[0]` = ptr, `ops[1]` = val |
| `BIR_ATOMIC_OR` | `atomic_or` | `*ptr \|= val` | `ops[0]` = ptr, `ops[1]` = val |
| `BIR_ATOMIC_XOR` | `atomic_xor` | `*ptr ^= val` | `ops[0]` = ptr, `ops[1]` = val |
| `BIR_ATOMIC_MIN` | `atomic_min` | `*ptr = min(*ptr, val)` | `ops[0]` = ptr, `ops[1]` = val |
| `BIR_ATOMIC_MAX` | `atomic_max` | `*ptr = max(*ptr, val)` | `ops[0]` = ptr, `ops[1]` = val |
| `BIR_ATOMIC_XCHG` | `atomic_xchg` | `*ptr = val` (exchange) | `ops[0]` = ptr, `ops[1]` = val |
| `BIR_ATOMIC_CAS` | `atomic_cas` | Compare-and-swap | `ops[0]` = ptr, `ops[1]` = cmp, `ops[2]` = new |
| `BIR_ATOMIC_LOAD` | `atomic_load` | Atomic load with ordering | `ops[0]` = ptr |
| `BIR_ATOMIC_STORE` | `atomic_store` | Atomic store with ordering | `ops[0]` = val, `ops[1]` = ptr |

The backend dispatches to DS (LDS/shared) or GLOBAL atomic instructions based on the address space of the pointer.

**Textual format:**
```
%5 = atomic_add relaxed i32 %0, %4
%8 = atomic_cas seq_cst i32 %0, %6, %7
    atomic_store release i32 %3, %1
```

### Warp-Level Primitives

Warp/wavefront communication primitives. These are hardware-agnostic; backends lower them to the appropriate mechanism (e.g., `ds_bpermute_b32` on AMDGPU).

| Opcode | Mnemonic | CUDA Equivalent | Description |
|--------|----------|-----------------|-------------|
| `BIR_SHFL` | `shfl` | `__shfl_sync` | Read from arbitrary lane |
| `BIR_SHFL_UP` | `shfl_up` | `__shfl_up_sync` | Read from lane - delta |
| `BIR_SHFL_DOWN` | `shfl_down` | `__shfl_down_sync` | Read from lane + delta |
| `BIR_SHFL_XOR` | `shfl_xor` | `__shfl_xor_sync` | Read from lane ^ mask |
| `BIR_BALLOT` | `ballot` | `__ballot_sync` | Bitmask of lanes where predicate is true |
| `BIR_VOTE_ANY` | `vote_any` | `__any_sync` | True if any lane's predicate is true |
| `BIR_VOTE_ALL` | `vote_all` | `__all_sync` | True if all lanes' predicate is true |

**Textual format:**
```
%5 = shfl_xor i32 %3, %4
%7 = ballot i32 %6
%8 = vote_all i32 %6
```

### Miscellaneous

#### `call`

```
%5 = call T @func_name(%arg0, %arg1, ...)
```

Calls a function. `ops[0]` is the callee function index (into `funcs[]`). Remaining operands are arguments. Uses overflow mode for functions with more than 5 arguments.

- **Operands:** `ops[0]` = callee function index, `ops[1..N]` = arguments
- **Result type:** Function return type

#### `select`

```
%3 = select T %cond, %true_val, %false_val
```

Ternary select (equivalent to the C ternary operator `?:`). If `%cond` is true, produces `%true_val`; otherwise `%false_val`.

- **Operands:** `ops[0]` = condition (`i1`), `ops[1]` = true value, `ops[2]` = false value
- **Result type:** Same as the true/false value types

#### `inline_asm`

```
inline_asm "..."
```

Placeholder for inline assembly. Not yet fully implemented.

**Textual format:**
```
%6 = call f32 @add_vectors(%3, %5)
%4 = select i32 %2, %0, %1
```

---

## Comparison Predicates

The `subop` field of `icmp` and `fcmp` instructions selects the comparison predicate.

### Integer Predicates

| Predicate | Mnemonic | Description |
|-----------|----------|-------------|
| `BIR_ICMP_EQ` | `eq` | Equal |
| `BIR_ICMP_NE` | `ne` | Not equal |
| `BIR_ICMP_SLT` | `slt` | Signed less than |
| `BIR_ICMP_SLE` | `sle` | Signed less than or equal |
| `BIR_ICMP_SGT` | `sgt` | Signed greater than |
| `BIR_ICMP_SGE` | `sge` | Signed greater than or equal |
| `BIR_ICMP_ULT` | `ult` | Unsigned less than |
| `BIR_ICMP_ULE` | `ule` | Unsigned less than or equal |
| `BIR_ICMP_UGT` | `ugt` | Unsigned greater than |
| `BIR_ICMP_UGE` | `uge` | Unsigned greater than or equal |

### Floating-Point Predicates

Floating-point comparisons distinguish **ordered** (O-prefix) and **unordered** (U-prefix) behavior in the presence of NaN:
- **Ordered:** If either operand is NaN, the result is `false`.
- **Unordered:** If either operand is NaN, the result is `true`.

| Predicate | Mnemonic | Description |
|-----------|----------|-------------|
| `BIR_FCMP_OEQ` | `oeq` | Ordered equal |
| `BIR_FCMP_ONE` | `one` | Ordered not equal |
| `BIR_FCMP_OLT` | `olt` | Ordered less than |
| `BIR_FCMP_OLE` | `ole` | Ordered less than or equal |
| `BIR_FCMP_OGT` | `ogt` | Ordered greater than |
| `BIR_FCMP_OGE` | `oge` | Ordered greater than or equal |
| `BIR_FCMP_UEQ` | `ueq` | Unordered equal |
| `BIR_FCMP_UNE` | `une` | Unordered not equal |
| `BIR_FCMP_ULT` | `ult` | Unordered less than |
| `BIR_FCMP_ULE` | `ule` | Unordered less than or equal |
| `BIR_FCMP_UGT` | `ugt` | Unordered greater than |
| `BIR_FCMP_UGE` | `uge` | Unordered greater than or equal |
| `BIR_FCMP_ORD` | `ord` | Ordered (true if neither operand is NaN) |
| `BIR_FCMP_UNO` | `uno` | Unordered (true if either operand is NaN) |

---

## Memory Ordering

The `subop` field of atomic instructions encodes the memory ordering constraint:

| Ordering | Mnemonic | Description |
|----------|----------|-------------|
| `BIR_ORDER_RELAXED` | `relaxed` | No ordering constraint; only atomicity guaranteed |
| `BIR_ORDER_ACQUIRE` | `acquire` | Subsequent reads/writes cannot be reordered before this operation |
| `BIR_ORDER_RELEASE` | `release` | Prior reads/writes cannot be reordered after this operation |
| `BIR_ORDER_ACQ_REL` | `acq_rel` | Both acquire and release semantics |
| `BIR_ORDER_SEQ_CST` | `seq_cst` | Sequentially consistent; total order with all other seq_cst operations |

---

## Operand Overflow

Most instructions use 0-6 inline operand slots, which covers 99% of real instructions. When more operands are needed (phi nodes with many predecessors, function calls with many arguments, switch statements with many cases), the **overflow** mechanism is used:

- `num_operands` is set to `BIR_OPERANDS_OVERFLOW` (`0xFF`).
- `ops[0]` = start index into `extra_operands[]`.
- `ops[1]` = count of extra operands.

The extra operands are stored contiguously in the module's `extra_operands[]` array. The interpretation of the overflow data depends on the opcode:

| Opcode | Overflow Layout |
|--------|----------------|
| `BIR_PHI` | Alternating `(block_index, value_ref)` pairs |
| `BIR_CALL` | `extra[0]` = callee func index, `extra[1..]` = arguments |
| `BIR_SWITCH` | `extra[0]` = selector value, `extra[1]` = default block, then `(const_value, block)` pairs |

---

## IR Construction

### From CUDA Source (Normal Pipeline)

The `bir_lower()` function lowers an AST to BIR. It is the primary entry point for IR construction:

```c
int bir_lower(const parser_t *P, uint32_t ast_root, bir_module_t *M,
              const sema_ctx_t *sema);
```

The lowering proceeds in three passes:

1. **Collect declarations** -- Scans top-level AST nodes for struct definitions, enum definitions, typedefs, and global variables (`__shared__`, `__device__`, `__constant__`).

2. **Lower device functions** -- For each function marked `__global__` or `__device__`, creates a `bir_func_t` with an entry block, emits `BIR_PARAM` instructions, and recursively lowers the function body.

3. **Scan host functions** -- Inspects host-side functions for kernel launch syntax (`<<<...>>>`), triggering template instantiation as needed.

**Key lowering strategies:**
- **Local variables** are lowered to `alloca` + `store`/`load`. The `mem2reg` pass promotes these to SSA registers afterward.
- **CUDA builtins** (`threadIdx.x`, `blockIdx.y`, etc.) are pattern-matched during member access lowering and emitted as `thread_id`, `block_id`, etc. with the appropriate dimension subop.
- **Short-circuit `&&`/`||`** are lowered to conditional branches with `alloca` + `store` patterns. `mem2reg` converts these to phi nodes.
- **Operator overloading** is resolved by looking up `operator+`, `operator*`, etc. in the module's function list and emitting `call` instructions.
- **CUDA atomics** (`atomicAdd`, etc.) are lowered directly to the corresponding `atomic_*` instruction.
- **Half precision** builtins (`__float2half`, `__half2float`) are lowered to `fptrunc`/`fpext`.
- **Cooperative groups** methods (`.sync()`, `.thread_rank()`, `.size()`) are lowered to `barrier`, `thread_id`, and `block_dim` respectively.

### Programmatic Construction

The module can also be built programmatically using the API:

```c
bir_module_t M;
bir_module_init(&M);

// Create types
uint32_t i32  = bir_type_int(&M, 32);
uint32_t f32  = bir_type_float(&M, 32);
uint32_t ptr  = bir_type_ptr(&M, f32, BIR_AS_GLOBAL);

// Create constants
uint32_t zero = bir_const_int(&M, i32, 0);

// Add strings
uint32_t name = bir_add_string(&M, "my_kernel", 9);

// ... then populate funcs[], blocks[], insts[] directly
```

---

## Optimization Passes

### mem2reg

The sole optimization pass currently implemented. Classic SSA construction that promotes stack allocas to SSA registers:

```c
int bir_mem2reg(bir_module_t *M);  // Returns number of allocas promoted
```

**Algorithm:**

1. **Identify promotable allocas** -- Scalar allocas (int, float, ptr) whose only uses are plain loads and stores. Excludes: GEP uses, call arguments, volatile accesses, address-taken allocas, array/struct allocas.
2. **Build predecessor lists** from block terminators.
3. **Compute dominator tree** using the Cooper-Harvey-Kennedy iterative algorithm.
4. **Compute dominance frontiers.**
5. **Insert phi nodes** at iterated dominance frontiers of each alloca's store set.
6. **Rename variables** via iterative domtree walk with a save/restore stack. Stores become definitions; loads resolve to the current reaching definition.
7. **Compact** -- Dead `alloca`/`store`/`load` instructions are removed. All references are remapped and instructions are slid to close gaps.

**Before mem2reg:**
```
entry:
    %2 = alloca ptr<private, i32>
    store i32 %0, %2
    br bb1
bb1:
    %5 = load i32, %2
    ...
```

**After mem2reg:**
```
entry:
    br bb1
bb1:
    %5 = phi i32 [entry: %0], ...
    ...
```

---

## Textual IR Format

The textual IR is produced by `bir_print_module()` and can be dumped with `./kath --ir kernel.cu`. The format follows these conventions:

### Module Structure
```
; Booth IR

@global_name = global|constant TYPE [initializer] [cuda_flags]

func @function_name(TYPE %0, TYPE %1, ...) [cuda_flags] [launch_bounds] {
block_label:
    INSTRUCTION  ; line N
    ...
}
```

### Value Numbering

Values are numbered sequentially within each function, starting at `%0` for the first parameter. The numbering is based on instruction index relative to the function's first instruction.

### Source Locations

Each instruction is annotated with its source line number as a trailing comment:
```
    %5 = add i32 %3, %4  ; line 42
```

### Complete Example

```
; Booth IR

func @vector_add(ptr<global, f32> %0, ptr<global, f32> %1, ptr<global, f32> %2, i32 %3) __global__ {
entry:
    %4 = thread_id.x  ; line 3
    %5 = block_id.x  ; line 3
    %6 = block_dim.x  ; line 3
    %7 = mul i32 %5, %6  ; line 3
    %8 = add i32 %4, %7  ; line 3
    %9 = icmp slt i32 %8, %3  ; line 4
    br_cond %9, if.then, if.end  ; line 4

if.then:
    %10 = sext i32 %8 to i64  ; line 5
    %11 = gep ptr<global, f32>, %1, %10  ; line 5
    %12 = load f32, %11  ; line 5
    %13 = gep ptr<global, f32>, %2, %10  ; line 5
    %14 = load f32, %13  ; line 5
    %15 = fadd f32 %12, %14  ; line 5
    %16 = gep ptr<global, f32>, %0, %10  ; line 5
    store f32 %15, %16  ; line 5
    br if.end  ; line 5

if.end:
    ret void  ; line 6
}
```

---

## Limits

All array sizes are compile-time constants defined in `bir.h`:

| Constant | Value | Description |
|----------|-------|-------------|
| `BIR_MAX_TYPES` | 2^14 = 16,384 | Maximum interned types |
| `BIR_MAX_TYPE_FIELDS` | 2^16 = 65,536 | Maximum struct/function field entries |
| `BIR_MAX_CONSTS` | 2^16 = 65,536 | Maximum constants |
| `BIR_MAX_INSTS` | 2^18 = 262,144 | Maximum instructions (all functions combined) |
| `BIR_MAX_EXTRA_OPS` | 2^16 = 65,536 | Maximum overflow operands |
| `BIR_MAX_BLOCKS` | 2^16 = 65,536 | Maximum basic blocks |
| `BIR_MAX_FUNCS` | 2^12 = 4,096 | Maximum functions |
| `BIR_MAX_GLOBALS` | 2^12 = 4,096 | Maximum global variables |
| `BIR_MAX_STRINGS` | 2^20 = 1,048,576 | Maximum string table bytes (1 MB) |

These limits are designed to handle real-world CUDA programs. If exceeded, the relevant construction function returns 0 (the void type / first constant index) as a safe fallback.
