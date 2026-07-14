# BIR Instruction Set

This is the complete instruction listing for the Booth Intermediate Representation. Each entry specifies the mnemonic, syntax, operand layout, subop encoding, result type, and operational semantics.

Conventions used throughout this document:

- `%N` refers to an SSA value (instruction result, referenced by index).
- Bare integers (`0`, `42`, `-1`) and floats (`3.14`) are inline constants.
- `null`, `undef`, `zeroinit` are special constant forms.
- `T` stands for any BIR type (`i32`, `f32`, `ptr<global, f32>`, etc.).
- Block labels appear as `bb0`, `bb1`, or by name (`entry`, `if.then`).
- `subop` is a per-instruction modifier field packed into the instruction word. Its meaning depends on the opcode.

---

## Arithmetic

All binary. Both operands and the result share the same type.

### `add`

Integer addition.

```
%r = add T %a, %b
```

| Field | Value |
|-------|-------|
| Operands | `ops[0]` = lhs, `ops[1]` = rhs |
| Result | `T` (same as operands) |
| subop | unused |

Computes `%a + %b`. Wraps on overflow (two's complement).

---

### `sub`

Integer subtraction.

```
%r = sub T %a, %b
```

| Field | Value |
|-------|-------|
| Operands | `ops[0]` = lhs, `ops[1]` = rhs |
| Result | `T` |
| subop | unused |

Computes `%a - %b`. Wraps on underflow.

---

### `mul`

Integer multiplication.

```
%r = mul T %a, %b
```

| Field | Value |
|-------|-------|
| Operands | `ops[0]` = lhs, `ops[1]` = rhs |
| Result | `T` |
| subop | unused |

Computes `%a * %b`. Result truncated to the operand width.

---

### `sdiv`

Signed integer division.

```
%r = sdiv T %a, %b
```

| Field | Value |
|-------|-------|
| Operands | `ops[0]` = dividend, `ops[1]` = divisor |
| Result | `T` |
| subop | unused |

Computes `%a / %b`, treating both operands as signed. Division by zero is undefined.

---

### `udiv`

Unsigned integer division.

```
%r = udiv T %a, %b
```

| Field | Value |
|-------|-------|
| Operands | `ops[0]` = dividend, `ops[1]` = divisor |
| Result | `T` |
| subop | unused |

Computes `%a / %b`, treating both operands as unsigned. Division by zero is undefined.

---

### `srem`

Signed integer remainder.

```
%r = srem T %a, %b
```

| Field | Value |
|-------|-------|
| Operands | `ops[0]` = dividend, `ops[1]` = divisor |
| Result | `T` |
| subop | unused |

Computes `%a % %b` with the sign of the dividend. Division by zero is undefined.

---

### `urem`

Unsigned integer remainder.

```
%r = urem T %a, %b
```

| Field | Value |
|-------|-------|
| Operands | `ops[0]` = dividend, `ops[1]` = divisor |
| Result | `T` |
| subop | unused |

Computes `%a % %b`, treating both operands as unsigned. Division by zero is undefined.

---

### `fadd`

Floating-point addition.

```
%r = fadd T %a, %b
```

| Field | Value |
|-------|-------|
| Operands | `ops[0]` = lhs, `ops[1]` = rhs |
| Result | `T` (f16, f32, or f64) |
| subop | unused |

IEEE 754 addition.

---

### `fsub`

Floating-point subtraction.

```
%r = fsub T %a, %b
```

| Field | Value |
|-------|-------|
| Operands | `ops[0]` = lhs, `ops[1]` = rhs |
| Result | `T` |
| subop | unused |

IEEE 754 subtraction.

---

### `fmul`

Floating-point multiplication.

```
%r = fmul T %a, %b
```

| Field | Value |
|-------|-------|
| Operands | `ops[0]` = lhs, `ops[1]` = rhs |
| Result | `T` |
| subop | unused |

IEEE 754 multiplication.

---

### `fdiv`

Floating-point division.

```
%r = fdiv T %a, %b
```

| Field | Value |
|-------|-------|
| Operands | `ops[0]` = dividend, `ops[1]` = divisor |
| Result | `T` |
| subop | unused |

IEEE 754 division.

---

### `frem`

Floating-point remainder.

```
%r = frem T %a, %b
```

| Field | Value |
|-------|-------|
| Operands | `ops[0]` = dividend, `ops[1]` = divisor |
| Result | `T` |
| subop | unused |

IEEE 754 remainder.

---

## Bitwise

Integer bitwise and shift operations. Both operands and the result share the same integer type.

### `and`

Bitwise AND.

```
%r = and T %a, %b
```

| Field | Value |
|-------|-------|
| Operands | `ops[0]` = lhs, `ops[1]` = rhs |
| Result | `T` |
| subop | unused |

---

### `or`

Bitwise OR.

```
%r = or T %a, %b
```

| Field | Value |
|-------|-------|
| Operands | `ops[0]` = lhs, `ops[1]` = rhs |
| Result | `T` |
| subop | unused |

---

### `xor`

Bitwise XOR.

```
%r = xor T %a, %b
```

| Field | Value |
|-------|-------|
| Operands | `ops[0]` = lhs, `ops[1]` = rhs |
| Result | `T` |
| subop | unused |

---

### `shl`

Shift left.

```
%r = shl T %a, %b
```

| Field | Value |
|-------|-------|
| Operands | `ops[0]` = value, `ops[1]` = shift amount |
| Result | `T` |
| subop | unused |

Shifts `%a` left by `%b` bits. Vacated low bits are filled with zero.

---

### `lshr`

Logical shift right.

```
%r = lshr T %a, %b
```

| Field | Value |
|-------|-------|
| Operands | `ops[0]` = value, `ops[1]` = shift amount |
| Result | `T` |
| subop | unused |

Shifts `%a` right by `%b` bits. Vacated high bits are filled with zero.

---

### `ashr`

Arithmetic shift right.

```
%r = ashr T %a, %b
```

| Field | Value |
|-------|-------|
| Operands | `ops[0]` = value, `ops[1]` = shift amount |
| Result | `T` |
| subop | unused |

Shifts `%a` right by `%b` bits. Vacated high bits are filled with copies of the sign bit.

---

## Comparison

Both comparison instructions produce an `i1` result. The `subop` field selects the predicate.

### `icmp`

Integer comparison.

```
%r = icmp PRED OPERAND_TYPE %a, %b
```

| Field | Value |
|-------|-------|
| Operands | `ops[0]` = lhs, `ops[1]` = rhs |
| Result | `i1` |
| subop | comparison predicate (see table below) |

Note: the printed type is the *operand* type, not the result type.

**Integer predicates:**

| subop | Mnemonic | Meaning |
|-------|----------|---------|
| 0 | `eq` | equal |
| 1 | `ne` | not equal |
| 2 | `slt` | signed less than |
| 3 | `sle` | signed less than or equal |
| 4 | `sgt` | signed greater than |
| 5 | `sge` | signed greater than or equal |
| 6 | `ult` | unsigned less than |
| 7 | `ule` | unsigned less than or equal |
| 8 | `ugt` | unsigned greater than |
| 9 | `uge` | unsigned greater than or equal |

---

### `fcmp`

Floating-point comparison.

```
%r = fcmp PRED OPERAND_TYPE %a, %b
```

| Field | Value |
|-------|-------|
| Operands | `ops[0]` = lhs, `ops[1]` = rhs |
| Result | `i1` |
| subop | comparison predicate (see table below) |

**Floating-point predicates:**

Ordered predicates (O-prefix) return `false` if either operand is NaN. Unordered predicates (U-prefix) return `true` if either operand is NaN.

| subop | Mnemonic | Meaning |
|-------|----------|---------|
| 10 | `oeq` | ordered equal |
| 11 | `one` | ordered not equal |
| 12 | `olt` | ordered less than |
| 13 | `ole` | ordered less than or equal |
| 14 | `ogt` | ordered greater than |
| 15 | `oge` | ordered greater than or equal |
| 16 | `ueq` | unordered equal |
| 17 | `une` | unordered not equal |
| 18 | `ult` | unordered less than |
| 19 | `ule` | unordered less than or equal |
| 20 | `ugt` | unordered greater than |
| 21 | `uge` | unordered greater than or equal |
| 22 | `ord` | ordered (true if neither operand is NaN) |
| 23 | `uno` | unordered (true if either operand is NaN) |

---

## Conversion

Unary type conversion instructions. Each takes a single operand and converts it to the instruction's result type.

### `trunc`

Truncate integer to a narrower width.

```
%r = trunc SRC_TYPE %a to DST_TYPE
```

| Field | Value |
|-------|-------|
| Operands | `ops[0]` = source value |
| Result | narrower integer type |
| subop | unused |

The source width must be strictly greater than the destination width. High bits are discarded.

---

### `zext`

Zero-extend integer to a wider width.

```
%r = zext SRC_TYPE %a to DST_TYPE
```

| Field | Value |
|-------|-------|
| Operands | `ops[0]` = source value |
| Result | wider integer type |
| subop | unused |

The destination width must be strictly greater than the source width. High bits are filled with zero.

---

### `sext`

Sign-extend integer to a wider width.

```
%r = sext SRC_TYPE %a to DST_TYPE
```

| Field | Value |
|-------|-------|
| Operands | `ops[0]` = source value |
| Result | wider integer type |
| subop | unused |

High bits are filled with copies of the source sign bit.

---

### `fptrunc`

Truncate floating-point to a narrower width.

```
%r = fptrunc SRC_TYPE %a to DST_TYPE
```

| Field | Value |
|-------|-------|
| Operands | `ops[0]` = source value |
| Result | narrower float type |
| subop | unused |

Converts between float widths (e.g., `f64` to `f32`, `f32` to `f16`). May lose precision.

---

### `fpext`

Extend floating-point to a wider width.

```
%r = fpext SRC_TYPE %a to DST_TYPE
```

| Field | Value |
|-------|-------|
| Operands | `ops[0]` = source value |
| Result | wider float type |
| subop | unused |

Lossless widening (e.g., `f16` to `f32`, `f32` to `f64`).

---

### `fptosi`

Floating-point to signed integer.

```
%r = fptosi SRC_TYPE %a to DST_TYPE
```

| Field | Value |
|-------|-------|
| Operands | `ops[0]` = float value |
| Result | integer type |
| subop | unused |

Truncates toward zero. Overflow is undefined.

---

### `fptoui`

Floating-point to unsigned integer.

```
%r = fptoui SRC_TYPE %a to DST_TYPE
```

| Field | Value |
|-------|-------|
| Operands | `ops[0]` = float value |
| Result | integer type |
| subop | unused |

Truncates toward zero. Negative inputs and overflow are undefined.

---

### `sitofp`

Signed integer to floating-point.

```
%r = sitofp SRC_TYPE %a to DST_TYPE
```

| Field | Value |
|-------|-------|
| Operands | `ops[0]` = signed integer value |
| Result | float type |
| subop | unused |

May lose precision for large values.

---

### `uitofp`

Unsigned integer to floating-point.

```
%r = uitofp SRC_TYPE %a to DST_TYPE
```

| Field | Value |
|-------|-------|
| Operands | `ops[0]` = unsigned integer value |
| Result | float type |
| subop | unused |

May lose precision for large values.

---

### `ptrtoint`

Pointer to integer.

```
%r = ptrtoint SRC_TYPE %a to DST_TYPE
```

| Field | Value |
|-------|-------|
| Operands | `ops[0]` = pointer value |
| Result | integer type |
| subop | unused |

Reinterprets the pointer address as an integer. The integer width should be large enough to hold the address.

---

### `inttoptr`

Integer to pointer.

```
%r = inttoptr SRC_TYPE %a to DST_TYPE
```

| Field | Value |
|-------|-------|
| Operands | `ops[0]` = integer value |
| Result | pointer type |
| subop | unused |

Reinterprets an integer as a pointer address.

---

### `bitcast`

Bitwise reinterpretation.

```
%r = bitcast SRC_TYPE %a to DST_TYPE
```

| Field | Value |
|-------|-------|
| Operands | `ops[0]` = source value |
| Result | destination type |
| subop | unused |

Source and destination must have the same bit width. No bits are changed; only the type interpretation differs.

---

## Memory

### `alloca`

Per-thread stack allocation.

```
%r = alloca PTR_TYPE [, align N]
```

| Field | Value |
|-------|-------|
| Operands | none |
| Result | `ptr<private, T>` where `T` is the allocated element type |
| subop | `log2(alignment)` (0 = default) |

Allocates space for one element of type `T` on the per-thread stack. The result is a pointer in the `private` address space.

Scalar allocas (int, float, ptr) that are only used by plain loads and stores are promoted to SSA registers by the `mem2reg` pass.

---

### `shared_alloc`

Per-block shared memory allocation.

```
%r = shared_alloc PTR_TYPE
```

| Field | Value |
|-------|-------|
| Operands | none |
| Result | `ptr<shared, T>` |
| subop | unused |

Allocates per-block shared memory for CUDA `__shared__` variables. This memory lives in the shared address space (LDS on AMD hardware). All threads in a block see the same allocation.

---

### `global_ref`

Reference to a global variable.

```
%r = global_ref PTR_TYPE, @name
```

| Field | Value |
|-------|-------|
| Operands | none |
| Result | pointer to the global's type |
| subop | index into the module's `globals[]` array |

Produces a pointer to a `__device__` or `__constant__` global variable.

---

### `load`

Load from memory.

```
%r = load [volatile] T, %ptr
```

| Field | Value |
|-------|-------|
| Operands | `ops[0]` = address (pointer) |
| Result | `T` (the loaded value type) |
| subop | 0 = normal, 1 = volatile |

Reads a value of type `T` from the given address. The backend selects the appropriate load instruction based on the address space of the pointer type (constant, shared, global, or private).

Volatile loads are never reordered or eliminated.

---

### `store`

Store to memory.

```
store [volatile] T %val, %ptr
```

| Field | Value |
|-------|-------|
| Operands | `ops[0]` = value, `ops[1]` = address (pointer) |
| Result | `void` |
| subop | 0 = normal, 1 = volatile |

Writes `%val` to the given address. This instruction produces no result value.

Volatile stores are never reordered or eliminated.

---

### `gep`

Get element pointer. Computes a derived pointer from a base pointer and typed indices.

```
%r = gep PTR_TYPE, %base, %idx0 [, %idx1, ...]
```

| Field | Value |
|-------|-------|
| Operands | `ops[0]` = base pointer, `ops[1..N]` = indices |
| Result | pointer type (same address space as base) |
| subop | unused |

Adds byte offsets to the base pointer computed from each index multiplied by the corresponding element size. The result pointer inherits the address space of the base.

---

## Control Flow

Terminator instructions. Every basic block must end with exactly one of these.

### `br`

Unconditional branch.

```
br TARGET
```

| Field | Value |
|-------|-------|
| Operands | `ops[0]` = target block index |
| Result | `void` |
| subop | unused |

Transfers control to the target block unconditionally.

---

### `br_cond`

Conditional branch.

```
br_cond %cond, TRUE_BLOCK, FALSE_BLOCK
```

| Field | Value |
|-------|-------|
| Operands | `ops[0]` = condition (`i1`), `ops[1]` = true block, `ops[2]` = false block |
| Result | `void` |
| subop | unused |

If `%cond` is nonzero, transfers control to `TRUE_BLOCK`. Otherwise transfers to `FALSE_BLOCK`.

On GPU backends, divergent conditional branches (where threads within a wavefront disagree on the condition) are handled via execution mask manipulation.

---

### `switch`

Multi-way branch.

```
switch %val, default DEFAULT_BLOCK [, [CONST: BLOCK], ...]
```

| Field | Value |
|-------|-------|
| Operands | `ops[0]` = selector value, `ops[1]` = default block. Case pairs in overflow. |
| Result | `void` |
| subop | unused |

Compares `%val` against each case constant. If a match is found, branches to the corresponding block. If no match, branches to the default block.

Uses the overflow mechanism when there are case entries. In overflow mode: `extra[0]` = selector, `extra[1]` = default block, then alternating `(constant, block)` pairs.

---

### `ret`

Return from function.

```
ret T %val
ret void
```

| Field | Value |
|-------|-------|
| Operands | `ops[0]` = return value (omitted for void) |
| Result | `void` |
| subop | unused |

When `num_operands` is 0, this is a void return. Otherwise `ops[0]` holds the return value.

---

### `unreachable`

Marks unreachable code.

```
unreachable
```

| Field | Value |
|-------|-------|
| Operands | none |
| Result | `void` |
| subop | unused |

Indicates that this point in the program should never be executed. Reaching this instruction is undefined behavior. On AMDGPU, lowered to `s_endpgm`.

---

## SSA

### `phi`

SSA phi node.

```
%r = phi T [BLOCK_A: %val_a], [BLOCK_B: %val_b], ...
```

| Field | Value |
|-------|-------|
| Operands | alternating `(block_index, value_ref)` pairs |
| Result | `T` |
| subop | unused |

Selects a value based on which predecessor block control flow came from. Each `(block, value)` pair specifies: "if we arrived from this block, use this value."

Phi nodes must appear at the beginning of a block, before any non-phi, non-param instructions. They are inserted by the `mem2reg` pass.

With 3 or fewer predecessors, the pairs fit in the 6 inline operand slots. With more predecessors, the overflow mechanism is used.

---

### `param`

Function parameter.

```
%r = param T
```

| Field | Value |
|-------|-------|
| Operands | none |
| Result | `T` (the parameter type) |
| subop | parameter index (0, 1, 2, ...) |

Declares a function parameter. These appear at the start of the entry block, in order. For kernel functions (`__global__`), parameters are loaded from the kernel argument segment. For device functions (`__device__`), parameters arrive in registers.

---

## Thread Model

Hardware-agnostic thread identification. These have no operands and produce an `i32` result. The `subop` field selects the dimension: 0 = x, 1 = y, 2 = z.

### `thread_id`

Per-thread index within the block.

```
%r = thread_id.x
%r = thread_id.y
%r = thread_id.z
```

| Field | Value |
|-------|-------|
| Operands | none |
| Result | `i32` |
| subop | dimension (0=x, 1=y, 2=z) |

CUDA equivalent: `threadIdx.x`, `threadIdx.y`, `threadIdx.z`.

This value is **always divergent** -- different threads within a wavefront hold different values.

---

### `block_id`

Block index within the grid.

```
%r = block_id.x
%r = block_id.y
%r = block_id.z
```

| Field | Value |
|-------|-------|
| Operands | none |
| Result | `i32` |
| subop | dimension (0=x, 1=y, 2=z) |

CUDA equivalent: `blockIdx.x`, `blockIdx.y`, `blockIdx.z`.

This value is **uniform** -- all threads in a block see the same value.

---

### `block_dim`

Block dimensions (number of threads per block).

```
%r = block_dim.x
%r = block_dim.y
%r = block_dim.z
```

| Field | Value |
|-------|-------|
| Operands | none |
| Result | `i32` |
| subop | dimension (0=x, 1=y, 2=z) |

CUDA equivalent: `blockDim.x`, `blockDim.y`, `blockDim.z`.

This value is **uniform**.

---

### `grid_dim`

Grid dimensions (number of blocks in the grid).

```
%r = grid_dim.x
%r = grid_dim.y
%r = grid_dim.z
```

| Field | Value |
|-------|-------|
| Operands | none |
| Result | `i32` |
| subop | dimension (0=x, 1=y, 2=z) |

CUDA equivalent: `gridDim.x`, `gridDim.y`, `gridDim.z`.

This value is **uniform**.

---

## Barriers

Synchronization points. These produce no value and have no operands.

### `barrier`

Block-level thread synchronization.

```
barrier
```

| Field | Value |
|-------|-------|
| Operands | none |
| Result | `void` |
| subop | unused |

All threads in the block must reach this point before any thread proceeds past it. Memory operations issued before the barrier are visible to all threads in the block after the barrier.

CUDA equivalent: `__syncthreads()`.

---

### `barrier_group`

Group-level synchronization.

```
barrier_group
```

| Field | Value |
|-------|-------|
| Operands | none |
| Result | `void` |
| subop | unused |

Synchronizes threads within a cooperative group. Currently lowered identically to `barrier` on supported backends.

---

## Atomics

Atomic memory operations. All read-modify-write instructions take a pointer and a value, perform the operation atomically, and return the **old** value at the memory location. The `subop` field encodes the memory ordering.

**Memory orderings (subop values):**

| subop | Mnemonic | Meaning |
|-------|----------|---------|
| 0 | `relaxed` | No ordering constraint, only atomicity |
| 1 | `acquire` | Subsequent operations cannot move before this one |
| 2 | `release` | Prior operations cannot move after this one |
| 3 | `acq_rel` | Both acquire and release |
| 4 | `seq_cst` | Total order with all other seq_cst operations |

The backend selects DS (shared memory) or GLOBAL atomic instructions based on the address space of the pointer operand.

### `atomic_add`

Atomic addition.

```
%r = atomic_add ORDER T %ptr, %val
```

| Field | Value |
|-------|-------|
| Operands | `ops[0]` = pointer, `ops[1]` = value |
| Result | `T` (old value at `%ptr`) |
| subop | memory ordering |

Atomically replaces `*ptr` with `*ptr + val`, returns the old value.

---

### `atomic_sub`

Atomic subtraction.

```
%r = atomic_sub ORDER T %ptr, %val
```

| Field | Value |
|-------|-------|
| Operands | `ops[0]` = pointer, `ops[1]` = value |
| Result | `T` (old value) |
| subop | memory ordering |

Atomically replaces `*ptr` with `*ptr - val`, returns the old value.

---

### `atomic_and`

Atomic bitwise AND.

```
%r = atomic_and ORDER T %ptr, %val
```

| Field | Value |
|-------|-------|
| Operands | `ops[0]` = pointer, `ops[1]` = value |
| Result | `T` (old value) |
| subop | memory ordering |

Atomically replaces `*ptr` with `*ptr & val`, returns the old value.

---

### `atomic_or`

Atomic bitwise OR.

```
%r = atomic_or ORDER T %ptr, %val
```

| Field | Value |
|-------|-------|
| Operands | `ops[0]` = pointer, `ops[1]` = value |
| Result | `T` (old value) |
| subop | memory ordering |

Atomically replaces `*ptr` with `*ptr | val`, returns the old value.

---

### `atomic_xor`

Atomic bitwise XOR.

```
%r = atomic_xor ORDER T %ptr, %val
```

| Field | Value |
|-------|-------|
| Operands | `ops[0]` = pointer, `ops[1]` = value |
| Result | `T` (old value) |
| subop | memory ordering |

Atomically replaces `*ptr` with `*ptr ^ val`, returns the old value.

---

### `atomic_min`

Atomic minimum.

```
%r = atomic_min ORDER T %ptr, %val
```

| Field | Value |
|-------|-------|
| Operands | `ops[0]` = pointer, `ops[1]` = value |
| Result | `T` (old value) |
| subop | memory ordering |

Atomically replaces `*ptr` with `min(*ptr, val)`, returns the old value.

---

### `atomic_max`

Atomic maximum.

```
%r = atomic_max ORDER T %ptr, %val
```

| Field | Value |
|-------|-------|
| Operands | `ops[0]` = pointer, `ops[1]` = value |
| Result | `T` (old value) |
| subop | memory ordering |

Atomically replaces `*ptr` with `max(*ptr, val)`, returns the old value.

---

### `atomic_xchg`

Atomic exchange.

```
%r = atomic_xchg ORDER T %ptr, %val
```

| Field | Value |
|-------|-------|
| Operands | `ops[0]` = pointer, `ops[1]` = value |
| Result | `T` (old value) |
| subop | memory ordering |

Atomically replaces `*ptr` with `val`, returns the old value.

---

### `atomic_cas`

Atomic compare-and-swap.

```
%r = atomic_cas ORDER T %ptr, %cmp, %new
```

| Field | Value |
|-------|-------|
| Operands | `ops[0]` = pointer, `ops[1]` = expected value, `ops[2]` = new value |
| Result | `T` (old value at `%ptr`) |
| subop | memory ordering |

If `*ptr == cmp`, atomically replaces `*ptr` with `new`. In either case, returns the old value. The caller can compare the returned value against `cmp` to determine whether the swap occurred.

---

### `atomic_load`

Atomic load.

```
%r = atomic_load ORDER T, %ptr
```

| Field | Value |
|-------|-------|
| Operands | `ops[0]` = pointer |
| Result | `T` (loaded value) |
| subop | memory ordering |

Atomically reads the value at `%ptr` with the specified ordering guarantee.

---

### `atomic_store`

Atomic store.

```
atomic_store ORDER T %val, %ptr
```

| Field | Value |
|-------|-------|
| Operands | `ops[0]` = value, `ops[1]` = pointer |
| Result | `void` |
| subop | memory ordering |

Atomically writes `%val` to `%ptr` with the specified ordering guarantee.

---

## Warp-Level Primitives

Cross-lane communication within a wavefront. These are target-independent; backends lower them to the appropriate hardware mechanism.

### `shfl`

Warp shuffle. Read a value from an arbitrary lane.

```
%r = shfl T %val, %lane [, %mask]
```

| Field | Value |
|-------|-------|
| Operands | `ops[0]` = source value, `ops[1]` = source lane, `ops[2]` = width mask (if present) |
| Result | `T` |
| subop | unused |

CUDA equivalent: `__shfl_sync`.

---

### `shfl_up`

Shuffle up. Read from a lower-numbered lane.

```
%r = shfl_up T %val, %delta [, %mask]
```

| Field | Value |
|-------|-------|
| Operands | `ops[0]` = source value, `ops[1]` = delta, `ops[2]` = width mask (if present) |
| Result | `T` |
| subop | unused |

Reads the value from lane `(current_lane - delta)`. If the source lane is out of range, the thread's own value is returned.

CUDA equivalent: `__shfl_up_sync`.

---

### `shfl_down`

Shuffle down. Read from a higher-numbered lane.

```
%r = shfl_down T %val, %delta [, %mask]
```

| Field | Value |
|-------|-------|
| Operands | `ops[0]` = source value, `ops[1]` = delta, `ops[2]` = width mask (if present) |
| Result | `T` |
| subop | unused |

Reads the value from lane `(current_lane + delta)`. If the source lane is out of range, the thread's own value is returned.

CUDA equivalent: `__shfl_down_sync`.

---

### `shfl_xor`

Shuffle XOR. Read from a lane computed by XOR.

```
%r = shfl_xor T %val, %mask_val [, %width]
```

| Field | Value |
|-------|-------|
| Operands | `ops[0]` = source value, `ops[1]` = XOR mask, `ops[2]` = width (if present) |
| Result | `T` |
| subop | unused |

Reads the value from lane `(current_lane ^ mask_val)`. Commonly used for butterfly reduction patterns.

CUDA equivalent: `__shfl_xor_sync`.

---

### `ballot`

Warp ballot. Returns a bitmask of active lanes where the predicate is true.

```
%r = ballot T %pred
```

| Field | Value |
|-------|-------|
| Operands | `ops[0]` = predicate value |
| Result | `i32` (bitmask) |
| subop | unused |

Each bit in the result corresponds to one lane. Bit `N` is set if lane `N`'s predicate is nonzero.

CUDA equivalent: `__ballot_sync`.

---

### `vote_any`

True if any lane's predicate is true.

```
%r = vote_any T %pred
```

| Field | Value |
|-------|-------|
| Operands | `ops[0]` = predicate value |
| Result | `i32` (boolean: 0 or 1) |
| subop | unused |

Returns nonzero if at least one active lane has a nonzero predicate.

CUDA equivalent: `__any_sync`.

---

### `vote_all`

True if all lanes' predicates are true.

```
%r = vote_all T %pred
```

| Field | Value |
|-------|-------|
| Operands | `ops[0]` = predicate value |
| Result | `i32` (boolean: 0 or 1) |
| subop | unused |

Returns nonzero only if every active lane has a nonzero predicate.

CUDA equivalent: `__all_sync`.

---

## Miscellaneous

### `call`

Function call.

```
%r = call T @function_name(%arg0, %arg1, ...)
```

| Field | Value |
|-------|-------|
| Operands | `ops[0]` = callee function index (into `funcs[]`), `ops[1..N]` = arguments |
| Result | `T` (function return type) |
| subop | unused |

Calls the named function with the given arguments. The return type is determined by the callee's function type.

With 5 or fewer arguments, the callee index and arguments fit in the 6 inline operand slots. With more arguments, the overflow mechanism is used: `extra[0]` = callee index, `extra[1..]` = arguments.

---

### `select`

Conditional value selection.

```
%r = select T %cond, %true_val, %false_val
```

| Field | Value |
|-------|-------|
| Operands | `ops[0]` = condition (`i1`), `ops[1]` = value if true, `ops[2]` = value if false |
| Result | `T` (same as the true/false value types) |
| subop | unused |

If `%cond` is nonzero, produces `%true_val`. Otherwise produces `%false_val`. This is the SSA-friendly equivalent of the C ternary operator and does not introduce control flow.

---

### `inline_asm`

Inline assembly block.

```
inline_asm "..."
```

| Field | Value |
|-------|-------|
| Operands | implementation-defined |
| Result | implementation-defined |
| subop | unused |

Placeholder for inline assembly. Allows direct emission of target-specific instructions. Not yet fully implemented.

---

## Reserved

Opcodes 128-143 (`BIR_TEX_BASE` through `BIR_TEX_MAX`) are reserved for future texture and surface operations. These are not yet implemented.
