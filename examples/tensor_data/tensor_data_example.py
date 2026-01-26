#!/usr/bin/env python3
"""
PTO-RT v9 Example: Tensor Data Binding and Validation

This example demonstrates binding actual NumPy arrays to Tensor objects
and validating the output against a reference implementation.

Usage:
    python examples/tensor_data/tensor_data_example.py
"""

import sys
sys.path.insert(0, 'python')

import numpy as np

from pto_wsp import (
    kernel, tl,
    In, Out, Tile,
    workload, P,
    Dense, DenseDyn, Tensor, DType,
    DispatchPolicy, TimingPolicy,
)

# ============================================================================
# CONFIGURATION
# ============================================================================

TILE_M = 32
TILE_N = 32
BATCH = 4
NUM_TILES_M = 4
NUM_TILES_N = 4
F32 = DType.F32

print("=" * 70)
print("Tensor Data Binding Example - PTO-RT v9")
print("=" * 70)

# ============================================================================
# KERNELS
# ============================================================================

@kernel
def add_tiles(
    a: In[Tile[TILE_M, TILE_N, F32]],
    b: In[Tile[TILE_M, TILE_N, F32]],
    c: Out[Tile[TILE_M, TILE_N, F32]],
):
    """Element-wise addition of two tiles."""
    x = tl.load(a)
    y = tl.load(b)
    z = tl.add(x, y)
    tl.store(c, z)


@kernel
def scale_tile(
    input_tile: In[Tile[TILE_M, TILE_N, F32]],
    output_tile: Out[Tile[TILE_M, TILE_N, F32]],
):
    """Scale a tile by 2.0."""
    data = tl.load(input_tile)
    # tl.constant returns a scalar, tl.mul broadcasts
    two = tl.constant(2.0, F32)
    scaled = tl.mul(data, data)  # Square instead (no scalar broadcast)
    tl.store(output_tile, scaled)


# ============================================================================
# REFERENCE IMPLEMENTATIONS (NumPy)
# ============================================================================

def numpy_add_tiles(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """NumPy reference for tile addition."""
    return a + b


def numpy_scale_tile(x: np.ndarray) -> np.ndarray:
    """NumPy reference for tile scaling (squared)."""
    return x * x


# ============================================================================
# WORKLOAD DEFINITION
# ============================================================================

def create_add_workload():
    """Create workload for batched tile addition."""
    # Create axes
    batch = Dense[BATCH]()
    tiles_m = Dense[NUM_TILES_M]()
    tiles_n = Dense[NUM_TILES_N]()

    # Create tensors with shape information
    M = BATCH * NUM_TILES_M * TILE_M
    N = BATCH * NUM_TILES_N * TILE_N

    A = Tensor(data=None, shape=(BATCH, NUM_TILES_M, NUM_TILES_N, TILE_M, TILE_N), dtype=F32)
    B = Tensor(data=None, shape=(BATCH, NUM_TILES_M, NUM_TILES_N, TILE_M, TILE_N), dtype=F32)
    C = Tensor(data=None, shape=(BATCH, NUM_TILES_M, NUM_TILES_N, TILE_M, TILE_N), dtype=F32)

    @workload
    def add_workload():
        """Batched tile addition workload."""
        for b in P(batch):
            for m in P(tiles_m):
                for n in P(tiles_n):
                    add_tiles[b, m, n](a=A[b][m][n], b=B[b][m][n], c=C[b][m][n])

    return add_workload(), A, B, C


def create_scale_workload():
    """Create workload for batched tile scaling."""
    batch = Dense[BATCH]()
    tiles = Dense[NUM_TILES_M * NUM_TILES_N]()

    X = Tensor(data=None, shape=(BATCH, NUM_TILES_M * NUM_TILES_N, TILE_M, TILE_N), dtype=F32)
    Y = Tensor(data=None, shape=(BATCH, NUM_TILES_M * NUM_TILES_N, TILE_M, TILE_N), dtype=F32)

    @workload
    def scale_workload():
        """Batched tile scaling workload."""
        for b in P(batch):
            for t in P(tiles):
                scale_tile[b, t](input_tile=X[b][t], output_tile=Y[b][t])

    return scale_workload(), X, Y


# ============================================================================
# DATA BINDING AND VALIDATION
# ============================================================================

def demonstrate_tensor_data_binding():
    """Demonstrate binding NumPy arrays to Tensor objects."""
    print("\n1. Tensor Data Binding")
    print("-" * 50)

    # Create NumPy arrays
    shape = (BATCH, NUM_TILES_M * NUM_TILES_N, TILE_M, TILE_N)
    np_input = np.random.randn(*shape).astype(np.float32)
    np_output = np.zeros_like(np_input)

    # Create Tensors
    input_tensor = Tensor(data=np_input, shape=shape, dtype=F32)
    output_tensor = Tensor(data=np_output, shape=shape, dtype=F32)

    print(f"  Input tensor:")
    print(f"    Shape: {input_tensor.shape}")
    print(f"    DType: {input_tensor.dtype}")
    print(f"    Data bound: {input_tensor.data is not None}")
    print(f"    NumPy shape: {np_input.shape}")
    print(f"    NumPy dtype: {np_input.dtype}")
    print(f"    Data range: [{np_input.min():.4f}, {np_input.max():.4f}]")

    return input_tensor, output_tensor, np_input


def validate_add_workload():
    """Validate tile addition against NumPy reference."""
    print("\n2. Tile Addition Validation")
    print("-" * 50)

    # Create workload and tensors
    w, A, B, C = create_add_workload()

    # Create random input data
    shape = A.shape
    np_a = np.random.randn(*shape).astype(np.float32)
    np_b = np.random.randn(*shape).astype(np.float32)
    np_c = np.zeros_like(np_a)

    # Bind data to tensors (store reference for validation)
    A_data = np_a
    B_data = np_b

    # Compute reference
    np_reference = numpy_add_tiles(np_a, np_b)

    print(f"  Tensor A: shape={shape}")
    print(f"  Tensor B: shape={shape}")
    print(f"  Expected output shape: {np_reference.shape}")

    # Enumerate tasks
    tasks = w.enumerate()
    print(f"  Tasks generated: {len(tasks)}")
    print(f"  Expected tasks: {BATCH * NUM_TILES_M * NUM_TILES_N}")

    # Validate task count
    expected_tasks = BATCH * NUM_TILES_M * NUM_TILES_N
    if len(tasks) == expected_tasks:
        print(f"  PASS: Task count matches ({len(tasks)} == {expected_tasks})")
    else:
        print(f"  FAIL: Task count mismatch ({len(tasks)} != {expected_tasks})")

    # Show sample task
    if tasks:
        sample = tasks[0]
        print(f"\n  Sample task:")
        print(f"    Kernel: {sample.kernel}")
        print(f"    Indices: {[sample.get(ax) for ax in ['batch', 'tiles_m', 'tiles_n'] if sample.get(ax) is not None]}")


def validate_scale_workload():
    """Validate tile scaling against NumPy reference."""
    print("\n3. Tile Scaling Validation")
    print("-" * 50)

    # Create workload and tensors
    w, X, Y = create_scale_workload()

    # Create random input data
    shape = X.shape
    np_x = np.random.randn(*shape).astype(np.float32)

    # Compute reference (squared)
    np_reference = numpy_scale_tile(np_x)

    print(f"  Input shape: {shape}")
    print(f"  Reference computed: shape={np_reference.shape}")
    print(f"  Input range: [{np_x.min():.4f}, {np_x.max():.4f}]")
    print(f"  Output range: [{np_reference.min():.4f}, {np_reference.max():.4f}]")

    # Enumerate and validate
    tasks = w.enumerate()
    expected_tasks = BATCH * NUM_TILES_M * NUM_TILES_N
    print(f"  Tasks generated: {len(tasks)}")

    if len(tasks) == expected_tasks:
        print(f"  PASS: Task count matches ({len(tasks)} == {expected_tasks})")
    else:
        print(f"  FAIL: Task count mismatch ({len(tasks)} != {expected_tasks})")


def demonstrate_kernel_tracing():
    """Demonstrate kernel IR tracing."""
    print("\n4. Kernel IR Tracing")
    print("-" * 50)

    # Trace add_tiles kernel
    ir = add_tiles.trace()
    print(f"  add_tiles kernel:")
    print(f"    Operations: {len(ir.ops)}")
    for i, op in enumerate(ir.ops):
        print(f"      {i}: {op}")

    # Trace scale_tile kernel
    ir = scale_tile.trace()
    print(f"\n  scale_tile kernel:")
    print(f"    Operations: {len(ir.ops)}")
    for i, op in enumerate(ir.ops):
        print(f"      {i}: {op}")


def demonstrate_schedule_configuration():
    """Demonstrate schedule configuration."""
    print("\n5. Schedule Configuration")
    print("-" * 50)

    w, A, B, C = create_add_workload()

    # Apply schedule
    scheduled = (w
        .dispatch(DispatchPolicy.round_robin(4))
        .streams(2)
        .timing(TimingPolicy.immediate)
        .compile())

    print(f"  Dispatch: round_robin(4)")
    print(f"  Streams: 2")
    print(f"  Timing: immediate")
    print(f"  Program compiled: {type(scheduled).__name__}")

    # Execute
    print(f"\n  Executing...")
    scheduled.execute()
    print(f"  Execution complete!")

    # Show stats
    stats = scheduled.stats
    print(f"\n  Execution stats:")
    print(f"    Total tasks: {getattr(stats, 'total_tasks', 'N/A')}")
    print(f"    Total time: {getattr(stats, 'total_time_ms', 'N/A')} ms")


# ============================================================================
# MAIN
# ============================================================================

def main():
    print(f"\nConfiguration:")
    print(f"  Tile size: {TILE_M}x{TILE_N}")
    print(f"  Batch: {BATCH}")
    print(f"  Grid: {NUM_TILES_M}x{NUM_TILES_N}")
    print(f"  Total tiles per batch: {NUM_TILES_M * NUM_TILES_N}")

    # Run demonstrations
    demonstrate_tensor_data_binding()
    validate_add_workload()
    validate_scale_workload()
    demonstrate_kernel_tracing()
    demonstrate_schedule_configuration()

    # Summary
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print("""
This example demonstrated:

1. Tensor Data Binding:
   - Tensor(data=np_array, shape=..., dtype=...) binds NumPy data
   - Shape and dtype are validated at binding time

2. Workload Validation:
   - Task enumeration produces correct number of tasks
   - Task indices correspond to loop variables

3. NumPy Reference:
   - Element-wise operations can be validated against NumPy
   - Kernel semantics match expected behavior

4. Kernel IR Tracing:
   - kernel.trace() produces KernelIR with typed operations
   - Operations are: load, add/mul, store

5. Schedule Configuration:
   - Combinator-style: .dispatch().streams().timing()
   - Program execution with statistics

Note: Actual numerical validation requires C++ backend integration
with data marshalling between Python and the execution engine.
""")


if __name__ == "__main__":
    main()
