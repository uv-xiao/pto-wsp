# Tensor Data Example

This example demonstrates binding actual NumPy arrays to Tensor objects
and validating workload output against reference implementations.

## Overview

The example shows:
1. **Data Binding**: Creating Tensors with NumPy arrays
2. **Validation**: Comparing against NumPy reference implementations
3. **Tracing**: Kernel IR generation and inspection
4. **Execution**: Schedule configuration and program execution

## Running the Example

```bash
# From repository root
make -C examples/tensor_data run

# Or directly
python examples/tensor_data/tensor_data_example.py
```

## Features Demonstrated

### 1. Tensor Data Binding
```python
import numpy as np
from pto_wsp import Tensor, DType

# Create NumPy array
np_data = np.random.randn(4, 16, 32, 32).astype(np.float32)

# Bind to Tensor
tensor = Tensor(data=np_data, shape=(4, 16, 32, 32), dtype=DType.F32)
```

### 2. Workload Validation
```python
# Create workload
w, A, B, C = create_add_workload()

# Enumerate tasks
tasks = w.enumerate()

# Validate task count
assert len(tasks) == expected_tasks
```

### 3. Kernel Tracing
```python
@kernel
def add_tiles(a: In[Tile], b: In[Tile], c: Out[Tile]):
    x = tl.load(a)
    y = tl.load(b)
    z = tl.add(x, y)
    tl.store(c, z)

# Trace to IR
ir = add_tiles.trace()
print(ir.ops)  # [load, load, add, store]
```

### 4. Reference Validation
```python
# NumPy reference
np_reference = np_a + np_b

# Kernel semantics should match
# (Actual numerical validation requires C++ backend)
```

## Expected Output

```
======================================================================
Tensor Data Binding Example - PTO-RT v9
======================================================================

Configuration:
  Tile size: 32x32
  Batch: 4
  Grid: 4x4
  Total tiles per batch: 16

1. Tensor Data Binding
--------------------------------------------------
  Input tensor:
    Shape: (4, 16, 32, 32)
    DType: F32
    Data bound: True
    ...

2. Tile Addition Validation
--------------------------------------------------
  Tasks generated: 64
  Expected tasks: 64
  PASS: Task count matches (64 == 64)
  ...

3. Tile Scaling Validation
--------------------------------------------------
  PASS: Task count matches
  ...

4. Kernel IR Tracing
--------------------------------------------------
  add_tiles kernel:
    Operations: 4
      0: v3 = Load(v1)
      1: v4 = Load(v2)
      2: v5 = Add(v3, v4)
      3: Store(v5, v5)
  ...
```

## Pass/Fail Criteria

The example passes if:
1. Tensor data binding succeeds
2. Task enumeration produces correct count
3. Kernel tracing produces expected operations
4. Program execution completes without errors

## Files

- `tensor_data_example.py` - Main example code
- `Makefile` - Build and run targets
- `README.md` - This file

## Notes

- Actual numerical validation requires C++ backend with data marshalling
- Current validation checks task count and kernel structure
- Future: Add C++ backend execution with tensor data transfer
