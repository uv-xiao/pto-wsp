# CSP Pipeline Example

This example demonstrates CSP (Communicating Sequential Processes) primitives for pipeline-parallel execution in PTO-RT v9.

## Overview

The example implements a three-stage data processing pipeline:

```
┌──────────┐    Channel    ┌──────────┐    Channel    ┌──────────┐
│  Loader  │──────────────>│ Computer │──────────────>│  Storer  │
└──────────┘ load_to_compute └──────────┘ compute_to_store └──────────┘
     │                           │                           │
   F32→F16                   square                      F16→F32
```

## CSP Primitives Demonstrated

### 1. Channel[T] - Typed Communication
```python
ch = Channel[Tensor](name="data_channel", depth=2)
```
- Bounded, typed channel for inter-process communication
- `depth` controls buffer size (backpressure when full)

### 2. Process - Pipeline Stage
```python
loader = (Process("loader")
    .produces(channel)
    .body_func(lambda: [...]))
```
- Named process with explicit input/output channels
- `.consumes()` - declares input channel
- `.produces()` - declares output channel

### 3. send/consume - Channel Operations
```python
send(channel, kernel(...))      # Send data to channel
consume(channel, lambda data:   # Receive and process
    compute(data))
```

### 4. connect - Pipeline Composition
```python
pipeline = connect(loader, computer, storer)
```
- Composes processes into a pipeline
- Verifies channel connections

### 5. Event Synchronization
```python
event = Event(name="phase_done")
record(event)           # Mark event occurred
query(event)            # Check if occurred
synchronize(event)      # Wait for event
```

## Running the Example

```bash
# From repository root
make -C examples/csp_pipeline run

# Or directly
python examples/csp_pipeline/csp_pipeline_example.py
```

## Expected Output

```
======================================================================
CSP Pipeline Example - PTO-RT v9
======================================================================

Configuration:
  Tile size: 128x128
  Number of tiles: 8

Kernels (@kernel with tl.* primitives):
  load_kernel: tl.load, tl.cast, tl.store
  compute_kernel: tl.load, tl.mul, tl.store
  store_kernel: tl.load, tl.cast, tl.store

======================================================================
1. CSP Pipeline Pattern
======================================================================

Pipeline created:
  Channels: load_to_compute (depth=2)
            compute_to_store (depth=2)
  Processes: loader -> computer -> storer
Pipeline structure: Pipeline

Kernel IR (traced from tl.* primitives):
  load_kernel: 3 operations
    load src
    cast F16
    store dst

======================================================================
2. Simple CSP Workload (with P.pipe())
======================================================================
Workload created: Workload

======================================================================
3. Data-Parallel Equivalent (for comparison)
======================================================================
Program compiled: Program
Tasks enumerated: 24
Task breakdown:
  load_kernel: 8 tasks
  compute_kernel: 8 tasks
  store_kernel: 8 tasks

======================================================================
4. Event Synchronization
======================================================================

Event Synchronization Demo:
  Created events: load_done, compute_done
  Recorded: load_done
  Query load_done: True
  Synchronized on: load_done

======================================================================
5. Execution
======================================================================
Executing data-parallel workload...
Execution complete!

======================================================================
CSP Pipeline Example Summary
======================================================================

Key v9 CSP patterns demonstrated:
...
```

## CSP vs Data-Parallel

| Aspect | CSP Pipeline | Data-Parallel |
|--------|--------------|---------------|
| Pattern | Producer-consumer with explicit channels | Independent parallel tasks |
| Best for | Streaming, pipelined computation | Batch processing |
| Synchronization | Channel-based (bounded buffers) | Barrier-based or independent |
| Memory | Can overlap stages (double buffering) | All data resident |

## Files

- `csp_pipeline_example.py` - Main example code
- `Makefile` - Build and run targets
- `README.md` - This file

## Related Examples

- `examples/data_parallel/` - Data-parallel patterns
- `examples/moe/` - Mixture of Experts with select()
- `examples/attention/` - Attention workload patterns
