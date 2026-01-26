#!/usr/bin/env python3
"""
PTO-RT v9 End-to-End Example: CSP Pipeline

This example demonstrates CSP (Communicating Sequential Processes) primitives
for pipeline-parallel execution:
- Channel: Typed, bounded communication channel
- process: Named process with input/output channels
- send/consume: Channel operations
- connect: Pipeline composition

Architecture:
- Producer-consumer pattern with explicit channels
- Three-stage pipeline: Load -> Compute -> Store
- Demonstrates data flow between processes

Usage:
    python examples/csp_pipeline/csp_pipeline_example.py
"""

import sys
sys.path.insert(0, 'python')

from pto_wsp import (
    kernel, tl,
    In, Out, Tile,
    workload, P,
    Dense, DenseDyn, Tensor, DType,
    DispatchPolicy, TimingPolicy,
    task, combine, for_each, parallel_for,
)

from pto_wsp.csp import (
    Channel,
    process,
    send,
    consume,
    connect,
    Event,
    record,
    synchronize,
    query,
)

# ============================================================================
# CONSTANTS
# ============================================================================

TILE_SIZE = 128
NUM_TILES = 8
F32 = DType.F32
F16 = DType.F16

print("=" * 70)
print("CSP Pipeline Example - PTO-RT v9")
print("=" * 70)

# ============================================================================
# KERNELS (with tl.* primitives)
# ============================================================================

@kernel
def load_kernel(
    src: In[Tile[TILE_SIZE, TILE_SIZE, F16]],
    dst: Out[Tile[TILE_SIZE, TILE_SIZE, F16]],
):
    """Load data from source to destination."""
    # Load source tile
    data = tl.load(src)
    # Store to destination
    tl.store(dst, data)


@kernel
def compute_kernel(
    input_tile: In[Tile[TILE_SIZE, TILE_SIZE, F16]],
    output_tile: Out[Tile[TILE_SIZE, TILE_SIZE, F16]],
):
    """Compute on tile (example: element-wise square)."""
    data = tl.load(input_tile)
    squared = tl.mul(data, data)
    tl.store(output_tile, squared)


@kernel
def store_kernel(
    src: In[Tile[TILE_SIZE, TILE_SIZE, F16]],
    dst: Out[Tile[TILE_SIZE, TILE_SIZE, F16]],
):
    """Store data from source to destination."""
    data = tl.load(src)
    tl.store(dst, data)


# ============================================================================
# CSP PIPELINE DEFINITION
# ============================================================================

def create_csp_pipeline(num_tiles: int):
    """Create a three-stage CSP pipeline.

    Pipeline stages:
    1. Loader: Reads data and sends to compute channel
    2. Computer: Receives data, processes, sends to store channel
    3. Storer: Receives processed data and writes output

    Args:
        num_tiles: Number of tiles to process

    Returns:
        Pipeline workload ready for execution
    """
    # Channels for inter-process communication
    load_to_compute = Channel[Tensor](name="load_to_compute", depth=2)
    compute_to_store = Channel[Tensor](name="compute_to_store", depth=2)

    # Axis for tile iteration
    tiles = Dense[num_tiles]()

    # Define processes using process() builder
    # Process bodies use declarative workload primitives

    # Process 1: Loader - produces tiles to load_to_compute channel
    loader = (process("loader")
        .produces(load_to_compute)
        .body(for_each(tiles, lambda t:
            send(load_to_compute, task("load_kernel", [t], ["input_data", "temp_f16"])))))

    # Process 2: Computer - consumes from load channel, produces to store channel
    computer = (process("computer")
        .consumes(load_to_compute)
        .produces(compute_to_store)
        .body(consume(load_to_compute, lambda tile_data:
            send(compute_to_store, task("compute_kernel", [tile_data], ["temp_f16"])))))

    # Process 3: Storer - consumes from store channel, writes output
    storer = (process("storer")
        .consumes(compute_to_store)
        .body(consume(compute_to_store, lambda tile_data:
            task("store_kernel", [tile_data], ["temp_f16", "output_data"]))))

    # Connect processes into pipeline
    pipeline = connect(
        [loader, computer, storer],
        [load_to_compute, compute_to_store]
    )

    print(f"\nPipeline created:")
    print(f"  Channels: {load_to_compute.name} (depth={load_to_compute.depth})")
    print(f"            {compute_to_store.name} (depth={compute_to_store.depth})")
    print(f"  Processes: {loader.name} -> {computer.name} -> {storer.name}")

    return pipeline


def create_simple_csp_workload(num_tiles: int):
    """Create a simpler CSP example using @workload with P.pipe().

    This demonstrates CSP primitives integrated with the @workload decorator.
    """
    # Tensors
    input_data = Tensor(data=None, shape=(num_tiles, TILE_SIZE, TILE_SIZE), dtype=F16)
    output_data = Tensor(data=None, shape=(num_tiles, TILE_SIZE, TILE_SIZE), dtype=F16)
    temp = Tensor(data=None, shape=(num_tiles, TILE_SIZE, TILE_SIZE), dtype=F16)

    tiles = Dense[num_tiles]()

    @workload
    def csp_workload():
        """CSP workload with pipeline context."""
        # Pipeline context allows channel operations
        with P.pipe():
            # Create channel for producer-consumer
            ch = Channel[Tensor](name="data_channel", depth=2)

            # Producer: Load tiles - using task() within P() loop
            for t in P(tiles):
                send(ch, task("load_kernel", [t], [input_data[t], temp[t]]))

            # Consumer: Process tiles (would receive from channel)
            # Note: In actual execution, consume() blocks until data available
            consume(ch, lambda data:
                task("compute_kernel", [data], [temp]))

    return csp_workload()


# ============================================================================
# ALTERNATIVE: Data-Parallel Approach (for comparison)
# ============================================================================

def create_data_parallel_workload(num_tiles: int):
    """Create equivalent workload using data-parallel pattern.

    This shows the same computation without CSP, for comparison.
    """
    input_data = Tensor(data=None, shape=(num_tiles, TILE_SIZE, TILE_SIZE), dtype=F16)
    output_data = Tensor(data=None, shape=(num_tiles, TILE_SIZE, TILE_SIZE), dtype=F16)
    temp = Tensor(data=None, shape=(num_tiles, TILE_SIZE, TILE_SIZE), dtype=F16)

    tiles = Dense[num_tiles]()

    @workload
    def parallel_workload():
        """Data-parallel version - all tiles processed independently."""
        for t in P(tiles):
            # Load
            load_kernel[t](src=input_data[t], dst=temp[t])
            # Compute
            compute_kernel[t](input_tile=temp[t], output_tile=temp[t])
            # Store
            store_kernel[t](src=temp[t], dst=output_data[t])

    return parallel_workload()


# ============================================================================
# EVENT-BASED SYNCHRONIZATION DEMO
# ============================================================================

def demo_event_synchronization():
    """Demonstrate event-based synchronization."""
    print("\nEvent Synchronization Demo:")

    # Create events for coordination
    load_done = Event(name="load_done")
    compute_done = Event(name="compute_done")

    print(f"  Created events: {load_done.name}, {compute_done.name}")

    # Simulate recording events
    record(load_done)
    print(f"  Recorded: {load_done.name}")

    # Query event status (would return True if event occurred)
    status = query(load_done)
    print(f"  Query {load_done.name}: {status}")

    # Synchronize (would block until event)
    synchronize(load_done)
    print(f"  Synchronized on: {load_done.name}")


# ============================================================================
# CSP PRIMITIVES DEMONSTRATION
# ============================================================================

def demo_csp_primitives():
    """Demonstrate CSP primitives without full pipeline."""
    print("\nCSP Primitives Demo:")

    # 1. Channel creation
    ch = Channel[int](name="demo_channel", depth=3)
    print(f"  Created channel: {ch.name} (depth={ch.depth})")
    print(f"    empty: {ch.empty()}, full: {ch.full()}, size: {ch.size()}")

    # 2. Process builder pattern
    tiles = Dense[4]()

    producer = (process("producer")
        .produces(ch)
        .body(for_each(tiles, lambda i:
            send(ch, task("produce", [i], [])))))

    consumer = (process("consumer")
        .consumes(ch)
        .body(consume(ch, lambda data:
            task("consume", [data], []))))

    print(f"  Producer: {producer.name}")
    print(f"    produces: {[c.name for c in producer.produces]}")
    print(f"  Consumer: {consumer.name}")
    print(f"    consumes: {[c.name for c in consumer.consumes]}")

    # 3. Pipeline connection
    pipeline = connect([producer, consumer], [ch])
    print(f"  Pipeline: {type(pipeline).__name__}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    print(f"\nConfiguration:")
    print(f"  Tile size: {TILE_SIZE}x{TILE_SIZE}")
    print(f"  Number of tiles: {NUM_TILES}")

    # Kernel info
    print(f"\nKernels (@kernel with tl.* primitives):")
    kernels = [
        ("load_kernel", load_kernel, "tl.load, tl.store"),
        ("compute_kernel", compute_kernel, "tl.load, tl.mul, tl.store"),
        ("store_kernel", store_kernel, "tl.load, tl.store"),
    ]
    for name, k, ops in kernels:
        print(f"  {name}: {ops}")

    # Demo CSP primitives
    print("\n" + "=" * 70)
    print("1. CSP Primitives Demo")
    print("=" * 70)
    demo_csp_primitives()

    # Create CSP pipeline
    print("\n" + "=" * 70)
    print("2. CSP Pipeline Pattern")
    print("=" * 70)
    pipeline = create_csp_pipeline(NUM_TILES)
    print(f"Pipeline structure: {type(pipeline).__name__}")

    # Trace kernels to show IR
    print(f"\nKernel IR (traced from tl.* primitives):")
    ir = load_kernel.trace()
    print(f"  load_kernel: {len(ir.ops)} operations")
    for op in ir.ops[:3]:
        print(f"    {op}")

    # Create simple CSP workload
    print("\n" + "=" * 70)
    print("3. Simple CSP Workload (with P.pipe())")
    print("=" * 70)
    try:
        w = create_simple_csp_workload(NUM_TILES)
        print(f"Workload created: {type(w).__name__}")
    except Exception as e:
        print(f"Note: P.pipe() context not fully implemented: {e}")

    # Create data-parallel equivalent
    print("\n" + "=" * 70)
    print("4. Data-Parallel Equivalent (for comparison)")
    print("=" * 70)
    dp_workload = create_data_parallel_workload(NUM_TILES)

    # Apply schedule
    scheduled = (dp_workload
        .dispatch(DispatchPolicy.round_robin(4))
        .streams(2)
        .timing(TimingPolicy.immediate)
        .compile())

    print(f"Program compiled: {type(scheduled).__name__}")

    # Enumerate tasks
    tasks = dp_workload.enumerate()
    print(f"Tasks enumerated: {len(tasks)}")

    # Show task breakdown
    kernels_used = {}
    for t in tasks:
        k = t.kernel
        kernels_used[k] = kernels_used.get(k, 0) + 1
    print(f"Task breakdown:")
    for k, count in kernels_used.items():
        print(f"  {k}: {count} tasks")

    # Event synchronization demo
    print("\n" + "=" * 70)
    print("5. Event Synchronization")
    print("=" * 70)
    demo_event_synchronization()

    # Execute
    print("\n" + "=" * 70)
    print("6. Execution")
    print("=" * 70)
    print("Executing data-parallel workload...")
    scheduled.execute()
    print("Execution complete!")

    # Summary
    print(f"\n" + "=" * 70)
    print("CSP Pipeline Example Summary")
    print("=" * 70)
    print("""
Key v9 CSP patterns demonstrated:

1. Channel[T]: Typed, bounded communication channel
   - Channel[Tensor](name="ch", depth=2)
   - Provides producer-consumer synchronization

2. process(): Process builder with fluent API
   - .consumes(channel) - Input channel
   - .produces(channel) - Output channel
   - .body(workload) - Process body (declarative)

3. send(channel, value): Send data to channel
   - send(ch, task("kernel", [i], resources))

4. consume(channel, handler): Receive and process data
   - consume(ch, lambda data: task("process", [data], []))

5. connect([processes], [channels]): Compose pipeline
   - Connects processes via channels

6. Event-based synchronization:
   - Event(name) - Named synchronization point
   - record(event) - Mark event occurred
   - query(event) - Check event status
   - synchronize(event) - Wait for event

7. P.pipe() context: Pipeline-aware workload
   - Enables channel operations within @workload

CSP vs Data-Parallel:
- CSP: Better for streaming/pipelined computation
- Data-Parallel: Better for independent batch processing
- v9 supports both patterns with consistent API
""")


if __name__ == "__main__":
    main()
