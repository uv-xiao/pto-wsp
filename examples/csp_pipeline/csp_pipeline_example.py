#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent / "_harness"))
sys.path.insert(0, str(_HERE))

from harness import CycleCheck, run_example  # noqa: E402
from golden import square_pipeline_ref  # noqa: E402
from pto_wsp_impl import run_csp_square_pipeline  # noqa: E402


def main() -> bool:
    tiles, h, w = 8, 64, 64
    seed = 0

    rng = np.random.default_rng(seed)
    x = rng.standard_normal((tiles, h, w), dtype=np.float32)

    try:
        run_example(
            "csp_pipeline",
            run_pto=lambda: run_csp_square_pipeline(x),
            run_golden=lambda: square_pipeline_ref(x),
            rtol=1e-6,
            atol=1e-6,
            cycles=CycleCheck(expected=114688, rel_tol=0.20, min_cycles=1),
        )
        return True
    except Exception as e:  # noqa: BLE001
        print(f"csp_pipeline: FAIL ({e})")
        return False


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
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
    print(f"\nKernels (@kernel with pto.* primitives):")
    kernels = [
        ("load_kernel", load_kernel, "pto.load, pto.store"),
        ("compute_kernel", compute_kernel, "pto.load, pto.mul, pto.store"),
        ("store_kernel", store_kernel, "pto.load, pto.store"),
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
    print(f"\nKernel IR (traced from pto.* primitives):")
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
    np.random.seed(0)
    input_np = np.random.randn(NUM_TILES, TILE_SIZE, TILE_SIZE).astype(np.float16)
    output_np = np.zeros_like(input_np)
    temp_np = np.zeros_like(input_np)
    ref_np = (input_np.astype(np.float32) * input_np.astype(np.float32)).astype(np.float16)

    input_data = Tensor(data=input_np, shape=input_np.shape, dtype=F16)
    output_data = Tensor(data=output_np, shape=output_np.shape, dtype=F16)
    temp = Tensor(data=temp_np, shape=temp_np.shape, dtype=F16)

    dp_workload = create_data_parallel_workload(input_data, output_data, temp, NUM_TILES)

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
    scheduled.synchronize()
    print("Execution complete!")

    stats = scheduled.stats() if callable(scheduled.stats) else scheduled.stats
    print(f"Total cycles: {getattr(stats, 'total_cycles', 'N/A')}")

    max_err = float(np.max(np.abs(output_np.astype(np.float32) - ref_np.astype(np.float32))))
    print(f"Max error:  {max_err:.3e}")
    if max_err < 1e-2:
        print("Status: PASS")
    else:
        print("Status: FAIL")

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
