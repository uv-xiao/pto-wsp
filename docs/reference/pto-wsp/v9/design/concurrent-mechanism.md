# PTO Workload-Schedule Programming (PTO-WSP) framework v9: Concurrent Mechanism Design

This document describes the concurrent execution mechanisms in PTO-WSP, with detailed analysis of patterns from pto-isa-lh (task graph runtime) and pto-isa-wc (work-stealing runtime).

---

## 1. Overview

PTO-WSP supports two primary concurrent execution models:

| Model | Source | API | Use Case |
|-------|--------|-----|----------|
| **FIFO Task Graph** | pto-isa-lh | `ReadyPolicy.fifo()` | Single-stream NPU, ordered execution |
| **Work-Stealing** | pto-isa-wc | `ReadyPolicy.work_steal()` | Multi-core CPU, multi-stream NPU |

---

## 2. pto-isa-lh Concurrent Features (Task Graph Runtime)

### 2.1 Core Data Structures

From `pto_runtime.h`:

```c
typedef struct {
    PendingTask       pend_task[524288];    // Task table (sliding window)
    TensorMapEntry*   tensor_map[16384];    // Producer lookup (hash map)
    int32_t           ready_queue[262144];  // Tasks with fanin==0
    int32_t           next_task_id;
    pthread_mutex_t   queue_mutex;
    pthread_cond_t    queue_not_empty;
} PTORuntime;
```

### 2.2 TensorMap: Exact-Match Dependency Tracking

**Purpose**: Track which task produces each tensor region for automatic dependency inference.

**Key Structure**:
```c
typedef struct {
    void*    raw_tensor;   // Base pointer
    int64_t  row_offset;   // Region offset
    int64_t  col_offset;
    int64_t  rows;         // Region extent
    int64_t  cols;
} TensorRegion;
```

**Algorithm**:
1. When task writes output: `tensor_map[(hash(tensor, offset, shape)] = task_id`
2. When task reads input: lookup producer in tensor_map, add to fanin
3. Key: `(ptr, row_offset, col_offset, rows, cols)` - **exact match only**

**Limitation**: Exact-match misses overlapping regions with different offsets/shapes.

**v9 Extension**: `Deps.infer_bytes_overlap()` detects overlapping byte ranges.

### 2.3 ReadyQueue: FIFO Task Scheduling

**Purpose**: Hold tasks with zero remaining dependencies.

**Implementation**:
```c
// Single shared queue with mutex
int32_t ready_queue[262144];
int32_t ready_head, ready_tail;
pthread_mutex_t queue_mutex;

// Enqueue when fanin reaches 0
void task_ready(runtime, task_id) {
    pthread_mutex_lock(&queue_mutex);
    ready_queue[ready_tail++] = task_id;
    pthread_cond_signal(&queue_not_empty);
    pthread_mutex_unlock(&queue_mutex);
}

// Dequeue for execution
int32_t get_next_task(runtime) {
    pthread_mutex_lock(&queue_mutex);
    while (ready_head == ready_tail && !shutdown) {
        pthread_cond_wait(&queue_not_empty, &queue_mutex);
    }
    int32_t task_id = ready_queue[ready_head++];
    pthread_mutex_unlock(&queue_mutex);
    return task_id;
}
```

**Characteristics**:
- Strict FIFO ordering (preserves submission order)
- Single lock (potential contention with many workers)
- Simple, predictable behavior

### 2.4 WindowState: Task Window Management

**Purpose**: Limit in-flight task metadata to prevent memory exhaustion.

**Modes**:
| Mode | Behavior |
|------|----------|
| `stall` | Block task submission until window has space |
| `abort` | Fail immediately on overflow |
| `benchmark` | Record overflow events for profiling |

**Implementation** (in `runtime.cpp`):
```cpp
class WindowState {
    int32_t max_tasks;      // Window capacity (e.g., 8192)
    int32_t active_tasks;   // Currently in-flight
    WindowMode overflow_mode;

    bool can_submit() {
        return active_tasks < max_tasks;
    }

    void wait_for_space() {
        while (active_tasks >= max_tasks) {
            if (overflow_mode == ABORT) throw WindowOverflow();
            if (overflow_mode == BENCHMARK) record_overflow();
            yield();
        }
    }
};
```

### 2.5 IssueGate: Pipeline Depth Control

**Purpose**: Implement double/triple buffering via counting semaphore.

**Scopes**:
| Scope | Description |
|-------|-------------|
| `GLOBAL` | Single limit across all streams |
| `PER_STREAM` | Each stream has own limit |
| `PER_POOL` | Separate limits for vector/cube pools |

**Implementation**:
```cpp
class IssueGate {
    int32_t capacity;     // Max in-flight (e.g., 2 for double buffer)
    int32_t current;      // Currently in-flight
    std::mutex mutex;
    std::condition_variable cv;

    void acquire() {
        std::unique_lock lock(mutex);
        cv.wait(lock, [this] { return current < capacity; });
        current++;
    }

    void release() {
        std::unique_lock lock(mutex);
        current--;
        cv.notify_one();
    }
};
```

### 2.6 DepBatcher: Batched Dependency Resolution

**Purpose**: Reduce lock contention by batching dependency updates.

**Implementation**:
```cpp
class DepBatcher {
    std::vector<int32_t> pending_updates;
    int32_t threshold;    // Batch size before flush

    void add_completion(task_id) {
        pending_updates.push_back(task_id);
        if (pending_updates.size() >= threshold) {
            flush();
        }
    }

    void flush() {
        std::lock_guard lock(dep_mutex);
        for (auto task_id : pending_updates) {
            for (auto downstream : fanout[task_id]) {
                if (--fanin[downstream] == 0) {
                    ready_queue.push(downstream);
                }
            }
        }
        pending_updates.clear();
    }
};
```

### 2.7 Target Applicability

| Feature | CPU Simulation | Single-Stream NPU | Multi-Stream NPU |
|---------|---------------|-------------------|------------------|
| TensorMap | ✓ | ✓ | ✓ |
| FIFO Queue | ✓ | ✓ | Per-stream |
| WindowState | ✓ | ✓ | ✓ |
| IssueGate | ✓ | ✓ (double buffer) | Per-stream |
| DepBatcher | ✓ | ✓ | ✓ |

---

## 3. pto-isa-wc Concurrent Features (Work-Stealing Runtime)

### 3.1 WorkStealingDeque: Per-Worker Queue

**Purpose**: Lock-free task queue optimized for local LIFO access.

**Algorithm** (Chase-Lev deque):
- Owner thread: push/pop from bottom (LIFO - cache locality)
- Thieves: steal from top (FIFO - oldest tasks)

**Implementation** (in `ready_queue.hpp`):
```cpp
template<typename T>
class WorkStealingDeque {
    std::vector<T> buffer;
    std::atomic<int64_t> top;     // Steal from here
    std::atomic<int64_t> bottom;  // Owner pushes/pops here

    void push(T item) {
        int64_t b = bottom.load(relaxed);
        buffer[b] = item;
        std::atomic_thread_fence(release);
        bottom.store(b + 1, relaxed);
    }

    std::optional<T> pop() {
        int64_t b = bottom.load(relaxed) - 1;
        bottom.store(b, relaxed);
        std::atomic_thread_fence(seq_cst);
        int64_t t = top.load(relaxed);

        if (t <= b) {
            T item = buffer[b];
            if (t == b) {
                // Last item - CAS to avoid race with stealer
                if (!top.compare_exchange_strong(t, t + 1)) {
                    bottom.store(b + 1, relaxed);
                    return std::nullopt;
                }
                bottom.store(b + 1, relaxed);
            }
            return item;
        }
        bottom.store(b + 1, relaxed);
        return std::nullopt;
    }

    std::optional<T> steal() {
        int64_t t = top.load(acquire);
        std::atomic_thread_fence(seq_cst);
        int64_t b = bottom.load(acquire);

        if (t < b) {
            T item = buffer[t];
            if (top.compare_exchange_strong(t, t + 1)) {
                return item;
            }
        }
        return std::nullopt;
    }
};
```

### 3.2 WorkStealingQueueSet: Multi-Queue Distribution

**Purpose**: Coordinate multiple work-stealing deques.

```cpp
class WorkStealingQueueSet {
    std::vector<WorkStealingDeque<Task>> queues;  // One per worker
    size_t num_workers;

    void push(Task task, size_t worker_id) {
        queues[worker_id].push(task);
    }

    std::optional<Task> pop(size_t worker_id) {
        // Try local queue first
        if (auto task = queues[worker_id].pop()) {
            return task;
        }
        // Try stealing from random other worker
        for (int attempts = 0; attempts < num_workers * 2; attempts++) {
            size_t victim = random() % num_workers;
            if (victim != worker_id) {
                if (auto task = queues[victim].steal()) {
                    return task;
                }
            }
        }
        return std::nullopt;
    }
};
```

### 3.3 Affinity-Based Task Placement

**Purpose**: Improve cache locality by placing related tasks on same worker.

```cpp
size_t compute_affinity(Task& task) {
    // Hash task properties to worker ID
    size_t key = 0;
    if (auto batch = task.get("batch")) {
        key = hash_combine(key, batch);
    }
    return key % num_workers;
}

void submit_with_affinity(Task task) {
    size_t worker = compute_affinity(task);
    queues[worker].push(task);
}
```

### 3.4 Target Applicability

| Feature | Multi-Core CPU | Multi-Stream NPU | Distributed |
|---------|---------------|------------------|-------------|
| Work-Stealing Deque | ✓ | Per-AICPU | N/A |
| Affinity Placement | ✓ | Per-Stream | N/A |
| Multi-Queue | ✓ | ✓ | N/A |

---

## 4. Multi-Domain Execution (CPU ↔ Accelerator)

### 4.1 DomainHandshake Protocol

**Purpose**: Coordinate execution across CPU and accelerator domains.

From pto-isa-wc `handshake.h`:
```c
struct Handshake {
    volatile uint32_t aicpu_ready;   // AICPU signals scheduler ready
    volatile uint32_t aicore_done;   // AICore signals task complete
    volatile uint64_t task;          // Task pointer (AICPU → AICore)
    volatile int32_t  task_status;   // 1 = assigned, 0 = complete
    volatile int32_t  control;       // Shutdown signal
};
```

**Protocol**:
1. AICPU sets `aicpu_ready = 1` when scheduler initialized
2. AICPU polls ready queue, writes task to `task`, sets `task_status = 1`
3. AICore reads task, executes kernel, sets `task_status = 0`
4. AICPU updates dependencies, finds next ready task
5. On shutdown: AICPU sets `control = 1`, AICore exits

### 4.2 PTO-WSP Implementation

```cpp
// In include/pto/wsp/concurrent/utilities.hpp
class DomainHandshake {
    std::atomic<bool> cpu_ready_{false};
    std::atomic<bool> accelerator_ready_{false};
    std::atomic<bool> transfer_complete_{false};
    std::mutex mutex_;
    std::condition_variable cv_;

    void cpu_ready() {
        std::lock_guard lock(mutex_);
        cpu_ready_ = true;
        cv_.notify_all();
    }

    void wait_accelerator_ready() {
        std::unique_lock lock(mutex_);
        cv_.wait(lock, [this] { return accelerator_ready_; });
    }

    void reset() {
        cpu_ready_ = false;
        accelerator_ready_ = false;
        transfer_complete_ = false;
    }
};
```

### 4.3 Cross-Domain Dependency Tracking

For heterogeneous execution:
1. CPU tasks track dependencies normally
2. Before accelerator launch: synchronize all CPU dependencies
3. Accelerator executes (may have internal parallelism)
4. After accelerator complete: update CPU-side fanout

---

## 5. Comparison Matrix

| Feature | FIFO (lh-style) | Work-Stealing (wc-style) |
|---------|-----------------|--------------------------|
| **Queue Structure** | Single shared queue | Per-worker deques |
| **Task Ordering** | Strict FIFO | Local LIFO, steal FIFO |
| **Lock Contention** | Higher (single lock) | Lower (per-queue locks) |
| **Cache Locality** | Lower | Higher (local tasks hot) |
| **Load Balancing** | Automatic | Requires stealing |
| **Predictability** | Higher | Lower (stealing is random) |
| **Implementation Complexity** | Simple | Complex |
| **Best For** | Single-stream, ordered | Multi-core, parallel |

---

## 6. When to Use Which Approach

### 6.1 Use FIFO (`ReadyPolicy.fifo()`)

- Single-stream NPU execution
- Ordered task completion required
- Simple debugging/profiling
- Low task parallelism (< 4 concurrent tasks)

### 6.2 Use Work-Stealing (`ReadyPolicy.work_steal()`)

- Multi-core CPU with 4+ cores
- Multi-stream NPU execution
- High task parallelism (many independent tasks)
- Variable task durations (load imbalance)

---

## 7. Integration with PTO-WSP API

### 7.1 Python API

```python
from pto_wsp import workload, P, Deps, ReadyPolicy, StartPolicy, TracePolicy

# FIFO execution (pto-isa-lh compatible)
program = (workload()
    .task_graph(
        deps=Deps.infer_tensor_map_exact(),
        ready=ReadyPolicy.fifo(),
        start=StartPolicy.after_orchestration(),
    )
    .compile())

# Work-stealing execution
program = (workload()
    .task_graph(
        deps=Deps.infer_tensor_map_exact(),
        ready=ReadyPolicy.work_steal(),
        start=StartPolicy.immediate(),
    )
    .compile())
```

### 7.2 C++ Implementation Mapping

| Python API | C++ Implementation |
|------------|-------------------|
| `ReadyPolicy.fifo()` | `FIFOReadyQueue` in `ready_queue.hpp` |
| `ReadyPolicy.work_steal()` | `WorkStealingQueueSet` in `ready_queue.hpp` |
| `StartPolicy.after_orchestration()` | Wait for all tasks submitted |
| `StartPolicy.immediate()` | Execute as tasks become ready |
| `StartPolicy.threshold(n)` | Wait for n tasks, then start |
| `TracePolicy.cycles(fn)` | Record task start/end timestamps |

---

## 8. Target-Specific Recommendations

| Target | Recommended Ready Policy | Notes |
|--------|-------------------------|-------|
| **CPU Simulation** | Work-steal (4+ cores) | Maximize core utilization |
| **Ascend NPU (single stream)** | FIFO | Matches device ordering |
| **Ascend NPU (multi-stream)** | FIFO per stream | Each stream ordered |
| **AMD AIE** | Work-steal (many tiles) | Spatial parallelism |

---

## References

- pto-isa-lh: `docs/reference/13_pto_isa_lh.md`
- pto-isa-wc: `docs/reference/14_pto_isa_wc.md`
- C++ Utilities: `include/pto/wsp/concurrent/utilities.hpp`
- Ready Queue: `include/pto/wsp/graph/ready_queue.hpp`
- Runtime: `include/pto/wsp/graph/runtime.hpp`
