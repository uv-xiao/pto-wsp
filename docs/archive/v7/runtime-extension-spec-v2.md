# PTO Workload-Schedule Programming (PTO-WSP): API Specification (v7.1 - TVM Series Enhanced)

## 1. Overview

This specification extends v7 with insights from TVM, SparseTIR, and Relax:
- **Axis-based iteration** for sparse/dynamic task spaces
- **Symbolic shapes** for tracking dynamic dimensions
- **Task fusion** based on operator categories
- **Three-stage compilation** pipeline
- **Cross-level** workload-kernel abstraction

## 2. Symbolic Shape System (From Relax)

### 2.1 Symbolic Expressions

```cpp
namespace pto::wsp::sym {

// Symbolic variable
class SymVar {
public:
    static SymVar create(const char* name);
    const char* name() const;
    int64_t value() const;  // Runtime: concrete value
    bool is_bound() const;
};

// Symbolic expression
class SymExpr {
public:
    // Constructors
    static SymExpr var(const char* name);
    static SymExpr constant(int64_t value);

    // Arithmetic
    SymExpr operator+(SymExpr rhs) const;
    SymExpr operator-(SymExpr rhs) const;
    SymExpr operator*(SymExpr rhs) const;
    SymExpr operator/(SymExpr rhs) const;
    SymExpr operator%(SymExpr rhs) const;

    // Comparison (returns SymBool)
    SymBool operator<(SymExpr rhs) const;
    SymBool operator<=(SymExpr rhs) const;
    SymBool operator>(SymExpr rhs) const;
    SymBool operator>=(SymExpr rhs) const;
    SymBool operator==(SymExpr rhs) const;
    SymBool operator!=(SymExpr rhs) const;

    // Functions
    static SymExpr min(SymExpr a, SymExpr b);
    static SymExpr max(SymExpr a, SymExpr b);
    static SymExpr ceildiv(SymExpr a, SymExpr b);
    static SymExpr floordiv(SymExpr a, SymExpr b);

    // Query
    bool is_constant() const;
    int64_t as_constant() const;  // Throws if not constant
    std::vector<SymVar> free_vars() const;

    // Simplify
    SymExpr simplify() const;

    // Substitute
    SymExpr substitute(const std::map<SymVar, SymExpr>& subs) const;
};

// Symbolic boolean
class SymBool {
public:
    static SymBool constant(bool value);
    SymBool operator&&(SymBool rhs) const;
    SymBool operator||(SymBool rhs) const;
    SymBool operator!() const;

    bool can_prove_true() const;
    bool can_prove_false() const;
};

}  // namespace pto::wsp::sym
```

### 2.2 Symbolic Tensor

```cpp
namespace pto::wsp {

// Tensor with symbolic shape
class SymTensor {
public:
    // Constructors
    SymTensor(std::vector<SymExpr> shape, DType dtype);
    SymTensor(std::initializer_list<SymExpr> shape, DType dtype);

    // Accessors
    const std::vector<SymExpr>& shape() const;
    SymExpr shape(int dim) const;
    int ndim() const;
    DType dtype() const;

    // Derived properties
    SymExpr numel() const;  // Product of shape dimensions
    SymExpr size_bytes() const;

    // Shape operations
    SymTensor reshape(std::vector<SymExpr> new_shape) const;
    SymTensor flatten() const;
    SymTensor squeeze(int dim) const;
    SymTensor unsqueeze(int dim) const;

    // Indexing (returns SymTensor with reduced dims)
    SymTensor operator[](SymExpr index) const;
    SymTensor slice(int dim, SymExpr start, SymExpr end) const;
};

// match_cast: Assert shape with new symbolic variables
SymTensor match_cast(SymTensor tensor, std::vector<SymExpr> expected_shape);

}  // namespace pto::wsp
```

### 2.3 Shape Deduction

```cpp
namespace pto::wsp {

// Shape deduction rule
using ShapeDeduceFunc = std::function<
    std::vector<SymExpr>(
        const std::vector<SymTensor>& inputs,
        const std::map<std::string, SymExpr>& attrs
    )
>;

// Register deduction rules
class ShapeDeducer {
public:
    static ShapeDeducer& instance();

    void register_rule(const char* op_name, ShapeDeduceFunc rule);

    std::vector<SymExpr> deduce(
        const char* op_name,
        const std::vector<SymTensor>& inputs,
        const std::map<std::string, SymExpr>& attrs
    );
};

// Built-in rules
namespace shape_rules {
    ShapeDeduceFunc elementwise();  // Same as first input
    ShapeDeduceFunc broadcast();    // Broadcast rules
    ShapeDeduceFunc matmul();       // (M,K) @ (K,N) -> (M,N)
    ShapeDeduceFunc concat(int axis);
    ShapeDeduceFunc split(int axis, int num_splits);
    ShapeDeduceFunc reshape(std::vector<SymExpr> target);
}

}  // namespace pto::wsp
```

## 3. Axis System (From SparseTIR)

### 3.1 Axis Types

```cpp
namespace pto::wsp::axis {

// Base axis interface
class Axis {
public:
    virtual ~Axis() = default;

    // Properties
    virtual bool is_dense() const = 0;
    virtual bool is_fixed() const = 0;
    virtual SymExpr extent() const = 0;  // Total iteration count

    // Iteration
    virtual AxisIterator begin() const = 0;
    virtual AxisIterator end() const = 0;
};

// Dense-Fixed: known size at compile time (batch, heads)
class DenseFixed : public Axis {
public:
    explicit DenseFixed(SymExpr size);

    bool is_dense() const override { return true; }
    bool is_fixed() const override { return true; }
    SymExpr extent() const override { return size_; }

    SymExpr size() const { return size_; }

private:
    SymExpr size_;
};

// Dense-Variable: ragged dimension (variable seq_len per batch)
class DenseVariable : public Axis {
public:
    DenseVariable(SymExpr outer_size, int* lengths);

    bool is_dense() const override { return true; }
    bool is_fixed() const override { return false; }
    SymExpr extent() const override;  // Sum of lengths

    int length(int outer_idx) const { return lengths_[outer_idx]; }
    int* lengths() const { return lengths_; }

private:
    SymExpr outer_size_;
    int* lengths_;
};

// Sparse-Variable: CSR-like (MoE routing)
class SparseVariable : public Axis {
public:
    SparseVariable(Axis* parent, SymExpr max_size, SymExpr nnz,
                   int* indptr, int* indices);

    bool is_dense() const override { return false; }
    bool is_fixed() const override { return false; }
    SymExpr extent() const override { return nnz_; }

    Axis* parent() const { return parent_; }
    int* indptr() const { return indptr_; }
    int* indices() const { return indices_; }

    // Get range for given parent index
    std::pair<int, int> range(int parent_idx) const {
        return {indptr_[parent_idx], indptr_[parent_idx + 1]};
    }

    // Get coordinate from position
    int coordinate(int position) const { return indices_[position]; }

private:
    Axis* parent_;
    SymExpr max_size_;
    SymExpr nnz_;
    int* indptr_;
    int* indices_;
};

// Sparse-Fixed: ELL-like (fixed nnz per row)
template<int NNZ_PER_ROW>
class SparseFixed : public Axis {
public:
    SparseFixed(Axis* parent, SymExpr max_size, int* indices);

    bool is_dense() const override { return false; }
    bool is_fixed() const override { return true; }
    SymExpr extent() const override;

    static constexpr int nnz_per_row = NNZ_PER_ROW;
    int* indices() const { return indices_; }

    int coordinate(int parent_idx, int local_idx) const {
        return indices_[parent_idx * NNZ_PER_ROW + local_idx];
    }

private:
    Axis* parent_;
    SymExpr max_size_;
    int* indices_;
};

}  // namespace pto::wsp::axis
```

### 3.2 Axis-Based Iteration

```cpp
namespace pto::wsp {

// Sparse iteration over multiple axes
class TaskIter {
public:
    // Constructor with axes and iteration type
    // iter_types: S=Spatial, R=Reduction
    TaskIter(std::vector<std::shared_ptr<axis::Axis>> axes,
             const char* iter_types);

    // Set task generator function
    template<typename F>
    TaskIter& body(F&& generator);

    // Convert to Workload
    Workload to_workload() const;
};

// Convenience function
template<typename... Axes, typename F>
Workload task_iter(std::tuple<Axes...> axes, const char* iter_types, F&& body);

// Example usage:
Workload moe = task_iter(
    {batch_axis, expert_axis},  // Axes
    "SR",                       // Spatial, Reduction
    [&](auto batch, auto expert) {
        return task(expert_kernels[expert], {tokens[batch]});
    }
);

}  // namespace pto::wsp
```

## 4. Task Categories (From TVM)

### 4.1 Category Definitions

```cpp
namespace pto::wsp {

// Task/operator categories for fusion
enum class TaskCategory {
    INJECTIVE,           // 1-to-1 element-wise (relu, add, mul)
    BROADCAST,           // Broadcasting (add with broadcast)
    REDUCTION,           // Many-to-1 (sum, mean, softmax)
    COMPLEX_OUT_FUSABLE, // Can fuse element-wise at output (matmul, conv)
    OPAQUE               // Cannot analyze, no fusion
};

// Category analysis result
struct TaskAnalysis {
    TaskCategory category;
    AccessPattern input_pattern;   // How inputs are accessed
    AccessPattern output_pattern;  // How outputs are written
    bool can_fuse_input;           // Can fuse at input
    bool can_fuse_output;          // Can fuse at output
};

// Analyze kernel to determine category
TaskAnalysis analyze_kernel(KernelId kernel);

}  // namespace pto::wsp
```

### 4.2 Fusion Primitives

```cpp
namespace pto::wsp {

// Fusion configuration
struct FusionConfig {
    bool fuse_injective_chains = true;
    bool fuse_to_complex_out = true;
    bool fuse_broadcast = true;
    int max_fused_ops = 8;
};

class Schedule {
public:
    // Automatic task annotation
    Schedule& auto_annotate();

    // Fusion primitives
    Schedule& fuse_by_category();
    Schedule& fuse_by_category(FusionConfig config);

    // Manual fusion
    Schedule& fuse_tasks(std::vector<TaskId> tasks);
    Schedule& prevent_fusion(TaskId task);
};

}  // namespace pto::wsp
```

## 5. Format Decomposition (From SparseTIR)

### 5.1 Task Formats

```cpp
namespace pto::wsp::format {

// Composable task format specification
class TaskFormat {
public:
    // hyb(c, k): Hybrid format for load balancing
    // c = column partition factor
    // k = bucket factor (2^k buckets)
    static TaskFormat hyb(int col_partition, int bucket_factor);

    // ell(nnz): Fixed nnz per row
    static TaskFormat ell(int nnz_per_row);

    // csr(): Variable nnz per row (default)
    static TaskFormat csr();

    // bsr(block_size): Block sparse
    static TaskFormat bsr(int block_size);

    // Custom format
    static TaskFormat custom(
        std::function<DecomposedWorkload(Workload)> decompose_fn
    );
};

// Decomposed workload result
struct DecomposedWorkload {
    std::vector<Workload> partitions;      // Partitioned workloads
    std::vector<int> bucket_assignments;   // Task → bucket mapping
};

}  // namespace pto::wsp::format

namespace pto::wsp {

class Schedule {
public:
    // Format decomposition
    Schedule& decompose(format::TaskFormat fmt);

    // Bucket-aware scheduling
    Schedule& dispatch(DispatchPolicy policy);  // Updated to handle buckets
    Schedule& bucket_affinity();                // Same bucket → same core
};

}  // namespace pto::wsp
```

## 6. Three-Stage Compilation

### 6.1 Stage I: Coordinate Space

```cpp
namespace pto::wsp::stage1 {

// Coordinate space workload
class CoordWorkload {
public:
    // From user specification
    static CoordWorkload from_workload(Workload w);

    // Stage I transformations
    CoordWorkload& sparse_reorder(std::vector<axis::Axis*> new_order);
    CoordWorkload& sparse_fuse(std::vector<axis::Axis*> to_fuse);
    CoordWorkload& decompose(format::TaskFormat fmt);

    // Lower to Stage II
    stage2::PositionWorkload lower() const;
};

}  // namespace pto::wsp::stage1
```

### 6.2 Stage II: Position Space

```cpp
namespace pto::wsp::stage2 {

// Position space workload (after iteration lowering)
class PositionWorkload {
public:
    // Stage II transformations (TVM-like schedule primitives)
    PositionWorkload& split(axis::Axis* ax, int factor,
                            axis::Axis** outer, axis::Axis** inner);
    PositionWorkload& fuse(axis::Axis* a, axis::Axis* b, axis::Axis** fused);
    PositionWorkload& reorder(std::vector<axis::Axis*> order);
    PositionWorkload& tile(axis::Axis* a, axis::Axis* b,
                           int tile_a, int tile_b);

    // Binding
    PositionWorkload& bind(axis::Axis* ax, const char* resource);
    PositionWorkload& parallel(axis::Axis* ax);
    PositionWorkload& vectorize(axis::Axis* ax);

    // Memory
    PositionWorkload& cache_read(BufferId buf, const char* scope);
    PositionWorkload& cache_write(BufferId buf, const char* scope);

    // Lower to Stage III
    stage3::CompiledProgram compile() const;
};

}  // namespace pto::wsp::stage2
```

### 6.3 Stage III: AICPU Binary

```cpp
namespace pto::wsp::stage3 {

// Final compiled program
class CompiledProgram {
public:
    // Binary access
    const uint8_t* binary() const;
    size_t binary_size() const;

    // Metadata
    size_t state_size() const;
    size_t stack_size() const;
    std::vector<std::string> required_bindings() const;

    // Binding
    void bind(const char* name, int64_t value);
    void bind(const char* name, void* ptr);
    void bind(const char* name, SymTensor tensor);

    // Execution
    void execute(AICPUContext* ctx);
};

}  // namespace pto::wsp::stage3
```

## 7. Cross-Level Abstraction (From Relax)

### 7.1 Unified Module

```cpp
namespace pto::wsp {

// Cross-level module containing workloads, kernels, and library calls
class Module {
public:
    // Add workloads
    void add_workload(const char* name, Workload w);

    // Add tensor programs (kernels)
    void add_tensorir(const char* name, TensorIR tir);

    // Add external library functions
    void add_library(const char* name, const char* lib_func);

    // Accessors
    Workload get_workload(const char* name) const;
    TensorIR get_tensorir(const char* name) const;

    // Cross-level transformations
    Module& lift_workspace(const char* tir_name);
    Module& fuse_tensorir(const char* a, const char* b);
    Module& inline_tensorir(const char* name);

    // Analysis
    Module& analyze_kernels();  // Populate task categories
    TaskAnalysis get_analysis(const char* kernel_name) const;
};

}  // namespace pto::wsp
```

### 7.2 Cross-Level Function Calls

```cpp
namespace pto::wsp {

// call_tir: Call tensor program with DPS semantics
Workload call_tir(
    const char* tir_name,
    std::vector<SymTensor> inputs,
    SymTensor output_annotation
);

// call_dps_library: Call external library with DPS semantics
Workload call_dps_library(
    const char* lib_func,
    std::vector<SymTensor> inputs,
    SymTensor output_annotation
);

// Partial lowering
class PartialLower {
public:
    PartialLower& to_library(const char* lib_func, Pattern pattern);
    PartialLower& to_tensorir(const char* tir_name, Pattern pattern);
    PartialLower& remaining_to_tir();

    Schedule finalize();
};

// Usage:
Schedule sched = workload.schedule()
    .partial_lower()
        .to_library("cublas.gemm", pattern::gemm())
        .to_library("flashattn", pattern::attention())
        .remaining_to_tir()
    .dispatch(round_robin(4));

}  // namespace pto::wsp
```

### 7.3 Dataflow Blocks

```cpp
namespace pto::wsp {

// Dataflow block: pure region for safe optimization
template<typename F>
Workload dataflow(F&& body);

// Example:
Workload w = dataflow([&]() {
    auto t1 = task(gemm, {A, B});
    auto t2 = task(relu, {t1});
    auto t3 = task(add, {t2, C});
    return t3;
});

// Properties:
// - No side effects within block
// - Safe to reorder tasks
// - Automatic dead code elimination
// - Fusion-friendly

}  // namespace pto::wsp
```

## 8. Memory Planning Enhancements

### 8.1 Symbolic Memory Planning

```cpp
namespace pto::wsp {

class MemoryPlanner {
public:
    // Analyze workload for memory requirements
    static MemoryPlanner analyze(Workload w);

    // Get buffer allocations
    struct BufferAlloc {
        SymExpr size;
        SymExpr alignment;
        Lifetime lifetime;
    };
    std::vector<BufferAlloc> allocations() const;

    // Apply optimizations
    MemoryPlanner& share_when_possible();  // Reuse same-shape buffers
    MemoryPlanner& static_upper_bound();   // Use upper bound for dynamic

    // Generate allocation plan
    MemoryPlan finalize();
};

// Memory plan result
class MemoryPlan {
public:
    // Total memory required
    SymExpr total_size() const;
    int64_t max_size() const;  // Upper bound

    // Individual buffer info
    struct BufferInfo {
        size_t offset;
        SymExpr size;
        bool shared;
    };
    BufferInfo get_buffer(BufferId id) const;

    // Apply to workload
    Workload apply(Workload w);
};

}  // namespace pto::wsp
```

### 8.2 Workspace Lifting

```cpp
namespace pto::wsp {

// Lift workspace from kernel to workload level
class WorkspaceLifter {
public:
    static WorkspaceLifter analyze(Module m, const char* kernel);

    // Get workspace requirements
    struct WorkspaceReq {
        SymExpr size;
        const char* scope;  // "global", "ub", "l1"
    };
    std::vector<WorkspaceReq> requirements() const;

    // Perform lifting
    Module lift();
};

// Schedule primitive for workspace sharing
Schedule& share_workspace(std::vector<TaskId> tasks);

}  // namespace pto::wsp
```

## 9. Enhanced Schedule API

### 9.1 Complete Schedule Class

```cpp
namespace pto::wsp {

class Schedule {
public:
    // === Stage I (Coordinate Space) ===

    // Axis transformations
    Schedule& sparse_reorder(std::vector<axis::Axis*> order);
    Schedule& sparse_fuse(std::vector<axis::Axis*> axes);

    // Format decomposition
    Schedule& decompose(format::TaskFormat fmt);

    // === Analysis ===

    // Automatic annotation
    Schedule& auto_annotate();  // Analyze kernels → categories

    // Fusion
    Schedule& fuse_by_category();
    Schedule& fuse_by_category(FusionConfig config);
    Schedule& fuse_tasks(std::vector<TaskId> tasks);

    // === Partial Lowering ===

    PartialLower partial_lower();

    // === Stage II (Position Space) ===

    // Loop transformations
    Schedule& tile(axis::Axis* ax, int factor);
    Schedule& split(axis::Axis* ax, int factor);
    Schedule& fuse_axes(axis::Axis* a, axis::Axis* b);
    Schedule& reorder(std::vector<axis::Axis*> order);

    // Dispatch (Host → AICPU)
    Schedule& dispatch(DispatchPolicy policy);
    Schedule& colocate(axis::Axis* ax);
    Schedule& colocate(SymExpr key);

    // Issue (AICPU → AICore)
    Schedule& issue(IssueOrder order);
    Schedule& bind(axis::Axis* ax, CoreSet cores);
    Schedule& steal();
    Schedule& bucket_affinity();

    // Pipeline
    Schedule& pipeline(int depth);
    Schedule& double_buffer();
    Schedule& prefetch(int n);

    // Stitch
    static Schedule stitch(Schedule a, Schedule b);
    static Schedule interleave(Schedule a, Schedule b);

    // Memory
    Schedule& share_workspace(std::vector<TaskId> tasks);
    Schedule& cache(BufferId buf, const char* scope);

    // === Compilation ===

    CompiledProgram compile();
    CompiledProgram compile(CompilerOptions opts);
};

}  // namespace pto::wsp
```

## 10. Complete Example

### 10.1 MoE with All Features

```cpp
using namespace pto::wsp;
using namespace pto::wsp::sym;
using namespace pto::wsp::axis;

// === Symbolic dimensions ===
auto batch = SymExpr::var("batch");
auto tokens = SymExpr::var("tokens");
auto num_experts = SymExpr::var("num_experts");
auto hidden = SymExpr::constant(4096);

// === Axis definitions ===
auto batch_axis = std::make_shared<DenseFixed>(batch);
auto token_axis = std::make_shared<DenseVariable>(batch, seq_lens);
auto expert_axis = std::make_shared<SparseVariable>(
    token_axis.get(), num_experts, total_routed,
    routing_indptr, routing_indices
);

// === Workload specification ===
Workload moe = dataflow([&]() {
    // Router stage
    auto routing = task_iter(
        {batch_axis, token_axis},
        "SS",
        [&](auto b, auto t) {
            return task(router_kernel,
                {input_tokens[b][t]},
                SymTensor{{num_experts}, f32}
            );
        }
    );

    // Expert stage with sparse iteration
    auto experts = task_iter(
        {batch_axis, expert_axis},
        "SR",
        [&](auto b, auto e) {
            return task(expert_kernels[e],
                {routed_tokens[b]},
                SymTensor{{hidden}, f16}
            );
        }
    );

    return routing.then(experts);
});

// === Module with cross-level info ===
Module m;
m.add_workload("moe", moe);
m.add_tensorir("expert_ffn", expert_ffn_tir);
m.add_library("cublas_gemm", "cublas.gemm");

// === Schedule ===
Schedule sched = moe.schedule()
    // Stage I: Format decomposition for load balance
    .decompose(format::TaskFormat::hyb(4, 3))

    // Analysis
    .auto_annotate()
    .fuse_by_category()

    // Partial lowering
    .partial_lower()
        .to_library("cublas.gemm", pattern::gemm())
        .remaining_to_tir()

    // Stage II: Dispatch and issue
    .dispatch(DispatchPolicy::dynamic())
    .issue(IssueOrder::priority([](Task& t) {
        int expert = t.params.get<int>(1);
        return -expert_load[expert];
    }))
    .colocate(expert_axis.get())
    .bucket_affinity()

    // Pipeline
    .pipeline(3)
    .steal()

    // Memory
    .share_workspace({expert_tasks});

// === Compile ===
CompiledProgram prog = sched.compile({
    .opt_level = OptLevel::O3,
    .enable_symbolic_planning = true
});

// === Execute ===
prog.bind("batch", batch_size);
prog.bind("tokens", token_count);
prog.bind("num_experts", 8);
prog.bind("routing_indptr", routing_indptr);
prog.bind("routing_indices", routing_indices);
prog.bind("input_tokens", input_tensor);
prog.bind("expert_weights", weight_tensors);

AICPUContext ctx;
prog.execute(&ctx);
```

## 11. Summary

### 11.1 New APIs from TVM Series

| Source | API | Purpose |
|--------|-----|---------|
| Relax | `SymExpr`, `SymVar` | Symbolic dimension tracking |
| Relax | `SymTensor` | Tensor with symbolic shape |
| Relax | `match_cast()` | Assert data-dependent shapes |
| Relax | `dataflow()` | Pure computation regions |
| Relax | `call_tir()`, `call_dps_library()` | Cross-level calls |
| Relax | `partial_lower()` | Incremental lowering |
| SparseTIR | `DenseFixed`, `SparseVariable`, etc. | Axis types |
| SparseTIR | `task_iter()` | Axis-based iteration |
| SparseTIR | `TaskFormat::hyb()` | Load-balanced decomposition |
| SparseTIR | `sparse_reorder()`, `sparse_fuse()` | Stage I transforms |
| TVM | `TaskCategory` | Fusion classification |
| TVM | `auto_annotate()` | Automatic category detection |
| TVM | `fuse_by_category()` | Category-based fusion |
| TVM | `tile()`, `split()`, `reorder()` | Loop transformations |

### 11.2 Compilation Pipeline

```
User Code
    │
    ▼
┌─────────────────────────────────────────┐
│ Stage I: Coordinate Space                │
│ - Workload with axes and symbolic shapes │
│ - Format decomposition                   │
│ - Sparse transformations                 │
└─────────────────────────────────────────┘
    │ lower()
    ▼
┌─────────────────────────────────────────┐
│ Stage II: Position Space                 │
│ - Loop structures                        │
│ - Dispatch/issue primitives              │
│ - Pipeline/stealing configuration        │
└─────────────────────────────────────────┘
    │ compile()
    ▼
┌─────────────────────────────────────────┐
│ Stage III: AICPU Binary                  │
│ - JIT-compiled executable                │
│ - Optimized for target hardware          │
└─────────────────────────────────────────┘
```

---
*Version: 7.1*
*Last Updated: 2025-01-17*
