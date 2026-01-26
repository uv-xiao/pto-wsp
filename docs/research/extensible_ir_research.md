I can’t write into this repo from the current environment (both `apply_patch` and shell writes are blocked), so below is a concrete, “paste-into-headers” IR infrastructure design that fits `docs/ir-design.md` but adds the missing extensibility: `WorkloadLevel`, a clean visitor + walker, and a minimal pass/rewriter system (still far simpler than MLIR).

## 1) Core IR model (hierarchical CPU/NPU)

### Key idea
Keep the current “Workload IR + Schedule IR + Module” shape, but make **level an explicit property on every IR node** and make hierarchy explicit via a **cross-level call node**.

### Types to add (new headers)
Create these headers (suggested layout):
- `include/pto/rt/ir/core.hpp` (IDs, levels, base node, factory)
- `include/pto/rt/ir/nodes.hpp` (axis/workload/csp/schedule nodes + `ExtOpNode`)
- `include/pto/rt/ir/module.hpp` (Module, WorkloadDef, ScheduleDef, symbol tables)
- `include/pto/rt/ir/visitor.hpp` (visitor + walker)
- `include/pto/rt/ir/pass.hpp` (passes + pass manager + diagnostics)
- `include/pto/rt/ir/rewriter.hpp` (tree rewriter used by transform passes)
- `include/pto/rt/ir/ir.hpp` (umbrella include)

### `WorkloadLevel` (required)
```cpp
enum class WorkloadLevel : uint8_t { CPU, NPU, Any };
```
- `Any` is useful for level-agnostic nodes (e.g., axis types) and future extensions.
- Every `IRNode` stores `WorkloadLevel level`.

### Base node (immutable, shared, simple)
- Use `std::shared_ptr<const T>` everywhere for immutability.
- Keep `NodeKind` for fast dispatch (no RTTI dependence).

```cpp
using NodeId = uint64_t;

enum class NodeKind : uint16_t {
  // axis...
  // workload...
  Call,   // NEW: cross-level call (hierarchy)
  Ext,    // NEW: extension op
  // schedule...
};

struct IRNode {
  const NodeId id;
  const NodeKind kind;
  const WorkloadLevel level;

  virtual ~IRNode() = default;
  virtual void print(std::ostream&, int indent=0) const = 0;

  using ChildFn = std::function<void(const std::shared_ptr<const IRNode>&)>;
  virtual void forEachChild(const ChildFn&) const {} // default: leaf
};

class IRFactory { /* monotonic NodeId + create<T>(...) */ };
```

### Hierarchical workload-schedule model (CPU ↔ NPU)
Add a **workload node** that references another workload (typically NPU) by symbol:
```cpp
struct CallNode : WorkloadNode {
  const std::string callee_workload;          // symbol name
  const std::optional<std::string> callee_schedule; // optional override
  const WorkloadLevel callee_level;           // usually NPU when called from CPU
  const std::vector<std::string> args;        // keep strings for now (like current TaskNode)
  const std::vector<std::string> resources;

  // children: none (symbol edge), OR optionally attach resolved ptrs later via analysis
};
```

In `module.hpp`, make defs level-aware:
```cpp
struct WorkloadDef {
  std::string name;
  WorkloadLevel level; // NEW
  std::vector<std::pair<std::string, IRPtr<AxisNode>>> params;
  IRPtr<WorkloadNode> body;
};

struct ScheduleDef {
  std::string name;
  std::string workload_name;
  WorkloadLevel level; // NEW (should match workload’s level)

  // IMPORTANT change for extensibility:
  // store schedule statements as a list, not fixed fields.
  std::vector<IRPtr<ScheduleNode>> directives;
};
```

Why the list form matters:
- You can add new schedule primitives later without changing `ScheduleDef` (and without “optional field explosion” like `spatial_map`, `layouts`, …).

## 2) Visitor pattern + traversal (clean, extensible)

### Goals
- Works for both built-in nodes and extension nodes.
- Can be used for analyses, verification, printing, and passes.
- Keeps traversal logic *outside* node classes (nodes only expose children).

### Design
Use:
- `dispatch(node.kind)` → calls the most specific `visitor.enter(const XNode&)`.
- A generic `walk(root, visitor)` handles recursion and skip/abort control.

```cpp
enum class WalkControl { Continue, SkipChildren, Abort };

struct IRVisitor {
  virtual ~IRVisitor() = default;

  virtual WalkControl enter(const IRNode&) { return WalkControl::Continue; }

  // Optional typed hooks (override only what you need).
  virtual WalkControl enter(const TaskNode& n) { return enter((const IRNode&)n); }
  virtual WalkControl enter(const CallNode& n) { return enter((const IRNode&)n); }
  virtual WalkControl enter(const ExtOpNode& n) { return enter((const IRNode&)n); }
  // ...one per concrete node type...
};

// walker does:
// 1) dispatch enter()
// 2) if Continue -> recurse via forEachChild()
// 3) optional post hook if you want (keep it minimal unless needed)
WalkControl walk(const IRNodePtr& root, IRVisitor& v);
```

**Module-aware traversal (optional but recommended)**  
Add a `ModuleWalker` that can optionally “follow” `CallNode` edges by resolving symbols via a `SymbolTable` built from `Module`. This is how you traverse hierarchical CPU→NPU programs without embedding sub-IR directly inside nodes.

## 3) Transformation pass infrastructure (simple but scalable)

### Pass API
Keep it minimal: passes run on `Module` and may replace `WorkloadDef.body`, schedule directives, pipelines, etc.

```cpp
enum class DiagnosticSeverity { Note, Warning, Error };

struct Diagnostic {
  DiagnosticSeverity severity;
  std::string pass;
  std::string message;
  std::optional<NodeId> node;
};

struct PassResult {
  bool changed = false;
  bool ok = true;
};

struct PassContext {
  IRFactory& factory;              // create new nodes
  std::vector<Diagnostic>& diags;   // collect diagnostics
  // optional: SymbolTable*, options, etc
};

class Pass {
public:
  virtual ~Pass() = default;
  virtual std::string_view name() const = 0;
  virtual PassResult run(Module& m, PassContext& ctx) = 0;
};

class PassManager {
public:
  void add(std::unique_ptr<Pass> p);
  PassResult run(Module& m); // stops early on errors
};
```

### Rewriter (enables most transforms without MLIR)
Provide a tree-rewriter that:
- Recursively rewrites children
- Rebuilds nodes if any child changed
- Lets subclasses override specific rewrite points

```cpp
class IRRewriter {
public:
  explicit IRRewriter(IRFactory& f) : f_(f) {}

  IRNodePtr rewrite(const IRNodePtr& n);              // dispatch by kind
  IRPtr<WorkloadNode> rewriteWorkload(const IRPtr<WorkloadNode>& n);
  IRPtr<ScheduleNode> rewriteSchedule(const IRPtr<ScheduleNode>& n);

protected:
  // Override points (default: rebuild after rewriting children)
  virtual IRNodePtr rewriteTask(const IRPtr<TaskNode>& n);
  virtual IRNodePtr rewriteCall(const IRPtr<CallNode>& n);
  virtual IRNodePtr rewriteExt(const IRPtr<ExtOpNode>& n);

private:
  IRFactory& f_;
};
```

This stays “MLIR-light”:
- no SSA values, blocks, dominance, etc.
- still supports systematic transforms and future growth.

## 4) Extension points (new primitives, new backends)

### New primitives: `ExtOpNode`
Add a single “escape hatch” op that backends/passes can recognize by name + attrs.

```cpp
using Attr = std::variant<int64_t, double, bool, std::string,
                          std::vector<int64_t>, std::vector<std::string>>;

enum class ExtClass : uint8_t { Axis, Workload, Schedule, CSP };

struct ExtOpNode : IRNode {
  const ExtClass cls;
  const std::string op_name; // e.g. "npu.double_buffer", "cpu.task_window"
  const std::unordered_map<std::string, Attr> attrs;
  const std::vector<IRNodePtr> children;

  void forEachChild(const ChildFn& fn) const override;
};
```

This enables:
- adding schedule directives like `pipeline_depth`, `task_window`, `dispatch_threshold` without changing core node kinds
- backend-specific annotations without polluting common IR

### New backends (future)
Keep backend integration out of the IR headers, but plan for:
- `Backend` interface: `supports(NodeKind or op_name)`, `lower(Module)`
- per-backend pass pipelines: `PassManager pm; pm.add(...);`

## 5) Minimal changes you should apply to `docs/ir-design.md`
- Add `WorkloadLevel` to `IRNode` (and mention `Any` as optional).
- Add `CallNode` (cross-level hierarchy primitive).
- Change `ScheduleDef` from fixed optional fields to `directives: vector<ScheduleNode>`.
- Add `ExtOpNode` as the official extension mechanism.
- Add a short “Passes & Visitors” section describing `IRVisitor`, `walk`, `PassManager`, and `IRRewriter`.

If you want, I can also provide a concrete “v1 header set” (full compilable code for the headers above + a tiny `tests/ir_smoke_test.cpp`) in one message so you can paste it directly into the repo.