# PTO Workload-Schedule Programming (PTO-WSP) framework: Design Redundancy Analysis

This document identifies and analyzes potentially redundant or overlapping constructs in the PTO Workload-Schedule Programming (PTO-WSP) framework design.

## Methodology

1. **Semantic Analysis**: Determine if constructs mean the same thing
2. **Type System Check**: Compare types produced by each construct
3. **Implementation Review**: Check if code is shared or duplicated
4. **Example Validation**: Verify actual usage in deepseek/megakernel examples
5. **Double-Check**: Identify subtle differences that may justify separate existence

---

## Category 1: Axis Type Redundancies

### 1.1 DenseDyn vs Range

| Aspect | DenseDyn | Range |
|--------|----------|-------|
| **Definition** | `struct DenseDyn { int64_t size; }` | `struct Range { int64_t size; }` |
| **Purpose** | Dense axis with dynamic size | Simple [0, n) iteration |
| **Index Type** | `int64_t` | `int64_t` |
| **Methods** | `get_size()` | `get_size()` |
| **Usage in Examples** | 17 occurrences | 3 occurrences (tests only) |

**Analysis:**
- **Semantically identical**: Both represent iteration from 0 to size-1
- **Same type signature**: `int64_t size`, `int64_t Index`, `get_size()`
- **Same implementation**: Identical member variables and methods
- **Naming difference**: "Dense" implies density attribute, "Range" implies simple iteration

**Example Usage:**
```cpp
// DenseDyn used everywhere in actual examples
parallel_for(DenseDyn(num_tokens), [](int64_t t) { ... });

// Range only used in unit tests
auto w = parallel_for(Range(10), [](int64_t i) { ... });
```

**Verdict: MERGE** → Deprecate `Range` as alias, keep `DenseDyn`
- `DenseDyn` is consistent with `Dense<N>` (static) naming
- All actual examples use `DenseDyn`
- `Range` adds no value over `DenseDyn`
- **Note (codex review)**: Consider deprecating as alias rather than hard removal since `Range` is used in tests and spec examples (`spec.md:95`)

Author review: we should just remove Range to make things clean.

---

### 1.2 DenseVar vs DenseDyn

| Aspect | DenseVar (analysis.md) | DenseDyn (implementation) |
|--------|------------------------|---------------------------|
| **Definition** | `struct DenseVar { int64_t* size; }` | `struct DenseDyn { int64_t size; }` |
| **Purpose** | Dynamic size via pointer | Dynamic size via value |
| **Storage** | Pointer to size | Direct value |

**Analysis:**
- `DenseVar` exists only in `analysis.md` line 599
- Not implemented in `types.hpp`
- `DenseDyn` has constructor `DenseDyn(const int64_t* ptr)` that achieves same effect

**Verdict: ALREADY RESOLVED** → `DenseVar` is design artifact, `DenseDyn` handles both cases

Authoer review: remove `DenseVar` everywhere.

---

### 1.3 Unit Axis

| Aspect | Unit | Direct task() |
|--------|------|---------------|
| **Purpose** | Single-element axis for leaf tasks | Create single task directly |
| **Size** | Always 1 | N/A |
| **Usage** | Internal in task() primitive | User-facing API |

**Analysis:**
- `Unit` is used internally by `task()` to create workloads
- Users never directly construct `Unit` axes
- Provides type consistency: all workloads have axes

**Verdict: KEEP** → `Unit` is internal implementation detail, not user-facing redundancy

Aurthor review: keep

---

## Category 2: Dependency Type Redundancies

### 2.1 Independent vs None

| Aspect | Independent | None |
|--------|-------------|------|
| **Purpose** | All tasks can run in parallel | No dependency (for leaf tasks) |
| **Used by** | `parallel_for` | `task()` |
| **Semantic** | Parallelism assertion | Absence of dependency |

**Analysis:**
- `Independent` is returned by `parallel_for` - asserts all generated tasks are independent
- `None` is returned by `task()` - a single task has no internal dependencies
- Subtle difference: `Independent` is about relationships between tasks, `None` is absence of relationships

**Verdict: KEEP BOTH** → Different semantic meanings
- `Independent`: "These tasks have no dependencies on each other"
- `None`: "This is a single task with no dependency structure"

Author review: keep
---

### 2.2 ChannelDep vs EventDep

| Aspect | ChannelDep | EventDep |
|--------|------------|----------|
| **Definition** | `struct ChannelDep {}` (types.hpp) | Mentioned in spec only |
| **Purpose** | CSP channel-based dependency | Event-based dependency |
| **Implementation** | Exists | NOT implemented |

**Analysis:**
- `EventDep` is mentioned in `analysis.md` line 278, 393: "EventDep<E> ≡ ChannelDep<Channel<Signal, 0>>"
- Design explicitly states they are equivalent
- Only `ChannelDep` is implemented

**Verdict: ALREADY RESOLVED** → `EventDep` was design concept, unified under `ChannelDep`
- Event = Channel<Signal, 0>, so EventDep = ChannelDep
- **Note (codex review)**: Implementation uses non-parameterized `ChannelDep` (less expressive than docs' `ChannelDep<Ch>` / `EventDep<E>`)

Author review: I think we can keep the current status for now.

---

### 2.3 Reduction (unused)

| Aspect | Reduction |
|--------|-----------|
| **Definition** | `struct Reduction {}` (types.hpp:127) |
| **Purpose** | Reduction pattern dependency |
| **Usage** | In DepType variant, but never instantiated |
| **Related primitive** | `reduce()` mentioned in analysis.md but not implemented |

**Analysis:**
- `Reduction` exists in types.hpp as part of DepType variant
- No `reduce()` primitive implemented
- Not used in any examples

**Verdict: REMOVE** (or implement reduce primitive if needed)
- Currently dead code
- If `reduce()` is needed, implement it; otherwise remove `Reduction`

Author review: just remove it.

---

### 2.4 DAG Dependency

| Aspect | DAG (spec.md) | depends_on primitive |
|--------|---------------|----------------------|
| **Definition** | `struct DAG { vector<Edge> edges; }` | `workload.depends_on(Event e)` |
| **Implementation** | NOT implemented | Mentioned in spec, not implemented |

**Analysis:**
- DAG dependency type defined in spec.md:228 but not in implementation
- `depends_on` primitive mentioned but not implemented
- Current design uses structural dependencies (inferred from composition) and ChannelDep

**Verdict: REMOVE from spec** → Not implemented, not needed
- Structural dependencies cover most use cases
- ChannelDep covers explicit synchronization
- Explicit DAG adds complexity without clear benefit

Author review: We can remove the deponds_on and DAG things. But meanwhile, I found it hard to describe the explicit synchronization even with the ChannelDep. For example, if we want task B to be executed after task A. We should send to a channel after A and the task B should recv from the channel before running. But we don't have the composition primitive for "send to a channel after A". The `combine` doesn't ensure the "after". Should we add a `sequential` for execution in sequence with dependencies?

**DECISION (2026-01-22):** Add `sequential(A, B, ...)` primitive for explicit ordering.
- `sequential(task_A, task_B)` means B depends on A completing
- Type: `Workload[A.Axes ∪ B.Axes, A.Task ∪ B.Task, Sequential]`
- CSP channels remain for pipeline parallelism (double buffering)
- `sequential()` handles common "do A then B" case more cleanly

---


## Category 3: CSP Redundancies

### 3.1 SignalChannel vs Event

| Aspect | SignalChannel | Event |
|--------|---------------|-------|
| **Definition** | `using SignalChannel = Channel<Signal, 0>` | `using Event = Channel<Signal, 0>` |
| **API** | `signal()`, `wait_signal()` | `record()`, `synchronize()`, `query()` |
| **Purpose** | CSP inter-process sync | Stream execution sync |

**Analysis:**
- Both are type aliases to the same type: `Channel<Signal, 0>`
- Different APIs provide different semantics:
  - SignalChannel: CSP-style `signal()`/`wait_signal()`
  - Event: CUDA-style `record()`/`synchronize()`

**Verdict: MERGE naming, KEEP both APIs**
- Use `Event` as the canonical name (more familiar to CUDA users)
- Keep both API styles as they serve different mental models
- Remove `SignalChannel` type alias (users can use `Event` directly)

Author review: agree on verdict

---

### 3.2 create_event vs Event constructor

| Aspect | create_event() | Event constructor |
|--------|----------------|-------------------|
| **Definition** | `Event create_event(string_view name)` | `Event(string_view name)` |
| **Usage** | Factory function | Direct construction |

**Analysis:**
- Both create the same thing
- Factory function adds no value

**Verdict: KEEP constructor only** → Remove `create_event()` factory
- Direct construction is simpler
- Factory pattern adds no value here
- **Note (codex review)**: Also remove `create_signal_channel()` (`csp.hpp:495`) for consistency

Author review: agree on codex.

---

### 3.3 consume vs while(recv())

| Aspect | consume() | while(recv()) |
|--------|-----------|---------------|
| **Type** | Declarative primitive | Imperative loop |
| **JIT-friendly** | Yes | No |
| **Implementation** | Uses while internally | Directly written |

**Analysis:**
- `consume()` is designed to replace `while(recv())` for JIT friendliness
- Both exist because `consume()` is implemented using `while(recv())` internally
- User code should only use `consume()`

**Verdict: KEEP both** → Different abstraction levels
- `consume()` is user-facing declarative API
- `while(recv())` is internal implementation
- Document that users should prefer `consume()`

Author review: I don't understand why while(recv()) exists. We should only have JIT-friendly design.

**DECISION (2026-01-22):** No action needed - `while(recv())` is internal implementation only.
- Users only see `consume()` which is JIT-friendly
- The imperative `while(recv())` loop is hidden inside `consume()` implementation
- This is an implementation detail, not a user-facing API redundancy

---

### 3.4 select (CSP) vs select (data-parallel)

| Aspect | select (CSP) | select (data-parallel) |
|--------|--------------|------------------------|
| **Purpose** | Wait on multiple channels | Sparse iteration for MoE |
| **Signature** | `select(on_recv(ch1), on_recv(ch2), ...)` | `select(Sparse indices, body)` |
| **Implementation** | spec.md only | primitives.hpp |

**Analysis:**
- Same name, completely different semantics
- CSP `select` is from spec.md, not implemented
- Data-parallel `select` is implemented for MoE routing

**Verdict: RENAME one** → Confusing to have same name
- Keep data-parallel `select` (already implemented and used)
- Rename CSP select to `select_channel()` or `alt()` (from CSP terminology)
- Or don't implement CSP select if not needed

Author review: Is the CSP select required? Consider megakernel example or more advanced work stealing. You should make the decision according to carefully example-driven analysis.

**DECISION (2026-01-22):** Remove CSP `select` from spec.
- Codex analysis confirmed: NOT used in any examples (deepseek, megakernel)
- Examples use dedicated channels with `consume()` - no multi-channel wait needed
- Work stealing would need design changes beyond just adding `select`
- Data-parallel `select` for MoE is already implemented and keeps the name
- If needed later, can add as `alt()` (Go/CSP terminology) or `select_channel()`

---

## Category 4: Policy Redundancies

### 4.1 hash dispatch vs dispatch_by

| Aspect | hash() | dispatch_by() |
|--------|--------|---------------|
| **Definition** | `DispatchPolicy::hash(key_fn, n)` | `DispatchPolicy::dispatch_by(fn)` |
| **Purpose** | Hash key for dispatch | Custom dispatch function |

**Analysis:**
- `hash()` is sugar for `dispatch_by([key_fn, n](Task t) { return hash(key_fn(t)) % n; })`
- Can be implemented using `dispatch_by()`

**Verdict: KEEP for convenience** → Sugar is useful
- `hash()` is clearer for hash-based dispatch
- Keeps API consistent with other built-in policies

Author review: keep.

---

### 4.2 lifo issue policy (spec only)

| Aspect | lifo() |
|--------|--------|
| **Definition** | `IssuePolicy::lifo()` in spec.md:798 |
| **Implementation** | NOT implemented |
| **Usage** | None |

**Verdict: REMOVE from spec** → Not implemented, unclear use case
- FIFO and priority cover most needs
- LIFO rarely needed in task scheduling

Author review: remove.

---

## Category 5: Incomplete Implementations

### 5.1 connect() - CSP composition

| Aspect | connect() |
|--------|-----------|
| **Spec** | `connect({procs}, {channels}) -> Workload[CSP]` (`spec.md:670`) |
| **Implementation** | `connect(...) -> Pipeline` (`csp.hpp:1031`) |
| **Usage** | Not used in examples |

**Analysis:**
- Defined in spec but examples use direct process.start()/join()
- CSP workload type not fully realized
- **Note (codex review)**: Not "missing" but has different API - returns `Pipeline` not `Workload[CSP]`

**Verdict: EITHER implement fully OR simplify API**
- Current: Processes are started/joined manually
- Option A: Implement `connect()` to return executable CSP workload matching spec
- Option B: Update spec to match implementation (`Pipeline` return type)

Author revieW: we should choose Option B. But I want details about the usage in example.

**DECISION (2026-01-22):** Update spec to match implementation (Option B).
- Implementation: `connect(...) -> Pipeline` with manual `pipeline.start()`/`pipeline.join()`
- Example usage (megakernel, tests): Processes started/joined manually, `connect()` used in tests
- Spec change: `connect({procs}, {channels}) -> Pipeline` (not `Workload[CSP]`)
- Pipeline provides explicit lifecycle control which examples actually use

---

### 5.2 reduce() primitive

| Aspect | reduce() |
|--------|----------|
| **Spec** | `reduce(axis, init, body)` in analysis.md:610 |
| **Implementation** | NOT implemented |
| **Dependency type** | `Reduction` exists but unused |

**Verdict: REMOVE or implement**
- Either implement `reduce()` with `Reduction` dependency
- Or remove both from spec and types.hpp

Author review: remove all reduce things.

---

## Category 6: Spec vs Implementation Gaps

### 6.1 Location enum casing

| Spec (spec.md) | Implementation (types.hpp) |
|----------------|----------------------------|
| `GLOBAL` | `Global` |
| `L2` | `L2` |
| `UB` | `UB` |
| `L1` | `L1` |

**Verdict: FIX** → Align spec with implementation (use PascalCase)

Author review: align.

---

## Category 7: Additional Issues (Codex Review)

*These issues were identified by automated codex analysis on 2026-01-20.*

### 7.1 Unused Type-Erasure Helpers

| Aspect | AnyAxis | DepType |
|--------|---------|---------|
| **Definition** | `types.hpp:106` | `types.hpp:140` |
| **Purpose** | Type-erased axis variant | Type-erased dependency variant |
| **Usage** | Only in tests | Only in tests |

**Analysis:**
- `AnyAxis = std::variant<Dense<N>..., DenseDyn, Ragged, Sparse, Unit>`
- `DepType = std::variant<Independent, Sequential, Reduction, ChannelDep, None, Combined>`
- Neither is used in the runtime implementation, only in unit tests

**Verdict: REMOVE or integrate**
- If runtime needs type erasure, integrate properly
- Otherwise remove from types.hpp (keep only in tests if needed)

Author review: use codex to figure out all the type erasure is not used. Discuss with me about it.

**DECISION (2026-01-22):** Remove `AnyAxis` and `DepType` from `types.hpp`.
- Codex analysis confirmed: Neither is used in runtime implementation
- `AnyAxis` at `types.hpp:106` - only referenced in unit tests
- `DepType` at `types.hpp:140` - only referenced in unit tests
- Runtime uses concrete types with templates, not type erasure
- If tests need type erasure, move to test utilities

---

### 7.2 Redundant Channel Factories

| Aspect | Factory | Constructor |
|--------|---------|-------------|
| **channel<T,N>()** | `csp.hpp:440` | `Channel<T,N>(name)` |
| **create_signal_channel()** | `csp.hpp:495` | `SignalChannel(name)` |
| **create_event()** | `csp.hpp:502` | `Event(name)` |

**Analysis:**
- All factory functions are thin wrappers over constructors
- Add no value over direct construction

**Verdict: REMOVE all factories** → Keep constructors only
- `channel<T,N>(name)` → `Channel<T,N>(name)`
- `create_signal_channel(name)` → `Event(name)` (after SignalChannel removal)
- `create_event(name)` → `Event(name)`

Author review: remove all factories.

---

### 7.3 Duplicate TaskId Assignment

| Location | Code |
|----------|------|
| `workload.hpp:118` | `t.id = TaskId(static_cast<uint64_t>(i))` in `enumerate()` |
| `simulation.hpp:375` | `tasks[i].id = TaskId(static_cast<uint64_t>(i))` in `compile()` |

**Analysis:**
- `Workload::enumerate()` assigns task IDs when generating tasks
- `Schedule::compile()` reassigns IDs, overwriting the previous assignment
- Redundant work and potential source of bugs if IDs differ

**Verdict: REMOVE duplicate**
- Assign IDs in one place only (preferably `Schedule::compile()` for consistency)
- Or ensure `enumerate()` doesn't assign IDs if they'll be reassigned

Author review: only keep in compile()
---

### 7.4 Unused External Dependency Flag

| Aspect | Details |
|--------|---------|
| **Definition** | `workload.hpp:96`: `bool has_external_dependency_ = false` |
| **Methods** | `with_dependency()`, `has_dependency()` at `workload.hpp:135` |
| **Usage** | Not wired to `depends_on` (which is spec-only) |

**Analysis:**
- Flag exists to mark workloads with external dependencies
- `depends_on` primitive mentioned in spec but not implemented
- Flag is set but never checked by runtime

**Verdict: REMOVE or implement**
- Either implement `depends_on` and use the flag
- Or remove the flag as dead code

Author review: remove.

---

## Summary Table

| # | Issue | Verdict | Priority |
|---|-------|---------|----------|
| 1 | DenseDyn vs Range | REMOVE Range | High |
| 2 | Independent vs None | KEEP BOTH | - |
| 3 | ChannelDep vs EventDep | ALREADY RESOLVED | - |
| 4 | Reduction unused | REMOVE | Medium |
| 5 | DAG dependency | REMOVE + ADD `sequential()` | Medium |
| 6 | SignalChannel vs Event | MERGE → use Event only | Medium |
| 7 | create_event() | REMOVE factory | Low |
| 8 | select name collision | REMOVE CSP select from spec | High |
| 9 | lifo policy | REMOVE from spec | Low |
| 10 | connect() incomplete | UPDATE spec → Pipeline return | Medium |
| 11 | reduce() missing | REMOVE | Medium |
| 12 | Location enum casing | FIX spec | Low |
| 13 | AnyAxis/DepType unused | REMOVE from types.hpp | Low |
| 14 | Redundant channel factories | REMOVE all factories | Low |
| 15 | Duplicate TaskId assignment | KEEP only in compile() | Medium |
| 16 | Unused external dependency flag | REMOVE | Low |

---

## Recommended Actions (Finalized 2026-01-22)

### Implementation Changes (types.hpp, csp.hpp, workload.hpp)
1. **Remove `Range`** from types.hpp - use `DenseDyn` everywhere
2. **Remove `Reduction`** from types.hpp
3. **Remove `SignalChannel`** type alias - use `Event` only
4. **Remove `AnyAxis`** from types.hpp (move to test utils if needed)
5. **Remove `DepType`** from types.hpp (move to test utils if needed)
6. **Remove factory functions** - `create_event()`, `create_signal_channel()`, `channel<T,N>()`
7. **Remove duplicate TaskId assignment** in `Workload::enumerate()` - keep only in `Schedule::compile()`
8. **Remove external dependency flag** - `has_external_dependency_`, `with_dependency()`, `has_dependency()`

### Spec Changes (spec.md, analysis.md)
9. **Remove `DAG`** dependency type from spec.md
10. **Add `sequential()` primitive** to spec.md and analysis.md
11. **Remove CSP `select`** from spec.md
12. **Update `connect()`** return type to `Pipeline` in spec.md
13. **Remove `lifo()`** from spec.md
14. **Fix `Location` enum** casing in spec.md (use PascalCase)
15. **Remove `DenseVar`** references from analysis.md

---

## Migration Guide

### Removing Range

```cpp
// Before
auto w = parallel_for(Range(10), body);

// After
auto w = parallel_for(DenseDyn(10), body);
```

### Using Event instead of SignalChannel

```cpp
// Before
SignalChannel ch = create_signal_channel("sync");
signal(ch);
wait_signal(ch);

// After
Event e("sync");
record(e);
synchronize(e);
```

---

*Document created: 2026-01-20*
*Updated: 2026-01-20 (codex review - added Category 7, 4 additional issues)*
*Updated: 2026-01-22 (author review + decisions finalized)*
*Status: DECISIONS FINALIZED - READY FOR IMPLEMENTATION*
