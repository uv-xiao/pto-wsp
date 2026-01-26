# Design Questions for PTO Workload-Schedule Programming (PTO-WSP)

Please edit this file with your answers. You can:
- Select options by marking with `[x]`
- Add free-form answers below questions
- Skip questions that are not relevant
- Add clarifications or new requirements

---

## 1. Scope & Generality

**Q1.1**: What is the primary use case scope?
- [x] (a) LLM inference only (attention, MLP, MoE)
- [ ] (b) General AI workloads (conv, pooling, custom ops)
- [ ] (c) Arbitrary kernel composition (user-defined operators)

**Answer**: Although we focus on LLM inference, we still need to keep generality for arbitratry kernels' composition running. But we just use LLM inference as examples.

**Q1.2**: Should the runtime support heterogeneous kernels in a single dispatch?
- Example: Can one dispatch include both a GEMM kernel and an attention kernel running on different AICores?
- [x] Yes
- [ ] No

**Answer**: We need to make it clear what is dispatch and what is issue. For dispatch, I want it to represent with AICPU should the tasksets (one taskset can be a kernel called multiple times, like with different outer loop indices, such as batche number or sequence number). For issue, I want it to represent the kernels in the same AICPU launched onto AICores. The kernel design (existing in PTO-ISA) determines the different AICores used.

**Q1.3**: For MoE specifically, what routing patterns need support?
- [ ] (a) Static routing (known at compile time)
- [x] (b) TopK routing (K experts per token, K known at compile time)
- [x] (c) Fully dynamic routing (variable K per token)

**Answer**: We need to have general dynamic support. But this should also be done with high-level abstraction, such as indexing of sparse algebra.

---

## 2. Task & Dependency Model

**Q2.1**: How should task dependencies be expressed?
- [x] (a) Implicit sequential: Tasks in same queue execute in order (like CUDA streams)
- [x] (b) Explicit events: Tasks signal completion, others wait (like cudaStreamWaitEvent)
- [ ] (c) Full DAG: User specifies complete dependency graph
- [ ] (d) Barrier-based: Bulk synchronization at defined points

**Answer**: This is very important. I think we can combine implicit sequential and explicit events. But I think the tasks in the same queue can still overlap. So the dependencies can be very flexible. CUDA streams, which works with work stealing by SMs, might not be the final solution. I think events are general to represent, but we need more discussion about this problem.

**Q2.2**: Should tasks be able to dynamically generate new tasks?
- Example: An MoE router task creates expert tasks based on routing results
- [x] Yes (open system - tasks can spawn tasks)
- [ ] No (closed system - all tasks known upfront)

**Answer**: This also relates to the routing pattern problem. The key is the abstraction.

**Q2.3**: What is the expected task granularity?
- [ ] (a) Coarse: One task = one layer (attention, MLP)
- [x] (b) Medium: One task = one tile/block of computation
- [ ] (c) Fine: One task = one AICore invocation
- [ ] (d) User-defined: Programmer chooses granularity

**Answer**: I think one unit task for the extension is a kernel call to existing pto-isa kernels. So b might be the correct answer. 

---

## 3. Dispatch & Scheduling

**Q3.1**: What dispatch strategies should be supported?
- [x] (a) Static only: Round-robin or pre-assigned to AICores
- [x] (b) Dynamic only: Work stealing from shared queue
- [x] (c) Hybrid: User chooses per-workload
- [x] (d) Adaptive: Runtime auto-selects based on workload characteristics

**Answer**: All the above should be covered by the extension.

**Q3.2**: Who decides the dispatch strategy?
- [x] (a) Compile-time (template parameter)
- [x] (b) Runtime configuration
- [x] (c) Per-dispatch decision

**Answer**: All supported.

**Q3.3**: For work stealing, what is the stealing granularity?
- [x] (a) Single task
- [ ] (b) Batch of tasks
- [ ] (c) Configurable

**Answer**: Single task.

---

## 4. Memory & Resource Management

**Q4.1**: Should tensors/buffers be first-class citizens in the API?
- [x] (a) Yes - explicit tensor handles with lifetime management
- [ ] (b) No - just pass pointers like current PTO-ISA
- [ ] (c) Optional - support both styles

**Answer**: I think PTO-ISA has first-class citizen tensor/buffers. Am I wrong?

**Q4.2**: How should intermediate buffers (between fused kernels) be managed?
- [x] (a) User allocates and manages
- [ ] (b) Runtime manages a buffer pool
- [ ] (c) Automatic allocation based on dependency analysis

**Answer**: User allocates. But use can also specify buffer pool for runtime allocation.

**Q4.3**: For paged KV cache (LLM inference), should the runtime provide built-in support?
- [ ] (a) Yes - first-class paged memory abstraction
- [x] (b) No - user implements on top of basic primitives
- [ ] (c) Optional library feature

**Answer**: Don't be so specific.

---

## 5. Programming Model

**Q5.1**: How much should the API constrain vs enable flexibility?
- [ ] (a) Structured: Fixed patterns (Plan-Dispatch-Execute), less flexibility
- [x] (b) Flexible: Primitives that compose freely, more complexity
- [ ] (c) Layered: Low-level primitives + high-level patterns on top

**Answer**: We must remain enough flexibility for humans.

**Q5.2**: Should there be a "Plan" phase or just direct dispatch?
- [x] (a) No explicit Plan API, just describe and dispatch
- [ ] (b) Planning happens implicitly/internally
- [ ] (c) Planning is optional for advanced users

**Answer**: Plan is left for programmers to create their own.

**Q5.3**: What syntax style is preferred?
- [ ] (a) Macro-based: `DISPATCH_TIERED(...)` (current v4 style)
- [ ] (b) Class-based: `Dispatcher().add(...).run()` (builder pattern)
- [x] (c) Function-based: `dispatch(tasks, config)` (simple calls)
- [ ] (d) DSL/declarative: Configuration files or decorators

**Answer**:

---

## 6. Stitching & Fusion

**Q6.1**: What does "stitching" mean in your context?
- [x] (a) Graph stitching: Combine multiple task graphs into one dispatch
- [ ] (b) Kernel fusion: Fuse multiple kernels into one
- [x] (c) Batch stitching: Combine requests from multiple batches
- [ ] (d) Layer stitching: Execute multiple layers without returning to host

**Answer**: It's every general. It's a combination of task dispatch and issueing.

**Q6.2**: Should stitching be explicit or automatic?
- [x] (a) User explicitly defines stitching points
- [ ] (b) Runtime automatically finds stitching opportunities
- [ ] (c) Compiler/JIT optimizes stitching

**Answer**: Currently, we let programmers do optimizations.

---

## 7. AICPU Role

**Q7.1**: What should run on AICPU vs Host?
- [x] (a) AICPU does everything: Planning, dispatch, coordination
- [ ] (b) Host plans, AICPU dispatches: Split responsibilities
- [ ] (c) Configurable: User chooses based on workload

**Answer**: The extension is for AICPU! The Host logic should be programmed by users.

**Q7.2**: Can AICPU run multiple threads for parallel planning?
- [x] Yes
- [ ] No
- [ ] Unknown / Need to investigate

**Answer**: Sure, multiple threads and multiple streams. But this might not be explicit in the extension.

**Q7.3**: Should AICPU maintain persistent state between dispatches?
- [x] (a) Yes - keep scheduler running, avoid cold start
- [ ] (b) No - stateless dispatch each time
- [ ] (c) Optional - user chooses

**Answer**: I think we need states for dependencies management.

---

## 8. Compatibility & Migration

**Q8.1**: How important is backward compatibility with existing PTO-ISA code?
- [x] (a) Must be drop-in compatible
- [ ] (b) Migration path required but can break API
- [ ] (c) Clean slate, new API

**Answer**: Extension must work with PTO-ISA

**Q8.2**: Should the runtime work with existing PyPTO/CANN integration?
- [ ] (a) Must integrate seamlessly
- [x] (b) Can require modifications to PyPTO
- [ ] (c) Separate system initially

**Answer**: Must work with CANN but can require later modifications to PyPTO.

---

## 9. Performance Priorities

**Q9.1**: Rank these priorities (1=highest, 5=lowest):
- [3] Minimize dispatch latency
- [2] Maximize throughput
- [1] Support dynamic workloads
- [5] Minimize memory footprint
- [4] Ease of programming

**Answer**: Human-in-the-loop is a fundamental requirements. Generality is very important for human to optimize the design.

**Q9.2**: What is the acceptable overhead for dynamic features?
- [ ] (a) Zero overhead when not used
- [ ] (b) Small fixed overhead acceptable (<1%)
- [x] (c) Performance vs flexibility tradeoff acceptable

**Answer**: No need to consider too much about overheads.

---

## 10. Concrete Examples

**Q10.1**: Besides FlashInfer-style decode attention, what are the top 3 workloads this must support well?

**Answer**: MoE. Just FlashInfer + MoE is good. The DeepSeek kernel is a complete example.

**Q10.2**: Can you provide a concrete MoE dispatch scenario?
- How many experts? How is routing decided? What happens after routing?

**Answer**: Look at /data/shibizhao/uvxiao/cann-recipes-infer/models/deepseek-v3.2-exp

**Q10.3**: What does a "stitched" execution look like in practice?
- Example: Two consecutive operations that should be stitched

**Answer**: Operations from two batches that have no dependencies and can run in parallel.

---

## 11. Additional Requirements

Please add any requirements, constraints, or considerations not covered above:

**Answer**: Always keep in mind docs/uv/requirements.md. Learn carefully from [text](research/01_flashinfer.md) [text](research/06_megakernels.md) [text](research/07_cuda_streams_workstealing.md)

