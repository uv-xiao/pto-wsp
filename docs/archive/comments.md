V9 version has many problems to fix:

1. We should not use the decorator style for the python binding. Instead, we should use still use declarative-style description in python.
2. The fluent style for schedule is not as good as the combinator style in v8. We prefer the combinator style: `schedule = workload.dispatch(...).stream(...)`, which provides better typing checking.
3. Project organization: similar to pto-isa-lh, but better than it, we should not extend the pto-isa. Instead, we should use pto-isa as a 3rdparty dependency. This requires carefully moving files under docs/uv to a new created repo, named pto-wsp in the directory the current pto-isa repo resides in. You should setup the new project and stop. Then I'll open claude code in the new repo and we can move on.

---

1. I also don't like the lambda expression. For the declarative python, we should use the `with` grammar in python (context manager).
2. As did under @references/pto-isa-lh, we should also provide in-python programming mechanism for kernels or in-task logics or InCore functions, which corresponds to the original pto-isa docs/research/04_pto_isa.md and 3rdparty/pto-isa 's abstraction.
3. We should carefully rethink the programming model. Previosly, we have "Workload-Schedule" model, where task is the atomic unit that is fixed and not-modified by any schedule primitives. I think we should provide a "Hierarchical Workload-Schedule" model, where in-task function (or InCore function, I prefer the term "npu" for InCore) is still programmed by description plus schedule (look at @docs/research/13_pto_isa_lh.md), and outer workload and dispatch-issue is kept. The analysis.md should carefully discuss about the rationale, which considerations to the multi-backend support and the backend's architecture. 
4. Can we improve the compiler infrastructure? We don't want to use the heavy MLIR, but the current design (IRNode) is too simple. We should think about the extensability of the framework. How we support the hiearchical workload-schedule model with workload and schedule primitives from different levels (cpu, npu, ...)
5. I don't think the existing primitives (dispatch, stream, timing) can cover the task-graph execution and the threshold-based pipelining from @references/pto-isa/lh. But we definitely have the potential to cover it. We need to extend our primitives to achieve the coverage. We must be stronger than pto-isa-lh's solution, providing more control over dispatch and issue.
6. You should also carefully think about the backends' implementation. This should be considered from compiler's view. How we reduce replicated coding?

---

1. I hate string-based behavior description, such as `task("attn_kernel", [b, h], [Q[b][h], K[b], V[b], O[b][h]])`. We should have a concise JIT support, which will make things more concise, clear, and intuitive. You can learn from triton/jax JIT. Indeed, tilelang also has the JIT support to be learned. You should work harder to make the python programming easy.
2. The type system is not clear now. We should have typing at different levels: for npu programming, we have typing about tile size and layout. For cpu programming, we have axis and another layout from Dato. Both layout and tile shape are about data (tensor/tile), and we should have a more unified type system for it. You should also learn from the triton's linear layout system to complete the design. Also, in v8, we have detailed typing for all workloads, schedule. We should keep them in v9. We should specify how the type checking can be conducted on python (which is better, since can reject wrong design earlier) and compiler levels. The data constructs should also be provided in python programming (JIT). We should have comprehensive and unified typing about layout.
3. We should avoid using v8's implementation directly. Things are completely different now.
4. We want our schedule to cover pto-isa-lh's, but our stream-based issueing is not as powerful as their task graph based one. So we should have task_graph as a alternative for stream. We should redo the dispatch-issue primitive design to make things concise and powerful. The coverage on pto-isa-lh's solution is a strict requirement.
5. Sharding and replicating should be refinement type as did in Dato, rather than schedule primitives.


---

1. Although we cannot validate NPU (Ascend) backends, we should validate the CPU simulation backend with full end-to-end examples, as did in pto-isa-lh's bgemm, softmax, llama, and we should also add a deepseek-v3.2-exp example.
2. The file structure is kind of wierd: all C++ things are hpp @include/pto/wsp. We should seperate header files and implementation. You should act as a C++ expert to architect the file organization and building system (CMake).
3. The documentation set should be consistent and up-to-date; avoid relying on separate “task plan” files as the source of truth.
4. We need to refine the documents: analysis.md is for rationale/requirement analysis and discussion, showing why our technique looks like this; spec.md holds the detailed design and implementation key points; some other documents hold the indivudial techniques' details; and we need a new document, features.md, which should iterate all features in current version, each one should be concisely explained (can be visual if necessary), and provide links to detailed explanation or codes.

---

1. Documents are a mess. We should put only reference analysis in @docs/reference (old documents @docs/research/id_xxx.md). Put intermediate analysis in @docs/research (such as @docs/research/backend_code_reuse_research.md, @docs/cpp_restructuring_plan.md). Put detailed design in @docs/design (such as @docs/backend-arch.md). And @docs/analysis.md, @docs/spec.md, @docs/features.md under @docs (keep their current location). You should write in @docs/README.md and @CLAUDE.md about how to put document files.
2. You must make sure all features are fully implemented: not only in python frontend/JIT, but also in compiler framework and backend codegen. Prefer tracking remaining gaps via issues/PRs and keeping `docs/spec.md` / `docs/features.md` aligned with implementation.
3. I'm not satisfied about the NPU workload programming. Current version looks like pto-isa-lh. But we need better JIT programming support, like Triton / JAX / Tilelang, you can find references from @docs/research. I especially hate string-based ref, such as tile("x", 32, 128, dtype=DType.F16). We also don't need the extreme combinator style. You should understand what pto-isa @docs/research/04_pto_isa.md and @3rdparty/pto-isa provides, to design the kernel programming DSL, since kernel will generate pto-isa (for Ascend NPU) directly. Why we cannot only use the @kernel programming for NPU? We should extend @kernel programming to include all npu programming primitives. Also, kernel schedule primitives are not present yet. Are they not documented or not implemented?
4. The features.md document should be better structured. The presentation order of the features must have good logic.
5. You write many tests, but I think they are not fully qualifiable. You should reduce the number but improve tests' quality. Especially, we should have end-to-end test to generate true target code for specific backends.
6. Most codes are not well documented. We need document str for every class and function, even for key statements or blocks inside functions/methods.
7. Should TracePolicy be included in task_graph primitive?
8. Do we have Triton's Linear Layout (https://arxiv.org/abs/2505.23819) in tensor memory layout?
9. What is the difference between fifo and work-steal for task-graph?
10. I don't think the extended schedule primitives are interesting. Maybe we just keep the dispatch_threshold and remove the others.
11. For C++ backend, we don't need NPU function/op. We should have complete support for kernel programming's compilation instead.
12. For features, one missing but pretty important is the concurrent mechanism for different backends. We should list what we have and require for CPU sim and Ascend NPU backends, individually. We must cover those in pto-isa-lh and pto-isa-wc (read code/commits carefully to learn). And the better solution is to extract some common things for the concurrent programming and provide them as tools/utilities.

---

For docs/v9_design_review_v2.md, the comments:

*1.1* : fix doc links. For versioning, we should use v9.3 as the design version. But the python package should have "version = 0.1.0", since we don't have a complete implementation yet. For licence, let's use a MIT license for the project, but provide a license under 3rdparty (LICENSE-3RD-PARTY.md) to emphasize pto-isa's CANN license. Also add the license in the README.

*1.2* : the `docs/spec.md` is somehow outdated. We need to use /codex to decide whether a feature is missing in implementaion or the feature should be removed from the document. In my mind, we should prefer consolidating things instead adding all features that might be useless. The document must align with implementation.

*1.3* : similarly, we need to align the document with codes.

*1.4*: I think python ir_passes is not useful and should be removed. The C++ compiler codes should implement the extensive IR features fully. Don't leave the features as only planned.

*2.1*: We should just remove the legacy things. We should not use the term `tl`, but `pto` is better. We should keep `python/pto_wsp/kernel.py` as canonical and remove duplicate things (ask /codex for final decisions).

*2.2*: Use real typing. Ask /codex to carefully design.

*2.3*, *2.4*, *2.5*: improve as recommended.

*2.6*: fix the binding.

For *3.x* items, I agree with the reviews and recommendation.

*4.1* : we should deliver typing things down to C++ compiler.

*4.2* : Option A is good.

*4.3* : We don't need the python executor anymore. But the IR bridge must pass all the things down to C++ to observe. This must be coherent.

Take advice for *5.x*.

*6.1* : the requirements explicitly demands variable-length. I think our design provide the capability. Why not supported in C++ lowering? /codex do analysis. Also, I think our workload design can describe tier-based kernel specialization. For first-class IR support for "descriptor arrays' or segmented KV chunks, I think they can still be represented by existing features. the flashinfer_decode example must provide evidence.

*6.2* : recommendation is right.

*6.3* : same issue as *1.4*, we need better compiler base.

All improvements must be done in v9, since they are all v9's implementation/design issues. No defering is allowed.

---

## Summary: How Comments Influenced v9 Design

### Major Design Shifts

1. **Declarative style** (R1): Removed decorator style, use `for b in P(axis)` syntax
2. **JIT kernels** (R7, L3): Replaced string-based refs with typed `pto.*` primitives
3. **Combinator schedule** (C1): `.dispatch().streams()` chain instead of fluent style
4. **Layout types** (R10): Dato-style refinement types, not schedule primitives
5. **Task graph** (R9): `.task_graph()` as alternative to streams

### User Decisions Applied

| Item | Decision |
|------|----------|
| Version | v9.3 design, `0.1.0` Python package |
| License | MIT with LICENSE-3RD-PARTY.md for pto-isa CANN |
| Execution | Option A - C++ backend required |
| Namespace | `pto` instead of `tl` |
| Schedule | Only `dispatch_threshold` kept (later removed) |

### Requirements Traceability

All requirements implemented and verified:
- **R1-R11**: Core design requirements (declarative style, JIT, layout types, task graph)
- **C1-C2**: Combinator schedule, pto-isa as 3rdparty
- **N1-N4**: End-to-end examples, C++ file structure, features.md
- **L1-L12**: Document organization, feature gaps, kernel JIT, test quality, documentation
