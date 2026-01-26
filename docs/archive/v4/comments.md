1. Plan should be excluded: I think the plan phase should be determined by programmers with their flexible coding preference. We don't need to provide abstraction for them. 
2. Generality: existing solution is too specific: WorkPlanner, PlanConfig, TierConfig, Tier, Accessors, ... They are all designed for the flashinfer-like operators. They are not general. We need the ISA extension to be very general. The extension should cover:
    1. Composition of kernel call into the whole work.
    2. How work is partitioned into AICPUs (dispatch) and then offloaded onto AICores (issue).
    3. Dependencies and synchronization between splited work (tasks). This is also described in @docs/uv/requirements.md (边执行边管理依赖). You should web search CUDA stream and events, and related techniques like work stealing.
How we provide the generality similar to C++'s language features while giving high-level abstraction to do analysis and optimization?
3. Data-Dependent must be supported: MoE in @docs/uv/requirements.md is another critical use case. This involves tasks' dynamic generation.
4. Stitching must be supported. This involves tasks' issueing strategies. Also, putting things across batches into one AICPU also involves the works' dispatching.
5. Tensor or memory resources are first-order citizens in the extension. This is similar to the pto-isa's existing constructs.
6. For the cold start or head overhead of stitching in @docs/uv/requirements.md, we should learn about "pipeline across instruction boundaries" from @references/megakernels.pdf, whose code is in references/Megakernels. You should first read the pdf and the code repo (docs, codes) carefully to understand all techniques involved. They are very inspiring to our design. You should write notes for them under @docs/uv/research
7. Treating @docs/uv/requirements.md with high priority. The FlashInfer's methodology is just one solution to support by the new extension, not everything. Solving the problems in the requirements document is more fundamental.

Based on the comments above and existing research notes and prior design, draft a new version of solution after very careful and comprehensive exploration and thinking. You can discuss with me through multi rounds of chatting.

