The v8 version is to some extent satisfiable. But we need to continue the iteration.

1. We have new references: references/pto-isa-lh, references/pto-isa-wc, references/allo (references/dato.pdf).
   1. pto-isa-lh is a reference for accurate cpu sim; pto-isa-wc is reference for npu target support. They are similar and provide some backend files to learn. Also, it provides mapping between python builder and assembly format, which we should learn. For the task graph data structure, we should make our schedule mechanism have the capability to cover or describe. Note that, we shouldn't become pto-isa-lh. It is not as powerful as ours. We should make sure to support it (perhaps with extension), but should keep ourselves.
   2. allo (also named dato in the paper) is a reference for targetting spatial architecture. I have ambition to make pto-isa's runtime multi-backend. I think our data-parallel+CSP workload description is powerful enough. The schedule primitives may need more extension to be suitable for the spatial architecture. This is what next version should carefully think and solve.
2. Detailed requirements:
   1. keep the oveall paradigm design in v8 version;
   2. new compiler stack: instead of C++ functions/classes for C++ JIT, we want python builder + C++ IR + assembly printer/parser + backend generator;
   3. prototyping should support three backends: cpu sim (for this backend, ) + npu backend + amd aie backend