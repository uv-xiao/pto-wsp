# REDO V10

I've review the v10 design by myself. And I found it not satisfying. Let's redo the design from scratch. 

## HOW TO?

We should create the docs/future-htp folder to hold the redo design. The principles of the redo process is listed below. You should generate: 

- analysis.md: motivation and fundamentail methodology
- features.md: feature summary and rationale discussion 
- implementations.md: architecture design and explanation
- examples.md: full examples (from kernel/megakernel/serving-routine programs to different backends with pipeline and passes) for demo.
- impls/ folder to hold more implementation details of every component
- feats/ folder to hold detailed explanation of every feature

You can learn from docs/future, but you must do fundamental REDO, not incremental fixes from v10 design. You must conduct very careful thinking and analysis as experts in related domains (PL, computer architect, LLM performance engineer, compiler, ...) For related work, you need to carefully read paper and code and reports.

This is a very large design work. You should do it with huge patience and frequent review. 

## The most important thing: POSITION

PTO-WSP = Parallel Tile Operations - Workload-Schedule Programming. Where, PTO is an ISA provided by Huawei Ascend CANN toolchain, and WSP is a programming paradigm. However, the goal of this project is to ease the programming of kernel and things above kernels (megakernels, or even serving routine) with different backend targets. The original v9 design which match the name PTO-WSP is completely outdated, failing to match the ambitions.

## RENAME

We need to rename the project to reflect the ambition. I think Heterogeneous Tile Programming (HTP) is a good candidate. 

## COMPONENTS

The key components of HTP is:

1. Programming entry: native Python.
2. Intermediate repr: Python AST (AST mutator as passes) and optional more compilation pipeline (like MLIR + passes). Python AST is the entry: any optional/additional compilation pipeline needs to be started from AST mutator (match + apply).
3. Codegen: generate artifacts towards different backends (Ascend a2a3sim / a2a3, AIE, ...)
4. Binding: start the compilation and load/run codegen artifacts.


## Extensibility is the most essential feature!

What extensibility we should support?

- Define new programming models, such as CSP and WSP. 
- Define backend-specific intrinsics: for example, define intrinsics for PTO ISA, and use them like `pto.add(tileA, tileB)`
- Attach layout typing to tile/tensor variables: for example, we should optionally attach Axe (docs/reference/20_axe.md) or Arknife (docs/reference/19_arknife.md); which should be used by passes and codegen in the pipeline targetting a specific hardware target.
- Add new passes (either Python AST mutator, or external MLIR pipeline triggered by AST mutator-based match-apply).
- Define new backend targets, including NPU architecture and codegen. 
- Define new pipeline to run specific passes and end by specific backend target's codegen and binding.
- ...


First, we need to provide the flexibility for extensions on every aspect. This requires very careful and expertise design of the framework architecture.

Second, which is also important and novel, is that all the extensible things are closely related. For example, both the CSP and the WSP programming model should be handled specifically by the a2a3sim backend, and the pipeline for the a2a3sim backend should only include passes for it, and the passes must handle the layout attached in the program, and the only the a2a3sim-compatible intrinsics should be included. So, what we need is a type system handling such complex dependencies.


## Must-support PROGRAMMING MODEL

### CSP

This should be an optional dialact to enable in programming, which should be supported by different backend targets.

### WSP

We should support use Python decorators to distinguish workload and schedule programming, and different backends should do specific handling (compilation) according to them.

## Must-support BACKENDS

### PTO

We must codegen artifacts to be further compiled by pto-runtime, for both a2a3sim and a2a3 backend support. Therefore, HTP should have similar position with the PyPTO (docs/reference/18_pypto.md) in terms of the PTO ecosystem.

### AIE

We must codegen artifacts to be compiled by mlir-aie, similar to what Allo's aie backend does and DATO does.

