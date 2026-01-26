# Linear Layout Research Note (L8)

Based on: [Linear Layouts: Robust Code Generation of Efficient Tensor Computation Using F₂](https://arxiv.org/abs/2505.23819)

Authors: Keren Zhou, Mario Lezcano, et al. (October 2025)

## Overview

Linear Layouts is a novel approach to modeling tensor layouts using linear algebra over **𝔽₂** (the binary field). Instead of case-by-case layout handling (which caused 12% of Triton's GitHub issues), this framework represents layouts as binary matrices acting on hardware resource bits.

## Key Technical Ideas

### 1. Binary Matrix Representation

A layout is a linear transformation `w = Av` where:
- `v` = input vector (hardware resource indices: register, thread, warp)
- `A` = layout matrix (binary)
- `w` = output vector (tensor coordinates)

Since GPU parameters are powers of two, bit-level operations (XOR, AND) naturally express layout structure.

### 2. Basis Vectors

The key insight: **A complete layout can be specified using only a few basis vectors** rather than enumerating all possible mappings.

For example, a blocked layout distributing 256 elements across 32 threads:
```
Thread = [0,0,0,0,0,1,1,1]  // bits 5-7 of element index
Element = [1,0,0,0,0,0,0,0] // bits 0-4 within thread's block
         [0,1,0,0,0,0,0,0]
         [0,0,1,0,0,0,0,0]
         ...
```

### 3. Automatic Swizzling

Algorithm to compute optimal swizzled memory layouts:
1. Define vectorization set V based on common register/bank dimensions
2. Construct subspace P combining memory bank constraints with thread distribution
3. Find largest subspace H where `P ∩ span(H) = {0}` (no bank conflicts)
4. Select index columns from H∪C to minimize unavoidable conflicts

**Result**: Provably maximal vectorization with minimal bank conflicts.

### 4. Warp-Shuffle Generation

For layout conversion A → B using `B⁻¹A` decomposition:
1. Compute vectorization size `n = |A_Reg ∩ B_Reg|`
2. Identify unchanged threads: `I = A_Thr ∩ B_Thr`
3. Compute XOR combinations: `G = {eᵢ ⊕ fᵢ}` for shuffle partners
4. Execute `2^|R|` shuffle rounds

### 5. Layout Propagation Rules

Distributed layouts are **closed under Triton shape operations** (transpose, reshape, join, split, expand_dims, broadcast). This enables:
- Forward/backward propagation without explicit conversions
- Automatic layout selection for operation chains
- Layout composition to eliminate data movement

## Relevance to PTO-RT

### Current Approach

PTO-RT's `MemLayout` in `types.py` uses an enum-based approach:
```python
class MemLayout(Enum):
    RowMajor = "row_major"
    ColMajor = "col_major"
    BlockedCOO = "blocked_coo"
```

This is sufficient for describing logical tensor layout but doesn't capture hardware-level distribution.

### Potential Integration

For NPU kernel optimization, Linear Layout concepts could help with:

1. **Ascend Cube/Vector tile programming**: Represent tile layouts as F₂ matrices for automatic swizzling and bank conflict avoidance

2. **Multi-core distribution**: Express how tiles distribute across AICore compute units

3. **Memory hierarchy**: Model UB/L1/HBM transfers as layout transformations

### Implementation Considerations

1. **Scope**: Linear Layout targets GPU-specific operations (warp shuffle, shared memory banking). Ascend NPU has different primitives (data mover, cube unit).

2. **Complexity vs. benefit**: For PTO-RT v9, enum-based layouts suffice. Linear Layout would be valuable for NPU kernel auto-tuning in future versions.

3. **Type-level integration**: Could extend `TensorLayout` to include F₂ matrix representation alongside distribution info.

## Conclusion

Linear Layout is a powerful formalism for GPU kernel optimization. For PTO-RT:

- **v9**: Document awareness; enum-based layouts adequate
- **Future**: Consider integration for NPU auto-tuning when targeting performance-critical kernels

## References

- [arXiv:2505.23819](https://arxiv.org/abs/2505.23819) - Linear Layouts paper
- [ML-Triton](https://arxiv.org/pdf/2503.14985) - Multi-Level Compilation for Triton
- [LEGO Layout Language](https://arxiv.org/html/2505.08091) - Related work on hierarchical mapping
