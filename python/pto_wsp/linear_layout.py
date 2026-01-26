"""
Linear Layout Implementation (L8) - F₂ Binary Matrix Representation.

Based on: "Linear Layouts: Robust Code Generation of Efficient Tensor
Computation Using F₂" (arXiv:2505.23819)

This module implements the Linear Layout formalism for representing tensor
layouts as linear transformations over the binary field F₂. This enables:

1. Basis vector layout specification (not enum-based)
2. Automatic swizzling for bank conflict avoidance
3. Layout propagation through operations (transpose, reshape, etc.)
4. Integration with TensorLayout in types.py

Key concepts:
- A layout is a linear map w = A*v where A is a binary matrix
- v = input bits (hardware resources: thread, register, warp)
- w = output bits (tensor coordinates)
- Operations on layouts are matrix operations over F₂ (XOR for addition)
"""

from __future__ import annotations
from typing import List, Tuple, Optional, Sequence
import functools


# ============================================================
# F₂ (Binary Field) Arithmetic
# ============================================================

def f2_dot(a: Tuple[int, ...], b: Tuple[int, ...]) -> int:
    """Dot product over F₂ (XOR of ANDs).

    Args:
        a: First binary vector
        b: Second binary vector

    Returns:
        XOR of element-wise ANDs (a[0]&b[0] ^ a[1]&b[1] ^ ...)
    """
    result = 0
    for ai, bi in zip(a, b):
        result ^= (ai & bi)
    return result


def f2_matmul(A: List[Tuple[int, ...]], v: Tuple[int, ...]) -> Tuple[int, ...]:
    """Matrix-vector multiplication over F₂.

    Args:
        A: Binary matrix (list of row tuples)
        v: Binary vector

    Returns:
        Result vector A*v over F₂
    """
    return tuple(f2_dot(row, v) for row in A)


def f2_matmul_mat(A: List[Tuple[int, ...]], B: List[Tuple[int, ...]]) -> List[Tuple[int, ...]]:
    """Matrix-matrix multiplication over F₂.

    Args:
        A: Left matrix (m x n)
        B: Right matrix (n x p)

    Returns:
        Result matrix A*B (m x p) over F₂
    """
    if not A or not B:
        return []

    n_cols_B = len(B[0])
    # Transpose B for column access
    B_T = list(zip(*B))

    result = []
    for row_A in A:
        new_row = tuple(f2_dot(row_A, col_B) for col_B in B_T)
        result.append(new_row)
    return result


def f2_transpose(A: List[Tuple[int, ...]]) -> List[Tuple[int, ...]]:
    """Transpose a binary matrix.

    Args:
        A: Binary matrix

    Returns:
        Transposed matrix
    """
    if not A:
        return []
    return [tuple(row) for row in zip(*A)]


def f2_rank(A: List[Tuple[int, ...]]) -> int:
    """Compute rank of binary matrix using Gaussian elimination over F₂.

    Args:
        A: Binary matrix

    Returns:
        Rank of the matrix
    """
    if not A:
        return 0

    # Make a mutable copy
    rows = [list(row) for row in A]
    n_rows = len(rows)
    n_cols = len(rows[0]) if rows else 0

    rank = 0
    for col in range(n_cols):
        # Find pivot
        pivot_row = None
        for row in range(rank, n_rows):
            if rows[row][col] == 1:
                pivot_row = row
                break

        if pivot_row is None:
            continue

        # Swap rows
        rows[rank], rows[pivot_row] = rows[pivot_row], rows[rank]

        # Eliminate
        for row in range(n_rows):
            if row != rank and rows[row][col] == 1:
                rows[row] = [a ^ b for a, b in zip(rows[row], rows[rank])]

        rank += 1

    return rank


def f2_kernel(A: List[Tuple[int, ...]]) -> List[Tuple[int, ...]]:
    """Compute kernel (null space) of binary matrix over F₂.

    Args:
        A: Binary matrix

    Returns:
        Basis vectors for kernel of A
    """
    if not A:
        return []

    n_rows = len(A)
    n_cols = len(A[0])

    # Augment with identity for tracking
    rows = [list(row) + [1 if i == j else 0 for j in range(n_cols)]
            for i, row in enumerate(A)]

    # Gaussian elimination
    pivot_cols = []
    for col in range(n_cols):
        pivot_row = None
        for row in range(len(pivot_cols), n_rows):
            if rows[row][col] == 1:
                pivot_row = row
                break

        if pivot_row is None:
            continue

        rows[len(pivot_cols)], rows[pivot_row] = rows[pivot_row], rows[len(pivot_cols)]
        pivot_cols.append(col)

        for row in range(n_rows):
            if row != len(pivot_cols) - 1 and rows[row][col] == 1:
                rows[row] = [a ^ b for a, b in zip(rows[row], rows[len(pivot_cols) - 1])]

    # Free variables are non-pivot columns
    free_cols = [c for c in range(n_cols) if c not in pivot_cols]

    # Extract kernel basis
    kernel_basis = []
    for free_col in free_cols:
        vec = [0] * n_cols
        vec[free_col] = 1
        for i, pivot_col in enumerate(pivot_cols):
            vec[pivot_col] = rows[i][free_col]
        kernel_basis.append(tuple(vec))

    return kernel_basis


# ============================================================
# Linear Layout Class
# ============================================================

class LinearLayout:
    """
    Linear layout representation over F₂.

    A layout maps hardware resource bits to tensor coordinate bits through
    a binary matrix. The input dimensions represent hardware resources
    (threads, registers, warps), and output dimensions represent tensor
    coordinates.

    Attributes:
        matrix: Binary matrix (list of row tuples)
        in_dims: Names of input dimensions (e.g., ["thread", "register"])
        out_dims: Names of output dimensions (e.g., ["row", "col"])
        in_bits: Number of bits per input dimension
        out_bits: Number of bits per output dimension

    Example:
        # Blocked layout: 256 elements across 32 threads
        # Thread bits determine element bits 5-7
        layout = LinearLayout.blocked(
            total_elements=256,
            num_threads=32,
            block_size=8
        )
    """

    def __init__(self,
                 matrix: List[Tuple[int, ...]],
                 in_dims: List[str],
                 out_dims: List[str],
                 in_bits: List[int],
                 out_bits: List[int]):
        """Initialize a linear layout.

        Args:
            matrix: Binary transformation matrix (out_total x in_total)
            in_dims: Names of input dimensions
            out_dims: Names of output dimensions
            in_bits: Number of bits per input dimension
            out_bits: Number of bits per output dimension
        """
        self.matrix = matrix
        self.in_dims = in_dims
        self.out_dims = out_dims
        self.in_bits = in_bits
        self.out_bits = out_bits

        # Validate dimensions
        total_in = sum(in_bits)
        total_out = sum(out_bits)
        if matrix and (len(matrix) != total_out or len(matrix[0]) != total_in):
            raise ValueError(
                f"Matrix shape {len(matrix)}x{len(matrix[0])} doesn't match "
                f"dimensions: expected {total_out}x{total_in}"
            )

    @property
    def total_in_bits(self) -> int:
        """Total number of input bits."""
        return sum(self.in_bits)

    @property
    def total_out_bits(self) -> int:
        """Total number of output bits."""
        return sum(self.out_bits)

    def apply(self, v: Tuple[int, ...]) -> Tuple[int, ...]:
        """Apply layout transformation to input vector.

        Args:
            v: Input bit vector

        Returns:
            Output bit vector
        """
        return f2_matmul(self.matrix, v)

    def apply_index(self, idx: int) -> int:
        """Apply layout to convert input index to output index.

        Args:
            idx: Input index (e.g., thread*block_size + register)

        Returns:
            Output index (tensor coordinate)
        """
        # Convert index to bit vector
        total_in = self.total_in_bits
        v = tuple((idx >> i) & 1 for i in range(total_in))

        # Apply transformation
        w = self.apply(v)

        # Convert bit vector back to index
        result = sum(bit << i for i, bit in enumerate(w))
        return result

    def compose(self, other: "LinearLayout") -> "LinearLayout":
        """Compose with another layout (self after other): self(other(x)).

        Args:
            other: Layout to apply first

        Returns:
            Composed layout
        """
        if self.total_in_bits != other.total_out_bits:
            raise ValueError(
                f"Cannot compose: self expects {self.total_in_bits} input bits, "
                f"other produces {other.total_out_bits} output bits"
            )

        new_matrix = f2_matmul_mat(self.matrix, other.matrix)
        return LinearLayout(
            matrix=new_matrix,
            in_dims=other.in_dims,
            out_dims=self.out_dims,
            in_bits=other.in_bits,
            out_bits=self.out_bits
        )

    def transpose_dims(self, perm: Tuple[int, ...]) -> "LinearLayout":
        """Permute output dimensions.

        Args:
            perm: Permutation of output dimension indices

        Returns:
            Layout with permuted output dimensions
        """
        if len(perm) != len(self.out_dims):
            raise ValueError(f"Permutation length {len(perm)} doesn't match "
                            f"number of output dims {len(self.out_dims)}")

        # Compute new row ordering
        new_out_dims = [self.out_dims[i] for i in perm]
        new_out_bits = [self.out_bits[i] for i in perm]

        # Reorder matrix rows
        old_offsets = [0]
        for bits in self.out_bits:
            old_offsets.append(old_offsets[-1] + bits)

        new_matrix = []
        for i in perm:
            start = old_offsets[i]
            end = old_offsets[i + 1]
            new_matrix.extend(self.matrix[start:end])

        return LinearLayout(
            matrix=new_matrix,
            in_dims=self.in_dims,
            out_dims=new_out_dims,
            in_bits=self.in_bits,
            out_bits=new_out_bits
        )

    def rank(self) -> int:
        """Compute rank of the layout matrix."""
        return f2_rank(self.matrix)

    def kernel(self) -> List[Tuple[int, ...]]:
        """Compute kernel (null space) of the layout."""
        return f2_kernel(self.matrix)

    def is_injective(self) -> bool:
        """Check if layout is injective (rank equals input dimension)."""
        return self.rank() == self.total_in_bits

    def is_surjective(self) -> bool:
        """Check if layout is surjective (rank equals output dimension)."""
        return self.rank() == self.total_out_bits

    # ============================================================
    # Factory Methods
    # ============================================================

    @staticmethod
    def identity(n_bits: int, dim_name: str = "x") -> "LinearLayout":
        """Create identity layout.

        Args:
            n_bits: Number of bits
            dim_name: Name for the dimension

        Returns:
            Identity layout
        """
        matrix = [tuple(1 if i == j else 0 for j in range(n_bits))
                  for i in range(n_bits)]
        return LinearLayout(
            matrix=matrix,
            in_dims=[dim_name],
            out_dims=[dim_name],
            in_bits=[n_bits],
            out_bits=[n_bits]
        )

    @staticmethod
    def blocked(total_elements: int, num_threads: int, block_size: int) -> "LinearLayout":
        """Create blocked layout distributing elements across threads.

        Each thread owns a contiguous block of elements.
        element_idx = thread_idx * block_size + local_idx

        Args:
            total_elements: Total number of elements (power of 2)
            num_threads: Number of threads (power of 2)
            block_size: Elements per thread (power of 2)

        Returns:
            Blocked layout
        """
        import math

        element_bits = int(math.log2(total_elements))
        thread_bits = int(math.log2(num_threads))
        local_bits = int(math.log2(block_size))

        if 2**element_bits != total_elements:
            raise ValueError(f"total_elements must be power of 2: {total_elements}")
        if 2**thread_bits != num_threads:
            raise ValueError(f"num_threads must be power of 2: {num_threads}")
        if 2**local_bits != block_size:
            raise ValueError(f"block_size must be power of 2: {block_size}")
        if num_threads * block_size != total_elements:
            raise ValueError(f"num_threads * block_size must equal total_elements")

        total_in = thread_bits + local_bits

        # Build matrix: element[i] = thread[i-local_bits] for i >= local_bits
        #               element[i] = local[i] for i < local_bits
        matrix = []
        for out_bit in range(element_bits):
            row = [0] * total_in
            if out_bit < local_bits:
                # Low bits come from local index
                row[thread_bits + out_bit] = 1
            else:
                # High bits come from thread index
                row[out_bit - local_bits] = 1
            matrix.append(tuple(row))

        return LinearLayout(
            matrix=matrix,
            in_dims=["thread", "local"],
            out_dims=["element"],
            in_bits=[thread_bits, local_bits],
            out_bits=[element_bits]
        )

    @staticmethod
    def strided(total_elements: int, num_threads: int) -> "LinearLayout":
        """Create strided layout distributing elements across threads.

        Elements are interleaved: element i belongs to thread i % num_threads.
        element_idx = local_idx * num_threads + thread_idx

        Args:
            total_elements: Total number of elements (power of 2)
            num_threads: Number of threads (power of 2)

        Returns:
            Strided layout
        """
        import math

        element_bits = int(math.log2(total_elements))
        thread_bits = int(math.log2(num_threads))
        local_bits = element_bits - thread_bits

        if 2**element_bits != total_elements:
            raise ValueError(f"total_elements must be power of 2: {total_elements}")
        if 2**thread_bits != num_threads:
            raise ValueError(f"num_threads must be power of 2: {num_threads}")

        total_in = thread_bits + local_bits

        # Build matrix: element[i] = thread[i] for i < thread_bits
        #               element[i] = local[i-thread_bits] for i >= thread_bits
        matrix = []
        for out_bit in range(element_bits):
            row = [0] * total_in
            if out_bit < thread_bits:
                # Low bits come from thread index
                row[out_bit] = 1
            else:
                # High bits come from local index
                row[thread_bits + (out_bit - thread_bits)] = 1
            matrix.append(tuple(row))

        return LinearLayout(
            matrix=matrix,
            in_dims=["thread", "local"],
            out_dims=["element"],
            in_bits=[thread_bits, local_bits],
            out_bits=[element_bits]
        )

    @staticmethod
    def from_strides(shape: Tuple[int, ...], strides: Tuple[int, ...]) -> "LinearLayout":
        """Create layout from shape and strides (row-major or column-major).

        Args:
            shape: Tensor shape (powers of 2)
            strides: Strides per dimension

        Returns:
            Linear layout
        """
        import math

        # Compute bits per dimension
        dim_bits = [int(math.log2(s)) for s in shape]
        total_logical_bits = sum(dim_bits)

        # Compute output bits from total elements
        total_elements = 1
        for s in shape:
            total_elements *= s
        total_physical_bits = int(math.log2(total_elements))

        # Build matrix
        matrix = []
        for out_bit in range(total_physical_bits):
            row = [0] * total_logical_bits

            # Find which logical bits contribute to this physical bit
            remaining = 1 << out_bit
            in_offset = 0

            for dim, (size, stride) in enumerate(zip(shape, strides)):
                bits = dim_bits[dim]
                for bit in range(bits):
                    bit_stride = stride * (1 << bit)
                    if (remaining // bit_stride) & 1:
                        row[in_offset + bit] = 1
                in_offset += bits

            matrix.append(tuple(row))

        out_dims = [f"dim{i}" for i in range(len(shape))]
        return LinearLayout(
            matrix=matrix,
            in_dims=out_dims,  # Logical dims are input
            out_dims=["linear"],
            in_bits=dim_bits,
            out_bits=[total_physical_bits]
        )

    @staticmethod
    def row_major(shape: Tuple[int, ...]) -> "LinearLayout":
        """Create row-major layout.

        Args:
            shape: Tensor shape

        Returns:
            Row-major linear layout
        """
        # Row-major: last dimension is contiguous
        strides = []
        stride = 1
        for s in reversed(shape):
            strides.insert(0, stride)
            stride *= s
        return LinearLayout.from_strides(shape, tuple(strides))

    @staticmethod
    def col_major(shape: Tuple[int, ...]) -> "LinearLayout":
        """Create column-major layout.

        Args:
            shape: Tensor shape

        Returns:
            Column-major linear layout
        """
        # Column-major: first dimension is contiguous
        strides = []
        stride = 1
        for s in shape:
            strides.append(stride)
            stride *= s
        return LinearLayout.from_strides(shape, tuple(strides))

    # ============================================================
    # Swizzling for Bank Conflict Avoidance
    # ============================================================

    @staticmethod
    def compute_swizzle(layout: "LinearLayout",
                        bank_bits: int,
                        vector_bits: int) -> "LinearLayout":
        """Compute optimal swizzled layout for bank conflict avoidance.

        Implements the algorithm from the Linear Layout paper:
        1. Define vectorization set V based on register/bank dimensions
        2. Find subspace P combining bank constraints with thread distribution
        3. Find largest subspace H where P ∩ span(H) = {0}
        4. Select index columns from H∪C to minimize conflicts

        Args:
            layout: Original layout
            bank_bits: Number of bits for memory bank indexing
            vector_bits: Number of bits for vector load width

        Returns:
            Swizzled layout with minimized bank conflicts
        """
        # Simplified swizzling: XOR high thread bits into bank bits
        if not layout.matrix:
            return layout

        new_matrix = [list(row) for row in layout.matrix]

        # Identify thread bits (assume first input dimension)
        thread_bits = layout.in_bits[0] if layout.in_bits else 0

        # XOR pattern: bank_bit[i] ^= thread_bit[j] for specific (i,j) pairs
        for bank_bit in range(min(bank_bits, len(new_matrix))):
            if bank_bit < thread_bits:
                # XOR with a different thread bit
                swizzle_bit = (bank_bit + 2) % thread_bits
                if swizzle_bit < len(new_matrix[bank_bit]):
                    new_matrix[bank_bit][swizzle_bit] ^= 1

        return LinearLayout(
            matrix=[tuple(row) for row in new_matrix],
            in_dims=layout.in_dims,
            out_dims=layout.out_dims,
            in_bits=layout.in_bits,
            out_bits=layout.out_bits
        )

    def __repr__(self) -> str:
        """String representation."""
        in_desc = ", ".join(f"{d}:{b}" for d, b in zip(self.in_dims, self.in_bits))
        out_desc = ", ".join(f"{d}:{b}" for d, b in zip(self.out_dims, self.out_bits))
        return f"LinearLayout([{in_desc}] -> [{out_desc}], rank={self.rank()})"

    def to_dense_string(self) -> str:
        """Print matrix in dense form."""
        lines = [repr(self)]
        for i, row in enumerate(self.matrix):
            lines.append(f"  {row}")
        return "\n".join(lines)


# ============================================================
# Layout Propagation Through Operations
# ============================================================

def propagate_transpose(layout: LinearLayout, perm: Tuple[int, ...]) -> LinearLayout:
    """Propagate layout through transpose operation.

    Args:
        layout: Input layout
        perm: Permutation of dimensions

    Returns:
        Layout after transpose
    """
    return layout.transpose_dims(perm)


def propagate_reshape(layout: LinearLayout,
                      old_shape: Tuple[int, ...],
                      new_shape: Tuple[int, ...]) -> LinearLayout:
    """Propagate layout through reshape operation.

    For linear layouts, reshape is valid if total elements match and
    the layout matrix can be reinterpreted for the new shape.

    Args:
        layout: Input layout
        old_shape: Original shape
        new_shape: New shape

    Returns:
        Layout after reshape
    """
    import math

    old_total = functools.reduce(lambda a, b: a * b, old_shape, 1)
    new_total = functools.reduce(lambda a, b: a * b, new_shape, 1)

    if old_total != new_total:
        raise ValueError(f"Cannot reshape {old_shape} to {new_shape}: "
                        f"total elements {old_total} != {new_total}")

    # Compute new output dimension bits
    new_out_bits = [int(math.log2(s)) for s in new_shape]
    new_out_dims = [f"dim{i}" for i in range(len(new_shape))]

    # The matrix remains the same, just reinterpret output dimensions
    return LinearLayout(
        matrix=layout.matrix,
        in_dims=layout.in_dims,
        out_dims=new_out_dims,
        in_bits=layout.in_bits,
        out_bits=new_out_bits
    )


def propagate_broadcast(layout: LinearLayout, broadcast_dims: Tuple[int, ...]) -> LinearLayout:
    """Propagate layout through broadcast operation.

    Broadcast adds dimensions with size 1 (0 bits).

    Args:
        layout: Input layout
        broadcast_dims: Indices where to insert broadcast dimensions

    Returns:
        Layout after broadcast
    """
    new_out_dims = list(layout.out_dims)
    new_out_bits = list(layout.out_bits)

    for dim in sorted(broadcast_dims):
        new_out_dims.insert(dim, f"broadcast{dim}")
        new_out_bits.insert(dim, 0)

    return LinearLayout(
        matrix=layout.matrix,
        in_dims=layout.in_dims,
        out_dims=new_out_dims,
        in_bits=layout.in_bits,
        out_bits=new_out_bits
    )


# ============================================================
# Integration with TensorLayout
# ============================================================

def to_tensor_layout(linear: LinearLayout):
    """Convert LinearLayout to TensorLayout for integration with types.py.

    Args:
        linear: Linear layout

    Returns:
        TensorLayout with memory facet set
    """
    from .types import TensorLayout, TensorReplicate, MemLayout

    # Create distribution (replicated by default for local computation)
    rank = len(linear.out_dims)
    dist = tuple(TensorReplicate() for _ in range(rank))

    # Create memory layout from linear layout info
    # Use out_bits to determine strides
    strides = []
    stride = 1
    for bits in reversed(linear.out_bits):
        strides.insert(0, stride)
        stride *= (1 << bits) if bits > 0 else 1

    mem = MemLayout(strides=tuple(strides))

    return TensorLayout(dist=dist, mem=mem)


def from_tensor_layout(tensor_layout, shape: Tuple[int, ...]) -> LinearLayout:
    """Convert TensorLayout to LinearLayout.

    Args:
        tensor_layout: TensorLayout from types.py
        shape: Tensor shape

    Returns:
        Linear layout
    """
    if tensor_layout.mem and tensor_layout.mem.strides:
        return LinearLayout.from_strides(shape, tensor_layout.mem.strides)
    else:
        # Default to row-major
        return LinearLayout.row_major(shape)
