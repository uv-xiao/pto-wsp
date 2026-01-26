"""
Tests for Linear Layout (L8) - F₂ Binary Matrix Representation.

These tests verify the linear layout implementation based on arXiv:2505.23819.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))

import pytest
from pto_wsp.linear_layout import (
    LinearLayout,
    f2_dot,
    f2_matmul,
    f2_matmul_mat,
    f2_transpose,
    f2_rank,
    f2_kernel,
    propagate_transpose,
    propagate_reshape,
    propagate_broadcast,
    to_tensor_layout,
    from_tensor_layout,
)


# ============================================================
# F₂ Arithmetic Tests
# ============================================================

class TestF2Arithmetic:
    """Test F₂ (binary field) arithmetic operations."""

    def test_f2_dot_basic(self):
        """Test dot product over F₂."""
        a = (1, 0, 1, 0)
        b = (1, 1, 0, 0)
        # 1&1 ^ 0&1 ^ 1&0 ^ 0&0 = 1 ^ 0 ^ 0 ^ 0 = 1
        assert f2_dot(a, b) == 1

    def test_f2_dot_zero(self):
        """Test dot product with zeros."""
        a = (0, 0, 0, 0)
        b = (1, 1, 1, 1)
        assert f2_dot(a, b) == 0

    def test_f2_dot_self_cancel(self):
        """Test that x XOR x = 0."""
        a = (1, 1, 1, 1)
        b = (1, 1, 1, 1)
        # 1&1 ^ 1&1 ^ 1&1 ^ 1&1 = 1 ^ 1 ^ 1 ^ 1 = 0
        assert f2_dot(a, b) == 0

    def test_f2_matmul_identity(self):
        """Test matrix-vector multiply with identity matrix."""
        identity = [
            (1, 0, 0),
            (0, 1, 0),
            (0, 0, 1),
        ]
        v = (1, 0, 1)
        result = f2_matmul(identity, v)
        assert result == (1, 0, 1)

    def test_f2_matmul_permutation(self):
        """Test matrix-vector multiply with permutation matrix."""
        # Permutation: swap first two elements
        perm = [
            (0, 1, 0),
            (1, 0, 0),
            (0, 0, 1),
        ]
        v = (1, 0, 1)
        result = f2_matmul(perm, v)
        assert result == (0, 1, 1)  # Swapped

    def test_f2_matmul_mat_identity(self):
        """Test matrix-matrix multiply with identity."""
        A = [(1, 0), (0, 1)]
        B = [(1, 1), (0, 1)]
        result = f2_matmul_mat(A, B)
        assert result == [(1, 1), (0, 1)]

    def test_f2_transpose(self):
        """Test matrix transpose."""
        A = [(1, 0, 1), (0, 1, 0)]
        result = f2_transpose(A)
        assert result == [(1, 0), (0, 1), (1, 0)]

    def test_f2_rank_full(self):
        """Test rank of full-rank matrix."""
        A = [(1, 0), (0, 1)]
        assert f2_rank(A) == 2

    def test_f2_rank_deficient(self):
        """Test rank of rank-deficient matrix."""
        # Rows are linearly dependent over F₂
        A = [(1, 1), (1, 1)]
        assert f2_rank(A) == 1

    def test_f2_kernel_empty(self):
        """Test kernel of full-rank matrix."""
        A = [(1, 0), (0, 1)]
        kernel = f2_kernel(A)
        assert kernel == []  # No non-trivial kernel

    def test_f2_kernel_nonempty(self):
        """Test kernel of rank-deficient matrix."""
        # Matrix [1 1] has kernel {(1,1)}
        A = [(1, 1)]
        kernel = f2_kernel(A)
        assert len(kernel) == 1
        # Verify kernel vector is in null space
        for v in kernel:
            result = f2_matmul(A, v)
            assert all(r == 0 for r in result)


# ============================================================
# LinearLayout Basic Tests
# ============================================================

class TestLinearLayoutBasic:
    """Test basic LinearLayout functionality."""

    def test_identity_layout(self):
        """Test identity layout creation."""
        layout = LinearLayout.identity(4, "x")

        assert layout.total_in_bits == 4
        assert layout.total_out_bits == 4
        assert layout.in_dims == ["x"]
        assert layout.out_dims == ["x"]
        assert layout.rank() == 4
        assert layout.is_injective()
        assert layout.is_surjective()

    def test_identity_apply_index(self):
        """Test identity layout preserves indices."""
        layout = LinearLayout.identity(8, "x")

        for idx in range(256):  # 2^8 = 256
            assert layout.apply_index(idx) == idx

    def test_blocked_layout(self):
        """Test blocked layout creation."""
        # 256 elements, 32 threads, 8 elements per thread
        layout = LinearLayout.blocked(
            total_elements=256,
            num_threads=32,
            block_size=8
        )

        assert layout.in_dims == ["thread", "local"]
        assert layout.out_dims == ["element"]
        assert layout.in_bits == [5, 3]  # 32 threads, 8 local
        assert layout.out_bits == [8]  # 256 elements

    def test_blocked_layout_mapping(self):
        """Test blocked layout maps correctly."""
        # 64 elements, 8 threads, 8 elements per thread
        layout = LinearLayout.blocked(
            total_elements=64,
            num_threads=8,
            block_size=8
        )

        # Thread 0, local 0 -> element 0
        # Thread 1, local 0 -> element 8
        # Thread 0, local 1 -> element 1

        # Apply to thread=0, local=0 (index 0 in packed form)
        assert layout.apply_index(0) == 0

        # Apply to thread=1, local=0 (index 1 in packed form: thread bits first)
        assert layout.apply_index(1) == 8

        # Apply to thread=0, local=1 (index 8 in packed form: local bits after thread)
        assert layout.apply_index(8) == 1

    def test_strided_layout(self):
        """Test strided layout creation."""
        layout = LinearLayout.strided(
            total_elements=64,
            num_threads=8
        )

        assert layout.in_dims == ["thread", "local"]
        assert layout.out_dims == ["element"]

    def test_strided_layout_mapping(self):
        """Test strided layout maps correctly."""
        # 64 elements, 8 threads
        layout = LinearLayout.strided(
            total_elements=64,
            num_threads=8
        )

        # Thread 0, local 0 -> element 0
        # Thread 1, local 0 -> element 1
        # Thread 0, local 1 -> element 8

        assert layout.apply_index(0) == 0  # thread=0, local=0
        assert layout.apply_index(1) == 1  # thread=1, local=0

    def test_row_major_layout(self):
        """Test row-major layout creation."""
        layout = LinearLayout.row_major((4, 8))

        # from_strides creates a layout mapping logical dims to linear address
        assert len(layout.in_dims) == 2  # Two input dims (dim0, dim1)
        assert layout.total_in_bits == 5  # 4*8 = 32 = 2^5

    def test_col_major_layout(self):
        """Test column-major layout creation."""
        layout = LinearLayout.col_major((4, 8))

        assert len(layout.in_dims) == 2  # Two input dims
        assert layout.total_in_bits == 5


# ============================================================
# LinearLayout Composition Tests
# ============================================================

class TestLinearLayoutComposition:
    """Test LinearLayout composition operations."""

    def test_compose_identities(self):
        """Test composing two identity layouts."""
        id1 = LinearLayout.identity(4, "x")
        id2 = LinearLayout.identity(4, "y")

        # Need compatible dimensions
        composed = id1.compose(id2)
        assert composed.rank() == 4

    def test_transpose_dims(self):
        """Test dimension permutation."""
        # Create 2D layout
        layout = LinearLayout(
            matrix=[
                (1, 0, 0, 0),  # out[0] = in[0]
                (0, 1, 0, 0),  # out[1] = in[1]
                (0, 0, 1, 0),  # out[2] = in[2]
                (0, 0, 0, 1),  # out[3] = in[3]
            ],
            in_dims=["row", "col"],
            out_dims=["dim0", "dim1"],
            in_bits=[2, 2],
            out_bits=[2, 2]
        )

        transposed = layout.transpose_dims((1, 0))

        assert transposed.out_dims == ["dim1", "dim0"]
        assert transposed.out_bits == [2, 2]


# ============================================================
# Layout Propagation Tests
# ============================================================

class TestLayoutPropagation:
    """Test layout propagation through operations."""

    def test_propagate_transpose(self):
        """Test transpose propagation."""
        layout = LinearLayout(
            matrix=[
                (1, 0, 0, 0),
                (0, 1, 0, 0),
                (0, 0, 1, 0),
                (0, 0, 0, 1),
            ],
            in_dims=["row", "col"],
            out_dims=["x", "y"],
            in_bits=[2, 2],
            out_bits=[2, 2]
        )

        transposed = propagate_transpose(layout, (1, 0))

        assert transposed.out_dims == ["y", "x"]

    def test_propagate_reshape(self):
        """Test reshape propagation."""
        layout = LinearLayout.row_major((4, 4))

        reshaped = propagate_reshape(layout, (4, 4), (2, 8))

        assert len(reshaped.out_dims) == 2

    def test_propagate_reshape_invalid(self):
        """Test reshape with incompatible shapes."""
        layout = LinearLayout.row_major((4, 4))

        with pytest.raises(ValueError, match="Cannot reshape"):
            propagate_reshape(layout, (4, 4), (2, 4))  # 16 != 8

    def test_propagate_broadcast(self):
        """Test broadcast propagation."""
        layout = LinearLayout(
            matrix=[(1, 0), (0, 1)],
            in_dims=["x"],
            out_dims=["dim0"],
            in_bits=[2],
            out_bits=[2]
        )

        broadcasted = propagate_broadcast(layout, (0,))

        assert len(broadcasted.out_dims) == 2
        assert "broadcast0" in broadcasted.out_dims


# ============================================================
# Swizzling Tests
# ============================================================

class TestSwizzling:
    """Test swizzling for bank conflict avoidance."""

    def test_compute_swizzle(self):
        """Test swizzle computation."""
        layout = LinearLayout.blocked(
            total_elements=256,
            num_threads=32,
            block_size=8
        )

        swizzled = LinearLayout.compute_swizzle(
            layout,
            bank_bits=5,  # 32 banks
            vector_bits=4  # 128-bit vectors
        )

        # Swizzled layout should have same dimensions
        assert swizzled.in_dims == layout.in_dims
        assert swizzled.out_dims == layout.out_dims

        # But different matrix
        assert swizzled.matrix != layout.matrix


# ============================================================
# Integration Tests
# ============================================================

class TestLinearLayoutIntegration:
    """Test integration with TensorLayout from types.py."""

    def test_to_tensor_layout(self):
        """Test conversion to TensorLayout."""
        linear = LinearLayout.row_major((4, 8))
        tensor_layout = to_tensor_layout(linear)

        assert tensor_layout is not None
        # Output dims determine distribution rank
        assert len(tensor_layout.dist) == len(linear.out_dims)
        assert tensor_layout.mem is not None

    def test_from_tensor_layout(self):
        """Test conversion from TensorLayout."""
        from pto_wsp.types import TensorLayout, TensorReplicate, MemLayout

        # Create TensorLayout with strides
        strides = (8, 1)  # Row-major for (4, 8)
        mem = MemLayout(strides=strides)
        tensor_layout = TensorLayout(
            dist=(TensorReplicate(), TensorReplicate()),
            mem=mem
        )

        linear = from_tensor_layout(tensor_layout, (4, 8))

        assert linear is not None
        assert linear.out_dims is not None


# ============================================================
# Property Tests
# ============================================================

class TestLinearLayoutProperties:
    """Test mathematical properties of linear layouts."""

    def test_injective_layout(self):
        """Test injective layout detection."""
        # Identity is injective
        layout = LinearLayout.identity(4, "x")
        assert layout.is_injective()

    def test_surjective_layout(self):
        """Test surjective layout detection."""
        # Identity is surjective
        layout = LinearLayout.identity(4, "x")
        assert layout.is_surjective()

    def test_rank_computation(self):
        """Test rank computation."""
        # Full rank identity
        layout = LinearLayout.identity(4, "x")
        assert layout.rank() == 4

    def test_repr(self):
        """Test string representation."""
        layout = LinearLayout.identity(4, "x")
        repr_str = repr(layout)

        assert "LinearLayout" in repr_str
        assert "x:4" in repr_str
        assert "rank=4" in repr_str


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
