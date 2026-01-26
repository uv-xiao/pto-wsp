"""
Type system for PTO Workload-Schedule Programming (PTO-WSP) framework.

Axis types describe iteration spaces.
"""

from __future__ import annotations
from typing import Generic, TypeVar, Any
from enum import Enum

N = TypeVar("N", bound=int)


class Dense(Generic[N]):
    """Static size axis.

    Usage:
        heads = Dense[8]()  # 8 elements, known at compile time
    """
    _size: int

    def __class_getitem__(cls, n: int) -> type:
        """Create Dense type with static size."""
        class DenseN(Dense):
            _size = n
        return DenseN

    def __init__(self):
        pass

    @property
    def size(self) -> int:
        return self._size


class DenseDyn:
    """Dynamic size axis.

    Usage:
        batch = DenseDyn(batch_size)  # Runtime size
    """
    def __init__(self, size: int):
        self._size = size

    @property
    def size(self) -> int:
        return self._size


class Ragged:
    """Variable length per outer element.

    Usage:
        tokens = Ragged(batch_size, lengths)  # lengths[i] = tokens in batch i
    """
    def __init__(self, outer_size: int, lengths: list[int]):
        self._outer_size = outer_size
        self._lengths = lengths

    @property
    def outer_size(self) -> int:
        return self._outer_size

    def length(self, idx: int) -> int:
        return self._lengths[idx]

    def total(self) -> int:
        return sum(self._lengths)


class Sparse:
    """CSR format for sparse iteration.

    Usage:
        routing = Sparse(batch_size, indptr, indices)  # MoE routing
    """
    def __init__(self, outer_size: int, indptr: list[int], indices: list[int]):
        self._outer_size = outer_size
        self._indptr = indptr
        self._indices = indices

    @property
    def outer_size(self) -> int:
        return self._outer_size

    def nnz(self) -> int:
        return self._indptr[self._outer_size]

    def row_nnz(self, row: int) -> int:
        return self._indptr[row + 1] - self._indptr[row]

    def __getitem__(self, row: int) -> list[int]:
        """Get indices for a row."""
        start = self._indptr[row]
        end = self._indptr[row + 1]
        return self._indices[start:end]


class DType(Enum):
    """Data types."""
    F16 = "f16"
    BF16 = "bf16"
    F32 = "f32"
    F64 = "f64"
    I8 = "i8"
    I16 = "i16"
    I32 = "i32"
    I64 = "i64"
    U8 = "u8"
    U16 = "u16"
    U32 = "u32"
    U64 = "u64"


class Location(Enum):
    """Memory location."""
    Global = "global"   # Global memory (HBM)
    L2 = "l2"          # L2 cache
    UB = "ub"          # Unified Buffer (AICore local)
    L1 = "l1"          # L1 buffer


# ============================================================
# Memory Layout (Triton-style) - R10
# ============================================================

class MemLayout:
    """Physical memory layout (Triton/CuTe-style).

    Represents the physical arrangement of tensor data:
    - strides: Stride per dimension
    - order: Iteration order (which dims are contiguous/vectorized)
    - swizzle: Optional bank-conflict avoidance pattern
    - pack: Optional vector packing specification

    This is the "Memory facet" of the unified Layout type.
    """
    def __init__(self,
                 strides: tuple[int, ...] = None,
                 order: tuple[int, ...] = None,
                 swizzle: Any = None,
                 pack: Any = None):
        self.strides = strides
        self.order = order
        self.swizzle = swizzle
        self.pack = pack

    def compose(self, other: "MemLayout") -> "MemLayout":
        """Compose with another layout (tile/reshape/permute)."""
        # Simplified composition - in practice this would be more complex
        return MemLayout(
            strides=other.strides or self.strides,
            order=other.order or self.order,
            swizzle=other.swizzle or self.swizzle,
            pack=other.pack or self.pack
        )

    def permute(self, perm: tuple[int, ...]) -> "MemLayout":
        """Permute dimensions."""
        if self.strides:
            new_strides = tuple(self.strides[i] for i in perm)
            return MemLayout(strides=new_strides, order=self.order,
                            swizzle=self.swizzle, pack=self.pack)
        return self

    def tile(self, tile_shape: tuple[int, ...]) -> "MemLayout":
        """Add tiling to layout."""
        # Simplified - would compute new strides based on tile shape
        return MemLayout(strides=self.strides, order=self.order,
                        swizzle=self.swizzle, pack=self.pack)

    @staticmethod
    def row_major(shape: tuple[int, ...]) -> "MemLayout":
        """Create row-major (C-contiguous) layout."""
        strides = []
        stride = 1
        for dim in reversed(shape):
            strides.insert(0, stride)
            stride *= dim
        return MemLayout(strides=tuple(strides), order=tuple(range(len(shape))))

    @staticmethod
    def col_major(shape: tuple[int, ...]) -> "MemLayout":
        """Create column-major (Fortran-contiguous) layout."""
        strides = []
        stride = 1
        for dim in shape:
            strides.append(stride)
            stride *= dim
        return MemLayout(strides=tuple(strides), order=tuple(reversed(range(len(shape)))))

    def __repr__(self):
        parts = []
        if self.strides:
            parts.append(f"strides={self.strides}")
        if self.order:
            parts.append(f"order={self.order}")
        if self.swizzle:
            parts.append(f"swizzle={self.swizzle}")
        return f"MemLayout({', '.join(parts)})"


# ============================================================
# Distribution Types (Dato-style) - R10
# ============================================================

class DistElem:
    """Base class for distribution elements."""
    pass


class TensorReplicate(DistElem):
    """Full copy on each worker (Dato's R)."""

    def __repr__(self):
        return "R"

    def __eq__(self, other):
        return isinstance(other, TensorReplicate)

    def __hash__(self):
        return hash("R")


class TensorShard(DistElem):
    """Partitioned along mesh axis (Dato's S(i))."""

    def __init__(self, mesh_axis: int):
        self.mesh_axis = mesh_axis

    def __repr__(self):
        return f"S({self.mesh_axis})"

    def __eq__(self, other):
        return isinstance(other, TensorShard) and self.mesh_axis == other.mesh_axis

    def __hash__(self):
        return hash(("S", self.mesh_axis))


# ============================================================
# Unified Layout = Distribution × Memory - R10
# ============================================================

class TensorLayout:
    """Unified layout type with distribution and memory facets.

    Per R10 design (docs/research/type_system_research.md):
    - Distribution facet: Dato-style S(mesh_axis) / R per dimension
    - Memory facet: Triton-style MemLayout with strides/order/swizzle

    Example:
        # Sharded along batch (dim 0) on mesh axis 0, replicated on dim 1
        layout = TensorLayout(
            dist=(TensorShard(0), TensorReplicate()),
            mem=MemLayout.row_major((batch, hidden))
        )
    """

    def __init__(self, dist: tuple[DistElem, ...], mem: MemLayout = None):
        self.dist = dist
        self.mem = mem

    @staticmethod
    def default(rank: int) -> "TensorLayout":
        """Create default replicated layout."""
        return TensorLayout(tuple(TensorReplicate() for _ in range(rank)))

    @staticmethod
    def sharded(dim: int, rank: int, mesh_axis: int = 0) -> "TensorLayout":
        """Create layout sharded along one dimension."""
        dist = [TensorReplicate() for _ in range(rank)]
        dist[dim] = TensorShard(mesh_axis)
        return TensorLayout(tuple(dist))

    def is_replicated(self) -> bool:
        """Check if fully replicated."""
        return all(isinstance(d, TensorReplicate) for d in self.dist)

    def sharded_dims(self) -> list[int]:
        """Get list of sharded dimensions."""
        return [i for i, d in enumerate(self.dist) if isinstance(d, TensorShard)]

    def __repr__(self):
        dist_str = ", ".join(repr(d) for d in self.dist)
        if self.mem:
            return f"TensorLayout({dist_str}, mem={self.mem})"
        return f"TensorLayout({dist_str})"


class LayoutIncompatibleError(TypeError):
    """Raised when layouts cannot be joined."""
    pass


def tensor_layout_join(a: TensorLayout, b: TensorLayout) -> TensorLayout:
    """Join layouts for elementwise operations.

    Implements Dato-style join rules:
    - R ⊔ R = R
    - R ⊔ S(i) = S(i)
    - S(i) ⊔ R = S(i)
    - S(i) ⊔ S(i) = S(i)
    - S(i) ⊔ S(j) = error if i ≠ j

    Memory layout: uses first non-None, or composes if both present.
    """
    if len(a.dist) != len(b.dist):
        raise LayoutIncompatibleError(
            f"Layout rank mismatch: {len(a.dist)} vs {len(b.dist)}"
        )

    result_dist = []
    for i, (da, db) in enumerate(zip(a.dist, b.dist)):
        if isinstance(da, TensorReplicate):
            result_dist.append(db)
        elif isinstance(db, TensorReplicate):
            result_dist.append(da)
        elif isinstance(da, TensorShard) and isinstance(db, TensorShard):
            if da.mesh_axis == db.mesh_axis:
                result_dist.append(da)
            else:
                raise LayoutIncompatibleError(
                    f"Incompatible sharding at dim {i}: "
                    f"S({da.mesh_axis}) vs S({db.mesh_axis})"
                )
        else:
            result_dist.append(da)

    # Compose memory layouts if both present
    result_mem = a.mem
    if b.mem:
        if result_mem:
            result_mem = result_mem.compose(b.mem)
        else:
            result_mem = b.mem

    return TensorLayout(tuple(result_dist), result_mem)


# ============================================================
# Tensor with Layout Refinement - R10
# ============================================================

class Tensor:
    """Tensor: first-class memory resource with layout refinement.

    Per R10 (docs/research/type_system_research.md), Tensor includes:
    - data: Underlying data buffer
    - shape: Tensor dimensions
    - dtype: Data type
    - location: Memory location (Global/L2/UB/L1)
    - layout: Optional TensorLayout with distribution and memory facets

    Example:
        # Create tensor with explicit layout
        t = Tensor(
            data=None,
            shape=(batch, hidden),
            dtype=DType.F16,
            layout=TensorLayout.sharded(dim=0, rank=2, mesh_axis=0)
        )
    """
    def __init__(self, data: Any, shape: tuple[int, ...], dtype: DType,
                 location: Location = Location.Global,
                 layout: TensorLayout = None):
        self.data = data
        self.shape = shape
        self.dtype = dtype
        self.location = location
        # R10: Layout as type refinement, not schedule primitive
        self.layout = layout or TensorLayout.default(len(shape))

    def __getitem__(self, idx: int) -> "Tensor":
        """Return sub-tensor view with layout propagation.

        When indexing into a tensor, the layout is adjusted:
        - The indexed dimension is removed from shape
        - Distribution elements are dropped for the indexed dimension
        - Memory layout is preserved

        Args:
            idx: Index along first dimension

        Returns:
            New Tensor with reduced rank and adjusted layout
        """
        if not self.shape:
            return self

        new_shape = self.shape[1:]
        new_layout = None
        if self.layout and len(self.layout.dist) > 1:
            # Remove first dimension from distribution
            new_layout = TensorLayout(
                dist=tuple(self.layout.dist[1:]),
                mem=self.layout.mem
            )
        elif self.layout:
            # Single dimension - return default layout for scalar
            new_layout = TensorLayout.default(len(new_shape)) if new_shape else None

        return Tensor(
            data=self.data,
            shape=new_shape,
            dtype=self.dtype,
            location=self.location,
            layout=new_layout
        )

    def slice(self, start: int, end: int) -> "Tensor":
        """Return slice view with layout propagation.

        Slicing preserves the tensor rank but changes the first dimension size.
        The layout is preserved since the structure remains the same.

        Args:
            start: Start index (inclusive)
            end: End index (exclusive)

        Returns:
            New Tensor with adjusted first dimension size
        """
        if not self.shape:
            return self

        new_shape = (end - start,) + self.shape[1:]
        return Tensor(
            data=self.data,
            shape=new_shape,
            dtype=self.dtype,
            location=self.location,
            layout=self.layout  # Layout preserved for slicing
        )

    def nbytes(self) -> int:
        """Return size in bytes.

        Calculates total memory footprint based on shape and dtype.

        Returns:
            Total bytes needed to store the tensor data
        """
        if not self.shape:
            return 0

        num_elements = 1
        for dim in self.shape:
            num_elements *= dim

        # Bytes per element based on dtype
        dtype_sizes = {
            DType.F32: 4,
            DType.F64: 8,
            DType.F16: 2,
            DType.BF16: 2,
            DType.I8: 1,
            DType.I16: 2,
            DType.I32: 4,
            DType.I64: 8,
            DType.U8: 1,
            DType.U16: 2,
            DType.U32: 4,
            DType.U64: 8,
        }
        bytes_per_element = dtype_sizes.get(self.dtype, 4)  # Default to 4 bytes

        return num_elements * bytes_per_element

    def with_layout(self, layout: TensorLayout) -> "Tensor":
        """Return new tensor with different layout annotation.

        Note: This doesn't move data - use relayout() for actual redistribution.
        """
        return Tensor(
            data=self.data,
            shape=self.shape,
            dtype=self.dtype,
            location=self.location,
            layout=layout
        )


# ============================================================
# Layout Transition Operations - R10
# ============================================================

def relayout(tensor: Tensor, to: TensorLayout) -> Tensor:
    """Explicit layout transition (data redistribution).

    Creates a new tensor with the specified layout. This operation
    may involve data movement (communication) if distribution changes.

    This is the explicit operation required when layouts are incompatible
    (per Dato's design: make communication visible).

    Example:
        # Redistribute from replicated to sharded
        sharded_t = relayout(replicated_t, TensorLayout.sharded(0, 2, mesh_axis=0))
    """
    # In actual implementation, this would:
    # 1. Compare source and target distribution
    # 2. Generate appropriate collective operation (allgather, reduce_scatter, etc.)
    # 3. Return tensor with new layout
    return Tensor(
        data=tensor.data,  # Would be new buffer in real impl
        shape=tensor.shape,
        dtype=tensor.dtype,
        location=tensor.location,
        layout=to
    )


def allreduce(tensor: Tensor, mesh_axis: int, op: str = "sum") -> Tensor:
    """Allreduce collective for sharded tensors.

    Reduces sharded tensor along mesh axis and replicates result.

    Args:
        tensor: Input tensor (must be sharded on mesh_axis)
        mesh_axis: Mesh axis to reduce along
        op: Reduction operation ("sum", "max", "min", "mean")

    Returns:
        Tensor with replicated layout along the previously sharded dimension
    """
    new_dist = list(tensor.layout.dist)
    for i, d in enumerate(new_dist):
        if isinstance(d, TensorShard) and d.mesh_axis == mesh_axis:
            new_dist[i] = TensorReplicate()

    new_layout = TensorLayout(tuple(new_dist), tensor.layout.mem)
    return Tensor(
        data=tensor.data,
        shape=tensor.shape,
        dtype=tensor.dtype,
        location=tensor.location,
        layout=new_layout
    )


def allgather(tensor: Tensor, dim: int, mesh_axis: int) -> Tensor:
    """Allgather collective to replicate a sharded dimension.

    Args:
        tensor: Input tensor
        dim: Tensor dimension to gather
        mesh_axis: Mesh axis along which to gather

    Returns:
        Tensor with replicated layout on the specified dimension
    """
    new_dist = list(tensor.layout.dist)
    if dim < len(new_dist):
        new_dist[dim] = TensorReplicate()

    new_layout = TensorLayout(tuple(new_dist), tensor.layout.mem)
    return Tensor(
        data=tensor.data,
        shape=tensor.shape,
        dtype=tensor.dtype,
        location=tensor.location,
        layout=new_layout
    )


def reduce_scatter(tensor: Tensor, dim: int, mesh_axis: int, op: str = "sum") -> Tensor:
    """Reduce-scatter collective to shard a replicated dimension.

    Args:
        tensor: Input tensor
        dim: Tensor dimension to scatter
        mesh_axis: Mesh axis along which to scatter
        op: Reduction operation

    Returns:
        Tensor with sharded layout on the specified dimension
    """
    new_dist = list(tensor.layout.dist)
    if dim < len(new_dist):
        new_dist[dim] = TensorShard(mesh_axis)

    new_layout = TensorLayout(tuple(new_dist), tensor.layout.mem)
    return Tensor(
        data=tensor.data,
        shape=tensor.shape,
        dtype=tensor.dtype,
        location=tensor.location,
        layout=new_layout
    )
