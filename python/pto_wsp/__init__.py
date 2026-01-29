"""
PTO Workload-Schedule Programming (PTO-WSP) framework

Typed workload expressions for dynamic LLM workloads with:
- Data-parallel primitives: parallel_for, for_each, select, cond
- Pipeline-parallel (CSP): Channel, Process, consume
- Combinator-style scheduling: workload.dispatch(...).streams(...)
- JIT-style kernel definitions: @kernel, direct kernel calls
- NPU function builder: npu("name").tile(...).load(...).build()
"""

__version__ = "0.1.0"

# ============================================================
# Import order matters - import decorator functions LAST
# to avoid being overwritten by module imports
# ============================================================

# ============================================================
# Axis types
# ============================================================

from pto_wsp.types import Dense, DenseDyn, Ragged, Sparse, Symbol

# ============================================================
# Legacy workload primitives (still available)
# ============================================================

from pto_wsp.primitives import (
    parallel_for,
    for_each,
    select,
    cond,
    task,
    combine,
    sequential,
    slot_set_u64,
    slot_load_u64,
)

# ============================================================
# CSP primitives (Pipeline-parallel)
# ============================================================

from pto_wsp.csp import (
    Channel,
    process,
    send,
    try_send,
    consume,
    connect,
    replicate,
    Event,
    record,
    synchronize,
    query,
)

# ============================================================
# Schedule policies
# ============================================================

from pto_wsp.schedule import (
    DispatchPolicy,
    TimingPolicy,
    # Task graph primitives (R9)
    WindowMode,
    TaskWindow,
    DepsMode,
    Deps,
    ReadyPolicy,
    StartPolicy,
    TracePolicy,
    Pools,
    TaskGraphConfig,
)

# ============================================================
# Spatial primitives (legacy module removed - use TensorShard/TensorReplicate)
# ============================================================

# Legacy: Shard/Replicate are now aliases for TensorShard/TensorReplicate
# This maintains backward compatibility without the deprecated module
from pto_wsp.types import TensorShard as Shard, TensorReplicate as Replicate

# ============================================================
# Tensor and types
# ============================================================

from pto_wsp.types import (
    Tensor, DType, Location,
    # R10: Layout as refinement types
    MemLayout,
    DistElem, TensorReplicate, TensorShard,
    TensorLayout, LayoutIncompatibleError, tensor_layout_join,
    # R10: Layout transition operations
    relayout, allreduce, allgather, reduce_scatter,
)

# ============================================================
# Execution
# ============================================================

from pto_wsp.program import Program

# ============================================================
# Profiling/Tracing
# ============================================================

from pto_wsp.program import (
    TraceLevel,
    TraceEvent,
    ExecutionTrace,
)

# ============================================================
# Workload class (for combinator-style schedule)
# Import module contents directly without module reference
# ============================================================

from pto_wsp.workload import Workload as Workload
from pto_wsp.workload import Task as Task

# ============================================================
# Type Checker (L1 - Python builder-time checking)
# ============================================================

from pto_wsp.type_checker import (
    TypeChecker,
    TypeCheckContext,
    TypeErrorInfo,
    TypeErrorKind,
    Layout,
    Shard as LayoutShard,
    Replicate as LayoutReplicate,
    layout_join,
    LayoutCompatibilityError,
    check_kernel_call,
    check_layouts_compatible,
    validate_axis_index,
)

# ============================================================
# P namespace for symbolic loops
# ============================================================

from pto_wsp.p_namespace import P

# ============================================================
# JIT Kernel Types (from kernel.py - low-level types)
# ============================================================

from pto_wsp.kernel import (
    pto,  # PTO tile language primitives
    Value,
    TileType,
    ScalarType,
    Tile,
    Scalar,
    KernelIR,
    CompiledKernel,
    OpKind,
)

# ============================================================
# PTO-ISA kernel authoring (Python -> C++ emission)
# ============================================================

from pto_wsp.ptoisa_jit import ptoisa, ptoisa_kernel

# ============================================================
# NEW: @workload decorator + unified @kernel decorator (RECOMMENDED - R11)
# The @kernel decorator now includes JIT support (L3)
# ============================================================

from pto_wsp.builder import (
    workload_decorator,
    kernel as kernel_decorator,
    Kernel,
    KernelRef,  # Alias for Kernel (backward compat)
    In, Out, InOut,
    Constexpr,
    KernelParam,
    extract_kernel_params,
    get_direction,
    get_inner_type,
)

# Create proper names for export
workload = workload_decorator
kernel = kernel_decorator

# Backward compatibility: jit_kernel is now just kernel
jit_kernel = kernel

# ============================================================
# __all__ exports
# ============================================================

__all__ = [
    # Version
    "__version__",

    # NEW: @workload + P namespace (RECOMMENDED)
    "workload",
    "kernel",
    "ptoisa",
    "ptoisa_kernel",
    "KernelRef",
    "P",
    "In", "Out", "InOut", "Constexpr",

    # Axis types
    "Dense",
    "DenseDyn",
    "Ragged",
    "Sparse",

    # Workload primitives (legacy)
    "parallel_for",
    "for_each",
    "select",
    "cond",
    "task",
    "combine",
    "sequential",
    "slot_set_u64",
    "slot_load_u64",

    # CSP primitives
    "Channel",
    "process",
    "send",
    "try_send",
    "consume",
    "connect",
    "replicate",
    "Event",
    "record",
    "synchronize",
    "query",

    # Schedule policies
    "DispatchPolicy",
    "TimingPolicy",
    # Task graph primitives (R9)
    "WindowMode",
    "TaskWindow",
    "DepsMode",
    "Deps",
    "ReadyPolicy",
    "StartPolicy",
    "TracePolicy",
    "Pools",
    "TaskGraphConfig",

    # Spatial
    "Shard",
    "Replicate",

    # Types
    "Tensor",
    "DType",
    "Location",
    "Task",
    "Workload",
    # R10: Layout as refinement types
    "MemLayout",
    "DistElem",
    "TensorReplicate",
    "TensorShard",
    "TensorLayout",
    "LayoutIncompatibleError",
    "tensor_layout_join",
    # R10: Layout transition operations
    "relayout",
    "allreduce",
    "allgather",
    "reduce_scatter",

    # Execution
    "Program",

    # Profiling/Tracing
    "TraceLevel",
    "TraceEvent",
    "ExecutionTrace",

    # JIT Kernel Programming (L3 - now unified with @kernel)
    "jit_kernel",  # Backward compat alias for @kernel
    "pto",  # PTO tile language primitives
    "Value",
    "TileType",
    "ScalarType",
    "Tile",
    "Scalar",
    "KernelIR",
    "CompiledKernel",
    "Kernel",
    "OpKind",

    # Type Checker
    "TypeChecker",
    "TypeCheckContext",
    "TypeErrorInfo",
    "TypeErrorKind",
    "Layout",
    "LayoutShard",
    "LayoutReplicate",
    "layout_join",
    "LayoutCompatibilityError",
    "check_kernel_call",
    "check_layouts_compatible",
    "validate_axis_index",

    # Kernel Parameter Extraction
    "KernelParam",
    "extract_kernel_params",

    # C++ IR bindings (pybind11)
    "cpp",
]

# ============================================================
# Linear Layout (L8 - F₂ binary matrix representation)
# ============================================================

from pto_wsp.linear_layout import (
    LinearLayout,
    f2_dot,
    f2_matmul,
    f2_rank,
    propagate_transpose,
    propagate_reshape,
    propagate_broadcast,
    to_tensor_layout,
    from_tensor_layout,
)

# Add to __all__
__all__.extend([
    "LinearLayout",
    "f2_dot",
    "f2_matmul",
    "f2_rank",
    "propagate_transpose",
    "propagate_reshape",
    "propagate_broadcast",
    "to_tensor_layout",
    "from_tensor_layout",
])

# ============================================================
# Python → C++ IR Bridge
# ============================================================

from pto_wsp.ir_bridge import (
    workload_to_ir,
    workload_to_codegen_ir,
    module_to_string,
    parse_ir_string,
    IRBridgeError,
    HAS_CPP_BINDINGS,
)

__all__.extend([
    "workload_to_ir",
    "workload_to_codegen_ir",
    "module_to_string",
    "parse_ir_string",
    "IRBridgeError",
    "HAS_CPP_BINDINGS",
])

# ============================================================
# Automatic Scheduling
# ============================================================

from pto_wsp.auto_schedule import (
    AutoScheduler,
    SearchSpace,
    ScheduleConfig,
    ExplorationResult,
    ExplorationSummary,
    auto_schedule,
)

__all__.extend([
    "AutoScheduler",
    "SearchSpace",
    "ScheduleConfig",
    "ExplorationResult",
    "ExplorationSummary",
    "auto_schedule",
])

# ============================================================
# Exception Hierarchy (Task 16.3)
# ============================================================

from pto_wsp.errors import (
    PtoError,
    CompileError,
    TypeCheckError,
    IRConversionError,
    ExecutionError,
    KernelError,
    ScheduleError,
    ChannelError,
    ChannelClosed,
    ChannelFull,
)

__all__.extend([
    "PtoError",
    "CompileError",
    "TypeCheckError",
    "IRConversionError",
    "ExecutionError",
    "KernelError",
    "ScheduleError",
    "ChannelError",
    "ChannelClosed",
    "ChannelFull",
])

# ============================================================
# C++ IR Bindings (pybind11)
# ============================================================

try:
    from pto_wsp import pto_ir_cpp as cpp
except ImportError:
    # C++ bindings not built - provide stub
    cpp = None
