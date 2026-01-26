"""Spatial layout primitives.

DEPRECATED: Use TensorShard/TensorReplicate from types.py or
LayoutShard/LayoutReplicate from type_checker.py instead.

This module is retained for backward compatibility only.
"""

import warnings


class Shard:
    """Shard data along a dimension.

    DEPRECATED: Use TensorShard(mesh_axis=dim) from types.py instead.
    Note: TensorShard uses mesh_axis (mesh dimension), not dim (tensor dimension).
    """
    def __init__(self, dim: int):
        warnings.warn(
            "spatial.Shard is deprecated. Use TensorShard from pto_wsp.types "
            "or LayoutShard from pto_wsp.type_checker instead.",
            DeprecationWarning,
            stacklevel=2
        )
        self.dim = dim


class Replicate:
    """Replicate data to all tiles.

    DEPRECATED: Use TensorReplicate from types.py instead.
    """
    def __init__(self):
        warnings.warn(
            "spatial.Replicate is deprecated. Use TensorReplicate from pto_wsp.types "
            "or LayoutReplicate from pto_wsp.type_checker instead.",
            DeprecationWarning,
            stacklevel=2
        )
