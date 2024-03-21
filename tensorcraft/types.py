"""Type aliases for TensorCraft."""

from typing import Tuple, TypeAlias

import numpy as np

MIndex: TypeAlias = np.ndarray[Tuple[int], np.dtype[np.int64]]
"""
MIndex is a type alias representing a multi-dimensional index array.

It is defined as a 1-D numpy ndarray integers, and a data type of np.int_.

Example usage:
    index: MIndex = np.array((1, 2, 3))
"""
