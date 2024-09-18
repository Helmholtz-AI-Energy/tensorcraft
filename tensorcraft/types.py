"""Type aliases for TensorCraft."""

from typing import Literal, Tuple, TypeAlias, TypeGuard

import numpy as np

MemLayoutNP = Literal["C", "F"]
MemLayout = Literal["R", "C"]

Index: TypeAlias = int | np.int_

IndexTuple: TypeAlias = Tuple[Index, ...]
"""
MIndex is a type alias representing a multi-dimensional index array.

It is defined as a tuple of integers, and a data type of int.

Example usage:
    index: MIndex = (1, 2, 3)
"""

TensorDataType: TypeAlias = np.ndarray | np.number | int | float


def is_tensor_type(value: TensorDataType) -> TypeGuard[TensorDataType]:
    """Check if the value is a tensor type.

    Parameters
    ----------
    value : TensorType
        The value to check.

    Returns
    -------
    TypeGuard[TensorType]
    True if the value is a tensor type, False otherwise.
    """
    return isinstance(value, (np.ndarray, np.number, int, float))


ScalarDataType: TypeAlias = np.number | int | float


def is_scalar_type(value: TensorDataType) -> TypeGuard[ScalarDataType]:
    """Check if the value is a scalar type.

    Parameters
    ----------
    value : ScalarType
        The value to check.

    Returns
    -------
    TypeGuard[ScalarType]
        True if the value is a scalar type, False otherwise.
    """
    return isinstance(value, (np.number, int, float))
