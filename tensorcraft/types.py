"""Type aliases for TensorCraft."""

from typing import Literal, Tuple, TypeAlias, TypeGuard

import numpy as np
import numpy.typing as npt

from tensorcraft.types import linear2multiIndex, multi2linearIndex

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


class Shape:
    """
    A class representing the shape of a tensor.

    Parameters
    ----------
    dims : npt.ArrayLike
        The dimensions of the tensor.

    Attributes
    ----------
    order : int
        The order of the tensor.
    shape : MIndex
        The shape of the tensor.
    size : int
        The size of the tensor.

    Methods
    -------
    getLinearIndex(indices: MIndex, order: MemLayout | MIndex = "R") -> int
        Obtain the multi-dimensional indices to a linear index.
    getMultiIndex(index: int, order: MemLayout = "R") -> MIndex
        Convert a linear index to multi-dimensional indices.
    info() -> None
        Print information about the tensor.
    """

    def __init__(self, dims: npt.ArrayLike) -> None:
        """
        Initialize a new instance of the Shape class.

        Parameters
        ----------
        dims : npt.ArrayLike
            The dimensions of the tensor.
        """
        if isinstance(dims, np.ndarray):
            if not np.issubdtype(dims.dtype, np.integer):
                raise ValueError("Dimensions must be integers")
            if len(dims.shape) != 1:
                raise ValueError("Must be a 1D array of dimensions")
            if len(dims) == 0:
                raise ValueError("Must have at least one dimension")
            if not np.all(dims > 0):
                raise ValueError("Dimensions must be positive")
            self._dims: IndexTuple = tuple(dims.astype(np.int32))
        elif isinstance(dims, (list, tuple)):
            if len(dims) == 0:
                raise ValueError("Must have at least one dimension")
            if not all(isinstance(d, int) for d in dims):
                raise ValueError("Dimensions must be integers")
            if not all(d > 0 for d in dims):
                raise ValueError("Dimensions must be positive")
            self._dims = dims if isinstance(dims, tuple) else tuple(dims)
        elif isinstance(dims, (np.int_, int)):
            if dims < 1:
                raise ValueError("Dimensions must be positive")
            self._dims = (dims,)  # type: ignore
        else:
            raise ValueError("Invalid dimensions")

    @property
    def order(self) -> int:
        """
        Returns the order given the shape dimentions.

        Returns
        -------
        int
            The number of dimentions.
        """
        return len(self._dims)

    @property
    def shape(self) -> IndexTuple:
        """
        Returns the shape as a tuple of integers.

        Returns
        -------
        MIndex
            Integer tuple with shape dimentions.
        """
        return self._dims

    @property
    def size(self) -> int:
        """
        Returns the total number of elements given the shape dimentions.

        Returns
        -------
        int
            Total number of elements.
        """
        return np.prod(self._dims, dtype=int)  # type: ignore

    def getLinearIndex(
        self, indices: IndexTuple, order: MemLayout | IndexTuple = "R"
    ) -> int:
        """
        Obtain the multi-dimensional indices to a linear index.

        Parameters
        ----------
        indices : MIndex
            The multi-dimensional indices.
        order : str or MIndex, optional
            The order of the indices. Defaults to "R" (row-major order).

        Returns
        -------
        int
            The linear index corresponding to the given multi-dimensional indices.

        Raises
        ------
        ValueError
            If the length of the indices is not equal to the length of the tensor's dimensions.
        ValueError
            If the order is invalid.
        """
        if order == "R":
            return multi2linearIndex(
                self._dims, indices, order=np.arange(len(self._dims))[::-1]
            )
        elif order == "C":
            return multi2linearIndex(self._dims, indices)
        else:
            try:
                return multi2linearIndex(self._dims, indices, np.array(order))
            except ValueError:
                raise ValueError("Invalid order")

    def getMultiIndex(self, index: int, order: MemLayout = "R") -> IndexTuple:
        """
        Convert a linear index to multi-dimensional indices.

        Parameters
        ----------
        index : int
            The linear index.
        order : str, optional
            The order of the multi-dimensional indices. Defaults to "R" (row-major order).

        Returns
        -------
        MIndex
            The multi-dimensional indices corresponding to the given linear index.

        Raises
        ------
        ValueError
            If the index is out of bounds.
        """
        return linear2multiIndex(index, self._dims, order=order)

    def info(self) -> None:
        """Print information about the tensor."""
        print(f"Order: {self.order}")
        print(f"Shape: {self.shape}")
        print(f"Size: {self.size}")
