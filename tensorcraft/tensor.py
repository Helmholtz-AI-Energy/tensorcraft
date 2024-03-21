"""Tensor class."""

import numpy as np
import numpy.typing as npt

from tensorcraft.types import MIndex
from tensorcraft.util import multi2linearIndex, order2npOrder


class Tensor:
    """
    A class representing a tensor with a given shape. This is purely an abstract representation and does not store any data.

    Parameters
    ----------
    dims : npt.ArrayLike
        The dimensions of the tensor.

    Attributes
    ----------
    _dims : MIndex
        The dimensions of the tensor.

    Methods
    -------
    __init__(dims: npt.ArrayLike) -> None
        Initializes a new instance of the Tensor class.
    order() -> int
        Returns the order of the tensor.
    shape() -> MIndex
        Returns the shape of the tensor.
    size() -> int
        Returns the size of the tensor.
    linearIndex(indices: MIndex, order: str | MIndex = "R") -> int
        Converts multi-dimensional indices to a linear index.
    getMultiIndex(index: int, order: str = "R") -> MIndex
        Converts a linear index to multi-dimensional indices.
    info() -> None
        Prints information about the tensor.
    """

    def __init__(self, dims: npt.ArrayLike) -> None:
        """
        Initialize a new instance of the Tensor class.

        Parameters
        ----------
        dims : npt.ArrayLike
            The dimensions of the tensor.
        """
        self._dims: MIndex = np.array(dims)

    @property
    def order(self) -> int:
        """
        Returns the order of the tensor.

        Returns
        -------
        int
            The order of the tensor.
        """
        return len(self._dims)

    @property
    def shape(self) -> MIndex:
        """
        Returns the shape of the tensor.

        Returns
        -------
        MIndex
            The shape of the tensor.
        """
        return self._dims

    @property
    def size(self) -> int:
        """
        Returns the size of the tensor.

        Returns
        -------
        int
            The size of the tensor.
        """
        return np.prod(self._dims, dtype=int)

    def getLinearIndex(self, indices: MIndex, order: str | MIndex = "R") -> int:
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
        if len(indices) != len(self._dims):
            raise ValueError(
                "Indices must have the same length as the tensor's dimensions"
            )
        idx_array = np.array(indices)

        if order == "R":
            return multi2linearIndex(
                self._dims, idx_array, order=np.arange(len(self._dims))[::-1]
            )
        elif order == "C":
            return multi2linearIndex(self._dims, idx_array)
        elif isinstance(order, (list, tuple)):
            order_array = np.array(order)
            return multi2linearIndex(self._dims, idx_array, order_array)
        else:
            raise ValueError("Invalid order")

    def getMultiIndex(self, index: int, order: str = "R") -> MIndex:
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
        if index < 0 or index >= self.size:
            raise ValueError("Index out of bounds")

        return np.array(np.unravel_index(index, self._dims, order=order2npOrder(order)))

    def info(self) -> None:
        """Print information about the tensor."""
        print(f"Order: {self.order}")
        print(f"Shape: {self.shape}")
        print(f"Size: {self.size}")
