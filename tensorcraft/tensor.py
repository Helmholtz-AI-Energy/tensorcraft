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
        if isinstance(dims, np.ndarray):
            if not np.issubdtype(dims.dtype, np.integer):
                raise ValueError("Dimensions must be integers")
            if len(dims.shape) != 1:
                raise ValueError("Must be a 1D array of dimensions")
            if len(dims) == 0:
                raise ValueError("Must have at least one dimension")
            if not np.all(dims > 0):
                raise ValueError("Dimensions must be positive")
            self._dims: MIndex = dims.astype(np.int64)
        elif isinstance(dims, (list, tuple)):
            if len(dims) == 0:
                raise ValueError("Must have at least one dimension")
            if not all(isinstance(d, int) for d in dims):
                raise ValueError("Dimensions must be integers")
            if not all(d > 0 for d in dims):
                raise ValueError("Dimensions must be positive")
            self._dims = np.array(dims, dtype=np.int64)
        elif isinstance(dims, int):
            if dims < 1:
                raise ValueError("Dimensions must be positive")
            self._dims = np.array([dims], dtype=np.int64)
        else:
            raise ValueError("Invalid dimensions")

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

    def getLinearIndex(
        self, indices: npt.ArrayLike, order: str | npt.ArrayLike = "R"
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
        idx_array = np.array(indices)
        if order == "R":
            return multi2linearIndex(
                self._dims, idx_array, order=np.arange(len(self._dims))[::-1]
            )
        elif order == "C":
            return multi2linearIndex(self._dims, idx_array)
        else:
            try:
                order_array = np.array(order)
                return multi2linearIndex(self._dims, idx_array, order_array)
            except ValueError:
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
