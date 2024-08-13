"""Utility functions for tensorcraft."""

from typing import Literal

import numpy as np

from tensorcraft.types import MIndex

_order2npOrder: dict[str, Literal["C", "F"]] = {"C": "F", "R": "C"}


def multi2linearIndex(
    dims: MIndex,
    indices: MIndex,
    order: MIndex | None = None,
) -> int:
    """Convert a multi-dimensional index to a linear index.

    This function takes a multi-dimensional index and converts it to a linear index
    based on the dimensions of the tensor. The linear index represents the position
    of the element in a flattened version of the tensor.

    Uses a column mayor indexing scheme by default.

    Parameters
    ----------
    dims : tuple | np.ndarray
        An array containing the dimensions of the tensor.

    indices : tuple | np.ndarray
        An array containing the multi-dimensional index.

    order : tuple | np.ndarray | None, optional
        An array specifying the order in which the dimensions should be considered
        when calculating the linear index. If None, the dimensions are considered
        in the default order, which is the same as the input order.

    Returns
    -------
    int
        The linear index corresponding to the multi-dimensional index.

    Raises
    ------
    ValueError
        If the length of the indices array is not equal to the length of the dims array.

    Examples
    --------
    >>> dims = np.array([2, 3])
    >>> indices = np.array([1, 1])
    >>> multi2linearIndex(dims, indices)
    3

    >>> dims = np.array([2, 3])
    >>> indices = np.array([1, 1])
    >>> order = np.array([1, 0])
    >>> multi2linearIndex(dims, indices, order)
    4

    """
    if len(indices) != len(dims):
        raise ValueError("Indices must have the same length as the tensor's dimensions")

    if order is None:
        indices_reorderd = indices
        dims_reorderd = dims
    else:
        if len(order) == 0 or len(order) > len(dims):
            raise ValueError("Invalid order dimensions")
        indices_reorderd = indices[order]
        dims_reorderd = dims[order]

    if not np.all(indices_reorderd >= 0) or not np.all(
        indices_reorderd < dims_reorderd
    ):
        raise ValueError("Indices out of bounds")

    result = 0
    for i in range(len(indices_reorderd)):
        result += indices_reorderd[i] * np.prod(dims_reorderd[:i])
    return result


def linear2multiIndex(index: int, dims: MIndex, order="R") -> MIndex:
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
    if index < 0 or index >= np.prod(dims, dtype=int):
        raise ValueError("Index out of bounds")

    return np.array(np.unravel_index(index, dims, order=order2npOrder(order)))


def order2npOrder(order: str) -> Literal["C", "F"]:
    """
    Convert the order of dimensions from a given order string to the corresponding NumPy order string.

    Parameters
    ----------
    order : str
        The order of dimensions as a string.

    Returns
    -------
    str
        The corresponding NumPy order string.

    Raises
    ------
    KeyError
        If the given order is not a valid order string.

    Examples
    --------
    >>> order2npOrder("C")
    'F'

    >>> order2npOrder("R")
    'C'

    >>> order2npOrder("Z")
    Traceback (most recent call last):
        ...
    KeyError: 'Z is not a valid order string.'
    """
    if order not in _order2npOrder:
        raise ValueError(f"{order} is not a valid order string.")
    return _order2npOrder[order]
