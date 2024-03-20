"""Utility functions for tensorcraft"""

import numpy as np

_order2npOrder = {"C": "F", "R": "C"}


def multi2linearIndex(
    dims: np.ndarray, indices: np.ndarray, order: np.ndarray | None = None
) -> int:
    """Converts a multi-dimensional index to a linear index

    This function takes a multi-dimensional index and converts it to a linear index
    based on the dimensions of the tensor. The linear index represents the position
    of the element in a flattened version of the tensor.

    Parameters
    ----------
    dims : np.ndarray
        An array containing the dimensions of the tensor.

    indices : np.ndarray
        An array containing the multi-dimensional index.

    order : np.ndarray | None, optional
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
    >>> indices = np.array([1, 2])
    >>> multi2linearIndex(dims, indices)
    5

    >>> dims = np.array([2, 3])
    >>> indices = np.array([1, 2])
    >>> order = np.array([1, 0])
    >>> multi2linearIndex(dims, indices, order)
    3

    """
    if len(indices) != len(dims):
        raise ValueError("Indices must have the same length as the tensor's dimensions")

    result: int = 0
    if order is None:
        for i in range(len(indices)):
            result += indices[i] * np.prod(dims[:i])
    else:
        indices = indices[order]
        dims = dims[order]
        for i in range(len(indices)):
            result += indices[i] * np.prod(dims[:i])
    return result


def order2npOrder(order: str) -> str:
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
    >>> order2npOrder('C')
    'F'

    >>> order2npOrder('R')
    'C'

    >>> order2npOrder('Z')
    Traceback (most recent call last):
        ...
    KeyError: 'Z is not a valid order string.'
    """
    return _order2npOrder[order]
