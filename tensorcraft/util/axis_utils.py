"""Utility functions for tensorcraft."""

import math
from typing import Literal, cast

import torch

MemLayoutNP = Literal["C", "F"]
MemLayout = Literal["R", "C"]

_order2npOrder: dict[MemLayout, MemLayoutNP] = {"C": "F", "R": "C"}


def multi2linearIndex(
    dims: torch.Size,
    indices: torch.Size,
    order: torch.Size | None = None,
) -> int:
    """Convert a multi-dimensional index to a linear index.

    This function takes a multi-dimensional index and converts it to a linear index
    based on the dimensions of the tensor. The linear index represents the position
    of the element in a flattened version of the tensor.

    Uses a row mayor indexing scheme by default.

    Parameters
    ----------
    dims : torch.Size
        An array containing the dimensions of the tensor.

    indices : torch.Size
        An array containing the multi-dimensional index.

    order : torch.Size, optional
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
    >>> dims = (2, 3)
    >>> indices = (1, 1)
    >>> multi2linearIndex(dims, indices)
    3

    >>> dims = (2, 3)
    >>> indices = (1, 1)
    >>> order = (1, 0)
    >>> multi2linearIndex(dims, indices, order)
    4

    """
    if len(indices) != len(dims):
        raise ValueError("Indices must have the same length as the tensor's dimensions")

    if order is None:
        indices_reorderd = torch.tensor(indices).flip(0)
        dims_reorderd = torch.tensor(dims).flip(0)
    else:
        if len(order) == 0 or len(order) > len(dims):
            raise ValueError("Invalid order dimensions")
        order_t = torch.tensor(order)
        indices_reorderd = torch.tensor(indices)[order_t].flip(0)
        dims_reorderd = torch.tensor(dims)[order_t].flip(0)

    if not torch.all(indices_reorderd >= 0) or not torch.all(
        indices_reorderd < dims_reorderd
    ):
        raise ValueError("Indices out of bounds")

    result = 0
    if indices_reorderd.size() == tuple():
        indices_reorderd = indices_reorderd.unsqueeze(-1)
        dims_reorderd = dims_reorderd.unsqueeze(-1)

    for i in range(len(indices_reorderd)):
        result += indices_reorderd[i] * torch.prod(dims_reorderd[:i])
    return result


def linear2multiIndex(
    index: int, dims: torch.Size, order: MemLayout = "R"
) -> torch.Size:
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
    if index < 0 or index >= math.prod(dims):
        raise ValueError("Index out of bounds")

    index_tensor = torch.tensor(index)

    if order == "R":
        return torch.Size(
            [cast(int, dim.item()) for dim in torch.unravel_index(index_tensor, dims)]
        )
    else:
        return torch.Size(
            [
                cast(int, dim.item())
                for dim in torch.unravel_index(
                    index_tensor,
                    dims[::-1],
                )
            ]
        )


def order2npOrder(order: MemLayout) -> MemLayoutNP:
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
    if order not in _order2npOrder.keys():
        raise ValueError(f"{order} is not a valid order string.")
    return _order2npOrder[order]
