# ruff: noqa: D102
"""Block distribution module."""

import numpy as np

from tensorcraft.distributions.dist import Dist
from tensorcraft.shape import Shape
from tensorcraft.types import IndexTuple


class SlabDist(Dist):
    """
    Represents a block distribution for a tensor.

    The tensor is split along a single axis, and each block is assigned to a processor in a round-robin fashion.

    Parameters
    ----------
    num_processors : int
        The number of processors.
    dim : int, optional
        The dimension along which the tensor is distributed. Defaults to 0.
    block_size : int, optional
        The size of each block. Defaults to 0. A block size of 0 will try to split the tensor evenly across processors.

    Attributes
    ----------
    _num_processors : int
        The number of processors.
    _dim : int
        The dimension along which the tensor is distributed.
    _block_size : int
        The size of each block.
    """

    def __init__(self, num_processors: int, dim: int = 0, block_size: int = 0) -> None:
        self._num_processors = num_processors
        self._dim = dim
        self._block_size = block_size

    @property
    def numProcessors(self):
        return self._num_processors

    @property
    def processorArrangement(self):
        return np.array((self._num_processors,))

    def compatible(self, shape: Shape):
        if len(shape) > self._dim and shape.getItem(self._dim) < self._num_processors:
            print(f"Tensor dimension {self._dim} is less than the number of processors")
            return False

        if (
            self._block_size != 0
            and self._num_processors * self._block_size > shape.getItem(self._dim)
        ):
            raise ValueError(
                "Block size is too big for the number of processors and tensor dimensions"
            )

        return True

    def getProcessorMultiIndex(self, index: int):
        return np.array((index,))

    def processorView(self, shape: Shape):
        if not self.compatible(shape):
            raise ValueError("The tensor is not compatible with the distribution")

        _, tile_ends = self.axisSplits(
            shape.getItem(self._dim),
            self._block_size,
            self._num_processors,
        )

        processor_view = np.zeros(
            (*shape.asTuple(), self._num_processors), dtype=np.bool_
        )

        prev_idx = 0
        for p, next_idx in enumerate(tile_ends):
            processor = p % self._num_processors
            slice_idx: tuple[slice | int, ...] = tuple(
                slice(None) if j != self._dim else slice(prev_idx, next_idx)
                for j in range(shape.order)
            )
            slice_idx += (processor,)
            processor_view[slice_idx] = True
            prev_idx = next_idx
        return processor_view

    def getIndexLocation(self, shape: Shape, index: IndexTuple | int):
        if isinstance(index, int):
            mindex = shape.getMultiIndex(index)
        else:
            mindex = index

        _, tile_ends = self.axisSplits(
            shape.getItem(self._dim),
            self._block_size,
            self._num_processors,  # type: ignore
        )
        dim_index = mindex[self._dim]
        block_idx = np.where(dim_index < tile_ends)[0][0]

        p_list = np.zeros((self._num_processors,), dtype=np.bool_)
        p_list[block_idx % self._num_processors] = True

        return p_list
