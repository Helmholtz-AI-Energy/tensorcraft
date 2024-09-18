"""TileDist class."""

import numpy as np

from tensorcraft.distributions.dist import Dist
from tensorcraft.util import multi2linearIndex


class TileDist(Dist):
    """
    A distribution class for tiling tensors across multiple processors.

    Splits a tensor into regular tiles, n-dimensional boxes of equal side lenght, and assigns each tile to a processor in a round-robin fashion.

    Parameters
    ----------
    num_processors : int
        The number of processors to distribute the tensor across.
    tile_size : int
        The size of each tile.

    Attributes
    ----------
    numProcessors : int
        The number of processors.
    processorArrangement : numpy.ndarray
        The arrangement of processors.
    """

    def __init__(self, num_processors: int, tile_size: int) -> None:
        self._num_processors = num_processors
        self._tile_size = tile_size

    @property
    def numProcessors(self):  # noqa: D102
        return self._num_processors

    @property
    def processorArrangement(self):  # noqa: D102
        return np.array((self._num_processors, 1))

    def compatible(self, shape):  # noqa: D102
        for dim, dim_size in enumerate(shape.asTuple()):
            if dim_size % self._tile_size != 0:
                print("Tile shape not divisible by tile size along dimension ", dim)
                return False

        return True

    def getProcessorMultiIndex(self, index):  # noqa: D102
        return np.array((index,))

    def processorView(self, shape):  # noqa: D102
        if not self.compatible(shape):
            raise ValueError("The tensor is not compatible with the distribution")

        processor_view = np.zeros(
            (*shape.asTuple(), self._num_processors), dtype=np.bool_
        )
        for i in range(shape.size):
            i_mi = shape.getMultiIndex(i)
            processor_view[tuple(i_mi) + (None,)] = self.getIndexLocation(shape, i_mi)

        return processor_view

    def getIndexLocation(self, shape, index):  # noqa: D102
        if not self.compatible(shape):
            raise ValueError("The tensor is not compatible with the distribution")

        shrinked_index = np.array(index) // self._tile_size
        shrinked_shape = np.array(shape.asTuple()) // self._tile_size
        shrinked_linear_index = multi2linearIndex(
            shrinked_shape, shrinked_index, order=np.arange(shape.order)[::-1]
        )

        p_list = np.zeros((self._num_processors,), dtype=np.bool_)
        # print(f"Index: {index}, Shrinked index: {shrinked_index}, Shrinked shape: {shrinked_shape}, Shrinked linear index: {shrinked_linear_index}, p: {shrinked_linear_index % self._num_processors}")

        p_list[shrinked_linear_index % self._num_processors] = True

        return p_list
