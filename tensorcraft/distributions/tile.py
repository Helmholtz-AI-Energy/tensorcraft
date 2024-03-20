import numpy as np

from tensorcraft.distributions.dist import Dist
from tensorcraft.tensor import Tensor
from tensorcraft.util import multi2linearIndex


class TileDist(Dist):
    def __init__(self, num_processors: int, tile_size: int) -> None:
        self._num_processors = num_processors
        self._tile_size = tile_size

    @property
    def numProcessors(self):
        return self._num_processors

    @property
    def processorArrangement(self):
        return np.array((self._num_processors, 1))

    def compatible(self, tensor: Tensor) -> bool:
        for dim, dim_size in enumerate(tensor.shape):
            if dim_size % self._tile_size != 0:
                print("Tile shape not divisible by tile size along dimension ", dim)
                return False

        return True

    def getProcessorMultiIndex(self, index: int):
        return np.array((index,))

    def processorView(self, tensor: Tensor):
        if not self.compatible(tensor):
            raise ValueError("The tensor is not compatible with the distribution")

        processor_view = np.zeros((*tensor.shape, self._num_processors), dtype=np.bool_)
        for i in range(tensor.size):
            i_mi = tensor.getMultiIndex(i) + (None,)
            processor_view[i_mi] = self.getIndexLocation(tensor, i_mi)

        return processor_view

    def getIndexLocation(
        self, tensor: Tensor, index: int | tuple | np.ndarray
    ) -> np.ndarray:
        if not self.compatible(tensor):
            raise ValueError("The tensor is not compatible with the distribution")

        if isinstance(index, int):
            index = tensor.getMultiIndex(index)

        shrinked_index = np.array(index) // self._tile_size
        shrinked_shape = tensor.shape // self._tile_size
        shrinked_linear_index = multi2linearIndex(
            shrinked_shape, shrinked_index, order=np.arange(tensor.order)[::-1]
        )

        p_list = np.zeros((self._num_processors,), dtype=np.bool_)
        # print(f"Index: {index}, Shrinked index: {shrinked_index}, Shrinked shape: {shrinked_shape}, Shrinked linear index: {shrinked_linear_index}, p: {shrinked_linear_index % self._num_processors}")

        p_list[shrinked_linear_index % self._num_processors] = True

        return p_list
