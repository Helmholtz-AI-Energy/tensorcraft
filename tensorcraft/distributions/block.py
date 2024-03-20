import numpy as np

from tensorcraft.distributions.dist import Dist
from tensorcraft.tensor import Tensor


class BlockDist(Dist):
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

    def compatible(self, tensor: Tensor):
        if tensor.shape[self._dim] < self._num_processors:
            print(f"Tensor dimension {self._dim} is less than the number of processors")
            return False

        return True

    def getProcessorMultiIndex(self, index: int):
        return np.array((index,))

    def processorView(self, tensor: Tensor):
        if not self.compatible(tensor):
            raise ValueError("The tensor is not compatible with the distribution")

        if self._block_size == 0:
            block_size = tensor.shape[self._dim] // self._num_processors
        else:
            block_size = self._block_size
            if self._num_processors * block_size > tensor.shape[self._dim]:
                raise ValueError(
                    "Block size is two big for the number of processors and tensor dimensions"
                )

        processor_view = np.zeros((*tensor.shape, self._num_processors), dtype=np.bool_)

        for i in range(tensor.shape[self._dim]):
            p = (i // block_size) % self._num_processors
            slice_idx = tuple(
                slice(None) if j != self._dim else i for j in range(tensor.order)
            )
            slice_idx += (p,)
            processor_view[slice_idx] = True
        return processor_view

    def getIndexLocation(self, tensor, index):
        if isinstance(index, int):
            index = tensor.getMultiIndex(index)

        p_list = np.zeros((self._num_processors,), dtype=np.bool_)
        p_list[
            np.floor(
                index[self._dim] / (tensor.shape[self._dim] / self._num_processors)
            )
        ] = True

        return p_list
