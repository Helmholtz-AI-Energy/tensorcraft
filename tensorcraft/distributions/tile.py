"""TileDist class."""

import torch

from tensorcraft.distributions.dist import Dist
from tensorcraft.util import linear2multiIndex, multi2linearIndex


class TileDist(Dist):
    def __init__(self, processor_mesh: int | torch.Size, tile_size: int) -> None:
        super().__init__(processor_mesh=processor_mesh)
        self._tile_size = tile_size

    def compatible(self, shape: torch.Size):  # noqa: D102
        for dim, dim_size in enumerate(shape):
            if dim_size % self._tile_size != 0:
                print("Tile shape not divisible by tile size along dimension ", dim)
                return False

        return True

    def processorView(self, shape: torch.Size):  # noqa: D102
        if not self.compatible(shape):
            raise ValueError("The tensor is not compatible with the distribution")

        processor_view = torch.zeros((*shape, self.numProcessors), dtype=torch.bool)
        for i in range(shape.numel()):
            i_mi = linear2multiIndex(i, shape)
            processor_view[tuple(i_mi) + (None,)] = self.getElementLocation(shape, i_mi)

        return processor_view

    def getElementLocation(self, shape: torch.Size, index: int | torch.Size):  # noqa: D102
        if not self.compatible(shape):
            raise ValueError("The tensor is not compatible with the distribution")

        if isinstance(index, int):
            index = multi2linearIndex(shape, index)

        shrinked_index = torch.tensor(index) // self._tile_size
        shrinked_shape = torch.tensor(shape) // self._tile_size
        shrinked_linear_index = multi2linearIndex(
            shrinked_shape, shrinked_index, order=torch.arange(len(shape)).flip(0)
        )

        p_list = torch.zeros((self.numProcessors,), dtype=torch.bool)
        # print(f"Index: {index}, Shrinked index: {shrinked_index}, Shrinked shape: {shrinked_shape}, Shrinked linear index: {shrinked_linear_index}, p: {shrinked_linear_index % self._num_processors}")

        p_list[shrinked_linear_index % self.numProcessors] = True

        return p_list

    def allGather(self, shape, mesh_axis=None):
        raise NotImplementedError("allGather is not implemented for TileDist")

    def scatter(self, shape, mesh_axis=None):
        raise NotImplementedError("scatter is not implemented for TileDist")

    def permute(self, shape, mesh_axis=None):
        raise NotImplementedError("permute is not implemented for TileDist")
