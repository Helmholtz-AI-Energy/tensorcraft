"""TileDist class."""

import torch

from tensorcraft.distributions.dist import Dist
from tensorcraft.util import linear2multiIndex, multi2linearIndex


class TileDist(Dist):
    """
    TileDist is a distribution class that divides a tensor into tiles and assigns each tile to a processor.

    Parameters
    ----------
    processor_mesh : int or torch.Size
        The number of processors or the size of the processor mesh.
    tile_size : int
        The size of each tile.

    Methods
    -------
    compatible(shape: torch.Size) -> bool
        Checks if the given shape is compatible with the tile size.
    processorView(shape: torch.Size) -> torch.Tensor
        Returns a boolean tensor indicating the processor assignment for each element in the tensor.
    getElementLocation(shape: torch.Size, index: int or torch.Size) -> torch.Tensor
        Returns a boolean tensor indicating the processor assignment for a specific element in the tensor.
    allGather(shape, mesh_axis=None)
        Not implemented. Raises NotImplementedError.
    split(shape, tensor_axis, mesh_axis)
        Not implemented. Raises NotImplementedError.
    reduce_scatter(shape, mesh_axis=None)
        Not implemented. Raises NotImplementedError.
    permute(shape, mesh_axis=None)
        Not implemented. Raises NotImplementedError.
    all2all(shape, from_tensor_axis, to_tensor_axis)
        Not implemented. Raises NotImplementedError.
    """

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

    def allGather(self, shape, mesh_axis=None):  # noqa: D102
        raise NotImplementedError("allGather is not implemented for TileDist")

    def split(self, shape, tensor_axis, mesh_axis):  # noqa: D102
        raise NotImplementedError("split is not implemented for TileDist")

    def permute(self, shape, mesh_axis):  # noqa: D102
        raise NotImplementedError("permute is not implemented for TileDist")

    def all2all(self, shape, from_tensor_axis, to_tensor_axis):  # noqa: D102
        raise NotImplementedError("all2all is not implemented for TileDist")
