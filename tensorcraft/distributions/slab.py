# ruff: noqa: D102
"""Block distribution module."""

import torch

from tensorcraft.distributions.dist import Dist
from tensorcraft.util import multi2linearIndex


class SlabDist(Dist):
    """
    Single axis distribution with configurable block size.

    Parameters
    ----------
    processor_mesh : int | torch.Size
        The number of processors in each dimension.
    dim : int, optional
        The dimension to distribute, by default 0.
    block_size : int, optional
        The size of the blocks, by default 0.

    Raises
    ------
    ValueError
        If the block size is too big for the number of processors and tensor dimensions.

    """

    def __init__(
        self, processor_mesh: int | torch.Size, dim: int = 0, block_size: int = 0
    ) -> None:
        super().__init__(processor_mesh=processor_mesh)
        self._dim = dim
        self._block_size = block_size

    def compatible(self, shape: torch.Size) -> bool:
        # Check that tensor has at least self._dim + 1 number of dimensions
        if len(shape) <= self._dim:
            print(f"Tensor must have at least {self._dim + 1} dimensions")
            return False

        if not self.compatibleAxis(
            self._dim, shape[self._dim], self._block_size, self.numProcessors
        ):
            print(f"Axis {self._dim}: Incompatible axis with distribution scheme")
            return False

        return True

    def processorView(self, shape: torch.Size) -> torch.Tensor:
        if not self.compatible(shape):
            raise ValueError("The tensor is not compatible with the distribution")

        _, tile_ends = self.axisSplits(
            shape[self._dim], self._block_size, self.numProcessors
        )

        processor_view = torch.zeros((*shape, self.numProcessors), dtype=torch.bool)

        prev_idx = 0
        for p, next_idx in enumerate(tile_ends):
            processor = p % self.numProcessors
            slice_idx: tuple[slice | int, ...] = tuple(
                slice(None) if j != self._dim else slice(prev_idx, next_idx)
                for j in range(len(shape))
            )
            slice_idx += (processor,)
            processor_view[slice_idx] = True
            prev_idx = next_idx
        return processor_view

    def getElementLocation(self, shape: torch.Size, index: torch.Size | int):
        if isinstance(index, int):
            mindex: torch.Size = multi2linearIndex(shape, index)
        else:
            mindex = index

        _, tile_ends = self.axisSplits(
            shape[self._dim],
            self._block_size,
            self.numProcessors,  # type: ignore
        )
        dim_index = mindex[self._dim]
        block_idx = torch.where(dim_index < tile_ends)[0][0]

        p_list = torch.zeros((self.numProcessors,), dtype=torch.bool)
        p_list[block_idx % self.numProcessors] = True

        return p_list
