# ruff: noqa: D102
"""Block distribution module."""

import logging
from typing import Any

import torch

from tensorcraft.distributions.dist import Dist
from tensorcraft.util.axis_utils import linear2multiIndex

log = logging.getLogger(__name__)


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

    __slots__ = ("_dim", "_block_size")

    def __init__(
        self, processor_mesh: int | torch.Size, dim: int = 0, block_size: int = 0
    ) -> None:
        super().__init__(processor_mesh=processor_mesh)
        self._dim = dim
        self._block_size = block_size

    def __eq__(self, other: Any) -> bool:
        if super().__eq__(other) and isinstance(other, SlabDist):
            return self._dim == other._dim and self._block_size == other._block_size
        else:
            return False

    def __str__(self) -> str:
        return f"D_[{self.numProcessors}]⊥{self._dim}({self._block_size})"

    def latexStr(self) -> str:
        return f"T_{{\\perp\\{{ {self._dim} \\}}({self._block_size})}}"

    def compatible(self, shape: torch.Size) -> bool:
        # Check that tensor has at least self._dim + 1 number of dimensions
        if len(shape) <= self._dim:
            log.debug(f"Tensor must have at least {self._dim + 1} dimensions")
            return False

        if not self.compatibleAxis(
            self._dim, shape[self._dim], self._block_size, self.numProcessors
        ):
            log.debug(f"Axis {self._dim}: Incompatible axis with distribution scheme")
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

    def getElementLocation(
        self, shape: torch.Size, index: torch.Size | int
    ) -> torch.Tensor:
        if isinstance(index, int):
            mindex: torch.Size = linear2multiIndex(index, shape)
        else:
            mindex = index

        _, tile_ends = self.axisSplits(
            shape[self._dim],
            self._block_size,
            self.numProcessors,
        )
        dim_index = mindex[self._dim]
        block_idx = torch.where(dim_index < tile_ends)[0][0]

        p_list = torch.zeros((self.numProcessors,), dtype=torch.bool)
        p_list[block_idx % self.numProcessors] = True

        return p_list

    def maxNumElements(self, shape: torch.Size) -> int:
        raise NotImplementedError("Not implemented for SlabDist")

    def neighbours(self, shape: torch.Size) -> list[tuple[str, "SlabDist", int, int]]:
        raise NotImplementedError("Not implemented for SlabDist")
