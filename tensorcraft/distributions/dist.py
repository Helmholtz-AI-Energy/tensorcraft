"""Abstract base class for distributions."""

from abc import ABC, abstractmethod

import math
import torch

from tensorcraft.util import linear2multiIndex

class Dist(ABC):
    """Abstract base class for distributions."""

    @staticmethod
    def axisSplits(
        axis_size: int, block_size: int, num_procs: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Split an axis into blocks.

        Parameters
        ----------
        axis_size : int
            The size of the axis.
        block_size : int
            The size of the blocks.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            The split sizes and the end of each of the blocks.
        """
        if block_size == 0:
            tile_dims = torch.zeros((num_procs,), dtype=torch.int)
            chunk_size = axis_size // num_procs
            remainder = axis_size % num_procs
            tile_dims[:] = chunk_size
            tile_dims[:remainder] += 1
        else:
            # Calculate the number of blocks
            n_blocks = axis_size // block_size
            if axis_size % block_size:
                n_blocks += 1
            tile_dims = torch.zeros((n_blocks,), dtype=torch.int)

            rest = axis_size % block_size
            tile_dims[:-1] = block_size
            tile_dims[-1] = block_size - rest

        return tile_dims, torch.cumsum(tile_dims, dim=0)

    def __init__(self, processor_mesh: int | torch.Size):
        if isinstance(processor_mesh, int):
            self._pmesh = torch.Size([processor_mesh])
        else:
            self._pmesh = processor_mesh


    @property
    def numProcessors(self) -> int:
        """
        Get the number of processors.

        Returns
        -------
        int
            The number of processors.
        """
        return math.prod(self._pmesh)

    @property
    def processorMesh(self) -> torch.Size:
        """
        Get the arrangement of processors.

        Returns
        -------
        torch.Size
            The arrangement of processors.
        """
        return self._pmesh

    def getProcessorMultiIndex(self, index: int) -> torch.Size:
        """
        Get the multi-index of a processor.

        Parameters
        ----------
        index : int
            The index of the processor.

        Returns
        -------
        IndexTuple
            The multi-index of the processor.
        """
        return linear2multiIndex(index, self._pmesh)


    @abstractmethod
    def processorView(self, shape: torch.Size) -> torch.Tensor:
        """
        Get the processor view of a tensor.

        The processor view is a boolean array that shares the same shape as the input tensors, where each of the elements is a a boolean array marking on which processors the element is located.

        Parameters
        ----------
        shape : torch.Size
            The shape of a tensor

        Returns
        -------
        torch.Tensor
            The processor view of the tensor.
        """
        pass

    @abstractmethod
    def getElementLocation(
        self, shape: torch.Size, index: torch.Size
    ) -> torch.BoolTensor:
        """
        Get the processors that hold a specific element of a tensor.

        Parameters
        ----------
        tensor : Tensor
            The input tensor.
        index : IndexTuple
            The multi-index.

        Returns
        -------
        torch.BoolTensor
            Boolean array marking the processors that hold the element.
        """
        pass

    @abstractmethod
    def compatible(self, shape: torch.Size) -> bool:
        """
        Check if a tensor is compatible with the distribution.

        Parameters
        ----------
        tensor : Tensor
            The input tensor.

        Returns
        -------
        bool
            True if the tensor is compatible, False otherwise.
        """
        pass

