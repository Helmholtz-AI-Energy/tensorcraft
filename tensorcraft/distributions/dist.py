"""Abstract base class for distributions."""

from abc import ABC, abstractmethod

import torch


class Dist(ABC):
    """Abstract base class for distributions."""

    @abstractmethod
    def processorView(self, shape: torch.Size) -> torch.Tensor:
        """
        Get the processor view of a tensor.

        The processor view is a boolean array that shares the same shape as the input tensors, where each of the elements is a a boolean array marking on which processors the element is located.

        Parameters
        ----------
        tensor : Tensor
            The input tensor.

        Returns
        -------
        np.ndarray
            The processor view of the tensor.
        """
        pass

    @property
    @abstractmethod
    def numProcessors(self) -> int:
        """
        Get the number of processors.

        Returns
        -------
        int
            The number of processors.
        """
        pass

    @property
    @abstractmethod
    def processorArrangement(self) -> torch.Size:
        """
        Get the arrangement of processors.

        Returns
        -------
        npt.NDArray[np.int_]
            The arrangement of processors.
        """
        pass

    @abstractmethod
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
        pass

    @abstractmethod
    def getIndexLocation(
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
        npt.NDArray[np.bool_]
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

        return tile_dims, torch.cumsum(tile_dims)
