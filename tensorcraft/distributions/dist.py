"""Abstract base class for distributions."""

from abc import ABC, abstractmethod

import numpy as np
import numpy.typing as npt

from tensorcraft import Tensor, MIndex


class Dist(ABC):
    """Abstract base class for distributions."""

    @abstractmethod
    def processorView(self, tensor: Tensor) -> np.ndarray:
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
    def processorArrangement(self) -> npt.NDArray[np.int_]:
        """
        Get the arrangement of processors.

        Returns
        -------
        npt.NDArray[np.int_]
            The arrangement of processors.
        """
        pass

    @abstractmethod
    def getProcessorMultiIndex(self, index: int) -> MIndex:
        """
        Get the multi-index of a processor.

        Parameters
        ----------
        index : int
            The index of the processor.

        Returns
        -------
        MIndex
            The multi-index of the processor.
        """
        pass

    @abstractmethod
    def getIndexLocation(self, tensor: Tensor, index: MIndex) -> npt.NDArray[np.bool_]:
        """
        Get the processors that hold a specific element of a tensor.

        Parameters
        ----------
        tensor : Tensor
            The input tensor.
        index : MIndex
            The multi-index.

        Returns
        -------
        npt.NDArray[np.bool_]
            Boolean array marking the processors that hold the element.
        """
        pass

    @abstractmethod
    def compatible(self, tensor: Tensor) -> bool:
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
    def axis_splits(
        axis_size: int | np.int_, block_size: int, num_procs: int
    ) -> tuple[np.ndarray, np.ndarray]:
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
            tile_dims = np.zeros((num_procs,), dtype=np.int_)
            chunk_size = axis_size // num_procs
            remainder = axis_size % num_procs
            tile_dims[:] = chunk_size
            tile_dims[:remainder] += 1
        else:
            # Calculate the number of blocks
            n_blocks = axis_size // block_size
            if axis_size % block_size:
                n_blocks += 1
            tile_dims = np.zeros((n_blocks,), dtype=np.int_)

            rest = axis_size % block_size
            tile_dims[:-1] = block_size
            tile_dims[-1] = block_size - rest

        return tile_dims, np.cumsum(tile_dims)
