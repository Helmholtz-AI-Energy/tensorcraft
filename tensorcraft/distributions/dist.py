"""Abstract base class for distributions."""

import logging
import math
from abc import ABC, abstractmethod
from typing import Optional

import torch
from typing_extensions import Self

from tensorcraft.util import linear2multiIndex

log = logging.getLogger("tensorcraft")


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
        num_procs : int
            The number of processors.

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
            rest = axis_size % block_size
            if rest:
                tile_dims = torch.zeros((n_blocks + 1,), dtype=torch.int)
                tile_dims[:-1] = block_size
                tile_dims[-1] = rest
            else:
                tile_dims = torch.zeros((n_blocks,), dtype=torch.int)
                tile_dims[:] = block_size

        return tile_dims, torch.cumsum(tile_dims, dim=0)

    @staticmethod
    def maxBlockSize(axis_size: int, num_procs: int) -> int:
        """
        Calculate the maximum allowed block size given the size of the axis and the number of workers.

        Parameters
        ----------
        axis_size : int
            The size of the axis.
        num_procs : int
            The number of processors.

        Returns
        -------
        int
            The maximum block size.
        """
        if num_procs == 1:
            return axis_size

        max_block_size = math.floor((axis_size - 1) / (num_procs - 1))

        if max_block_size < 1:
            max_block_size = 1
        return max_block_size

    @staticmethod
    def compatibleAxis(
        axis_index, axis_size: int, block_size: int, num_procs: int
    ) -> bool:
        """
        Given an axis size, block size, and number of processors, check if the axis can be distributed.

        Parameters
        ----------
        axis_index : int
            The index of the axis.
        axis_size : int
            The size of the axis.
        block_size : int
            The size of the blocks.
        num_procs : int
            The number of processors.

        Returns
        -------
        bool
            True if the axis can be distributed, False otherwise.
        """
        # 1) Number of processors must be less or equal the axis size, otherwise there are empty processors
        if axis_size < num_procs:
            print(
                f"Axis {axis_index}: Number of processors must be less or equal the axis size to avoid empty processors"
            )
            return False

        # 2.a) Block size must be greater or equal 0
        if block_size < 0:
            print(f"Axis {axis_index}: Block size must be greater or equal 0")
            return False

        # 2.b) Block size must ensure that each of the processors has access to at least one element
        max_block_size = Dist.maxBlockSize(axis_size, num_procs)
        if block_size > max_block_size:
            print(
                f"Axis {axis_index}: Block size is too big for the number of processors and tensor axis size"
            )
            return False

        return True

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
    def maxNumElements(self, shape: torch.Size) -> int:
        """Max number of elements held by a process, given a tensor shape."""
        raise NotImplementedError("Not implemented for abstract class")

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

    @abstractmethod
    def allGather(self, shape, mesh_axis: Optional[int] = None) -> tuple[Self, float]:
        """
        Return the distribution that results from gathering the tensor across the selected processor mesh axis.

        Parameters
        ----------
        shape : torch.Size
            The shape of the tensor.
        mesh_axis : int, optional
            The processor mesh axis, by default None, signifying an all-gather over all the processors.

        Returns
        -------
        Dist
            The distribution that results from the all-gather. None if the tensor is not compatible with the distribution.
        float
            The maximum expected communication cost.
        """
        log.warning("allGather not implemented for abstract class Dist")
        raise NotImplementedError()

    @abstractmethod
    def split(
        self,
        shape: torch.Size,
        tensor_axis: int,
        mesh_axis: int | tuple[int, ...],
        block_size=1,
    ) -> tuple[Self, float]:
        """
        Return the distribution that results from splitting the tensor across the selected tensor axis and processor mesh axis.

        Parameters
        ----------
        shape : torch.Size
            The shape of the tensor.
        tensor_axis : int
            The tensor axis to split.
        mesh_axis : int | tuple[int, ...]
            The processor mesh axis to split.
        block_size : int, optional
            The size of the blocks, by default 1.

        Returns
        -------
        Dist
            The distribution that results from the split. None if the tensor is not compatible with the distribution.
        float
            The maximum expected communication cost.
        """
        log.warning("split not implemented for abstract class Dist")
        raise NotImplementedError()

    @abstractmethod
    def permute(
        self, shape: torch.Size, mesh_axis: tuple[int, int]
    ) -> tuple[Self, float]:
        """
        Return the distribution that results from permuting the tensor across the selected processor mesh axes.

        Parameters
        ----------
        shape : torch.Size
            The shape of the tensor.
        mesh_axis : tuple[int, int]
            The processor mesh axes.

        Returns
        -------
        Dist
            The distribution that results from the permutation. None if the tensor is not compatible with the distribution.
        float
            The maximum expected communication cost.
        """
        log.warning("permute not implemented for abstract class Dist")
        raise NotImplementedError()

    @abstractmethod
    def all2all(
        self, shape: torch.Size, from_tensor_axis: int, to_tensor_axis: int, minor=True
    ) -> tuple[Self, float]:
        """
        Return the distribution that results from an all-to-all communication of the tensor across the selected tensor axes.

        Parameters
        ----------
        shape : torch.Size
            The shape of the tensor.
        from_tensor_axis : int
            The tensor axis to send data from.
        to_tensor_axis : int
            The tensor axis to receive data to.
        minor: bool, Default false
            If set to true, it will only exchange data between the minor distribution dimention of the source tensor axis.

        Returns
        -------
        Dist
            The distribution that results from the all-to-all communication. None if the tensor is not compatible with the distribution.
        float
            The maximum expected communication cost.
        """
        log.warning("all2all not implemented for abstract class Dist")
        raise NotImplementedError()
