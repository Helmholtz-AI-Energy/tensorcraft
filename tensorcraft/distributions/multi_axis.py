"""MultiAxisDist class."""
import math
from typing import TypeAlias

from tensorcraft.distributions.dist import Dist
from tensorcraft.util import multi2linearIndex, linear2multiIndex

import torch


DimsMapType: TypeAlias = tuple[tuple[int, ...], ...]
BlockSizesType: TypeAlias = tuple[int, ...]

class MultiAxisDist(Dist):

    def __init__(
        self, processor_mesh: int | torch.Size, dims_mapping: DimsMapType, block_sizes: BlockSizesType
    ) -> None:
        super().__init__(processor_mesh=processor_mesh)
        if len(dims_mapping) != len(block_sizes):
            raise ValueError("The number of dimensions and block sizes must match")
        for dim in dims_mapping:
            for mesh_axis in dim:
                if mesh_axis >= len(self._pmesh) or mesh_axis < 0:
                    raise ValueError("The dimension mapping is out of bounds")

        for block in block_sizes:
            if block < 0:
                raise ValueError("The block size must be greater or equal 0")

        self._dims_mapping = dims_mapping
        self._block_sizes = block_sizes


    def processorView(self, shape: torch.Size) -> torch.Tensor:
        """
        Get the processor view of a tensor.
        
        Parameters
        ----------
        shape : torch.Size
            The shape of the tensor.
        
        Returns
        -------
        torch.Tensor
            The processor view of the tensor.
        """
        if not self.compatible(shape):
            raise ValueError("The tensor is not compatible with the distribution")

        processor_view = torch.ones((*shape, self._pmesh.numel()), dtype=torch.bool)

        dist_axis_list = [self._distributeDim(i, shape[i]) for i in range(len(shape))]
        for i in range(shape.numel()):
            m_idx = linear2multiIndex(i, shape) + (None,)
            for j in range(len(shape)):
                processor_view[m_idx] &= dist_axis_list[j][m_idx[j], :]

        return processor_view

    def getElementLocation(self, shape, index):  # noqa: D102
        if isinstance(index, int):
            mindex = linear2multiIndex(index, shape)
        else:
            mindex = index

        p_list = torch.ones((self._pmesh.numel(),), dtype=torch.bool)
        for dim in range(len(shape)):
            distributed_dim = self._distributeDim(dim, shape[dim])
            p_list &= distributed_dim[mindex[dim], :]

        return p_list

    def compatible(self, shape):  # noqa: D102
        # Ensure that the tensor order and the number of dimensions in the distribution match
        if len(shape) != len(self._dims_mapping):
            print(
                f"Tensor order and the number of dimensions in the distribution must match: {len(shape)} != {len(self._dims_mapping)}"
            )
            return False

        # Ensure that the tensor dimensions are divisible by the block sizes
        if not all(
            [
                shape[dim] % block == 0
                for dim, (mapping, block) in enumerate(
                    zip(self._dims_mapping, self._block_sizes)
                )
                if len(mapping) > 0 and block > 0
            ]
        ):
            raise ValueError(
                "The tensor dimensions must be divisible by the block sizes"
            )

        # Block size must ensure that each of the mesh dimensions has access to at least one block
        for dim, block_size in enumerate(self._block_sizes):
            mesh_dims_idx = self._dims_mapping[dim]
            mesh_dims = [self._pmesh[i] for i in mesh_dims_idx]
            if block_size == 0:
                continue
            elif shape[dim] % block_size != 0:
                print(
                    f"Maximum block size exceeded for dimension {dim}: {block_size} > {torch.floor(shape[dim] / torch.prod(mesh_dims))}"
                )
                return False

        return True

    def _distributeDim(self, dim: int, dim_size: int) -> torch.Tensor:
        mesh_dims_idx = self._dims_mapping[dim]
        num_process = self._pmesh.numel()

        if len(mesh_dims_idx) == 0:
            return torch.ones((dim_size, num_process), dtype=torch.bool)

        dim_distribution = torch.ones((dim_size, num_process), dtype=torch.bool)
        mesh_dims = [self._pmesh[i] for i in mesh_dims_idx]
        block_size = self._block_sizes[dim]
        _, axis_chunk_ends = self.axisSplits(dim_size, block_size, math.prod(mesh_dims))  # type: ignore
        start_idx = 0
        for chunk_idx, end_idx in enumerate(axis_chunk_ends):
            left_eq = chunk_idx % math.prod(mesh_dims)
            for j in range(num_process):
                p_mi = self.getProcessorMultiIndex(j)
                right_eq = multi2linearIndex(
                    self._pmesh, p_mi, order=mesh_dims_idx
                ) % math.prod(mesh_dims)  # type: ignore
                dim_distribution[start_idx:end_idx, j] = left_eq == right_eq

            start_idx = end_idx

        return dim_distribution
