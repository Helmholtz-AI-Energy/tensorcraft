"""MultiAxisDist class."""

import logging
import math
from typing import Optional, TypeAlias

import torch

from tensorcraft.distributions.dist import Dist
from tensorcraft.distributions.util import allgather_bandwidth_cost
from tensorcraft.util import linear2multiIndex, multi2linearIndex

log = logging.getLogger("tensorcraft")

DimsMapType: TypeAlias = tuple[Optional[tuple[int, ...]], ...]
BlockSizesType: TypeAlias = int | tuple[int, ...]


class MultiAxisDist(Dist):
    """
    Distribution scheme that distributes a tensor over multiple axes of the processor mesh.

    Parameters
    ----------
    processor_mesh : int | torch.Size
        The processor mesh.
    dims_mapping : DimsMapType
        The mapping of tensor dimensions to processor mesh axes.
    block_sizes : BlockSizesType
        The block sizes for each dimension.

    Raises
    ------
    ValueError
        If the number of dimensions and block sizes do not match.
    ValueError
        If the dimension mapping is out of bounds.
    ValueError
        If the processor mesh axis is repeated.
    """

    def __init__(
        self,
        processor_mesh: int | torch.Size,
        dims_mapping: DimsMapType,
        block_sizes: BlockSizesType,
    ) -> None:
        super().__init__(processor_mesh=processor_mesh)
        if isinstance(block_sizes, int):
            block_sizes = (block_sizes,) * len(dims_mapping)

        if len(dims_mapping) != len(block_sizes):
            raise ValueError("The number of dimensions and block sizes must match")

        # Check if the dimension mapping is out of bounds and fill in the missing dimensions with empty tuples
        for dim in dims_mapping:
            if dim:
                for mesh_axis in dim:
                    if mesh_axis >= len(self._pmesh) or mesh_axis < 0:
                        raise ValueError("The dimension mapping is out of bounds")

        for block in block_sizes:
            if block < 0:
                raise ValueError("The block size must be greater or equal 0")

        # Check that no processor mesh axis is repeated
        mesh_dims = []
        for dim_mapping in dims_mapping:
            if dim_mapping:
                for axis in dim_mapping:
                    if axis in mesh_dims:
                        raise ValueError("The processor mesh axis must not be repeated")
                    mesh_dims.append(axis)

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

    def isDistributed(self) -> bool:
        """
        Check if the distribution is distributed.

        Returns
        -------
        bool
            True if the distribution is distributed, False otherwise.
        """
        dist = False
        for axis_mapping in self._dims_mapping:
            if axis_mapping and len(axis_mapping) > 0:
                dist = True
                break
        return dist

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
        if len(shape) != len(self._dims_mapping) or len(shape) != len(
            self._block_sizes
        ):
            print(
                f"Tensor order and the number of dimensions in the distribution must match: {len(shape)} != {len(self._dims_mapping)}"
            )
            return False

        # Check for axis to processor mesh compatibility
        for axis, (axis_size, assigned_dims, block_size) in enumerate(
            zip(shape, self._dims_mapping, self._block_sizes)
        ):
            if not assigned_dims:
                # Axis is not distributed
                continue
            mesh_dims = [self._pmesh[i] for i in assigned_dims]
            if not self.compatibleAxis(
                axis, axis_size, block_size, math.prod(mesh_dims)
            ):
                print(f"Axis {axis}: Incompatible axis with distribution scheme")
                return False
        return True

    def _distributeDim(self, dim: int, dim_size: int) -> torch.Tensor:
        mesh_dims_idx = self._dims_mapping[dim]
        num_process = self._pmesh.numel()

        if not mesh_dims_idx or len(mesh_dims_idx) == 0:
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

    def _max_block_size_n_blocks(self, shape):
        max_block_size = 1
        max_n_blocks = 1
        for i in range(len(shape)):
            mesh_dims_idx = self._dims_mapping[i]
            if not mesh_dims_idx or len(mesh_dims_idx) == 0:
                max_block_size *= shape[i]
                continue
            mesh_dims = [self._pmesh[i] for i in mesh_dims_idx]
            n_procs_axis = math.prod(mesh_dims)
            axis_splits, _ = self.axisSplits(
                shape[i], self._block_sizes[i], n_procs_axis
            )
            max_block_size *= max(axis_splits)
            max_n_blocks *= math.ceil(len(axis_splits) / n_procs_axis)
        return max_block_size, max_n_blocks

    def allGather(self, shape, mesh_axis=Optional[int]):  # noqa: D102
        if not self.isDistributed():
            log.warning(
                "The original distribution is not distributed, nothing is done."
            )
            return self, 0

        if not self.compatible(shape):
            raise ValueError("The tensor is not compatible with the distribution")

        if mesh_axis is None:
            # Results in full replication of the tensor on all processors
            new_dist = MultiAxisDist(
                self._pmesh, ((),) * len(shape), (0,) * len(shape)
            )  #

            # Cost of all-gather is the number of processors
            max_block_size, max_n_blocks = self._max_block_size_n_blocks(shape)
            involved_procs = math.prod(self._pmesh)

            return new_dist, allgather_bandwidth_cost(
                involved_procs, max_block_size * max_n_blocks
            )
        else:
            # Check that the mesh axis is valid
            tensor_axis = -1
            for axis, mappings in enumerate(self._dims_mapping):
                if mesh_axis in mappings:
                    dist_axis_i = mappings.index(mesh_axis)

                    print(f"Mesh axis idx: {dist_axis_i}")
                    if dist_axis_i != 0 and dist_axis_i != len(mappings) - 1:
                        log.warning(
                            f"Gather along axis {mesh_axis} leads to an undefined data distribution. Can only gather along the first or last axis withing a dimmension mapping."
                        )
                        return self, 0
                    tensor_axis = axis
                    break

            if tensor_axis == -1:
                log.warning(
                    "Tensor is not distributed along the specified axis, doing nothing"
                )
                return self, 0

            log.debug(f"Tensor axis: {tensor_axis}")
            log.debug(f"Mesh axis: {mesh_axis}")
            log.debug(f"Mappings: {mappings}")

            involved_procs = self._pmesh[mesh_axis]

            if dist_axis_i != 0:
                new_block_sizes = list(self._block_sizes)
                new_block_sizes[tensor_axis] *= involved_procs
            else:
                new_block_sizes = list(self._block_sizes)

            new_mapping = tuple(
                tuple(m_axis for m_axis in axis_mappings if m_axis != mesh_axis)
                if axis_mappings
                else None
                for axis_mappings in self._dims_mapping
            )

            new_dist = MultiAxisDist(self._pmesh, new_mapping, new_block_sizes)

            # Cost of all-gather is the number of processors
            max_block_size, max_n_blocks = self._max_block_size_n_blocks(shape)

            return new_dist, allgather_bandwidth_cost(
                involved_procs, max_block_size * max_n_blocks
            )

    def split(self, shape, tensor_axis, mesh_axis, block_size=1):  # noqa: D102
        # This will just split a along the scatter axis
        if not self.compatible(shape):
            raise ValueError("The tensor is not compatible with the distribution")

        if isinstance(mesh_axis, int):
            mesh_axis = (mesh_axis,)

        # When tensor is not distributed or the tensor axis is not split, simply apply
        if not self.isDistributed() or len(self._dims_mapping[tensor_axis]) == 0:
            new_dims_mapping = tuple(
                mesh_axis if i == tensor_axis else mapping
                for i, mapping in enumerate(self._dims_mapping)
            )
            new_block_size = tuple(
                block_size if i == tensor_axis else b_size
                for i, b_size in enumerate(self._block_sizes)
            )
            new_dist = MultiAxisDist(self._pmesh, new_dims_mapping, new_block_size)
        else:
            # Spliting an already split axis. New split mesh axis gets appended on the left (mayor), and block size is not changed.
            if block_size != 1:
                log.warning(
                    "Spliting an already split axis. Block size argument will be ignored."
                )
            new_dims_mapping = tuple(
                mesh_axis + mapping if i == tensor_axis else mapping
                for i, mapping in enumerate(self._dims_mapping)
            )
            new_block_size = self._block_sizes
            new_dist = MultiAxisDist(self._pmesh, new_dims_mapping, new_block_size)
        # Check that new dist is compatible with the tensor shape
        if not new_dist.compatible(shape):
            raise ValueError(
                "Tensor shape cannot be split with the given axis mapping and block size."
            )
        return new_dist, 0

    def reduce_scatter(self, shape, scatter_axis=None):  # noqa: D102
        raise NotImplementedError("Reduce scatter is not implemented for MultiAxisDist")

    def permute(self, shape, mesh_axis):  # noqa: D102
        raise NotImplementedError("Permute is not implemented for MultiAxisDist")

    def all2all(self, shape, from_tensor_axis, to_tensor_axis):  # noqa: D102
        raise NotImplementedError("All2ll is not implemented for MultiAxisDist")

    def __str__(self):
        return (
            f"MultiAxisDist({self._pmesh}, {self._dims_mapping}, {self._block_sizes})"
        )
