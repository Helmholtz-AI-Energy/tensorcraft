"""MultiAxisDist class."""

import itertools
import logging
import math
from typing import Any, Optional, TypeAlias

import torch
import torch.nn.functional as F
from typing_extensions import override

from tensorcraft.distributions.dist import Dist
from tensorcraft.util.axis_utils import linear2multiIndex, multi2linearIndex

log = logging.getLogger("tensorcraft")

DimsMapType: TypeAlias = tuple[tuple[int, ...], ...]
BlockSizesType: TypeAlias = tuple[int, ...]


class MultiAxisDist(Dist):
    """
    Distribution scheme that distributes a tensor over multiple axes of the processor mesh.

    Parameters
    ----------
    processor_mesh : int | torch.Size
        The processor mesh.
    dims_mapping : DimsMapType
        The mapping of tensor dimensions to processor mesh axes.
    block_sizes : int | Block
        The block sizes for each dimension. Block size must be greater than zero. To leave a block size empty, use either -1 or None. If left empty, but the corresponding axis is split, a default block size of 1 will be used.

    Raises
    ------
    ValueError
        If the number of dimensions and block sizes do not match.
    ValueError
        If the dimension mapping is out of bounds.
    ValueError
        If the processor mesh axis is repeated.
    """

    __slots__ = ("_dims_mapping", "_block_sizes")

    def __init__(
        self,
        processor_mesh: int | torch.Size,
        dims_mapping: tuple[Optional[tuple[int, ...]], ...],
        block_sizes: int | tuple[Optional[int], ...],
    ):
        super().__init__(processor_mesh=processor_mesh)
        if isinstance(block_sizes, int):
            block_sizes = (block_sizes,) * len(dims_mapping)

        if isinstance(processor_mesh, int):
            processor_mesh = torch.Size([processor_mesh])

        if len(processor_mesh) == 0:
            raise ValueError("The processor mesh must have at least one dimension")

        if len(dims_mapping) != len(block_sizes):
            raise ValueError("The number of dimensions and block sizes must match")

        # Check if the dimension mapping is out of bounds and fill in the missing dimensions with empty tuples
        for dim in dims_mapping:
            if dim:
                for mesh_axis in dim:
                    if mesh_axis >= len(self._pmesh) or mesh_axis < 0:
                        raise ValueError("The dimension mapping is out of bounds")

        for block in block_sizes:
            if block is not None:
                if block <= 0 and block != -1:
                    raise ValueError(
                        "The block size must be None, greater than zero or -1"
                    )

        # Check that no processor mesh axis is repeated
        mesh_dims = []
        dims_mapping_list: list[tuple[int, ...]] = []
        block_sizes_list: list[int] = []
        for dim_mapping, b_size in zip(dims_mapping, block_sizes):
            if dim_mapping:
                for axis in dim_mapping:
                    if axis in mesh_dims:
                        raise ValueError("The processor mesh axis must not be repeated")
                    mesh_dims.append(axis)
                dims_mapping_list.append(dim_mapping)
                block_sizes_list.append(b_size if b_size and b_size > 0 else 1)
            else:
                dims_mapping_list.append(())
                block_sizes_list.append(-1)

        self._dims_mapping: DimsMapType = tuple(dims_mapping_list)
        self._block_sizes: BlockSizesType = tuple(block_sizes_list)

    @property
    def dimsMapping(self) -> DimsMapType:
        """
        Get the dimension mapping.

        Returns
        -------
        DimsMapType
            The dimension mapping.
        """
        return self._dims_mapping

    @property
    def blockSizes(self) -> tuple[int, ...]:
        """
        Get the block sizes.

        Returns
        -------
        tuple[int, ...]
            The block sizes.
        """
        return self._block_sizes

    def __eq__(self, other: Any) -> bool:
        base_comp = super().__eq__(other)
        if base_comp and isinstance(other, MultiAxisDist):
            return (self._dims_mapping == other._dims_mapping) and (
                self._block_sizes == other._block_sizes
            )
        else:
            return False

    def __str__(self) -> str:
        mesh_str = ",".join([str(x) for x in self._pmesh])
        map_str = "{"
        for m in self._dims_mapping:
            if len(m) == 0:
                map_str += "∅,"
            elif len(m) == 1:
                map_str += f"{m[0]},"
            else:
                map_str += f"({','.join([str(x) for x in m])}),"
        map_str = map_str[:-1] + "}"
        b_str = ",".join([str(x) if x != -1 else "∅" for x in self._block_sizes])
        return f"D_[{mesh_str}]⊥{map_str}({b_str})"

    def latexStr(self) -> str:  # noqa: D102
        map_str = ""
        for m in self._dims_mapping:
            if len(m) == 0:
                map_str += "\\emptyset,"
            elif len(m) == 1:
                map_str += f"{m[0]},"
            else:
                map_str += f"({','.join([str(x) for x in m])}),"
        b_str = ",".join(
            [str(x) if x != -1 else "\\emptyset" for x in self._block_sizes]
        )
        return f"$T_{{\\perp\\{{ {map_str[:-1]} \\}}({b_str})}}$"

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
            m_idx = tuple(linear2multiIndex(i, shape)) + (None,)
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

    def getElementLocation(  # noqa: D102
        self, shape: torch.Size, index: int | torch.Size
    ) -> torch.Tensor:
        if isinstance(index, int):
            mindex = linear2multiIndex(index, shape)
        else:
            mindex = index

        p_list = torch.ones((self._pmesh.numel(),), dtype=torch.bool)
        for dim in range(len(shape)):
            distributed_dim = self._distributeDim(dim, shape[dim])
            p_list &= distributed_dim[mindex[dim], :]

        return p_list

    def compatible(self, shape: torch.Size) -> bool:  # noqa: D102
        # Ensure that the tensor order and the number of dimensions in the distribution match
        if len(shape) != len(self._dims_mapping) or len(shape) != len(
            self._block_sizes
        ):
            log.debug(
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
                log.debug(f"Axis {axis}: Incompatible axis with distribution scheme")
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
        _, axis_chunk_ends = self.axisSplits(dim_size, block_size, math.prod(mesh_dims))
        start_idx = 0
        for chunk_idx, end_idx in enumerate(axis_chunk_ends):
            left_eq = chunk_idx % math.prod(mesh_dims)
            for j in range(num_process):
                p_mi = self.getProcessorMultiIndex(j)
                right_eq = multi2linearIndex(
                    self._pmesh, p_mi, order=mesh_dims_idx
                ) % math.prod(mesh_dims)
                dim_distribution[start_idx:end_idx, j] = left_eq == right_eq

            start_idx = end_idx

        return dim_distribution

    def localShape(self, global_shape: torch.Size, rank: int) -> torch.Size:
        """
        Get the local shape of the tensor on the given rank.

        Parameters
        ----------
        global_shape : torch.Size
            The global shape of the tensor.
        rank : int
            The rank of the processor.

        Returns
        -------
        torch.Size
            The local shape of the tensor on the given rank.
        """
        # Check if global shape is compatible with the distribution
        if not self.compatible(global_shape):
            raise ValueError("The global shape is not compatible with the distribution")

        # Get the processor multi index for the given rank
        p_midx = self.getProcessorMultiIndex(rank)

        local_shape = [0] * len(global_shape)
        for axis, (axis_size, mapping, block_size) in enumerate(
            zip(global_shape, self._dims_mapping, self._block_sizes)
        ):
            if mapping and len(mapping) > 0:
                mesh_dims = [self._pmesh[i] for i in mapping]
                n_procs = math.prod(mesh_dims)

                axis_idx = multi2linearIndex(self._pmesh, p_midx, order=mapping)
                axis_splits, _ = self.axisSplits(axis_size, block_size, n_procs)
                local_shape[axis] = sum(axis_splits[axis_idx::n_procs])
            else:
                local_shape[axis] = axis_size

        return torch.Size(local_shape)

    def maxNumElements(self, shape: torch.Size) -> int:  # noqa: D102
        rank_0_shape = self.localShape(shape, 0)
        return math.prod(rank_0_shape)

    def allgather(
        self, shape: torch.Size, gather_dim: Optional[int] = None
    ) -> tuple["MultiAxisDist", int, int]:
        """
        Return the distribution that results from gathering the tensor across the selected processor mesh axis.

        Parameters
        ----------
        shape : torch.Size
            The shape of the tensor.
        mesh_axis : int, optional
            The processor mesh axis, by default None, signifying an allgather over all the processors.

        Returns
        -------
        MultiAxisDist
            The distribution that results from the all-gather. None if the tensor is not compatible with the distribution.
        int
            The maximum expected communication volume (n_elements).
        int
            The number of involved processes in each sub communicator.
        """
        if not self.isDistributed():
            log.debug("The original distribution is not distributed, nothing to do.")
            return self, 0, 0

        if not self.compatible(shape):
            raise ValueError("The tensor is not compatible with the distribution")

        if gather_dim is None:
            # Results in full replication of the tensor on all processors
            new_dist = MultiAxisDist(
                self._pmesh, ((),) * len(shape), (0,) * len(shape)
            )  #

            # Cost of all-gather is the number of processors
            comm_volume = self.maxNumElements(shape) * involved_procs
            involved_dims = [
                self._pmesh[mesh_dim]
                for axis_map in self._dims_mapping
                if len(axis_map) != 0
                for mesh_dim in axis_map
            ]
            involved_procs = math.prod(involved_dims)

            return (
                new_dist,
                comm_volume,
                involved_procs,
            )
        else:
            # Check that the mesh axis is valid
            tensor_axis = -1
            for axis, mappings in enumerate(self._dims_mapping):
                if gather_dim in mappings:
                    dist_axis_i = mappings.index(gather_dim)

                    if dist_axis_i != 0 and dist_axis_i != len(mappings) - 1:
                        log.debug(
                            f"Gather along axis {gather_dim} leads to an undefined data distribution. Can only gather along the first or last axis withing a dimmension mapping."
                        )
                        raise ValueError(
                            "Gather along axis leads to an undefined data distribution"
                        )

                    tensor_axis = axis
                    break

            if tensor_axis == -1:
                log.debug(
                    "Tensor is not distributed along the specified axis, doing nothing"
                )
                raise ValueError(
                    "The tensor is not distributed along the specified axis"
                )

            log.debug(f"Tensor axis: {tensor_axis}")
            log.debug(f"Mesh axis: {gather_dim}")
            log.debug(f"Mappings: {mappings}")

            involved_procs = self._pmesh[gather_dim]

            if dist_axis_i != 0:
                new_block_sizes = list(self._block_sizes)
                new_block_sizes[tensor_axis] *= involved_procs
            else:
                new_block_sizes = list(self._block_sizes)

            new_mapping = tuple(
                tuple(m_axis for m_axis in axis_mappings if m_axis != gather_dim)
                if axis_mappings
                else None
                for axis_mappings in self._dims_mapping
            )

            new_dist = MultiAxisDist(self._pmesh, new_mapping, tuple(new_block_sizes))

            # Cost of all-gather is the number of processors

            comm_volume = self.maxNumElements(shape) * involved_procs

            return new_dist, comm_volume, involved_procs

    def split(
        self,
        shape: torch.Size,
        tensor_axis: int,
        mesh_dims: int | tuple[int, ...],
        block_size: int = 1,
        minor: bool = False,
    ) -> tuple["MultiAxisDist", int, int]:
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
        minor : bool, optional
            If set to true, the mesh dimensions will be appended to the right (minor), by default False. If the axis is already split, it will change the existing block size.

        Returns
        -------
        MultiAxisDist
            The distribution that results from the split. None if the tensor is not compatible with the distribution.
        int
            The maximum expected communication volume (n_elements).
        int
            The number of involved processes in each sub communicator.
        """
        # This will just split a along the scatter axis
        if not self.compatible(shape):
            raise ValueError("The tensor is not compatible with the distribution")

        if isinstance(mesh_dims, int):
            mesh_dims = (mesh_dims,)

        # When tensor is not distributed or the tensor axis is not split, simply apply
        if not self.isDistributed() or len(self._dims_mapping[tensor_axis]) == 0:
            new_dims_mapping = tuple(
                mesh_dims if i == tensor_axis else mapping
                for i, mapping in enumerate(self._dims_mapping)
            )
            new_block_size_list = [
                block_size if i == tensor_axis else b_size
                for i, b_size in enumerate(self._block_sizes)
            ]
            new_dist = MultiAxisDist(
                self._pmesh, new_dims_mapping, tuple(new_block_size_list)
            )
        else:
            # Spliting an already split axis. New split mesh axis gets appended on the left (mayor), and block size is not changed.
            if block_size != 1:
                log.debug(
                    "Spliting an already split axis. Block size argument will be ignored."
                )

            if minor:
                new_dims_mapping_list = [
                    mapping + mesh_dims if i == tensor_axis else mapping
                    for i, mapping in enumerate(self._dims_mapping)
                ]
                new_block_size_list = list(self._block_sizes)
                previous_block_size = new_block_size_list[tensor_axis]

                involved_prods = math.prod([self._pmesh[x] for x in mesh_dims])
                if previous_block_size % involved_prods != 0:
                    raise ValueError(
                        "Incompatible resulting block size. The block size of the newly split tensor axis must be divisible by the involved mesh dimentions"
                    )
                new_block_size_list[tensor_axis] = previous_block_size // involved_prods
                new_dist = MultiAxisDist(
                    self._pmesh,
                    tuple(new_dims_mapping_list),
                    tuple(new_block_size_list),
                )

            else:
                new_dims_mapping_list = [
                    mesh_dims + mapping if i == tensor_axis else mapping
                    for i, mapping in enumerate(self._dims_mapping)
                ]
                new_block_size_list = list(self._block_sizes)
                new_dist = MultiAxisDist(
                    self._pmesh,
                    tuple(new_dims_mapping_list),
                    tuple(new_block_size_list),
                )

        # Check that new dist is compatible with the tensor shape
        if not new_dist.compatible(shape):
            raise ValueError(
                "Tensor shape cannot be split with the given axis mapping and block size."
            )
        return new_dist, 0, 0

    def permute(
        self, shape: torch.Size, mesh_dims: tuple[int, int]
    ) -> tuple["MultiAxisDist", int, int]:
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
        MultiAxisDist
            The distribution that results from the permutation. None if the tensor is not compatible with the distribution.
        int
            The maximum expected communication volume (n_elements).
        int
            The number of involved processes in each sub communicator.
        """
        # check for compatibility and if it is distributed
        if not self.compatible(shape):
            raise ValueError(
                "Provided shape is not compitble with original distribution"
            )

        if not self.isDistributed():
            raise ValueError("Operation not allowed on non-distributed tensors.")

        # Swaping mesh axis should have same size should have matching sizes
        if self._pmesh[mesh_dims[0]] != self._pmesh[mesh_dims[1]]:
            raise ValueError("Both mesh dimentions need to be on the same size.")

        new_dims_mapping: tuple[tuple[int, ...] | None, ...] = ()
        for i, axis_dims in enumerate(self._dims_mapping):
            if axis_dims:
                axis_dims_list = list(axis_dims)
                if mesh_dims[0] in axis_dims_list:
                    axis_dims_list[axis_dims.index(mesh_dims[0])] = mesh_dims[1]
                if mesh_dims[1] in axis_dims_list:
                    axis_dims_list[axis_dims.index(mesh_dims[1])] = mesh_dims[0]
                new_dims_mapping += (tuple(axis_dims_list),)
            else:
                new_dims_mapping += (axis_dims,)

        new_dim = MultiAxisDist(self._pmesh, new_dims_mapping, self._block_sizes)

        comm_volume = self.maxNumElements(shape)
        n_procs = 2

        return new_dim, comm_volume, n_procs

    def alltoall(
        self,
        shape: torch.Size,
        from_tensor_axis: int,
        to_tensor_axis: int,
        block_size: int = -1,
        minor: bool = False,
    ) -> tuple["MultiAxisDist", int, int]:
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
        block_size: int, Default -1
            If possible, the selected block size will be applied, otherwise, the block size of the reciving axis will remain unchanged. -1 will try to apply the block size of the source axis.
        minor: bool, Default false
            If set to true, it will only exchange data between the minor distribution dimention of the source tensor axis.

        Returns
        -------
        MultiAxisDist
            The distribution that results from the all-to-all communication. None if the tensor is not compatible with the distribution.
        int
            The maximum expected communication volume (n_elements).
        int
            The number of involved processes in each sub communicator.
        """
        if not self.isDistributed() or len(self._dims_mapping[from_tensor_axis]) == 0:
            raise ValueError("From axis needs to be distributed.")
        if not self.compatible(shape):
            raise ValueError("Not compatible starting shape for starting distribution.")

        new_dims_map_list = list(self._dims_mapping)
        new_block_size_list = list(self._block_sizes)

        block_size = (
            self._block_sizes[from_tensor_axis] if block_size < 0 else block_size
        )

        if minor:
            moved_mesh_dims: tuple[int, ...] = (
                self._dims_mapping[from_tensor_axis][-1],
            )
            relevant_proc = self._pmesh[moved_mesh_dims[0]]
            n_procs = relevant_proc

            # Find out new dims mapping
            new_from_t_axis_mapping = self._dims_mapping[from_tensor_axis][:-1]
            new_to_t_axis_mapping = self._dims_mapping[to_tensor_axis] + moved_mesh_dims

            new_dims_map_list[from_tensor_axis] = new_from_t_axis_mapping
            new_dims_map_list[to_tensor_axis] = new_to_t_axis_mapping

            new_from_t_axis_bs = (
                self._block_sizes[from_tensor_axis] * n_procs
                if new_dims_map_list[from_tensor_axis] != ()
                else -1
            )

            to_t_axis_bs = (
                self._block_sizes[to_tensor_axis]
                if self._dims_mapping[to_tensor_axis] != ()
                else shape[to_tensor_axis]
            )

            if to_t_axis_bs % n_procs != 0:
                raise ValueError(
                    "Incompatible resulting block size. The block size of the newly split tensor axis must be divisible by the involved mesh dimentions"
                )
            new_to_t_axis_bs = (
                to_t_axis_bs // n_procs
                if self._dims_mapping[to_tensor_axis] != ()
                else block_size
            )
        else:
            moved_mesh_dims = self._dims_mapping[from_tensor_axis]
            relevant_procs = [self._pmesh[x] for x in moved_mesh_dims]
            n_procs = math.prod(relevant_procs)

            # Find out new dims mapping
            new_dims_map_list[from_tensor_axis] = ()
            new_dims_map_list[to_tensor_axis] = (
                moved_mesh_dims + self._dims_mapping[to_tensor_axis]
            )

            new_from_t_axis_bs = -1

            new_to_t_axis_bs = (
                self._block_sizes[to_tensor_axis]
                if self._dims_mapping[to_tensor_axis] != ()
                else block_size
            )

        new_block_size_list[from_tensor_axis] = new_from_t_axis_bs
        new_block_size_list[to_tensor_axis] = new_to_t_axis_bs

        new_dist = MultiAxisDist(
            self._pmesh, tuple(new_dims_map_list), tuple(new_block_size_list)
        )

        if not new_dist.compatible(shape):
            raise ValueError(
                "Tensor shape cannot be redistributed with the given axis mapping and block size."
            )

        # Communication volume
        comm_volume = self.maxNumElements(shape)
        return new_dist, comm_volume * n_procs, n_procs

    def change_block_size(
        self, shape: torch.Size, tensor_axis: int, block_size: int
    ) -> tuple["MultiAxisDist", int, int]:
        """
        Change the blocks size of a distributed tensor axis.

        Parameters
        ----------
        shape : torch.Size
            The shape of the tensor.
        tensor_axis : int
            The target tensor axis.
        block_size : int
            Desired block size.

        Returns
        -------
        MultiAxisDist
            The distribution that results from the all-to-all communication. None if the tensor is not compatible with the distribution.
        int
            The maximum expected communication volume (n_elements).
        int
            The number of involved processes in each sub communicator.
        """
        if not self.isDistributed() or len(self._dims_mapping[tensor_axis]) == 0:
            raise ValueError("From axis needs to be distributed.")
        if not self.compatible(shape):
            raise ValueError("Not compatible starting shape for starting distribution.")

        new_block_size_list = list(self._block_sizes)

        moved_mesh_dims = self._dims_mapping[tensor_axis]
        involved_dims_sizes = [self._pmesh[x] for x in moved_mesh_dims]
        n_procs = math.prod(involved_dims_sizes)

        old_block_size = new_block_size_list[tensor_axis]
        if old_block_size == block_size:
            log.info(f"Block size is already {old_block_size}")
            raise ValueError("Invalid arguments")

        if block_size > old_block_size and block_size % old_block_size != 0:
            log.info("New block size must be a multiple of the old block size")
            raise ValueError("Invalid arguments")
        elif block_size < old_block_size and old_block_size % block_size != 0:
            log.info("Old block size must be a multiple of the new block size")
            raise ValueError("Invalid arguments")

        ## TODO: Very simplified shit. If implemented like this it would be extremely inefficient, as some of the redistributions would have a lot of empty alltoallw buffers.
        new_block_size_list[tensor_axis] = block_size
        new_dist = MultiAxisDist(
            self.processorMesh, self._dims_mapping, tuple(new_block_size_list)
        )

        if not new_dist.compatible(shape):
            log.info("Tensor shape not compatible with new block size.")
            raise ValueError("Incompatible redistribution.")

        comm_volume = self.maxNumElements(shape)
        return new_dist, comm_volume, n_procs

    @override
    def neighbours(
        self, shape: torch.Size, prefered_b_size: list[int] = []
    ) -> list[tuple[str, "MultiAxisDist", int, int]]:  # noqa: D102
        neighbours = []

        # Free dims:
        free_dims = []
        for dim in range(len(self._pmesh)):
            is_free = True
            for i in range(len(shape)):
                if dim in self._dims_mapping[i]:
                    is_free = False
                    break

            if is_free:
                free_dims.append(dim)

        # split
        if prefered_b_size:
            b_sizes = prefered_b_size if 1 in prefered_b_size else [1] + prefered_b_size
        else:
            b_sizes = [1]
        for free_dim in free_dims:
            for b_size in b_sizes:
                for axis in range(len(shape)):
                    operation = f"split_{axis}_{free_dim}_{b_size}"
                    try:
                        new_dist, _, _ = self.split(shape, axis, free_dim, b_size)
                        log.debug(f"Landed in {new_dist}")

                        neighbours.append((operation, new_dist, 0, 0))

                    except Exception:
                        log.debug(
                            f"Failed operation {operation} on dist {self} with shape {shape}"
                        )

                    operation = f"split_minor_{axis}_{free_dim}_{b_size}"
                    try:
                        new_dist, _, _ = self.split(shape, axis, free_dim, b_size, True)
                        log.debug(f"Landed in {new_dist}")

                        neighbours.append((operation, new_dist, 0, 0))

                    except Exception:
                        log.debug(
                            f"Failed operation {operation} on dist {self} with shape {shape}"
                        )

        # allgather
        try:
            operation = "allgather_*"
            new_dist, vol, n_procs = self.allgather(shape)
            log.debug(f"New neighbour: {new_dist}")

            neighbours.append((operation, new_dist, vol, n_procs))

        except Exception:
            log.debug(f"Failed operation {operation} on dist {self} with shape {shape}")

        non_free_dims = list(set(range(len(self._pmesh))) - set(free_dims))
        for non_free_dim in non_free_dims:
            operation = f"allgather_{non_free_dim}"
            try:
                new_dist, vol, n_procs = self.allgather(shape, non_free_dim)
                log.debug(f"New neighbour: {new_dist}")

                neighbours.append((operation, new_dist, vol, n_procs))
            except Exception:
                log.debug(
                    f"Failed operation {operation} on dist {self} with shape {shape}"
                )

        # permutation
        for axis in range(len(shape)):
            mapping = self._dims_mapping[axis]
            if len(mapping) > 1:
                possible_permutations = itertools.combinations(mapping, 2)
                for combination in possible_permutations:
                    operation = f"permute_{combination[0]}_{combination[1]}"
                    try:
                        new_dist, vol, n_procs = self.permute(shape, combination)
                        log.debug(f"New neighbour: {new_dist}")

                        neighbours.append((operation, new_dist, vol, n_procs))
                    except Exception:
                        log.debug(
                            f"Failed operation {operation} on dist {self} with shape {shape}"
                        )

        # alltoall
        for source_axis in range(len(shape)):
            if len(self._dims_mapping[source_axis]) != 0:
                for target_axis in range(len(shape)):
                    for b_size in [-1] + b_sizes:
                        operation = f"alltoall_{source_axis}_{target_axis}_{b_size}"
                        try:
                            new_dist, vol, n_procs = self.alltoall(
                                shape, source_axis, target_axis, block_size=b_size
                            )
                            log.debug(f"New neighbour: {new_dist}")

                            neighbours.append((operation, new_dist, vol, n_procs))
                        except Exception:
                            log.debug(
                                f"Failed operation {operation} on dist {self} with shape {shape}"
                            )

                        if len(self._dims_mapping[source_axis]) > 1:
                            operation = (
                                f"alltoall_minor_{source_axis}_{target_axis}_{b_size}"
                            )
                            try:
                                new_dist, vol, n_procs = self.alltoall(
                                    shape, source_axis, target_axis, b_size, True
                                )
                                log.debug(f"New neighbour: {new_dist}")

                                neighbours.append((operation, new_dist, vol, n_procs))

                            except Exception:
                                log.debug(
                                    f"Failed operation {operation} on dist {self} with shape {shape}"
                                )

        # change_block_size
        for axis in range(len(shape)):
            for b_size in b_sizes:
                operation = f"changeBlockSize_{axis}_{b_size}"
                try:
                    new_dist, vol, n_procs = self.change_block_size(shape, axis, b_size)
                    log.debug(f"New neighbour: {new_dist}")
                    neighbours.append((operation, new_dist, vol, n_procs))
                except Exception:
                    log.debug(
                        f"Failed operation {operation} on dist {self} with shape {shape}"
                    )
        return neighbours

    def apply(self, tensor: torch.Tensor, rank: int) -> torch.Tensor:
        """
        Apply the distribution to a tensor, assuming that it is replicated on all processors.

        Parameters
        ----------
        tensor : torch.Tensor
            The tensor to distribute.
        rank : int
            The rank of the processor.

        Returns
        -------
        torch.Tensor
            The distributed tensor.
        """
        if not self.compatible(tensor.shape):
            raise ValueError("The tensor is not compatible with the distribution")

        if not self.isDistributed():
            log.info("Tensor is not distributed, nothing to do.")
            return tensor

        if (0 <= rank < self.numProcessors) is False:
            raise ValueError("Rank must be in the range of the processor mesh.")

        p_midx = self.getProcessorMultiIndex(rank)
        log.info(f"R{rank}: Processor multi index: {p_midx}")

        og_shape = tensor.shape
        dims = len(og_shape)

        missing_elements = []
        n_full_blocks_per_axis = []
        for s, b in zip(tensor.shape, self._block_sizes):
            missing_elements.append(b - (s % b) if b != -1 else 0)
            n_full_blocks_per_axis.append(s // b if b != -1 else 1)
        log.info(f"R{rank}: Missing elements: {missing_elements}")
        log.info(f"R{rank}: N blocks per axis: {n_full_blocks_per_axis}")
        tensor = F.pad(
            tensor,
            [value for me in missing_elements[::-1] for value in (0, me)],
            value=0,
        )
        log.info(f"R{rank}: Padded tensor shape: {tensor.shape}")

        reshape_list = []
        permute_tuple: tuple[int, ...] = ()
        tile_slices = []

        for size, b_size, dim_map in zip(
            tensor.shape, self._block_sizes, self._dims_mapping
        ):
            if b_size == -1:
                reshape_list += [1, size]
            else:
                reshape_list += [size // b_size, b_size]
            permute_tuple += (len(reshape_list) - 1,)

            if len(dim_map) == 0:
                tile_slices.append(slice(None))
            else:
                idx = multi2linearIndex(
                    self._pmesh,
                    p_midx,
                    order=dim_map,
                )
                n_procs = math.prod([self._pmesh[x] for x in dim_map])

                tile_slices.append(slice(idx, None, n_procs))

        permute_tuple = (
            tuple(set(range(len(reshape_list))) - set(permute_tuple)) + permute_tuple
        )
        log.info(f"R{rank}: Permute tuple: {permute_tuple}")
        log.info(f"R{rank}: Reshape tuple: {reshape_list}")

        tensor = tensor.reshape(*reshape_list).permute(*permute_tuple)

        log.info(f"R{rank}: Tile Slices: {tile_slices}")
        local_tensor = tensor[tile_slices]

        log.info(f"R{rank}: Local tensor shape: {local_tensor.shape}")

        local_shape = [
            n_blocks * b_size if b_size != -1 else og_size
            for og_size, n_blocks, b_size in zip(
                og_shape, local_tensor.shape[:dims], self._block_sizes
            )
        ]
        log.info(f"R{rank}: Target local tensor shape: {local_shape}")

        # Reshape the tensor to the original shape
        local_tensor = local_tensor.permute(*permute_tuple).reshape(*local_shape)

        # Remove the padding from the relevant axes
        shaved_slices = []
        for i, r in enumerate(missing_elements):
            if r != 0:
                n_procs = math.prod([self._pmesh[x] for x in self._dims_mapping[i]])
                l_p_index = multi2linearIndex(
                    self._pmesh,
                    p_midx,
                    order=self._dims_mapping[i],
                )
                log.info(
                    f"R{rank}: Linear processor index: {l_p_index}, Residue: {r}, Block size: {self._block_sizes[i]}, axis: {i}, N procs: {n_procs}, N full blocks: {n_full_blocks_per_axis[i]}"
                )
                if n_full_blocks_per_axis[i] % n_procs == l_p_index:
                    shaved_slices.append(slice(0, -r))
                else:
                    shaved_slices.append(slice(None))
            else:
                shaved_slices.append(slice(None))

        log.info(f"R{rank}: Shaved slices: {shaved_slices}")
        local_tensor = local_tensor[shaved_slices]
        log.info(f"R{rank}: Final local tensor shape: {local_tensor.shape}")
        return local_tensor

    def apply_split(
        self,
        global_shape: torch.Size,
        local_tensor: torch.Tensor,
        rank: int,
        tensor_axis: int,
        mesh_dims: int | tuple[int, ...],
        block_size: int = 1,
        minor: bool = False,
    ) -> tuple["MultiAxisDist", torch.Tensor]:
        """
        Given a distributed tensor, apply a multi_axis split operation.

        Parameters
        ----------
        global_shape : torch.Size
            The shape of the tensor.
        local_tensor : torch.Tensor
            The local tensor to distribute.
        rank : int
            The rank of the processor.
        tensor_axis : int
            The tensor axis to split.
        mesh_axis : int | tuple[int, ...]
            The processor mesh axis to split.
        block_size : int, optional
            The size of the blocks, by default 1.
        minor : bool, optional
            If set to true, the mesh dimensions will be appended to the right (minor), by default False. If the axis is already split, it will change the existing block size.

        Returns
        -------
        MultiAxisDist
            The distribution that results from the split. None if the tensor is not compatible with the distribution.
        torch.Tensor
            The local distributed tensor belonging to the rank.
        """
        if not self.compatible(global_shape):
            raise ValueError("The tensor is not compatible with the distribution")

        if (0 <= rank < self.numProcessors) is False:
            raise ValueError("Rank must be in the range of the processor mesh.")

        log.info(f"R{rank}: Local tensor shape: {local_tensor.shape}")
        log.info(f"R{rank}: Predicted shape: {self.localShape(global_shape, rank)}")

        if local_tensor.shape != self.localShape(global_shape, rank):
            raise ValueError("Local tensor shape does not match the distribution.")

        new_dist = self.split(global_shape, tensor_axis, mesh_dims, block_size, minor)[
            0
        ]
        log.info(f"New distribution: {new_dist}")

        p_midx = self.getProcessorMultiIndex(rank)
        log.info(f"R{rank}: Processor multi index: {p_midx}")

        mesh_dims = mesh_dims if isinstance(mesh_dims, tuple) else (mesh_dims,)

        target_block_size = new_dist._block_sizes[tensor_axis]

        # Add padding to the target axis
        og_l_shape = local_tensor.shape
        missing_elements = target_block_size - (
            local_tensor.shape[tensor_axis] % target_block_size
        )
        log.info(f"R{rank}: Missing elements {missing_elements}")
        n_full_blocks = og_l_shape[tensor_axis] // target_block_size

        padding_tuple = [0] * len(og_l_shape) * 2
        padding_tuple[tensor_axis * 2] = missing_elements
        log.info(f"R{rank}: Padding tuple: {padding_tuple}")

        padded_tensor = F.pad(local_tensor, padding_tuple[::-1], value=0)
        log.info(f"R{rank}: Padded local tensor shape: {padded_tensor.shape}")

        reshape_list = list(padded_tensor.shape)
        reshape_list[tensor_axis] = (
            padded_tensor.shape[tensor_axis] // target_block_size
        )
        reshape_list.insert(tensor_axis + 1, target_block_size)
        log.info(f"R{rank}: Reshape list: {reshape_list}")

        # Create the permute tuple
        permute_list = list(range(len(reshape_list)))
        permute_list.remove(tensor_axis + 1)
        permute_list.append(tensor_axis + 1)
        log.info(f"R{rank}: Permute list: {permute_list}")

        idx = multi2linearIndex(
            self._pmesh,
            p_midx,
            order=mesh_dims,
        )
        n_procs = math.prod([self._pmesh[x] for x in mesh_dims])
        tile_slices = [slice(None)] * len(padded_tensor.shape)
        tile_slices[tensor_axis] = slice(idx, None, n_procs)
        log.info(f"R{rank}: Tile Slices: {tile_slices}")

        reshaped_tensor = padded_tensor.reshape(*reshape_list).permute(*permute_list)
        log.info(f"R{rank}: Reshaped tensor shape: {reshaped_tensor.shape}")

        # Apply the tile slices
        sliced_tensor = reshaped_tensor[tile_slices]
        log.info(f"R{rank}: Local tensor shape: {sliced_tensor.shape}")

        local_shape = list(og_l_shape)
        local_shape[tensor_axis] = sliced_tensor.shape[tensor_axis] * target_block_size

        log.info(f"R{rank}: Target local tensor shape: {local_shape}")

        # Reshape the tensor to the original shape
        pre_shaving_tensor = sliced_tensor.permute(*permute_list).reshape(*local_shape)
        log.info(f"R{rank}: Pre-shaving tensor shape: {pre_shaving_tensor.shape}")

        relevant_dims = new_dist._dims_mapping[tensor_axis]
        relevant_dims_tuple = (
            relevant_dims if isinstance(relevant_dims, tuple) else (relevant_dims,)
        )
        log.info(f"R{rank}: Relevant dims: {relevant_dims_tuple}")
        l_idx_axis = multi2linearIndex(
            self._pmesh,
            p_midx,
            order=relevant_dims_tuple,
        )
        if missing_elements != 0 and n_full_blocks % n_procs == l_idx_axis:
            shave_slices = [slice(None)] * len(og_l_shape)
            shave_slices[tensor_axis] = slice(0, -missing_elements)
            log.info(f"R{rank}: Shaving slices: {shave_slices}")
            shaved_tensor = pre_shaving_tensor[shave_slices]
        else:
            shaved_tensor = pre_shaving_tensor

        return new_dist, shaved_tensor

    def apply_allgather(
        self,
        global_shape: torch.Size,
        local_tensor: torch.Tensor,
        rank: int,
        gather_dim: int = -1,
    ) -> tuple["MultiAxisDist", torch.Tensor]:
        """
        Given a distributed tensor, apply a multi_axis allgather operation.

        Parameters
        ----------
        global_shape : torch.Size
            The shape of the tensor.
        local_tensor : torch.Tensor
            The local tensor to distribute.
        rank : int
            The rank of the processor.
        gather_dim : int, optional
            The tensor axis to gather data from, by default -1. If -1, it will use the first axis.

        Returns
        -------
        MultiAxisDist
            The distribution that results from the allgather communication. None if the tensor is not compatible with the distribution.
        torch.Tensor
            The local distributed tensor belonging to the rank.
        """
        if not self.compatible(global_shape):
            raise ValueError("The tensor is not compatible with the distribution")

        if (0 <= rank < self.numProcessors) is False:
            raise ValueError("Rank must be in the range of the processor mesh.")

        exp_l_shape = self.localShape(
            global_shape, rank
        )  # Shape the local tensor should have
        log.info(f"R{rank}: Local tensor shape: {local_tensor.shape}")
        log.info(f"R{rank}: Predicted shape: {exp_l_shape}")

        if local_tensor.shape != exp_l_shape:
            raise ValueError("Local tensor shape does not match the distribution.")

        new_dist = self.allgather(global_shape, gather_dim)[0]
        exp_t_l_shape = new_dist.localShape(
            global_shape, rank
        )  # Expected local shape of the outcome dist
        log.info(f"New distribution: {new_dist}, new shape: {exp_t_l_shape}")

        p_midx = self.getProcessorMultiIndex(rank)
        log.info(f"R{rank}: Processor multi index: {p_midx}")

        og_l_shape = local_tensor.shape
