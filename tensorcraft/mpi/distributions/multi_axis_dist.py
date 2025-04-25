"""MPI Implementatation of MultiAxisDist class."""

import logging
import math
from typing import Optional

import torch
import torch.nn.functional as F
from mpi4py import MPI

from tensorcraft.distributions import MultiAxisDist
from tensorcraft.mpi.mpi_utils import as_buffer, tensor2mpiBuffer
from tensorcraft.util import multi2linearIndex

log = logging.getLogger("tensorcraft")


class MPIMultiAxisDist(MultiAxisDist):
    """MPI implementation of the MultiAxisDist class."""

    @classmethod
    def fromMultiAxisDist(
        cls,
        dist: MultiAxisDist,
    ) -> "MPIMultiAxisDist":
        """
        Create a new MPIMultiAxisDist object from an existing MultiAxisDist object.

        Parameters
        ----------
        dist : MultiAxisDist
            The MultiAxisDist object to convert.

        Returns
        -------
        MPIMultiAxisDist
            The new MPIMultiAxisDist object.
        """
        if not isinstance(dist, MultiAxisDist):
            raise TypeError("dist must be an instance of MultiAxisDist")
        return cls(
            dist._pmesh,
            dist._dims_mapping,
            dist._block_sizes,
        )

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
                tile_slices += [slice(None), slice(None)]
            else:
                idx = multi2linearIndex(
                    self._pmesh,
                    p_midx,
                    order=dim_map,
                )
                n_procs = math.prod([self._pmesh[x] for x in dim_map])

                tile_slices += [slice(idx, None, n_procs), slice(None)]

        permute_tuple = (
            tuple(set(range(len(reshape_list))) - set(permute_tuple)) + permute_tuple
        )
        log.info(f"R{rank}: Permute tuple: {permute_tuple}")
        log.info(f"R{rank}: Reshape tuple: {reshape_list}")

        tensor = tensor.reshape(*reshape_list)

        log.info(f"R{rank}: Tile Slices: {tile_slices}")
        local_tensor = tensor[tile_slices]

        log.info(f"R{rank}: Local tensor shape: {local_tensor.shape}")

        local_shape = [
            n_blocks * b_size if b_size != -1 else og_size
            for og_size, n_blocks, b_size in zip(
                og_shape, local_tensor.shape[::2], self._block_sizes
            )
        ]
        log.info(f"R{rank}: Target local tensor shape: {local_shape}")

        # Reshape the tensor to the original shape
        local_tensor = local_tensor.reshape(*local_shape)
        # print(local_tensor)

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
    ) -> tuple["MPIMultiAxisDist", torch.Tensor]:
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

        res = local_tensor.shape[tensor_axis] % target_block_size
        missing_elements = (target_block_size - res) if res != 0 else 0
        log.info(f"R{rank}: Missing elements {missing_elements}")
        n_full_blocks = global_shape[tensor_axis] // target_block_size

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

        idx = multi2linearIndex(
            self._pmesh,
            p_midx,
            order=mesh_dims,
        )
        n_procs = math.prod([self._pmesh[x] for x in mesh_dims])
        tile_slices = [slice(None)] * len(padded_tensor.shape)
        tile_slices[tensor_axis] = slice(idx, None, n_procs)
        log.info(f"R{rank}: Tile Slices: {tile_slices}")

        reshaped_tensor = padded_tensor.reshape(*reshape_list)
        log.info(f"R{rank}: Reshaped tensor shape: {reshaped_tensor.shape}")

        # Apply the tile slices
        sliced_tensor = reshaped_tensor[tile_slices]
        log.info(f"R{rank}: Local tensor shape: {sliced_tensor.shape}")

        local_shape = list(og_l_shape)
        local_shape[tensor_axis] = sliced_tensor.shape[tensor_axis] * target_block_size

        log.info(f"R{rank}: Pre-shaved Target local tensor shape: {local_shape}")

        # Reshape the tensor to the original shape
        pre_shaving_tensor = sliced_tensor.reshape(*local_shape)
        log.info(f"R{rank}: Pre-shaving tensor shape: {pre_shaving_tensor.shape}")

        relevant_dims = new_dist._dims_mapping[tensor_axis]
        relevant_dims_tuple = (
            relevant_dims if isinstance(relevant_dims, tuple) else (relevant_dims,)
        )
        relevant_n_procs = math.prod([self._pmesh[x] for x in relevant_dims_tuple])
        log.info(f"R{rank}: Relevant dims: {relevant_dims_tuple}")
        l_idx_axis = multi2linearIndex(
            self._pmesh,
            p_midx,
            order=relevant_dims_tuple,
        )

        log.info(
            f"R{rank}: Linear processor index: {l_idx_axis}, Residue: {missing_elements}, Block size: {target_block_size}, axis: {tensor_axis}, N procs: {relevant_n_procs}, N full blocks: {n_full_blocks}"
        )
        if missing_elements != 0 and n_full_blocks % relevant_n_procs == l_idx_axis:
            shave_slices = [slice(None)] * len(og_l_shape)
            shave_slices[tensor_axis] = slice(0, -missing_elements)
            log.info(f"R{rank}: Shaving slices: {shave_slices}")
            shaved_tensor = pre_shaving_tensor[shave_slices]
        else:
            shaved_tensor = pre_shaving_tensor

        return self.fromMultiAxisDist(new_dist), shaved_tensor

    def apply_allgather(  # type: ignore[no-any-unimported]
        self,
        global_shape: torch.Size,
        local_tensor: torch.Tensor,
        comm: MPI.Comm,
        gather_mesh_dim: Optional[int] = None,
    ) -> tuple["MPIMultiAxisDist", torch.Tensor]:
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
        mesh_dim : int, optional
            The mesh dimension that will gather data, by default -1. If -1, it will gather along all the mesh dimensions, leading to a non-distributed tensor.

        Returns
        -------
        MultiAxisDist
            The distribution that results from the allgather communication. None if the tensor is not compatible with the distribution.
        torch.Tensor
            The local distributed tensor belonging to the rank.
        """
        rank = comm.Get_rank()
        world_size = comm.Get_size()

        if world_size != self.numProcessors:
            raise ValueError(
                f"World size {world_size} does not match the number of processors {self.numProcessors}"
            )

        if not self.compatible(global_shape):
            raise ValueError("The tensor is not compatible with the distribution")

        if (0 <= rank < self.numProcessors) is False:
            raise ValueError("Rank must be in the range of the processor mesh.")

        # Obtain the new distribution and the expected local shape
        new_dist = self.allgather(global_shape, gather_mesh_dim)[0]
        exp_t_l_shape = new_dist.localShape(
            global_shape, rank
        )  # Expected local shape of the outcome dist

        log.info(f"R{rank}: New distribution: {new_dist}, new shape: {exp_t_l_shape}")
        if gather_mesh_dim is not None:
            exp_l_shape = self.localShape(
                global_shape, rank
            )  # Shape the local tensor should have
            log.info(f"R{rank}: Local tensor shape: {local_tensor.shape}")
            log.info(f"R{rank}: Expected local shape: {exp_l_shape}")

            if local_tensor.shape != exp_l_shape:
                raise ValueError("Local tensor shape does not match the distribution.")

            changed_t_axis = -1
            minor = False
            for axis, mappings in enumerate(self._dims_mapping):
                if gather_mesh_dim in mappings:
                    minor = mappings.index(gather_mesh_dim) != 0
                    changed_t_axis = axis
                    break
            log.info(f"R{rank}: Changed tensor axis: {changed_t_axis}, minor: {minor}")

            p_midx = self.getProcessorMultiIndex(rank)
            log.info(f"R{rank}: Processor multi index: {p_midx}")

            n_procs = self._pmesh[gather_mesh_dim]
            log.info(f"R{rank}: N procs: {n_procs}")

            # Find the global rank of the first processor in the mesh dimension
            tmp_p_midx = list(p_midx)
            tmp_p_midx[gather_mesh_dim] = 0
            linear_max_g_rank = multi2linearIndex(
                self._pmesh,
                tmp_p_midx,
            )
            log.info(
                f"R{rank}: Rank of largest tensor in the subcommunicator: {tmp_p_midx} {linear_max_g_rank}"
            )

            max_local_shape = self.localShape(global_shape, linear_max_g_rank)
            n_elements = math.prod(max_local_shape)
            b_size = self._block_sizes[changed_t_axis]
            log.info(f"R{rank}: N elements: {n_elements}")
            log.info(f"R{rank}: Max local shape: {max_local_shape}")

            # Target buffer shape (n_procs, n_max_blocks, b_size, )
            n_max_blocks = math.ceil(
                max_local_shape[changed_t_axis] / self._block_sizes[changed_t_axis]
            )

            # Insert padding in case change_t_axis is not the first axis
            if changed_t_axis != 0:
                padding = (
                    n_max_blocks * self._block_sizes[changed_t_axis]
                    - local_tensor.shape[changed_t_axis]
                )
                log.info(f"R{rank}: Padding: {padding}")
                if padding > 0:
                    padding_tuple = [0] * len(local_tensor.shape) * 2
                    padding_tuple[changed_t_axis * 2] = padding
                    log.info(f"R{rank}: Padding tuple: {padding_tuple}")
                    local_tensor = F.pad(local_tensor, padding_tuple[::-1], value=0)

                    log.info(
                        f"R{rank}: Padded local tensor shape: {local_tensor.shape}"
                    )
                    log.info(f"R{rank}: Padded local tensor: {local_tensor}")

            # Send buffer
            send_buffer_tuple = tensor2mpiBuffer(local_tensor)

            tmp_shape = list(max_local_shape)
            tmp_shape[changed_t_axis] = b_size
            tmp_shape.insert(changed_t_axis, n_max_blocks)

            recv_tensor = torch.zeros(
                [
                    n_procs,
                ]
                + tmp_shape,
                dtype=local_tensor.dtype,
            )
            recv_buffer_tuple = (
                as_buffer(recv_tensor),
                n_elements,
                send_buffer_tuple[2],
            )
            log.info(f"R{rank}: Send buffer: {send_buffer_tuple}")
            log.info(f"R{rank}: Recv buffer: {recv_buffer_tuple}")

            # Create the subcommunicator
            cart_comm = comm.Create_cart(
                dims=self._pmesh, periods=[True] * len(self._pmesh), reorder=True
            )
            subs = [0] * len(self._pmesh)
            subs[gather_mesh_dim] = 1
            sub_comm = cart_comm.Sub(subs)

            # Perform the allgather operation
            sub_comm.Allgather(send_buffer_tuple, recv_tensor)
            log.info(f"R{rank}: Recv_tensor : {recv_tensor}")

            # Reshape the tensor to the original shape
            permute_list = list(range(1, len(recv_tensor.shape)))
            permute_list.insert(changed_t_axis + 1, 0)

            reshape_list = list(exp_t_l_shape)
            reshape_list[changed_t_axis] = n_max_blocks * b_size * n_procs

            recv_tensor = recv_tensor.permute(*permute_list).reshape(*reshape_list)
            log.info(f"R{rank}: Reshaped tensor shape: {recv_tensor.shape}")

            # Remove the padding from the relevant axes
            slices = [slice(None)] * len(exp_l_shape)
            slices[changed_t_axis] = slice(0, exp_t_l_shape[changed_t_axis])

            log.info(f"R{rank}: Slices: {slices}")
            recv_tensor = recv_tensor[slices]
            log.info(f"R{rank}: Final tensor shape: {recv_tensor.shape}")

            return self.fromMultiAxisDist(new_dist), recv_tensor

        else:
            raise NotImplementedError(
                "Gathering along all the mesh dimensions is not implemented yet."
            )
