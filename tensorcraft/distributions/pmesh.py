"""PMeshDist class."""

import numpy as np

from tensorcraft.distributions.dist import Dist
from tensorcraft.tensor import Tensor
from tensorcraft.types import MIndex
from tensorcraft.util import multi2linearIndex


class PMeshDist(Dist):
    """
    Represents a distribution for a tensor on a processor mesh.

    Each of the dimensions of the tensor is distributed across the assigned dimensions of a processor mesh.

    Parameters
    ----------
    mesh : Tensor
        The processor mesh.
    dims_mapping : tuple[tuple]
        The mapping of tensor dimensions to mesh dimensions.
    block_sizes : tuple
        The block sizes for each dimension.

    Raises
    ------
    ValueError
        If the number of dimensions and block sizes do not match.
    ValueError
        If the dimension mapping is out of bounds.
    ValueError
        If the block size is less than or equal to 0.

    Attributes
    ----------
    _mesh : Tensor
        The processor mesh.
    _dims_mapping : tuple[tuple]
        The mapping of tensor dimensions to mesh dimensions.
    _block_sizes : np.array
        The block sizes for each dimension.
    _omit_replication : None
        Placeholder attribute.

    Properties
    ----------
    numProcessors : int
        The number of processors in the mesh.
    processorArrangement : tuple
        The arrangement of processors in the mesh.

    Methods
    -------
    getProcessorMultiIndex(index)
        Returns the multi-index of the processor at the given index.
    processorView(tensor)
        Returns a boolean array indicating the processors that have access to each element of the tensor.
    getIndexLocation(tensor, index)
        Returns a boolean array indicating the processors that have access to the specified index of the tensor.
    compatible(tensor)
        Checks if the tensor is compatible with the distribution.

    Private Methods
    ---------------
    _dimIndexInProcessor(dim, dim_idx, p_mi)
        Checks if the given dimension index is in the specified processor.
    _distributeDim(dim, dim_size)
        Distributes a dimension across the processors.

    """

    def __init__(
        self, mesh: Tensor, dims_mapping: tuple[tuple], block_sizes: tuple
    ) -> None:
        if len(dims_mapping) != len(block_sizes):
            raise ValueError("The number of dimensions and block sizes must match")
        for dim in dims_mapping:
            for mesh_axis in dim:
                if mesh_axis >= mesh.order or mesh_axis < 0:
                    raise ValueError("The dimension mapping is out of bounds")

        for block in block_sizes:
            if block <= 0:
                raise ValueError("The block size must be greater than 0")

        self._mesh = mesh
        self._dims_mapping = dims_mapping
        self._block_sizes = np.array(block_sizes)
        self._omit_replication = None

    @property
    def numProcessors(self):  # noqa: D102
        return self._mesh.size

    @property
    def processorArrangement(self):  # noqa: D102
        return self._mesh.shape

    def getProcessorMultiIndex(self, index):  # noqa: D102
        return self._mesh.getMultiIndex(index)

    def processorView(self, tensor):  # noqa: D102
        if not self.compatible(tensor):
            raise ValueError("The tensor is not compatible with the distribution")

        processor_view = np.ones((*tensor.shape, self._mesh.size), dtype=np.bool_)

        dist_axis_list = [
            self._distributeDim(i, tensor.shape[i]) for i in range(tensor.order)
        ]
        for i in range(tensor.size):
            m_idx = tuple(tensor.getMultiIndex(i)) + (None,)
            for j in range(tensor.order):
                processor_view[m_idx] &= dist_axis_list[j][m_idx[j], :]

        return processor_view

    def getIndexLocation(self, tensor, index):  # noqa: D102
        if isinstance(index, int):
            index = tensor.getMultiIndex(index)

        p_list = np.ones((self._mesh.size,), dtype=np.bool_)
        for dim in range(tensor.order):
            for p in range(self._mesh.size):
                p_mi = self._mesh.getMultiIndex(p)
                p_list[p] &= self._dimIndexInProcessor(dim, index[dim], p_mi)

        return p_list

    def compatible(self, tensor):  # noqa: D102
        # Ensure that the tensor order and the number of dimensions in the distribution match
        if tensor.order != len(self._dims_mapping):
            print(
                f"Tensor order and the number of dimensions in the distribution must match: {tensor.order} != {len(self._dims_mapping)}"
            )
            return False

        # Ensure that the tensor dimensions are divisible by the block sizes
        if not all(
            [
                tensor.shape[dim] % block == 0
                for dim, (mapping, block) in enumerate(
                    zip(self._dims_mapping, self._block_sizes)
                )
                if len(mapping) > 0
            ]
        ):
            raise ValueError(
                "The tensor dimensions must be divisible by the block sizes"
            )

        # Block size must ensure that each of the mesh dimensions has access to at least one block
        for dim, block_size in enumerate(self._block_sizes):
            mesh_dims_idx = self._dims_mapping[dim]
            mesh_dims = self._mesh.shape[mesh_dims_idx]
            if tensor.shape[dim] % block_size != 0:
                print(
                    f"Maximum block size exceeded for dimension {dim}: {block_size} > {np.floor(tensor.shape[dim] / np.prod(mesh_dims))}"
                )
                return False

        return True

    def _dimIndexInProcessor(self, dim: int, dim_idx: int, p_mi: MIndex) -> bool:
        mesh_dims_idx = self._dims_mapping[dim]

        if len(mesh_dims_idx) == 0:
            return True

        mesh_dims = self._mesh.shape[mesh_dims_idx]
        block_size = self._block_sizes[dim]

        t_mi = multi2linearIndex(self._mesh.shape, p_mi, order=np.array(mesh_dims_idx))
        u = np.prod(mesh_dims)
        # print(f"{p_mi} : {dim_idx} / {block_size}  == {t_mi} (% {u}) -> {np.floor(dim_idx / block_size) % u} == {t_mi % u}")
        return np.floor(dim_idx / block_size) % u == (t_mi % u)

    def _distributeDim(self, dim: int, dim_size: int) -> np.ndarray:
        mesh_dims_idx = self._dims_mapping[dim]
        num_process = self._mesh.size

        if len(mesh_dims_idx) == 0:
            return np.ones((dim_size, num_process), dtype=np.bool_)

        dim_distribution = np.ones((dim_size, num_process), dtype=np.bool_)
        for i in range(dim_size):
            for j in range(num_process):
                p_mi = np.array(self._mesh.getMultiIndex(j))
                dim_distribution[i, j] = self._dimIndexInProcessor(dim, i, p_mi)

        return dim_distribution
