"""MultiAxisDist class."""

import numpy as np

from tensorcraft.distributions.dist import Dist
from tensorcraft.shape import Shape
from tensorcraft.types import Index
from tensorcraft.util import multi2linearIndex


class MultiAxisDist(Dist):
    """
    Represents a distribution for a tensor on a processor mesh.

    Each of the dimensions of the tensor is distributed across the assigned dimensions of a processor mesh.

    Parameters
    ----------
    mesh : Shape
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
    _mesh : Shape
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
    processorView(shape)
        Returns a boolean array indicating the processors that have access to each element of the tensor.
    getIndexLocation(shape, index)
        Returns a boolean array indicating the processors that have access to the specified index of the tensor.
    compatible(shape)
        Checks if the shape is compatible with the distribution.

    Private Methods
    ---------------
    _dimIndexInProcessor(dim, dim_idx, p_mi)
        Checks if the given dimension index is in the specified processor.
    _distributeDim(dim, dim_size)
        Distributes a dimension across the processors.

    """

    def __init__(
        self, mesh: Shape, dims_mapping: tuple[tuple], block_sizes: tuple
    ) -> None:
        if len(dims_mapping) != len(block_sizes):
            raise ValueError("The number of dimensions and block sizes must match")
        for dim in dims_mapping:
            for mesh_axis in dim:
                if mesh_axis >= mesh.order or mesh_axis < 0:
                    raise ValueError("The dimension mapping is out of bounds")

        for block in block_sizes:
            if block < 0:
                raise ValueError("The block size must be greater or equal 0")

        self._mesh = mesh
        self._dims_mapping = dims_mapping
        self._block_sizes = np.array(block_sizes)
        self._omit_replication = None

    @property
    def numProcessors(self):  # noqa: D102
        return self._mesh.size

    @property
    def processorArrangement(self):  # noqa: D102
        return self._mesh

    def getProcessorMultiIndex(self, index):  # noqa: D102
        return self._mesh.getMultiIndex(index)

    def processorView(self, shape):  # noqa: D102
        if not self.compatible(shape):
            raise ValueError("The tensor is not compatible with the distribution")

        processor_view = np.ones((*shape.asTuple(), self._mesh.size), dtype=np.bool_)
        print(processor_view.shape)

        dist_axis_list = [self._distributeDim(i, shape[i]) for i in range(shape.order)]
        for i in range(shape.size):
            m_idx = tuple(shape.getMultiIndex(i)) + (None,)
            for j in range(shape.order):
                processor_view[m_idx] &= dist_axis_list[j][m_idx[j], :]

        return processor_view

    def getIndexLocation(self, shape, index):  # noqa: D102
        if isinstance(index, int):
            index = shape.getMultiIndex(index)

        p_list = np.ones((self._mesh.size,), dtype=np.bool_)
        for dim in range(shape.order):
            distributed_dim = self._distributeDim(dim, shape[dim])
            p_list &= distributed_dim[index[dim], :]

        return p_list

    def compatible(self, shape):  # noqa: D102
        # Ensure that the tensor order and the number of dimensions in the distribution match
        if shape.order != len(self._dims_mapping):
            print(
                f"Tensor order and the number of dimensions in the distribution must match: {shape.order} != {len(self._dims_mapping)}"
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
            mesh_dims = [self._mesh[i] for i in mesh_dims_idx]
            if block_size == 0:
                continue
            elif shape[dim] % block_size != 0:
                print(
                    f"Maximum block size exceeded for dimension {dim}: {block_size} > {np.floor(shape[dim] / np.prod(mesh_dims))}"
                )
                return False

        return True

    def _distributeDim(self, dim: Index, dim_size: Index) -> np.ndarray:
        mesh_dims_idx = self._dims_mapping[dim]
        num_process = self._mesh.size

        if len(mesh_dims_idx) == 0:
            return np.ones((dim_size, num_process), dtype=np.bool_)

        dim_distribution = np.ones((dim_size, num_process), dtype=np.bool_)
        mesh_dims = [self._mesh[i] for i in mesh_dims_idx]
        block_size = self._block_sizes[dim]
        _, axis_chunk_ends = self.axisSplits(dim_size, block_size, np.prod(mesh_dims))  # type: ignore
        start_idx = 0
        for chunk_idx, end_idx in enumerate(axis_chunk_ends):
            left_eq = chunk_idx % np.prod(mesh_dims)  # type: ignore
            for j in range(num_process):
                p_mi = self._mesh.getMultiIndex(j)
                right_eq = multi2linearIndex(
                    self._mesh.asTuple(), p_mi, order=np.array(mesh_dims_idx)
                ) % np.prod(mesh_dims)  # type: ignore
                dim_distribution[start_idx:end_idx, j] = left_eq == right_eq

            start_idx = end_idx

        return dim_distribution
