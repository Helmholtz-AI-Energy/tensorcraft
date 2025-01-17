"""Datatype utilities."""

from typing import Tuple, TypeAlias

import torch
from mpi4py import MPI

MPIBuffer: TypeAlias = MPI.BufSpec | MPI.BufSpecB | MPI.BufSpecV | MPI.BufSpecW

MPI_INT_MAX = torch.iinfo(torch.int32).max

torch_type2mpi_type = {
    torch.float32: MPI.FLOAT,
    torch.float64: MPI.DOUBLE,
    torch.int8: MPI.CHAR,
    torch.uint8: MPI.UNSIGNED_CHAR,
    torch.int16: MPI.SHORT,
    torch.uint16: MPI.UNSIGNED_SHORT,
    torch.int32: MPI.INT,
    torch.uint32: MPI.UNSIGNED,
    torch.int64: MPI.LONG,
    torch.uint64: MPI.UNSIGNED_LONG,
    torch.complex64: MPI.COMPLEX,
    torch.complex128: MPI.DOUBLE_COMPLEX,
    torch.bool: MPI.BOOL,
}


def as_buffer(x: torch.Tensor) -> MPI.buffer:
    """
    Convert a PyTorch tensor to an MPI buffer.

    Parameters
    ----------
    x : torch.Tensor
        The input tensor.

    Returns
    -------
    MPI.buffer

    """
    return MPI.buffer.fromaddress(x.untyped_storage().data_ptr(), 0)


def _recursive_contiguous(
    n_elements: int, mpi_type: MPI.Datatype
) -> Tuple[MPI.Datatype, int]:
    new_count = n_elements // MPI_INT_MAX
    left_over = n_elements % MPI_INT_MAX

    if new_count > MPI_INT_MAX:
        raise ValueError("Tensor is too large, wtf are you doing?")
    vector_type = mpi_type.Create_vector(new_count, MPI_INT_MAX, MPI_INT_MAX)
    if left_over > 0:
        left_over_mpi_type = mpi_type.Create_contiguous(left_over).Commit()
        _, old_type_extent = mpi_type.Get_extent()
        disp = MPI_INT_MAX * new_count * old_type_extent
        struct_type = mpi_type.Create_struct(
            [1, 1], [0, disp], [vector_type, left_over_mpi_type]
        ).Commit()
        vector_type.Free()
        left_over_mpi_type.Free()
        return struct_type, 1
    else:
        return vector_type, 1


def _create_large_vector(n_elements: int, mpi_type: MPI.Datatype):
    n_vectors = n_elements // MPI_INT_MAX
    left_over = n_elements % MPI_INT_MAX

    vector_type = mpi_type.Create_vector(n_vectors, MPI_INT_MAX, 1).Commit()
    if left_over == 0:
        return vector_type
    else:
        vector_type = mpi_type.Create_vector(n_vectors, 1, MPI_INT_MAX)


def tensor2mpiBuffer(tensor: torch.Tensor) -> MPIBuffer:
    """
    Convert a PyTorch tensor to an MPI buffer.

    Parameters
    ----------
    tensor : torch.Tensor
        The input tensor.

    Returns
    -------
    MPIBuffer
    """
    tensor_stride = tensor.stride()
    tensor_offset = tensor.storage_offset()
    tensor_shape = tensor.size()
    tensor_dtype = tensor.dtype

    print(f"tensor_stride: {tensor_stride}")
    print(f"tensor_offset: {tensor_offset}")
    print(f"tensor_shape: {tensor_shape}")
    print(f"tensor_dtype: {tensor_dtype}")

    if tensor.is_contiguous():
        buffer = as_buffer(tensor)

        n_elements = tensor.numel()
        # Check
        if n_elements >= MPI_INT_MAX:
            if n_elements > MPI_INT_MAX**2:
                raise ValueError("Tensor is too large, wtf are you doing?")
            mpi_type, type_count = _recursive_contiguous(
                n_elements, torch_type2mpi_type[tensor_dtype]
            )
            return buffer, type_count, mpi_type
        else:
            return buffer, n_elements, torch_type2mpi_type[tensor_dtype]
    else:
        print("tensor is not contiguous")
        raise NotImplementedError("Non-contiguous tensors are not supported yet.")
