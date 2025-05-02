"""Datatype utilities."""

import logging
from typing import Tuple, TypeAlias

import torch
from mpi4py import MPI
from mpi4py.typing import BufSpec, BufSpecB, BufSpecV, BufSpecW

log = logging.getLogger(__name__)

MPIBuffer: TypeAlias = BufSpec | BufSpecB | BufSpecV | BufSpecW

MPI_INT_MAX = torch.iinfo(torch.int32).max
# MPI_INT_MAX = 128

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


def as_buffer(x: torch.Tensor, offset: int = 0) -> MPI.buffer:
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
    dtype_size = x.dtype.itemsize
    return MPI.buffer.fromaddress(
        x.untyped_storage().data_ptr() + offset * dtype_size, 0
    )


def _large_contiguous_vector(
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


def _create_recursive_vector(x: torch.Tensor) -> MPI.Datatype:
    subarray_sizes = x.size()
    tensor_stride = x.stride()

    datatype_history: list[MPI.Datatype] = []
    original_datatype = torch_type2mpi_type[x.dtype]
    current_datatype = original_datatype

    i = len(tensor_stride) - 1
    while i >= 0:
        current_stride = tensor_stride[i]
        current_size = subarray_sizes[i]
        # Define vector out of previous datatype with stride equals to current stride
        if i == len(tensor_stride) - 1 and current_stride == 1:
            i -= 1
            # Define vector out of previous datatype with stride equals to current stride
            current_stride = tensor_stride[i]
            next_size = subarray_sizes[i]
            new_vector_datatype = current_datatype.Create_vector(
                next_size, current_size, current_stride
            ).Commit()

        else:
            if i == len(tensor_stride) - 1:
                new_vector_datatype = current_datatype.Create_vector(
                    current_size, 1, current_stride
                ).Commit()
            else:
                new_vector_datatype = current_datatype.Create_vector(
                    current_size, 1, 1
                ).Commit()

        datatype_history.append(new_vector_datatype)
        # Set extent of the new datatype to the extent of the basic datatype to allow interweaving of data
        next_stride = tensor_stride[i - 1]
        new_resized_vector_datatype = new_vector_datatype.Create_resized(
            0, original_datatype.Get_extent()[1] * next_stride
        ).Commit()
        datatype_history.append(new_resized_vector_datatype)
        current_datatype = new_resized_vector_datatype

        i -= 1
    for dt in datatype_history[:-1]:
        dt.Free()
    return current_datatype


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

    log.debug(f"tensor_stride: {tensor_stride}")
    log.debug(f"tensor_offset: {tensor_offset}")
    log.debug(f"tensor_shape: {tensor_shape}")
    log.debug(f"tensor_dtype: {tensor_dtype}")

    buffer = as_buffer(tensor, int(tensor_offset))

    if tensor.is_contiguous():
        n_elements = tensor.numel()
        # Check
        if n_elements > MPI_INT_MAX:
            log.warning("tensor is too large, using tricks")
            if n_elements > MPI_INT_MAX**2:
                raise ValueError("Tensor is too large, wtf are you doing?")
            mpi_type, type_count = _large_contiguous_vector(
                n_elements, torch_type2mpi_type[tensor_dtype]
            )
            return buffer, type_count, mpi_type
        else:
            log.debug("Best case scenario, it is contiguous!")
            return buffer, n_elements, torch_type2mpi_type[tensor_dtype]
    else:
        # Check if the tensor stride is arranged in decending order
        recursive_dt = _create_recursive_vector(tensor)
        type_count = 1
        return buffer, type_count, recursive_dt
