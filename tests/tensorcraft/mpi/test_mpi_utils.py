import hypothesis.extra.numpy as npst
import mpi4py.MPI as MPI
import pytest
import torch
from hypothesis import given
from hypothesis import strategies as st

import tensorcraft as tc

pytestmark = pytest.mark.mpi_test(2)

comm = MPI.COMM_WORLD
mpi_size = comm.Get_size()
mpi_rank = comm.Get_rank()

_dtypes = [torch.int32, torch.int64, torch.float32, torch.float64]


@st.composite
def mpi_st(draw, strategy: st.SearchStrategy):
    """Decorator to make a strategy MPI-aware."""
    # Get the strategy
    data = draw(strategy)

    # Broadcast the strategy to all ranks
    sync_data = comm.bcast(data, root=0)

    return sync_data


@given(
    shape_dtype=mpi_st(
        st.tuples(
            npst.array_shapes(min_dims=1, max_dims=4, min_side=2, max_side=100),
            st.sampled_from(_dtypes),
        )
    )
)
def test_torch2mpiBuffer_full_tensor(shape_dtype):
    shape, dtype = shape_dtype

    print(f"R{mpi_rank}: Shape: {shape}, dtype: {dtype}")
    if mpi_rank == 0:
        if dtype in [torch.int32, torch.int64]:
            tensor = torch.randint(100, shape, dtype=dtype)
        else:
            tensor = torch.rand(shape, dtype=dtype)
    else:
        tensor = torch.zeros(shape, dtype=dtype)

    print(f"R{mpi_rank}: shape: {tensor.shape}")

    buffer, count, mpi_type = tc.mpi.tensor2mpiBuffer(tensor)
    print(f"R{mpi_rank}: buffer: {buffer}")
    comm.Bcast([buffer, count, mpi_type], root=0)
    print(f"R{mpi_rank}: After Bcast")

    validation_tensors: list[torch.Tensor] = comm.gather(tensor, root=0)

    if validation_tensors:
        print(f"R{mpi_rank}: N Validation tensors: {len(validation_tensors)}")
        for v_tensor in validation_tensors:
            assert torch.all(v_tensor == tensor)
    print(f"R{mpi_rank}: After validation --------------------------------------------")


@given(
    shape_dtype=mpi_st(
        st.tuples(
            npst.array_shapes(min_dims=2, max_dims=4, min_side=2, max_side=100),
            st.sampled_from(_dtypes),
        )
    )
)
def test_torch2mpiBuffer_transposed_first_last(shape_dtype):
    shape, dtype = shape_dtype
    if mpi_rank == 0:
        if dtype in [torch.int32, torch.int64]:
            base_tensor = torch.randint(100, shape, dtype=dtype)
        else:
            base_tensor = torch.rand(shape, dtype=dtype)

    # Transpose first and last dimension
    if mpi_rank == 0:
        tensor = base_tensor.transpose(0, len(shape) - 1)
    else:
        tensor = torch.zeros((shape[-1],) + shape[1:-1] + (shape[0],), dtype=dtype)

    print(f"R{mpi_rank}: shape: {tensor.shape}")
    buffer, count, mpi_type = tc.mpi.tensor2mpiBuffer(tensor)
    comm.Bcast([buffer, count, mpi_type], root=0)

    validation_tensors: list[torch.Tensor] = comm.gather(tensor, root=0)
    print(validation_tensors)

    if mpi_rank == 0:
        for v_tensor in validation_tensors:
            assert torch.all(v_tensor == tensor)


@given(
    shape_dtype=mpi_st(
        st.tuples(
            npst.array_shapes(min_dims=2, max_dims=4, min_side=2, max_side=100),
            st.sampled_from(_dtypes),
        )
    )
)
def test_torch2mpiBuffer_transposed_first_second(shape_dtype):
    shape, dtype = shape_dtype
    print(f"Rank {mpi_rank}")
    print(f"Ransk {comm.gather(mpi_rank, root=0)}")
    if mpi_rank == 0:
        if dtype in [torch.int32, torch.int64]:
            base_tensor = torch.randint(100, shape, dtype=dtype)
        else:
            base_tensor = torch.rand(shape, dtype=dtype)

    # Transpose first and second dims
    if mpi_rank == 0:
        tensor = base_tensor.transpose(0, 1)
    else:
        tensor = torch.zeros((shape[1],) + (shape[0],) + shape[2:], dtype=dtype)

    print(f"R{mpi_rank}: shape: {tensor.shape}")
    buffer, count, mpi_type = tc.mpi.tensor2mpiBuffer(tensor)
    comm.Bcast([buffer, count, mpi_type], root=0)

    validation_tensors: list[torch.Tensor] = comm.gather(tensor, root=0)
    print(validation_tensors)

    if mpi_rank == 0:
        for v_tensor in validation_tensors:
            assert torch.all(v_tensor == tensor)


@given(
    shape_dtype=mpi_st(
        st.tuples(
            npst.array_shapes(min_dims=1, max_dims=4, min_side=10, max_side=100),
            st.sampled_from(_dtypes),
        )
    )
)
def test_torch2mpiBuffer_cont_offset(shape_dtype):
    shape, dtype = shape_dtype
    print(f"Rank {mpi_rank}")
    print(f"Ransk {comm.gather(mpi_rank, root=0)}")
    if mpi_rank == 0:
        if dtype in [torch.int32, torch.int64]:
            base_tensor = torch.randint(100, shape, dtype=dtype)
        else:
            base_tensor = torch.rand(shape, dtype=dtype)

    # Contiguous with offset
    if mpi_rank == 0:
        tensor = base_tensor[3:]
    else:
        tensor = torch.zeros((shape[0] - 3,) + shape[1:], dtype=dtype)

    print(f"R{mpi_rank}: shape: {tensor.shape}")
    buffer, count, mpi_type = tc.mpi.tensor2mpiBuffer(tensor)
    comm.Bcast([buffer, count, mpi_type], root=0)

    validation_tensors: list[torch.Tensor] = comm.gather(tensor, root=0)
    print(validation_tensors)

    if mpi_rank == 0:
        for v_tensor in validation_tensors:
            assert torch.all(v_tensor == tensor)


@given(
    shape_dtype=mpi_st(
        st.tuples(
            npst.array_shapes(min_dims=1, max_dims=4, min_side=10, max_side=100),
            st.sampled_from(_dtypes),
        )
    )
)
def test_torch2mpiBuffer_non_cont_slice(shape_dtype):
    shape, dtype = shape_dtype
    print(f"Rank {mpi_rank}")
    print(f"Ransk {comm.gather(mpi_rank, root=0)}")
    if mpi_rank == 0:
        if dtype in [torch.int32, torch.int64]:
            base_tensor = torch.randint(100, shape, dtype=dtype)
        else:
            base_tensor = torch.rand(shape, dtype=dtype)

    # Slice causes non contiguous stuff
    if mpi_rank == 0:
        tensor = base_tensor[..., 3:]
    else:
        tensor = torch.zeros(shape[0:-1] + (shape[-1] - 3,), dtype=dtype)

    print(f"R{mpi_rank}: shape: {tensor.shape}")
    buffer, count, mpi_type = tc.mpi.tensor2mpiBuffer(tensor)
    comm.Bcast([buffer, count, mpi_type], root=0)

    validation_tensors: list[torch.Tensor] = comm.gather(tensor, root=0)
    print(validation_tensors)

    if mpi_rank == 0:
        for v_tensor in validation_tensors:
            assert torch.all(v_tensor == tensor)


@given(
    shape_dtype=mpi_st(
        st.tuples(
            npst.array_shapes(min_dims=2, max_dims=4, min_side=10, max_side=100),
            st.sampled_from(_dtypes),
        )
    )
)
def test_torch2mpiBuffer_slice_step(shape_dtype):
    shape, dtype = shape_dtype
    print(f"Rank {mpi_rank}")
    print(f"Ransk {comm.gather(mpi_rank, root=0)}")
    if mpi_rank == 0:
        if dtype in [torch.int32, torch.int64]:
            base_tensor = torch.randint(100, shape, dtype=dtype)
        else:
            base_tensor = torch.rand(shape, dtype=dtype)
    # Slice with step parameter on second dimension
    if mpi_rank == 0:
        tensor = base_tensor[:, ::2, ...]
    else:
        tensor = torch.zeros(
            (shape[0],) + ((shape[1] // 2) + (shape[1] % 2),) + shape[2:], dtype=dtype
        )

    print(f"R{mpi_rank}: shape: {tensor.shape}")
    buffer, count, mpi_type = tc.mpi.tensor2mpiBuffer(tensor)
    comm.Bcast([buffer, count, mpi_type], root=0)

    validation_tensors: list[torch.Tensor] = comm.gather(tensor, root=0)
    print(validation_tensors)

    if mpi_rank == 0:
        for v_tensor in validation_tensors:
            assert torch.all(v_tensor == tensor)
