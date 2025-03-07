import mpi4py.MPI as MPI
import pytest
import torch

import tensorcraft as tc

comm = MPI.COMM_WORLD
mpi_size = comm.Get_size()
mpi_rank = comm.Get_rank()

if not comm.Get_size() > 1:
    pytest.skip("Skipping MPI tests", allow_module_level=True)


@pytest.mark.parametrize("shape", [(10, 5), (5, 8, 7), (6, 7, 5, 3, 6)])
@pytest.mark.parametrize(
    "dtype", [torch.int32, torch.int64, torch.float32, torch.float64]
)
def test_torch2mpiBuffer_full_tensor(shape, dtype):
    print(f"Rank {mpi_rank}")
    print(f"Ransk {comm.gather(mpi_rank, root=0)}")
    if mpi_rank == 0:
        if dtype in [torch.int32, torch.int64]:
            base_tensor = torch.randint(100, shape, dtype=dtype)
        else:
            base_tensor = torch.rand(shape, dtype=dtype)

    # Send the full tensor, check that other ranks got the same data
    if mpi_rank == 0:
        tensor = base_tensor
    else:
        tensor = torch.zeros(shape, dtype=dtype)

    buffer, count, mpi_type = tc.util.tensor2mpiBuffer(tensor)
    comm.Bcast([buffer, count, mpi_type], root=0)

    validation_tensors: list[torch.Tensor] = comm.gather(tensor, root=0)
    print(validation_tensors)

    if mpi_rank == 0:
        for v_tensor in validation_tensors:
            assert torch.all(v_tensor == tensor)

    comm.Barrier()


@pytest.mark.parametrize("shape", [(10, 5), (5, 8, 7), (6, 7, 5, 3, 6)])
@pytest.mark.parametrize(
    "dtype", [torch.int32, torch.int64, torch.float32, torch.float64]
)
def test_torch2mpiBuffer_transposed_first_last(shape, dtype):
    print(f"Rank {mpi_rank}")
    print(f"Ransk {comm.gather(mpi_rank, root=0)}")
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
    buffer, count, mpi_type = tc.util.tensor2mpiBuffer(tensor)
    comm.Bcast([buffer, count, mpi_type], root=0)

    validation_tensors: list[torch.Tensor] = comm.gather(tensor, root=0)
    print(validation_tensors)

    if mpi_rank == 0:
        for v_tensor in validation_tensors:
            assert torch.all(v_tensor == tensor)


@pytest.mark.parametrize("shape", [(10, 5), (5, 8, 7), (6, 7, 5, 3, 6)])
@pytest.mark.parametrize(
    "dtype", [torch.int32, torch.int64, torch.float32, torch.float64]
)
def test_torch2mpiBuffer_transposed_first_second(shape, dtype):
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
    buffer, count, mpi_type = tc.util.tensor2mpiBuffer(tensor)
    comm.Bcast([buffer, count, mpi_type], root=0)

    validation_tensors: list[torch.Tensor] = comm.gather(tensor, root=0)
    print(validation_tensors)

    if mpi_rank == 0:
        for v_tensor in validation_tensors:
            assert torch.all(v_tensor == tensor)


@pytest.mark.parametrize("shape", [(10, 5), (5, 8, 7), (6, 7, 5, 3, 6)])
@pytest.mark.parametrize(
    "dtype", [torch.int32, torch.int64, torch.float32, torch.float64]
)
def test_torch2mpiBuffer_cont_offset(shape, dtype):
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
    buffer, count, mpi_type = tc.util.tensor2mpiBuffer(tensor)
    comm.Bcast([buffer, count, mpi_type], root=0)

    validation_tensors: list[torch.Tensor] = comm.gather(tensor, root=0)
    print(validation_tensors)

    if mpi_rank == 0:
        for v_tensor in validation_tensors:
            assert torch.all(v_tensor == tensor)


@pytest.mark.parametrize("shape", [(10, 5), (5, 8, 7), (6, 7, 5, 3, 6)])
@pytest.mark.parametrize(
    "dtype", [torch.int32, torch.int64, torch.float32, torch.float64]
)
def test_torch2mpiBuffer_non_cont_slice(shape, dtype):
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
    buffer, count, mpi_type = tc.util.tensor2mpiBuffer(tensor)
    comm.Bcast([buffer, count, mpi_type], root=0)

    validation_tensors: list[torch.Tensor] = comm.gather(tensor, root=0)
    print(validation_tensors)

    if mpi_rank == 0:
        for v_tensor in validation_tensors:
            assert torch.all(v_tensor == tensor)


@pytest.mark.parametrize("shape", [(10, 5), (5, 8, 7), (6, 7, 5, 3, 6)])
@pytest.mark.parametrize(
    "dtype", [torch.int32, torch.int64, torch.float32, torch.float64]
)
def test_torch2mpiBuffer_slice_step(shape, dtype):
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
    buffer, count, mpi_type = tc.util.tensor2mpiBuffer(tensor)
    comm.Bcast([buffer, count, mpi_type], root=0)

    validation_tensors: list[torch.Tensor] = comm.gather(tensor, root=0)
    print(validation_tensors)

    if mpi_rank == 0:
        for v_tensor in validation_tensors:
            assert torch.all(v_tensor == tensor)
