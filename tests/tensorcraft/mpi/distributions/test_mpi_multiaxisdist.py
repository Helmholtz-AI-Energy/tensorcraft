import math
from typing import Optional

import pytest
import torch
from hypothesis import assume, given, note, settings
from hypothesis import strategies as st
from mpi4py import MPI

from tensorcraft.mpi import MPIMultiAxisDist
from tests.tensorcraft.distributions.test_multi_axis import shape_and_dist


@st.composite
def mpi_shape_and_dist(
    draw,
) -> tuple[
    torch.Size, torch.Size, tuple[Optional[tuple[int, ...]], ...], tuple[int, ...]
]:
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    if rank == 0:
        shape_and_dist_tuple = draw(shape_and_dist())
    else:
        shape_and_dist_tuple = None

    shape_and_dist_tuple = comm.bcast(shape_and_dist_tuple, root=0)

    return shape_and_dist_tuple


@given(
    shape_and_dist=shape_and_dist(),
)
@settings(deadline=None)
def test_apply(
    shape_and_dist: tuple[
        torch.Size, torch.Size, tuple[Optional[tuple[int, ...]], ...], tuple[int, ...]
    ],
):
    shape, mesh, mapping, block_sizes = shape_and_dist
    dist = MPIMultiAxisDist(mesh, mapping, block_sizes)

    assume(dist.isDistributed())
    assume(dist.compatible(shape))

    note(f"Shape: {shape}")
    note(f"Dist: {dist}")

    # Create a tensor with the given shape
    n_elements = math.prod(shape)
    g_x = torch.arange(n_elements).reshape(shape)

    global_sum = torch.sum(g_x)

    # Apply the distribution to the tensor

    n_procs = dist.numProcessors

    dist_sum = 0
    for rank in range(n_procs):
        # Create a tensor for the rank
        l_x = dist.apply(g_x, rank)

        # Check that the shape matches the expected local shape
        assert l_x.shape == dist.localShape(shape, rank)

        sorted, _ = torch.sort(l_x)
        assert torch.all(sorted == l_x)

        dist_sum += torch.sum(l_x)

    # Check that the global sum is equal to the local sum
    assert dist_sum % global_sum == 0


@given(
    shape_and_dist=shape_and_dist(),
)
@settings(deadline=None)
def test_apply_split(
    shape_and_dist: tuple[
        torch.Size, torch.Size, tuple[Optional[tuple[int, ...]], ...], tuple[int, ...]
    ],
):
    shape, mesh, mapping, block_sizes = shape_and_dist
    dist = MPIMultiAxisDist(mesh, mapping, block_sizes)

    # Check that there are some non-assigned dimensions

    non_assigned_dims = set(range(len(mesh))) - set(
        [y for x in mapping if x is not None for y in x]
    )
    assume(len(non_assigned_dims) > 0)

    assume(dist.isDistributed())
    assume(dist.compatible(shape))

    note(f"Shape: {shape}")
    note(f"Dist: {dist}")

    n_procs = dist.numProcessors
    g_tensor = torch.arange(math.prod(shape)).reshape(shape)

    for rank in range(n_procs):
        l_tensor = dist.apply(g_tensor, rank)

        for split_dim in non_assigned_dims:
            for axis in range(len(shape)):
                for minor in [True, False]:
                    try:
                        split_dist, split_local = dist.apply_split(
                            shape,
                            l_tensor,
                            rank,
                            axis,
                            split_dim,
                            block_size=1,
                            minor=minor,
                        )
                        note(f"Rank: {rank}")
                        note(f"Split dim: {split_dim}")
                        note(f"Split axis: {axis}")
                        note(f"Minor: {minor}")
                        note(f"Split local shape: {split_local.shape}")
                        note(f"Split dist: {split_dist}")

                        assert split_local.shape == split_dist.localShape(shape, rank)

                        sorted, _ = torch.sort(split_local)
                        assert torch.all(sorted == split_local)

                        expected_split = split_dist.apply(g_tensor, rank)
                        assert split_local.shape == expected_split.shape
                        assert torch.all(split_local == expected_split)

                    except ValueError:
                        continue


@pytest.mark.mpi_test
@given(
    shape_and_dist=shape_and_dist(),
)
@settings(deadline=None)
def test_apply_allgather(
    shape_and_dist: tuple[
        torch.Size, torch.Size, tuple[Optional[tuple[int, ...]], ...], tuple[int, ...]
    ],
):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    shape_and_dist = comm.bcast(shape_and_dist, root=0)
    shape, mesh, mapping, block_sizes = shape_and_dist

    dist = MPIMultiAxisDist(mesh, mapping, block_sizes)

    # Check that there are some non-assigned dimensions

    assigned_dims = set([y for x in mapping if x is not None for y in x])
    assume(assigned_dims > 0)

    assume(dist.isDistributed())
    assume(dist.compatible(shape))

    note(f"Shape: {shape}")
    note(f"Dist: {dist}")

    n_procs = dist.numProcessors
    g_tensor = torch.arange(math.prod(shape)).reshape(shape)

    for rank in range(n_procs):
        l_tensor = dist.apply(g_tensor, rank)

        for split_dim in non_assigned_dims:
            for axis in len(shape):
                for minor in [True, False]:
                    try:
                        split_dist, split_local = dist.apply_split(
                            shape,
                            l_tensor,
                            rank,
                            axis,
                            split_dim,
                            block_size=1,
                            minor=minor,
                        )

                        assert split_local.shape == split_dist.localShape(shape, rank)

                        sorted, _ = torch.sort(split_local)
                        assert torch.all(sorted == split_local)

                        expected_split = split_dist.apply(g_tensor, rank)
                        assert split_local.shape == expected_split.shape
                        assert torch.all(split_local == expected_split)

                    except ValueError:
                        continue
