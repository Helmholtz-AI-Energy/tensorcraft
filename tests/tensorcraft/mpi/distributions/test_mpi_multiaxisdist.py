import math
from typing import Optional

import pytest
import torch
from hypothesis import assume, given, note, settings
from mpi4py import MPI

from tensorcraft.mpi import MPIMultiAxisDist
from tests.tensorcraft.distributions.test_multi_axis import shape_and_dist
from tests.tensorcraft.mpi.test_mpi_utils import AllAssert, mpi_st

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
world_size = comm.Get_size()


@given(
    shape_and_dist=shape_and_dist(is_distributed=True, is_compatible=True),
)
@settings(deadline=None)
def test_apply(
    shape_and_dist: tuple[
        torch.Size, torch.Size, tuple[Optional[tuple[int, ...]], ...], tuple[int, ...]
    ],
):
    shape, dist = shape_and_dist
    dist = MPIMultiAxisDist.fromMultiAxisDist(dist)

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
    shape_and_dist=shape_and_dist(is_compatible=True),
)
@settings(deadline=None)
def test_apply_split(
    shape_and_dist: tuple[
        torch.Size, torch.Size, tuple[Optional[tuple[int, ...]], ...], tuple[int, ...]
    ],
):
    shape, dist = shape_and_dist
    dist = MPIMultiAxisDist.fromMultiAxisDist(dist)

    mesh = dist.processorMesh
    mapping = dist._dims_mapping

    # Check that there are some non-assigned dimensions

    non_assigned_dims = set(range(len(mesh))) - set(
        [y for x in mapping if x is not None for y in x]
    )
    assume(len(non_assigned_dims) > 0)

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


@pytest.mark.mpi_test(4)
@given(
    shape_and_dist=mpi_st(
        shape_and_dist(
            mesh=torch.Size([2, 2]),
            is_compatible=True,
            is_distributed=True,
            max_axis_size=60,
        )
    )
)
@settings(deadline=None)
def test_apply_allgather_single_dim(
    shape_and_dist: tuple[torch.Size, MPIMultiAxisDist],
):
    shape, dist = shape_and_dist
    dist = MPIMultiAxisDist.fromMultiAxisDist(dist)

    # Check that there are some non-assigned dimensions
    assigned_dims = set([y for x in dist._dims_mapping if x is not None for y in x])

    if rank == 0:
        note(f"Shape: {shape}")
        note(f"Dist: {dist}")

    g_tensor = torch.arange(math.prod(shape)).reshape(shape)

    l_tensor = dist.apply(g_tensor, rank)

    for gather_dim in assigned_dims:
        try:
            gathered_dist, gathered_local = dist.apply_allgather(
                shape,
                l_tensor,
                comm,
                gather_dim,
            )

            assert gathered_local.shape == gathered_dist.localShape(shape, rank)

            sorted, _ = torch.sort(gathered_local)
            AllAssert(comm, torch.all(sorted == gathered_local))

            expected_split = gathered_dist.apply(g_tensor, rank)
            AllAssert(comm, gathered_local.shape == expected_split.shape)
            AllAssert(comm, torch.all(gathered_local == expected_split))

        except ValueError:
            continue


@pytest.mark.mpi_test(4)
@given(
    shape_and_dist=mpi_st(
        shape_and_dist(
            mesh=torch.Size([2, 2]),
            is_compatible=True,
            is_distributed=True,
            max_axis_size=75,
        )
    )
)
@settings(deadline=None)
def test_apply_allgather_all_dims(shape_and_dist: tuple[torch.Size, MPIMultiAxisDist]):
    shape, dist = shape_and_dist
    dist = MPIMultiAxisDist.fromMultiAxisDist(dist)

    if rank == 0:
        note(f"Shape: {shape}")
        note(f"Dist: {dist}")

    g_tensor = torch.arange(math.prod(shape)).reshape(shape)

    l_tensor = dist.apply(g_tensor, rank)

    gathered_dist, gathered_local = dist.apply_allgather(shape, l_tensor, comm)
    AllAssert(comm, gathered_local.shape == g_tensor.shape)
    AllAssert(comm, torch.all(gathered_local == g_tensor))

    assert not gathered_dist.isDistributed()


@pytest.mark.mpi_test(8)
@given(
    shape_and_dist=mpi_st(
        shape_and_dist(
            mesh=torch.Size([2, 2, 2]),
            is_compatible=True,
            is_distributed=True,
            max_axis_size=75,
        )
    )
)
@settings(deadline=None)
def test_apply_permute_cube(shape_and_dist: tuple[torch.Size, MPIMultiAxisDist]):
    shape, dist = shape_and_dist
    dist = MPIMultiAxisDist.fromMultiAxisDist(dist)

    target_axis = -1
    target_mapping = tuple()

    for axis, mapping in enumerate(dist._dims_mapping):
        if len(mapping) >= 2:
            target_axis = axis
            target_mapping = mapping

    assume(target_axis != -1)

    swap_mesh_dims = [target_mapping[0], target_mapping[-1]]

    if rank == 0:
        note(f"Shape: {shape}")
        note(f"Dist: {dist}")
        note(f"Swap: {swap_mesh_dims}")

    print("Hello")

    g_tensor = torch.arange(math.prod(shape)).reshape(shape)

    l_tensor = dist.apply(g_tensor, rank)

    print("Ready for the real test!")

    new_dist, new_l_tensor = dist.apply_permute(shape, l_tensor, comm, swap_mesh_dims)
    print("In the  middle of a good test!")
    expected_l_tensor = new_dist.apply(g_tensor, rank)
    print("Exited the big test!")

    AllAssert(comm, expected_l_tensor.shape == new_l_tensor.shape)
    AllAssert(comm, torch.all(expected_l_tensor == new_l_tensor))

    print("Bye!")


# @pytest.mark.mpi_test(8)
# @given(
#     shape_and_dist=mpi_st(
#         shape_and_dist(
#             mesh=torch.Size([2, 4]),
#             is_compatible=True,
#             is_distributed=True,
#             max_axis_size=75,
#         )
#     )
# )
# @settings(deadline=None)
# def test_apply_permute_rectangle(shape_and_dist: tuple[torch.Size, MPIMultiAxisDist]):

#     shape, dist = shape_and_dist
#     dist = MPIMultiAxisDist.fromMultiAxisDist(dist)

#     target_axis = -1
#     target_mapping = tuple()

#     for axis, mapping in enumerate(dist._dims_mapping):
#         if len(mapping) >= 2:
#             target_axis = axis
#             target_mapping = mapping

#     assume(target_axis != -1)

#     swap_mesh_dims = [0,1]

#     if rank == 0:
#         note(f"Shape: {shape}")
#         note(f"Dist: {dist}")
#         note(f"Swap: {swap_mesh_dims}")


#     g_tensor = torch.arange(math.prod(shape)).reshape(shape)

#     l_tensor = dist.apply(g_tensor, rank)

#     new_dist, new_l_tensor = dist.apply_permute(shape, l_tensor, comm, swap_mesh_dims)
#     expected_l_tensor = new_dist.apply(g_tensor, rank)

#     assert expected_l_tensor.shape == new_l_tensor.shape
#     assert torch.all(expected_l_tensor == new_l_tensor)
