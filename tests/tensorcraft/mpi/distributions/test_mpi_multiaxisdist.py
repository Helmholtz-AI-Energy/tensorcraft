import math
from typing import Optional

import torch
from hypothesis import assume, given, note, settings

from tensorcraft.mpi import MPIMultiAxisDist
from tests.tensorcraft.distributions.test_multi_axis import shape_and_dist


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
