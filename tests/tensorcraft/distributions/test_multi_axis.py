import pytest
import torch

from tensorcraft.distributions.multi_axis import MultiAxisDist


@pytest.mark.parametrize(
    "processor_mesh, dims_mapping, block_sizes, isDistributed",
    [
        (torch.Size([2, 2]), ((0,), (1,)), 1, True),
        (torch.Size([2, 2]), ((0,), (1,)), 2, True),
        (torch.Size([3, 2]), ((0,), (1,)), 3, True),
        (torch.Size([4, 5]), (None,), 4, False),
        (torch.Size([2, 2]), ((), ()), 4, False),
        (torch.Size([2, 5]), ((), None), 2, False),
    ],
)
def test_isDistributed(processor_mesh, dims_mapping, block_sizes, isDistributed):
    dist = MultiAxisDist(processor_mesh, dims_mapping, block_sizes)
    assert dist.isDistributed() == isDistributed
