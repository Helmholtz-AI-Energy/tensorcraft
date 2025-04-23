from typing import Optional

import pytest
import torch
from hypothesis import strategies as st

from tensorcraft.distributions.multi_axis import MultiAxisDist


@st.composite
def shape_and_dist(
    draw,
) -> tuple[
    torch.Size, torch.Size, tuple[Optional[tuple[int, ...]], ...], tuple[int, ...]
]:
    shape = torch.Size(
        draw(st.lists(st.integers(min_value=50, max_value=100), min_size=1, max_size=4))
    )
    mesh = draw(st.lists(st.integers(min_value=2, max_value=3), min_size=1, max_size=3))
    block_sizes = draw(st.integers(min_value=1, max_value=5))

    available_dims = list(range(len(mesh)))
    # Create mappings, as list with the same lenght as the shape and with non-repeating values from available_dims
    mappings = []
    for i in range(len(shape)):
        if len(available_dims) == 0:
            mappings.append(None)
        else:
            # Choose a random number of dimensions to map
            num_dims = draw(st.integers(min_value=0, max_value=len(available_dims)))
            axis_map = []
            for j in range(num_dims):
                dim = draw(st.sampled_from(available_dims))
                axis_map.append(dim)
                available_dims.remove(dim)

            # Append the mapping to the list
            mappings.append(tuple(axis_map))

    return shape, mesh, mappings, block_sizes


@pytest.mark.parametrize(
    "processor_mesh, dims_mapping, block_sizes, raise_error",
    [
        # Block sizes and dimensions must match
        (torch.Size([2, 2]), ((0,), (1,)), (2, 2, 2, 2), True),
        (torch.Size([2, 2]), ((0,), (1,)), (2,), True),
        (torch.Size([2, 2]), ((0,), (1,), None), (2, 2), True),
        # Processor mesh cannot be empty
        (torch.Size([]), ((0,), (1,)), 1, True),
        # Out of bounds mesh mappings
        (2, ((0,), (1,)), (2, 2, 2, 2), True),
        (torch.Size([2, 2]), ((0,), (2,)), (2,), True),
        (torch.Size([2, 2, 5, 2]), ((-1,), (1,), None), (2, 2), True),
        # Non-matching block sizes and mappings
        (torch.Size([2, 2]), ((0,), (1,)), (2, 2, 3), True),
        (torch.Size([2, 2]), ((0,), None), (2,), True),
        (torch.Size([2, 2]), ((0,), None, (1,)), (2, 3, 3, 3), True),
        # Repeated mesh dimensions
        (
            torch.Size([2, 2]),
            ((1,), (1,)),
            (
                2,
                2,
            ),
            True,
        ),
        (torch.Size([2, 2, 3]), ((2,), (2,)), 2, True),
        (torch.Size([2, 2, 3]), ((2, 2), None), 2, True),
        (torch.Size([2, 2]), ((0,), None, (1,), (0,)), (2, 3, 3, 3), True),
        # Block size <= 0
        (torch.Size([2, 2]), ((0,), (1,)), -2, True),
        (torch.Size([2, 2]), ((0,), (1,)), 0, True),
        (torch.Size([2, 2]), ((0,), (1,)), (2, -3), True),
        (torch.Size([2, 2]), ((0,), (1,)), (0, 1), True),
        # Valid cases
        (torch.Size([2, 3]), ((0, 1), ()), 1, False),
        (torch.Size([3, 3]), ((0,), (1,)), (5, 5), False),
        (torch.Size([3, 3]), ((1, 0), ()), (-1, 5), False),
        (torch.Size([3, 3]), ((0, 1), ()), (1, None), False),
    ],
)
def test_init(processor_mesh, dims_mapping, block_sizes, raise_error):
    if raise_error:
        with pytest.raises(ValueError):
            MultiAxisDist(processor_mesh, dims_mapping, block_sizes)
        return
    dist = MultiAxisDist(processor_mesh, dims_mapping, block_sizes)

    assert dist.processorMesh == processor_mesh
    assert isinstance(dist.dimsMapping, tuple)
    for mapping in dist.dimsMapping:
        assert isinstance(mapping, tuple)
    assert len(dist.dimsMapping) == len(dist.blockSizes)
    assert isinstance(dist.dimsMapping, tuple)
    for block_size in dist.blockSizes:
        assert isinstance(block_size, int) and (block_size > 0 or block_size == -1)


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


@pytest.mark.parametrize(
    "dist1, dist2, expected",
    [
        (
            MultiAxisDist(torch.Size([2, 2]), ((0,), (1,)), 1),
            MultiAxisDist(torch.Size([2, 2]), ((0,), (1,)), 1),
            True,
        ),
        (
            MultiAxisDist(torch.Size([2, 2]), ((0,), (1,)), 1),
            MultiAxisDist(torch.Size([2, 2]), ((0,), (1,)), 2),
            False,
        ),
        (
            MultiAxisDist(torch.Size([2, 2]), ((0,), (1,)), 1),
            MultiAxisDist(torch.Size([3, 3]), ((0,), (1,)), 1),
            False,
        ),
        (
            MultiAxisDist(torch.Size([3, 3]), ((0,), (1,)), 1),
            MultiAxisDist(torch.Size([3, 3]), ((0,), (1,)), 1),
            True,
        ),
        (
            MultiAxisDist(torch.Size([3, 3]), ((0,), (1,)), (1, 1)),
            MultiAxisDist(torch.Size([3, 3]), ((0,), (1,)), 1),
            True,
        ),
    ],
)
def test_equal(dist1, dist2, expected):
    assert (dist1 == dist2) == expected


@pytest.mark.parametrize(
    "dist, expected",
    [
        (MultiAxisDist(torch.Size([2, 2]), ((0,), ()), 4), "D_[2,2]⊥{0,∅}(4,∅)"),
        (
            MultiAxisDist(torch.Size([2, 2, 2]), ((0, 1), (2,)), 1),
            "D_[2,2,2]⊥{(0,1),2}(1,1)",
        ),
        (
            MultiAxisDist(torch.Size([2, 2, 2]), ((0, 2), (1,)), 4),
            "D_[2,2,2]⊥{(0,2),1}(4,4)",
        ),
        (
            MultiAxisDist(torch.Size([2, 2, 2]), ((), (0, 1, 2)), 3),
            "D_[2,2,2]⊥{∅,(0,1,2)}(∅,3)",
        ),
        (
            MultiAxisDist(torch.Size([2, 2, 2]), ((), (0, 1, 2)), 4),
            "D_[2,2,2]⊥{∅,(0,1,2)}(∅,4)",
        ),
    ],
)
def test_str(dist, expected):
    assert str(dist) == expected


@pytest.mark.parametrize(
    "dist, shape, maximum",
    [
        (
            MultiAxisDist(torch.Size([2, 3]), ((1, 0), ()), (2, 1)),
            torch.Size((20, 1)),
            1,
        ),
        (
            MultiAxisDist(torch.Size([2, 3]), ((0, 1), ()), (2, 1)),
            torch.Size((20, 1)),
            1,
        ),
        (MultiAxisDist(torch.Size([2, 2]), ((0,), ()), 4), torch.Size((20, 20)), 2),
        (MultiAxisDist(torch.Size([2, 2]), (None, None), 4), torch.Size((20, 20)), 4),
    ],
)
def test_processorView(dist: MultiAxisDist, shape: torch.Size, maximum: int):
    processor_view = dist.processorView(shape)

    assert isinstance(processor_view, torch.Tensor)

    # Check the shape of the processor view is equal to the shape + the number of processors in the mesh
    assert processor_view.shape == shape + (dist.numProcessors,)

    # Check that that at least every element appers once in the processor view
    summarized_pv = processor_view.sum(dim=-1)
    assert torch.all(processor_view.sum(dim=-1) > 0)

    # Check that the highest value corresponds is less than the number of processors
    assert torch.all(summarized_pv <= maximum)


@pytest.mark.parametrize(
    "dist, shape",
    [
        # Order and mapping dims do not match
        (MultiAxisDist(torch.Size([2, 3]), ((1, 0),), (2,)), torch.Size((20, 1))),
        (
            MultiAxisDist(torch.Size([2, 3, 2]), ((0, 1), (), (2,)), 1),
            torch.Size((20, 1)),
        ),
        # Non compatible axis
        (MultiAxisDist(torch.Size([2, 2]), ((0,), ()), 20), torch.Size((20, 20))),
        (
            MultiAxisDist(
                torch.Size(
                    [
                        10,
                    ]
                ),
                (None, (0,)),
                3,
            ),
            torch.Size((20, 20)),
        ),
    ],
)
def test_not_compatible(dist: MultiAxisDist, shape: torch.Size):
    assert not dist.compatible(shape)


@pytest.mark.parametrize(
    "kwargs, target_dist",
    [
        ({"gather_mesh_dim": 0}, MultiAxisDist(torch.Size([2, 2, 2]), ((1,), (2,)), 1)),
        (
            {"gather_mesh_dim": 1},
            MultiAxisDist(torch.Size([2, 2, 2]), ((0,), (2,)), (2, 1)),
        ),
        (
            {"gather_mesh_dim": 2},
            MultiAxisDist(torch.Size([2, 2, 2]), ((0, 1), None), 1),
        ),
    ],
)
def test_allgather(kwargs, target_dist):
    tensor_shape = torch.Size([10, 10])
    mesh = torch.Size([2, 2, 2])
    dist = MultiAxisDist(mesh, ((0, 1), (2,)), 1)

    result_dist, n_elements, cost = dist.allgather(tensor_shape, **kwargs)
    assert isinstance(result_dist, MultiAxisDist)
    assert result_dist == target_dist


@pytest.mark.parametrize(
    "kwargs, target_dist",
    [
        (
            {"tensor_axis": 0, "mesh_dims": 1, "block_size": 2},
            MultiAxisDist(torch.Size([2, 2]), ((1, 0), None), 4),
        ),
        (
            {"tensor_axis": 0, "mesh_dims": 1, "block_size": 2, "minor": True},
            MultiAxisDist(torch.Size([2, 2]), ((0, 1), None), 2),
        ),
        (
            {"tensor_axis": 1, "mesh_dims": 1, "block_size": 2},
            MultiAxisDist(torch.Size([2, 2]), ((0,), (1,)), (4, 2)),
        ),
        (
            {"tensor_axis": 1, "mesh_dims": 1, "block_size": 2, "minor": True},
            MultiAxisDist(torch.Size([2, 2]), ((0,), (1,)), (4, 2)),
        ),
    ],
)
def test_split(kwargs, target_dist):
    tensor_shape = torch.Size([20, 20])
    mesh = torch.Size([2, 2])
    dist = MultiAxisDist(mesh, ((0,), ()), 4)

    result_dist, n_elements, cost = dist.split(tensor_shape, **kwargs)
    assert isinstance(result_dist, MultiAxisDist)
    assert result_dist == target_dist
    assert n_elements == 0
    assert cost == 0


@pytest.mark.parametrize(
    "kwargs, target_dist",
    [
        (
            {"from_tensor_axis": 0, "to_tensor_axis": 1},
            MultiAxisDist(torch.Size([2, 2, 2]), (None, (0, 2, 1)), 4),
        ),
        (
            {"from_tensor_axis": 0, "to_tensor_axis": 1, "minor": True},
            MultiAxisDist(torch.Size([2, 2, 2]), ((0,), (1, 2)), (8, 2)),
        ),
        (
            {"from_tensor_axis": 1, "to_tensor_axis": 0},
            MultiAxisDist(torch.Size([2, 2, 2]), ((1, 0, 2), None), 4),
        ),
    ],
)
def test_alltoall(kwargs, target_dist):
    tensor_shape = torch.Size([32, 32])
    mesh = torch.Size([2, 2, 2])
    dist = MultiAxisDist(mesh, ((0, 2), (1,)), 4)

    result_dist, n_elements, cost = dist.alltoall(tensor_shape, **kwargs)
    assert isinstance(result_dist, MultiAxisDist)
    assert result_dist == target_dist


@pytest.mark.parametrize(
    "kwargs, target_dist",
    [
        (
            {"mesh_dims": (0, 1)},
            MultiAxisDist(torch.Size([2, 2, 2]), (None, (1, 0, 2)), 3),
        ),
        (
            {"mesh_dims": (0, 2)},
            MultiAxisDist(torch.Size([2, 2, 2]), (None, (2, 1, 0)), 3),
        ),
        (
            {"mesh_dims": (1, 2)},
            MultiAxisDist(torch.Size([2, 2, 2]), (None, (0, 2, 1)), 3),
        ),
    ],
)
def test_permute(kwargs, target_dist):
    tensor_shape = torch.Size([10, 24])
    mesh = torch.Size([2, 2, 2])
    dist = MultiAxisDist(mesh, (None, (0, 1, 2)), 3)

    result_dist, n_elements, cost = dist.permute(tensor_shape, **kwargs)
    assert isinstance(result_dist, MultiAxisDist)
    assert result_dist == target_dist


@pytest.mark.parametrize(
    "kwargs, target_dist",
    [
        (
            {"tensor_axis": 1, "block_size": 1},
            MultiAxisDist(torch.Size([2, 2, 2]), (None, (0, 1, 2)), 1),
        ),
        (
            {"tensor_axis": 1, "block_size": 2},
            MultiAxisDist(torch.Size([2, 2, 2]), (None, (0, 1, 2)), 2),
        ),
        (
            {"tensor_axis": 1, "block_size": 8},
            MultiAxisDist(torch.Size([2, 2, 2]), (None, (0, 1, 2)), 8),
        ),
    ],
)
def test_change_block_size(kwargs, target_dist):
    tensor_shape = torch.Size([10, 60])
    mesh = torch.Size([2, 2, 2])
    dist = MultiAxisDist(mesh, (None, (0, 1, 2)), 4)

    result_dist, n_elements, cost = dist.change_block_size(tensor_shape, **kwargs)
    assert isinstance(result_dist, MultiAxisDist)
    assert result_dist == target_dist
