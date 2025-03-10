import random

import hypothesis.strategies as st
from hypothesis import assume, example, given, note

import tensorcraft as tc


@given(
    axis_size=st.integers(min_value=1, max_value=100),
    num_procs=st.integers(min_value=1, max_value=64),
)
@example(axis_size=20, num_procs=3)
@example(axis_size=20, num_procs=8)
def test_maxBlockSize(axis_size, num_procs):
    assume(axis_size >= num_procs)
    max_block_size = tc.dist.Dist.maxBlockSize(axis_size, num_procs)
    note(f"Max block size: {max_block_size}")

    # Check that the block size is less than the axis size
    assert max_block_size <= axis_size
    assert max_block_size >= 1

    # The max_block size times the processor less should fit completely in the axis
    assert max_block_size * (num_procs - 1) <= axis_size

    # The max_block size + 1 times the number of processors should be equal or greater than the axis size
    assert (max_block_size + 1) * num_procs >= axis_size

    if axis_size == 20 and num_procs == 8:
        assert max_block_size == 2

    if axis_size == 20 and num_procs == 3:
        assert max_block_size == 9


@given(
    axis_size=st.integers(min_value=10, max_value=100),
    num_procs=st.integers(min_value=2, max_value=64),
)
def test_axisSplits(axis_size, num_procs):
    assume(axis_size >= num_procs)
    max_block_size = tc.dist.Dist.maxBlockSize(axis_size, num_procs)
    note(f"Max block size: {max_block_size}")

    rand_block_size = random.randint(0, max_block_size)
    note(f"Random block size: {rand_block_size}")

    tile_sizes, tile_ends = tc.dist.Dist.axisSplits(
        axis_size, rand_block_size, num_procs
    )
    note(f"Tile sizes: {tile_sizes}")
    note(f"Tile ends: {tile_ends}")
    assert tile_sizes.sum().item() == axis_size
    assert len(tile_sizes) == len(tile_ends)
    assert len(tile_sizes) >= num_procs
    assert tile_ends[-1] == axis_size
    assert tile_ends[0] == tile_sizes[0]

    assert all([0 < tile_sizes[i] <= max_block_size for i in range(len(tile_sizes))])
