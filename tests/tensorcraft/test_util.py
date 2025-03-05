import random

import torch
from hypothesis import given, note, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import array_shapes

from tensorcraft.util import linear2multiIndex, multi2linearIndex, order2npOrder


@given(
    dims=array_shapes(min_dims=1, max_dims=5, min_side=1, max_side=5),
    use_order=st.booleans(),
)
def test_multi2linearIndex(dims, use_order):
    nDims = len(dims)
    if use_order:
        order = torch.randperm(nDims).tolist()
    else:
        order = None

    a = torch.randn(dims)

    indices = tuple(random.randint(0, d - 1) for d in dims)

    if order is not None:
        a_reorderd = a.permute(order)
        indices_reorderd = tuple(indices[o] for o in order)
    else:
        a_reorderd = a
        indices_reorderd = indices
    a_flat = a_reorderd.flatten()
    idx_flat = multi2linearIndex(dims, indices, order)

    note(f"Dimensions: {dims}")
    note(f"Indices: {indices}")
    note(f"Order: {order}")
    note(f"a: {a}")
    note(f"a_flat: {a_flat}")
    note(f"Reordered array: {a_reorderd.shape}")
    note(f"Reordered indices: {indices_reorderd}")
    note(f"Linear index: {idx_flat}")
    note(f"Wanted value: {a_reorderd[indices_reorderd].item()}")
    note(f"Obtained value: {a_flat[idx_flat].item()}")
    assert a_flat[idx_flat].item() == a_reorderd[indices_reorderd].item()


@given(
    dims=array_shapes(min_dims=1, max_dims=5, min_side=1, max_side=5),
    order=st.sampled_from(["R", "C"]),
)
@settings(deadline=None)
def test_linear2multiIndex(dims, order):
    a = torch.randn(dims)

    if order == "C":
        a = a.permute(*torch.arange(a.ndim - 1, -1, -1))
        a_flat = a.flatten()
    else:
        a_flat = a.flatten()

    idx = random.randint(0, a.numel() - 1)
    midx = linear2multiIndex(idx, dims, order)

    note(f"Dimensions: {dims}")
    note(f"Order: {order}")
    note(f"a: {a}")
    note(f"a_flat: {a_flat}")
    note(f"Linear index: {idx}")
    note(f"MIndex: {midx}")

    expected = a_flat[idx].item()
    result = a[midx].item()
    note(f"Wanted value: {expected}")
    note(f"Obtained value: {result}")
    assert expected == result


def test_order2npOrder():
    # Test case 1
    order = "C"
    expected_result = "F"
    assert order2npOrder(order) == expected_result  # type: ignore

    # Test case 2
    order = "R"
    expected_result = "C"
    assert order2npOrder(order) == expected_result  # type: ignore

    # Additional test cases...
    # Test case 3
    order = "A"
    # Assuming "A" is not a valid order, you can add an assertion to check for an expected exception
    try:
        order2npOrder(order)  # type: ignore
        assert False, "Expected ValueError"
    except ValueError:
        assert True
