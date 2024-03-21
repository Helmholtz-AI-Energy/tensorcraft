import numpy as np
from hypothesis import given, note
from hypothesis import strategies as st
from hypothesis.extra.numpy import array_shapes

from tensorcraft.util import multi2linearIndex, order2npOrder


@given(
    dims=array_shapes(min_dims=1, max_dims=5, min_side=1, max_side=5),
    use_order=st.booleans(),
)
def test_multi2linearIndex(dims, use_order):
    dims = np.array(dims)
    nDims = len(dims)
    if use_order:
        order = np.arange(nDims)
        np.random.shuffle(order)
    else:
        order = None

    a = np.random.random(size=dims)

    indices = np.array([np.random.randint(0, d) for d in dims])

    if order is not None:
        a_reorderd = a.transpose(order)
        indices_reorderd = indices[order]
    else:
        a_reorderd = a
        indices_reorderd = indices
    a_flat = a_reorderd.flatten("F")
    idx_flat = multi2linearIndex(dims, indices, order)

    note(f"Dimensions: {dims}")
    note(f"Indices: {indices}")
    note(f"Order: {order}")
    note(f"Reordered array: {a_reorderd.shape}")
    note(f"Reordered indices: {indices_reorderd}")
    note(f"Linear index: {idx_flat}")
    note(f"Wanted value: {a_reorderd.item(*indices_reorderd)}")
    note(f"Obtained value: {a_flat.item(idx_flat)}")
    assert a_flat.item(idx_flat) == a_reorderd.item(*indices_reorderd)


def test_order2npOrder():
    # Test case 1
    order = "C"
    expected_result = "F"
    assert order2npOrder(order) == expected_result

    # Test case 2
    order = "R"
    expected_result = "C"
    assert order2npOrder(order) == expected_result

    # Additional test cases...
    # Test case 3
    order = "A"
    # Assuming "A" is not a valid order, you can add an assertion to check for an expected exception
    try:
        order2npOrder(order)
        assert False, "Expected ValueError"
    except ValueError:
        assert True
