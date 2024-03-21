import pytest
import numpy as np
from tensorcraft.tensor import Tensor
from hypothesis import given, note, strategies as st
from hypothesis.extra.numpy import arrays, array_shapes, from_dtype

@given(
    list_of_ints=st.one_of(
        st.none(),
        st.integers(min_value=-10000, max_value=10000),
        st.floats(),
        st.lists(st.integers(min_value=-10000, max_value=10000)),
        st.tuples(st.integers(min_value=-10000, max_value=10000)),
        arrays(dtype=np.number, shape=array_shapes(min_dims=1, max_dims=3, min_side=0, max_side=5))
    )
)
def test_tensor_constructor(list_of_ints):
    note(f"list_of_ints: {list_of_ints}")
    if isinstance(list_of_ints, float):
        with pytest.raises(ValueError) as exc_info:
            tensor = Tensor(list_of_ints)
        assert "Invalid dimensions" in str(exc_info.value)
        return 
    if isinstance(list_of_ints, int):
        if list_of_ints < 1:
            with pytest.raises(ValueError) as exc_info:
                tensor = Tensor(list_of_ints)
            assert "Dimensions must be positive" in str(exc_info.value)
            return
        else:
            tensor = Tensor(list_of_ints)
    if isinstance(list_of_ints, (list, tuple)):
        if len(list_of_ints) == 0:
            with pytest.raises(ValueError) as exc_info:
                tensor = Tensor(list_of_ints)
            assert "Must have at least one dimension" in str(exc_info.value)
            return
        elif isinstance(list_of_ints[0], float):
            with pytest.raises(ValueError) as exc_info:
                tensor = Tensor(list_of_ints)
            assert "Dimensions must be integers" in str(exc_info.value)
            return
        elif not np.all(np.array(list_of_ints) > 0):
            with pytest.raises(ValueError) as exc_info:
                tensor = Tensor(list_of_ints)
            assert "Dimensions must be positive" in str(exc_info.value)
            return
        else:
            tensor = Tensor(list_of_ints)

    if isinstance(list_of_ints, np.ndarray):
        if not np.issubdtype(list_of_ints.dtype, np.integer):
            with pytest.raises(ValueError) as exc_info:
                tensor = Tensor(list_of_ints)
            assert "Dimensions must be integers" in str(exc_info.value)
            return
        elif len(list_of_ints.shape) != 1:
            with pytest.raises(ValueError) as exc_info:
                tensor = Tensor(list_of_ints)
            assert "Must have at least one dimension" in str(exc_info.value)
            return
        elif len(list_of_ints) == 0:
            with pytest.raises(ValueError) as exc_info:
                tensor = Tensor(list_of_ints)
            assert "Must have at least one dimension" in str(exc_info.value)
            return
        elif not np.all(list_of_ints > 0):
            with pytest.raises(ValueError) as exc_info:
                tensor = Tensor(list_of_ints)
            assert "Dimensions must be positive" in str(exc_info.value)
            return
        else:
            tensor = Tensor(list_of_ints)

@given(list_of_ints=st.lists(st.integers(min_value=1, max_value=100), min_size=1, max_size=5))
def test_order(list_of_ints):
    tensor = Tensor(list_of_ints)
    expected_result = len(list_of_ints)
    assert tensor.order == expected_result

@given(list_of_ints=st.lists(st.integers(min_value=1, max_value=100), min_size=1, max_size=5))
def test_shape(list_of_ints):
    tensor = Tensor(list_of_ints)
    expected_result = np.array(list_of_ints, dtype=np.int_)
    assert np.all(tensor.shape == expected_result)

@given(list_of_ints=st.lists(st.integers(min_value=1, max_value=100), min_size=1, max_size=5))
def test_size(list_of_ints):
    tensor = Tensor(list_of_ints)
    expected_result = np.prod(np.array(list_of_ints), dtype=int)
    assert tensor.size == expected_result

@given(
    list_of_ints=st.lists(st.integers(min_value=1, max_value=100), min_size=1, max_size=5),
    row_order=st.booleans()
)
def test_index_tranformations(list_of_ints, row_order):
    tensor = Tensor(list_of_ints)
    note(f"Tensor shape: {tensor.shape}")

    order = "R" if row_order else "C"

    mindex = np.array([np.random.randint(0, d) for d in tensor.shape])
    note(f"Multi-index: {mindex}")
    linear_index = tensor.getLinearIndex(mindex, order)
    note(f"Linear index: {linear_index}")
    result_mindex = tensor.getMultiIndex(linear_index, order)
    note(f"Resulting multi-index: {result_mindex}")
    assert np.all(result_mindex == mindex)

    idx = np.random.randint(0, tensor.size)
    note(f"Linear index: {idx}")
    mindex = tensor.getMultiIndex(idx, order)
    note(f"Resulting multi-index: {mindex}")
    linear_index = tensor.getLinearIndex(mindex, order)
    note(f"Resulting linear index: {linear_index}")
    assert linear_index == idx

