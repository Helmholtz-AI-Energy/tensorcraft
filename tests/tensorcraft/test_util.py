import numpy as np

from tensorcraft.util import multi2linearIndex, order2npOrder


def test_multi2linearIndex():
    # Test case 1
    dims = np.array([2, 3])
    indices = np.array([1, 1])
    expected_result = 3
    assert multi2linearIndex(dims, indices) == expected_result

    # Test case 2
    dims = np.array([2, 3])
    indices = np.array([1, 1])
    order = np.array([1, 0])
    expected_result = 4
    assert multi2linearIndex(dims, indices, order) == expected_result

    # Additional test cases...
    # Test case 3
    dims = np.array([3, 6])
    indices = np.array([2, 4])
    expected_result = 14
    assert multi2linearIndex(dims, indices) == expected_result

    # Test case 4
    dims = np.array([3, 6])
    indices = np.array([2, 4])
    order = np.array([1, 0])
    expected_result = 16
    assert multi2linearIndex(dims, indices, order) == expected_result

    # Test case 5
    dims = np.array([3, 6, 4, 7])
    indices = np.array([2, 4, 3, 5])
    order = np.array([1, 2, 0])
    expected_result = 70
    assert multi2linearIndex(dims, indices, order) == expected_result


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
