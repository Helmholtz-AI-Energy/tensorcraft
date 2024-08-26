from typing import Callable

import hypothesis.extra.numpy as npst
import networkx as nx
import numpy as np
import pytest
from hypothesis import given, note, settings
from hypothesis import strategies as st

import tensorcraft as tc
from tensorcraft.compiler.model import Program

TOL = 1e-2

# [Operation string, op_count, loop_depth]
_operations = [
    ("A = B - C", 1, 0),  # Scalar Substraction
    ("A[i,k] = B[i,j] * C[j,k]", 1, 3),  # Matrix Matrix Multiplication
    ("A[i] = B[i] + C[i]", 1, 1),  # Element-wise Vector Addition
    ("A[i,j,k] = B[i,j,k] / C[i,j,k]", 1, 3),  # Element-wise Tensor Division
    ("A = B[i] * C[i]", 1, 1),  # Dot Product
    ("A[i,j] = B[i,j] + C", 1, 2),  # Scalar Addition
    ("A = B[i]", 0, 1),  # Vector Summation
    ("A[i] = B[i,j]", 0, 2),  # Row summation
    ("A[j] = B[i,j]", 0, 2),  # Column summation
    ("A[i] = B[i,i]", 0, 1),  # Diagonal
    ("A[i] = i", 0, 1),  # Range vector
    (
        "Dist[j,k] = (Samples[j,l] - Clusters[k,l]) * (Samples[j,l] - Clusters[k,l])",
        3,
        3,
    ),  # Distance computation
    ("Zeros[i,j,k] = 0", 0, 3),  # Zero tensor
    # "Min[j] = (Dist[j,k] < Min[j]) * (Dist[j,k] - Min[j])", # Minimum value
]

index_names = ["i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t"]


@given(program=st.text())
def test_invalid_programs(program: str):
    with pytest.raises(ValueError) as exc_info:
        tc.compile(program)
    print(exc_info.value)
    assert "Invalid program" in str(exc_info.value)


@pytest.mark.parametrize("operations,op_count,loop_depth", _operations)
def test_valid_operations(operations: str, op_count: int, loop_depth: int):
    program = tc.compile(operations)
    assert program is not None
    assert isinstance(program, Program)
    assert len(program.tensor_expressions) == 1
    assert program.graph is not None
    assert isinstance(program.graph, nx.DiGraph)
    assert len(program.variables) - len(program.input_variables) == 1

    # Check if the expression graph is correct
    print(program.tensor_expressions)
    op_graph = program.tensor_expressions[1].op_graph
    inputs = program.tensor_expressions[1].inputs
    with pytest.raises(nx.NetworkXNoCycle) as e:
        nx.find_cycle(op_graph, [node for node, _ in inputs])
    assert e is not None

    # Check if the loop depth is correct
    print(program.tensor_expressions[1])
    assert program.tensor_expressions[1].loop_count == loop_depth

    # Check if the operation count is correct
    assert program.tensor_expressions[1].op_count == op_count


@pytest.mark.filterwarnings("ignore:overflow:RuntimeWarning")
@given(
    data=st.data(),
    op=st.one_of(
        st.just(("+", np.add)),
        st.just(("-", np.subtract)),
        st.just(("*", np.multiply)),
        st.just(("/", np.divide)),
        st.just(("<", np.less)),
        st.just(("<=", np.less_equal)),
        st.just(("==", np.equal)),
        st.just(("!=", np.not_equal)),
        st.just((">", np.greater)),
        st.just((">=", np.greater_equal)),
    ),
    dtype=st.one_of(npst.floating_dtypes(), npst.integer_dtypes()),
)
def test_scalar_ops(data, op: tuple[str, Callable], dtype: np.dtype):
    a = data.draw(npst.from_dtype(dtype, allow_nan=False, allow_infinity=False))
    b = data.draw(npst.from_dtype(dtype, allow_nan=False, allow_infinity=False))
    if op[0] == "/" and b == 0:
        b = 1
    expected = op[1](a, b)

    program = tc.compile(f"C = A {op[0]} B")
    note(f"Expected: {expected}")
    result = program.tensor_expressions[1]({"A": a, "B": b})
    note(f"Result: {result}")
    assert result == expected


@pytest.mark.filterwarnings("ignore:overflow:RuntimeWarning")
@given(
    op=st.one_of(
        st.just(("+", np.add)),
        st.just(("-", np.subtract)),
        st.just(("*", np.multiply)),
        st.just(("/", np.divide)),
        st.just(("<", np.less)),
        st.just(("<=", np.less_equal)),
        st.just(("==", np.equal)),
        st.just(("!=", np.not_equal)),
        st.just((">", np.greater)),
        st.just((">=", np.greater_equal)),
    ),
    vector=npst.arrays(
        dtype=np.dtype("float32"),
        shape=npst.array_shapes(min_dims=1, max_dims=5, max_side=5),
        elements=npst.from_dtype(
            np.dtype("float32"), allow_nan=False, allow_infinity=False, max_value=1000
        ),
    ),
    scalar=npst.from_dtype(
        np.dtype("float32"), allow_nan=False, allow_infinity=False, max_value=1000
    ),
)
@settings(deadline=None)
def test_tensor_scalar_ops(op, vector, scalar):
    if op[0] == "/" and scalar == 0:
        scalar = 1

    expected = op[1](vector, scalar)
    note(f"Expected: {expected}")

    idx_str = ",".join([index_names[i] for i in range(len(vector.shape))])
    note(f"Index string: {idx_str}")

    program = tc.compile(f"C[{idx_str}] = A[{idx_str}] {op[0]} B")
    result = program.tensor_expressions[1]({"A": vector, "B": scalar})
    note(f"Result: {result}")
    assert np.allclose(result, expected, atol=TOL)


@pytest.mark.filterwarnings("ignore:overflow:RuntimeWarning")
@given(
    data=st.data(),
    op=st.one_of(
        st.just(("+", np.add)),
        st.just(("-", np.subtract)),
        st.just(("*", np.multiply)),
        st.just(("/", np.divide)),
        st.just(("<", np.less)),
        st.just(("<=", np.less_equal)),
        st.just(("==", np.equal)),
        st.just(("!=", np.not_equal)),
        st.just((">", np.greater)),
        st.just((">=", np.greater_equal)),
    ),
    shape=npst.array_shapes(min_dims=1, max_dims=4, min_side=1, max_side=5),
)
@settings(deadline=None)
def test_tensor_elementwise_ops(data, op, shape):
    a = data.draw(
        npst.arrays(
            dtype=np.dtype("float32"),
            shape=shape,
            elements=npst.from_dtype(
                np.dtype("float32"),
                allow_nan=False,
                allow_infinity=False,
                max_value=1000,
            ),
        )
    )
    b = data.draw(
        npst.arrays(
            dtype=np.dtype("float32"),
            shape=shape,
            elements=npst.from_dtype(
                np.dtype("float32"),
                allow_nan=False,
                allow_infinity=False,
                max_value=1000,
            ),
        )
    )

    if op[0] == "/" and np.any(b == 0):
        b[b == 0] = 1
    expected = op[1](a, b)

    idx_str = ",".join([index_names[i] for i in range(len(a.shape))])

    program = tc.compile(f"C[{idx_str}] = A[{idx_str}] {op[0]} B[{idx_str}]")
    result = program.tensor_expressions[1]({"A": a, "B": b})
    assert np.allclose(result, expected, atol=TOL)


@pytest.mark.filterwarnings("ignore:overflow:RuntimeWarning")
@given(
    data=st.data(),
    shape=npst.array_shapes(min_dims=1, max_dims=1, max_side=100),
)
def test_vector_dot(data, shape):
    a = data.draw(
        npst.arrays(
            dtype=np.dtype("float32"),
            shape=shape,
            elements=npst.from_dtype(
                np.dtype("float32"),
                allow_nan=False,
                allow_infinity=False,
                min_value=-100,
                max_value=100,
            ),
        )
    )
    b = data.draw(
        npst.arrays(
            dtype=np.dtype("float32"),
            shape=shape,
            elements=npst.from_dtype(
                np.dtype("float32"),
                allow_nan=False,
                allow_infinity=False,
                min_value=-100,
                max_value=100,
            ),
        )
    )
    expected = a @ b
    note(f"Expected: {expected}, {expected.dtype}")

    program = tc.compile("C += A[i] * B[i]")
    result = program.tensor_expressions[1]({"A": a, "B": b})
    note(f"Result: {result}, {result.dtype}")
    assert np.allclose(result, expected, atol=TOL)


@pytest.mark.filterwarnings("ignore:overflow:RuntimeWarning")
@given(
    data=st.data(),
    shape=npst.array_shapes(min_dims=3, max_dims=3, min_side=2),
)
def test_matrix_dot(data, shape):
    A = data.draw(
        npst.arrays(
            dtype=np.dtype("float32"),
            shape=(shape[0], shape[1]),
            elements=npst.from_dtype(
                np.dtype("float32"),
                allow_nan=False,
                allow_infinity=False,
                min_value=-100,
                max_value=100,
            ),
        )
    )
    B = data.draw(
        npst.arrays(
            dtype=np.dtype("float32"),
            shape=(shape[1], shape[2]),
            elements=npst.from_dtype(
                np.dtype("float32"),
                allow_nan=False,
                allow_infinity=False,
                min_value=-100,
                max_value=100,
            ),
        )
    )

    expected = np.dot(A, B)
    note(f"Expected: {expected}, {expected.dtype}")

    program = tc.compile("C[i,j] += A[i,k] * B[k,j]")
    result = program.tensor_expressions[1]({"A": A, "B": B})
    note(f"Result: {result}, {result.dtype}")
    assert np.allclose(result, expected, atol=TOL)


@given(
    data=st.data(),
    shape=npst.array_shapes(min_dims=2, max_dims=4, max_side=5),
)
def test_reduction(data, shape):
    A = data.draw(
        npst.arrays(
            dtype=np.dtype("float32"),
            shape=shape,
            elements=npst.from_dtype(
                np.dtype("float32"),
                allow_nan=False,
                allow_infinity=False,
                min_value=-100,
                max_value=100,
            ),
        )
    )
    expected = np.sum(A)

    idx_str = ",".join([index_names[i] for i in range(len(shape))])
    program = tc.compile(f"C += A[{idx_str}]")
    result = program.tensor_expressions[1]({"A": A})
    assert np.allclose(result, expected, atol=TOL)

    expected = np.sum(A, axis=tuple(range(1, len(shape))))
    program = tc.compile(f"C[i] += A[{idx_str}]")
    result = program.tensor_expressions[1]({"A": A})
    assert np.allclose(result, expected, atol=TOL)


@given(
    data=st.data(),
    shape=npst.array_shapes(min_dims=2, max_dims=2, min_side=2),
)
def test_reshape(data, shape):
    A = data.draw(
        npst.arrays(
            dtype=np.dtype("float32"),
            shape=shape,
            elements=npst.from_dtype(
                np.dtype("float32"),
                allow_nan=False,
                allow_infinity=False,
                min_value=-100,
                max_value=100,
            ),
        )
    )
    excepted = A.reshape(shape[0] * shape[1])
    program = tc.compile("C[(ij)] = A[i,j]")
    result = program.tensor_expressions[1]({"A": A})
    assert np.allclose(result, excepted, atol=TOL)

    A = data.draw(
        npst.arrays(
            dtype=np.dtype("float32"),
            shape=(shape[0] * shape[1]),
            elements=npst.from_dtype(
                np.dtype("float32"),
                allow_nan=False,
                allow_infinity=False,
                min_value=-100,
                max_value=100,
            ),
        )
    )
    excepted = A.reshape(shape)
    program = tc.compile("C[i,j] = A[(ij)]")
    result = program.tensor_expressions[1]({"A": A}, output_shape_hint=shape)
    assert np.allclose(result, excepted, atol=TOL)
