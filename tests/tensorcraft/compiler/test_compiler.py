import networkx as nx
import pytest
from hypothesis import given
from hypothesis import strategies as st

import tensorcraft as tc
from tensorcraft.compiler.model import Program

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
