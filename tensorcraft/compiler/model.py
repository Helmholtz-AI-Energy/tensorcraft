"""A module containing classes for representing a tensor expression and a program."""

from dataclasses import dataclass

import networkx as nx


@dataclass
class TensorVariable:
    """A class representing a variable in a tensor expression."""

    name: str
    order: int
    lines: list[int]

    def __str__(self):
        """Return the string representation of the tensor variable."""
        return f"{self.name} {self.order} {self.lines}"


@dataclass
class TensorExpression:
    """A class representing an operation in a tensor expression."""

    line: int
    raw: str
    inputs: list[tuple[str, list[str]]]
    output: tuple[str, list[str]]
    loop_count: int
    op_graph: nx.DiGraph
    op_count: int

    def __str__(self):
        """Return the string representation of the tensor expression."""
        return f"Line: {self.line}: {self.raw}"


class Program:
    """A class representing a program containing tensor expressions."""

    def __init__(
        self,
        graph: nx.DiGraph,
        input_variables: list[str],
        variables: dict[str, TensorVariable],
        tensor_expressions: dict[int, TensorExpression],
        max_loop_depth: tuple[int, int],
        max_op_count: tuple[int, int],
    ):
        self._graph = graph
        self._input_variables = input_variables
        self._variables = variables
        self._tensor_expressions = tensor_expressions
        self._max_loop_depth = max_loop_depth
        self._max_op_count = max_op_count

        if not self._is_correct():
            raise ValueError("The program is not correct.")

    def _is_correct(self) -> bool:
        """Check if the program is correct.

        1) Variables should keep a consistent shape throughout the program.
        2) The output of an operation should match the shape of the output variable.

        """
        return True

    @property
    def graph(self) -> nx.DiGraph:
        """Program graph as networkx DiGraph object."""
        return self._graph

    @property
    def input_variables(self) -> list[str]:
        """Input variables in the program."""
        return self._input_variables

    @property
    def variables(self) -> dict[str, TensorVariable]:
        """Program variables. Keys are variable names."""
        return self._variables

    @property
    def tensor_expressions(self) -> dict[int, TensorExpression]:
        """Dictionary with tensor expressions in the program. Keys are line numbers."""
        return self._tensor_expressions
