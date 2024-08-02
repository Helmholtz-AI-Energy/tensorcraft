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
    op_count: int

    def __str__(self):
        """Return the string representation of the tensor expression."""
        return f"Line: {self.line}: {self.raw}"


@dataclass
class Program:
    """A class representing a program containing tensor expressions."""

    graph: nx.DiGraph
    tensor_expressions: dict[int, TensorExpression]
    max_loop_depth: tuple[int, int]  # (max_loop_depth, line_number)
    variables: dict[str, TensorVariable]
    input_variables: list[str]
