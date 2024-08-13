"""A module containing classes for representing a tensor expression and a program."""

from dataclasses import dataclass
from typing import Optional

import networkx as nx
import numpy as np

from tensorcraft.types import MIndex
from tensorcraft.util import linear2multiIndex


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

    line: int  # Line number in the program
    raw: str  # Raw string of the expression
    inputs: list[tuple[str, list[str]]]  # List of input variables and their shapes
    output: tuple[str, list[str]]  # Output variable and its shape
    loop_count: int  # Number of loops in the expression
    op_graph: nx.DiGraph  # Graph representing the expression
    op_count: int  # Number of operations in the expression

    def __str__(self):
        """Return the string representation of the tensor expression."""
        return f"Line: {self.line}: {self.raw}"

    @property
    def index_variables(self) -> list[str]:
        """
        Returns a list of indexed variables used in the model.

        Returns
        -------
        list[str]
            List of indexed variables used in the model.
        """
        if not self._index_variables:
            vars = set()

            for _, input_shape in self.inputs:
                for idx_var in input_shape:
                    vars.add(idx_var)
            for idx_var in self.output[1]:
                vars.add(idx_var)
            self._index_variables: list[str] = list(vars)

        return self._index_variables

    def __call__(
        self, inputs: dict[str, np.ndarray], output_shape: Optional[MIndex] = None
    ) -> np.ndarray:
        """Evaluate the tensor expression.

        Parameters
        ----------
        inputs : dict[str, np.ndarray]
            Dictionary containing the input arrays. Keys are the variable names.
        output_shape : tuple[int], optional
            Shape of the output array. If None, the shape is inferred from the input arrays.

        Returns
        -------
        np.ndarray
            The output array resulting from the evaluation of the expression.
        """
        # Check if the given arrays match the input shapes
        index_variables_sizes = {idx_var: 0 for idx_var in self.index_variables}

        for input_name, input_shape in self.inputs:
            if input_name not in inputs:
                raise ValueError(f"Input {input_name} is missing.")
            if len(inputs[input_name].shape) != len(input_shape):
                raise ValueError(f"Input {input_name} has a wrong order.")
            for idx_var, size in zip(input_shape, inputs[input_name].shape):
                if index_variables_sizes == 0:
                    index_variables_sizes[idx_var] = size
                elif index_variables_sizes[idx_var] != size:
                    raise ValueError("Input variables have non-compatile shapes.")

        if output_shape is not None:
            if len(output_shape) != len(self.output[1]):
                raise ValueError("Output has a wrong order.")
            for idx_var, size in zip(self.output[1], output_shape):
                if index_variables_sizes == 0:
                    index_variables_sizes[idx_var] = size
                elif index_variables_sizes[idx_var] != size:
                    raise ValueError("Output has a non-compatible shape.")
        else:
            output_shape = np.array(
                [index_variables_sizes[idx_var] for idx_var in self.output[1]]
            )

        # Check that all index variables have a size
        for idx_var in self.index_variables:
            if index_variables_sizes[idx_var] == 0:
                raise ValueError(f"Index variable {idx_var} has no size.")

        # Initialize the output array
        output_array = np.zeros(output_shape)

        # Loop over the index_variables
        index_vars_names = self.index_variables
        index_var_dims = np.array(
            [index_variables_sizes[idx_var] for idx_var in index_vars_names]
        )

        # Prep the string for evaluations

        for i in range(np.prod(index_var_dims)):
            loop_mindex = linear2multiIndex(i, index_var_dims)
            print(loop_mindex)

            # Read elements from input arrays
            # input_values = {}

        return output_array


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
        2) The graph should be acyclic.
        3) Each expression should be acyclic.

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
