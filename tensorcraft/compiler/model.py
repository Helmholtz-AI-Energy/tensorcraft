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


class TensorExpression:
    """A class representing an operation in a tensor expression."""

    def __init__(
        self,
        line: int,
        raw: str,
        inputs: list[tuple[str, list[str]]],
        output: tuple[str, list[str]],
        loop_count: int,
        op_graph: nx.DiGraph,
        op_count: int,
    ):
        self.line = line  # Line number in the program
        self.raw = raw  # Raw string of the expression
        self.inputs = inputs  # List of input variables and their shapes
        self.output = output  # Output variable and its shape
        self.loop_count = loop_count  # Number of loops in the expression
        self.op_graph = op_graph  # Graph representing the expression
        self.op_count = op_count  # Number of operations in the expression
        self._index_variables: list[
            str
        ] = []  # List of indexed variables used in the model

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
            self._index_variables = list(vars)

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
                if index_variables_sizes[idx_var] == 0:
                    index_variables_sizes[idx_var] = size
                elif index_variables_sizes[idx_var] != size:
                    raise ValueError("Input variables have non-compatile shapes.")

        if self.output[0] in inputs:
            for idx_var, size in zip(self.output[1], inputs[self.output[0]].shape):
                if index_variables_sizes[idx_var] != size:
                    raise ValueError("Output has a non-compatible shape.")

            output_array = inputs[self.output[0]]
        else:
            if output_shape is not None:
                if len(output_shape) != len(self.output[1]):
                    raise ValueError("Output has a wrong order.")
                for idx_var, size in zip(self.output[1], output_shape):
                    if index_variables_sizes[idx_var] == 0:
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
        index_var_names = self.index_variables
        index_var_dims = np.array(
            [index_variables_sizes[idx_var] for idx_var in index_var_names]
        )
        var_idx_masks = {
            f"{var}[{','.join(idxs)}]": (var, [index_var_names.index(i) for i in idxs])
            for (var, idxs) in self.inputs
        }
        output_index_mask = [index_var_names.index(i) for i in self.output[1]]

        # Prep the string for evaluations

        for i in range(np.prod(index_var_dims)):
            # Read elements from input arrays, write them on the operation string and evaluate
            loop_mindex = linear2multiIndex(i, index_var_dims)
            op_string = self.raw.split("=")[1].strip()
            for var_id, (var_name, mask) in var_idx_masks.items():
                idxs = tuple(loop_mindex[mask])
                value = inputs[var_name][idxs]
                op_string = op_string.replace(var_id, str(value))

            for index_var in index_var_names:
                op_string = op_string.replace(
                    index_var, str(loop_mindex[index_var_names.index(index_var)])
                )

            value = eval(op_string)
            output_index = tuple(loop_mindex[output_index_mask])

            output_array[output_index] += value

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
