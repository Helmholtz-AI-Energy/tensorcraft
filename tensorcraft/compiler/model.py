"""A module containing classes for representing a tensor expression and a program."""

import logging
from dataclasses import dataclass
from typing import Optional

import networkx as nx
import numpy as np

from tensorcraft.compiler.util import _opGraph2Func
from tensorcraft.types import MIndex, TensorType, is_scalar_type
from tensorcraft.util import linear2multiIndex

log = logging.getLogger("tensorcraft")


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
        self,
        input_data: dict[str, TensorType],
        output_shape_hint: Optional[MIndex] = None,
    ) -> TensorType:
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
            if input_name not in input_data:
                raise ValueError(f"Input {input_name} is missing.")

            if is_scalar_type(input_data[input_name]):
                data_shape: list[int] = []
                data_order = 0
            else:
                data_shape = input_data[input_name].shape  # type: ignore
                data_order = len(data_shape)

            if data_order != len(input_shape):
                raise ValueError(f"Input {input_name} has a wrong order.")
            for idx_var, size in zip(input_shape, data_shape):
                if index_variables_sizes[idx_var] == 0:
                    index_variables_sizes[idx_var] = size
                elif index_variables_sizes[idx_var] != size:
                    raise ValueError("Input variables have non-compatile shapes.")

        if self.output[0] in input_data:
            if is_scalar_type(input_data[self.output[0]]):
                output_shape: MIndex = tuple([])
                output_order = 0
            else:
                output_shape = input_data[self.output[0]].shape  # type: ignore
                output_order = len(output_shape)

            if output_order != len(self.output[1]):
                raise ValueError("Output data has a wrong order.")

            for idx_var, size in zip(self.output[1], output_shape):  # type: ignore
                if index_variables_sizes[idx_var] != size:
                    raise ValueError("Output has a non-compatible shape.")

            output_array = input_data[self.output[0]]
        else:
            if output_shape_hint is not None:
                if len(output_shape_hint) != len(self.output[1]):
                    raise ValueError("Output has a wrong order.")
                for idx_var, size in zip(self.output[1], output_shape_hint):  # type: ignore
                    if index_variables_sizes[idx_var] == 0:
                        index_variables_sizes[idx_var] = size
                    elif index_variables_sizes[idx_var] != size:
                        raise ValueError("Output has a non-compatible shape.")
                output_shape = output_shape_hint
            else:
                output_shape = tuple(
                    [index_variables_sizes[idx_var] for idx_var in self.output[1]]
                )

            # Check that all index variables have a size
            for idx_var in self.index_variables:
                if index_variables_sizes[idx_var] == 0:
                    raise ValueError(
                        f"Index variable {idx_var} has no size. Please provide an output shape."
                    )

            # Initialize the output array
            output_array = np.zeros(
                output_shape, dtype=np.result_type(*input_data.values())
            )

        # Loop over the index_variables
        index_var_names = self.index_variables
        index_var_dims = tuple(
            index_variables_sizes[idx_var] for idx_var in index_var_names
        )
        var_idx_masks = {
            f"{var}[{','.join(idxs)}]": (var, [index_var_names.index(i) for i in idxs])
            for (var, idxs) in self.inputs
        }
        output_index_mask = [index_var_names.index(i) for i in self.output[1]]

        op_func = _opGraph2Func(self.op_graph)

        # If there are no index variables, evaluate the expression directly, it is a scalar operation
        if len(index_var_names) == 0:
            elementwise_inputs = {}
            for var_id, (var_name, mask) in var_idx_masks.items():
                value = input_data[var_name]
                elementwise_inputs[var_id] = value

            value = op_func(**elementwise_inputs)
            output_array = value

        else:
            for i in range(np.prod(index_var_dims)):
                # Read elements from input arrays, write them on the operation string and evaluate
                loop_mindex = np.array(linear2multiIndex(i, index_var_dims))
                elementwise_inputs = {}

                for var_id, (var_name, mask) in var_idx_masks.items():
                    idxs = tuple(loop_mindex[mask])
                    if is_scalar_type(input_data[var_name]):
                        value = input_data[var_name]
                    else:
                        value = input_data[var_name][idxs]  # type: ignore
                    elementwise_inputs[var_id] = value

                for index_var in index_var_names:
                    elementwise_inputs[index_var] = loop_mindex[
                        index_var_names.index(index_var)
                    ]

                value = op_func(**elementwise_inputs)

                output_index = tuple(loop_mindex[output_index_mask])

                output_array[output_index] += value  # type: ignore

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
        for line, tensor_expression in self.tensor_expressions.items():
            for var_name, var_shape in tensor_expression.inputs:
                if var_name not in self.variables:
                    log.error(
                        f"Variable {var_name} in {line} not found in the program."
                    )
                    return False
                if self.variables[var_name].order != len(var_shape):
                    log.error(
                        f"Variable {var_name} in {line} has an incompatible order."
                    )
                    return False
            if tensor_expression.output[0] not in self.variables:
                log.error(f"Variable {var_name} in {line} not found in the program.")
                return False

            if self.variables[tensor_expression.output[0]].order != len(
                tensor_expression.output[1]
            ):
                log.error(f"Variable {var_name} in {line} has an incompatible order.")
                return False

        if not nx.is_directed_acyclic_graph(self.graph):
            log.error("The graph is not acyclic.")
            return False

        for tensor_expression in self.tensor_expressions.values():
            if not nx.is_directed_acyclic_graph(tensor_expression.op_graph):
                log.error(
                    f"Expression graph for {tensor_expression.line} is not acyclic."
                )
                return False
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
