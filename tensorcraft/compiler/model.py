"""A module containing classes for representing a tensor expression and a program."""

import enum
import logging
import re
from dataclasses import dataclass
from typing import Optional

import networkx as nx
import numpy as np

from tensorcraft.compiler.util import idx_exp2multiIdx, idx_exp_compatible, opGraph2Func
from tensorcraft.types import MIndex, TensorType, is_scalar_type
from tensorcraft.util import linear2multiIndex

log = logging.getLogger("tensorcraft")


class AssignmentType(enum.Enum):
    """An enumeration of the different types of assignments in a tensor expression."""

    ASSIGN = "="
    ADD = "+="
    SUB = "-="
    MUL = "*="
    DIV = "/="


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
        assignment_type: AssignmentType,
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
        self.assignment_type = assignment_type  # Type of assignment
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
                    if re.match(r"[a-z]{2,}", idx_var):
                        for char in idx_var:
                            vars.add(char)
                    elif re.match(r"[a-z]", idx_var):
                        vars.add(idx_var)
            for idx_var in self.output[1]:
                if re.match(r"[a-z]{2,}", idx_var):
                    for char in idx_var:
                        vars.add(char)
                elif re.match(r"[a-z]", idx_var):
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
        # If there are no index variables, evaluate the expression directly, it is a scalar operation
        op_func = opGraph2Func(self.op_graph)
        output_array: TensorType = 0

        if len(self.index_variables) == 0:
            elementwise_inputs = {}
            var_id_name_dict = {
                f"{var}[{','.join(idxs)}]": var for var, idxs in self.inputs
            }

            for var_id, var_name in var_id_name_dict.items():
                value = input_data[var_name]
                elementwise_inputs[var_id] = value

            value = op_func(**elementwise_inputs)
            output_array = value

        else:
            # Check if the given arrays match the input shapes
            index_variables_sizes = {idx_var: 0 for idx_var in self.index_variables}

            for input_name, input_idx_exp in self.inputs:
                if input_name not in input_data:
                    raise ValueError(f"Input {input_name} is missing.")
                if not idx_exp_compatible(
                    input_name,
                    input_idx_exp,
                    input_data[input_name],
                    index_variables_sizes,
                ):
                    raise ValueError(f"Input {input_name} has an incompatible shape.")

            # If the output array is also an input array, check for consistency with between the two.
            if self.output[0] in input_data:
                if not idx_exp_compatible(
                    self.output[0],
                    self.output[1],
                    input_data[self.output[0]],
                    index_variables_sizes,
                ):
                    raise ValueError(
                        f"Output {self.output[0]} has an incompatible shape."
                    )

                output_array = input_data[self.output[0]]
            else:
                # If the output shape is provided, check the shape_hint for consistency and initialize the output array
                if output_shape_hint is not None:
                    # Initialize the output array
                    if len(input_data) == 0:
                        output_array = np.zeros(output_shape_hint, dtype=np.float64)
                    else:
                        dtype = np.result_type(*input_data.values())
                        output_array = np.zeros(output_shape_hint, dtype=dtype)

                    if not idx_exp_compatible(
                        self.output[0],
                        self.output[1],
                        output_array,
                        index_variables_sizes,
                    ):
                        raise ValueError(
                            f"Output {self.output[0]} has an incompatible shape."
                        )

                else:
                    output_shape: MIndex = tuple()
                    for idx_var in self.output[1]:
                        if re.match(r"\d+", idx_var):
                            raise ValueError(
                                "Writing on a slice of output variable that has not been provided as an input is not supported."
                            )
                        elif re.match(r"[a-z]{2,}", idx_var):
                            size = 1
                            for char in idx_var:
                                if index_variables_sizes[char] == 0:
                                    raise ValueError(
                                        f"Index variable {char} has no size. Please provide an output shape."
                                    )
                                size *= index_variables_sizes[char]
                            output_shape += (size,)
                        elif re.match(r"[a-z]", idx_var):
                            if index_variables_sizes[idx_var] == 0:
                                raise ValueError(
                                    f"Index variable {idx_var} has no size. Please provide an output shape."
                                )
                            output_shape += (index_variables_sizes[idx_var],)

                    if len(input_data) == 0:
                        output_array = np.zeros(output_shape, dtype=np.float64)
                    else:
                        dtype = np.result_type(*input_data.values())
                        output_array = np.zeros(output_shape, dtype=dtype)

                    if not idx_exp_compatible(
                        self.output[0],
                        self.output[1],
                        output_array,
                        index_variables_sizes,
                    ):
                        raise ValueError(
                            f"Output {self.output[0]} has an incompatible shape."
                        )

            # Check that all index variables have a size
            for idx_var in self.index_variables:
                if index_variables_sizes[idx_var] == 0:
                    raise ValueError(
                        f"Index variable {idx_var} has no size. Please provide an output shape."
                    )

            # Prep idx masks for each input and output variable
            index_var_names = self.index_variables
            index_var_dims = tuple(
                index_variables_sizes[idx_var] for idx_var in index_var_names
            )
            var_idx_exp_dict = {
                f"{var}[{','.join(idxs)}]": (var, idxs) for (var, idxs) in self.inputs
            }

            print(f"Line: {self.raw}: index vars = {index_var_names}")
            for i in range(np.prod(index_var_dims)):
                # Read elements from input arrays, write them on the operation string and evaluate
                loop_mindex = np.array(linear2multiIndex(i, index_var_dims))
                elementwise_inputs = {}

                for var_id, (var_name, idx_exp_list) in var_idx_exp_dict.items():
                    if is_scalar_type(input_data[var_name]):
                        value = input_data[var_name]
                    else:
                        var_midx = idx_exp2multiIdx(
                            idx_exp_list,  # type: ignore
                            index_var_names,  # type: ignore
                            loop_mindex,  # type: ignore
                            index_var_dims,  # type: ignore
                        )
                        value = input_data[var_name][var_midx]  # type: ignore
                    elementwise_inputs[var_id] = value

                for index_var in index_var_names:
                    elementwise_inputs[index_var] = loop_mindex[
                        index_var_names.index(index_var)
                    ]

                value = op_func(**elementwise_inputs)

                output_midx = idx_exp2multiIdx(
                    self.output[1],  # type: ignore
                    index_var_names,  # type: ignore
                    loop_mindex,  # type: ignore
                    index_var_dims,  # type: ignore
                )

                match self.assignment_type:
                    case AssignmentType.ASSIGN:
                        output_array[output_midx] = value  # type: ignore
                    case AssignmentType.ADD:
                        output_array[output_midx] += value  # type: ignore
                    case AssignmentType.SUB:
                        output_array[output_midx] -= value  # type: ignore
                    case AssignmentType.MUL:
                        output_array[output_midx] *= value  # type: ignore
                    case AssignmentType.DIV:
                        output_array[output_midx] /= value  # type: ignore

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

    def execute(
        self, inputs: dict[str, TensorType], shape_hints: dict[str, MIndex]
    ) -> dict[str, TensorType]:
        """Execute the program.

        Parameters
        ----------
        inputs : dict[str, np.ndarray]
            Dictionary containing the input arrays. Keys are the variable names.

        Returns
        -------
        dict[str, np.ndarray]
            Dictionary containing the output arrays. Keys are the variable names.
        """
        # 1. Check that all variable shapes are provided
        for var_name, var_metadata in self.variables.items():
            if var_name in self.input_variables:
                if var_name not in inputs:
                    raise ValueError(f"Input {var_name} is missing.")
                if not is_scalar_type(inputs[var_name]):
                    if var_metadata.order != len(inputs[var_name].shape):  # type: ignore
                        raise ValueError(f"Input {var_name} has an incompatible order.")
                else:
                    if var_metadata.order != 0:
                        raise ValueError(f"Input {var_name} has an incompatible order.")
            else:
                if var_name not in shape_hints:
                    raise ValueError(f"Variable {var_name} has no shape hint.")
                if var_metadata.order != len(shape_hints[var_name]):
                    raise ValueError(f"Variable {var_name} has an incompatible order")

        # 2. Execute the expressions
        nodes = list(nx.topological_sort(self.graph))
        outputs: dict[str, TensorType] = {}
        for node in nodes:
            if isinstance(node, int):
                # Execute the expression
                tensor_expression = self.tensor_expressions[node]
                exp_output_var = tensor_expression.output[0]
                exp_input = {
                    exp_input_var: outputs[exp_input_var]
                    for exp_input_var, _ in tensor_expression.inputs
                }

                output_shape_hint = shape_hints[exp_output_var]
                outputTensor = tensor_expression(exp_input, output_shape_hint)
                outputs[exp_output_var] = outputTensor
            else:
                # This is a an input variable, just get the value from the inputs
                outputs[node] = inputs[node]

        return outputs
