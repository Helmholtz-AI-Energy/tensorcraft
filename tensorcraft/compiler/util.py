"""Utility functions for the compiler module."""

import logging
import re
from typing import Callable

import networkx as nx
import numpy as np

from tensorcraft.types import MIndex, ScalarType, TensorType, is_scalar_type
from tensorcraft.util import multi2linearIndex

log = logging.getLogger("tensorcraft")

TOL = 1e-11

_numpy_ops: dict[str, Callable[..., np.number]] = {
    "+": np.add,
    "-": np.subtract,
    "*": np.multiply,
    "/": np.divide,
    "<": np.less,
    "<=": np.less_equal,
    "==": np.equal,
    "!=": np.not_equal,
    ">=": np.greater_equal,
    ">": np.greater,
    "&&": np.logical_and,
    "||": np.logical_or,
}


def opGraph2Func(op_graph: nx.DiGraph) -> Callable[..., ScalarType]:
    """Convert a directed graph of operations to a function.

    This function takes a directed graph of operations and converts it to a function
    that can be used to compute the final result of the operations.

    Parameters
    ----------
    op_graph : nx.DiGraph
        A directed graph of operations.

    Returns
    -------
    Callable[..., ScalarType]
        A function that can be used to compute the final result of the operations.
    """
    sorted_nodes = list(nx.topological_sort(op_graph))

    def compute(**kwargs: ScalarType) -> ScalarType:
        results = {}

        for node in sorted_nodes:
            if node in kwargs:
                results[node] = kwargs[node]
            elif re.match(r"[\+\-\*\>\/\<\=\!\&\|]{1,2}\s\d+", node):
                op_id = node.split(" ")[0]
                op_inputs = [results[inp] for inp in op_graph.predecessors(node)]
                if np.any(
                    [isinstance(x, np.floating) for x in op_inputs]
                ) and op_id in ["==", "!="]:
                    if op_id == "==":
                        results[node] = np.abs(np.subtract(*op_inputs)) < TOL  # type: ignore
                    elif op_id == "!=":
                        results[node] = np.abs(np.subtract(*op_inputs)) >= TOL  # type: ignore
                else:
                    results[node] = _numpy_ops[op_id](*op_inputs)
            else:
                results[node] = np.float64(node) if "." in node else np.int64(node)

        return results[sorted_nodes[-1]]

    return compute


def idx_exp_compatible(
    var_name: str, idx_exp: list[str], tensor: TensorType, idx_var_sizes: dict[str, int]
) -> bool:
    """Check if an index expression is compatible with a tensor.

    This function checks if an index expression is compatible with a tensor. The index
    expression is compatible with the tensor if the index variables in the expression
    are compatible with the tensor's shape. If there is no existing information about the index variables, it will be stored on the idx_var_sizes dictionary.

    Parameters
    ----------
    idx_exp : str
        The index expression to check.
    tensor : np.ndarray
        The tensor to check the index expression against.
    idx_var_sizes : dict[str, int]
        A dictionary mapping index variables to their sizes.

    Returns
    -------
    bool
        True if the index expression is compatible with the tensor, False otherwise.
    """
    if is_scalar_type(tensor):
        data_shape: list[int] = []
        data_order = 0
    else:
        data_shape = tensor.shape  # type: ignore
        data_order = len(data_shape)

    if data_order != len(idx_exp):
        log.error(f"Variable {var_name} has incompatible order.")
        return False

    for idx_var, size in zip(idx_exp, data_shape):
        if re.match(r"\d+", idx_var):
            if int(idx_var) > size:
                log.error(f"Index {idx_var} is out of bounds for variable {var_name}.")
                return False
        elif re.match(r"[a-z]{2,}", idx_var):
            min_size = 1
            for char in idx_var:
                if idx_var_sizes[char] == 0:
                    min_size *= 1
                else:
                    min_size *= idx_var_sizes[char]

            if min_size > size:
                log.error(f"Variable {var_name} has incompatible shape.")
                return False

        elif re.match(r"[a-z]", idx_var):
            if idx_var_sizes[idx_var] == 0:
                idx_var_sizes[idx_var] = size
            elif idx_var_sizes[idx_var] != size:
                log.error(
                    f"Variable {var_name} has non-compatile non-compatible shape."
                )
                return False

    return True


def idx_exp2multiIdx(
    idx_exp: list[str],
    idx_var_names: list[str],
    current_loop_midx: list[int],
    idx_var_sizes: list[int],
) -> MIndex:
    """Convert an index expression to a multi-index.

    This function converts an index expression to a multi-index. The index expression
    is a list of strings that represent the index variables and constants that are
    used to access a tensor. The index variables are mapped to their current values
    in the current loop iteration.

    Parameters
    ----------
    idx_exp : list[str]
        The index expression to convert.
    idx_var_names : list[str]
        A list of index variable names.
    current_loop_midx : list[int]
        A list of the current values of the index variables.
    idx_var_sizes : list[int]
        A list of the sizes of the index variables.

    Returns
    -------
    MIndex
        The multi-index that corresponds to the index expression.
    """
    current_tensor_midx = [
        0,
    ] * len(idx_exp)
    for i, sub_idx in enumerate(idx_exp):
        if re.match(r"\d+", sub_idx):
            current_tensor_midx[i] = int(sub_idx)
        elif re.match(r"[a-z]{2,}", sub_idx):
            tmp_midx = []
            tmp_shape = []
            for char in sub_idx:
                idx_var_idx = idx_var_names.index(char)
                tmp_midx.append(current_loop_midx[idx_var_idx])
                tmp_shape.append(idx_var_sizes[idx_var_idx])
            current_tensor_midx[i] = multi2linearIndex(
                tuple(tmp_shape), tuple(tmp_midx), np.array(range(len(tmp_shape)))[::-1]
            )
        elif re.match(r"[a-z]", sub_idx):
            current_tensor_midx[i] = current_loop_midx[idx_var_names.index(sub_idx)]
    return tuple(current_tensor_midx)
