"""Utility functions for the compiler module."""

import logging
import re
from typing import Callable

import networkx as nx
import torch

from tensorcraft.util.axis_utils import multi2linearIndex

log = logging.getLogger("tensorcraft")

TOL = torch.tensor(1e-14)

_torch_ops: dict[str, Callable[..., torch.Tensor]] = {
    "+": torch.add,
    "-": torch.subtract,
    "*": torch.multiply,
    "/": torch.divide,
    "<": torch.less,
    "<=": torch.less_equal,
    "==": torch.eq,
    "!=": torch.not_equal,
    ">=": torch.greater_equal,
    ">": torch.greater,
    "&&": torch.logical_and,
    "||": torch.logical_or,
    "^": torch.pow,
    "abs": torch.abs,
    "exp": torch.exp,
    "log": torch.log,
    "log2": torch.log2,
    "log10": torch.log10,
    "sqrt": torch.sqrt,
    "sin": torch.sin,
    "cos": torch.cos,
    "tan": torch.tan,
    "asin": torch.arcsin,
    "acos": torch.arccos,
    "atan": torch.arctan,
    "sinh": torch.sinh,
    "cosh": torch.cosh,
    "tanh": torch.tanh,
    "asinh": torch.arcsinh,
    "acosh": torch.arccosh,
    "atanh": torch.arctanh,
    "ceil": torch.ceil,
    "floor": torch.floor,
}


def opGraph2Func(op_graph: nx.DiGraph) -> Callable[..., torch.Tensor]:
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

    def compute(**kwargs: torch.Tensor) -> dict[str, torch.Tensor]:
        results: dict[str, torch.Tensor] = {}

        for node in sorted_nodes:
            if node in kwargs:
                result = kwargs[node]
                # log.debug(f"Node: {node}, Result: {result}")
            elif node.split(" ")[0] in _torch_ops:
                op_id = node.split(" ")[0]
                op_inputs = [results[inp] for inp in op_graph.predecessors(node)]
                if any([x.dtype == torch.float for x in op_inputs]) and op_id in [
                    "==",
                    "!=",
                ]:
                    if op_id == "==":
                        result = torch.lt(torch.abs(torch.subtract(*op_inputs)), TOL)
                    elif op_id == "!=":
                        result = torch.ge(torch.abs(torch.subtract(*op_inputs)), TOL)
                else:
                    result = _torch_ops[op_id](*op_inputs)
                    if not isinstance(result, torch.Tensor):
                        result = torch.tensor(result)

                # log.debug(f"Node: {node}, Inputs: {op_inputs}, Result: {result}")
            else:
                result = (
                    torch.tensor(int(node))
                    if "." in node
                    else torch.tensor(float(node), dtype=torch.float64)
                )

            results[node] = result

        return results[sorted_nodes[-1]]

    return compute


def idx_exp_compatible(
    var_name: str,
    idx_exp: list[str],
    tensor: torch.Tensor,
    idx_var_sizes: dict[str, int],
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
) -> torch.Size:
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
    torch.Size
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
                tuple(tmp_shape), tuple(tmp_midx), tuple(range(len(tmp_shape)))
            )
        elif re.match(r"[a-z]", sub_idx):
            current_tensor_midx[i] = current_loop_midx[idx_var_names.index(sub_idx)]
    return torch.Size(current_tensor_midx)
