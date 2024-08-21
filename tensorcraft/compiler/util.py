"""Utility functions for the compiler module."""

import re
from typing import Callable

import networkx as nx
import numpy as np

from tensorcraft.types import ScalarType

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


def _opGraph2Func(op_graph: nx.DiGraph) -> Callable[..., ScalarType]:
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
                results[node] = _numpy_ops[op_id](*op_inputs)
            else:
                results[node] = float(node) if "." in node else int(node)

        return results[sorted_nodes[-1]]

    return compute
