"""Module for visualizing the program graph using networkx."""

import logging
import math
from typing import Any

import networkx as nx

from tensorcraft.compiler import Program, TensorExpression

from .util import get_n_colors, rgba2hex

log = logging.getLogger("tensorcraft")


def draw_program_graph(program: Program, color_by="loops") -> None:
    """Draw the program graph using networkx.

    Parameters
    ----------
    program : Program
        Object containing the program graph.
    """
    node_pos = _position_nodes(program.graph, program.input_variables)
    node_colors = _color_nodes(program, color_by=color_by)
    nx.draw(
        program.graph,
        node_pos,
        node_color=node_colors,
        with_labels=True,
        font_weight="bold",
    )


def draw_expression_graph(tensor_expression: TensorExpression) -> None:
    """Draw the expression graph using networkx.

    Parameters
    ----------
    tensor_expressoin : TensorExpression
        Object containing the tensor expression graph.
    """
    inputs = [f"{name}[{','.join(shape)}]" for name, shape in tensor_expression.inputs]
    node_pos = _position_nodes(tensor_expression.op_graph, inputs)
    neg_attributes = nx.get_node_attributes(tensor_expression.op_graph, "neg")
    node_labels = {
        node: f"-{node}"
        if node in neg_attributes and neg_attributes[node]
        else str(node)
        for node in tensor_expression.op_graph.nodes
    }
    nx.draw(
        tensor_expression.op_graph,
        node_pos,
        with_labels=True,
        font_weight="bold",
        labels=node_labels,
    )


def _position_nodes(
    G: nx.DiGraph, input_variables: list[str]
) -> dict[Any, list[float]]:
    """
    Position the nodes in the graph.

    Parameters
    ----------
    program : Program
        The program containing the tensor expressions and variable information.

    Returns
    -------
    dict[Any, tuple[float, float]]
        The positions of the nodes in the graph.

    """
    log.debug("Positioning nodes")
    positions = {}

    # Root nodes on level 0
    assigned_levels = {node: 0 for node in input_variables}
    assigned_nodes: dict[int, list[Any]] = {0: input_variables, 1: []}
    elemens_per_level: dict[int, int] = {0: len(input_variables), 1: 0}

    # Add successors of root nodes to open_nodes
    open_nodes = []
    for node in input_variables:
        open_nodes.extend([s for s in G.successors(node) if s not in open_nodes])

    # Find other nodes without predecessors, and add their successors to open_nodes
    for node in G.nodes:
        if node not in assigned_levels and len(list(G.predecessors(node))) == 0:
            assigned_levels[node] = 1
            assigned_nodes[1].append(node)
            elemens_per_level[1] += 1
            open_nodes.extend([s for s in G.successors(node) if s not in open_nodes])

    max_level = 1

    while open_nodes:
        node = open_nodes.pop(0)
        predecessor_levels = []
        missing_predecessors = False
        for predecessor in G.predecessors(node):
            if predecessor in assigned_levels:
                predecessor_levels.append(assigned_levels[predecessor])
            else:
                if predecessor not in open_nodes:
                    open_nodes.append(predecessor)
                missing_predecessors = True

        if not missing_predecessors:
            level = max(predecessor_levels) + 1
            if level not in elemens_per_level:
                elemens_per_level[level] = 1
            else:
                elemens_per_level[level] += 1

            if level not in assigned_nodes:
                assigned_nodes[level] = [node]
            else:
                assigned_nodes[level].append(node)

            max_level = max(level, max_level)
            assigned_levels[node] = level
            open_nodes.extend([s for s in G.successors(node) if s not in open_nodes])
        else:
            open_nodes.append(node)

    x_offset = {}
    for level, nodes in assigned_nodes.items():
        for node in sorted(nodes):
            if level not in x_offset:
                base_offset = -2.5 * elemens_per_level[level] * math.sqrt(level + 1)
                even_odd_offset = (
                    2.5
                    if len(elemens_per_level) % 2 == 1
                    else 2.5 * math.sqrt(level + 1)
                )
                even_odd_offset = (
                    -even_odd_offset if level % 2 == 0 else even_odd_offset
                )
                x_offset[level] = base_offset + even_odd_offset

                x = x_offset[level]
            else:
                x_offset[level] += 5 * math.sqrt(level + 1)
                x = x_offset[level]

            positions[node] = [x, -5 * level]

    return positions


def _color_nodes(program: Program, color_by: str = "type") -> list[str]:
    """Color the nodes in the graph.

    Parameters
    ----------
    program : Program
        The program containing the tensor expressions and variable information.
    color_by : str
        The attribute to use for coloring the nodes.

    Returns
    -------
    dict[Any, str]
        The colors of the nodes in the graph.
    """
    match color_by:
        case "type":
            return _color_nodes_by_type(program)
        case "loops":
            return _color_nodes_by_loops(program)
        case "opcount":
            return _color_nodes_by_opcount(program)
        case _:
            raise ValueError(f"Invalid value for 'color_by': {color_by}.")


def _color_nodes_by_loops(program: Program) -> list[str]:
    """Color the nodes in the graph by the number of loops they are in.

    Parameters
    ----------
    program : Program
        The program containing the tensor expressions and variable information.

    Returns
    -------
    dict[Any, str]
        The colors of the nodes in the graph.
    """
    colors = get_n_colors(program._max_loop_depth[0] + 1, colormap="plasma")

    color_list = []
    for node in program.graph:
        if node in program.input_variables:
            color_list.append("gray")
        else:
            color = colors[program.tensor_expressions[node].loop_count]
            color_list.append(rgba2hex(color))
    return color_list


def _color_nodes_by_opcount(program: Program) -> list[str]:
    """Color the nodes in the graph by the number of basic operations that take place on each loop.

    Parameters
    ----------
    program : Program
        The program containing the tensor expressions and variable information.

    Returns
    -------
    dict[Any, str]
        The colors of the nodes in the graph.
    """
    colors = get_n_colors(program._max_op_count[0] + 1, colormap="plasma")

    color_list = []
    for node in program.graph:
        if node in program.input_variables:
            color_list.append("gray")
        else:
            color = colors[program.tensor_expressions[node].op_count]
            color_list.append(rgba2hex(color))
    return color_list


def _color_nodes_by_type(program: Program) -> list[str]:
    """Color the nodes in the graph by type.

    Parameters
    ----------
    program : Program
        The program containing the tensor expressions and variable information.

    Returns
    -------
    dict[Any, str]
        The colors of the nodes in the
    """
    node_colors = []
    for node in program.graph.nodes:
        if node in program.input_variables:
            node_colors.append("green")
        else:
            node_colors.append("blue")

    return node_colors
