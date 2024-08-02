"""Module for visualizing the program graph using networkx."""

from typing import Any

import networkx as nx

from tensorcraft.compiler import Program


def draw_program_graph(program: Program):
    """Draw the program graph using networkx.

    Parameters
    ----------
    program : Program
        Object containing the program graph.
    """
    node_pos = _position_nodes(program)
    node_colors = _color_nodes(program, by="type")
    nx.draw(
        program.graph,
        node_pos,
        node_color=node_colors.values(),
        with_labels=True,
        font_weight="bold",
    )


def _position_nodes(program: Program) -> dict[Any, list[float]]:
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
    print("Positioning nodes")
    positions = {}
    G = program.graph

    # Root nodes on level 0
    assigned_levels = {node: 0 for node in program.input_variables}
    assigned_nodes: dict[int, list[Any]] = {0: program.input_variables, 1: []}
    elemens_per_level: dict[int, int] = {0: len(program.input_variables), 1: 0}

    # Add successors of root nodes to open_nodes
    open_nodes = []
    for node in program.input_variables:
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
                x_offset[level] = (
                    -2.5 * elemens_per_level[level] + 0 if level % 2 == 0 else -2.5
                )
                x = x_offset[level]
            else:
                x_offset[level] += 5
                x = x_offset[level]

            positions[node] = [x, -5 * level]

    return positions


def _color_nodes(program: Program, by: str = "type") -> dict[Any, str]:
    """Color the nodes in the graph.

    Parameters
    ----------
    program : Program
        The program containing the tensor expressions and variable information.
    by : str
        The attribute to use for coloring the nodes.

    Returns
    -------
    dict[Any, str]
        The colors of the nodes in the graph.
    """
    match by:
        case "type":
            return _color_nodes_by_type(program)
        case _:
            raise ValueError(f"Invalid value for 'by': {by}.")


def _color_nodes_by_type(program: Program) -> dict[Any, str]:
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
    node_colors = {}
    for node in program.graph.nodes:
        if node in program.input_variables:
            node_colors[node] = "green"
        else:
            node_colors[node] = "blue"

    return node_colors
