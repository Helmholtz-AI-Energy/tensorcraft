"""3D mesh visualization module."""

import networkx as nx
import numpy as np
from matplotlib.axes import Axes

from tensorcraft.tensor import Tensor
from tensorcraft.viz.util import getNColors, meshGrid, rgba2hex


def draw3DMesh(axes: Axes, mesh: Tensor) -> None:
    """
    Draw a 3D mesh using matplotlib.

    Parameters
    ----------
    mesh : Tensor
        The 3D mesh to be drawn.

    Raises
    ------
    ValueError
        If the provided mesh is not a 3D mesh.

    Returns
    -------
    None
    """
    if mesh.order != 3:
        raise ValueError("Must provide a 3D mesh")

    graph = nx.grid_graph(dim=tuple(mesh.shape[::-1]))
    colors = getNColors(mesh.size)
    hexColors = [rgba2hex(color) for color in colors]
    pos = meshGrid(mesh)

    node_xyz = np.array([pos[v] for v in graph.nodes()])
    edge_xyz = np.array([[pos[u], pos[v]] for u, v in graph.edges()])

    axes.scatter(*node_xyz.T, c=hexColors, s=100, alpha=1.0)
    for node, node_pos in zip(graph.nodes(), node_xyz):
        axes.text(*node_pos, f"{node}", fontsize=8, ha="right", va="bottom")
    for edge in edge_xyz:
        axes.plot(*edge.T, c="black", alpha=0.5, linewidth=0.5)

    axes.set_axis_off()
    axes.grid(False)
