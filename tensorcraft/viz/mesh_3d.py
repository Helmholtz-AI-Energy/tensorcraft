"""3D mesh visualization module."""

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from tensorcraft.tensor import Tensor
from tensorcraft.viz import getNColors, meshGrid, rgba2hex


def draw3DMesh(mesh: Tensor) -> None:
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

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    ax.scatter(*node_xyz.T, c=hexColors)
    for edge in edge_xyz:
        ax.plot(*edge.T, c="black")

    fig.tight_layout()
    plt.show()
