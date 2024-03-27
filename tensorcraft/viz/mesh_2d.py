"""2D mesh visualization module."""

import networkx as nx
import numpy as np
from matplotlib.axes import Axes

from tensorcraft.tensor import Tensor
from tensorcraft.viz.util import getNColors, meshGrid, rgba2hex


def draw2DMesh(axes: Axes, mesh: Tensor) -> None:
    """
    Plot a 2D mesh.

    Parameters
    ----------
    mesh : Tensor
        The 2D mesh to plot.

    Returns
    -------
    None
    """
    if mesh.order <= 2:
        graph = nx.grid_graph(dim=tuple(mesh.shape[::-1]))
    else:
        raise ValueError("Only 1D and 2D meshes are supported")

    colors = getNColors(mesh.size)
    hexColors = [rgba2hex(color) for color in colors]
    pos = meshGrid(mesh)
    offset = np.array([0.15 / dim for dim in mesh.shape])
    labelPos = {key: pos + offset for key, pos in pos.items()}

    nx.draw(graph, pos, node_color=hexColors, ax=axes)
    nx.draw_networkx_labels(
        graph,
        labelPos,
        font_size=8,
        bbox={"boxstyle": "circle", "x": 1.0, "y": 1.0, "alpha": 0.0},
        ax=axes,
    )
