"""2D mesh visualization module."""

import networkx as nx
import torch
from matplotlib.axes import Axes

from tensorcraft.viz.util import get_n_colors, mesh_grid, rgba2hex


def draw_2d_mesh(axes: Axes, mesh: torch.Size) -> None:
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
    if len(mesh) <= 2:
        graph = nx.grid_graph(dim=tuple(mesh[::-1]))
    else:
        raise ValueError("Only 1D and 2D meshes are supported")

    colors = get_n_colors(mesh.numel())
    hexColors = [rgba2hex(color) for color in colors]
    pos = mesh_grid(mesh)
    offset = torch.tensor([0.15 / dim for dim in mesh])
    labelPos = {key: pos + offset for key, pos in pos.items()}

    nx.draw(graph, pos, node_color=hexColors, ax=axes)
    nx.draw_networkx_labels(
        graph,
        labelPos,
        font_size=8,
        bbox={"boxstyle": "circle", "x": 1.0, "y": 1.0, "alpha": 0.0},
        ax=axes,
    )
