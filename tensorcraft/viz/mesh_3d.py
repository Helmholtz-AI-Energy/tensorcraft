"""3D mesh visualization module."""
import torch
import networkx as nx
from matplotlib.axes import Axes

from tensorcraft.viz.util import get_n_colors, mesh_grid, rgba2hex


def draw_3d_mesh(axes: Axes, mesh: torch.Size) -> None:
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
    if len(mesh) != 3:
        raise ValueError("Must provide a 3D mesh")

    graph = nx.grid_graph(dim=tuple(mesh[::-1]))
    colors = get_n_colors(mesh.numel())
    hexColors = [rgba2hex(color) for color in colors]
    pos = mesh_grid(mesh)

    node_xyz = torch.stack([pos[v] for v in graph.nodes()])
    edge_xyz = torch.stack([torch.stack([pos[u], pos[v]]) for u, v in graph.edges()])

    axes.scatter(*node_xyz.T, c=hexColors, s=100, alpha=1.0)
    for node, node_pos in zip(graph.nodes(), node_xyz):
        axes.text(*node_pos, f"{node}", fontsize=8, ha="right", va="bottom")
    for edge in edge_xyz:
        axes.plot(*edge.T, c="black", alpha=0.5, linewidth=0.5)

    axes.set_axis_off()
    axes.grid(False)
