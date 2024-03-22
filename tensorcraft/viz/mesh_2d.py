from tensorcraft.tensor import Tensor
from tensorcraft.viz import getNColors, rgba2hex
import networkx as nx
import matplotlib.pyplot as plt

def draw2DMesh(mesh: Tensor) -> None:
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
    if mesh.order == 1:
        graph = nx.grid_2d_graph(mesh.size, 1)
    elif mesh.order == 2:
        graph = nx.grid_2d_graph(*mesh.shape)
    else:
        raise ValueError("Only 1D and 2D meshes are supported")

    colors = getNColors(mesh.size)
    hexColors = [rgba2hex(color) for color in colors]
    pos = nx.spring_layout(graph)
    nx.draw(graph, pos, node_color=hexColors)
    plt.show()
