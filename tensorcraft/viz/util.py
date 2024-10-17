"""Utility functions for visualization."""

import matplotlib as mpl
from matplotlib.axes import Axes
import torch

from tensorcraft.util import multi2linearIndex, linear2multiIndex


def get_n_colors(n: int, colormap: str = "viridis") -> torch.IntTensor:
    """
    Get an array of n colors from a colormap.

    Parameters
    ----------
    n : int
        The number of colors.
    colormap : str, optional
        The name of the colormap (default is "viridis").

    Returns
    -------
    ndarray
        An array of n colors.
    """
    return mpl.colormaps[colormap].resampled(n).colors


def rgba2hex(rgba: torch.Tensor) -> str:
    """
    Convert an RGBA color array to a hexadecimal color string.

    Parameters
    ----------
    rgba : ndarray
        The RGBA color array.

    Returns
    -------
    str
        The hexadecimal color string.
    """
    RGBA = rgba * 255
    RGBA = RGBA.dtype(torch.uint8)
    return "#{:02x}{:02x}{:02x}{:02x}".format(*RGBA)


def draw_2d_grid(ax: Axes, shape: tuple | torch.Tensor, color: str = "black") -> None:
    """
    Set the axis ticks and labels for a 2D tensor plot.

    Parameters
    ----------
    ax : Axes
        The axes object.
    shape : tuple or ndarray
        The shape of the tensor.
    color : str, optional
        The color of the axis ticks (default is "black").

    Returns
    -------
    None
    """
    # Ticks
    ax.set_xticks(torch.arange(0.0, shape[1], 1.0))
    ax.set_yticks(torch.arange(0.0, shape[0], 1.0))

    ax.set_xticks(torch.arange(-0.5, float(shape[1]) - 0.5, 1.0), minor=True)
    ax.set_yticks(torch.arange(-0.5, float(shape[0]) - 0.5, 1.0), minor=True)

    ax.grid(which="minor", color="black", linestyle="-", linewidth=0.5)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.tick_params(which="major", bottom=False, left=False)

    # Labels
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    ax.set_xlabel("Axis 1")
    ax.set_ylabel("Axis 0")


def draw_color_bar(fig, axs, colors: torch.Tensor, shrink=1.0, orientation="horizontal"):
    """
    Draw a color bar for the given colors.

    Parameters
    ----------
    fig : Figure
        The figure object.
    axs : Axes
        The axes object.
    colors : ndarray
        An array of colors.

    Returns
    -------
    None
    """
    location = "bottom" if orientation == "horizontal" else "right"
    cmap = mpl.colors.ListedColormap(colors)
    norm = mpl.colors.BoundaryNorm(torch.arange(-0.5, len(colors), 1), cmap.N)
    cbar = fig.colorbar(
        mpl.cm.ScalarMappable(cmap=cmap, norm=norm),
        ax=axs,
        orientation=orientation,
        shrink=shrink,
        ticks=torch.arange(0, len(colors), 1),
        location=location,
        panchor=(0.5, 0.5),
    )
    if orientation == "horizontal":
        cbar.ax.set_xticklabels(torch.arange(0, len(colors), 1))
    else:
        cbar.ax.set_yticklabels(torch.arange(0, len(colors), 1))
    cbar.set_label("Processor index")


def explode(data: torch.Tensor) -> torch.Tensor:
    """
    Explode a 3D array by inserting zeros between each element.

    Parameters
    ----------
    data : ndarray
        The 3D array to explode.

    Returns
    -------
    ndarray
        The exploded 3D array.
    """
    size = torch.Tensor(data.shape) * 2
    data_e = torch.Tensor(size - 1, dtype=data.dtype)
    data_e[::2, ::2, ::2] = data
    return data_e


def mesh_grid(mesh: torch.Size) -> dict[tuple[int], torch.tensor]:
    """
    Generate a mesh grid based on the given tensor.

    Parameters
    ----------
    mesh : Tensor
        The input tensor.

    Returns
    -------
    dict[tuple[int], torch.Tensor]
        A dictionary containing the positions of each element in the mesh grid.
    """
    positions: dict = {}
    for i in range(mesh.numel()):
        mindex = linear2multiIndex(i, mesh)
        pos = [
            float(dimSize - dim) / (dimSize - 1) for dim, dimSize in zip(mindex, mesh)
        ]
        if len(mesh) <= 1:
            pos += [0.5]
            pos[-1] = 1 - pos[-1]
            positions[mindex[0]] = torch.tensor(pos)[::-1]
        else:
            pos[-1] = 1 - pos[-1]
            positions[tuple(mindex)] = torch.tensor(pos)[::-1]

    return positions


def latex2figSize(
    width: float, fraction: float = 1, ratio=16 / 9
) -> tuple[float, float]:
    """Set figure dimensions to avoid scaling in LaTeX.

    Parameters
    ----------
    width: float
            Document textwidth or columnwidth in pts
    fraction: float, optional
            Fraction of the width which you wish the figure to occupy

    Returns
    -------
    fig_dim: tuple
            Dimensions of figure in inches
    """
    # Width of figure (in pts)
    fig_width_pt = width * fraction

    # Convert from pt to inches
    inches_per_pt = 1 / 72.27

    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt
    # Figure height in inches
    fig_height_in = fig_width_in / ratio

    fig_dim = (fig_width_in, fig_height_in)

    return fig_dim
