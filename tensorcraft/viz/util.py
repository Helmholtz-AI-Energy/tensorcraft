"""Utility functions for visualization."""

import matplotlib as mpl
import numpy as np

from tensorcraft.tensor import Tensor


def getNColors(n: int | np.int64, colormap: str = "viridis") -> np.ndarray:
    """
    Get an array of n colors from a colormap.

    Parameters
    ----------
    n : int or np.int64
        The number of colors.
    colormap : str, optional
        The name of the colormap (default is "viridis").

    Returns
    -------
    ndarray
        An array of n colors.
    """
    return mpl.colormaps[colormap].resampled(n).colors


def rgba2hex(rgba: np.ndarray) -> str:
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
    RGBA = RGBA.astype(np.uint8)
    return "#{:02x}{:02x}{:02x}{:02x}".format(*RGBA)


def draw2DGrid(ax, shape: tuple | np.ndarray, color: str = "black") -> None:
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
    ax.set_xticks(np.arange(0.0, shape[1], 1.0))
    ax.set_yticks(np.arange(0.0, shape[0], 1.0))

    ax.set_xticks(np.arange(-0.5, float(shape[1]) - 0.5, 1.0), minor=True)
    ax.set_yticks(np.arange(-0.5, float(shape[0]) - 0.5, 1.0), minor=True)

    ax.grid(which="minor", color="black", linestyle="-", linewidth=0.5)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.tick_params(which="major", bottom=False, left=False)

    # Labels
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    ax.set_xlabel("Axis 1")
    ax.set_ylabel("Axis 0")


def drawColorBar(fig, axs, colors: np.ndarray, shrink=1.0, orientation="horizontal"):
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
    norm = mpl.colors.BoundaryNorm(np.arange(-0.5, len(colors), 1), cmap.N)
    cbar = fig.colorbar(
        mpl.cm.ScalarMappable(cmap=cmap, norm=norm),
        ax=axs,
        orientation=orientation,
        shrink=shrink,
        ticks=np.arange(0, len(colors), 1),
        location=location,
        panchor=(0.5, 0.5),
    )
    if orientation == "horizontal":
        cbar.ax.set_xticklabels(np.arange(0, len(colors), 1))
    else:
        cbar.ax.set_yticklabels(np.arange(0, len(colors), 1))
    cbar.set_label("Processor index")


def explode(data: np.ndarray) -> np.ndarray:
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
    size = np.array(data.shape) * 2
    data_e = np.zeros(size - 1, dtype=data.dtype)
    data_e[::2, ::2, ::2] = data
    return data_e


def meshGrid(mesh: Tensor) -> dict[tuple[int], np.ndarray]:
    """
    Generate a mesh grid based on the given tensor.

    Parameters
    ----------
    mesh : Tensor
        The input tensor.

    Returns
    -------
    dict[tuple[int], np.ndarray]
        A dictionary containing the positions of each element in the mesh grid.
    """
    positions: dict = {}
    for i in range(mesh.size):
        mindex = mesh.getMultiIndex(i)
        pos = [
            float(dimSize - dim) / (dimSize - 1)
            for dim, dimSize in zip(mindex, mesh.shape)
        ]
        if mesh.order == 1:
            pos += [0.5]
            pos[-1] = 1 - pos[-1]
            positions[mindex[0]] = np.array(pos)[::-1]
        else:
            pos[-1] = 1 - pos[-1]
            positions[tuple(mindex)] = np.array(pos)[::-1]

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
