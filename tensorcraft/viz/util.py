import matplotlib as mpl
import numpy as np

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

def drawColorBar(fig, axs, colors: np.ndarray):
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
    cmap = mpl.colors.ListedColormap(colors)
    norm = mpl.colors.BoundaryNorm(np.arange(-0.5, len(colors), 1), cmap.N)
    cbar = fig.colorbar(
        mpl.cm.ScalarMappable(cmap=cmap, norm=norm),
        ax=axs,
        orientation="horizontal",
        ticks=np.arange(0, len(colors), 1),
        location="bottom",
    )
    cbar.ax.set_xticklabels(np.arange(0, len(colors), 1))
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
