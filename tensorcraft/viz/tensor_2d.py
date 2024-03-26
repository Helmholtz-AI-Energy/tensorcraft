"""2D tensor visualization functions."""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from tensorcraft.distributions import Dist
from tensorcraft.tensor import Tensor
from tensorcraft.viz.util import draw2DGrid, drawColorBar, getNColors


def draw2DTensor(
    fig: Figure, tensor: Tensor, distribution: Dist, cbar: bool = False
) -> None:
    """
    Plot a 2D tensor.

    Parameters
    ----------
    tensor : Tensor
        The 2D tensor to plot.
    distribution : PMeshDist
        The distribution of the tensor.
    cbar : bool, optional
        Whether to show the color bar (default is True).

    Returns
    -------
    None
    """
    if tensor.order > 2:
        raise ValueError(
            f"Only 2D tensors are supported, the provided tensor has {tensor.order}"
        )

    if len(distribution.processorArrangement) > 2:
        raise ValueError("Only 2D meshes are supported")

    axis = fig.add_subplot(111)

    processor_view = distribution.processorView(tensor)

    if tensor.order == 1:
        img_shape = tensor.shape.reshape(-1, 1)
    else:
        img_shape = tensor.shape

    colors = getNColors(distribution.numProcessors)
    img = np.zeros((*img_shape, 4))
    for i in range(tensor.size):
        m_idx = tensor.getMultiIndex(i)
        img[m_idx[0], m_idx[1], :] = colors[
            np.argmax(processor_view[m_idx[0], m_idx[1], :])
        ]

    axis.imshow(img[::-1], origin="lower", aspect="equal")

    # Ticks
    draw2DGrid(axis, img_shape)

    if cbar:
        drawColorBar(fig, axis, colors)


def draw2DProcessorView(
    fig: Figure, tensor: Tensor, distribution: Dist, cbar: bool = False
) -> None:
    """
    Plot the processor view of a 2D tensor.

    Parameters
    ----------
    tensor : Tensor
        The 2D tensor to plot.
    distribution : Dist
        The distribution of the tensor.
    cbar : bool, optional
        Whether to show the color bar (default is True).

    Returns
    -------
    None
    """
    if tensor.order > 2:
        raise ValueError(
            "Only 2D tensors are supported, please provide the dimensions to print"
        )

    if len(distribution.processorArrangement) > 2:
        raise ValueError("Only 2D meshes are supported")

    processor_view = distribution.processorView(tensor)

    if tensor.order == 1:
        img_shape = tensor.shape.reshape(-1, 1)
    else:
        img_shape = tensor.shape

    processorArragement = distribution.processorArrangement
    subplot_x = processorArragement[0]
    subplot_y = (
        processorArragement[1] if len(distribution.processorArrangement) > 1 else 1
    )

    colors = getNColors(distribution.numProcessors)
    gs = fig.add_gridspec(nrows=subplot_x, ncols=subplot_y)
    axs = gs.subplots(
        sharex=True,
        sharey=True,
    )

    for p in range(distribution.numProcessors):
        p_midx = distribution.getProcessorMultiIndex(p)

        img = np.apply_along_axis(
            lambda a: colors[p] if a[p] else [0.0, 0.0, 0.0, 0.0], -1, processor_view
        )
        axs[p_midx[0], p_midx[1]].imshow(img, origin="upper", aspect="equal")
        draw2DGrid(axs[p_midx[0], p_midx[1]], img_shape)

        axs[p_midx[0], p_midx[1]].title.set_text(f"P {p_midx}")

    ## Add discrete colorbar with the processor index and colors

    if cbar:
        drawColorBar(fig, axs, colors)

    plt.show()
