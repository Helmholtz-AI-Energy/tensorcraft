"""2D tensor visualization functions."""

import numpy as np
from matplotlib.axes import Axes

from tensorcraft.distributions import Dist
from tensorcraft.tensor import Tensor
from tensorcraft.viz.util import draw2DGrid, drawColorBar, getNColors


def draw2DTensor(
    axes: Axes, tensor: Tensor, distribution: Dist, cbar: bool = False
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

    processor_view = distribution.processorView(tensor)

    if tensor.order == 1:
        img_shape = np.array(tensor.shape).reshape(-1, 1)
    else:
        img_shape = np.array(tensor.shape)

    colors = getNColors(distribution.numProcessors)
    img = np.zeros((*img_shape, 4))
    for i in range(tensor.size):
        m_idx = tensor.getMultiIndex(i)
        img[m_idx[0], m_idx[1], :] = colors[
            np.argmax(processor_view[m_idx[0], m_idx[1], :])
        ]

    axes.imshow(img[::-1], origin="lower", aspect="equal")

    # Ticks
    draw2DGrid(axes, img_shape)

    if cbar:
        drawColorBar(axes.get_figure(), axes, colors)


def draw2DProcessorView(
    axes: Axes, tensor: Tensor, distribution: Dist, processor: int, cbar: bool = False
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
        img_shape = np.array(tensor.shape).reshape(-1, 1)
    else:
        img_shape = np.array(tensor.shape)

    colors = getNColors(distribution.numProcessors)

    p_midx = distribution.getProcessorMultiIndex(processor)

    img = np.apply_along_axis(
        lambda a: colors[processor] if a[processor] else [0.0, 0.0, 0.0, 0.0],
        -1,
        processor_view,
    )
    axes.imshow(img, origin="upper", aspect="equal")
    draw2DGrid(axes, img_shape)

    axes.title.set_text(f"P {[int(x) for x in p_midx]}")

    ## Add discrete colorbar with the processor index and colors

    if cbar:
        drawColorBar(axes.get_figure(), axes, colors)
