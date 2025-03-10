"""2D tensor visualization functions."""

import math

import torch
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from tensorcraft.distributions import Dist
from tensorcraft.util.axis_utils import linear2multiIndex, multi2linearIndex
from tensorcraft.viz.util import draw_2d_grid, draw_color_bar, get_n_colors


def draw_2d_tensor(
    axes: Axes, shape: torch.Size, dist: Dist, cbar: bool = False
) -> None:
    """
    Plot a 2D tensor.

    Parameters
    ----------
    tensor : Tensor
        The 2D tensor to plot.
    distribution : MultiAxisDist
        The distribution of the tensor.
    cbar : bool, optional
        Whether to show the color bar (default is True).

    Returns
    -------
    None
    """
    if len(shape) > 2:
        raise ValueError(
            f"Only 2D tensors are supported, the provided tensor has {len(shape)} dimensions"
        )

    # if len(tensor.dist.processorArrangement) > 2:
    #     raise ValueError("Only 2D meshes are supported")

    processor_view = dist.processorView(shape)

    if len(shape) == 1:
        img_shape = torch.Size([shape[0], 1])
    elif len(shape) == 0:
        img_shape = torch.Size([1, 1])
    else:
        img_shape = shape

    colors = get_n_colors(dist.numProcessors)
    img = torch.zeros((*img_shape, 4))
    for i in range(shape.numel()):
        m_idx = linear2multiIndex(i, img_shape)
        img[m_idx[0], m_idx[1], :] = colors[
            processor_view[m_idx[0], m_idx[1], :].nonzero()
        ]

    axes.imshow(img.flip(0), origin="lower", aspect="equal")

    # Ticks
    draw_2d_grid(axes, img_shape)

    if cbar:
        draw_color_bar(axes.get_figure(), axes, colors)


def draw_2d_processor_view(
    axes: Axes, shape: torch.Size, dist: Dist, processor: int, cbar: bool = False
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
    if len(shape) > 2:
        raise ValueError(
            "Only 2D tensors are supported, please provide the dimensions to print"
        )

    if dist is None:
        raise ValueError("The tensor is not distributed")

    processor_view = dist.processorView(shape)

    if len(shape) == 1:
        img_shape = torch.tensor(shape).reshape(-1, 1)
    else:
        img_shape = torch.tensor(shape)

    colors = get_n_colors(dist.numProcessors)

    p_midx = dist.getProcessorMultiIndex(processor)

    img = torch.stack(
        [
            colors[processor] if a[processor] else torch.zeros(4, dtype=torch.float32)
            for a in processor_view.reshape(-1, dist.numProcessors).unbind(0)
        ]
    ).reshape(*shape, 4)
    axes.imshow(img, origin="upper", aspect="equal")
    draw_2d_grid(axes, img_shape)

    axes.title.set_text(f"P {[int(x) for x in p_midx]}")

    ## Add discrete colorbar with the processor index and colors

    if cbar:
        draw_color_bar(axes.get_figure(), axes, colors)


def draw_processor_grid(
    fig: Figure, tensor_shape: torch.Size, dist: Dist, cbar: bool = False
):
    """
    Plot the processor grid of a 2D tensor.

    Parameters
    ----------
    fig : Figure
        The figure to plot the tensor on.
    tensor_shape : torch.Size
        The shape of the tensor.
    distribution : Dist
        The distribution of the tensor.
    cbar : bool, optional
        Whether to show the color bar (default is True).
    """
    mesh = dist.processorMesh

    subplot_x = mesh[0]
    subplot_y = math.prod(mesh[1:]) if len(mesh) > 1 else 1

    gs = fig.add_gridspec(nrows=subplot_x, ncols=subplot_y)
    axs = gs.subplots(
        sharex=True,
        sharey=True,
    )

    for p in range(dist.numProcessors):
        p_midx = dist.getProcessorMultiIndex(p)

        y_idx = multi2linearIndex(mesh[1:], p_midx[1:]) if len(mesh) > 1 else 0
        if subplot_y == 1:
            draw_2d_processor_view(axs[p_midx[0]], tensor_shape, dist, p)
        else:
            draw_2d_processor_view(axs[p_midx[0], y_idx], tensor_shape, dist, p)

    if cbar:
        draw_color_bar(fig, axs, get_n_colors(dist.numProcessors), shrink=0.5)
