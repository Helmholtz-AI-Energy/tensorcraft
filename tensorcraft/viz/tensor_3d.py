"""3D tensor visualization."""

import logging

import torch
from mpl_toolkits.mplot3d.axes3d import Axes3D

from tensorcraft.distributions import Dist
from tensorcraft.viz.util import draw_color_bar, explode, get_n_colors

log = logging.getLogger("tensorcraft")


def draw_3d_tensor(
    axes: Axes3D, shape: torch.Size, dist: Dist, cbar: bool = False
) -> None:
    """
    Plot a 3D tensor.

    Parameters
    ----------
    tensor : Tensor
        The 3D tensor to plot.
    distribution : Dist
        The distribution of the tensor.
    cbar : bool, optional
        Whether to show the color bar (default is True).

    Returns
    -------
    None
    """
    if len(shape) != 3:
        raise ValueError("Only 3D tensors are supported")

    processorView = dist.processorView(shape)
    colors = get_n_colors(dist.numProcessors)
    colors_edges = [color * 0.8 for color in colors]

    # build up the numpy logo
    filled = torch.ones(shape, dtype=torch.float)
    facecolors_list = []
    edgecolors_list = []
    for a in processorView.reshape(-1, dist.numProcessors).unbind(0):
        facecolors_list.append(colors[a.nonzero()])
        edgecolors_list.append(colors_edges[a.nonzero()])

    facecolors = torch.stack(facecolors_list).reshape(shape + (4,))
    edgecolors = torch.stack(edgecolors_list).reshape(shape + (4,))

    # upscale the above voxel image, leaving gaps
    filled_2 = explode(filled)
    fcolors_2 = explode(facecolors)
    ecolors_2 = explode(edgecolors)

    log.debug(filled_2.shape, fcolors_2.shape, ecolors_2.shape)
    # Shrink the gaps

    # x, y, z = torch.meshgrid(torch.arange(shape[0]), torch.arange(shape[1]), torch.arange(shape[2]))  # type: ignore
    # x, y, z = np.indices(np.array(filled_2.shape) + 1).astype(float) // 2  # type: ignore
    x, y, z = torch.meshgrid(
        torch.arange(filled_2.shape[0] + 1),
        torch.arange(filled_2.shape[1] + 1),
        torch.arange(filled_2.shape[2] + 1),
    )
    x = x.contiguous().float()
    y = y.contiguous().float()
    z = z.contiguous().float()

    x[0::2, :, :] += 0.1
    y[:, 0::2, :] += 0.1
    z[:, :, 0::2] += 0.1
    x[1::2, :, :] += 0.9
    y[:, 1::2, :] += 0.9
    z[:, :, 1::2] += 0.9

    axes.voxels(
        x,
        y,
        z,
        filled_2.numpy(),
        facecolors=fcolors_2.numpy(),
        edgecolors=ecolors_2.numpy(),
    )

    axes.view_init(25, -135, 0)
    axes.set_proj_type("persp")
    axes.set_xticklabels(labels=[])
    axes.set_yticklabels(labels=[])
    axes.set_zticklabels(labels=[])  # type: ignore[operator]
    axes.set_xlabel("Axis 0", labelpad=-15)
    axes.set_ylabel("Axis 1", labelpad=-15)
    axes.set_zlabel("Axis 2", labelpad=-15)
    axes.set_aspect("equal")
    axes.grid(False)

    if cbar:
        draw_color_bar(axes.get_figure(), axes, colors)
