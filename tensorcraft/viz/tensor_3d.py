import numpy as np
import matplotlib.pyplot as plt

from tensorcraft.tensor import Tensor
from tensorcraft.distributions import Dist
from tensorcraft.viz import getNColors, rgba2hex, explode, drawColorBar

def draw3DTensor(tensor: Tensor, distribution: Dist, cbar: bool = True) -> None:
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
    if tensor.order != 3:
        raise ValueError("Only 3D tensors are supported")

    processorView = distribution.processorView(tensor)
    colors = getNColors(distribution.numProcessors)
    colors_hex = [rgba2hex(color) for color in colors]
    colors_edges = [rgba2hex(color * 0.8) for color in colors]

    x, y, z = np.indices(tuple(tensor.shape))

    # build up the numpy logo
    filled = np.ones(tensor.shape)
    facecolors = np.apply_along_axis(
        lambda a: colors_hex[np.argmax(a)], -1, processorView
    )
    edgecolors = np.apply_along_axis(
        lambda a: colors_edges[np.argmax(a)], -1, processorView
    )

    # upscale the above voxel image, leaving gaps
    filled_2 = explode(filled)
    fcolors_2 = explode(facecolors)
    ecolors_2 = explode(edgecolors)

    # Shrink the gaps

    x, y, z = np.indices(np.array(filled_2.shape) + 1).astype(float) // 2  # type: ignore
    x[0::2, :, :] += 0.1
    y[:, 0::2, :] += 0.1
    z[:, :, 0::2] += 0.1
    x[1::2, :, :] += 0.9
    y[:, 1::2, :] += 0.9
    z[:, :, 1::2] += 0.9

    ax = plt.figure().add_subplot(projection="3d")
    ax.voxels(x, y, z, filled_2, facecolors=fcolors_2, edgecolors=ecolors_2)
    ax.view_init(25, -135, 0)
    ax.set_proj_type("persp")
    ax.set_xlabel("Axis 0")
    ax.set_ylabel("Axis 1")
    ax.set_zlabel("Axis 2")
    ax.set_aspect("equal")
    ax.grid(False)

    if cbar:
        drawColorBar(plt.gcf(), ax, colors)

    plt.show()
