"""Visualization of tensor operations."""

from typing import Optional

import drawsvg as draw
import numpy as np

from tensorcraft.compiler.model import TensorExpression
from tensorcraft.tensor import Tensor
from tensorcraft.types import Index, IndexTuple


def _highlight_index(axis: int, index: Index, highlight: Optional[IndexTuple]) -> bool:
    if highlight is None:
        return index == 0
    else:
        return highlight[axis] == index or highlight[axis] is None


def _orthogonal_projection(theta: float, gamma: float, phi: float) -> np.ndarray:
    # Rotate the 3D point (x, y, z) by theta, gamma and phi angles
    # around the x, y and z axis respectively
    # The rotation matrix is given by:
    # R = Rz * Ry * Rx

    # Rotation around the x axis
    Rx = np.array(
        [
            [1, 0, 0],
            [0, np.cos(theta), -np.sin(theta)],
            [0, np.sin(theta), np.cos(theta)],
        ]
    )

    # Rotation around the y axis
    Ry = np.array(
        [
            [np.cos(gamma), 0, np.sin(gamma)],
            [0, 1, 0],
            [-np.sin(gamma), 0, np.cos(gamma)],
        ]
    )

    # Rotation around the z axis
    Rz = np.array(
        [[np.cos(phi), -np.sin(phi), 0], [np.sin(phi), np.cos(phi), 0], [0, 0, 1]]
    )

    R = Rz @ Ry @ Rx
    return R


def draw_tensor(
    d: draw.Drawing | draw.Group,
    tensor: Tensor,
    cell_size: float = 1,
    highlight_color: str = "blue",
    stroke_color: str = "black",
    mindex_highlight: Optional[IndexTuple] = None,
    stroke_width: float = 0.1,
):
    """Draw a tensor as a grid of cells.

    Parameters
    ----------
    d : draw.Drawing
        The drawing object to append the tensor to.
    tensor : Tensor
        The tensor to draw.
    cell_size : float, optional
        The size of each cell, by default 1.
    highlight_color : str, optional
        The color to highlight the cells, by default "blue".
    stroke_color : str, optional
        The color of the stroke, by default "black".
    mindex_highlight : Optional[IndexTuple], optional
        The index to highlight, by default None.
    stroke_width : float, optional
        The width of the stroke, by default 0.1.

    Raises
    ------
    ValueError
        If the tensor order is greater than 3.
    ValueError
        If the highlight index has a different order than the tensor.
    """
    if tensor.order > 3:
        raise ValueError("Cannot draw tensors with order greater than 3")

    if mindex_highlight is not None and len(mindex_highlight) != tensor.order:
        raise ValueError("Highlight index must have the same order as the tensor")

    # If 0 order, draw a simple square and highlight with
    if tensor.order == 0:
        start_x, start_y = -cell_size / 2, -cell_size / 2
        d.append(
            draw.Rectangle(
                start_x,
                start_y,
                cell_size,
                cell_size,
                fill=highlight_color,
                stroke=stroke_color,
            )
        )

    if tensor.order == 1:
        # Rectangle
        cells = tensor.size
        start_x, start_y = -cell_size / 2, -cell_size * cells / 2
        for i in range(tensor.size):
            fill = (
                highlight_color if _highlight_index(0, i, mindex_highlight) else "white"
            )
            d.append(
                draw.Rectangle(
                    start_x,
                    start_y + i * cell_size,
                    cell_size,
                    cell_size,
                    fill=fill,
                    stroke=stroke_color,
                    stroke_width=stroke_width,
                )
            )

    if tensor.order == 2:
        rows, cols = tensor.shape
        start_x, start_y = -cell_size * cols / 2, -cell_size * rows / 2
        for i in range(rows):
            fill_row = _highlight_index(0, i, mindex_highlight)
            for j in range(cols):
                fill = (
                    highlight_color
                    if fill_row & _highlight_index(1, j, mindex_highlight)
                    else "white"
                )
                d.append(
                    draw.Rectangle(
                        start_x + j * cell_size,
                        start_y + i * cell_size,
                        cell_size,
                        cell_size,
                        fill=fill,
                        stroke=stroke_color,
                        stroke_width=stroke_width,
                    )
                )

    if tensor.order == 3:
        # Draw a 3D tensor as a stack of 2D tensors
        rows, cols, depth = tensor.shape
        start_x, start_y, start_z = (
            -cell_size * cols / 2,
            -cell_size * rows / 2,
            -cell_size * depth / 2,
        )
        theta = -10 * np.pi / 180
        gamma = -10 * np.pi / 180
        phi = -0 * np.pi / 180
        R = _orthogonal_projection(theta, gamma, phi)
        for k in reversed(range(depth)):
            fill_depth = _highlight_index(2, k, mindex_highlight)
            opacity = 1.0 - k / depth
            for i in range(rows):
                fill_row = _highlight_index(0, i, mindex_highlight)
                for j in range(cols):
                    fill_col = _highlight_index(1, j, mindex_highlight)
                    fill = (
                        highlight_color
                        if fill_row & fill_col & fill_depth
                        else "#00000000"
                    )
                    coord_matrix = np.array(
                        [
                            [
                                start_x + j * cell_size,
                                start_y + i * cell_size,
                                start_z + k * cell_size,
                            ],
                            [
                                start_x + (j + 1) * cell_size,
                                start_y + i * cell_size,
                                start_z + k * cell_size,
                            ],
                            [
                                start_x + (j + 1) * cell_size,
                                start_y + (i + 1) * cell_size,
                                start_z + k * cell_size,
                            ],
                            [
                                start_x + j * cell_size,
                                start_y + (i + 1) * cell_size,
                                start_z + k * cell_size,
                            ],
                        ]
                    )
                    coord_matrix = coord_matrix @ R
                    d.append(
                        draw.Lines(
                            coord_matrix[0][0],
                            coord_matrix[0][1],
                            coord_matrix[1][0],
                            coord_matrix[1][1],
                            coord_matrix[2][0],
                            coord_matrix[2][1],
                            coord_matrix[3][0],
                            coord_matrix[3][1],
                            close=True,
                            fill=fill,
                            stroke=stroke_color,
                            stroke_width=stroke_width,
                            opacity=opacity,
                        )
                    )


def draw_op(
    d: draw.Drawing,
    op: TensorExpression,
    tensor_shapes: dict[str, Tensor],
    mindex_highlight: Optional[IndexTuple],
):
    """Draw a tensor operation as a graph.

    Parameters
    ----------
    d : draw.Drawing
        The drawing object to append the operation to.
    op : TensorExpression
        The tensor operation to draw.
    tensor_shapes : dict[str, Tensor]
        A dictionary of tensor shapes.
    mindex_highlight : Optional[IndexTuple]
        The multi-index to highlight

    Raises
    ------
    ValueError
        If the output tensor has an order greater than 3.
    ValueError
        If the highlight index has a different order than the output tensor.
    """
    canvas_w = d.width
    # canvas_h = d.height

    cell_size = 3
    padding_marging = 10
    font_size_ops = 8
    font_size_labels = 4

    # Draw the output tensor
    index_variables = op.index_variables
    print(f"Index variables: {index_variables}")
    output_shape = tensor_shapes[op.output[0]]
    output_index_variables = op.output[1]
    print(f"Output shape: {output_shape}")
    print(f"Output index variables: {output_index_variables}")

    if mindex_highlight is None:
        mindex_highlight = tuple([0 for _ in output_shape.shape])

    # Draw the output tensor
    output_tensor_width = (
        output_shape.shape[1] * cell_size if output_shape.order > 1 else cell_size
    )
    current_x = padding_marging - canvas_w / 2
    print(f"Current x: {current_x}")
    next_x = current_x + output_tensor_width
    group = draw.Group(transform=f"translate({current_x + output_tensor_width/2}, 0)")
    draw_tensor(group, output_shape, cell_size, mindex_highlight=mindex_highlight)
    d.append(group)

    # Label at the top
    label_h = -output_shape.shape[0] * cell_size / 2 - padding_marging
    d.append(
        draw.Text(
            f"{op.output[0]} {output_shape.shape}",
            font_size_labels,
            current_x + output_tensor_width / 2,
            label_h,
            center=True,
        )
    )

    # Draw the equal sign
    current_x = next_x + padding_marging
    print(f"Current x: {current_x}")
    next_x = current_x + font_size_ops / 2
    sign = op.assignment_type.value
    d.append(draw.Text(sign, 8, current_x, 0, center=True))

    # Draw the input tensors
    for i, (tensor_name, tensor_idx_exp) in enumerate(op.inputs):
        tensor_shape = tensor_shapes[tensor_name]
        tensor_width = (
            tensor_shape.shape[1] * cell_size if tensor_shape.order > 1 else cell_size
        )
        current_x = next_x + padding_marging
        group = draw.Group(transform=f"translate({current_x + tensor_width/2}, 0)")
        t_mindx_h = tuple(
            [
                mindex_highlight[output_index_variables.index(idx_var)]
                if idx_var in output_index_variables
                else None
                for idx_var in tensor_idx_exp
            ]
        )
        print(f"Tensor {tensor_name} highlight: {t_mindx_h}")
        draw_tensor(group, tensor_shape, cell_size, mindex_highlight=t_mindx_h)  # type: ignore
        d.append(group)

        # Label at the top
        label_h = -tensor_shape.shape[0] * cell_size / 2 - padding_marging
        d.append(
            draw.Text(
                f"{tensor_name} {tensor_shape.shape}",
                font_size_labels,
                current_x + tensor_width / 2,
                label_h,
                center=True,
            )
        )

        next_x = current_x + tensor_width + padding_marging
