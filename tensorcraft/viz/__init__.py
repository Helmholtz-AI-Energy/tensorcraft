"""Visualization tools for tensorcraft."""

from tensorcraft.viz.mesh_2d import draw_2d_mesh
from tensorcraft.viz.mesh_3d import draw_3d_mesh
from tensorcraft.viz.ops import draw_op, draw_tensor
from tensorcraft.viz.program_graph import draw_expression_graph, draw_program_graph
from tensorcraft.viz.tensor_2d import draw_2d_processor_view, draw_2d_tensor
from tensorcraft.viz.tensor_3d import draw_3d_tensor
from tensorcraft.viz.util import (
    draw_2d_grid,
    draw_color_bar,
    explode,
    get_n_colors,
    latex2figSize,
    mesh_grid,
    rgba2hex,
)

__all__ = [
    "draw_2d_grid",
    "draw_color_bar",
    "explode",
    "get_n_colors",
    "latex2figSize",
    "mesh_grid",
    "rgba2hex",
    "draw_2d_mesh",
    "draw_3d_mesh",
    "draw_2d_processor_view",
    "draw_2d_tensor",
    "draw_3d_tensor",
    "draw_program_graph",
    "draw_expression_graph",
    "draw_tensor",
    "draw_op",
]
