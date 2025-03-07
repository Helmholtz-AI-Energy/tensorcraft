"""TensorCraft is a collection of tools for working and visualizing distributed multi-dimensional tensors."""

__version__ = "0.0.0"

from tensorcraft.logging import init_logging

init_logging("INFO")

import torch

torch.autograd.set_grad_enabled(False)

# Stack
import tensorcraft.compiler as compiler

# Lower level modules
import tensorcraft.distributions as dist

# High level modules
import tensorcraft.optim as optim
import tensorcraft.util as util
import tensorcraft.viz as viz

_compiler = compiler.Compiler()
compile = _compiler.compile


__all__ = [
    "util",
    "dist",
    "compiler",
    "viz",
    "optim",
]
