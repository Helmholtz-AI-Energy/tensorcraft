"""TensorCraft is a collection of tools for working and visualizing distributed multi-dimensional tensors."""

__version__ = "0.0.0"

import torch

torch.autograd.set_grad_enabled(False)

# Lower level modules
import tensorcraft.compiler as compiler
import tensorcraft.distributions as dist
import tensorcraft.mpi as mpi

# High level modules
import tensorcraft.optim as optim
import tensorcraft.util as util
import tensorcraft.viz as viz

from .logging import set_logger_config

_compiler = compiler.Compiler()
compile = _compiler.compile


__all__ = [
    "util",
    "mpi",
    "dist",
    "compiler",
    "compile",
    "viz",
    "optim",
    "set_logger_config",
]
