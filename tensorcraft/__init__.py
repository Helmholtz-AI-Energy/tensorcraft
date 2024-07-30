"""TensorCraft is a collection of tools for working and visualizing distributed multi-dimensional tensors."""

__version__ = "0.0.0"

# Import everything and put it into __all__
from tensorcraft import compiler, logging, viz
from tensorcraft.tensor import Tensor
from tensorcraft.types import MIndex
from tensorcraft.util import multi2linearIndex, order2npOrder

logging.init_logging("WARNING")

_compiler = compiler.Compiler()
compile = _compiler.compile


__all__ = ["compiler", "viz", "MIndex", "Tensor", "multi2linearIndex", "order2npOrder"]
