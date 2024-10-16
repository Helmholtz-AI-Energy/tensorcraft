"""TensorCraft is a collection of tools for working and visualizing distributed multi-dimensional tensors."""

__version__ = "0.0.0"

# Classes to be used as tc.<class> / tc.<func>()
# Modules to be used as tc.<module>.<function>
# import tensorcraft.distributions as dist
# from tensorcraft import compiler, viz
from tensorcraft.util import linear2multiIndex, multi2linearIndex, order2npOrder

# _compiler = compiler.Compiler()
# compile = _compiler.compile


__all__ = [
    "dist",
    "compiler",
    "viz",
    "Shape",
    "Tensor",
    "multi2linearIndex",
    "linear2multiIndex",
    "order2npOrder",
]
