"""Distribution module for TensorCraft."""

from tensorcraft.distributions.dist import Dist
from tensorcraft.distributions.multi_axis import MultiAxisDist
from tensorcraft.distributions.slab import SlabDist
from tensorcraft.distributions.tile import TileDist

__all__ = ["Dist", "SlabDist", "TileDist", "MultiAxisDist"]
