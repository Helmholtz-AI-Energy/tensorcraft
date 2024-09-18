"""Tensor class."""

from typing import Optional

import numpy as np
import numpy.typing as npt

from tensorcraft.distributions.dist import Dist
from tensorcraft.shape import Shape
from tensorcraft.types import Index, IndexTuple


class Tensor:
    """A class representing a tensor with a given shape and distribution. This is purely an abstract representation and does not store any data."""

    def __init__(
        self,
        dims: npt.ArrayLike | Shape,
        dist: Optional[Dist] = None,
        num_workers: int = 0,
    ) -> None:
        if isinstance(dims, Shape):
            self._shape = dims
        else:
            self._shape = Shape(dims)

        self._dist = dist
        if not dist:
            self._n_procs = num_workers
        else:
            self._n_procs = dist.numProcessors

        self._processor_view: Optional[np.ndarray] = None

    @property
    def shape(self) -> Shape:
        """Get the shape of the tensor."""
        return self._shape

    @property
    def numWorkers(self) -> int:
        """Get the number of workers."""
        return self._n_procs

    def getMultiIndex(self, index: Index) -> IndexTuple:
        """Transform linear index to multi-index."""
        return self._shape.getMultiIndex(index)

    def getLinearIndex(self, index: IndexTuple) -> Index:
        """Transform multi-index to linear index."""
        return self._shape.getLinearIndex(index)

    @property
    def size(self) -> int:
        """Get the size of the tensor."""
        return self._shape.size

    @property
    def order(self) -> int:
        """Get the order of the tensor."""
        return self._shape.order

    @property
    def dist(self) -> Optional[Dist]:
        """Get the distribution of the tensor."""
        return self._dist

    @dist.setter
    def dist(self, new_dist: Optional[Dist]) -> None:
        """Set the distribution of the tensor."""
        if new_dist and not new_dist.compatible(self.shape):
            raise ValueError("Distribution is not compatible with tensor")
        self._processor_view = None
        self._dist = new_dist

    def processorView(self) -> np.ndarray:
        """Get the processor view of the tensor."""
        if self._processor_view is None:
            if not self.dist:
                self._processor_view = np.ones((*self.shape, self._n_procs), dtype=bool)
            else:
                self._processor_view = self.dist.processorView(self._shape)
        return self._processor_view

    def getIndexLocation(self, index: IndexTuple) -> np.ndarray:
        """Get the processors that hold a specific element of a tensor."""
        processor_view = self.processorView()
        if isinstance(index, int):
            index = self.getMultiIndex(index)
        return processor_view[index]

    def __repr__(self) -> str:
        """Get the string representation of the tensor."""
        return f"Tensor(shape: {self.shape}, dist: {self.dist}, nworkers: {self.numWorkers})"
