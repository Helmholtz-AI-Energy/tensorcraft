"""Base Redistributor module."""

import abc
from typing import Any

from tensorcraft.optim.cost import Cost, CostModel


class Redistributor(abc.ABC):
    """Base redistributer class."""

    def __init__(self, costModel: CostModel):
        """
        Redistributer initialization.

        Params
        ------
        costModel: CostModel
            The cost model to be used for optimizing redistributions.
        """
        self._cm = costModel
        self._setup()

    @abc.abstractmethod
    def _setup(self):
        """Redistributor setup."""
        raise NotImplementedError()

    @abc.abstractmethod
    def redistribute(
        self, shape, start_dist, target_dist
    ) -> tuple[list[tuple[str, tuple[Any], Cost]], Cost]:
        """
        Given a tensor shape, a starting distribution, and a target distribution, it will return a sequence of collective operations to reach the target distribution.

        Parameters
        ----------
        shape : tuple of int
            The shape of the tensor to be redistributed.
        start_dist : Any
            The starting distribution of the tensor.
        target_dist : Any
            The target distribution of the tensor.

        Returns
        -------
        list of (str, tuple of Any)
            A list of collective operations required to transform the tensor from the starting distribution to the target distribution.
        """
        raise NotImplementedError()
