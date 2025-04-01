"""Base Redistributor module."""

import abc
import logging

import torch

from tensorcraft.distributions import Dist, MultiAxisDist, SlabDist, TileDist
from tensorcraft.optim.cost import Cost, CostModel

log = logging.getLogger("tensorcraft")

OperationSchedule = list[tuple[str, Dist, float]]


class Redistributor(abc.ABC):
    """Base redistributer class."""

    def __init__(
        self,
        costModel: CostModel,
        alpha: float = 1.0,
        beta: float = 1.0,
        gamma: float = 1.0,
        epsilon: float = 0.0,
    ):
        """
        Redistributer initialization.

        Params
        ------
        costModel: CostModel
            The cost model to be used for optimizing redistributions.
        """
        self._cm = costModel
        self._alpha = alpha
        self._beta = beta
        self._gamma = gamma
        self._epsilon = epsilon
        self._setup()

    def _setup(self) -> None:
        """Redistributor setup."""
        return

    def _edge_weight(self, cost: Cost) -> float:
        return (
            cost.latency * self._alpha
            + cost.bandwidth * self._beta
            + self._gamma * cost.computation
            + self._epsilon * cost.max_memory_delta
        )

    def _compatible(
        self, shape: torch.Size, start_dist: Dist, target_dist: Dist
    ) -> bool:
        if not start_dist.compatible(shape):
            log.error("Shape is not compatible with starting distributions.")
            return False
        if not target_dist.compatible(shape):
            log.error("Shape is not compatible with target distributions.")
            return False

        if start_dist.processorMesh != target_dist.processorMesh:
            log.error(
                "Both source and target distributions should use the same processor mesh"
            )
            return False

        if type(start_dist) is not type(target_dist):
            log.error("Both source and target distributions should be of the same type")
            return False

        return True

    def redistribute(
        self, shape: torch.Size, start_dist: Dist, target_dist: Dist
    ) -> tuple[OperationSchedule, float]:
        """
        Given a tensor shape, a starting distribution, and a target distribution, it will return a sequence of collective operations to reach the target distribution.

        Parameters
        ----------
        shape : tuple of int
            The shape of the tensor to be redistributed.
        start_dist : Dist
            The starting distribution of the tensor.
        target_dist : Dist
            The target distribution of the tensor.

        Returns
        -------
        tuple[list[tuple[str, Dist, Cost]], Cost]
            A list of collective operations required to transform the tensor from the starting distribution to the target distribution. And the total cost of the redistribution.
        """
        if not self._compatible(shape, start_dist, target_dist):
            raise ValueError("Incompatible arguments.")

        if isinstance(start_dist, MultiAxisDist):
            return self._redistribute_multi_axis(shape, start_dist, target_dist)  # type: ignore
        elif isinstance(start_dist, TileDist):
            return self._redistribute_tile(shape, start_dist, target_dist)  # type: ignore
        elif isinstance(start_dist, SlabDist):
            return self._redistribute_slab(shape, start_dist, target_dist)  # type: ignore
        else:
            raise ValueError("Unsupported distribution type.")

    @abc.abstractmethod
    def _redistribute_multi_axis(
        self, shape: torch.Size, start_dist: MultiAxisDist, target_dist: MultiAxisDist
    ) -> tuple[OperationSchedule, float]:
        raise NotImplementedError

    @abc.abstractmethod
    def _redistribute_tile(
        self, shape: torch.Size, start_dist: TileDist, target_dist: TileDist
    ) -> tuple[OperationSchedule, float]:
        raise NotImplementedError

    @abc.abstractmethod
    def _redistribute_slab(
        self, shape: torch.Size, start_dist: SlabDist, target_dist: SlabDist
    ) -> tuple[OperationSchedule, float]:
        raise NotImplementedError
