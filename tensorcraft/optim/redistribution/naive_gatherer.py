"""NaiveGatherer Redistributor module."""

import logging

import torch

from tensorcraft.distributions import Dist, MultiAxisDist
from tensorcraft.distributions.slab import SlabDist
from tensorcraft.distributions.tile import TileDist
from tensorcraft.optim.cost import Cost

from .redistributor import OperationSchedule, Redistributor

log = logging.getLogger("tensorcraft")


class NaiveGathererRedist(Redistributor):
    """
    Naive Gatherer Redistributor class.

    The strategy is to simply apply an AllGather, and then apply no-communication splits, until the desired distribution is reached.
    """

    def _setup(self) -> None:  # noqa: D102
        return

    def _redistribute_multi_axis(
        self, shape: torch.Size, start_dist: MultiAxisDist, target_dist: MultiAxisDist
    ) -> tuple[OperationSchedule, float]:
        operations: list[tuple[str, Dist, float]] = [("", start_dist, 0)]
        total_cost = Cost()

        # First allgather
        starting_memory_usage = start_dist.maxNumElements(shape)
        non_split_dist, comm_volume, n_procs = start_dist.allgather(shape)
        log.info(f"Dist {non_split_dist}, volume: {comm_volume}, n_procs {n_procs}")
        non_split_memory_usage = non_split_dist.maxNumElements(shape)
        all_gather_cost = self._cm.allgather(n_procs, comm_volume)
        all_gather_cost.max_memory_delta = (
            non_split_memory_usage - starting_memory_usage
        )

        total_cost += all_gather_cost
        operations.append(
            ("allgather_*", non_split_dist, self._edge_weight(all_gather_cost))
        )

        # Then split
        post_split_memory = target_dist.maxNumElements(shape)
        split_cost = Cost(0, 0, 0, post_split_memory - non_split_memory_usage)
        operations.append(("split_*", target_dist, 0))

        total_cost += split_cost

        return operations, self._edge_weight(total_cost)

    def _redistribute_slab(
        self, shape: torch.Size, start_dist: SlabDist, target_dist: SlabDist
    ) -> tuple[OperationSchedule, float]:
        raise NotImplementedError(
            "Slab redistribution is not supported by the NaiveGathererRedist."
        )

    def _redistribute_tile(
        self, shape: torch.Size, start_dist: TileDist, target_dist: TileDist
    ) -> tuple[OperationSchedule, float]:
        raise NotImplementedError(
            "Tile redistribution is not supported by the NaiveGathererRedist."
        )
