"""NaiveGatherer Redistributor module."""

from typing import Any

from tensorcraft.optim.cost import Cost

from .redistributor import Redistributor


class NaiveGathererRedist(Redistributor):
    """
    Naive Gatherer Redistributor class.

    The strategy is to simply apply an AllGather, and then apply no-communication splits, until the desired distribution is reached.
    """

    def _setup(self):  # noqa: D102
        return

    def redistribute(self, shape, start_dist, target_dist):  # noqa: D102
        if not self._compatible(shape, start_dist, target_dist):
            raise ValueError("Incompatible arguments.")

        # This requirenment could be relax for this Redistributor
        if start_dist.processorMesh != target_dist.processorMesh:
            raise ValueError(
                "Starting and target distributions needs to have the be on the same processor mesh."
            )

        operations: list[tuple[str, tuple[Any], Cost]] = []
        total_cost = Cost()

        # First allgather
        starting_memory_usage = start_dist.maxNumElements(shape)
        non_split_dist, comm_volume, n_procs = start_dist.allGather(shape)
        non_split_memory_usage = non_split_dist.maxNumElements(shape)
        all_gather_cost = self._cm.allgather(n_procs, comm_volume)
        all_gather_cost.max_memory_delta = (
            non_split_memory_usage - starting_memory_usage
        )

        total_cost += all_gather_cost

        # Then split
        operations.append(("allgather", (shape,), all_gather_cost))
        post_split_memory = target_dist.maxNumElements(shape)
        split_cost = Cost(0, 0, 0, post_split_memory - non_split_memory_usage)
        operations.append(("split", (shape,), split_cost))

        total_cost += split_cost

        return operations, total_cost
