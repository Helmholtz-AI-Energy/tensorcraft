"""Memory Constrained Redistributor module."""

import logging
from typing import Any

import torch

from tensorcraft.distributions.multi_axis import MultiAxisDist
from tensorcraft.optim.cost import Cost

from .redistributor import Redistributor

log = logging.getLogger("tensorcraft")


class MemoryConstrainedRedist(Redistributor):
    """
    Redistributor that optimizes for memory usage per rank.

    Losely based on this: N. A. Rink, A. Paszke, D. Vytiniotis, and G. S. Schmid, “Memory-efficient array redistribution through portable collective communication,” Nov. 28, 2022, arXiv: arXiv:2112.01075. Accessed: Sep. 15, 2023. [Online]. Available: http://arxiv.org/abs/2112.01075

    """

    def _setup(self):
        return super()._setup()

    def redistribute(self, shape, start_dist, target_dist):  # noqa: D102
        if not self._compatible(shape, start_dist=start_dist, target_dist=target_dist):
            raise ValueError("Incompatible arguments.")

        match start_dist.__qualname__:
            case MultiAxisDist.__qualname__:
                return self._redistribute_multi_axis(shape, start_dist, target_dist)
            case _:
                raise NotImplementedError(
                    "Redistributor not implemented for the given distribution type."
                )

    def _redistribute_multi_axis(
        self, shape: torch.Size, start_dist: MultiAxisDist, target_dist: MultiAxisDist
    ):
        mesh = start_dist.processorMesh

        operations: list[tuple[str, tuple[Any], Cost]] = []
        total_cost = Cost()

        # 1) Identify tensor axes with discrepancies
        target_axes = [
            i
            for i, (x, y) in enumerate(
                zip(start_dist._dims_mapping, target_dist._dims_mapping)
            )
            if x != y
        ]
        log.info(f"Target axes: {target_axes}")

        # 2) Identify replication dims
        start_rep_dims = []
        target_rep_dims = []
        for dim in range(len(mesh)):
            in_start = False
            in_target = False
            for i in range(len(mesh)):
                if in_start and (dim in start_dist._dims_mapping[i]):  # type: ignore
                    in_start = True
                if in_target and (dim in target_dist._dims_mapping[i]):  # type: ignore
                    in_target = True

            if not in_start:
                start_rep_dims.append(dim)
            if not in_target:
                target_rep_dims.append(dim)

        log.info(
            f"Replicated dims: start - {start_rep_dims}, target - {target_rep_dims}"
        )

        # 3) Move mesh dims around

        # 4) Adjust block sizes

        return operations, total_cost
