"""IdealLowerBoundsCM module."""

import math

from .cost_model import Cost, CostModel


class IdealLowerBoundsCM(CostModel):
    """
    Lower bounds calculation cost model.

    Based on E. Chan, M. Heimlich, A. Purkayastha, and R. van de Geijn, “Collective communication: theory, practice, and experience,” Concurrency Computat.: Pract. Exper., vol. 19, no. 13, pp. 1749–1783, Sep. 2007, doi: 10.1002/cpe.1206.
    """

    @staticmethod
    def allgather(n_procs: int, n_elements: int) -> Cost:  # noqa: D102
        return Cost(
            math.ceil(math.log2(n_procs)),
            (n_procs - 1) * n_elements * n_procs / n_procs,
            0,
            0,
        )

    @staticmethod
    def permute(n_procs: int, n_elements: int) -> Cost:  # noqa: D102
        return Cost(1, n_elements * n_procs, 0, 0)

    @staticmethod
    def allreduce(n_procs: int, n_elements: int):  # noqa: D102
        return Cost(
            math.ceil(math.log2(n_procs)),
            2 * (n_procs - 1) * n_elements * n_procs / n_procs,
            (n_procs - 1) * n_elements * n_procs / n_procs,
            0,
        )

    @staticmethod
    def reduce_scatter(n_procs: int, n_elements: int) -> Cost:  # noqa: D102
        return Cost(
            math.ceil(math.log2(n_procs)),
            (n_procs - 1) * n_elements * n_procs / n_procs,
            (n_procs - 1) * n_elements / n_procs,
            0,
        )

    @staticmethod
    def alltoall(n_procs: int, n_elements: int) -> Cost:  # noqa: D102
        return Cost(
            math.ceil(math.log2(n_procs)),
            (n_procs - 1) * n_elements * n_procs / n_procs,
            0,
            0,
        )
