"""Utility function for redistributions."""

import abc
import dataclasses

from typing_extensions import Self


@dataclasses.dataclass
class Cost:
    """Data class that collectes the different costs."""

    latency: float
    bandwidth: float
    computation: float
    max_memory_delta: float

    def __add__(self, other: Self):
        self.latency += other.latency
        self.bandwidth += other.bandwidth
        self.computation += other.computation
        self.max_memory_delta = max(self.max_memory_delta, other.max_memory_delta)


class CostModel(abc.ABC):
    """
    CostModel class.

    Given an operation and the necessary context for it, calculates the expected latency, bandwidth, computation and memory costs.
    """

    @staticmethod
    def allgather(n_procs: int, n_elements: int) -> Cost:
        """Cost of doing an allgather on n_procs processors with n_elements each."""
        raise NotImplementedError()

    @staticmethod
    def permute(n_procs: int, n_elements: int) -> Cost:
        """Cost of doing a permute on n_procs processors with n_elements each."""
        raise NotImplementedError()

    @staticmethod
    def allreduce(n_procs: int, n_elements: int) -> Cost:
        """Cost of doing an allreduce on n_procs processors with n_elements each."""
        raise NotImplementedError()

    @staticmethod
    def reduce_scatter(n_procs: int, n_elements: int) -> Cost:
        """Cost of doing an reduce_scatter on n_procs processors with n_elements each."""
        raise NotImplementedError()

    @staticmethod
    def all2all(n_procs: int, n_elements: int) -> Cost:
        """Cost of doing an all2all on n_procs processors with n_elements each."""
        raise NotImplementedError()
