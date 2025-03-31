"""Memory Constrained Redistributor module."""

import logging
from typing import Any

import jellyfish
import torch

from tensorcraft.distributions import MultiAxisDist
from tensorcraft.distributions.slab import SlabDist
from tensorcraft.distributions.tile import TileDist
from tensorcraft.optim import Cost
from tensorcraft.optim.cost.cost_model import CostModel
from tensorcraft.util.route_finder import RouteNode, find_routes

from .redistributor import OperationSchedule, Redistributor

log = logging.getLogger("tensorcraft")


class AStarRedistributor(Redistributor):
    """Redistributor that tries to reach the target as fast as possible using the A* algorithm."""

    def __init__(
        self,
        costModel: CostModel,
        alpha: float = 1.0,  # Operation Latency Cost
        beta: float = 1.0,  # Communication volume cost / per element
        gamma: float = 1.0,  # Computation cost / per element
        epsilon: float = 0.0,  # Memory Cost
        **kwargs: Any,
    ):
        super().__init__(costModel, alpha, beta, gamma, epsilon)
        self._kwargs = kwargs
        self._path_cost_w = kwargs["path_cost_w"] if "path_cost_w" in kwargs else 1.0
        self._estimate_w = kwargs["estimate_w"] if "estimete_w" in kwargs else 1.0

    def _redistribute_slab(
        self, shape: torch.Size, start_dist: SlabDist, target_dist: SlabDist
    ) -> tuple[OperationSchedule, float]:
        raise NotImplementedError("Slab redistribution not implemented")

    def _redistribute_tile(
        self, shape: torch.Size, start_dist: TileDist, target_dist: TileDist
    ) -> tuple[OperationSchedule, float]:
        raise NotImplementedError("Tile redistribution not implemented")

    def _redistribute_multi_axis(
        self, shape: torch.Size, start_dist: MultiAxisDist, target_dist: MultiAxisDist
    ) -> tuple[OperationSchedule, float]:
        log.info(start_dist)
        log.info(target_dist)

        preferred_block_sizes = list(
            set(filter(lambda x: x > 0, target_dist.blockSizes))
        )

        def neighbours(node: RouteNode[MultiAxisDist]) -> list[tuple[str, RouteNode]]:
            current_memory_usage = node.obj.maxNumElements(shape)

            neighbours_list: list[tuple[str, MultiAxisDist, int, int]] = (
                node.obj.neighbours(shape, preferred_block_sizes)
            )
            real_neighbours: list[tuple[str, RouteNode[MultiAxisDist]]] = []
            for op_id, n_dist, vol, n_procs in neighbours_list:
                n_memory_usage = n_dist.maxNumElements(shape)

                mem_delta = n_memory_usage - current_memory_usage
                log.debug(
                    f"Op_id: {op_id}, Neighbour: {n_dist}, Memory delta: {mem_delta}, Volume: {vol}, n_procs: {n_procs}"
                )
                match op_id.split("_")[0]:
                    case "split":
                        mem_delta = n_dist.maxNumElements(shape) - current_memory_usage
                        edge_cost = Cost(0, 0, 0, mem_delta)
                    case "alltoall":
                        edge_cost = self._cm.alltoall(n_procs=n_procs, n_elements=vol)
                    case "allgather":
                        edge_cost = self._cm.allgather(n_procs=n_procs, n_elements=vol)
                    case "permute":
                        edge_cost = self._cm.permute(n_procs=n_procs, n_elements=vol)
                    case "changeBlockSize":
                        edge_cost = self._cm.alltoall(n_procs=n_procs, n_elements=vol)
                    case _:
                        log.warning(f"Unknown operation: {op_id}. Ignoring.")
                        continue
                log.debug(f"Op_id: {op_id}, Neighbour: {n_dist}, Cost: {edge_cost}")

                n_node = RouteNode(
                    node,
                    op_id,
                    n_dist,
                    {},
                    self._edge_weight(edge_cost),
                )
                real_neighbours.append((str(n_dist), n_node))
                node.children[str(n_dist)] = n_node

            return real_neighbours

        def priority_func(node: RouteNode[MultiAxisDist]) -> float:
            elements = shape.numel()
            estimate_dist: int = 0
            for i in range(len(shape)):
                str1 = ",".join(node.obj._dims_mapping[i])
                str2 = ",".join(target_dist._dims_mapping[i])
                estimate_dist += jellyfish.damerau_levenshtein_distance(str1, str2)

            str1 = ",".join(node.obj._block_sizes)
            str2 = ",".join(target_dist._block_sizes)
            estimate_dist = jellyfish.damerau_levenshtein_distance(str1, str2)

            _, cost = node.path_depth_cost
            f1 = self._path_cost_w * cost
            f2 = self._estimate_w * estimate_dist * elements
            log.debug(
                f"Path cost: {self._path_cost_w} * {cost} ={f1}, Estimate cost:{self._estimate_w} * {estimate_dist} * {elements} = {f2}"
            )
            value = f1 + f2
            return value

        paths = find_routes(
            RouteNode(None, "", start_dist, {}, 0),
            RouteNode(None, "", target_dist, {}, 0),
            neighbours,
            priority_func**self._kwargs,
        )

        best_path = min(paths, key=lambda x: x[1])
        return best_path
