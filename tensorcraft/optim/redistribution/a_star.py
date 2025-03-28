"""Memory Constrained Redistributor module."""

import logging

import torch

from tensorcraft.distributions import Dist, MultiAxisDist
from tensorcraft.optim import Cost
from tensorcraft.util.route_finder import RouteNode, find_routes

from .redistributor import Redistributor

log = logging.getLogger("tensorcraft")


class AStarRedistributor(Redistributor):
    """Redistributor that tries to reach the target as fast as possible using the A* algorithm."""

    def __init__(self, costModel, alpha=1, beta=1, gamma=1, epsilon=0, **kwargs):
        super().__init__(costModel, alpha, beta, gamma, epsilon)
        self._epsilon = epsilon
        self._kwargs = kwargs

    def _redistribute_slab(self, shape, start_dist, target_dist):
        raise NotImplementedError("Slab redistribution not implemented")

    def _redistribute_tile(self, shape, start_dist, target_dist):
        raise NotImplementedError("Tile redistribution not implemented")

    def _redistribute_multi_axis(
        self, shape: torch.Size, start_dist: MultiAxisDist, target_dist: MultiAxisDist
    ) -> tuple[list[tuple[str, Dist, float]], float]:
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

        paths = find_routes(
            RouteNode(None, "", start_dist, {}, 0),
            RouteNode(None, "", target_dist, {}, 0),
            neighbours,
            **self._kwargs,
        )

        best_path = min(paths, key=lambda x: x[1])
        return best_path
