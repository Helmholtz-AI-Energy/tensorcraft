"""Memory Constrained Redistributor module."""

import dataclasses
import logging
from typing import Optional

import torch
from typing_extensions import Self

from tensorcraft.distributions.multi_axis import MultiAxisDist
from tensorcraft.optim.cost import Cost

from .redistributor import Redistributor

log = logging.getLogger("tensorcraft")


@dataclasses.dataclass
class Node:
    """Distribution graph nodes."""

    parent_node: Optional[Self]
    parent_node_op: Optional[str]
    dist: MultiAxisDist
    children: dict[str, Self]
    edge_cost: Cost
    cum_cost: Cost
    depth: int = 0

    def path_to_root(self) -> list[tuple[str, MultiAxisDist, Cost]]:
        path = []
        current_node = self
        while current_node:
            path.insert(
                0,
                (
                    current_node.parent_node_op,
                    current_node.dist,
                    current_node.edge_cost,
                ),
            )
            current_node = current_node.parent_node
        return path

    def __str__(self):
        return f"Node: {self.dist}, Cost: {self.edge_cost}, Depth: {self.depth}, parent_op: {self.parent_node_op}"


class MemoryConstrainedRedist(Redistributor):
    """
    Redistributor that optimizes for memory usage per rank.

    Losely based on this: N. A. Rink, A. Paszke, D. Vytiniotis, and G. S. Schmid, “Memory-efficient array redistribution through portable collective communication,” Nov. 28, 2022, arXiv: arXiv:2112.01075. Accessed: Sep. 15, 2023. [Online]. Available: http://arxiv.org/abs/2112.01075
    """

    def __init__(self, costModel, alpha=1, beta=1, gamma=1, epsilon=1, max_depth=5):
        super().__init__(costModel, alpha, beta, gamma, epsilon)
        self._epsilon = 0.0
        self._max_depth = 5

    def _redistribute_multi_axis(
        self, shape: torch.Size, start_dist: MultiAxisDist, target_dist: MultiAxisDist
    ):
        log.info(start_dist)
        log.info(target_dist)

        open_nodes: list[Node] = []
        close_nodes: list[Node] = []
        end_nodes: list[tuple[Node, float]] = []

        nodes_dict: dict[str, tuple[Node, float]] = {}

        starter_node = Node(
            None, None, start_dist, {}, Cost(0, 0, 0, 0), Cost(0, 0, 0, 0)
        )
        open_nodes.append(starter_node)

        nodes_dict[str(start_dist)] = (starter_node, 0)

        base_memory = start_dist.maxNumElements(shape)
        memory_limit = max(base_memory, target_dist.maxNumElements(shape))

        current_depth = 0
        preferred_block_sizes = list(
            set(filter(lambda x: x > 0, target_dist._block_sizes))
        )

        while len(open_nodes) > 0 and current_depth < self._max_depth:
            log.debug(f"Open nodes: {len(open_nodes)}")
            log.debug(f"Close nodes: {len(close_nodes)}")
            log.debug(f"End nodes: {len(end_nodes)}")
            current_node = open_nodes.pop(0)
            current_depth = current_node.depth

            log.debug(f"Current node: {current_node.dist}")
            log.debug(f"Current depth: {current_depth}")

            close_nodes.append(current_node)
            current_memory_usage = current_node.dist.maxNumElements(shape)

            neighbours = current_node.dist.neighbours(shape, preferred_block_sizes)
            log.debug(f"Neighbours: {neighbours}")
            for id, n_dist, vol, n_procs in neighbours:
                # 1) Check if the memory usage is within the limit
                n_memory_usage = n_dist.maxNumElements(shape)
                if n_memory_usage > memory_limit:
                    continue

                mem_delta = n_memory_usage - current_memory_usage

                # 2) Calculate the edge cost

                match id.split("_")[0]:
                    case "split":
                        mem_delta = n_dist.maxNumElements(shape) - current_memory_usage
                        edge_cost = Cost(0, 0, 0, mem_delta)
                    case "alltoall":
                        edge_cost = self._cm.all2all(n_procs=n_procs, n_elements=vol)
                    case "allgather":
                        edge_cost = self._cm.allgather(n_procs=n_procs, n_elements=vol)
                    case "permute":
                        edge_cost = self._cm.permute(n_procs=n_procs, n_elements=vol)
                    case _:
                        log.warning(f"Unknown operation: {id}. Ignoring.")
                        continue

                cum_cost = edge_cost + current_node.cum_cost
                path_cost = self._edge_weight(cum_cost)

                # 3) Create the new node
                n_node = Node(
                    current_node,
                    id,
                    n_dist,
                    {},
                    edge_cost,
                    cum_cost,
                    current_node.depth + 1,
                )
                current_node.children[id] = n_node

                # 4) Check if the node is the target distribution
                if n_node.dist == target_dist:
                    end_nodes.append((n_node, path_cost))
                    continue

                # 5) Check if the node is already in the open list, if so update the path cost
                n_dist_str = str(n_dist)
                if n_dist_str in nodes_dict:
                    node, alternate_path_cost = nodes_dict[n_dist_str]

                    if alternate_path_cost > path_cost or (
                        alternate_path_cost == path_cost and n_node.depth < node.depth
                    ):
                        nodes_dict[n_dist_str] = (n_node, path_cost)

                    continue
                else:
                    nodes_dict[n_dist_str] = (n_node, path_cost)
                    open_nodes.append(n_node)

        # Get the best path
        best_path = []
        best_cost = float("inf")
        for node, path_cost in end_nodes:
            if path_cost < best_cost:
                best_cost = path_cost
                best_path = node.path_to_root()

        return best_path, best_cost
