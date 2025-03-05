"""Memory Constrained Redistributor module."""

import dataclasses
import logging
from typing import Any, Optional

import torch
from typing_extensions import Self

from tensorcraft.distributions import Dist
from tensorcraft.distributions.multi_axis import MultiAxisDist
from tensorcraft.optim.cost import Cost

from .redistributor import Redistributor

log = logging.getLogger("tensorcraft")


@dataclasses.dataclass
class Node:
    """Distribution graph nodes."""

    parent_node: Optional[Self]
    dist: MultiAxisDist
    children: dict[str, Self]
    cost: Cost
    depth: int = 0


class MemoryConstrainedRedist(Redistributor):
    """
    Redistributor that optimizes for memory usage per rank.

    Losely based on this: N. A. Rink, A. Paszke, D. Vytiniotis, and G. S. Schmid, “Memory-efficient array redistribution through portable collective communication,” Nov. 28, 2022, arXiv: arXiv:2112.01075. Accessed: Sep. 15, 2023. [Online]. Available: http://arxiv.org/abs/2112.01075

    """

    def _setup(self):
        self._epsilon = 0.0
        self._max_depth = 10
        return super()._setup()

    def redistribute(self, shape, start_dist, target_dist):  # noqa: D102
        if not self._compatible(shape, start_dist=start_dist, target_dist=target_dist):
            raise ValueError("Incompatible arguments.")

        match start_dist:
            case MultiAxisDist():
                return self._redistribute_multi_axis(shape, start_dist, target_dist)
            case _:
                raise NotImplementedError(
                    "Redistributor not implemented for the given distribution type."
                )

    def _redistribute_multi_axis(
        self, shape: torch.Size, start_dist: MultiAxisDist, target_dist: MultiAxisDist
    ):
        mesh = start_dist.processorMesh
        log.info(start_dist)
        log.info(target_dist)

        operations: list[tuple[str, tuple[Any], Cost]] = []
        total_cost = Cost()

        # 1) Identify tensor axes with discrepancies, and the relevant dims
        target_axes = [
            i
            for i, (x, y) in enumerate(
                zip(start_dist._dims_mapping, target_dist._dims_mapping)
            )
            if x != y
        ]
        log.info(f"Target axes: {target_axes}")

        # 2) Identify replication dims
        start_rep_dims = set()
        target_rep_dims = set()
        for dim in range(len(mesh)):
            in_start = False
            in_target = False
            for i in range(len(shape)):
                if not in_start and (dim in start_dist._dims_mapping[i]):  # type: ignore
                    in_start = True
                if not in_target and (dim in target_dist._dims_mapping[i]):  # type: ignore
                    in_target = True

            if not in_start:
                start_rep_dims.add(dim)
            if not in_target:
                target_rep_dims.add(dim)

        log.info(
            f"Replicated dims: start - {start_rep_dims}, target - {target_rep_dims}"
        )
        helper_dims = start_rep_dims & target_rep_dims
        log.info(f"Helper dims: {helper_dims}")

        # 3) Move mesh dims around
        open_nodes: list[Node] = []
        close_nodes: list[Node] = []
        end_nodes: list[Node] = []

        nodes_dict: dict[Dist, tuple[Node, float]] = {}

        starter_node = Node(None, start_dist, {}, Cost(0, 0, 0, 0))
        open_nodes.append(starter_node)

        nodes_dict[start_dist] = starter_node

        base_memory = start_dist.maxNumElements()
        memory_limit = max(base_memory, target_dist.maxNumElements())

        while len(open_nodes) > 0:
            current_node = open_nodes.pop(0)
            close_nodes.append(current_node)
            current_memory_usage = current_node.dist.maxNumElements(shape)
            
            neighbours = current_node.dist.neighbours(shape)
            for id, n_dist, vol, n_procs in neighbours:
            
                mem_delta = n_dist.maxNumElements(shape) - current_memory_usage

                match id.split("_")[0]:
                    case "split":
                        mem_delta = n_dist.maxNumElements(shape) - current_memory_usage
                        edge_cost = Cost(0, 0, 0, mem_delta)
                    case "alltoall":
                        edge_cost = self._cm.all2all(n_procs=n_procs, vol=vol)
                    case "allgather":
                        edge_cost = self._cm.allgather(n_procs=n_procs, vol=vol)
                    case "permute":
                        edge_cost = self._cm.permute(n_procs=n_procs, vol=vol)
                    case _:
                        log.warning(f"Unknown operation: {id}. Ignoring.")
                        continue
                
                n_node = Node(current_node, n_dist, {}, edge_cost, current_node.depth + 1)
                path_cost = self._edge_weight(edge_cost + current_node.cost)
                if n_dist in nodes_dict:
                    node, alternate_path_cost = nodes_dict[n_dist]
                    
                    if  alternate_path_cost > path_cost:
                        nodes_dict[n_dist] = (n_node, path_cost)
                
                else:
                    nodes_dict[n_dist] = (n_node, path_cost)
                        






        return operations, total_cost
