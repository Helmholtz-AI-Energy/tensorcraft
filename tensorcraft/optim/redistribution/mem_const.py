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
                         
        alpha = 0.1 # Latency weight
        beta = 1    # Bandwidth weight
        gamma = 1   # Memory weight

        starter_node = Node(None, start_dist, {}, Cost(0, 0, 0, 0))
        open_nodes.append(starter_node)
        
        nodes_dict[start_dist] = starter_node

        base_memory = start_dist.maxNumElements()
        memory_limit = max(base_memory, target_dist.maxNumElements())

        while len(open_nodes) > 0:
            current_node = open_nodes.pop(0)
            close_nodes.append(current_node)
            current_memory_usage = current_node.dist.maxNumElements(shape)
            
            # Free dims:
            free_dims = []
            for dim in range(len(mesh)):
                is_free = True
                for i in range(len(shape)):
                    if (dim in start_dist._dims_mapping[i]):  # type: ignore
                        is_free = False
                        break

                if is_free:
                    free_dims.append(dim)
            
            # 1) Further split (use target block sizes)
            for free_dim in free_dims:
                if free_dim in helper_dims:
                    for axis in range(len(shape)):
                        if axis not in target_axes:
                            operation = f"split_{axis}_{free_dim}_1"
                            try:
                                new_dist, _, _ = current_node.dist.split(shape, axis, free_dim, 1)
                                log.debug(f"Landed in {new_dist}")
                                
                                memory_usage = new_dist.maxNumElements(memory_usage)
                                
                                op_cost = current_node.cost + Cost(0, 0, 0, memory_usage - current_memory_usage)
                                
                                result_node = Node(
                                    current_node,
                                    new_dist,
                                    {},
                                    op_cost
                                )
                                current_node.children[operation, result_node]
                                
                                if new_dist == target_dist:
                                    end_nodes.append(result_node)
                                else:
                                    open_nodes.append(result_node)

                            except:
                                log.debug(f"Failed operation {operation} on dis {current_node.dist} with shape {shape}")
                            
            for target_axis in target_axes:
                if current_node.dist._dims_mapping[target_axis] != target_dist._dims_mapping[target_axis]:
                    c_axis_set = set(current_node.dist._dims_mapping[target_axis])
                    t_axis_set = set(target_dist._dims_mapping[target_axis]) 
                    extra_dims = c_axis_set - t_axis_set
                    missing_dims = t_axis_set - c_axis_set
                    

                    if len(extra_dims) == 0 and len(missing_dims) == 0:
                        # Needs permutation
                        log.debug("Requires permutation")
                    else:
                        # Apply splits
                        for m_dim in missing_dims:
                            if m_dim in free_dim:
                                target_block_size = target_dist._block_sizes[target_axis]
                                operation = f"split_{target_axis}_{m_dim}_{target_block_size}"
                                try:
                                    new_dist, _, _ = current_node.dist.split(shape, target_axis, free_dim, target_block_size)
                                    log.debug(f"Landed in {new_dist}")
                                    
                                    memory_usage = new_dist.maxNumElements(memory_usage)
                                    
                                    op_cost = current_node.cost + Cost(0, 0, 0, memory_usage - current_memory_usage)
                                    
                                    result_node = Node(
                                        current_node,
                                        new_dist,
                                        {},
                                        op_cost
                                    )
                                    current_node.children[operation, result_node]
                                    
                                    if new_dist == target_dist:
                                        end_nodes.append(result_node)
                                    else:
                                        open_nodes.append(result_node)

                                except:
                                    log.debug(f"Failed operation {operation} on dis {current_node.dist} with shape {shape}")
                        
                        for e_dim in extra_dims:
                            in_other_target_dim = False
                            for other_t_axis in target_axes:
                                if target_axis != other_t_axis:
                                    c_map = current_node.dist._dims_mapping[other_t_axis]
                                    if e_dim in c_map:
                                        index = c_map.index(e_dim)
                                        if index == 0 and len(c_map) == 1:

                                            

                                        if index == len(current_node.dist._dims_mapping[other_t_axis]) -1:
                                        else:
                                        
                                        break
                                    

                            
                                
                    



        # 4) Adjust block sizes

        return operations, total_cost
