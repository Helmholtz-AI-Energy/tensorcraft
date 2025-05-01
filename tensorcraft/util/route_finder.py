"""Route finder module."""

import bisect
import dataclasses
import logging
from typing import Any, Callable, Generic, Optional, TypeAlias, TypeVar

log = logging.getLogger(__name__)

T = TypeVar("T")


@dataclasses.dataclass
class RouteNode(Generic[T]):
    """Distribution graph nodes."""

    parent_node: Optional["RouteNode"]
    parent_node_edge_op: str
    obj: T
    children: dict[str, "RouteNode"]
    edge_cost: float
    modified: bool = True

    def path_to_root(self) -> tuple[list[tuple[str, T, float]], float]:
        """
        Get the path to the root node.

        Returns
        -------
        list[tuple[str, Dist, Cost]]
            The path to the root node.
        """
        path: list[tuple[str, T, float]] = []
        current_node: "RouteNode" | None = self
        path_cost = 0.0
        while current_node:
            path_cost += current_node.edge_cost
            path.insert(
                0,
                (
                    current_node.parent_node_edge_op,
                    current_node.obj,
                    current_node.edge_cost,
                ),
            )
            current_node = current_node.parent_node
        return path, path_cost

    def path_depth_cost(self) -> tuple[int, float]:
        """
        Get the path cost.

        Returns
        -------
        int
            The path depth.
        float
            The path cost.
        """
        current_node: "RouteNode" | None = self
        path_cost = 0.0
        path_depth = -1

        while current_node:
            path_cost += current_node.edge_cost
            current_node = current_node.parent_node
            path_depth += 1

        return path_depth, path_cost

    def __str__(self) -> str:
        return f"Node: {self.obj}, Cost: {self.edge_cost}, parent_op: {self.parent_node_edge_op}"


neighbours_type: TypeAlias = Callable[[RouteNode], list[tuple[str, RouteNode]]]
priority_func_type: TypeAlias = Callable[[RouteNode], float]


def find_routes(
    start_node: RouteNode,
    end_node: RouteNode,
    neighbours: neighbours_type,
    priority_func: priority_func_type | None = None,
    max_depth: int = -1,
    node_limit: int = 1000,
    top_k: int = -1,
) -> list[tuple[list[tuple[str, Any, float]], float]]:
    """
    Find routes between two nodes.

    Parameters
    ----------
    start_node : Node
        The starting node.
    end_node : Node
        The ending node.
    neighbours : Callable[[Node], list[tuple[str, Node]]]
        The neighbour function.
    priority_func : Optional[Callable[[Node], float]], optional
        The priority function, by default None.
    max_depth : int, optional
        The maximum depth, by default -1.
    node_limit : int, optional
        The node limit, by default 1000.

    Returns
    -------
    list[tuple[list[tuple[str, Any, float]], float]]
        The list of paths.
    """
    open_nodes: list[RouteNode] = []
    close_nodes: list[RouteNode] = []
    end_nodes: list[RouteNode] = []
    nodes_dict: dict[str, RouteNode] = {}

    open_nodes.append(start_node)
    nodes_dict[str(start_node)] = start_node

    node_count = 0

    while len(open_nodes) > 0 and node_count < node_limit and len(end_nodes) != top_k:
        log.debug(f"Open nodes: {len(open_nodes)}")
        log.debug(f"Close nodes: {len(close_nodes)}")
        log.debug(f"End nodes: {len(end_nodes)}")
        node_count += 1
        current_node = open_nodes.pop(0)
        current_depth, current_path_cost = current_node.path_depth_cost()

        close_nodes.append(current_node)

        if max_depth > 0 and current_depth >= max_depth:
            continue

        neighbours_list = neighbours(current_node)
        log.debug(f"Current node: {current_node} -> ")
        for id, n_node in neighbours_list:
            log.debug(f"             {n_node.obj}")

            path_cost = n_node.edge_cost + current_path_cost

            if n_node.obj == end_node.obj:
                end_nodes.append(n_node)
                continue

            if id in nodes_dict:
                alt_node = nodes_dict[id]
                alt_depth, alt_cost = alt_node.path_depth_cost()

                if alt_cost > path_cost or (
                    alt_cost == path_cost and alt_depth > current_depth + 1
                ):
                    alt_node.parent_node = current_node
                    alt_node.edge_cost = n_node.edge_cost
                    alt_node.parent_node_edge_op = n_node.parent_node_edge_op
                continue
            else:
                nodes_dict[id] = n_node
                if priority_func:
                    score = priority_func(n_node)
                    score_list: list[float] = list(map(priority_func, open_nodes))

                    idx = bisect.bisect(score_list, score)
                    open_nodes.insert(idx, n_node)
                else:
                    open_nodes.append(n_node)

    log.info(f"Explored {node_count} nodes, found {len(end_nodes)} possible paths.")
    paths = []
    for end_node in end_nodes:
        paths.append(end_node.path_to_root())

    return paths
