import abc
import dataclasses
import logging
from typing import Self

import torch

from tensorcraft.compiler.model import TensorExpression
from tensorcraft.distributions.dist import Dist

log = logging.getLogger("tensorcraft")


@dataclasses.dataclass
class ExpressionCost:
    computation: float
    bandwidth: float
    latency: float

    def __add__(self, other: Self) -> Self:
        return ExpressionCost(
            self.computation + other.computation,
            self.bandwidth + other.bandwidth,
            self.latency + other.latency,
        )

    def __str__(self):
        return (
            f"ExpressionCost: {self.latency}α + {self.bandwidth}β + {self.computation}γ"
        )


class TensorExpressionOptimizer(abc.ABC):
    def __init__(self):
        super().__init__()
        self.setup()

    @abc.abstractmethod
    def setup(self):
        raise NotImplementedError()

    @abc.abstractmethod
    def optimizeOp(
        self,
        tensor_exp: TensorExpression,
        tensor_shapes: dict[str, torch.Size],
        tensor_dist: dict[str, Dist],
    ) -> tuple[list[tuple[str, ExpressionCost]], ExpressionCost]:
        """
        Optimize the given tensor expression based on tensor shapes and distribution.

        Parameters
        ----------
        tensor_exp : TensorExpression
            The tensor expression to be optimized.
        tensor_shapes : dict[str, torch.Size]
            A dictionary mapping tensor names to their shapes.
        tensor_dist : dict[str, Dist]
            A dictionary mapping tensor names to their distribution information.

        Returns
        -------
        tuple[list[tuple[str, ExpressionCost]], ExpressionCost]
            A tuple containing a list of tuples with the redistributions/op names and their associated expression costs,
            and the overall expression cost.
        """
        raise NotImplementedError()

    def validArguments(
        self,
        tensor_exp: TensorExpression,
        tensor_shapes: dict[str, torch.Size],
        tensor_dist: dict[str, Dist],
    ) -> bool:
        # 1) Check requirenents.
        # 1a) All variable shapes presents and compatible.
        # 1b) All tensor need a explicit distribution.
        for tensor_name, tensor_idx_vars in tensor_exp.inputs:
            if tensor_name not in tensor_shapes:
                log.error(
                    f"Missing shape of {tensor_name}. Please provide a shape for each tensor in the expression."
                )
                return False
            if len(tensor_idx_vars) != len(tensor_shapes):
                log.error(
                    f"Incompatible index expression with provided shape for tensor {tensor_name}"
                )
                return False
            if tensor_name not in tensor_dist:
                log.error(f"Tensor {tensor_name} needs to be distributed.")
                return False
            if not tensor_dist[tensor_name]:
                log.error(f"Missing dist information for {tensor_name}")
                return False

        # 1c) All distributions use the same mesh.
        # 1d) All distributions of the same type.
        g_mesh: torch.Size = None
        dist_type = None
        for t_name, t_dist in tensor_dist.items():
            if g_mesh:
                if g_mesh != t_dist.processorMesh:
                    log.error(
                        "All tensors must be distributed on the same processor mesh."
                    )
                    return False
            else:
                g_mesh == t_dist.processorMesh

            if dist_type:
                if dist_type != type(t_dist):
                    log.error("All tensors must use the same distribution type.")
                    return False
            else:
                dist_type = type(t_dist)

        # 1e) All tensor shapes compatible with their distributions.
        for t_name, t_shape in tensor_shapes.items():
            if not tensor_dist[t_name].compatible(t_shape):
                log.error(
                    f"Distribution and shape of tensor {t_name} are not compatible."
                )
                return False

        return True
