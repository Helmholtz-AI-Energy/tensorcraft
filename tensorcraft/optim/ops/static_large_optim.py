from tensorcraft.optim.ops.tensor_exp_optim import TensorExpressionOptimizer


class StaticLargeOptim(TensorExpressionOptimizer):
    def _setup(self):
        return

    def optimizeOp(self, tensor_exp, tensor_shapes, tensor_dist):
        if not self.validArguments(tensor_exp, tensor_shapes, tensor_dist):
            raise ValueError("Invalid argument combination. See logs.")
