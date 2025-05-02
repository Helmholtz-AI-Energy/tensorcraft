import logging
import random
import time

import propulate
import torch
from mpi4py import MPI

import tensorcraft as tc

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
print(f"Hello from rank {rank}!")

ALPHA = 1e-6  # 1 micro second of latency (Maybe bigger)
BETA = 64.0 / (200.0 * 1e9)  # 200 GBits per second bandwidth

search_space = {
    "path_cost_w": (0.0, 1000.0),
    "estimate_w": (0.0, 1000.0),
    "max_depth": (5, 100),
    "node_limit": (100, 10000),
    "top_k": (1, 100),
}

# Shape, start_dist, target_dist
problems = [
    (
        torch.Size([100000, 10000]),
        tc.dist.MultiAxisDist(torch.Size([2, 2]), ((0,), (1,)), 100),
        tc.dist.MultiAxisDist(torch.Size([2, 2]), ((0, 1), None), 100),
    ),
    (
        torch.Size([500, 100, 100, 8]),
        tc.dist.MultiAxisDist(torch.Size([4, 4, 2]), ((0,), None, None, (1, 2)), 1),
        tc.dist.MultiAxisDist(
            torch.Size([4, 4, 2]), ((0,), (1,), (2,), None), (1, 25, 25, None)
        ),
    ),
    (
        torch.Size([1000, 1000, 1000]),
        tc.dist.MultiAxisDist(torch.Size([2, 2, 2]), ((0,), (1,), (2,)), 1),
        tc.dist.MultiAxisDist(
            torch.Size([2, 2, 2]), ((0, 1), (2,), None), (1, 50, None)
        ),
    ),
]


def mem_constrained_filter(
    shape: torch.Size,
    start_dist: tc.dist.MultiAxisDist,
    target_dist: tc.dist.MultiAxisDist,
    current_dist: tc.dist.MultiAxisDist,
) -> bool:
    max_n_elements = max(
        start_dist.maxNumElements(shape), target_dist.maxNumElements(shape)
    )
    return max_n_elements < current_dist.maxNumElements(shape)


def cost_function(params: dict[str, any]):
    cost_model = tc.optim.IdealLowerBoundsCM()
    redistributor = tc.optim.AStarRedistributor(
        cost_model, alpha=ALPHA, beta=BETA, **params
    )

    start_time = time.time()

    total_redist_time = 0.0
    for shape, start_dist, target_dist in problems:
        sequence, redist_time_cost = redistributor.redistribute(
            shape, start_dist, target_dist
        )
        total_redist_time += redist_time_cost

    end_time = time.time()
    scheduling_time = end_time - start_time
    return total_redist_time * 10000 + scheduling_time


tc.set_logger_config(level=logging.INFO, log_to_stdout=True)

propulate.set_logger_config(
    level=logging.INFO,
    log_to_stdout=True,
    log_file="./prop_cp/prop.log",
    log_rank=True,
    colors=True,
)


rng = random.Random(3459 + comm.rank)

propagator = propulate.get_default_propagator(
    pop_size=comm.Get_size(),
    limits=search_space,
)

propulator = propulate.Propulator(
    loss_fn=cost_function,
    propagator=propagator,
    rng=rng,
    island_comm=comm,
    checkpoint_path="./prop_cp/",
)

propulator.propulate(debug=1)

propulator.summarize(
    top_n=1,
    debug=1,
)
