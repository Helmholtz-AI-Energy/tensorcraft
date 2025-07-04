import time
import jax
from jax.random import f
from jax.sharding import PartitionSpec as P, NamedSharding, AxisType, set_mesh, get_abstract_mesh
import jax.sharding as jsh
import jax.numpy as jnp
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
world_size = comm.Get_size()

jax.distributed.initialize("localhost:12345", world_size, rank)

print(jax.devices("cpu"))
print(jax.device_count())
assert jax.device_count() == world_size



mesh = jax.make_mesh((2, 2), ("x", "y"), axis_types=(AxisType.Explicit, AxisType.Explicit))


set_mesh(mesh)
print(f"Current mesh: {get_abstract_mesh()}")

n = 100000
n_elements = n * n
replicated_array = jnp.arange(n_elements).reshape(n, n)
print(f"R{rank}/{world_size}: Replicated array shape: {replicated_array.shape}, dtype: {replicated_array.dtype}, memory size: {replicated_array.nbytes * 10**-9:.2f} GB")

start = time.time_ns()
sharded_array = jax.device_put(replicated_array, device=P(("x","y"), None))
mark_1 = time.time_ns()

resharded_array = jax.device_put(sharded_array, device=P(None, ("x", "y")))
mark_2 = time.time_ns()

if rank == 0:
    print(f"R{rank}/{world_size}: Time to shard: {(mark_1 - start) / 1e6} ms")
    print(f"R{rank}/{world_size}: Time to reshuffle: {(mark_2 - mark_1) / 1e6} ms")
    print(f"R{rank}/{world_size}: Total time: {(mark_2 - start) / 1e6} ms")
