import jax
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
world_size = comm.Get_size()

jax.distributed.initialize("localhost:12345", world_size, rank)

print(jax.devices("cpu"))
print(jax.device_count())
assert jax.device_count() == world_size

mesh = jax.make_mesh((2, 2), ("x", "y"))
# Create an array of random values:
x = jax.random.normal(jax.random.key(0), (256, 256))
# and use jax.device_put to distribute it across devices:
y = jax.device_put(x, NamedSharding(mesh, P("x", "y")))
print(f"R{rank}/{world_size}: {y.addressable_shards}")
