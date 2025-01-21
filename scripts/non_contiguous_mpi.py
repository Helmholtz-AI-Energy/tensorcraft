import torch
import tensorcraft as tc
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Contiguous without offset
shape = (10, 10)
if comm.Get_rank() == 0:
    A = torch.randn(shape)
    print(A)
else:
    A = torch.zeros(shape)

buffer, count, mpi_type = tc.mpi4torch.tensor2mpiBuffer(A)
comm.Bcast(buf=[buffer, count, mpi_type], root=0)

if comm.Get_rank() == 0:
    print("R0: Sent!")
    del A
else:
    print("R1: Received!")
    print("Got ", A)
    del A

comm.Barrier()

# Contiguous with offset
if comm.Get_rank() == 0:
    B = torch.randn(shape)
    print(B)
    A = B[5:]
    print(A)
else:
    A = torch.zeros((5, 10))

buffer, count, mpi_type = tc.mpi4torch.tensor2mpiBuffer(A)
comm.Bcast(buf=[buffer, count, mpi_type], root=0)

if comm.Get_rank() == 0:
    print("R0: Sent!")
    del A
else:
    print("R1: Received!")
    print("Got ", A)
    del A

comm.Barrier()

# Non-contiguous
if comm.Get_rank() == 0:
    C = torch.randn(shape)
    print(C)
    A = C[..., 5:]
    print(A)
else:
    A = torch.zeros((10, 5))

buffer, count, mpi_type = tc.mpi4torch.tensor2mpiBuffer(A)
comm.Bcast(buf=[buffer, count, mpi_type], root=0)

if comm.Get_rank() == 0:
    print("R0: Sent!")
    del A
else:
    print("R1: Received!")
    print("Got ", A)
    del A



