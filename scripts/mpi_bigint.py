import torch
import tensorcraft as tc
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

options  = [torch.iinfo(torch.int32).max, torch.iinfo(torch.uint32).max, torch.iinfo(torch.uint32).max * 10]

# options = [128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768]
for possible_max in options:
    print(f"Trying {possible_max}")
    if comm.Get_rank() == 0:
        A = torch.randn(possible_max)
        print(A.dtype)
        print(A[:10])

        print(f"R0: Sending {possible_max} elements")

    else:
        A = torch.zeros(possible_max, dtype=torch.float32)


    buffer, count, mpi_type = tc.mpi4torch.tensor2mpiBuffer(A)
    comm.Bcast(buf=[buffer, count, mpi_type], root=0)

    if comm.Get_rank() == 0:
        print("R0: Sent!")
        del A
    else:
        print("R1: Received!")
        print(A[:10])
        del A
    comm.Barrier()