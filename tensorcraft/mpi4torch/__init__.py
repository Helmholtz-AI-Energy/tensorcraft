"""MPI + PyTorch utilities."""

from tensorcraft.mpi4torch.util import as_buffer, tensor2mpiBuffer

__all__ = ["as_buffer", "tensor2mpiBuffer"]
