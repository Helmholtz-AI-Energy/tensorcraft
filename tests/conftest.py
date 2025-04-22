import mpi4py.MPI as MPI
import pytest

comm = MPI.COMM_WORLD
mpi_size = comm.Get_size()
mpi_rank = comm.Get_rank()


def pytest_configure(config):
    # Register the custom mark to avoid PytestUnknownMarkWarning
    config.addinivalue_line(
        "markers",
        "mpi_test(n_ranks): Mark test to run when when MPI is available, and optionaly, only with the specified ranks",
    )


def pytest_runtest_setup(item):
    skip_marker = item.get_closest_marker("mpi_test")
    if skip_marker:
        print("Skip marker: ", skip_marker.args)
        ranks = int(skip_marker.args[0]) if len(skip_marker.args) == 1 else None
        print("Runtest setup: ", ranks)
        if mpi_size <= 1 or (ranks is not None and ranks != mpi_size):
            pytest.skip(
                f"Skipping MPI test, only {mpi_size} ranks available, but {ranks if ranks else '> 1'} required"
            )
