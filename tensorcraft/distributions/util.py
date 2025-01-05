def allgather_bandwidth_cost(n_procs: int, n_elements: int) -> float:
    """
    Calculate the bandwidth of an allgather operation.

    Parameters
    ----------
    n_procs : int
        The number of processors.
    n_elements : int
        The number of elements to communicate.

    Returns
    -------
    float
        The bandwidth of the allgather operation.
    """
    return (n_procs - 1) * n_elements / n_procs


def permute_bandwith_cost(n_procs: int, n_elements: int) -> float:
    """
    Calculate the bandwidth of a permute operation.

    Parameters
    ----------
    n_procs : int
        The number of processors.
    n_elements : int
        The number of elements to communicate.

    Returns
    -------
    float
        The bandwidth of the permute operation.
    """
    return n_elements


def scatter_bandwidth_cost(n_procs: int, n_elements: int) -> float:
    """
    Calculate the bandwidth of a scatter operation.

    Parameters
    ----------
    n_procs : int
        The number of processors.
    n_elements : int
        The number of elements to communicate.

    Returns
    -------
    float
        The bandwidth of the scatter operation.
    """
    return (n_procs - 1) * n_elements / n_procs
