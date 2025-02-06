"""Utility function for redistributions."""


def allgather_bandwidth_cost(n_procs: int, n_elements: int) -> float:
    """
    Calculate the bandwidth cost of an allgather operation.

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
    Calculate the bandwidth cost of a permute operation.

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


def allreduce_bandwidth_cost(n_procs: int, n_elements: int) -> float:
    """
    Calculate the bandwidth cost of an all-reduce operation.

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


def reduce_scatter_bandwidth_cost(n_procs: int, n_elements: int) -> float:
    """
    Calculate the bandwidth cost of a reduce-scatter operation.

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


def all2all_bandwidth_cost(n_procs: int, n_elements: int) -> float:
    """
    Calculate the bandwidthco cost of an all2all operation.

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
