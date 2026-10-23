"""
Define the data paths, the constants, and the executors shared by the dataset modules.
"""

import concurrent.futures
import os
import random
import sys
from collections.abc import Callable, Iterable, Iterator
from typing import Any, Self

# Resolve root directory
if "google.colab" in sys.modules:
    ROOT_DIR = "/content/drive/MyDrive/pac-gcnn"
    DATA_PAC5_SQLITE = os.path.join(ROOT_DIR, "data", "pac5.db")
    DATA_PAC5_EXCEL = os.path.join(ROOT_DIR, "data", "pac5.xlsx")
    DATA_PUBCHEMQC_SQLITE = os.path.join(ROOT_DIR, "data", "pubchemqc.db")

else:
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_PAC5_SQLITE = os.path.join(ROOT_DIR, "../data", "pac5.db")
    DATA_PAC5_EXCEL = os.path.join(ROOT_DIR, "../data", "pac5.xlsx")
    DATA_PUBCHEMQC_SQLITE = os.path.join(ROOT_DIR, "../data", "pubchemqc.db")
SEED = 42
random.seed(SEED)


class MainProcessExecutor:
    """
    Run a computation in the main process with the interface of a process pool.

    `create_executor` returns this executor for 0 worker processes. The class has the context manager methods and the
    `map` method of `concurrent.futures.ProcessPoolExecutor`, making a ``with`` block that calls ``executor.map`` valid
    for both executors.
    """

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exception_information: object) -> None:
        return None

    def map[ResultType](
        self,
        function: Callable[..., ResultType],
        *iterables: Iterable[Any],
        chunksize: int = 1,
    ) -> Iterator[ResultType]:
        """
        Apply a function to the items of the iterables in the main process.

        The method returns the iterator of the built-in `map`, which calls the function on the next items only when the
        iterator is advanced. The results are in the order of the items.

        Parameters
        ----------
        function : callable
            Function with 1 positional parameter for each iterable.
        *iterables : iterable
            Arguments of the function calls. The iterator stops at the end of the shortest iterable.
        chunksize : int, default 1
            Number of items in each task of `concurrent.futures.ProcessPoolExecutor.map`. The parameter does not have an
            effect, because the main process does not divide the items into tasks.

        Returns
        -------
        iterator
            Return value of each function call, in the order of the items.
        """
        return map(function, *iterables)


def create_executor(
    num_workers: int,
) -> concurrent.futures.ProcessPoolExecutor | MainProcessExecutor:
    """
    Create the executor of a computation with the given number of worker processes.

    The function returns a `MainProcessExecutor` when `num_workers` is 0 and a `concurrent.futures.ProcessPoolExecutor`
    with `num_workers` worker processes otherwise. The ``map`` method of both executors returns the results in the order
    of the items.

    Parameters
    ----------
    num_workers : int
        Number of worker processes.

    Returns
    -------
    concurrent.futures.ProcessPoolExecutor or MainProcessExecutor
        Context manager with the method ``map``.
    """
    if num_workers == 0:
        return MainProcessExecutor()
    return concurrent.futures.ProcessPoolExecutor(max_workers=num_workers)
