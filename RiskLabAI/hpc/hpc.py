"""
High-Performance Computing (HPC) utilities for parallel processing.

Provides a set of functions to parallelize operations, especially
on pandas objects, using Python's multiprocessing.
"""

import multiprocessing as mp
import pandas as pd
import numpy as np
import datetime as dt
import time
from typing import List, Dict, Any, Callable, Tuple, Union, Iterable
from typing import Optional

def parallel_run(
    func: Callable[..., Any],
    iterable: Iterable[Any],
    num_cpus: int = -1,
    lin_partition: bool = False,  # <-- RECOMMENDED: Change default to False
    **kwargs
) -> List[Any]:
    """
    Executes a function in parallel over an iterable.

    Parameters
    ----------
    func : Callable
        The function to be parallelized.
    iterable : Iterable
        The iterable to loop over.
    num_cpus : int, default=-1
        The number of CPUs to use. -1 means all available CPUs.
    lin_partition : bool, default=False
        How to partition the work:
        - If False (default): Item-by-item. `func` receives one item
          from `iterable` at a time.
        - If True: Chunked. `iterable` is split into `num_cpus` chunks
          (by index). `func` must be designed to receive a list of
          indices (e.g., [0, 1, 2]) and process them.
    **kwargs :
        Additional keyword arguments to be passed to `func`.

    Returns
    -------
    List[Any]
        A list of the results from all parallel executions.
    """
    if num_cpus == -1:
        num_cpus = multiprocessing.cpu_count()

    if lin_partition:
        # --- Chunked Partitioning ---
        # func must accept a list of indices
        num_atoms = num_cpus
        iterable_partition = np.array_split(range(len(iterable)), num_atoms)
        jobs = (iterable_partition[i] for i in range(num_atoms))

        results = joblib.Parallel(n_jobs=num_cpus)(
            joblib.delayed(func)(job, **kwargs) for job in jobs
        )
        
        # Flatten the list of lists
        return [item for sublist in results for item in sublist]

    else:
        # --- Item-by-Item Partitioning (Standard) ---
        # func accepts a single item
        jobs = (iterable[i] for i in range(len(iterable)))

        results = joblib.Parallel(n_jobs=num_cpus)(
            joblib.delayed(func)(job, **kwargs) for job in jobs
        )
        return results


def report_progress(
    job_number: int, total_jobs: int, start_time: float, task: str
) -> None:
    """
    Report the progress of a computing task to the console.

    Parameters
    ----------
    job_number : int
        The current job number (e.g., `i` in a loop).
    total_jobs : int
        The total number of jobs.
    start_time : float
        The time the process started (e.g., `time.time()`).
    task : str
        A description of the task being performed.
    """
    progress = job_number / total_jobs
    elapsed_time_min = (time.time() - start_time) / 60.0
    remaining_time_min = elapsed_time_min * (1 / progress - 1)
    
    timestamp = str(dt.datetime.fromtimestamp(time.time()))
    message = (
        f"{timestamp} {progress*100:.2f}% {task} done after "
        f"{elapsed_time_min:.2f} minutes. Remaining {remaining_time_min:.2f} minutes."
    )
    
    if job_number < total_jobs:
        print(message, end='\r')
    else:
        print(message)


def expand_call(kargs: Dict[str, Any]) -> Any:
    """
    Wrapper function to expand keyword arguments for a callback.

    This is used by `process_jobs` to unpack a dictionary of arguments
    and call the target function.

    Parameters
    ----------
    kargs : dict
        A dictionary of arguments, which *must* include a 'func' key
        mapping to the function to be called.

    Returns
    -------
    Any
        The output of the callback function.
    """
    func = kargs.pop('func')
    return func(**kargs)


def process_jobs(
    jobs: List[Dict[str, Any]], task: Optional[str] = None, num_threads: int = 24
) -> List[Any]:
    """
    Process a list of jobs in parallel using multiprocessing.

    Parameters
    ----------
    jobs : List[Dict[str, Any]]
        A list of job dictionaries. Each dict must contain a 'func' key
        and all arguments required by that function.
    task : str, optional
        A name for the task, used for progress reporting. If None,
        the function name from the first job is used.
    num_threads : int, default=24
        The number of parallel processes to spawn.

    Returns
    -------
    List[Any]
        A list containing the results from all jobs.
    """
    if task is None:
        task = jobs[0]['func'].__name__

    with mp.Pool(processes=num_threads) as pool:
        outputs = []
        start_time = time.time()
        
        # Use imap_unordered for efficient processing
        imap_results = pool.imap_unordered(expand_call, jobs)
        
        # Process results as they complete
        for i, result in enumerate(imap_results, 1):
            outputs.append(result)
            report_progress(i, len(jobs), start_time, task)
            
    return outputs


def process_jobs_sequential(jobs: List[Dict[str, Any]]) -> List[Any]:
    """
    Run jobs sequentially (single-thread). Useful for debugging.

    Parameters
    ----------
    jobs : List[Dict[str, Any]]
        A list of job dictionaries.

    Returns
    -------
    List[Any]
        A list containing the results from all jobs.
    """
    output = []
    for job in jobs:
        output_ = expand_call(job)
        output.append(output_)
    return output


def linear_partitions(num_atoms: int, num_threads: int) -> np.ndarray:
    """
    Generate linear partitions (indices) for parallel computation.

    Splits `num_atoms` into `num_threads` (or fewer) roughly equal parts.

    Parameters
    ----------
    num_atoms : int
        The total number of items to split.
    num_threads : int
        The desired number of partitions (threads).

    Returns
    -------
    np.ndarray
        An array of partition boundary indices.
    """
    n_parts = min(num_threads, num_atoms)
    partitions = np.linspace(0, num_atoms, n_parts + 1)
    partitions = np.ceil(partitions).astype(int)
    return partitions


def nested_partitions(
    num_atoms: int, num_threads: int, upper_triangle: bool = False
) -> np.ndarray:
    r"""
    Generate nested partitions, useful for tasks with nested loops.

    The formula is designed to create partitions of varying sizes,
    which can be more efficient for tasks where work is not
    distributed linearly (e.g., matrix operations).

    .. math::
       p_i = \frac{-1 + \sqrt{1 + 4(p_{i-1}^2 + p_{i-1} + N(N+1)/K)}}{2}

    Parameters
    ----------
    num_atoms : int
        Number of atoms (N).
    num_threads : int
        Number of threads (K).
    upper_triangle : bool, default=False
        If True, sort partitions to be heaviest first.

    Returns
    -------
    np.ndarray
        An array of partition boundary indices.
    """
    partitions = [0]
    n_threads_ = min(num_threads, num_atoms)
    
    for _ in range(n_threads_):
        last_part = partitions[-1]
        part_size = 1 + 4 * (
            last_part**2 + last_part + num_atoms * (num_atoms + 1.0) / n_threads_
        )
        part_val = (-1 + part_size**0.5) / 2.0
        partitions.append(part_val)
        
    partitions = np.round(partitions).astype(int)
    
    if upper_triangle:  # The first rows are the heaviest
        partitions = np.cumsum(np.diff(partitions)[::-1])
        partitions = np.append(np.array([0]), partitions)
        
    return partitions


def mp_pandas_obj(
    func: Callable[..., pd.Series],
    pandas_object: Tuple[str, pd.Index],
    num_threads: int = 2,
    mp_batches: int = 1,
    linear_partition: bool = True,
    **kwargs: Any
) -> Union[pd.DataFrame, pd.Series]:
    """
    Parallelize a function call on a pandas object (DataFrame/Series).

    This function splits the `pandas_object` index into partitions and
    calls `func` on each partition in parallel.

    Parameters
    ----------
    func : Callable
        The function to be parallelized. It must accept a pandas
        object as its first argument.
    pandas_object : Tuple[str, pd.Index]
        A tuple where:
        - [0] (str): The name of the argument in `func` that
                     receives the partition (e.g., 'molecule').
        - [1] (pd.Index): The index to be partitioned.
    num_threads : int, default=2
        Number of parallel processes.
    mp_batches : int, default=1
        Number of batches to split the jobs into.
    linear_partition : bool, default=True
        If True, use `linear_partitions`.
        If False, use `nested_partitions`.
    **kwargs : Any
        Other keyword arguments to be passed to `func`.

    Returns
    -------
    Union[pd.DataFrame, pd.Series]
        The concatenated results from all parallel calls.
    """
    if linear_partition:
        parts = linear_partitions(
            len(pandas_object[1]), num_threads * mp_batches
        )
    else:
        parts = nested_partitions(
            len(pandas_object[1]), num_threads * mp_batches
        )

    jobs = []
    for i in range(1, len(parts)):
        job = {
            pandas_object[0]: pandas_object[1][parts[i - 1] : parts[i]],
            "func": func,
        }
        job.update(kwargs)
        jobs.append(job)

    if num_threads == 1:
        out = process_jobs_sequential(jobs)
    else:
        out = process_jobs(jobs, num_threads=num_threads)

    if isinstance(out[0], pd.DataFrame):
        result_df = pd.concat(out)
    elif isinstance(out[0], pd.Series):
        result_df = pd.concat(out)
    else:
        return out

    return result_df.sort_index()