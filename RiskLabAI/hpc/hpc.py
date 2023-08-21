import multiprocessing as mp
import pandas as pd
import numpy as np
import datetime as dt
import time


def report_progress(
    job_number: int,
    total_jobs: int,
    start_time: float,
    task: str
) -> None:
    """
    Report the progress of a computing task.

    :param job_number: The current job number.
    :type job_number: int
    :param total_jobs: The total number of jobs.
    :type total_jobs: int
    :param start_time: The start time of the computation.
    :type start_time: float
    :param task: The task being performed.
    :type task: str
    :return: None
    """
    progress = [float(job_number) / total_jobs, (time.time() - start_time) / 60.]   # compute remaining time
    progress.append(progress[1] * (1 / progress[0] - 1))  # append remaining time
    timestamp = str(dt.datetime.fromtimestamp(time.time()))  # local time
    message = f"{timestamp} {round(progress[0] * 100, 2)}% {task} done after {round(progress[1], 2)} minutes. " \
              f"Remaining {round(progress[2], 2)} minutes."  # create message
    if job_number < total_jobs:
        print(message, end='\r')
    else:
        print(message)


def process_jobs(
    jobs: list,
    task: str = None,
    num_threads: int = 24
) -> list:
    """
    Process multiple jobs in parallel.

    :param jobs: A list of jobs to be processed.
    :type jobs: list
    :param task: The task being performed.
    :type task: str
    :param num_threads: Number of threads to be used.
    :type num_threads: int
    :return: Outputs of the jobs.
    :rtype: list
    """
    if task is None:
        task = jobs[0]['func'].__name__  # initial func
    pool = mp.Pool(processes=num_threads)  # thread pool
    outputs, output, time0 = pool.imap_unordered(expand_call, jobs), [], time.time()  # initial task to processors

    # Process asynchronous output, report progress
    for i, out_ in enumerate(outputs):
        output.append(out_)
        report_progress(i, len(jobs), time0, task)
    pool.close()
    pool.join()  # this is needed to prevent memory leaks
    return output


def expand_call(kargs: dict):
    """
    Expand the arguments of a callback function, kargs['func'].

    :param kargs: Arguments for the callback function.
    :type kargs: dict
    :return: Output of the callback function.
    """
    func = kargs['func']
    del kargs['func']
    output = func(**kargs)
    return output


def process_jobs_sequential(jobs: list) -> list:
    """
    Single-thread execution, for debugging.

    :param jobs: A list of jobs to be processed.
    :type jobs: list
    :return: Outputs of the jobs.
    :rtype: list
    """
    output = []
    for job in jobs:
        output_ = expand_call(job)
        output.append(output_)
    return output


def linear_partitions(
    num_atoms: int,
    num_threads: int
) -> list:
    """
    Generate linear partitions for parallel computation.

    :param num_atoms: Number of atoms.
    :type num_atoms: int
    :param num_threads: Number of threads.
    :type num_threads: int
    :return: The partitions.
    :rtype: list
    """
    partitions = np.linspace(0, num_atoms, min(num_threads, num_atoms) + 1)  # split [0...nAtoms) into partitions
    partitions = np.ceil(partitions).astype(int)  # integerize(!) number
    return partitions


def nested_partitions(
    num_atoms: int,
    num_threads: int,
    upper_triangle: bool = False
) -> list:
    """
    Generate nested partitions for parallel computation.

    :param num_atoms: Number of atoms.
    :type num_atoms: int
    :param num_threads: Number of threads.
    :type num_threads: int
    :param upper_triangle: Whether to generate partitions for the upper triangle.
    :type upper_triangle: bool
    :return: The partitions.
    :rtype: list

    The formula for partition size is given by:

    .. math::

       partitions = \frac{-1 + \sqrt{1 + 4 \cdot (partitions[-1]^2 + partitions[-1] + \frac{n_atoms \cdot (n_atoms + 1)}{n_threads})}}{2}
    """
    partitions, n_threads_ = [0], min(num_threads, num_atoms)
    for num in range(n_threads_):
        partitions_value = 1 + 4 * (partitions[-1] ** 2 + partitions[-1] + num_atoms * (num_atoms + 1.) / n_threads_)
        partitions_value = (-1 + partitions_value ** .5) / 2.
        partitions.append(partitions_value)
    partitions = np.round(partitions).astype(int)
    if upper_triangle:  # the first rows are the heaviest
        partitions = np.cumsum(np.diff(partitions)[::-1])
        partitions = np.append(np.array([0]), partitions)
    return partitions


def mp_pandas_obj(
    function,
    pandas_object: tuple,
    num_threads: int = 2,
    mp_batches: int = 1,
    linear_partition: bool = True,
    **kwargs
) -> pd.DataFrame or pd.Series:
    """
    Parallelize jobs and return a DataFrame or Series.

    :param function: The function to be parallelized.
    :param pandas_object: A tuple containing the name of the argument used to pass the molecule and a list of atoms
                          that will be grouped into molecules.
    :type pandas_object: tuple
    :param num_threads: Number of threads to be used.
    :type num_threads: int
    :param mp_batches: Number of batches for multiprocessing.
    :type mp_batches: int
    :param linear_partition: Whether to use linear partitioning or nested partitioning.
    :type linear_partition: bool
    :param kwargs: Other arguments needed by the function.
    :return: The result of the function parallelized.
    :rtype: DataFrame or Series
    """
    if linear_partition:
        parts = linear_partitions(len(pandas_object[1]), num_threads * mp_batches)
    else:
        parts = nested_partitions(len(pandas_object[1]), num_threads * mp_batches)

    jobs = []
    for i in range(1, len(parts)):
        job = {pandas_object[0]: pandas_object[1][parts[i - 1]:parts[i]], 'func': function}
        job.update(kwargs)
        jobs.append(job)

    if num_threads == 1:
        out = process_jobs_sequential(jobs)
    else:
        out = process_jobs(jobs, num_threads=num_threads)

    if isinstance(out[0], pd.DataFrame):
        result = pd.DataFrame()
    elif isinstance(out[0], pd.Series):
        result = pd.Series()
    else:
        return out

    for i in out:
        result = result.append(i)
    result = result.sort_index()
    return result
