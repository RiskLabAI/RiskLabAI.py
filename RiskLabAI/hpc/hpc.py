import multiprocessing as mp
import pandas as pd
import numpy as np
import datetime as dt
import time


def report_progress(job_number, n_jobs, start_time, task):
    """
    Reporting progress of computing.

    :param job_number: int, the current job number
    :param n_jobs: int, total number of jobs
    :param start_time: float, the start time of the computation
    :param task: str, the task being performed
    :return: None
    """
    # Report progress as async jobs are completed
    progress = [float(job_number) / n_jobs, (time.time() - start_time) / 60.]   # compute remaining time
    progress.append(progress[1] * (1 / progress[0] - 1))  # append remaining time
    timestamp = str(dt.datetime.fromtimestamp(time.time()))  # local time
    message = f"{timestamp} {round(progress[0] * 100, 2)}% {task} done after {round(progress[1], 2)} minutes. " \
              f"Remaining {round(progress[2], 2)} minutes."  # create message
    if job_number < n_jobs:
        print(message, end='\r')
    else:
        print(message)
    return


def process_jobs(jobs, task=None, n_threads=24):
    """
    Process multiple jobs in parallel.

    :param jobs: list, a list of jobs to be processed
    :param task: str, the task being performed
    :param n_threads: int, number of threads to be used
    :return: list, outputs of the jobs
    """
    # Run in parallel.
    # jobs must contain a 'func' callback, for expand_call
    if task is None:
        task = jobs[0]['func'].__name__  # initial func
    pool = mp.Pool(processes=n_threads)  # thread pool
    outputs, output, time0 = pool.imap_unordered(expand_call, jobs), [], time.time()  # initial task to processors

    # Process asynchronous output, report progress
    for i, out_ in enumerate(outputs):
        output.append(out_)
        report_progress(i, len(jobs), time0, task)
    pool.close()
    pool.join()  # this is needed to prevent memory leaks
    return output


def expand_call(kargs):
    """
    Expand the arguments of a callback function, kargs['func']

    :param kargs: dict, arguments for the callback function
    :return: output of the callback function
    """
    func = kargs['func']
    del kargs['func']
    output = func(**kargs)
    return output


def process_jobs_sequential(jobs):
    """
    Single-thread execution, for debugging.

    :param jobs: list, a list of jobs to be processed
    :return: list, outputs of the jobs
    """
    output = []
    for job in jobs:
        output_ = expand_call(job)
        output.append(output_)
    return output


def linear_partitions(n_atoms, n_threads):
    """
    Generate linear partitions for parallel computation.

    :param n_atoms: int, number of atoms
    :param n_threads: int, number of threads
    :return: list, the partitions
    """
    partitions = np.linspace(0, n_atoms, min(n_threads, n_atoms) + 1)  # split [0...nAtoms) into partitions
    partitions = np.ceil(partitions).astype(int)  # integerize(!) number
    return partitions


def nested_partitions(n_atoms, n_threads, upper_triangle=False):
    """
    Generate nested partitions for parallel computation.

    :param n_atoms: int, number of atoms
    :param n_threads: int, number of threads
    :param upper_triangle: bool, whether to generate partitions for the upper triangle
    :return: list, the partitions
    """
    partitions, n_threads_ = [0], min(n_threads, n_atoms)
    for num in range(n_threads_):
        # compute parts size according to formula on page 308 of the book
        partitions = 1 + 4 * (partitions[-1] ** 2 + partitions[-1] + n_atoms * (n_atoms + 1.) / n_threads_)
        partitions = (-1 + partitions ** .5) / 2.
        partitions.append(partitions)
    partitions = np.round(partitions).astype(int)
    if upper_triangle:  # the first rows are the heaviest
        partitions = np.cumsum(np.diff(partitions)[::-1])
        partitions = np.append(np.array([0]), partitions)
    return partitions


def mp_pandas_obj(function, pandas_object, n_threads=2, mp_batches=1, linear_partition=True, **kwargs):
    """
    Parallelize jobs, return a DataFrame or Series.

    :param function: function, the function to be parallelized
    :param pandas_object: tuple, a tuple containing the name of the argument used to pass the molecule
        and a list of atoms that will be grouped into molecules
    :param n_threads: int, number of threads to be used
    :param mp_batches: int, number of batches for multiprocessing
    :param linear_partition: bool, whether to use linear partitioning or nested partitioning
    :param kwargs: other arguments needed by function
    :return: DataFrame or Series, the result of the function parallelized
    """
    if linear_partition:  # check which partition algorithm is most used
        parts = linear_partitions(len(pandas_object[1]), n_threads * mp_batches)
    else:
        parts = nested_partitions(len(pandas_object[1]), n_threads * mp_batches)

    jobs = []
    for i in range(1, len(parts)):
        job = {pandas_object[0]: pandas_object[1][parts[i - 1]:parts[i]], 'func': function}  # create job
        job.update(kwargs)  # append arguments of func to job
        jobs.append(job)  # append job to jobs

    if n_threads == 1:  # check number of threads
        out = process_jobs_sequential(jobs)
    else:
        out = process_jobs(jobs, num_threads=n_threads)

    if isinstance(out[0], pd.DataFrame):  # check type of out
        df0 = pd.DataFrame()
    elif isinstance(out[0], pd.Series):
        df0 = pd.Series()
    else:
        return out
    # according to output of func, sort index because multiprocessing approach reorder output!
    for i in out:
        df0 = df0.append(i)
    df0 = df0.sort_index()
    return df0
