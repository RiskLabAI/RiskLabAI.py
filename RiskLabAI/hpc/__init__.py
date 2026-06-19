"""
RiskLabAI High-Performance Computing (HPC) Module

Provides utilities for multiprocessing and parallel execution.
"""

from .hpc import (
    expand_call,
    linear_partitions,
    mp_pandas_obj,
    nested_partitions,
    parallel_run,
    process_jobs,
    process_jobs_sequential,
    report_progress,
)

__all__ = [
    "report_progress",
    "expand_call",
    "process_jobs",
    "process_jobs_sequential",
    "linear_partitions",
    "nested_partitions",
    "mp_pandas_obj",
    "parallel_run",
]
