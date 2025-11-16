"""
Provides a simple terminal-style progress bar.
"""

import sys
import time
from typing import Optional

def progress_bar(
    current_progress: int,
    total_progress: int,
    start_time: float,
    bar_length: int = 20,
) -> None:
    r"""
    Display a terminal-style progress bar.

    Shows completion percentage and estimated time remaining.

    Parameters
    ----------
    current_progress : int
        Current iteration number (e.g., `i`).
    total_progress : int
        Total number of items to process.
    start_time : float
        The time the task started (from `time.time()`).
    bar_length : int, default=20
        Length of the progress bar in characters.
    """
    if total_progress == 0:
        return
    
    # Check if task is completed
    if current_progress == total_progress:
        sys.stdout.write(
            f"\rCompleted: [{'-' * bar_length}] 100% - Task completed!          \n"
        )
    else:
        percentage = current_progress / total_progress
        
        # Handle the very first iteration
        if percentage == 0:
            arrow = ""
        else:
            # Safely calculate arrow length, ensuring it doesn't go below -1
            arrow_length = int(round(percentage * bar_length) - 1)
            arrow = "-" * max(0, arrow_length) + ">"
        
        spaces = " " * (bar_length - len(arrow))
        elapsed_time_sec = time.time() - start_time

        # Calculate remaining time
        if current_progress == 0:
            remaining_time_str = "Calculating..."
        else:
            remaining_time_min = (
                (elapsed_time_sec / current_progress)
                * (total_progress - current_progress)
            ) / 60
            remaining_time_str = f"{remaining_time_min:.0f} minutes remaining"

        # Assemble and print the bar
        sys.stdout.write(
            f"\rCompleted: [{arrow + spaces}] {percentage*100:.0f}% - {remaining_time_str}."
        )
    
    sys.stdout.flush()