import sys
import time
from typing import NoReturn

def progress_bar(current_progress: int, total_progress: int, start_time: float, bar_length: int = 20) -> NoReturn:
    """
    Display a terminal-style progress bar with completion percentage and remaining time.

    :param current_progress: Current value indicating the progress made.
    :type current_progress: int
    :param total_progress: Total value representing the completion of the task.
    :type total_progress: int
    :param start_time: The time at which the task started, typically acquired via time.time().
    :type start_time: float
    :param bar_length: Length of the progress bar in terminal characters. Default is 20.
    :type bar_length: int

    :return: None
    """

    if current_progress == 0:
        remaining_time = "Calculating..."
    else:
        percentage = float(current_progress) / total_progress
        arrow = '-' * int(round(percentage * bar_length) - 1) + '>'
        spaces = ' ' * (bar_length - len(arrow))

        elapsed_time = time.time() - start_time
        remaining_time = int((elapsed_time / current_progress) * (total_progress - current_progress) / 60)

    if current_progress == total_progress:
        sys.stdout.write("\rCompleted: [{0}] 100% - Task completed!\n".format('-' * bar_length))
    else:
        sys.stdout.write("\rCompleted: [{0}] {1}% - {2} minutes remaining.".format(
            arrow + spaces, int(round(percentage * 100)), remaining_time))
    sys.stdout.flush()
