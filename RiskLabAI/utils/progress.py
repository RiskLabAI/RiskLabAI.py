# import sys
# import time
# from typing import NoReturn

# """
#     Display a progress bar for a task with a given completion percentage and remaining time.

#     Args:
#         current_progress (int): Current progress value of the task.
#         total_progress (int): Total value of the task, representing 100% completion.
#         start_time (float): Start time of the task (typically from time.time()).
#         bar_length (int, optional): Length of the progress bar. Default is 20.
# """
# def progress_bar(
#     current_progress: int,
#     total_progress: int,
#     start_time: float,
#     bar_length: int = 20
# ) -> NoReturn:
    
#     percentage = float(current_progress) / total_progress
#     arrow = '-' * int(round(percentage * bar_length) - 1) + '>'
#     spaces = ' ' * (bar_length - len(arrow))

#     elapsed_time = time.time() - start_time
#     remaining_time = int(((elapsed_time / current_progress) * (total_progress - current_progress)) / 60)

#     sys.stdout.write("\rCompleted: [{0}] {1}% - {2} minutes remaining.".format(
#                      arrow + spaces, int(round(percentage * 100)), remaining_time))
#     sys.stdout.flush()



import sys
import time
from typing import NoReturn

def progress_bar(
    current_progress: int,
    total_progress: int,
    start_time: float,
    bar_length: int = 20
) -> NoReturn:
    """
    Display a terminal-style progress bar with completion percentage and remaining time.

    Args:
    - current_progress (int): Current value indicating the progress made.
    - total_progress (int): Total value representing the completion of the task.
    - start_time (float): The time at which the task started, typically acquired via time.time().
    - bar_length (int, optional): Length of the progress bar in terminal characters. Default is 20.

    Example:
    >>> start = time.time()
    >>> for i in range(101):
    ...     time.sleep(0.1)
    ...     progress_bar(i, 100, start)
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