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
    Display a terminal-style progress bar with completion percentage and estimated remaining time.

    :param current_progress: Current value indicating the progress made.
    :param total_progress: Total value representing the completion of the task.
    :param start_time: The time at which the task started, typically acquired via time.time().
    :param bar_length: Length of the progress bar in terminal characters, default is 20.
    
    The displayed progress bar uses the formula:

    .. math::
        \text{percentage} = \frac{\text{current\_progress}}{\text{total\_progress}}

    The estimated remaining time is calculated based on elapsed time and progress made:

    .. math::
        \text{remaining\_time} = \frac{\text{elapsed\_time} \times (\text{total\_progress} - \text{current\_progress})}{\text{current\_progress}}

    :return: None
    """

    percentage = current_progress / total_progress
    arrow = '-' * int(round(percentage * bar_length) - 1) + '>'
    spaces = ' ' * (bar_length - len(arrow))

    elapsed_time = time.time() - start_time
    
    if current_progress == 0:
        remaining_time = "Calculating..."
    else:
        remaining_time = int(
            (elapsed_time / current_progress) * (total_progress - current_progress) / 60
        )

    if current_progress == total_progress:
        sys.stdout.write("\rCompleted: [{0}] 100% - Task completed!\n".format('-' * bar_length))
    
    else:
        sys.stdout.write("\rCompleted: [{0}] {1}% - {2} minutes remaining.".format(
            arrow + spaces, int(round(percentage * 100)), remaining_time))
    
    sys.stdout.flush()
