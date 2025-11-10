"""
Tests for utils/progress.py
"""

import pytest
import time
from RiskLabAI.utils.progress import progress_bar

def test_progress_bar_start(capsys):
    """Test the progress bar at the start (0%)."""
    progress_bar(0, 100, time.time())
    captured = capsys.readouterr()
    assert "Completed: [                    ] 0% - Calculating..." in captured.out

def test_progress_bar_mid(capsys):
    """Test the progress bar in the middle (50%)."""
    start_time = time.time() - 10 # Pretend 10s have passed
    progress_bar(50, 100, start_time)
    captured = capsys.readouterr()
    
    assert "Completed: [--------->" in captured.out
    assert "] 50% - " in captured.out
    assert "minutes remaining" in captured.out

def test_progress_bar_end(capsys):
    """Test the progress bar at the end (100%)."""
    progress_bar(100, 100, time.time())
    captured = capsys.readouterr()
    assert "Completed: [--------------------] 100% - Task completed!" in captured.out
    assert captured.out.endswith('\n') # Should print a newline