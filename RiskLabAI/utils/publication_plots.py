"""
Utilities for creating publication-quality plots with Matplotlib
and Seaborn, using Times New Roman font and high DPI.
"""

import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, List

def setup_publication_style() -> None:
    """
    Sets the global Matplotlib rcParams for a consistent,
    publication-quality (Times New Roman, 300 DPI) style.
    
    Call this function once at the beginning of your notebook.
    """
    
    # Check if Times New Roman is available
    try:
        plt.rc('font', family='Times New Roman')
    except:
        print("Warning: Times New Roman not found. Defaulting to serif.")
        plt.rc('font', family='serif')

    params = {
        'font.size': 12,
        'axes.labelsize': 12,
        'axes.titlesize': 14,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12,
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'savefig.transparent': True,
        'axes.grid': True,
        'grid.linestyle': '--',
        'grid.alpha': 0.5,
        'axes.edgecolor': 'black',
        'axes.linewidth': 1.2,
    }
    
    plt.rcParams.update(params)
    sns.set_style("whitegrid", params)
    print("Matplotlib style updated for publication.")


def apply_plot_style(
    ax: plt.Axes,
    title: str,
    xlabel: str,
    ylabel: str,
    legend_title: Optional[str] = None
) -> None:
    """
    Applies the standardized style to a specific Matplotlib Axes object.

    Parameters
    ----------
    ax : plt.Axes
        The Matplotlib axes to style.
    title : str
        The title for the plot.
    xlabel : str
        The label for the x-axis.
    ylabel : str
        The label for the y-axis.
    legend_title : str, optional
        Title for the legend, if any.
    """
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    
    if ax.get_legend():
        ax.legend(title=legend_title, fontsize=12)