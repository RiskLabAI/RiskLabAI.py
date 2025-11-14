"""
Utilities for creating publication-quality plots with Matplotlib
and Seaborn, using Times New Roman font and high DPI.

This module provides 6 themes:
- 'light'
- 'medium'
- 'dark'
- 'light-transparent'
- 'medium-transparent'
- 'dark-transparent'
"""

import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, Dict, Any

# Define the color palettes for the base themes
THEMES: Dict[str, Dict[str, Any]] = {
    'light': {
        'figure.facecolor': '#FFFFFF',
        'axes.facecolor': '#FFFFFF',
        'text.color': '#000000',
        'axes.labelcolor': '#000000',
        'axes.edgecolor': '#000000',
        'xtick.color': '#000000',
        'ytick.color': '#000000',
        'grid.color': '#CCCCCC',
        'legend.facecolor': '#FFFFFF',
        'legend.edgecolor': '#B0B0B0',
    },
    'medium': {
        'figure.facecolor': '#E5E5E5',
        'axes.facecolor': '#E5E5E5',
        'text.color': '#000000',
        'axes.labelcolor': '#000000',
        'axes.edgecolor': '#000000',
        'xtick.color': '#000000',
        'ytick.color': '#000000',
        'grid.color': '#B0B0B0',
        'legend.facecolor': '#E5E5E5',
        'legend.edgecolor': '#B0B0B0',
    },
    'dark': {
        'figure.facecolor': '#2E2E2E',
        'axes.facecolor': '#2E2E2E',
        'text.color': '#F0F0F0',
        'axes.labelcolor': '#F0F0F0',
        'axes.edgecolor': '#F0F0F0',
        'xtick.color': '#F0F0F0',
        'ytick.color': '#F0F0F0',
        'grid.color': '#6A6A6A',
        'legend.facecolor': '#2E2E2E',
        'legend.edgecolor': '#F0F0F0',
    }
}

def setup_publication_style(
    theme: str = 'light',
    quality: int = 300
) -> None:
    """
    Sets the global Matplotlib rcParams for a consistent,
    publication-quality style based on a theme.
    
    Call this function once at the beginning of your notebook.

    Parameters
    ----------
    theme : str, optional
        The theme to apply. One of: 'light', 'medium', 'dark',
        'light-transparent', 'medium-transparent', 'dark-transparent'.
        Defaults to 'light'.
    quality : int, optional
        The DPI (dots per inch) for the figures. Defaults to 300.
    """
    
    # --- 1. Parse the theme string ---
    is_transparent = False
    base_theme_name = theme
    
    if theme.endswith('-transparent'):
        is_transparent = True
        base_theme_name = theme.replace('-transparent', '')

    if base_theme_name not in THEMES:
        print(f"Warning: Base theme '{base_theme_name}' not recognized. Defaulting to 'light'.")
        base_theme_name = 'light'
        
    # --- 2. Get base theme parameters ---
    try:
        params = THEMES[base_theme_name].copy()
    except KeyError:
        print(f"Warning: Theme '{theme}' not found. Defaulting to 'light'.")
        params = THEMES['light'].copy()

    # --- 3. Add common parameters ---
    common_params = {
        'font.size': 12,
        'axes.labelsize': 12,
        'axes.titlesize': 14,
        'axes.titleweight': 'bold',
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12,
        'legend.title_fontsize': 13,
        'figure.dpi': quality,
        'savefig.dpi': quality,
        'axes.grid': True,
        'grid.linestyle': '--',
        'grid.alpha': 0.7,
        'axes.linewidth': 1.2,
    }
    params.update(common_params)

    # --- 4. Handle transparency override ---
    if is_transparent:
        # Use RGBA tuple for full transparency
        params['figure.facecolor'] = (0, 0, 0, 0)
        params['axes.facecolor'] = (0, 0, 0, 0)
        params['savefig.transparent'] = True
        # Ensure legend is also transparent
        params['legend.facecolor'] = (0, 0, 0, 0) 
    else:
        params['savefig.transparent'] = False

    # --- 5. Set the font ---
    try:
        plt.rc('font', family='Times New Roman')
    except:
        print("Warning: Times New Roman not found. Defaulting to serif.")
        plt.rc('font', family='serif')
        
    # --- 6. Apply all parameters ---
    plt.rcParams.update(params)
    
    # --- 7. Set Seaborn style ---
    # Use the appropriate base style for Seaborn
    sns_style = "darkgrid" if base_theme_name == 'dark' else "whitegrid"
    sns.set_style(sns_style, rc=params) 
    
    print(f"Matplotlib style updated. Theme: '{theme}', Quality: {quality} DPI.")


def apply_plot_style(
    ax: plt.Axes,
    title: str,
    xlabel: str,
    ylabel: str,
    legend_title: Optional[str] = None
) -> None:
    """
    Applies standardized labels and legend to a specific Matplotlib Axes object,
    respecting the globally set rcParams.

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
    # Set labels, allowing rcParams to control font size and weight
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    
    # Update legend title if one exists and is provided
    if ax.get_legend() and legend_title is not None:
        ax.legend(title=legend_title)