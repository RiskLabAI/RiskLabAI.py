"""
Utilities for creating publication-quality plots with Matplotlib
and Seaborn, using Times New Roman font and high DPI.

Provides 6 themes and a configuration-based saving function.
"""

import matplotlib.pyplot as plt
import matplotlib.figure as fig  # For type hinting
import seaborn as sns
import os
from typing import Optional, Dict, Any

# [THEMES dictionary remains the same]
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
        'figure.facecolor': '#B0B0B0',  # A more solid, medium grey
        'axes.facecolor': '#B0B0B0',
        'text.color': '#FFFFFF',         # White text (like the dark theme)
        'axes.labelcolor': '#FFFFFF',
        'axes.edgecolor': '#FFFFFF',
        'xtick.color': '#FFFFFF',
        'ytick.color': '#FFFFFF',
        'grid.color': '#E0E0E0',         # Lighter grid lines on medium bg
        'legend.facecolor': '#B0B0B0',
        'legend.edgecolor': '#FFFFFF',
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

# --- MODULE-LEVEL CONFIGURATION ---
# This dictionary will store the settings from setup_publication_style
_CONFIG = {
    'save_plots': False,
    'save_dir': 'figs'
}

# --- UPDATED FUNCTION ---
def setup_publication_style(
    theme: str = 'light',
    quality: int = 300,
    save_plots: bool = False,  # <-- New parameter
    save_dir: str = 'figs'       # <-- New parameter
) -> None:
    """
    Sets the global Matplotlib rcParams and saving configuration.
    
    Call this function once at the beginning of your notebook.

    Parameters
    ----------
    theme : str, optional
        The theme to apply. Defaults to 'light'.
    quality : int, optional
        The DPI for the figures. Defaults to 300.
    save_plots : bool, optional
        Global switch to enable/disable saving plots. Defaults to False.
    save_dir : str, optional
        The directory to save figures in. Defaults to 'figs'.
    """
    
    # [All the theme parsing and styling code remains the same]
    # ... (omitted for brevity) ...
    is_transparent = False
    base_theme_name = theme
    if theme.endswith('-transparent'):
        is_transparent = True
        base_theme_name = theme.replace('-transparent', '')
    if base_theme_name not in THEMES:
        base_theme_name = 'light'
    params = THEMES[base_theme_name].copy()
    common_params = {
        'font.size': 12, 'axes.labelsize': 12, 'axes.titlesize': 14,
        'axes.titleweight': 'bold', 'xtick.labelsize': 12, 'ytick.labelsize': 12,
        'legend.fontsize': 12, 'legend.title_fontsize': 13,
        'figure.dpi': quality, 'savefig.dpi': quality, 'axes.grid': True,
        'grid.linestyle': '--', 'grid.alpha': 0.7, 'axes.linewidth': 1.2,
    }
    params.update(common_params)
    if is_transparent:
        params['figure.facecolor'] = (0, 0, 0, 0)
        params['axes.facecolor'] = (0, 0, 0, 0)
        params['savefig.transparent'] = True
        params['legend.facecolor'] = (0, 0, 0, 0) 
    else:
        params['savefig.transparent'] = False
    try:
        plt.rc('font', family='Times New Roman')
    except:
        print("Warning: Times New Roman not found. Defaulting to serif.")
        plt.rc('font', family='serif')
    plt.rcParams.update(params)
    sns_style = "darkgrid" if base_theme_name == 'dark' else "whitegrid"
    sns.set_style(sns_style, rc=params)
    
    # --- Store saving configuration ---
    _CONFIG['save_plots'] = save_plots
    _CONFIG['save_dir'] = save_dir
    
    print(f"Matplotlib style updated. Theme: '{theme}', Quality: {quality} DPI.")
    if save_plots:
        print(f"Plot saving enabled. Saving to: '{save_dir}'")
    else:
        print("Plot saving disabled.")

# [apply_plot_style function remains exactly the same]
def apply_plot_style(
    ax: plt.Axes,
    title: str,
    xlabel: str,
    ylabel: str,
    legend_title: Optional[str] = None
) -> None:
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if ax.get_legend() and legend_title is not None:
        ax.legend(title=legend_title)

# --- UPDATED FUNCTION ---
def finalize_plot(
    fig: fig.Figure,
    filename: str
) -> None:
    """
    Shows the plot and saves it *if* saving was enabled in
    setup_publication_style.

    Parameters
    ----------
    fig : plt.Figure
        The figure object to save.
    filename : str
        The name of the file (e.g., 'model_performance.png').
        This is required, but only used if saving is enabled.
    """
    
    # --- 1. Save the figure if global switch is on ---
    if _CONFIG['save_plots']:
        save_dir = _CONFIG['save_dir']
        
        # Create the directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)
        
        # Construct the full path
        full_path = os.path.join(save_dir, filename)
        
        # Save the figure
        fig.savefig(full_path, bbox_inches='tight')
        
        print(f"Figure saved to: {full_path}")
    
    # --- 2. Always show the plot ---
    plt.show()
    
    # --- 3. Close the figure object ---
    plt.close(fig)