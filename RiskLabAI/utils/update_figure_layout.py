"""
A helper function to apply a consistent, dark-themed layout
to Plotly figures.
"""

import plotly.graph_objects as go
from typing import Optional

def update_figure_layout(
    fig: go.Figure,
    title: str,
    xaxis_title: str,
    yaxis_title: str,
    legend_x: float = 1.0,
    legend_y: float = 1.0,
) -> go.Figure:
    """
    Apply a standardized dark-theme layout to a Plotly figure.

    This function modifies the figure in-place.

    Parameters
    ----------
    fig : go.Figure
        The Plotly figure object to update.
    title : str
        The title for the chart.
    xaxis_title : str
        The title for the x-axis.
    yaxis_title : str
        The title for the y-axis.
    legend_x : float, default=1.0
        The x-position of the legend.
    legend_y : float, default=1.0
        The y-position of the legend.

    Returns
    -------
    go.Figure
        The modified figure.
    """
    fig.update_layout(
        title=title,
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title,
        template="plotly_dark",
        plot_bgcolor="rgba(0,0,0,0)",  # Transparent background
        paper_bgcolor="rgba(0,0,0,0)", # Transparent background
        legend=dict(
            x=legend_x,
            y=legend_y,
            xanchor="auto",
            yanchor="auto",
        ),
    )
    return fig