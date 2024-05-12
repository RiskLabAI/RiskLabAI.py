import plotly.graph_objects as go

def update_figure_layout(fig, title, xaxis_title, yaxis_title):
    fig.update_layout(
        title=title,
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title,
        template="plotly_dark",
        plot_bgcolor='rgba(0,0,0,0)',  # Setting transparent background
        paper_bgcolor='rgba(0,0,0,0)', # Setting transparent background
        legend=dict(
            x=1,           # X position of the legend (1 is at the far right of the plot)
            y=1,           # Y position of the legend (1 is at the top of the plot)
            xanchor='auto',  # Anchor point of the legend
            yanchor='auto'   # Anchor point of the legend
        )
    )
