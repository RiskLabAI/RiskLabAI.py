import plotly.graph_objects as go

def update_figure_layout(fig, title, xaxis_title, yaxis_title, legend_x=1, legend_y=1):
    fig.update_layout(
        title=title,
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title,
        template="plotly_dark",
        plot_bgcolor='rgba(0,0,0,0)',  # Setting transparent background
        paper_bgcolor='rgba(0,0,0,0)', # Setting transparent background
        legend=dict(
            x=legend_x,       # X position of the legend
            y=legend_y,       # Y position of the legend
            xanchor='auto',   # Anchor point of the legend
            yanchor='auto'    # Anchor point of the legend
        )
    )
