from __future__ import annotations
import numpy as np
import plotly.graph_objects as go
from typing import List, Optional


def pareto_3d_figure(F: np.ndarray, obj_names: List[str], title: str = "Pareto Front (3D)", highlight: Optional[np.ndarray] = None) -> go.Figure:
    if F is None or len(F) == 0:
        fig = go.Figure()
        fig.update_layout(title=title)
        return fig

    x, y, z = F[:, 0], F[:, 1], F[:, 2]
    data_traces = [
        go.Scatter3d(
            x=x,
            y=y,
            z=z,
            mode="markers",
            marker=dict(size=4, color=z, colorscale="Viridis", opacity=0.9),
            name="Solutions",
        )
    ]
    if highlight is not None and len(highlight) > 0:
        hx, hy, hz = highlight[:, 0], highlight[:, 1], highlight[:, 2]
        data_traces.append(
            go.Scatter3d(
                x=hx,
                y=hy,
                z=hz,
                mode="markers",
                marker=dict(size=7, color="red", opacity=1.0, symbol="diamond"),
                name="Top 5",
            )
        )
    fig = go.Figure(data=data_traces)
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title=obj_names[0],
            yaxis_title=obj_names[1],
            zaxis_title=obj_names[2],
        ),
        margin=dict(l=0, r=0, t=40, b=0),
        height=520,
    )
    return fig


def pareto_2d_figure(F: np.ndarray, i: int, j: int, obj_names: List[str], title: str | None = None, highlight: Optional[np.ndarray] = None) -> go.Figure:
    if F is None or len(F) == 0:
        fig = go.Figure()
        fig.update_layout(title=title or f"Pareto Front: {obj_names[i]} vs {obj_names[j]}")
        return fig

    data_traces = [
        go.Scatter(
            x=F[:, i],
            y=F[:, j],
            mode="markers",
            marker=dict(size=6, color=F[:, i], colorscale="Viridis", opacity=0.9),
            name="Solutions",
        )
    ]
    if highlight is not None and len(highlight) > 0:
        data_traces.append(
            go.Scatter(
                x=highlight[:, i],
                y=highlight[:, j],
                mode="markers",
                marker=dict(size=9, color="red", opacity=1.0, symbol="diamond"),
                name="Top 5",
            )
        )
    fig = go.Figure(data=data_traces)
    fig.update_layout(
        title=title or f"Pareto Front: {obj_names[i]} vs {obj_names[j]}",
        xaxis_title=obj_names[i],
        yaxis_title=obj_names[j],
        height=420,
        margin=dict(l=10, r=10, t=40, b=10),
    )
    return fig
