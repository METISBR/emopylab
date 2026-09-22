"""
Plotly WebGL visualization bridge for PySide6.
Integrates Plotly.js seamlessly into Qt using QWebEngineView and QWebChannel.
"""
from __future__ import annotations

import json
import math
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
from PySide6.QtCore import QObject, QUrl, Signal, Slot
from PySide6.QtWebChannel import QWebChannel
from PySide6.QtWebEngineWidgets import QWebEngineView
from PySide6.QtWidgets import QWidget

import plotly.graph_objects as go
import plotly.offline

_CACHED_HTML_PATH: Path | None = None


def _get_or_create_plotly_html() -> Path:
    """Create or return cached standalone HTML template containing offline Plotly.js and WebChannel."""
    global _CACHED_HTML_PATH
    if _CACHED_HTML_PATH is not None and _CACHED_HTML_PATH.exists():
        return _CACHED_HTML_PATH

    plotly_js = plotly.offline.get_plotlyjs()
    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8" />
    <script src="qrc:///qtwebchannel/qwebchannel.js"></script>
    <script type="text/javascript">
{plotly_js}
    </script>
    <style>
        body, html {{
            margin: 0;
            padding: 0;
            width: 100vw;
            height: 100vh;
            overflow: hidden;
            background-color: transparent;
        }}
        #chart {{
            width: 100%;
            height: 100%;
        }}
    </style>
</head>
<body>
    <div id="chart"></div>
    <script type="text/javascript">
        window.bridge = null;
        window.plotly_config = {{
            responsive: true,
            displayModeBar: true,
            displaylogo: false,
            modeBarButtonsToRemove: ['sendDataToCloud', 'hoverClosestCartesian', 'hoverCompareCartesian']
        }};

        new QWebChannel(qt.webChannelTransport, function (channel) {{
            window.bridge = channel.objects.bridge;
        }});

        function update_figure(fig_json) {{
            try {{
                var fig = JSON.parse(fig_json);
                var chartDiv = document.getElementById('chart');
                var data = fig.data || [];
                var layout = fig.layout || {{}};
                Plotly.react(chartDiv, data, layout, window.plotly_config).then(function() {{
                    if (!chartDiv._has_click_handler) {{
                        chartDiv._has_click_handler = true;
                        chartDiv.on('plotly_click', function(data) {{
                            if (window.bridge && data.points && data.points.length > 0) {{
                                var pt = data.points[0];
                                window.bridge.on_point_clicked(pt.pointNumber !== undefined ? pt.pointNumber : pt.pointIndex);
                            }}
                        }});
                    }}
                }});
            }} catch(e) {{
                console.error("Plotly.react error:", e);
            }}
        }}

        function set_empty_message(msg, bg_color, text_color) {{
            try {{
                var layout = {{
                    paper_bgcolor: bg_color || '#1E293B',
                    plot_bgcolor: bg_color || '#1E293B',
                    xaxis: {{ visible: false }},
                    yaxis: {{ visible: false }},
                    annotations: [{{
                        text: msg,
                        xref: 'paper',
                        yref: 'paper',
                        showarrow: false,
                        font: {{ size: 14, color: text_color || '#94A3B8' }}
                    }}]
                }};
                Plotly.react('chart', [], layout, window.plotly_config);
            }} catch(e) {{
                console.error("set_empty_message error:", e);
            }}
        }}
    </script>
</body>
</html>
"""
    tmp = tempfile.NamedTemporaryFile("w", suffix=".html", delete=False, encoding="utf-8")
    tmp.write(html_content)
    tmp.close()
    _CACHED_HTML_PATH = Path(tmp.name)
    return _CACHED_HTML_PATH


class PlotlyBridge(QObject):
    """Bridge object communicating between JavaScript and Python."""
    point_clicked = Signal(int)

    @Slot(int)
    def on_point_clicked(self, point_index: int) -> None:
        self.point_clicked.emit(point_index)


class PlotlyWidget(QWebEngineView):
    """
    Unified high-performance 2D/3D Plotly WebGL chart widget.
    Drop-in replacement for QChartView, Q3DScatter, and MplCanvas.
    """
    point_clicked = Signal(int)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._is_ready = False
        self._pending_figure: str | None = None

        # Setup WebChannel
        self._bridge = PlotlyBridge(self)
        self._bridge.point_clicked.connect(self.point_clicked.emit)

        self._channel = QWebChannel(self)
        self._channel.registerObject("bridge", self._bridge)
        self.page().setWebChannel(self._channel)

        # Load offline template
        html_path = _get_or_create_plotly_html()
        self.loadFinished.connect(self._on_load_finished)
        self.setUrl(QUrl.fromLocalFile(str(html_path)))

    def _on_load_finished(self, ok: bool) -> None:
        self._is_ready = ok
        if ok and self._pending_figure is not None:
            fig_json = self._pending_figure
            self._pending_figure = None
            self._execute_update(fig_json)

    def set_figure(self, fig: go.Figure) -> None:
        """Render or update a Plotly Figure via WebGL (hardware accelerated)."""
        fig_json = fig.to_json()

        if not self._is_ready:
            self._pending_figure = fig_json
            return

        self._execute_update(fig_json)

    def _execute_update(self, fig_json: str) -> None:
        self.page().runJavaScript(f"update_figure({json.dumps(fig_json)});")

    def set_empty(self, message: str, bg_color: str = "#1E293B", text_color: str = "#94A3B8") -> None:
        """Show an empty placeholder message."""
        if not self._is_ready:
            return
        self.page().runJavaScript(
            f"set_empty_message({json.dumps(message)}, {json.dumps(bg_color)}, {json.dumps(text_color)});"
        )


# --- High-level figure builders ---

def make_convergence_figure(
    algo_name: str,
    run_label: str,
    run_timestamp: str,
    metric_name: str,
    x_history: np.ndarray,
    history: np.ndarray,
    bg_color: str = "#1E293B",
    text_color: str = "#F8FAFC",
    line_color: str = "#38BDF8",
    grid_color: str = "#334155",
) -> go.Figure:
    fig = go.Figure()
    valid_mask = np.isfinite(history)
    x_clean = x_history[valid_mask] if len(x_history) == len(history) else np.arange(1, np.sum(valid_mask) + 1)
    y_clean = history[valid_mask]

    fig.add_trace(go.Scattergl(
        x=x_clean,
        y=y_clean,
        mode="lines+markers",
        name=f"{algo_name} {run_label}",
        line=dict(color=line_color, width=2.5),
        marker=dict(size=4, color=line_color),
    ))

    fig.update_layout(
        title=dict(text=f"Convergence - {metric_name} - {algo_name} - {run_label} | {run_timestamp}", font=dict(size=12, color=text_color)),
        xaxis=dict(title="Function evaluations (FE)", color=text_color, gridcolor=grid_color, zerolinecolor=grid_color),
        yaxis=dict(title=metric_name or "Metric", color=text_color, gridcolor=grid_color, zerolinecolor=grid_color),
        paper_bgcolor=bg_color,
        plot_bgcolor=bg_color,
        font=dict(color=text_color, size=11),
        margin=dict(l=45, r=20, t=40, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(color=text_color)),
    )
    return fig


def make_multi_convergence_figure(
    series_data: list[dict[str, Any]],
    metric_name: str,
    title_suffix: str = "",
    bg_color: str = "#1E293B",
    text_color: str = "#F8FAFC",
    palette: tuple[str, ...] = ("#38BDF8", "#F43F5E", "#10B981", "#F59E0B", "#8B5CF6"),
    grid_color: str = "#334155",
) -> go.Figure:
    fig = go.Figure()
    for idx, item in enumerate(series_data):
        algo = item["algo"]
        x_vals = item["x"]
        y_vals = item["y"]
        color = palette[idx % len(palette)]
        fig.add_trace(go.Scattergl(
            x=x_vals,
            y=y_vals,
            mode="lines",
            name=algo,
            line=dict(color=color, width=2),
        ))

    title_text = f"Convergence - {metric_name}" + (f" - {title_suffix}" if title_suffix else "")
    fig.update_layout(
        title=dict(text=title_text, font=dict(size=12, color=text_color)),
        xaxis=dict(title="Function Evaluations (FE)", color=text_color, gridcolor=grid_color, zerolinecolor=grid_color),
        yaxis=dict(title=metric_name, color=text_color, gridcolor=grid_color, zerolinecolor=grid_color),
        paper_bgcolor=bg_color,
        plot_bgcolor=bg_color,
        font=dict(color=text_color, size=11),
        margin=dict(l=45, r=20, t=40, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(color=text_color)),
    )
    return fig


def make_pareto_figure(
    front: np.ndarray,
    problem_pf: np.ndarray | None,
    title: str,
    anchor: bool = False,
    axis_labels: tuple[str, ...] = ("f1", "f2", "f3"),
    selected_point: np.ndarray | None = None,
    point_label: str = "Obtained",
    bg_color: str = "#1E293B",
    surface_color: str = "#0F172A",
    text_color: str = "#F8FAFC",
    primary_color: str = "#38BDF8",
    pf_color: str = "#94A3B8",
    grid_color: str = "#334155",
) -> go.Figure:
    fig = go.Figure()
    front = np.atleast_2d(np.asarray(front, dtype=float))
    n_obj = front.shape[1]

    if n_obj == 2:
        # Reference Pareto Front
        if problem_pf is not None and problem_pf.ndim == 2 and problem_pf.shape[1] >= 2:
            pf_2d = problem_pf[:, :2]
            pf_2d = pf_2d[np.argsort(pf_2d[:, 0])]
            fig.add_trace(go.Scattergl(
                x=pf_2d[:, 0], y=pf_2d[:, 1],
                mode="lines",
                name="Reference PF",
                line=dict(color=pf_color, width=2),
                hoverinfo="skip"
            ))

        # Obtained front
        fig.add_trace(go.Scattergl(
            x=front[:, 0], y=front[:, 1],
            mode="markers",
            name=point_label,
            marker=dict(color=primary_color, size=7, line=dict(color="#0284C7", width=1)),
            hovertemplate=f"{axis_labels[0]}: %{{x:.4f}}<br>{axis_labels[1]}: %{{y:.4f}}<extra></extra>"
        ))

        # Selected MCDM point
        if selected_point is not None and len(selected_point) >= 2:
            fig.add_trace(go.Scattergl(
                x=[selected_point[0]], y=[selected_point[1]],
                mode="markers",
                name="MCDM selected",
                marker=dict(color="#DC2626", size=14, symbol="star", line=dict(color="#7F1D1D", width=1.5)),
                hovertemplate="<b>Selected Point</b><br>f1: %{x:.4f}<br>f2: %{y:.4f}<extra></extra>"
            ))

        xaxis_kwargs: dict[str, Any] = dict(title=axis_labels[0], color=text_color, gridcolor=grid_color, zerolinecolor=grid_color)
        yaxis_kwargs: dict[str, Any] = dict(title=axis_labels[1], color=text_color, gridcolor=grid_color, zerolinecolor=grid_color)
        if anchor:
            xaxis_kwargs["rangemode"] = "tozero"
            yaxis_kwargs["rangemode"] = "tozero"

        fig.update_layout(
            title=dict(text=title, font=dict(size=12, color=text_color)),
            xaxis=xaxis_kwargs,
            yaxis=yaxis_kwargs,
            paper_bgcolor=bg_color,
            plot_bgcolor=bg_color,
            font=dict(color=text_color, size=11),
            margin=dict(l=45, r=20, t=40, b=40),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(color=text_color)),
        )

    elif n_obj == 3:
        # Reference Pareto Front
        if problem_pf is not None and problem_pf.ndim == 2 and problem_pf.shape[1] >= 3:
            fig.add_trace(go.Scatter3d(
                x=problem_pf[:, 0], y=problem_pf[:, 1], z=problem_pf[:, 2],
                mode="markers",
                name="Reference PF",
                marker=dict(color=pf_color, size=3, opacity=0.35),
                hoverinfo="skip"
            ))

        # Obtained front
        fig.add_trace(go.Scatter3d(
            x=front[:, 0], y=front[:, 1], z=front[:, 2],
            mode="markers",
            name=point_label,
            marker=dict(color=primary_color, size=4.5, opacity=0.85, line=dict(color="#0284C7", width=0.5)),
            hovertemplate=f"{axis_labels[0]}: %{{x:.4f}}<br>{axis_labels[1]}: %{{y:.4f}}<br>{axis_labels[2]}: %{{z:.4f}}<extra></extra>"
        ))

        # Selected MCDM point
        if selected_point is not None and len(selected_point) >= 3:
            fig.add_trace(go.Scatter3d(
                x=[selected_point[0]], y=[selected_point[1]], z=[selected_point[2]],
                mode="markers",
                name="MCDM selected",
                marker=dict(color="#DC2626", size=10, symbol="diamond", line=dict(color="#7F1D1D", width=1.5)),
                hovertemplate="<b>Selected Point</b><br>f1: %{x:.4f}<br>f2: %{y:.4f}<br>f3: %{z:.4f}<extra></extra>"
            ))

        scene_dict: dict[str, Any] = dict(
            xaxis=dict(title=axis_labels[0], color=text_color, gridcolor=grid_color, backgroundcolor=surface_color),
            yaxis=dict(title=axis_labels[1], color=text_color, gridcolor=grid_color, backgroundcolor=surface_color),
            zaxis=dict(title=axis_labels[2], color=text_color, gridcolor=grid_color, backgroundcolor=surface_color),
        )
        if anchor:
            scene_dict["xaxis"]["rangemode"] = "tozero"
            scene_dict["yaxis"]["rangemode"] = "tozero"
            scene_dict["zaxis"]["rangemode"] = "tozero"

        fig.update_layout(
            title=dict(text=title, font=dict(size=12, color=text_color)),
            scene=scene_dict,
            paper_bgcolor=bg_color,
            font=dict(color=text_color, size=11),
            margin=dict(l=10, r=10, t=40, b=10),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(color=text_color)),
        )

    else:
        # Many-objective: Parallel Coordinates
        dims = []
        for i in range(n_obj):
            col_vals = front[:, i]
            col_min = float(np.min(col_vals)) if len(col_vals) > 0 else 0.0
            col_max = float(np.max(col_vals)) if len(col_vals) > 0 else 1.0
            if col_min == col_max:
                col_max = col_min + 1.0
            label = axis_labels[i] if i < len(axis_labels) else f"f{i+1}"
            dims.append(dict(range=[col_min, col_max], label=label, values=col_vals))

        fig.add_trace(go.Parcoords(
            line=dict(color=front[:, 0], colorscale="Viridis", showscale=True),
            dimensions=dims
        ))
        fig.update_layout(
            title=dict(text=title, font=dict(size=12, color=text_color)),
            paper_bgcolor=bg_color,
            font=dict(color=text_color, size=11),
            margin=dict(l=50, r=50, t=50, b=30),
        )

    return fig
