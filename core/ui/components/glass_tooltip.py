"""Glassmorphic Tooltip and Floating Card Overlay for Scientific Plotting (EmoPyLab 2026).

Provides an elegant, translucent, frosted-glass HUD card that renders
solution details (objective coordinates, constraint violations, decision variables)
with subpixel anti-aliasing and theme-aware contrast.
"""

from __future__ import annotations

from typing import Any, Sequence
import numpy as np

from PySide6.QtCore import QPoint, QRectF, QSize, Qt
from PySide6.QtGui import QBrush, QColor, QFont, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import QGraphicsDropShadowEffect, QLabel, QVBoxLayout, QWidget

from core.ui.styles import AppStyles


class GlassmorphismTooltip(QWidget):
    """Floating HUD tooltip with modern glassmorphism styling for scatter plots and Pareto fronts."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowFlags(Qt.WindowType.ToolTip | Qt.WindowType.FramelessWindowHint)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating, True)

        self._title = "Pareto Solution"
        self._objectives: list[float] = []
        self._rank: int | None = None
        self._cv: float | None = None
        self._score: float | None = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(4)

        self.label = QLabel(self)
        self.label.setTextFormat(Qt.TextFormat.RichText)
        self.label.setStyleSheet("color: #F8FAFC; background: transparent;")
        layout.addWidget(self.label)

        # Drop shadow for elevation depth
        shadow = QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(16)
        shadow.setColor(QColor(0, 0, 0, 120))
        shadow.setOffset(0, 4)
        self.setGraphicsEffect(shadow)

    def set_solution_data(
        self,
        *,
        title: str = "Pareto Point",
        objectives: Sequence[float],
        rank: int | None = None,
        score: float | None = None,
        cv: float | None = None,
    ) -> None:
        self._title = title
        self._objectives = list(objectives)
        self._rank = rank
        self._score = score
        self._cv = cv

        # Build Rich Text HTML
        lines = [f"<b style='color: #F59E0B; font-size: 13px;'>{title}</b>"]
        if rank is not None:
            lines.append(f"<span style='color: #94A3B8;'>Front Rank:</span> <b>{rank}</b>")
        if score is not None:
            lines.append(f"<span style='color: #10B981;'>MCDM Score:</span> <b>{score:.4f}</b>")
        if cv is not None and cv > 0:
            lines.append(f"<span style='color: #EF4444;'>Constraint Violation:</span> <b>{cv:.4e}</b>")

        obj_str = ", ".join([f"f<sub>{i+1}</sub>={v:.4f}" for i, v in enumerate(self._objectives[:6])])
        lines.append(f"<div style='margin-top: 4px; font-family: monospace; color: #E2E8F0;'>{obj_str}</div>")

        self.label.setText("<br>".join(lines))
        self.adjustSize()

    def paintEvent(self, event: Any) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        rect = QRectF(self.rect()).adjusted(1, 1, -1, -1)
        path = QPainterPath()
        path.addRoundedRect(rect, 8.0, 8.0)

        # Frosted glass obsidian background with subtle gradient
        bg_brush = QBrush(QColor(17, 24, 39, 230))
        painter.fillPath(path, bg_brush)

        # Translucent border highlight
        border_pen = QPen(QColor(245, 158, 11, 140), 1.0)
        painter.strokePath(path, border_pen)
