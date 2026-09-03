"""Hardware Acceleration & System Status Badge (EmoPyLab 2026).

Renders an Apple/Linear-style pulsing status pill in the bottom status bar indicating:
- ⚡ JAX Acceleration (Metal / CUDA / TPU / CPU Vectorized)
- 🍏 Apple MLX Neural Engine
- 🖥️ CPU Multi-Core Engine
"""

from __future__ import annotations

from typing import Any

from PySide6.QtCore import QPoint, QRectF, QSize, Qt
from PySide6.QtGui import QBrush, QColor, QFont, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import QHBoxLayout, QLabel, QWidget

from core.ui.icons import get_icon


class HardwareStatusBadge(QWidget):
    """Pill-shaped badge indicating hardware execution backend."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("hardwareStatusBadge")
        self.setFixedHeight(24)

        self._backend = "cpu"
        self._device = "CPU (NumPy)"
        self._is_accelerated = False

        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 2, 8, 2)
        layout.setSpacing(6)

        self.dot_label = QLabel(self)
        self.dot_label.setFixedSize(8, 8)
        self.dot_label.setStyleSheet("border-radius: 4px; background-color: #10B981;")
        layout.addWidget(self.dot_label)

        self.text_label = QLabel(self)
        self.text_label.setStyleSheet("font-size: 11px; font-weight: 600; color: #E2E8F0; background: transparent;")
        layout.addWidget(self.text_label)

        self.set_backend("auto")

    def set_backend(self, backend: str, device_label: str | None = None) -> None:
        backend_clean = str(backend).strip().lower()
        if backend_clean in {"jax", "gpu"}:
            self._backend = "jax"
            self._device = device_label or "⚡ JAX Vectorized"
            self._is_accelerated = True
            dot_color = "#10B981"  # Neon Emerald
            badge_bg = "rgba(16, 185, 129, 0.15)"
            badge_border = "rgba(16, 185, 129, 0.4)"
        elif backend_clean in {"mlx", "apple"}:
            self._backend = "mlx"
            self._device = device_label or "🍏 Apple MLX Engine"
            self._is_accelerated = True
            dot_color = "#06B6D4"  # Cyan
            badge_bg = "rgba(6, 182, 212, 0.15)"
            badge_border = "rgba(6, 182, 212, 0.4)"
        else:
            self._backend = "cpu"
            self._device = device_label or "🖥️ CPU Core Engine"
            self._is_accelerated = False
            dot_color = "#F59E0B"  # Amber Gold
            badge_bg = "rgba(245, 158, 11, 0.15)"
            badge_border = "rgba(245, 158, 11, 0.4)"

        self.dot_label.setStyleSheet(f"border-radius: 4px; background-color: {dot_color};")
        self.text_label.setText(self._device)
        self.setStyleSheet(f"""
            #hardwareStatusBadge {{
                background-color: {badge_bg};
                border: 1px solid {badge_border};
                border-radius: 12px;
            }}
        """)
        self.adjustSize()
