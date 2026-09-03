"""Scientific Empty State & Welcome Canvas (EmoPyLab 2026).

Renders a beautiful placeholder with geometric manifolds, clear call-to-actions,
and keyboard shortcuts when no experiment data has been loaded yet.
"""

from __future__ import annotations

from typing import Callable, Optional

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QFont
from PySide6.QtWidgets import (
    QGraphicsDropShadowEffect,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from core.ui.icons import get_icon
from core.ui.styles import AppStyles


class ScientificEmptyState(QWidget):
    """Modern illustrated empty state container for analytical workspaces."""

    def __init__(
        self,
        *,
        title: str = "No Optimization Campaign Loaded",
        description: str = "Launch a single-run test diagnosis or execute a multi-algorithm benchmark campaign to visualize Pareto fronts, convergence trajectories, and MCDM trade-offs.",
        action_text: str = "Start Test Diagnosis",
        action_callback: Optional[Callable[[], None]] = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("scientificEmptyState")

        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.setSpacing(16)
        layout.setContentsMargins(32, 48, 32, 48)

        # 1. Icon Container
        self.icon_label = QLabel(self)
        icon = get_icon("analytics", color="#F59E0B")
        if icon and not icon.isNull():
            self.icon_label.setPixmap(icon.pixmap(56, 56))
        self.icon_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.icon_label)

        # 2. Title
        self.title_label = QLabel(title, self)
        self.title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.title_label.setStyleSheet("font-size: 18px; font-weight: 700; color: #F8FAFC;")
        layout.addWidget(self.title_label)

        # 3. Description
        self.desc_label = QLabel(description, self)
        self.desc_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.desc_label.setWordWrap(True)
        self.desc_label.setMaximumWidth(520)
        self.desc_label.setStyleSheet("font-size: 13px; color: #94A3B8; line-height: 1.4;")
        layout.addWidget(self.desc_label)

        # 4. Action Button
        if action_text and action_callback:
            self.action_btn = QPushButton(action_text, self)
            btn_icon = get_icon("play", color="#FFFFFF")
            if btn_icon:
                self.action_btn.setIcon(btn_icon)
            self.action_btn.setCursor(Qt.CursorShape.PointingHandCursor)
            self.action_btn.setFixedHeight(34)
            self.action_btn.setStyleSheet("""
                QPushButton {
                    background-color: #B45309;
                    color: #FFFFFF;
                    font-weight: 600;
                    font-size: 13px;
                    border-radius: 7px;
                    padding: 0 18px;
                    border: none;
                }
                QPushButton:hover {
                    background-color: #D97706;
                }
                QPushButton:pressed {
                    background-color: #92400E;
                }
            """)
            self.action_btn.clicked.connect(action_callback)

            btn_layout = QHBoxLayout()
            btn_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
            btn_layout.addWidget(self.action_btn)
            layout.addLayout(btn_layout)
