"""Navigation and tab bar components for EmoPyLab."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from PySide6.QtCore import QPoint, QRect, QSize, Qt
from PySide6.QtGui import QColor, QFont, QIcon, QPainter, QPixmap
from PySide6.QtWidgets import QSizePolicy, QTabBar, QWidget

from core.ui.styles import AppStyles


class PrimaryWorkflowTabBar(QTabBar):
    """Top navigation tab bar with icon-over-text tabs."""

    _minimum_brand_width = 190

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("primaryWorkflowTabBar")
        self.setDrawBase(False)
        self.setDocumentMode(True)
        self.setExpanding(False)
        self.setElideMode(Qt.TextElideMode.ElideNone)
        self.setIconSize(QSize(22, 22))
        self.setUsesScrollButtons(True)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setMinimumHeight(64)

        root_dir = Path(__file__).resolve().parent.parent.parent.parent
        logo_path = root_dir / "emopylab.png"
        self._brand_pixmap = self._trim_transparent_padding(
            QPixmap(str(logo_path))
        )

    @staticmethod
    def _trim_transparent_padding(pixmap: QPixmap) -> QPixmap:
        if pixmap.isNull():
            return pixmap
        image = pixmap.toImage()
        if image.isNull() or not image.hasAlphaChannel():
            return pixmap

        width = image.width()
        height = image.height()
        min_x, min_y = width, height
        max_x, max_y = -1, -1
        for y in range(height):
            for x in range(width):
                if image.pixelColor(x, y).alpha() > 8:
                    if x < min_x: min_x = x
                    if x > max_x: max_x = x
                    if y < min_y: min_y = y
                    if y > max_y: max_y = y

        if min_x > max_x or min_y > max_y:
            return pixmap
        return pixmap.copy(QRect(min_x, min_y, max_x - min_x + 1, max_y - min_y + 1))

    def tabSizeHint(self, index: int) -> QSize:
        text = self.tabText(index)
        font_metrics = self.fontMetrics()
        text_width = font_metrics.horizontalAdvance(text) if text else 0
        width = max(96, text_width + 32)
        return QSize(width, 60)
