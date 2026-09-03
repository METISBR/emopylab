"""Unified QtAwesome Icon Subsystem for EmoPyLab.

Provides robust, HiDPI-crisp semantic icon resolution backed exclusively by
QtAwesome (FontAwesome 5/6 solid/regular, Material Design, and Phosphor icons).
Guarantees theme-aware contrast and graceful fallback.
"""

from __future__ import annotations

from typing import Optional

try:
    import qtawesome as qta
    from PySide6.QtGui import QIcon
    _HAS_QTAWESOME = True
except Exception:  # noqa: BLE001
    qta = None
    QIcon = None  # type: ignore[assignment, misc]
    _HAS_QTAWESOME = False


# Canonical Semantic Map -> QtAwesome Glyph identifiers
SEMANTIC_ICON_MAP: dict[str, str] = {
    # Navigation & Core Workspaces
    "science": "fa5s.vial",
    "biotech": "fa5s.flask",
    "analytics": "fa5s.chart-line",
    "smart_toy": "fa5s.robot",
    "code": "fa5s.code",
    "settings": "fa5s.cogs",
    "dashboard": "fa5s.tachometer-alt",
    "experiment": "fa5s.project-diagram",
    
    # Theme & Appearance
    "dark_mode": "fa5s.moon",
    "light_mode": "fa5s.sun",
    "palette": "fa5s.palette",
    
    # Execution & Control
    "play_arrow": "fa5s.play",
    "play": "fa5s.play",
    "stop": "fa5s.stop",
    "pause": "fa5s.pause",
    "restart_alt": "fa5s.redo-alt",
    "redo": "fa5s.redo-alt",
    "refresh": "fa5s.sync-alt",
    
    # Data & Storage
    "save": "fa5s.save",
    "folder_open": "fa5s.folder-open",
    "folder": "fa5s.folder",
    "delete": "fa5s.trash-alt",
    "trash": "fa5s.trash-alt",
    "remove": "fa5s.minus",
    "add": "fa5s.plus",
    "content_copy": "fa5s.copy",
    "copy": "fa5s.copy",
    "upload_file": "fa5s.file-upload",
    "download": "fa5s.file-download",
    "history": "fa5s.history",
    
    # Analysis, Metrics & Decision Making
    "table_view": "fa5s.table",
    "table_chart": "fa5s.table",
    "table": "fa5s.table",
    "fact_check": "fa5s.check-double",
    "description": "fa5s.file-alt",
    "file": "fa5s.file",
    "functions": "fa5s.square-root-alt",
    "math": "fa5s.calculator",
    "mcdm": "fa5s.balance-scale",
    "pareto": "fa5s.bezier-curve",
    "filter": "fa5s.filter",
    "sort": "fa5s.sort-amount-down",
    
    # Hardware & System
    "memory": "fa5s.microchip",
    "cpu": "fa5s.microchip",
    "gpu": "fa5s.server",
    "seed": "fa5s.seedling",
    "security": "fa5s.shield-alt",
    "lock": "fa5s.lock",
    "check": "fa5s.check",
    "cross": "fa5s.times",
    "warning": "fa5s.exclamation-triangle",
    "info": "fa5s.info-circle",
}

DEFAULT_FALLBACK_GLYPH = "fa5s.circle"


def get_icon(name: str, color: Optional[str] = None, active_color: Optional[str] = None) -> QIcon:
    """Resolve a semantic name or direct QtAwesome glyph to a QIcon.

    Args:
        name: Semantic name (e.g. 'science', 'play') or raw QtAwesome key (e.g. 'fa5s.vial').
        color: Primary color string (hex or named color).
        active_color: Optional color for the active/selected state.

    Returns:
        A QIcon instance styled with QtAwesome.
    """
    if not _HAS_QTAWESOME or qta is None or QIcon is None:
        return QIcon() if QIcon is not None else None  # type: ignore[return-value]

    glyph = SEMANTIC_ICON_MAP.get(name, name)
    kwargs: dict[str, Any] = {}
    if color:
        kwargs["color"] = color
    if active_color:
        kwargs["color_active"] = active_color

    try:
        return qta.icon(glyph, **kwargs)
    except Exception:
        try:
            return qta.icon(DEFAULT_FALLBACK_GLYPH, **kwargs)
        except Exception:
            return QIcon()


def get_navigation_icon(name: str, primary_color: str = "#FFB300", inactive_color: str = "#8E9AA8") -> QIcon:
    """Build a contrast-aware dual-state navigation tab icon.

    Args:
        name: Semantic name for the icon.
        primary_color: Color when tab is selected/active.
        inactive_color: Color when tab is unfocused.
    """
    if not _HAS_QTAWESOME or qta is None:
        return QIcon() if QIcon is not None else None  # type: ignore[return-value]

    glyph = SEMANTIC_ICON_MAP.get(name, name)
    try:
        return qta.icon(
            glyph,
            color=inactive_color,
            color_active=primary_color,
            color_selected=primary_color,
        )
    except Exception:
        return get_icon(DEFAULT_FALLBACK_GLYPH, color=primary_color)
