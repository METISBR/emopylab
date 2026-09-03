# Author: Professor Thiago Santos at UFOP, Brazil
from __future__ import annotations

import csv
import ast
import importlib
import importlib.util
import inspect
import json
import hashlib
import math
import os
import pkgutil
import re
import sys
import time
import types
import urllib.parse
import urllib.request
from datetime import datetime
from functools import lru_cache, partial
from pathlib import Path
from typing import Any, Callable, cast

import numpy as np


def to_numpy(value: Any) -> np.ndarray:
    """Convert array-like values to NumPy arrays (CPU-only runtime)."""
    getter = getattr(value, "get", None)
    if callable(getter):
        try:
            value = getter()
        except Exception:  # noqa: BLE001
            pass
    return np.asarray(value)

try:
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as MplCanvas
    from matplotlib.figure import Figure as MplFigure
    _HAS_MPL_3D = True
except ImportError:
    _HAS_MPL_3D = False
from PySide6.QtCharts import QChart, QChartView, QLineSeries, QScatterSeries, QValueAxis
from PySide6.QtCore import QObject, Qt, QThread, Signal, Slot, QPropertyAnimation, QEasingCurve, QParallelAnimationGroup, QTimer, QUrl, QRect, QSize
from PySide6.QtGui import QAction, QBrush, QColor, QFont, QIcon, QPainter, QPen, QPixmap, QDesktopServices
from PySide6.QtWidgets import (
    QApplication,
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QSplitter,
    QStackedWidget,
    QTableWidget,
    QTableWidgetItem,
    QTabBar,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

# Skill: qt_material must be imported AFTER PySide6
from qt_material import apply_stylesheet
import qtawesome as qta

# Skill: Import centralized styles from styles.py
from styles import AppStyles as StylesAppStyles
import emopylab_metadata
from core.execution.backend_runtime import (
    build_gpu_status_text as core_build_gpu_status_text,
    detect_gpu_runtime as core_detect_gpu_runtime,
    apple_silicon_available as core_apple_silicon_available,
    detect_mlx_runtime as core_detect_mlx_runtime,
    build_backend_status_text as core_build_backend_status_text,
)
from core.execution.qt_bridges import LLMTaskBridge
from core.llm.splash import (
    DEFAULT_BASE_URL as SPLASH_DEFAULT_BASE_URL,
    DEFAULT_MLX_PORT as SPLASH_DEFAULT_MLX_PORT,
    LocalLLMStartupCoordinator,
    base_url_for_port,
)
from core.analysis.stat_tests import (
    SCIPY_STATS_AVAILABLE as core_scipy_stats_available,
    SCIPY_STATS_ERROR as core_scipy_stats_error,
    run_friedman as core_run_friedman,
    run_wilcoxon as core_run_wilcoxon,
    summarize_stat_results as core_summarize_stat_results,
)
from core.execution.reproducibility import (
    build_execution_manifest as core_build_execution_manifest,
    build_seed_plan as core_build_seed_plan,
)
from core.llm.formulation import LLMFormulationService as CoreLLMFormulationService
from core.mcdm.decision import select_compromise_solution
from core.registry.backend_selection import (
    iter_operator_runtime_candidates as core_iter_operator_runtime_candidates,
    looks_like_jax_identifier as core_looks_like_jax_identifier,
    map_selected_ids_to_backend as core_map_selected_ids_to_backend,
    normalize_backend_token as core_normalize_backend_token,
    resolve_available_operator_target as core_resolve_available_operator_target,
    select_specs_for_backend as core_select_specs_for_backend,
    strip_jax_suffix as core_strip_jax_suffix,
)
from core.registry.rollout import (
    backend_aware_loading_enabled as core_backend_aware_loading_enabled,
    rollout_allows_domain as core_rollout_allows_domain,
    rollout_stage as core_rollout_stage,
)
from core.registry.plugin_dirs import ensure_plugin_directories as core_ensure_plugin_directories
from core.registry.specs import AlgorithmSpec, MetricSpec, OperatorSpec, ProblemSpec


DEFAULT_PROBLEM_DIMS: dict[str, tuple[int, int]] = {
    "zdt1": (30, 2),
    "zdt2": (30, 2),
    "zdt3": (30, 2),
    "zdt4": (10, 2),
    "zdt5": (11, 2),
    "zdt6": (10, 2),
    "dtlz1": (7, 3),
    "dtlz2": (10, 3),
    "dtlz3": (10, 3),
    "dtlz4": (10, 3),
    "dtlz5": (10, 3),
    "dtlz6": (10, 3),
    "dtlz7": (10, 3),
    "wfg1": (24, 3),
    "wfg2": (24, 3),
    "wfg3": (24, 3),
    "wfg4": (24, 3),
    "wfg5": (24, 3),
    "wfg6": (24, 3),
    "wfg7": (24, 3),
    "wfg8": (24, 3),
    "wfg9": (24, 3),
}

BACKEND_OPTIONS: dict[str, str] = {
    "cpu": "CPU (NumPy)",
    "gpu": "JAX",
    "mlx": "MLX (Apple Silicon)",
}

EMOPYLAB_CATALOG_API_TREE_URL = "https://api.github.com/repos/anyoptimization/pymoo/git/trees/main-recursive=1"
EMOPYLAB_CATALOG_RAW_BASE_URL = "https://raw.githubusercontent.com/anyoptimization/pymoo/main/"
EMOPYLAB_CATALOG_USER_AGENT = "EmoPyLab-dynamic-registry/2.0"
PYMOO_GITHUB_API_TREE_URL = EMOPYLAB_CATALOG_API_TREE_URL
PYMOO_GITHUB_RAW_BASE_URL = EMOPYLAB_CATALOG_RAW_BASE_URL
PYMOO_GITHUB_USER_AGENT = EMOPYLAB_CATALOG_USER_AGENT

OPERATOR_CATEGORY_META: dict[str, tuple[str, str, str]] = {
    "crossover": ("operators.crossover", "core.crossover", "Crossover"),
    "mutation": ("operators.mutation", "core.mutation", "Mutation"),
    "selection": ("operators.selection", "core.selection", "Selection"),
    "sampling": ("operators.sampling", "core.sampling", "Sampling"),
}

LEGACY_OPERATOR_ALIASES: dict[str, dict[str, tuple[str, str]]] = {
    "crossover": {
        "sbx": ("operators.crossover.sbx", "SBX"),
        "pmx": ("operators.crossover.pmx", "PMX"),
        "ox": ("operators.crossover.ox", "OrderCrossover"),
        "ux": ("operators.crossover.ux", "UX"),
        "dex": ("operators.crossover.dex", "DEX"),
        "hux": ("operators.crossover.hux", "HUX"),
        "erx": ("operators.crossover.erx", "ERX"),
    },
    "mutation": {
        "pm": ("operators.mutation.pm", "PM"),
        "sbm": ("operators.mutation.bitflip", "BitflipMutation"),
        "bitflip": ("operators.mutation.bitflip", "BitflipMutation"),
        "gaussian": ("operators.mutation.gauss", "GaussianMutation"),
    },
    "selection": {
        "tournament": ("operators.selection.tournament", "TournamentSelection"),
        "random": ("operators.selection.rnd", "RandomSelection"),
        "best": ("operators.selection.tournament", "TournamentSelection"),
        "random_binary": ("operators.selection.tournament", "TournamentSelection"),
    },
    "sampling": {
        "float_random": ("operators.sampling.rnd", "FloatRandomSampling"),
        "int_random": ("operators.sampling.rnd", "IntegerRandomSampling"),
        "binary_random": ("operators.sampling.rnd", "BinaryRandomSampling"),
        "perm_random": ("operators.sampling.rnd", "PermutationRandomSampling"),
        "lhs": ("operators.sampling.lhs", "LHS"),
    },
}

DEFAULT_MAX_FE = 25_000
DEFAULT_EXPERIMENT_N_RUNS = 30
DEFAULT_EXPERIMENT_POP_SIZE = 100
EXP_CONFIG_FILENAME = "last_experiment_config.json"
EXPERIMENT_MANIFEST_VERSION = 1
SEED_MODE_RANDOM = "random"
SEED_MODE_FIXED = "fixed"
SEED_MODE_SEQUENCE = "sequence"
DEFAULT_LOCAL_LMSTUDIO_MODEL = "loaded-model"
DEFAULT_LOCAL_LMSTUDIO_CHAT_URL = "http://127.0.0.1:1234/v1/chat/completions"
LLM_LOCAL_MAX_PARALLEL_WORKERS = 3

OBJECTIVE_FILTER_VALUES = ("single", "multi", "many")
ENCODING_FILTER_VALUES = ("real", "integer", "mixed", "binary", "permutation")
DIFFICULTY_FILTER_VALUES = (
    "constrained",
    "dynamic",
)
ALGORITHM_FLAG_VALUES = ("single", "multi", "many", "neural")

ALGORITHM_FLAG_ALIASES: dict[str, str] = {
    "mono": "single",
    "singleobjective": "single",
    "single-objective": "single",
    "soo": "single",
    "multiobjective": "multi",
    "multi-objective": "multi",
    "moo": "multi",
    "manyobjective": "many",
    "many-objective": "many",
    "maop": "many",
    "maoea": "many",
    "nn": "neural",
    "dl": "neural",
    "deep": "neural",
    "mlp": "neural",
}

MANY_OBJECTIVE_ALGO_HINTS = {
    "nsga3",
    "unsga3",
    "rnsga3",
    "rvea",
    "moead",
    "ctaea",
    "age",
    "agemoea",
    "agemoeaii",
}

LOCAL_ALGORITHM_EXCLUDED_ROOTS = {"base", "moo", "soo", "__pycache__"}
LOCAL_ALGORITHM_EXCLUDED_FILES = {"hyperparameters.py", "ssw_rdpa copy.py"}
OPTIONAL_BUILTIN_ALGORITHM_MODULE_HINTS = {"optuna"}
EXCLUDED_BUILTIN_ALGORITHM_MODULE_HINTS = {"mopso_cd"}
OPTIONAL_PYMOO_ALGORITHM_MODULE_HINTS = OPTIONAL_BUILTIN_ALGORITHM_MODULE_HINTS
EXCLUDED_PYMOO_ALGORITHM_MODULE_HINTS = EXCLUDED_BUILTIN_ALGORITHM_MODULE_HINTS
LOCAL_METRIC_FOLDERS = ("metrics",)

SOURCE_EMOPYLAB = "emopylab"
SOURCE_BUILTIN = "emopylab"
SOURCE_PYMOO = "emopylab"
SOURCE_LOCAL = "local"
SOURCE_GUIPYMOO = SOURCE_LOCAL
SOURCE_LOCAL_LEGACY = SOURCE_LOCAL
CUSTOM_SOURCE_VALUES = {SOURCE_LOCAL, "guipymoo", "local"}


def _is_custom_source(source: str) -> bool:
    return str(source).strip().lower() in CUSTOM_SOURCE_VALUES

CONSTRAINED_PROBLEM_HINTS = {
    "bnh",
    "osy",
    "tnk",
    "ctp",
    "srn",
    "c1dtlz1",
    "c1dtlz3",
    "c2dtlz2",
    "c3dtlz4",
}


class AppStylesMeta(type):
    """Metaclass providing dynamic property access for AppStyles tokens."""
    @property
    def BG(cls) -> str: return StylesAppStyles.colors.background
    @property
    def PANEL(cls) -> str: return StylesAppStyles.colors.surface
    @property
    def PANEL_ALT(cls) -> str: return StylesAppStyles.colors.surface_variant
    @property
    def BORDER(cls) -> str: return StylesAppStyles.colors.border
    @property
    def PRIMARY(cls) -> str: return StylesAppStyles.colors.primary
    @property
    def PRIMARY_DARK(cls) -> str: return StylesAppStyles.colors.primary_dark
    @property
    def PRIMARY_LIGHT(cls) -> str: return StylesAppStyles.colors.primary_light
    @property
    def PRIMARY_SUBTLE(cls) -> str: return StylesAppStyles.colors.primary_subtle
    @property
    def PRIMARY_HOVER(cls) -> str: return StylesAppStyles.colors.primary_light
    @property
    def TEXT(cls) -> str: return StylesAppStyles.colors.text_primary
    @property
    def TEXT_PRIMARY(cls) -> str: return StylesAppStyles.colors.text_primary
    @property
    def TEXT_MUTED(cls) -> str: return StylesAppStyles.colors.text_muted
    @property
    def TEXT_SECONDARY(cls) -> str: return StylesAppStyles.colors.text_secondary
    @property
    def TEXT_DISABLED(cls) -> str: return StylesAppStyles.colors.text_disabled
    @property
    def TEXT_ON_PRIMARY(cls) -> str: return StylesAppStyles.colors.text_on_primary
    @property
    def SUCCESS(cls) -> str: return StylesAppStyles.colors.success
    @property
    def WARNING(cls) -> str: return StylesAppStyles.colors.warning
    @property
    def ERROR(cls) -> str: return StylesAppStyles.colors.danger
    @property
    def INFO(cls) -> str: return StylesAppStyles.colors.info
    @property
    def ACCENT_BLUE(cls) -> str: return StylesAppStyles.colors.accent_blue
    @property
    def ACCENT_PROBLEM(cls) -> str: return StylesAppStyles.colors.accent_orange
    @property
    def SELECTION_BLUE(cls) -> str: return StylesAppStyles.colors.selection_blue
    @property
    def SELECTION_PROBLEM(cls) -> str: return StylesAppStyles.colors.selection_orange
    @property
    def BORDER_LIGHT(cls) -> str: return StylesAppStyles.colors.border_light


class AppStyles(metaclass=AppStylesMeta):
    """
    Dynamic wrapper for styles.py guaranteeing immediate reactive theme adaptation.
    """
    @classmethod
    def stylesheet(cls) -> str:
        """Return complementary stylesheet from styles.py."""
        return StylesAppStyles.get_stylesheet()

    @classmethod
    def search_input_style(cls) -> str:
        """Return the shared input treatment from styles.py."""
        return StylesAppStyles.get_search_input_style()

    @classmethod
    def chart_palette(cls) -> tuple[str, ...]:
        """Return the shared scientific palette from styles.py."""
        return StylesAppStyles.get_chart_palette()

    @classmethod
    def qt_material_theme(cls) -> dict[str, str]:
        """Return Qt Material compatibility tokens from styles.py."""
        return StylesAppStyles.get_qt_material_theme()

    @classmethod
    def algorithm_list_style(cls) -> str:
        return StylesAppStyles.get_algorithm_list_style()

    @classmethod
    def problem_list_style(cls) -> str:
        return StylesAppStyles.get_problem_list_style()

    @classmethod
    def metric_list_style(cls) -> str:
        return StylesAppStyles.get_metric_list_style()

    @classmethod
    def selection_card_style(cls, color: str | None = None) -> str:
        return StylesAppStyles.get_selection_card_style(color=color)

    @classmethod
    def title_style(cls) -> str:
        return StylesAppStyles.get_title_style()

    @classmethod
    def muted_style(cls) -> str:
        return StylesAppStyles.get_muted_style()

    @classmethod
    def helper_text_style(cls) -> str:
        return StylesAppStyles.get_helper_text_style()

    @classmethod
    def section_subtitle_style(cls) -> str:
        return StylesAppStyles.get_section_subtitle_style()

    @classmethod
    def count_badge_style(cls) -> str:
        return StylesAppStyles.get_count_badge_style()

    @classmethod
    def panel_hint_style(cls) -> str:
        return StylesAppStyles.get_panel_hint_style()

    @classmethod
    def guided_panel_style(cls) -> str:
        return StylesAppStyles.get_guided_panel_style()

    @classmethod
    def primary_button_style(cls) -> str:
        return StylesAppStyles.get_primary_button_style()

    @classmethod
    def secondary_button_style(cls) -> str:
        return StylesAppStyles.get_secondary_button_style()

    @classmethod
    def danger_button_style(cls) -> str:
        return StylesAppStyles.get_danger_button_style()

    @classmethod
    def status_pill_style(cls, color: str | None = None) -> str:
        return StylesAppStyles.get_status_pill_style(color=color)

    @classmethod
    def metric_tile_style(cls, accent: str | None = None) -> str:
        return StylesAppStyles.get_metric_tile_style(accent=accent)

    @classmethod
    def log_view_style(cls) -> str:
        return StylesAppStyles.get_log_view_style()

    @classmethod
    def table_style(cls, object_name: str | None = None) -> str:
        return StylesAppStyles.get_table_style(object_name=object_name)

    @staticmethod
    def rgba(hex_color: str, alpha: float) -> str:
        color = str(hex_color).strip().lstrip("#")
        if len(color) != 6:
            return f"rgba(0,0,0,{alpha})"
        r = int(color[0:2], 16)
        g = int(color[2:4], 16)
        b = int(color[4:6], 16)
        return f"rgba({r},{g},{b},{alpha})"


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
        # Brand artwork painted in the left slot of the bar (transparent PNG).
        logo_path = Path(__file__).resolve().parent / "emopylab.png"
        if not logo_path.exists():
            logo_path = Path(__file__).resolve().parent / "emopylab.png"
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
                if image.pixelColor(x, y).alpha() > 1:
                    min_x = min(min_x, x)
                    min_y = min(min_y, y)
                    max_x = max(max_x, x)
                    max_y = max(max_y, y)

        if max_x < min_x or max_y < min_y:
            return pixmap

        padding = 6
        left = max(0, min_x - padding)
        top = max(0, min_y - padding)
        right = min(width - 1, max_x + padding)
        bottom = min(height - 1, max_y + padding)
        return pixmap.copy(QRect(left, top, right - left + 1, bottom - top + 1))

    def tabSizeHint(self, index: int) -> QSize:
        width = self._base_tab_width(index)
        if index == 0:
            width += self._brand_width()
        return QSize(width, 64)

    def resizeEvent(self, event: object) -> None:
        super().resizeEvent(event)
        self.updateGeometry()

    def _base_tab_width(self, index: int) -> int:
        base = super().tabSizeHint(index)
        text_width = self.fontMetrics().horizontalAdvance(self.tabText(index))
        return max(116, text_width + 50, base.width() + 24)

    def _brand_width(self) -> int:
        if self.count() <= 0:
            return self._minimum_brand_width
        tabs_width = sum(self._base_tab_width(index) for index in range(self.count()))
        return max(self._minimum_brand_width, self.width() - tabs_width)

    def paintEvent(self, event: object) -> None:
        _ = event
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        # Soft dark background for the top header bar (#1E293B / #0F172A slate tone)
        bar_bg = QColor("#0F172A")
        painter.fillRect(self.rect(), bar_bg)
        painter.setPen(QPen(QColor("#334155"), 1))
        painter.drawLine(self.rect().bottomLeft(), self.rect().bottomRight())
        for index in range(self.count()):
            rect = self.tabRect(index)
            if not rect.isValid():
                continue

            selected = index == self.currentIndex()
            enabled = self.isTabEnabled(index)
            brand_width = self._brand_width() if index == 0 else 0

            if index == 0:
                brand_rect = QRect(rect.left() + 16, rect.top(), brand_width - 28, rect.height())
                if not self._brand_pixmap.isNull():
                    target_h = 50
                    target_w = round(
                        self._brand_pixmap.width() * target_h / self._brand_pixmap.height()
                    )
                    logo_rect = QRect(
                        brand_rect.left(),
                        brand_rect.top() + (brand_rect.height() - target_h) // 2,
                        target_w,
                        target_h,
                    )
                    painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
                    painter.drawPixmap(logo_rect, self._brand_pixmap)
                else:
                    brand_font = QFont(self.font())
                    brand_font.setPointSize(18)
                    brand_font.setBold(True)
                    painter.setFont(brand_font)
                    painter.setPen(QColor(AppStyles.TEXT))
                    painter.drawText(
                        brand_rect,
                        Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
                        "EmoPyLab",
                    )

            tab_rect = rect.adjusted(brand_width, 0, 0, 0)

            if selected:
                selected_rect = tab_rect.adjusted(6, 8, -6, -8)
                painter.setPen(Qt.PenStyle.NoPen)
                # Subtle glow container for active tab
                painter.setBrush(QColor("#1E293B"))
                painter.drawRoundedRect(selected_rect, 8, 8)

                underline = QRect(
                    selected_rect.left() + 10,
                    rect.bottom() - 3,
                    max(24, selected_rect.width() - 20),
                    3,
                )
                painter.setBrush(QColor("#F59E0B"))  # Vibrant gold-amber indicator
                painter.drawRoundedRect(underline, 2, 2)
            icon_size = 20
            icon_rect = QRect(
                tab_rect.left() + 18,
                tab_rect.top() + 22,
                icon_size,
                icon_size,
            )
            icon = self.tabIcon(index)
            if not icon.isNull():
                icon.paint(
                    painter,
                    icon_rect,
                    Qt.AlignmentFlag.AlignCenter,
                    QIcon.Mode.Normal if enabled else QIcon.Mode.Disabled,
                    QIcon.State.On if selected else QIcon.State.Off,
                )

            label_font = QFont(self.font())
            label_font.setPointSize(10)
            label_font.setBold(selected)
            painter.setFont(label_font)
            painter.setPen(
                QColor(
                    "#F8FAFC"  # High contrast crisp white for active tab
                    if selected and enabled
                    else "#94A3B8"  # Soft slate muted gray for inactive tabs
                    if enabled
                    else "#475569"
                )
            )
            text_rect = QRect(tab_rect.left() + 44, tab_rect.top() + 18, tab_rect.width() - 52, 30)
            painter.drawText(
                text_rect,
                Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter | Qt.TextFlag.TextSingleLine,
                self.tabText(index),
            )


def _navigation_icon(name: str) -> QIcon:
    """Create an accessible QtAwesome navigation icon with active/inactive contrast states for dark header."""
    try:
        from core.ui.icons import get_navigation_icon
        return get_navigation_icon(name, primary_color="#F59E0B", inactive_color="#94A3B8")
    except Exception:
        try:
            return qta.icon("fa5s.circle", color="#F59E0B")
        except Exception:
            return QIcon()


def app_icon(name: str, color: Optional[str] = None) -> QIcon:
    """Return a unified QtAwesome icon for UI buttons and views."""
    col = color or AppStyles.TEXT_PRIMARY
    try:
        from core.ui.icons import get_icon
        return get_icon(name, color=col)
    except Exception:
        try:
            return qta.icon("fa5s.circle", color=col)
        except Exception:
            return QIcon()

def _normalize_algorithm_flags(raw: Any) -> set[str]:
    if raw is None:
        return set()

    if isinstance(raw, str):
        chunks = re.split(r"[,\s;/|]+", raw.strip())
    elif isinstance(raw, (list, tuple, set, frozenset)):
        chunks = [str(v) for v in raw]
    else:
        chunks = [str(raw)]

    normalized: set[str] = set()
    for chunk in chunks:
        token = str(chunk).strip().lower()
        if not token:
            continue
        token = ALGORITHM_FLAG_ALIASES.get(token, token)
        if token in ALGORITHM_FLAG_VALUES:
            normalized.add(token)
    return normalized


def infer_algorithm_traits(spec: AlgorithmSpec) -> tuple[str, set[str], set[str], set[str]]:
    name = spec.name.lower().strip()
    module = spec.module.lower().strip()
    flags = _normalize_algorithm_flags(getattr(spec, "flags", set()))

    objective = "multi"
    if ".soo." in module:
        objective = "single"
    elif any(hint in name for hint in MANY_OBJECTIVE_ALGO_HINTS):
        objective = "many"

    if "single" in flags:
        objective = "single"
    elif "many" in flags:
        objective = "many"
    elif "multi" in flags:
        objective = "multi"

    encoding: set[str]
    if _is_custom_source(spec.source):
        encoding = {str(value) for value in ENCODING_FILTER_VALUES}
    else:
        encoding = {"real"}
        if any(tok in name for tok in {"binary", "bin", "bool"}):
            encoding.add("binary")
        if any(tok in name for tok in {"int", "integer", "discrete"}):
            encoding.add("integer")
        if any(tok in name for tok in {"perm", "tsp", "qap", "order"}):
            encoding.add("permutation")
        if any(tok in name for tok in {"mixed", "choice"}):
            encoding.add("mixed")

    difficulties: set[str] = set()
    if any(tok in name for tok in {"ctaea", "eps", "constraint", "constrained"}):
        difficulties.add("constrained")
    if any(tok in name for tok in {"dyn", "dynamic", "dmo", "dnsga"}):
        difficulties.add("dynamic")

    return objective, encoding, difficulties, flags


def infer_problem_traits(spec: ProblemSpec) -> tuple[str, set[str], set[str]]:
    name = spec.name.lower().strip()

    if spec.default_n_obj <= 1:
        objective = "single"
    elif spec.default_n_obj >= 4:
        objective = "many"
    else:
        objective = "multi"

    if name.startswith("dtlz") or name.startswith("wfg"):
        objective = "many"

    encoding: set[str]
    if _is_custom_source(spec.source):
        encoding = {str(value) for value in ENCODING_FILTER_VALUES}
    elif name in {"zdt5"} or "binary" in name:
        encoding = {"binary"}
    elif any(tok in name for tok in {"perm", "tsp", "qap", "flowshop", "pfsp"}):
        encoding = {"permutation"}
    elif any(tok in name for tok in {"int", "integer", "discrete"}):
        encoding = {"integer"}
    elif any(tok in name for tok in {"mixed", "choice"}):
        encoding = {"mixed"}
    else:
        encoding = {"real"}

    difficulties: set[str] = set()
    if (
        name in CONSTRAINED_PROBLEM_HINTS
        or name.startswith("c")
        or "constraint" in name
        or "constrained" in name
    ):
        difficulties.add("constrained")
    if name.startswith("df") or "dynamic" in name:
        difficulties.add("dynamic")

    return objective, encoding, difficulties


def _fmt_number(value: float | int | None, digits: int = 6) -> str:
    number = _float_or_none(value)
    if number is None:
        return "-"
    if not math.isfinite(number):
        return "-"
    return f"{number:.{digits}g}"


def _float_or_none(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _float_or_nan(value: Any) -> float:
    converted = _float_or_none(value)
    if converted is None:
        return float("nan")
    return converted


def _mean_std(values: list[float]) -> tuple[float, float]:
    arr = np.asarray([v for v in values if math.isfinite(v)], dtype=float)
    if arr.size == 0:
        return float("nan"), float("nan")
    return float(np.mean(arr)), float(np.std(arr))


def _positive_int(value: Any, default: int, *, minimum: int = 1) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return max(minimum, parsed)


def resolve_parallel_worker_limits() -> tuple[int, int]:
    """
    Return (recommended_default, max_supported) for worker threads.

    EmoPyLab UI policy:
    - default = half of physical CPUs
    - max = half of logical CPUs
    - if logical < physical (unexpected/virtualized edge case), use half of physical for max
    """
    logical_cpus = int(os.cpu_count() or 1)
    physical_cpus = None
    try:
        import psutil  # type: ignore

        physical_cpus = psutil.cpu_count(logical=False)
    except Exception:  # noqa: BLE001
        physical_cpus = None

    if physical_cpus is None:
        physical_cpus = logical_cpus

    try:
        physical_cpus = int(physical_cpus)
    except Exception:  # noqa: BLE001
        physical_cpus = logical_cpus
    physical_cpus = max(1, physical_cpus)

    recommended = max(1, physical_cpus // 2)
    if logical_cpus < physical_cpus:
        max_supported = max(1, physical_cpus // 2)
    else:
        max_supported = max(1, logical_cpus // 2)

    # Keep UI sane if physical//2 exceeds logical//2 due to odd/virtualized counts.
    max_supported = max(1, max_supported)
    if recommended > max_supported:
        recommended = max_supported
    return recommended, max_supported


def _algorithm_uses_local_llm(spec: AlgorithmSpec) -> bool:
    if not isinstance(spec, AlgorithmSpec):
        return False
    fields = (
        str(getattr(spec, "name", "")),
        str(getattr(spec, "module", "")),
        str(getattr(spec, "id", "")),
    )
    joined = " ".join(fields).lower()
    if "larc" in joined or "ollama" in joined:
        return True

    try:
        flags = {str(flag).strip().lower() for flag in (spec.flags or set())}
    except Exception:  # noqa: BLE001
        flags = set()
    return any(flag in {"llm", "local_llm", "ollama", "larc"} for flag in flags)


def _resolve_parallel_workers_for_algorithms(
    requested_workers: int,
    max_supported_workers: int,
    selected_algorithms: list[AlgorithmSpec],
) -> tuple[int, list[str]]:
    requested = max(1, int(requested_workers))
    max_supported = max(1, int(max_supported_workers))
    effective = min(requested, max_supported)
    warnings: list[str] = []

    if requested > max_supported:
        warnings.append(
            f"parallel_workers={requested} exceeds machine max ({max_supported}). Using {effective}."
        )

    llm_selected = any(_algorithm_uses_local_llm(spec) for spec in selected_algorithms)
    if llm_selected and effective > LLM_LOCAL_MAX_PARALLEL_WORKERS:
        warnings.append(
            "Local LLM algorithm selected (LARC). "
            f"Limiting parallel_workers to {LLM_LOCAL_MAX_PARALLEL_WORKERS} to reduce memory pressure."
        )
        effective = LLM_LOCAL_MAX_PARALLEL_WORKERS

    return effective, warnings


def _build_lmstudio_models_url(chat_url: str) -> str:
    raw = str(chat_url or DEFAULT_LOCAL_LMSTUDIO_CHAT_URL).strip()
    if "/chat/completions" in raw:
        return raw.rsplit("/chat/completions", 1)[0] + "/models"
    parsed = urllib.parse.urlsplit(raw)
    if not parsed.netloc:
        parsed = urllib.parse.urlsplit(f"http://{raw.lstrip('/')}")
    if not parsed.netloc:
        return "http://127.0.0.1:1234/v1/models"
    return urllib.parse.urlunsplit((parsed.scheme or "http", parsed.netloc, "/v1/models", "", ""))


def _assert_local_lmstudio_model_available(
    *,
    model: str,
    chat_url: str,
    timeout_s: float = 5.0,
) -> None:
    models_url = _build_lmstudio_models_url(chat_url)
    request = urllib.request.Request(
        models_url,
        headers={"Accept": "application/json"},
        method="GET",
    )
    try:
        with urllib.request.urlopen(request, timeout=float(timeout_s)) as response:
            payload = response.read().decode("utf-8")
    except Exception as exc:
        raise RuntimeError(
            "LM Studio local is unavailable for LARC_NSGA3. "
            f"URL={chat_url!r}, expected model={model!r}."
        ) from exc

    try:
        data = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"LM Studio local returned invalid JSON from /v1/models: {models_url!r}.") from exc

    available_models: set[str] = set()
    if isinstance(data, dict):
        raw_models = data.get("data", [])
        if isinstance(raw_models, list):
            for item in raw_models:
                if isinstance(item, dict):
                    name = item.get("id")
                    if isinstance(name, str) and name.strip():
                        available_models.add(name.strip())

    if not available_models:
        raise RuntimeError("LM Studio local is active, but no models are loaded in memory.")

    if model in {None, "", "loaded-model"}:
        return

    matched = False
    for av_m in available_models:
        if model.lower() in av_m.lower() or av_m.lower() in model.lower():
            matched = True
            break

    if not matched:
        available_str = ", ".join(sorted(available_models))
        raise RuntimeError(
            "LM Studio model required by LARC_NSGA3 is not available in memory. "
            f"expected={model!r}; loaded={available_str}."
        )


def _random_seed() -> int:
    return int(np.random.default_rng().integers(1, 2_147_483_647))


def _normalize_seed_value(value: Any, default: int = 1) -> int:
    seed = _positive_int(value, default, minimum=1)
    return int(max(1, min(seed, 2_147_483_647)))


def _canonical_json_dumps(data: Any) -> str:
    return json.dumps(data, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _sanitize_config_for_manifest(config: dict[str, Any]) -> dict[str, Any]:
    safe = dict(config)
    safe.pop("__single_problem_mode__", None)
    safe.pop("__suppress_progress__", None)
    return safe


def build_seed_plan(config: dict[str, Any], total_slots: int) -> dict[str, Any]:
    slots = max(1, int(total_slots))
    mode_raw = str(config.get("seed_mode", SEED_MODE_RANDOM)).strip().lower()
    if mode_raw not in {SEED_MODE_RANDOM, SEED_MODE_FIXED, SEED_MODE_SEQUENCE}:
        mode_raw = SEED_MODE_RANDOM

    base_seed = _normalize_seed_value(config.get("seed_base", config.get("seed", 1)), default=1)
    step = _positive_int(config.get("seed_step", 1), 1, minimum=1)
    sequence_raw = config.get("seed_sequence", [])

    sequence: list[int] = []
    if isinstance(sequence_raw, str):
        chunks = re.split(r"[,;\s]+", sequence_raw.strip())
        sequence = [_normalize_seed_value(chunk, default=1) for chunk in chunks if chunk.strip()]
    elif isinstance(sequence_raw, (list, tuple)):
        for item in sequence_raw:
            try:
                sequence.append(_normalize_seed_value(item, default=1))
            except Exception:  # noqa: BLE001
                continue

    if mode_raw == SEED_MODE_FIXED:
        seeds = [base_seed for _ in range(slots)]
    elif mode_raw == SEED_MODE_SEQUENCE:
        if sequence:
            seeds = [sequence[idx % len(sequence)] for idx in range(slots)]
        else:
            seeds = [_normalize_seed_value(base_seed + idx * step, default=1) for idx in range(slots)]
    else:
        seeds = [_random_seed() for _ in range(slots)]

    return {
        "mode": mode_raw,
        "deterministic": mode_raw in {SEED_MODE_FIXED, SEED_MODE_SEQUENCE},
        "base_seed": base_seed,
        "step": int(step),
        "provided_sequence": sequence,
        "seeds": seeds,
    }


def build_execution_manifest(
    *,
    config: dict[str, Any],
    seed_plan: dict[str, Any],
    selected_problem_ids: list[str],
    selected_algorithm_ids: list[str],
    selected_metric_ids: list[str],
    execution_backend: str,
    execution_backend_label: str,
) -> dict[str, Any]:
    manifest = {
        "manifest_version": EXPERIMENT_MANIFEST_VERSION,
        "timestamp_en_us": format_timestamp_en_us(),
        "config": _sanitize_config_for_manifest(config),
        "selection": {
            "problem_ids": list(selected_problem_ids),
            "algorithm_ids": list(selected_algorithm_ids),
            "metric_ids": list(selected_metric_ids),
        },
        "seed_plan": {
            "mode": seed_plan.get("mode", SEED_MODE_RANDOM),
            "deterministic": bool(seed_plan.get("deterministic", False)),
            "base_seed": int(seed_plan.get("base_seed", 1)),
            "step": int(seed_plan.get("step", 1)),
            "provided_sequence": list(seed_plan.get("provided_sequence", [])),
            "seeds": list(seed_plan.get("seeds", [])),
        },
        "execution_backend": execution_backend,
        "execution_backend_label": execution_backend_label,
    }
    manifest_json = _canonical_json_dumps(manifest)
    manifest["manifest_sha256"] = _sha256_text(manifest_json)
    return manifest


_EN_US_WEEKDAYS = (
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
)
_EN_US_MONTHS = (
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
)


def _format_en_us_utc_offset(dt: datetime) -> str:
    offset = dt.utcoffset()
    if offset is None:
        return "UTC"
    total_minutes = int(offset.total_seconds() // 60)
    sign = "+" if total_minutes >= 0 else "-"
    total_minutes = abs(total_minutes)
    hours = total_minutes // 60
    minutes = total_minutes % 60
    return f"GMT{sign}{hours:02d}:{minutes:02d}"


def format_timestamp_en_us(value: datetime | None = None) -> str:
    dt = value if isinstance(value, datetime) else datetime.now()
    dt = dt.astimezone()
    hour_12 = dt.hour % 12 or 12
    am_pm = "AM" if dt.hour < 12 else "PM"
    weekday = _EN_US_WEEKDAYS[dt.weekday()]
    month = _EN_US_MONTHS[dt.month - 1]
    utc_offset = _format_en_us_utc_offset(dt)
    return (
        f"{weekday}, {month} {dt.day:02d}, {dt.year} "
        f"{hour_12:02d}:{dt.minute:02d}:{dt.second:02d} {am_pm} {utc_offset}"
    )


def detect_gpu_runtime() -> dict[str, Any]:
    info: dict[str, Any] = {
        "jax_ok": False,
        "jax_version": None,
        "cuda_ok": False,
        "cuda_device_name": None,
        "device_count": 0,
        "all_devices": [],
        "error": None,
    }
    try:
        import jax
    except Exception as exc:  # noqa: BLE001
        info["error"] = f"JAX not available: {exc}"
        return info

    info["jax_ok"] = True
    info["jax_version"] = str(getattr(jax, "__version__", "unknown"))
    try:
        devices = list(jax.devices())
    except Exception as exc:  # noqa: BLE001
        info["error"] = f"Could not query JAX devices: {exc}"
        return info

    info["device_count"] = len(devices)
    info["all_devices"] = [f"{getattr(d, 'platform', '-')}:{getattr(d, 'device_kind', d)}" for d in devices]

    gpu_devices = [d for d in devices if str(getattr(d, "platform", "")).lower() in {"gpu", "cuda", "rocm"}]
    if gpu_devices:
        info["cuda_ok"] = True
        info["cuda_device_name"] = str(getattr(gpu_devices[0], "device_kind", gpu_devices[0]))

    return info


def build_gpu_status_text(info: dict[str, Any]) -> str:
    if info.get("cuda_ok"):
        return f"JAX GPU available: {info.get('cuda_device_name', 'GPU device')}"
    if info.get("jax_ok"):
        return "JAX available, but no GPU device detected. CPU execution will be used."
    if info.get("error"):
        return str(info.get("error"))
    return "JAX runtime unavailable. CPU execution will be used."


def ensure_plugin_directories(base_dir: Path) -> dict[str, Path]:
    folders = {
        "algorithms": base_dir / "algorithms",
        "problems": base_dir / "problems",
        "metrics": base_dir / "metrics",
        "operators": base_dir / "operators",
    }
    for path in folders.values():
        path.mkdir(parents=True, exist_ok=True)
        init_file = path / "__init__.py"
        if not init_file.exists():
            init_file.write_text("", encoding="utf-8")

    legacy_metrics_dir = base_dir / "indicators" / "custom_metrics"
    metrics_dir = folders["metrics"]
    if legacy_metrics_dir.exists() and legacy_metrics_dir.is_dir():
        for legacy_file in sorted(legacy_metrics_dir.glob("*.py")):
            if legacy_file.name == "__init__.py":
                continue
            target_file = metrics_dir / legacy_file.name
            if target_file.exists():
                continue
            try:
                legacy_file.replace(target_file)
            except Exception:
                continue

        legacy_init = legacy_metrics_dir / "__init__.py"
        if legacy_init.exists():
            try:
                legacy_init.unlink()
            except Exception:
                pass

        try:
            if not any(legacy_metrics_dir.iterdir()):
                legacy_metrics_dir.rmdir()
        except Exception:
            pass

        legacy_indicators_root = base_dir / "indicators"
        try:
            if legacy_indicators_root.exists() and not any(legacy_indicators_root.iterdir()):
                legacy_indicators_root.rmdir()
        except Exception:
            pass
    return folders


# Phase 7 bridge: delegate core responsibilities to modular package.
detect_gpu_runtime = core_detect_gpu_runtime
build_gpu_status_text = core_build_gpu_status_text
build_seed_plan = core_build_seed_plan
build_execution_manifest = core_build_execution_manifest
ensure_plugin_directories = core_ensure_plugin_directories
LLMFormulationService = CoreLLMFormulationService


def import_module_from_file(py_file: Path, prefix: str) -> tuple[Any | None, str | None]:
    unique = abs(hash(str(py_file.resolve())))
    module_name = f"{prefix}_{py_file.stem}_{unique}"
    try:
        spec = importlib.util.spec_from_file_location(module_name, py_file)
        if spec is None or spec.loader is None:
            return None, "invalid module spec"
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module, None
    except Exception as exc:  # noqa: BLE001
        return None, str(exc)


def _ensure_virtual_package(name: str) -> types.ModuleType:
    existing = sys.modules.get(name)
    if isinstance(existing, types.ModuleType):
        return existing

    try:
        mod = importlib.import_module(name)
        return mod
    except Exception:
        pass

    module = types.ModuleType(name)
    module.__path__ = []  # type: ignore[attr-defined]
    sys.modules[name] = module
    return module


def _attach_module_alias(alias_name: str, module_obj: types.ModuleType) -> None:
    sys.modules[alias_name] = module_obj
    parent_name, _, child_name = alias_name.rpartition(".")
    if parent_name and child_name:
        parent_module = _ensure_virtual_package(parent_name)
        setattr(parent_module, child_name, module_obj)


def _install_legacy_runtime_aliases() -> None:
    """
    Provide lightweight legacy import aliases used by local legacy plugins.

    This maps any legacy pymoo module requests (pymoo.core.*, pymoo.util.*,
    pymoo.algorithms.*, pymoo.optimize, pymoo.termination, pymoo.operators.*)
    directly to EmoPyLab's native core.*, util.*, algorithms.*, optimize,
    termination, and operators.* modules in sys.modules.
    """
    # Prefer the real backend-aware modules (MLX/JAX) when importable.
    for _real_module in (
        "util.array_backend",
        "core.algorithm",
        "core.problem",
        "core.population",
        "core.individual",
        "core.duplicate",
        "core.mating",
        "core.mixed",
        "core.parameters",
        "core.variable",
        "core.crossover",
        "core.mutation",
        "core.sampling",
        "core.selection",
        "core.repair",
        "core.replacement",
        "core.survival",
        "core.infill",
        "core.evaluator",
        "core.callback",
        "core.result",
        "core.termination",
        "core.optimize",
        "util.misc",
        "util.ref_dirs",
        "util.optimum",
        "util.dominator",
        "util.nds.non_dominated_sorting",
        "util.display.multi",
    ):
        if _real_module not in sys.modules:
            try:
                importlib.import_module(_real_module)
            except Exception:
                pass

    # Ensure base packages exist in sys.modules.
    _ensure_virtual_package("core")
    _ensure_virtual_package("util")
    _ensure_virtual_package("operators")
    for operator_pkg in ("operators.crossover", "operators.mutation", "operators.selection", "operators.sampling", "operators.repair", "operators.survival"):
        try:
            importlib.import_module(operator_pkg)
        except Exception:
            _ensure_virtual_package(operator_pkg)
    try:
        importlib.import_module("algorithms")
    except Exception:
        _ensure_virtual_package("algorithms")
    _ensure_virtual_package("algorithms.moo")
    _ensure_virtual_package("util.nds")
    _ensure_virtual_package("util.display")

    try:
        from util.misc import default_random_state as native_default_random_state
        util_pkg = _ensure_virtual_package("util")
        setattr(util_pkg, "default_random_state", native_default_random_state)
    except Exception:
        pass

    # Map legacy pymoo module paths to native EmoPyLab modules in sys.modules.
    pymoo_mappings = {
        "pymoo.core": "core",
        "pymoo.core.algorithm": "core.algorithm",
        "pymoo.core.problem": "core.problem",
        "pymoo.core.population": "core.population",
        "pymoo.core.individual": "core.individual",
        "pymoo.core.duplicate": "core.duplicate",
        "pymoo.core.mating": "core.mating",
        "pymoo.core.mixed": "core.mixed",
        "pymoo.core.parameters": "core.parameters",
        "pymoo.core.variable": "core.variable",
        "pymoo.core.crossover": "core.crossover",
        "pymoo.core.mutation": "core.mutation",
        "pymoo.core.sampling": "core.sampling",
        "pymoo.core.selection": "core.selection",
        "pymoo.core.repair": "core.repair",
        "pymoo.core.replacement": "core.replacement",
        "pymoo.core.survival": "core.survival",
        "pymoo.core.infill": "core.infill",
        "pymoo.core.evaluator": "core.evaluator",
        "pymoo.core.callback": "core.callback",
        "pymoo.core.result": "core.result",
        "pymoo.core.termination": "core.termination",
        "pymoo.core.operator": "core.operator",
        "pymoo.core.initialization": "core.initialization",
        "pymoo.optimize": "core.optimize",
        "pymoo.termination": "core.termination",
        "pymoo.util": "util",
        "pymoo.util.misc": "util.misc",
        "pymoo.util.ref_dirs": "util.ref_dirs",
        "pymoo.util.optimum": "util.optimum",
        "pymoo.util.dominator": "util.dominator",
        "pymoo.util.array_backend": "util.array_backend",
        "pymoo.util.randomized_argsort": "util.randomized_argsort",
        "pymoo.util.nds.non_dominated_sorting": "util.nds.non_dominated_sorting",
        "pymoo.util.display.multi": "util.display.multi",
        "pymoo.operators": "operators",
        "pymoo.operators.crossover": "operators.crossover",
        "pymoo.operators.crossover.sbx": "operators.crossover.sbx",
        "pymoo.operators.crossover.ux": "operators.crossover.ux",
        "pymoo.operators.crossover.binx": "operators.crossover.binx",
        "pymoo.operators.crossover.dex": "operators.crossover.dex",
        "pymoo.operators.crossover.expx": "operators.crossover.expx",
        "pymoo.operators.crossover.erx": "operators.crossover.erx",
        "pymoo.operators.crossover.hux": "operators.crossover.hux",
        "pymoo.operators.crossover.nox": "operators.crossover.nox",
        "pymoo.operators.crossover.ox": "operators.crossover.ox",
        "pymoo.operators.crossover.pcx": "operators.crossover.pcx",
        "pymoo.operators.crossover.pntx": "operators.crossover.pntx",
        "pymoo.operators.crossover.spx": "operators.crossover.spx",
        "pymoo.operators.mutation": "operators.mutation",
        "pymoo.operators.mutation.pm": "operators.mutation.pm",
        "pymoo.operators.mutation.bitflip": "operators.mutation.bitflip",
        "pymoo.operators.mutation.gauss": "operators.mutation.gauss",
        "pymoo.operators.mutation.inversion": "operators.mutation.inversion",
        "pymoo.operators.mutation.nom": "operators.mutation.nom",
        "pymoo.operators.mutation.rm": "operators.mutation.rm",
        "pymoo.operators.repair": "operators.repair",
        "pymoo.operators.repair.bounds_repair": "operators.repair.bounds_repair",
        "pymoo.operators.repair.rounding": "operators.repair.rounding",
        "pymoo.operators.repair.to_bound": "operators.repair.to_bound",
        "pymoo.operators.repair.bounce_back": "operators.repair.bounce_back",
        "pymoo.operators.repair.inverse_penalty": "operators.repair.inverse_penalty",
        "pymoo.operators.repair.vtype": "operators.repair.vtype",
        "pymoo.operators.sampling": "operators.sampling",
        "pymoo.operators.sampling.lhs": "operators.sampling.lhs",
        "pymoo.operators.sampling.rnd": "operators.sampling.rnd",
        "pymoo.operators.selection": "operators.selection",
        "pymoo.operators.selection.tournament": "operators.selection.tournament",
        "pymoo.operators.selection.rnd": "operators.selection.rnd",
        "pymoo.operators.survival": "operators.survival",
        "pymoo.operators.survival.rank_and_crowding": "operators.survival.rank_and_crowding",
        "pymoo.operators.survival.rank_and_crowding.classes": "operators.survival.rank_and_crowding.classes",
        "pymoo.operators.survival.rank_and_crowding.metrics": "operators.survival.rank_and_crowding.metrics",
        "pymoo.algorithms": "algorithms",
        "pymoo.algorithms.moo": "algorithms.moo",
        "pymoo.algorithms.moo.nsga2": "algorithms.moo.nsga2",
        "pymoo.algorithms.moo.nsga3": "algorithms.moo.nsga3",
        "pymoo.algorithms.moo.moead": "algorithms.moo.moead",
        "pymoo.algorithms.moo.rvea": "algorithms.moo.rvea",
        "pymoo.algorithms.moo.age2": "algorithms.moo.age2",
    }

    _ensure_virtual_package("pymoo")

    # Register parents before child modules so packages exist before child attributes are attached
    for alias_name, target_name in sorted(pymoo_mappings.items(), key=lambda item: item[0].count(".")):
        try:
            target_module = importlib.import_module(target_name)
        except Exception:
            continue
        _attach_module_alias(alias_name, target_module)


_install_legacy_runtime_aliases()


def _collect_algorithm_flags(module: Any, entry_name: str, entry_obj: Any) -> set[str]:
    flags: set[str] = set()

    # Module-level defaults or mappings.
    module_flags = getattr(module, "ALGORITHM_FLAGS", None)
    if isinstance(module_flags, dict):
        candidates = [
            str(entry_name),
            getattr(entry_obj, "__name__", None),
            "__default__",
            "default",
            "*",
        ]
        for key in candidates:
            if key is None:
                continue
            if key in module_flags:
                flags |= _normalize_algorithm_flags(module_flags[key])
    else:
        flags |= _normalize_algorithm_flags(module_flags)

    flags |= _normalize_algorithm_flags(getattr(module, "ALGO_FLAGS", None))

    # Entry-level declarations.
    for attr in (
        "ALGORITHM_FLAGS",
        "ALGO_FLAGS",
        "FLAGS",
        "TAGS",
        "OBJECTIVE_SCOPE",
        "OBJECTIVE_TYPE",
        "OBJECTIVE",
        "CATEGORY",
    ):
        if hasattr(entry_obj, attr):
            flags |= _normalize_algorithm_flags(getattr(entry_obj, attr))

    return flags


def _patch_known_pymoo_algorithm_flags(display_name: str, flags: set[str]) -> set[str]:
    """Add missing scope flags for known pymoo algorithms when discovery metadata is incomplete."""
    fixed = set(flags or set())
    key = _normalize_type_name_key(display_name)

    if key in {"nsgaiii", "nsga3"}:
        fixed |= {"multi", "many"}

    return fixed


def _normalize_type_name_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).strip().lower())


def _natural_lexicographic_key(value: str) -> tuple[Any, ...]:
    """Natural lexicographic sort key (e.g., NSGA2 < NSGA10)."""
    parts = re.split(r"(\d+)", str(value).strip().lower())
    key: list[Any] = []
    for part in parts:
        if not part:
            continue
        if part.isdigit():
            key.append(int(part))
        else:
            key.append(part)
    return tuple(key)


def _canonical_problem_name(value: str) -> str:
    token = str(value).strip()
    if not token:
        return token

    suffix = ""
    core = token
    if token.upper().endswith("_JAX"):
        suffix = "_JAX"
        core = token[:-4]

    # Canonicalize constrained-DTLZ style aliases:
    # C1DTLZ1 -> C1_DTLZ1, DC2DTLZ3 -> DC2_DTLZ3
    core = re.sub(r"^([A-Za-z]+[0-9]+)(DTLZ[0-9]+)$", r"\1_\2", core, flags=re.I)
    return f"{core}{suffix}" if suffix else core


def _strip_matlab_comments(text: str) -> str:
    lines: list[str] = []
    for line in str(text).splitlines():
        pos = line.find("%")
        if pos >= 0:
            line = line[:pos]
        lines.append(line)
    return "\n".join(lines)


def _extract_matlab_method_body(text: str, method_name: str) -> str:
    pat = re.compile(
        rf"^\s*function\b[^\n]*\b{re.escape(method_name)}\s*\([^\n]*\)\s*$",
        re.M,
    )
    match = pat.search(text)
    if match is None:
        pat = re.compile(
            rf"^\s*function\b[^\n]*\b{re.escape(method_name)}\b[^\n]*$",
            re.M,
        )
        match = pat.search(text)
        if match is None:
            return ""

    start_idx = match.end()
    if start_idx < len(text) and text[start_idx] == "\r":
        start_idx += 1
    if start_idx < len(text) and text[start_idx] == "\n":
        start_idx += 1

    next_match = re.search(r"^\s*function\b", text[start_idx:], re.M)
    end_idx = start_idx + next_match.start() if next_match else len(text)
    body = text[start_idx:end_idx]

    lines = body.splitlines()
    while lines and lines[-1].strip().lower() == "end":
        lines.pop()
    return "\n".join(lines).strip()


def _find_matlab_default_assignment(setting_text: str, attr: str) -> str | None:
    clean = _strip_matlab_comments(setting_text)

    cond_pat = re.compile(
        rf"if\s+isempty\s*\(\s*obj\.{re.escape(attr)}\s*\)(-P<body>.*-)end",
        re.I | re.S,
    )
    for cond in cond_pat.finditer(clean):
        assign = re.search(
            rf"obj\.{re.escape(attr)}\s*=\s*(-P<expr>[^;\n]+)\s*;",
            cond.group("body"),
            re.I,
        )
        if assign is not None:
            return str(assign.group("expr")).strip()

    direct = re.search(
        rf"obj\.{re.escape(attr)}\s*=\s*(-P<expr>[^;\n]+)\s*;",
        clean,
        re.I,
    )
    if direct is None:
        return None
    return str(direct.group("expr")).strip()


def _eval_matlab_positive_int_expr(expr: str | None, context: dict[str, Any]) -> int | None:
    if not expr:
        return None

    candidate = str(expr).strip()
    candidate = candidate.replace("^", "**")
    candidate = candidate.replace(".*", "*").replace("./", "/")
    candidate = re.sub(r"\bpi\b", "math.pi", candidate, flags=re.I)
    candidate = re.sub(r"\bceil\s*\(", "math.ceil(", candidate, flags=re.I)
    candidate = re.sub(r"\bfloor\s*\(", "math.floor(", candidate, flags=re.I)

    if re.search(r"[\[\]\{\}:]", candidate):
        return None

    for key, value in context.items():
        if isinstance(value, (int, float)):
            candidate = re.sub(
                rf"\bobj\.{re.escape(key)}\b",
                str(float(value)),
                candidate,
                flags=re.I,
            )

    if re.search(r"\bobj\.[A-Za-z_][A-Za-z0-9_]*\b", candidate):
        return None

    value = _safe_numeric_eval_expr(candidate)
    if value is None:
        return None
    if not isinstance(value, (int, float)):
        return None
    if not math.isfinite(float(value)):
        return None

    resolved = int(round(float(value)))
    return resolved if resolved > 0 else None


_SAFE_MATLAB_EXPR_BINOPS: dict[type[ast.operator], Callable[[float, float], float]] = {
    ast.Add: lambda a, b: a + b,
    ast.Sub: lambda a, b: a - b,
    ast.Mult: lambda a, b: a * b,
    ast.Div: lambda a, b: a / b,
    ast.Pow: lambda a, b: a**b,
    ast.Mod: lambda a, b: a % b,
    ast.FloorDiv: lambda a, b: a // b,
}
_SAFE_MATLAB_EXPR_UNARYOPS: dict[type[ast.unaryop], Callable[[float], float]] = {
    ast.UAdd: lambda a: +a,
    ast.USub: lambda a: -a,
}
_SAFE_MATLAB_EXPR_FUNCS: dict[str, Callable[..., float]] = {
    "max": max,
    "min": min,
    "abs": abs,
    "round": round,
    "int": int,
    "float": float,
}


def _safe_numeric_eval_expr(expr: str) -> float | None:
    try:
        tree = ast.parse(str(expr), mode="eval")
    except Exception:  # noqa: BLE001
        return None

    def _resolve_attr(node: ast.AST) -> Any:
        if isinstance(node, ast.Name) and node.id == "math":
            return math
        if isinstance(node, ast.Attribute):
            base = _resolve_attr(node.value)
            if base is None:
                return None
            name = str(node.attr)
            if base is math and name in {"pi", "ceil", "floor"}:
                return getattr(math, name, None)
        return None

    def _eval(node: ast.AST) -> float:
        if isinstance(node, ast.Expression):
            return _eval(node.body)
        if isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float)):
                return float(node.value)
            raise ValueError("non-numeric constant")
        if isinstance(node, ast.UnaryOp):
            op = _SAFE_MATLAB_EXPR_UNARYOPS.get(type(node.op))
            if op is None:
                raise ValueError("unsupported unary op")
            return float(op(float(_eval(node.operand))))
        if isinstance(node, ast.BinOp):
            op = _SAFE_MATLAB_EXPR_BINOPS.get(type(node.op))
            if op is None:
                raise ValueError("unsupported binary op")
            return float(op(float(_eval(node.left)), float(_eval(node.right))))
        if isinstance(node, ast.Call):
            if node.keywords:
                raise ValueError("keywords not allowed")
            fn: Any = None
            if isinstance(node.func, ast.Name):
                fn = _SAFE_MATLAB_EXPR_FUNCS.get(str(node.func.id))
            else:
                fn = _resolve_attr(node.func)
            if fn is None or not callable(fn):
                raise ValueError("unsupported function")
            args = [float(_eval(arg)) for arg in node.args]
            return float(fn(*args))
        if isinstance(node, ast.Attribute):
            value = _resolve_attr(node)
            if isinstance(value, (int, float)):
                return float(value)
            raise ValueError("unsupported attribute")
        raise ValueError(f"unsupported node: {type(node).__name__}")

    try:
        value = _eval(tree)
    except Exception:  # noqa: BLE001
        return None
    try:
        out = float(value)
    except Exception:  # noqa: BLE001
        return None
    return out if math.isfinite(out) else None


def _urlopen_http_only(req: urllib.request.Request, *, timeout: float):
    url = str(getattr(req, "full_url", "") or "")
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError(f"Unsupported URL scheme for remote fetch: {parsed.scheme or '<empty>'}")
    return urllib.request.urlopen(req, timeout=timeout)  # nosec B310


def _discover_matlab_problem_catalog(base_dir: Path, warnings: list[str]) -> list[dict[str, Any]]:
    problems_root = base_dir / "problems"
    root = None
    if problems_root.exists():
        for candidate in problems_root.iterdir():
            if not candidate.is_dir():
                continue
            if (candidate / "Multi-objective optimization").exists() or (candidate / "Single-objective optimization").exists():
                root = candidate
                break
    if root is None or not root.exists():
        return []

    class_re = re.compile(
        r"^\s*classdef\s+([A-Za-z][A-Za-z0-9_]*)\s*<\s*([A-Za-z][A-Za-z0-9_]*)",
        re.M,
    )
    tag_re = re.compile(r"<([^>]+)>")

    catalog: list[dict[str, Any]] = []
    for m_file in sorted(root.rglob("*.m")):
        try:
            text = m_file.read_text(encoding="utf-8", errors="ignore")
        except Exception as exc:  # noqa: BLE001
            warnings.append(f"Could not read MATLAB problem file {m_file}: {exc}")
            continue

        class_match = class_re.search(text)
        if class_match is None:
            continue

        class_name = str(class_match.group(1)).strip()
        base_class_name = str(class_match.group(2)).strip()
        if base_class_name.lower() != "problem":
            continue
        rel_source = m_file.relative_to(root).as_posix()
        rel_source_lower = rel_source.lower()
        is_single = "single-objective optimization/" in rel_source_lower

        tags: list[str] = []
        for line in text.splitlines()[:30]:
            if not line.lstrip().startswith("%"):
                continue
            for match in tag_re.finditer(line):
                tag = str(match.group(1)).strip().lower()
                if tag:
                    tags.append(tag)

        setting_body = _extract_matlab_method_body(text, "Setting")
        setting_clean = _strip_matlab_comments(setting_body)

        default_n_obj = 1 if is_single else 2
        context: dict[str, Any] = {}
        if not is_single:
            m_expr = _find_matlab_default_assignment(setting_clean, "M")
            parsed_m = _eval_matlab_positive_int_expr(m_expr, context)
            if parsed_m is not None:
                default_n_obj = max(1, int(parsed_m))
            context["M"] = default_n_obj
            if any("many" in tag for tag in tags) and default_n_obj < 3:
                default_n_obj = 3
                context["M"] = default_n_obj

        d_expr = _find_matlab_default_assignment(setting_clean, "D")
        parsed_d = _eval_matlab_positive_int_expr(d_expr, context)
        if parsed_d is not None:
            default_n_var = int(parsed_d)
        else:
            default_n_var = 10 if is_single else (30 if default_n_obj <= 2 else default_n_obj + 9)

        if is_single:
            group = "single"
        elif any("many" in tag for tag in tags) or default_n_obj > 2:
            group = "many"
        else:
            group = "multi"

        if group == "single":
            default_n_obj = 1
        elif group == "many" and default_n_obj < 4:
            default_n_obj = 4

        catalog.append(
            {
                "name": _canonical_problem_name(class_name),
                "source_rel": rel_source,
                "group": group,
                "default_n_var": int(max(1, default_n_var)),
                "default_n_obj": int(max(1, default_n_obj)),
            }
        )

    return catalog


def build_reference_dirs(n_obj: int, target: int) -> np.ndarray:
    target = max(2, int(target))
    n_obj = max(2, int(n_obj))
    try:
        from util.ref_dirs import get_reference_directions
    except Exception:  # noqa: BLE001
        rng = np.random.default_rng(42)
        dirs = rng.random((target, n_obj))
        dirs /= np.sum(dirs, axis=1, keepdims=True)
        return dirs

    last_dirs = None
    for partitions in range(1, 120):
        try:
            dirs = np.asarray(
                get_reference_directions("das-dennis", n_obj, n_partitions=partitions),
                dtype=float,
            )
        except Exception:  # noqa: BLE001
            break
        last_dirs = dirs
        if dirs.shape[0] >= target:
            return dirs

    if last_dirs is not None and last_dirs.size:
        return last_dirs

    rng = np.random.default_rng(42)
    dirs = rng.random((target, n_obj))
    dirs /= np.sum(dirs, axis=1, keepdims=True)
    return dirs


def _resolve_operator_module_class(
    operator_type: str,
    operator_name: str,
) -> tuple[str, str] | None:
    value = str(operator_name).strip()
    if not value:
        return None

    if "::" in value:
        _source, fqcn = value.split("::", 1)
        if "." in fqcn:
            module_path, class_name = fqcn.rsplit(".", 1)
            return module_path, class_name

    lowered = value.lower()
    aliases = LEGACY_OPERATOR_ALIASES.get(operator_type, {})
    if lowered in aliases:
        return aliases[lowered]

    if "." in value:
        module_path, class_name = value.rsplit(".", 1)
        return module_path, class_name

    return None


def _looks_like_jax_identifier(value: Any) -> bool:
    return bool(core_looks_like_jax_identifier(value))


def _strip_jax_suffix(value: str) -> str:
    return core_strip_jax_suffix(value)


def _iter_operator_runtime_candidates(
    module_path: str,
    class_name: str,
    *,
    prefer_jax: bool,
) -> list[tuple[str, str]]:
    return core_iter_operator_runtime_candidates(
        module_path,
        class_name,
        prefer_jax=prefer_jax,
    )


def _resolve_available_operator_target(
    module_path: str,
    class_name: str,
    *,
    prefer_jax: bool,
) -> tuple[str, str] | None:
    return core_resolve_available_operator_target(
        module_path,
        class_name,
        prefer_jax=prefer_jax,
    )


def _instantiate_operator_from_class(
    module_path: str,
    class_name: str,
    operator_type: str,
    **params: Any,
) -> Any:
    try:
        module = importlib.import_module(module_path)
        cls = getattr(module, class_name)
    except Exception as exc:  # noqa: BLE001
        print(f"Warning: Could not import {module_path}.{class_name}: {exc}")
        return "default"

    prepared: dict[str, Any] = {}
    if operator_type == "crossover":
        cx_eta = float(params.get("crossover_eta", 15))
        cx_prob = float(params.get("crossover_prob", 0.9))
        prepared = {
            "eta": cx_eta,
            "prob": cx_prob,
            "prob_var": cx_prob,
            "n_points": _positive_int(params.get("crossover_n_points", 2), 2),
        }
    elif operator_type == "mutation":
        m_eta = float(params.get("mutation_eta", 20))
        m_prob_raw = params.get("mutation_prob", None)
        prepared = {
            "eta": m_eta,
            "prob": None if m_prob_raw is None else float(m_prob_raw),
        }
    elif operator_type == "selection":
        prepared = {
            "func_comp": np.less,
            "pressure": _positive_int(params.get("selection_pressure", 2), 2),
        }

    kwargs: dict[str, Any] = {}
    try:
        sig = inspect.signature(cls.__init__)
        for name, param in list(sig.parameters.items())[1:]:
            if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
                continue
            if name in prepared and prepared[name] is not None:
                kwargs[name] = prepared[name]
    except Exception:  # noqa: BLE001
        kwargs = {}

    try:
        return cls(**kwargs)
    except Exception:
        try:
            return cls()
        except Exception as exc:  # noqa: BLE001
            print(f"Warning: Could not instantiate {module_path}.{class_name}: {exc}")
            return "default"


def create_pymoo_operator(
    operator_type: str,
    operator_name: str,
    n_obj: int = 2,
    pop_size: int = 100,
    **params: Any,
) -> Any:
    """
    Create a pymoo operator instance from its name or fully-qualified spec id.

    Args:
        operator_type: Type of operator ('crossover', 'mutation', 'selection', 'sampling')
        operator_name: Alias ('sbx'), spec id ('pymoo::module.Class'), or 'default'/'none'
        n_obj: Number of objectives (kept for compatibility)
        pop_size: Population size (kept for compatibility)
        **params: Operator hyperparameters from configuration.

    Returns:
        Instance of the operator, None for 'none', or 'default' to use pymoo defaults.
    """
    _ = n_obj
    _ = pop_size

    if operator_name == "default" or operator_name is None:
        return "default"

    if operator_name == "none":
        return None

    resolved = _resolve_operator_module_class(operator_type, str(operator_name))
    if resolved is None:
        return "default"

    module_path, class_name = resolved
    prefer_jax = bool(params.get("use_gpu", False)) or (
        str(params.get("array_backend", "")).strip().lower() in {"jax", "gpu"}
    )
    prefer_jax = bool(prefer_jax and core_rollout_allows_domain("operators"))
    target = _resolve_available_operator_target(
        module_path,
        class_name,
        prefer_jax=prefer_jax,
    )
    if target is None:
        return "default"

    module_path, class_name = target
    params_copy = dict(params)
    alias_name = str(operator_name).strip().lower()
    if operator_type == "selection" and alias_name == "best":
        params_copy["selection_pressure"] = max(
            10,
            _positive_int(params.get("selection_pressure", 10), 10),
        )
    elif operator_type == "selection" and alias_name == "random_binary":
        params_copy["selection_pressure"] = 1

    return _instantiate_operator_from_class(
        module_path,
        class_name,
        operator_type=operator_type,
        **params_copy,
    )


def instantiate_algorithm_class(cls: Any, config: dict[str, Any]) -> Any:
    """
    Instantiate a pymoo algorithm class with configuration from config dict.

    Parameters are read from config in priority order:
    1. Explicitly provided in config (e.g., crossover, mutation, selection)
    2. Computed from other config values (pop_size, n_obj, ref_dirs)
    3. pymoo default values (for operators like crossover, mutation, selection)

    Args:
        cls: The algorithm class to instantiate
        config: Configuration dictionary with algorithm parameters

    Returns:
        Instance of the algorithm class
    """
    pop_size = int(config.get("pop_size", 100))
    n_obj = int(config.get("n_obj", 2))
    n_inds = int(config.get("n_inds", pop_size))  # For some algorithms

    sig = inspect.signature(cls.__init__)
    kwargs: dict[str, Any] = {}

    # Build default prepared values - these are computed from config
    prepared = {
        "pop_size": pop_size,
        "n_inds": n_inds,
        "ref_dirs": build_reference_dirs(n_obj=n_obj, target=max(12, pop_size)),
        "n_neighbors": max(2, min(20, pop_size // 4)),
        "prob_neighbor_mating": 0.9,
        "weights": np.ones(max(1, n_obj), dtype=float) / max(1, n_obj),
        "ref_points": np.vstack(
            [np.full(max(1, n_obj), 0.2, dtype=float), np.full(max(1, n_obj), 0.8, dtype=float)]
        ),
    }

    # List of operator parameter names - these should NOT be passed as strings
    operator_params = ["crossover", "mutation", "selection", "repair", "sampling"]

    # Extract user-configurable operator parameters from config
    operator_extra_params: dict[str, Any] = {}
    for key in ("crossover_eta", "crossover_prob", "mutation_eta", "mutation_prob",
                "selection_pressure"):
        if key in config and config[key] is not None:
            operator_extra_params[key] = config[key]
    operator_extra_params["use_gpu"] = bool(config.get("use_gpu", False))
    operator_extra_params["array_backend"] = str(config.get("array_backend", "numpy"))

    # Algorithm operators from config - allow user to override defaults
    # These can be operator instances or string names (which get converted to instances)
    for op_param in operator_params:
        if op_param in config and config[op_param] is not None:
            op_value = config[op_param]
            # If it's a string, try to create the operator instance
            if isinstance(op_value, str):
                if op_value == "default":
                    continue
                else:
                    operator = create_pymoo_operator(
                        op_param, op_value,
                        n_obj=n_obj, pop_size=pop_size,
                        **operator_extra_params,
                    )
                    # Only set if we got a valid operator (not "default" string and not None)
                    if operator != "default" and operator is not None:
                        kwargs[op_param] = operator
            elif op_value is not None:
                # It's already an operator instance (or None for "none" option)
                kwargs[op_param] = op_value

    for name, param in list(sig.parameters.items())[1:]:
        if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            continue

        # If already set by operator_params, skip
        if name in kwargs:
            continue

        # Skill: CRITICAL FIX - Skip operator params in second loop to avoid passing strings
        # Operator params were already processed above and should not be passed as strings
        if name in operator_params:
            continue

        # Check if value exists in config first (user-specified)
        if name in config:
            kwargs[name] = config[name]
        # Otherwise use prepared values if available
        elif name in prepared:
            kwargs[name] = prepared[name]
        # If parameter has a default, skip it (pymoo will use its default)
        elif param.default is not inspect._empty:
            continue
        # Required parameter with no default - raise error
        else:
            raise RuntimeError(f"{cls.__name__} requires unsupported init parameter: '{name}'")

    return cls(**kwargs)


def instantiate_problem_class(cls: Any, config: dict[str, Any]) -> Any:
    has_n_var = "n_var" in config
    has_n_obj = "n_obj" in config
    n_var = int(config.get("n_var", 30))
    n_obj = int(config.get("n_obj", 2))
    sig = inspect.signature(cls.__init__)
    kwargs: dict[str, Any] = {}

    for name, param in list(sig.parameters.items())[1:]:
        if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            continue

        if name == "n_var":
            if has_n_var:
                kwargs[name] = n_var
                continue
            if param.default is inspect._empty:
                kwargs[name] = n_var
            continue

        if name == "n_obj":
            if has_n_obj:
                kwargs[name] = n_obj
                continue
            if param.default is inspect._empty:
                kwargs[name] = n_obj
            continue

        if param.default is inspect._empty:
            raise RuntimeError(f"{cls.__name__} requires unsupported init parameter: '{name}'")

    return cls(**kwargs)


def _extract_mapping_entries(candidate: Any) -> list[tuple[str, Any]]:
    if isinstance(candidate, dict):
        return [(str(k), v) for k, v in candidate.items()]
    if isinstance(candidate, (list, tuple)):
        out: list[tuple[str, Any]] = []
        for item in candidate:
            if isinstance(item, (list, tuple)) and len(item) == 2:
                out.append((str(item[0]), item[1]))
        return out
    return []


def _extract_class_names_from_module_source(
    module_name: str,
    base_name_hints: set[str],
) -> list[str]:
    try:
        spec = importlib.util.find_spec(module_name)
    except Exception:  # noqa: BLE001
        spec = None
    if spec is None or spec.origin is None:
        return []

    try:
        source = Path(spec.origin).read_text(encoding="utf-8")
    except Exception:  # noqa: BLE001
        return []

    try:
        tree = ast.parse(source, filename=spec.origin)
    except Exception:  # noqa: BLE001
        return []

    names: list[str] = []
    hint_lower = {hint.lower() for hint in base_name_hints}

    for node in tree.body:
        if not isinstance(node, ast.ClassDef):
            continue
        if node.name.startswith("_"):
            continue

        base_names: set[str] = set()
        for base in node.bases:
            if isinstance(base, ast.Name):
                base_names.add(base.id.lower())
            elif isinstance(base, ast.Attribute):
                base_names.add(base.attr.lower())

        if base_names & hint_lower:
            names.append(node.name)

    return names


def _extract_class_names_from_python_source(source: str, base_name_hints: set[str]) -> list[str]:
    if not source:
        return []
    try:
        tree = ast.parse(source)
    except Exception:  # noqa: BLE001
        return []

    names: list[str] = []
    hint_lower = {hint.lower() for hint in base_name_hints}
    for node in tree.body:
        if not isinstance(node, ast.ClassDef):
            continue
        if node.name.startswith("_"):
            continue
        base_names: set[str] = set()
        for base in node.bases:
            if isinstance(base, ast.Name):
                base_names.add(base.id.lower())
            elif isinstance(base, ast.Attribute):
                base_names.add(base.attr.lower())
        if base_names & hint_lower:
            names.append(node.name)
    return names


@lru_cache(maxsize=1)
def _fetch_emopylab_catalog_tree_paths() -> tuple[str, ...]:
    req = urllib.request.Request(
        EMOPYLAB_CATALOG_API_TREE_URL,
        headers={"User-Agent": EMOPYLAB_CATALOG_USER_AGENT},
    )
    try:
        with _urlopen_http_only(req, timeout=12.0) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except Exception:  # noqa: BLE001
        return tuple()

    paths: list[str] = []
    for entry in payload.get("tree", []) if isinstance(payload, dict) else []:
        if not isinstance(entry, dict):
            continue
        if entry.get("type") != "blob":
            continue
        path = str(entry.get("path", "")).strip()
        if path:
            paths.append(path)
    return tuple(sorted(paths))


def _collect_remote_module_names(
    folder_prefix: str,
    module_prefix: str,
) -> set[str]:
    modules: set[str] = set()
    normalized_prefix = folder_prefix.strip("/").replace("\\", "/") + "/"
    for path in _fetch_emopylab_catalog_tree_paths():
        if not path.startswith(normalized_prefix):
            continue
        if not path.endswith(".py"):
            continue
        if path.endswith("/__init__.py"):
            continue
        mod = path[:-3].replace("/", ".")
        if mod.startswith(module_prefix):
            modules.add(mod)
    return modules


def _module_name_to_repo_path(module_name: str) -> str | None:
    mod = str(module_name).strip()
    if not mod.startswith("pymoo."):
        return None
    return mod.replace(".", "/") + ".py"


@lru_cache(maxsize=2048)
def _fetch_emopylab_catalog_raw_source(repo_path: str) -> str:
    path = str(repo_path).strip().lstrip("/")
    if not path:
        return ""
    url = f"{EMOPYLAB_CATALOG_RAW_BASE_URL}{path}"
    req = urllib.request.Request(url, headers={"User-Agent": EMOPYLAB_CATALOG_USER_AGENT})
    try:
        with _urlopen_http_only(req, timeout=10.0) as response:
            return response.read().decode("utf-8", errors="ignore")
    except Exception:  # noqa: BLE001
        return ""


def _extract_class_names_from_remote_module_source(
    module_name: str,
    base_name_hints: set[str],
) -> list[str]:
    repo_path = _module_name_to_repo_path(module_name)
    if not repo_path:
        return []
    source = _fetch_emopylab_catalog_raw_source(repo_path)
    return _extract_class_names_from_python_source(source, base_name_hints)


def _is_expected_local_import_failure(message: str | None) -> bool:
    if not message:
        return False
    lowered = str(message).lower()
    expected_missing = (
        "no module named 'util'",
        "no module named 'core'",
        "no module named 'gradient'",
        "no module named 'vendor'",
        "no module named 'functions'",
    )
    return any(token in lowered for token in expected_missing)


def _is_expected_local_problem_probe_unavailable(message: str | None) -> bool:
    if not message:
        return False
    lowered = str(message).lower()
    expected_missing_runtime = (
        "could not locate matlab community problems folder",
        "matlab source folder for real-world mops not found",
        "scipy is required for realworld_mops problems",
    )
    return any(token in lowered for token in expected_missing_runtime)


def _call_factory_callable(factory: Callable[..., Any], config: dict[str, Any], role: str) -> Any:
    sig = inspect.signature(factory)
    params = [
        p
        for p in sig.parameters.values()
        if p.kind not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
    ]
    if not params:
        return factory()

    named = {
        "config": config,
        "cfg": config,
        "n_var": int(config.get("n_var", 30)),
        "n_obj": int(config.get("n_obj", 2)),
        "pop_size": int(config.get("pop_size", 100)),
        "ref_dirs": build_reference_dirs(int(config.get("n_obj", 2)), int(config.get("pop_size", 100))),
    }

    kwargs: dict[str, Any] = {}
    for p in params:
        if p.name in named:
            kwargs[p.name] = named[p.name]

    required_missing = [p.name for p in params if p.default is inspect._empty and p.name not in kwargs]
    if not required_missing:
        return factory(**kwargs)

    if len(params) == 1 and params[0].name not in kwargs:
        return factory(config)

    missing = ", ".join(required_missing)
    raise RuntimeError(f"Could not call {role} factory '{factory.__name__}'. Missing args: {missing}")


def _build_problem_from_pymoo_name(name: str, config: dict[str, Any]) -> Any:
    from problems import get_problem
    candidates: list[dict[str, Any]] = []
    n_var_raw = config.get("n_var")
    n_obj_raw = config.get("n_obj")
    if n_var_raw is not None and n_obj_raw is not None:
        candidates.append({"n_var": int(n_var_raw), "n_obj": int(n_obj_raw)})
    if n_var_raw is not None:
        candidates.append({"n_var": int(n_var_raw)})
    if n_obj_raw is not None:
        candidates.append({"n_obj": int(n_obj_raw)})
    candidates.append({})

    for kwargs in candidates:
        try:
            return get_problem(name, **kwargs)
        except Exception:  # noqa: BLE001
            continue
    return get_problem(name)


def _extract_problem_names_from_get_problem_source(source: str) -> set[str]:
    names: set[str] = set()
    if not source:
        return names

    try:
        tree = ast.parse(source)
    except Exception:  # noqa: BLE001
        return names

    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        target_ids = [t.id for t in node.targets if isinstance(t, ast.Name)]
        if "PROBLEM" not in target_ids or not isinstance(node.value, ast.Dict):
            continue
        for key in node.value.keys:
            if isinstance(key, ast.Constant) and isinstance(key.value, str):
                value = key.value.strip().lower()
                if value:
                    names.add(value)
        break
    return names


@lru_cache(maxsize=1)
def _fetch_problem_names_from_catalog_github() -> tuple[str, ...]:
    """
    Fetch the problem registry from upstream catalog reference.
    Falls back to empty list when offline or unavailable.
    """
    source = _fetch_emopylab_catalog_raw_source("pymoo/problems/__init__.py")
    if not source:
        return tuple()
    names = _extract_problem_names_from_get_problem_source(source)
    return tuple(sorted(names))

_fetch_problem_names_from_pymoo_github = _fetch_problem_names_from_catalog_github

def _parse_builtin_problem_names() -> list[str]:
    names: set[str] = set(DEFAULT_PROBLEM_DIMS)

    # 1) Prefer upstream pymoo main branch for the latest published registry.
    names.update(_fetch_problem_names_from_catalog_github())

    try:
        from problems.registry import list_problem_names
        names.update(list_problem_names())
    except Exception:  # noqa: BLE001
        pass
    return sorted(names)


def _infer_problem_dims(problem_name: str) -> tuple[int, int]:
    raw_name = str(problem_name).strip().lower()
    candidate_name = raw_name[:-4] if raw_name.endswith("_jax") else raw_name

    if candidate_name in DEFAULT_PROBLEM_DIMS:
        return DEFAULT_PROBLEM_DIMS[candidate_name]

    match = re.search(r"(dtlz[1-7]|wfg[1-9]|zdt[1-6])", candidate_name)
    if match is not None and match.group(1) in DEFAULT_PROBLEM_DIMS:
        return DEFAULT_PROBLEM_DIMS[match.group(1)]

    return 30, 2


def _build_local_jax_problem_factory(
    base_factory: Callable[[dict[str, Any]], Any],
    entry_name: str,
) -> Callable[[dict[str, Any]], Any]:
    def _jax_factory(cfg: dict[str, Any]) -> Any:
        base_problem = base_factory(cfg)
        gpu_dtype = str(cfg.get("gpu_dtype", "float64")).lower()
        wrapped = ExperimentBridge._try_create_jax_problem(
            base_problem,
            gpu_dtype=gpu_dtype,
            problem_name=f"{entry_name}_jax",
        )
        if wrapped is None:
            raise RuntimeError(f"Could not create JAX wrapper for problem '{entry_name}'.")
        return wrapped

    return _jax_factory


def _signature_default_positive_int(callable_obj: Any, parameter_name: str) -> int | None:
    try:
        signature = inspect.signature(callable_obj)
    except Exception:  # noqa: BLE001
        return None

    param = signature.parameters.get(str(parameter_name))
    if param is None or param.default is inspect._empty:
        return None

    value = _positive_int(param.default, 0, minimum=0)
    if value <= 0:
        return None
    return int(value)


def _signature_parameter_defaults_to_none(callable_obj: Any, parameter_name: str) -> bool:
    try:
        signature = inspect.signature(callable_obj)
    except Exception:  # noqa: BLE001
        return False

    param = signature.parameters.get(str(parameter_name))
    if param is None:
        return False
    return param.default is None


def _signature_parameter_required(callable_obj: Any, parameter_name: str) -> bool:
    try:
        signature = inspect.signature(callable_obj)
    except Exception:  # noqa: BLE001
        return False

    param = signature.parameters.get(str(parameter_name))
    if param is None:
        return False
    return param.default is inspect._empty


def _build_indicator_instance(cls: Any, context: dict[str, Any]) -> Any:
    sig = inspect.signature(cls.__init__)
    kwargs: dict[str, Any] = {}
    pf = context.get("pareto_front")

    prepared = {
        "ref_point": context.get("ref_point"),
        "reference_point": context.get("ref_point"),
        "pf": pf,
        "pareto_front": pf,
        "ref_pf": pf,
        "weights": np.ones(max(1, int(context.get("n_obj", 2))), dtype=float) / max(1, int(context.get("n_obj", 2))),
        "ideal": None if pf is None else np.min(pf, axis=0),
        "nadir": None if pf is None else np.max(pf, axis=0),
    }

    for name, param in list(sig.parameters.items())[1:]:
        if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            continue

        if name in prepared and prepared[name] is not None:
            kwargs[name] = prepared[name]
            continue

        if param.default is inspect._empty:
            if name in {"pf", "pareto_front", "ref_pf"}:
                raise RuntimeError(f"{cls.__name__} requires Pareto front, but none is available.")
            if name in {"ref_point", "reference_point"}:
                raise RuntimeError(f"{cls.__name__} requires ref_point, but none is available.")
            raise RuntimeError(f"{cls.__name__} requires unsupported init parameter: '{name}'")

    return cls(**kwargs)


def _build_metric_callable_from_entry(entry: Any, context: dict[str, Any]) -> Callable[[np.ndarray], float]:
    def _indicator_to_metric(indicator_obj: Any) -> Callable[[np.ndarray], float]:
        if callable(indicator_obj):
            indicator_call = cast(Callable[[np.ndarray], Any], indicator_obj)
            return lambda front: float(indicator_call(np.asarray(front, dtype=float)))

        indicator_do = getattr(indicator_obj, "do", None)
        if callable(indicator_do):
            do_call = cast(Callable[[np.ndarray], Any], indicator_do)
            return lambda front: float(do_call(np.asarray(front, dtype=float)))

        raise RuntimeError(f"Metric indicator is not callable: {type(indicator_obj)}")

    try:
        from metrics.indicators import Indicator as PymooIndicator
    except Exception:  # noqa: BLE001
        PymooIndicator = None  # type: ignore[assignment]

    if PymooIndicator is not None and inspect.isclass(entry) and issubclass(entry, PymooIndicator):
        indicator = _build_indicator_instance(entry, context)
        return _indicator_to_metric(indicator)

    if callable(entry):
        sig = inspect.signature(entry)
        params = [
            p
            for p in sig.parameters.values()
            if p.kind not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
        ]

        if params and params[0].name.lower() in {"front", "f", "values", "points"}:
            return lambda front: float(_call_metric_fn(entry, np.asarray(front, dtype=float), context))

        produced = _call_factory_callable(
            entry,
            {
                "config": context,
                "n_var": int(context.get("n_var", 30)),
                "n_obj": int(context.get("n_obj", 2)),
                "pop_size": int(context.get("pop_size", 100)),
            },
            role="metric",
        )

        if isinstance(produced, (int, float, np.floating)):
            value = float(produced)
            return lambda _front: value

        if callable(produced):
            return lambda front: float(_call_metric_fn(produced, np.asarray(front, dtype=float), context))

        if PymooIndicator is not None and isinstance(produced, PymooIndicator):
            return _indicator_to_metric(produced)

        raise RuntimeError(f"Metric factory returned unsupported object: {type(produced)}")

    raise RuntimeError(f"Unsupported metric entry type: {type(entry)}")


def _call_metric_fn(fn: Callable[..., Any], front: np.ndarray, context: dict[str, Any]) -> float:
    sig = inspect.signature(fn)
    params = [
        p
        for p in sig.parameters.values()
        if p.kind not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
    ]

    if not params:
        return float(fn())

    kwargs: dict[str, Any] = {}
    for p in params:
        name = p.name.lower()
        if name in {"front", "f", "values", "points"}:
            kwargs[p.name] = front
        elif name in {"context", "ctx", "config", "cfg"}:
            kwargs[p.name] = context

    missing = [p for p in params if p.default is inspect._empty and p.name not in kwargs]
    if not missing:
        return float(fn(**kwargs))

    if len(params) == 1:
        return float(fn(front))
    if len(params) == 2:
        return float(fn(front, context))

    missing_names = ", ".join(p.name for p in missing)
    raise RuntimeError(f"Metric callable '{fn.__name__}' missing required args: {missing_names}")


def _build_dynamic_hypervolume_metric(
    cls: Any,
    context: dict[str, Any],
    *,
    init_kwargs: dict[str, Any] | None = None,
) -> Callable[[np.ndarray], float]:
    ref_point = np.asarray(context.get("ref_point"), dtype=float)
    if ref_point.ndim != 1 or ref_point.size == 0:
        raise RuntimeError(f"{cls.__name__} requires a valid ref_point.")

    kwargs = dict(init_kwargs or {})
    if cls.__name__.lower().endswith("2d") and ref_point.size != 2:
        raise RuntimeError(f"{cls.__name__} is only available for bi-objective problems (n_obj=2).")

    def _metric(front: np.ndarray) -> float:
        values = np.asarray(front, dtype=float)
        if values.ndim == 1:
            values = values.reshape(1, -1)
        if values.size == 0:
            return float("nan")
        if values.shape[1] != ref_point.size:
            raise RuntimeError(
                f"{cls.__name__} expected {ref_point.size} objectives, got {values.shape[1]}."
            )

        indicator = cls(ref_point=np.asarray(ref_point, dtype=float), **kwargs)
        add_fn = getattr(indicator, "add", None)
        if callable(add_fn):
            add_fn(values)
            hv_value = getattr(indicator, "hv", None)
            if hv_value is not None:
                return float(hv_value)

        calc_fn = getattr(indicator, "calc", None)
        if callable(calc_fn):
            calc_value = calc_fn()
            if isinstance(calc_value, tuple):
                calc_value = calc_value[0]
            return float(cast(Any, calc_value))

        raise RuntimeError(f"{cls.__name__} does not expose add()/calc() for HV evaluation.")

    return _metric


def _register_builtin_hypervolume_specs(specs: dict[str, MetricSpec], warnings: list[str]) -> None:
    hv_specs: list[tuple[str, str, str, dict[str, Any] | None]] = [
        (
            "pymoo.indicators.hv.approximate",
            "ApproximateHypervolume",
            "HV Monte Carlo",
            {
                "n_samples_key": "hv_mc_samples",
                "n_samples_default": 10000,
                "n_exclusive_key": "hv_mc_exclusive",
                "n_exclusive_default": 1,
            },
        ),
    ]

    for module_name, class_name, display_name, mc_cfg in hv_specs:
        try:
            module = importlib.import_module(module_name)
            cls = getattr(module, class_name)
        except Exception as exc:  # noqa: BLE001
            warnings.append(f"Could not register {module_name}.{class_name}: {exc}")
            continue

        spec_id = f"pymoo::{module_name}.{class_name}"
        if spec_id in specs:
            continue

        if mc_cfg is None:
            factory = lambda context, cls=cls: _build_dynamic_hypervolume_metric(cls, context)
        else:
            factory = (
                lambda context, cls=cls, mc_cfg=mc_cfg: _build_dynamic_hypervolume_metric(
                    cls,
                    context,
                    init_kwargs={
                        "n_samples": _positive_int(
                            context.get(mc_cfg["n_samples_key"], mc_cfg["n_samples_default"]),
                            int(mc_cfg["n_samples_default"]),
                        ),
                        "n_exclusive": _positive_int(
                            context.get(mc_cfg["n_exclusive_key"], mc_cfg["n_exclusive_default"]),
                            int(mc_cfg["n_exclusive_default"]),
                        ),
                    },
                )
            )

        specs[spec_id] = MetricSpec(
            id=spec_id,
            name=display_name,
            source="pymoo",
            module=module_name,
            factory=factory,
        )


def discover_algorithm_specs(base_dir: Path, warnings: list[str]) -> dict[str, AlgorithmSpec]:
    specs: dict[str, AlgorithmSpec] = {}

    # Map class name -> display name from metadata for better UI
    algo_display_names = {item["class_name"]: item["name"] for item in getattr(emopylab_metadata, "EMOPYLAB_ALGORITHMS", emopylab_metadata.PYMOO_ALGORITHMS)}
    algo_display_names_norm = {
        _normalize_type_name_key(item["class_name"]): item["name"]
        for item in getattr(emopylab_metadata, "EMOPYLAB_ALGORITHMS", emopylab_metadata.PYMOO_ALGORITHMS)
    }

    try:
        from core.algorithm import Algorithm
        module_names: set[str] = set()

        try:
            import algorithms.moo as moo_algorithms
            module_names.update(
                mod_info.name
                for mod_info in pkgutil.walk_packages(moo_algorithms.__path__, prefix="algorithms.moo.")
            )
        except Exception as exc:  # noqa: BLE001
            pass

        module_names.update(_collect_remote_module_names("pymoo/algorithms", "pymoo.algorithms."))

        for module_name in sorted(module_names):
            if ".moo" not in module_name and ".soo" not in module_name:
                continue
            if any(f".{hint}" in module_name for hint in OPTIONAL_PYMOO_ALGORITHM_MODULE_HINTS):
                continue
            if any(f".{hint}" in module_name for hint in EXCLUDED_PYMOO_ALGORITHM_MODULE_HINTS):
                continue
            try:
                module = importlib.import_module(module_name)
            except Exception as exc:  # noqa: BLE001
                exc_msg = str(exc)
                is_optional_numba = "please install numba" in exc_msg.lower()
                msg_lower = exc_msg.lower()
                pure_missing_module = (
                    "no module named" in msg_lower
                    and (
                        module_name.lower() in msg_lower
                        or module_name.split(".")[-1].lower() in msg_lower
                    )
                )
                if not pure_missing_module and not is_optional_numba:
                    warnings.append(f"Could not import {module_name}: {exc}")

                fallback_classes = _extract_class_names_from_module_source(
                    module_name,
                    {"Algorithm", "GeneticAlgorithm", "EvolutionaryAlgorithm"},
                )
                if not fallback_classes:
                    fallback_classes = _extract_class_names_from_remote_module_source(
                        module_name,
                        {"Algorithm", "GeneticAlgorithm", "EvolutionaryAlgorithm"},
                    )
                for class_name in fallback_classes:
                    spec_id = f"pymoo::{module_name}.{class_name}"
                    if spec_id in specs:
                        continue

                    display_name = algo_display_names.get(
                        class_name,
                        algo_display_names_norm.get(_normalize_type_name_key(class_name), class_name),
                    )

                    def _lazy_algo_factory(
                        cfg: dict[str, Any],
                        module_name: str = module_name,
                        class_name: str = class_name,
                    ) -> Any:
                        module_local = importlib.import_module(module_name)
                        cls_local = getattr(module_local, class_name)
                        return instantiate_algorithm_class(cls_local, cfg)

                    specs[spec_id] = AlgorithmSpec(
                        id=spec_id,
                        name=display_name,
                        source="pymoo",
                        module=module_name,
                        factory=_lazy_algo_factory,
                        flags=_patch_known_pymoo_algorithm_flags(display_name, set()),
                    )
                continue

            for class_name, cls in inspect.getmembers(module, inspect.isclass):
                if class_name.startswith("_"):
                    continue
                if cls.__module__ != module_name:
                    continue
                try:
                    is_algo = issubclass(cls, Algorithm)
                except Exception:  # noqa: BLE001
                    is_algo = False
                if not is_algo or cls is Algorithm:
                    continue

                spec_id = f"pymoo::{cls.__module__}.{cls.__name__}"
                if spec_id in specs:
                    continue

                display_name = algo_display_names.get(
                    cls.__name__,
                    algo_display_names_norm.get(_normalize_type_name_key(cls.__name__), cls.__name__),
                )

                specs[spec_id] = AlgorithmSpec(
                    id=spec_id,
                    name=display_name,
                    source="pymoo",
                    module=cls.__module__,
                    factory=lambda cfg, cls=cls: instantiate_algorithm_class(cls, cfg),
                    flags=_patch_known_pymoo_algorithm_flags(
                        display_name,
                        _collect_algorithm_flags(module, cls.__name__, cls),
                    ),
                )
    except Exception as exc:  # noqa: BLE001
        warnings.append(f"pymoo algorithms discovery failed: {exc}")

    folder = base_dir / "algorithms"
    for py_file in sorted(folder.rglob("*.py")):
        rel_file = py_file.relative_to(folder)
        if "__pycache__" in rel_file.parts:
            continue
        if rel_file.parts and rel_file.parts[0] in LOCAL_ALGORITHM_EXCLUDED_ROOTS:
            continue
        if py_file.name.startswith("_") or py_file.name == "__init__.py":
            continue
        if py_file.name in LOCAL_ALGORITHM_EXCLUDED_FILES:
            continue

        module = None
        err = None
        rel_mod = ".".join(rel_file.with_suffix("").parts)
        module_tag = rel_file.as_posix()

        # Prefer package import so local relative imports from helper modules work.
        import_candidates = [
            f"algorithms.{rel_mod}",
            f"{base_dir.name}.algorithms.{rel_mod}",
        ]
        tried: set[str] = set()
        for candidate in import_candidates:
            if candidate in tried:
                continue
            tried.add(candidate)
            try:
                module = importlib.import_module(candidate)
                err = None
                break
            except Exception as exc:  # noqa: BLE001
                err = str(exc)

        if module is None:
            module, file_err = import_module_from_file(py_file, "local_alg")
            if module is None:
                final_err = file_err if file_err is not None else err
                if not _is_expected_local_import_failure(final_err):
                    warnings.append(f"Could not load local algorithm module {module_tag}: {final_err}")
                continue

        entries: list[tuple[str, Any]] = []
        if hasattr(module, "get_algorithms"):
            try:
                entries.extend(_extract_mapping_entries(module.get_algorithms()))
            except Exception as exc:  # noqa: BLE001
                warnings.append(f"{module_tag}.get_algorithms() failed: {exc}")
        if hasattr(module, "ALGORITHMS"):
            entries.extend(_extract_mapping_entries(getattr(module, "ALGORITHMS")))

        if hasattr(module, "create_algorithm") and callable(module.create_algorithm):
            entries.append((py_file.stem, module.create_algorithm))

        try:
            from core.algorithm import Algorithm
            known_entries = {id(obj) for _, obj in entries}

            for class_name, cls in inspect.getmembers(module, inspect.isclass):
                if class_name.startswith("_"):
                    continue
                if cls.__module__ != module.__name__:
                    continue
                if id(cls) in known_entries:
                    continue
                try:
                    if issubclass(cls, Algorithm) and cls is not Algorithm:
                        entries.append((class_name, cls))
                except Exception:  # noqa: BLE001
                    continue
        except Exception:  # noqa: BLE001
            pass

        for entry_name, entry_obj in entries:
            spec_id = f"local::{module.__name__}.{entry_name}"
            if spec_id in specs:
                continue
            entry_flags = _collect_algorithm_flags(module, str(entry_name), entry_obj)

            def _local_factory(cfg: dict[str, Any], entry_obj: Any = entry_obj) -> Any:
                if inspect.isclass(entry_obj):
                    return instantiate_algorithm_class(entry_obj, cfg)
                if callable(entry_obj):
                    return _call_factory_callable(entry_obj, cfg, role="algorithm")
                    raise RuntimeError(f"Unsupported local algorithm entry type: {type(entry_obj)}")

            specs[spec_id] = AlgorithmSpec(
                id=spec_id,
                name=str(entry_name),
                source=SOURCE_GUIPYMOO,
                module=module.__name__,
                factory=_local_factory,
                flags=entry_flags,
            )

    return specs

def discover_problem_specs(base_dir: Path, warnings: list[str]) -> dict[str, ProblemSpec]:
    specs: dict[str, ProblemSpec] = {}
    chosen_by_name: dict[str, tuple[str, tuple[int, int, int]]] = {}
    dedup_dropped = 0
    unavailable_runtime_warnings_emitted: set[str] = set()

    sync_marker_prefix = "Generated by tools/sync_"
    sync_marker_suffix = "_multiobjective.py"
    file_marker_cache: dict[Path, bool] = {}

    def _has_sync_marker(path: Path) -> bool:
        path = path.resolve()
        if path in file_marker_cache:
            return file_marker_cache[path]
        has_marker = False
        try:
            text = path.read_text(encoding="utf-8-sig", errors="ignore")
            has_marker = sync_marker_prefix in text and sync_marker_suffix in text
        except Exception:  # noqa: BLE001
            has_marker = False
        file_marker_cache[path] = bool(has_marker)
        return bool(has_marker)

    def _is_sync_generated_module_file(path: Path) -> bool:
        # Directly generated source file.
        if _has_sync_marker(path):
            return True

        # *_JAX generated counterpart of a sync-generated source file.
        if path.stem.upper().endswith("_JAX"):
            base_stem = path.stem[:-4]
            src_file = path.with_name(f"{base_stem}.py")
            if src_file.exists() and _has_sync_marker(src_file):
                return True
        return False

    def _entry_priority(*, entry_name: str, py_file: Path, module_sync_generated: bool) -> tuple[int, int, int]:
        name_upper = str(entry_name).strip().upper()
        module_is_jax = py_file.stem.upper().endswith("_JAX")
        entry_is_jax = name_upper.endswith("_JAX")

        # Higher tuple is better.
        return (
            1 if not module_sync_generated else 0,      # Prefer canonical (non sync-generated) modules.
            1 if entry_is_jax == module_is_jax else 0,  # Prefer suffix-aligned entries.
            1 if not module_is_jax else 0,              # Prefer CPU module for non-JAX names.
        )

    folder = base_dir / "problems"
    for py_file in sorted(folder.rglob("*.py")):
        rel_file = py_file.relative_to(folder)
        if "__pycache__" in rel_file.parts:
            continue
        if py_file.name.startswith("_") or py_file.name == "__init__.py":
            continue

        module = None
        err = None
        rel_mod = ".".join(rel_file.with_suffix("").parts)
        module_tag = rel_file.as_posix()

        import_candidates = [
            f"problems.{rel_mod}",
            f"{base_dir.name}.problems.{rel_mod}",
        ]
        tried: set[str] = set()
        for candidate in import_candidates:
            if candidate in tried:
                continue
            tried.add(candidate)
            try:
                module = importlib.import_module(candidate)
                err = None
                break
            except Exception as exc:  # noqa: BLE001
                err = str(exc)

        if module is None:
            module, file_err = import_module_from_file(py_file, "local_prob")
            if module is None:
                err = file_err if file_err is not None else err

        if module is None:
            if not _is_expected_local_import_failure(err):
                warnings.append(f"Could not load local problem module {module_tag}: {err}")
            continue

        module_sync_generated = _is_sync_generated_module_file(py_file)
        module_is_jax_file = py_file.stem.upper().endswith("_JAX")

        entries: list[tuple[str, Any]] = []
        if hasattr(module, "get_problems"):
            try:
                entries.extend(_extract_mapping_entries(module.get_problems()))
            except Exception as exc:  # noqa: BLE001
                warnings.append(f"{module_tag}.get_problems() failed: {exc}")
        if hasattr(module, "PROBLEMS"):
            entries.extend(_extract_mapping_entries(getattr(module, "PROBLEMS")))

        if hasattr(module, "create_problem") and callable(module.create_problem):
            entries.append((py_file.stem, module.create_problem))

        try:
            from core.problem import Problem
            for class_name, cls in inspect.getmembers(module, inspect.isclass):
                if class_name.startswith("_"):
                    continue
                if cls.__module__ != module.__name__:
                    continue
                try:
                    if issubclass(cls, Problem) and cls is not Problem:
                        entries.append((class_name, cls))
                except Exception:  # noqa: BLE001
                    continue
        except Exception:  # noqa: BLE001
            pass

        entry_names_lower = {str(name).strip().lower() for name, _ in entries}
        entry_names_compact = [
            _canonical_problem_name(str(name)).replace("_", "").lower()
            for name, _ in entries
        ]
        has_sibling_jax_file = py_file.with_name(f"{py_file.stem}_JAX.py").exists()

        for entry_name, entry_obj in entries:
            entry_name_raw = str(entry_name)
            entry_name_canonical = _canonical_problem_name(entry_name_raw)
            entry_name_compact = entry_name_canonical.replace("_", "").lower()

            # Skip family/base classes (e.g., ZDT/WFG/DASCMOP) when the module also
            # contains numbered variants in the same family. These base classes are
            # helpers, not benchmark problems, and can generate invalid defaults/JAX entries.
            if inspect.isclass(entry_obj):
                module_stem_compact = py_file.stem.replace("_", "").lower()
                if entry_name_compact == module_stem_compact:
                    has_numbered_sibling = any(
                        other != entry_name_compact
                        and other.startswith(entry_name_compact)
                        and any(ch.isdigit() for ch in other[len(entry_name_compact):])
                        for other in entry_names_compact
                    )
                    if has_numbered_sibling:
                        dedup_dropped += 1
                        continue

            # Avoid duplicating CPU names from *_JAX modules. Keep only explicit *_JAX names there.
            if module_is_jax_file and not entry_name_raw.strip().upper().endswith("_JAX"):
                dedup_dropped += 1
                continue

            spec_id = f"local::{module.__name__}.{entry_name_raw}"
            if spec_id in specs:
                continue

            def _local_problem_factory(cfg: dict[str, Any], entry_obj: Any = entry_obj) -> Any:
                if inspect.isclass(entry_obj):
                    return instantiate_problem_class(entry_obj, cfg)
                if callable(entry_obj):
                    return _call_factory_callable(entry_obj, cfg, role="problem")
                raise RuntimeError(f"Unsupported local problem entry type: {type(entry_obj)}")

            default_n_var, default_n_obj = _infer_problem_dims(entry_name_raw)
            signature_target = entry_obj.__init__ if inspect.isclass(entry_obj) else entry_obj
            default_from_sig_n_var = _signature_default_positive_int(signature_target, "n_var")
            default_from_sig_n_obj = _signature_default_positive_int(signature_target, "n_obj")
            if default_from_sig_n_var is not None:
                default_n_var = int(default_from_sig_n_var)
            if default_from_sig_n_obj is not None:
                default_n_obj = int(default_from_sig_n_obj)
            n_var_defaults_to_none = _signature_parameter_defaults_to_none(signature_target, "n_var")
            n_obj_defaults_to_none = _signature_parameter_defaults_to_none(signature_target, "n_obj")
            n_var_required = _signature_parameter_required(signature_target, "n_var")
            n_obj_required = _signature_parameter_required(signature_target, "n_obj")

            probe_candidates: list[dict[str, Any]] = []
            # Some local benchmark families use (n_var=None, n_obj=None) and derive the
            # canonical defaults from an internal family index. In those cases, forcing the
            # heuristic fallback n_obj=2 during probing can lock the catalog to a wrong
            # objective count (for example 3-objective variants discovered as 2-objective).
            # Prefer the default constructor first when both dimensions default to None.
            prefer_default_ctor_probe = bool(
                inspect.isclass(entry_obj)
                and n_var_defaults_to_none
                and n_obj_defaults_to_none
                and not n_var_required
                and not n_obj_required
            )
            if prefer_default_ctor_probe:
                probe_candidates.append({})

            # Prefer sparse probing when constructor computes one dimension from
            # the other (for example n_var=None -> n_var=f(n_obj)).
            if n_obj_defaults_to_none and default_n_var > 0:
                probe_candidates.append({"n_var": default_n_var})
            if n_var_defaults_to_none and default_n_obj > 0:
                probe_candidates.append({"n_obj": default_n_obj})

            # If both parameters are mandatory, avoid sparse configs that would fail.
            if n_var_required and n_obj_required:
                if default_n_var > 0 and default_n_obj > 0:
                    probe_candidates.append({"n_var": default_n_var, "n_obj": default_n_obj})
            else:
                if default_n_var > 0 and default_n_obj > 0:
                    probe_candidates.append({"n_var": default_n_var, "n_obj": default_n_obj})
                if default_n_var > 0 and not n_obj_required:
                    probe_candidates.append({"n_var": default_n_var})
                if default_n_obj > 0 and not n_var_required:
                    probe_candidates.append({"n_obj": default_n_obj})
                if not n_var_required and not n_obj_required:
                    probe_candidates.append({})

            seen_probe_keys: set[tuple[tuple[str, Any], ...]] = set()
            probe_failed_messages: list[str] = []
            probe_succeeded = False
            for probe_cfg in probe_candidates:
                probe_key = tuple(sorted(probe_cfg.items()))
                if probe_key in seen_probe_keys:
                    continue
                seen_probe_keys.add(probe_key)
                try:
                    probe = _local_problem_factory(probe_cfg)
                    probed_n_var = int(getattr(probe, "n_var", default_n_var))
                    probed_n_obj = int(getattr(probe, "n_obj", default_n_obj))
                    if probed_n_var > 0:
                        default_n_var = probed_n_var
                    if probed_n_obj > 0:
                        default_n_obj = probed_n_obj
                    probe_succeeded = True
                    break
                except Exception as exc:  # noqa: BLE001
                    probe_failed_messages.append(str(exc))
                    continue

            if not probe_succeeded and any(
                _is_expected_local_problem_probe_unavailable(msg) for msg in probe_failed_messages
            ):
                if module_tag not in unavailable_runtime_warnings_emitted:
                    sample_msg = next(
                        (msg for msg in probe_failed_messages if _is_expected_local_problem_probe_unavailable(msg)),
                        "external runtime dependency unavailable",
                    )
                    warnings.append(
                        f"Skipping local problem module {module_tag}: {sample_msg}"
                    )
                    unavailable_runtime_warnings_emitted.add(module_tag)
                continue

            candidate_spec = ProblemSpec(
                id=spec_id,
                name=entry_name_canonical,
                source=SOURCE_GUIPYMOO,
                module=module.__name__,
                default_n_var=default_n_var,
                default_n_obj=default_n_obj,
                factory=_local_problem_factory,
            )

            entry_name_key = _normalize_type_name_key(entry_name_canonical)
            cand_priority = _entry_priority(
                entry_name=entry_name_canonical,
                py_file=py_file,
                module_sync_generated=module_sync_generated,
            )
            previous = chosen_by_name.get(entry_name_key)
            if previous is not None:
                prev_spec_id, prev_priority = previous
                if cand_priority <= prev_priority:
                    dedup_dropped += 1
                    continue
                specs.pop(prev_spec_id, None)
                dedup_dropped += 1

            specs[spec_id] = candidate_spec
            chosen_by_name[entry_name_key] = (spec_id, cand_priority)

            base_name = entry_name_canonical
            jax_name = f"{base_name}_JAX"
            if (
                not base_name.upper().endswith("_JAX")
                and not has_sibling_jax_file
                and jax_name.lower() not in entry_names_lower
            ):
                jax_name = f"{base_name}_JAX"
                jax_spec_id = f"local::{module.__name__}.{jax_name}"
                if jax_spec_id not in specs:
                    jax_candidate = ProblemSpec(
                        id=jax_spec_id,
                        name=jax_name,
                        source=SOURCE_GUIPYMOO,
                        module=module.__name__,
                        default_n_var=default_n_var,
                        default_n_obj=default_n_obj,
                        factory=_build_local_jax_problem_factory(_local_problem_factory, base_name),
                    )
                    jax_name_key = _normalize_type_name_key(jax_name)
                    jax_priority = _entry_priority(
                        entry_name=jax_name,
                        py_file=py_file,
                        module_sync_generated=module_sync_generated,
                    )
                    previous_jax = chosen_by_name.get(jax_name_key)
                    if previous_jax is not None:
                        prev_spec_id, prev_priority = previous_jax
                        if jax_priority <= prev_priority:
                            dedup_dropped += 1
                            continue
                        specs.pop(prev_spec_id, None)
                        dedup_dropped += 1

                    specs[jax_spec_id] = jax_candidate
                    chosen_by_name[jax_name_key] = (jax_spec_id, jax_priority)

    matlab_catalog = _discover_matlab_problem_catalog(base_dir, warnings)
    placeholder_added = 0
    for meta in matlab_catalog:
        raw_name = str(meta.get("name", "")).strip()
        if not raw_name:
            continue
        name = _canonical_problem_name(raw_name)
        key = _normalize_type_name_key(name)
        if key in chosen_by_name:
            continue

        source_rel = str(meta.get("source_rel", "")).replace("\\", "/")
        if not source_rel:
            continue
        module_hint = source_rel[:-2].replace("/", ".") if source_rel.lower().endswith(".m") else source_rel.replace("/", ".")
        spec_id = f"matlab::{source_rel}::{name}"
        if spec_id in specs:
            continue

        default_n_var = int(max(1, int(meta.get("default_n_var", 10))))
        default_n_obj = int(max(1, int(meta.get("default_n_obj", 2))))

        def _missing_problem_factory(
            cfg: dict[str, Any],
            *,
            problem_name: str = name,
            matlab_source: str = source_rel,
        ) -> Any:
            _ = cfg
            raise RuntimeError(
                f"Problem '{problem_name}' has not been converted to Python yet. "
                f"MATLAB source: {matlab_source}"
            )

        specs[spec_id] = ProblemSpec(
            id=spec_id,
            name=name,
            source=SOURCE_GUIPYMOO,
            module=f"problems._matlab_source_catalog.{module_hint}",
            default_n_var=default_n_var,
            default_n_obj=default_n_obj,
            factory=_missing_problem_factory,
        )
        chosen_by_name[key] = (spec_id, (-1, -1, -1))
        placeholder_added += 1

        if not name.upper().endswith("_JAX"):
            jax_name = f"{name}_JAX"
            jax_key = _normalize_type_name_key(jax_name)
            if jax_key not in chosen_by_name:
                jax_spec_id = f"{spec_id}::JAX"

                def _missing_problem_factory_jax(
                    cfg: dict[str, Any],
                    *,
                    problem_name: str = jax_name,
                    matlab_source: str = source_rel,
                ) -> Any:
                    _ = cfg
                    raise RuntimeError(
                        f"Problem '{problem_name}' has not been converted to Python yet. "
                        f"MATLAB source: {matlab_source}"
                    )

                specs[jax_spec_id] = ProblemSpec(
                    id=jax_spec_id,
                    name=jax_name,
                    source=SOURCE_GUIPYMOO,
                    module=f"problems._matlab_source_catalog.{module_hint}_JAX",
                    default_n_var=default_n_var,
                    default_n_obj=default_n_obj,
                    factory=_missing_problem_factory_jax,
                )
                chosen_by_name[jax_key] = (jax_spec_id, (-1, -1, -1))
                placeholder_added += 1

    if placeholder_added > 0:
        warnings.append(
            f"Loaded {placeholder_added} MATLAB catalog placeholder(s) awaiting Python conversion."
        )

    if dedup_dropped > 0:
        warnings.append(f"Problem deduplication removed {dedup_dropped} duplicate entry candidate(s).")

    return specs


def discover_metric_specs(base_dir: Path, warnings: list[str]) -> dict[str, MetricSpec]:
    specs: dict[str, MetricSpec] = {}

    try:
        from metrics.indicators import Indicator
        official_metrics = getattr(emopylab_metadata, "EMOPYLAB_METRICS", getattr(emopylab_metadata, "PYMOO_METRICS", {}))
        module_names: set[str] = set()
        try:
            import metrics as metrics_pkg
            module_names.update(
                mod_info.name
                for mod_info in pkgutil.walk_packages(metrics_pkg.__path__, prefix="metrics.")
            )
        except Exception:
            pass
        module_names.update(_collect_remote_module_names("pymoo/indicators", "pymoo.indicators."))
        for module_name in sorted(module_names):
            try:
                module = importlib.import_module(module_name)
            except Exception as exc:  # noqa: BLE001
                exc_msg = str(exc)
                msg_lower = exc_msg.lower()
                pure_missing_module = (
                    "no module named" in msg_lower
                    and (
                        module_name.lower() in msg_lower
                        or module_name.split(".")[-1].lower() in msg_lower
                    )
                )
                if not pure_missing_module:
                    warnings.append(f"Could not import {module_name}: {exc}")

                fallback_classes = _extract_class_names_from_module_source(module_name, {"Indicator"})
                if not fallback_classes:
                    fallback_classes = _extract_class_names_from_remote_module_source(module_name, {"Indicator"})

                for class_name in fallback_classes:
                    spec_id = f"pymoo::{module_name}.{class_name}"
                    if spec_id in specs:
                        continue

                    display_name = str(official_metrics.get(class_name, class_name))
                    display_key = re.sub(r"\s+", " ", display_name.strip().lower())
                    if display_key in pymoo_metric_name_keys:
                        continue
                    pymoo_metric_name_keys.add(display_key)

                    def _lazy_metric_factory(
                        context: dict[str, Any],
                        module_name: str = module_name,
                        class_name: str = class_name,
                    ) -> Callable[[np.ndarray], float]:
                        module_local = importlib.import_module(module_name)
                        cls_local = getattr(module_local, class_name)
                        return _build_metric_callable_from_entry(cls_local, context)

                    specs[spec_id] = MetricSpec(
                        id=spec_id,
                        name=display_name,
                        source=SOURCE_PYMOO,
                        module=module_name,
                        factory=_lazy_metric_factory,
                    )
                continue

            for class_name, cls in inspect.getmembers(module, inspect.isclass):
                if class_name.startswith("_"):
                    continue
                if cls.__module__ != module_name:
                    continue
                try:
                    is_indicator = issubclass(cls, Indicator)
                except Exception:  # noqa: BLE001
                    is_indicator = False
                if not is_indicator or cls is Indicator:
                    continue

                spec_id = f"pymoo::{cls.__module__}.{cls.__name__}"
                if spec_id in specs:
                    continue

                display_name = str(official_metrics.get(class_name, class_name))
                display_key = re.sub(r"\s+", " ", display_name.strip().lower())
                if display_key in pymoo_metric_name_keys:
                    continue
                pymoo_metric_name_keys.add(display_key)
                specs[spec_id] = MetricSpec(
                    id=spec_id,
                    name=display_name,
                    source=SOURCE_PYMOO,
                    module=cls.__module__,
                    factory=lambda context, cls=cls: (
                        lambda front: float(_build_indicator_instance(cls, context)(np.asarray(front, dtype=float)))
                    ),
                )
    except Exception as exc:  # noqa: BLE001
        warnings.append(f"pymoo indicators discovery failed: {exc}")

    _register_builtin_hypervolume_specs(specs, warnings)

    local_metric_name_keys: set[str] = set()
    for relative_folder in LOCAL_METRIC_FOLDERS:
        folder = base_dir / relative_folder
        if not folder.exists():
            continue

        for py_file in sorted(folder.rglob("*.py")):
            rel_file = py_file.relative_to(folder)
            if "__pycache__" in rel_file.parts:
                continue
            if py_file.name.startswith("_") or py_file.name == "__init__.py":
                continue

            module = None
            err = None
            rel_mod = ".".join(rel_file.with_suffix("").parts)
            module_tag = f"{relative_folder}/{rel_file.as_posix()}"

            import_candidates = [
                f"metrics.{rel_mod}",
                f"{base_dir.name}.metrics.{rel_mod}",
            ]
            tried: set[str] = set()
            for candidate in import_candidates:
                if candidate in tried:
                    continue
                tried.add(candidate)
                try:
                    module = importlib.import_module(candidate)
                    err = None
                    break
                except Exception as exc:  # noqa: BLE001
                    err = str(exc)

            if module is None:
                module, file_err = import_module_from_file(py_file, "local_metric")
                if module is None:
                    err = file_err if file_err is not None else err

            if module is None:
                if not _is_expected_local_import_failure(err):
                    warnings.append(f"Could not load local metric module {module_tag}: {err}")
                continue

            entries: list[tuple[str, Any]] = []
            if hasattr(module, "get_metrics"):
                try:
                    entries.extend(_extract_mapping_entries(module.get_metrics()))
                except Exception as exc:  # noqa: BLE001
                    warnings.append(f"{module_tag}.get_metrics() failed: {exc}")
            if hasattr(module, "METRICS"):
                entries.extend(_extract_mapping_entries(getattr(module, "METRICS")))

            if hasattr(module, "create_metric") and callable(module.create_metric):
                entries.append((py_file.stem, module.create_metric))

            try:
                from metrics.indicators import Indicator

                for class_name, cls in inspect.getmembers(module, inspect.isclass):
                    if class_name.startswith("_"):
                        continue
                    if cls.__module__ != module.__name__:
                        continue
                    try:
                        if issubclass(cls, Indicator) and cls is not Indicator:
                            entries.append((class_name, cls))
                    except Exception:  # noqa: BLE001
                        continue
            except Exception:  # noqa: BLE001
                pass

            for entry_name, entry_obj in entries:
                name_key = str(entry_name).strip().lower()
                if name_key in local_metric_name_keys:
                    continue

                spec_id = f"local::{module.__name__}.{entry_name}"
                if spec_id in specs:
                    continue

                local_metric_name_keys.add(name_key)
                specs[spec_id] = MetricSpec(
                    id=spec_id,
                    name=str(entry_name),
                    source=SOURCE_GUIPYMOO,
                    module=module.__name__,
                    factory=lambda context, entry_obj=entry_obj: _build_metric_callable_from_entry(entry_obj, context),
                )

    # Project policy: use only local/manual metric implementations.
    specs = {
        sid: spec
        for sid, spec in specs.items()
        if str(getattr(spec, "source", "")).strip().lower() != str(SOURCE_PYMOO).strip().lower()
    }

    return specs


def discover_operator_specs(
    warnings: list[str],
    base_dir: Path | None = None,
) -> dict[str, dict[str, OperatorSpec]]:
    specs: dict[str, dict[str, OperatorSpec]] = {
        category: {} for category in OPERATOR_CATEGORY_META
    }
    base_cls_by_category: dict[str, Any] = {}

    for category, (package_name, base_module_name, base_class_name) in OPERATOR_CATEGORY_META.items():
        try:
            package = importlib.import_module(package_name)
            base_module = importlib.import_module(base_module_name)
            base_cls = getattr(base_module, base_class_name)
            base_cls_by_category[category] = base_cls
        except Exception as exc:  # noqa: BLE001
            warnings.append(f"pymoo operator discovery failed for {category}: {exc}")
            continue

        try:
            module_names: set[str] = set(
                mod_info.name
                for mod_info in pkgutil.walk_packages(package.__path__, prefix=f"{package_name}.")
            )
            module_names.update(
                _collect_remote_module_names(package_name.replace(".", "/"), f"{package_name}.")
            )
        except Exception as exc:  # noqa: BLE001
            warnings.append(f"Could not enumerate {package_name}: {exc}")
            continue

        for module_name in sorted(module_names):
            try:
                module = importlib.import_module(module_name)
            except Exception as exc:  # noqa: BLE001
                exc_msg = str(exc)
                msg_lower = exc_msg.lower()
                pure_missing_module = (
                    "no module named" in msg_lower
                    and (
                        module_name.lower() in msg_lower
                        or module_name.split(".")[-1].lower() in msg_lower
                    )
                )
                if not pure_missing_module:
                    warnings.append(f"Could not import {module_name}: {exc}")

                fallback_classes = _extract_class_names_from_module_source(module_name, {base_class_name})
                if not fallback_classes:
                    fallback_classes = _extract_class_names_from_remote_module_source(
                        module_name,
                        {base_class_name},
                    )

                for class_name in fallback_classes:
                    spec_id = f"pymoo::{module_name}.{class_name}"
                    if spec_id in specs[category]:
                        continue

                    specs[category][spec_id] = OperatorSpec(
                        id=spec_id,
                        name=class_name,
                        source=SOURCE_PYMOO,
                        category=category,
                        module=module_name,
                        class_name=class_name,
                    )
                continue

            for class_name, cls in inspect.getmembers(module, inspect.isclass):
                if class_name.startswith("_") or cls.__module__ != module_name:
                    continue
                try:
                    if not issubclass(cls, base_cls) or cls is base_cls:
                        continue
                except Exception:  # noqa: BLE001
                    continue

                spec_id = f"pymoo::{module_name}.{class_name}"
                if spec_id in specs[category]:
                    continue

                specs[category][spec_id] = OperatorSpec(
                    id=spec_id,
                    name=class_name,
                    source="pymoo",
                    category=category,
                    module=module_name,
                    class_name=class_name,
                )

    if base_dir is not None:
        local_ops_root = base_dir / "operators"
        if local_ops_root.exists():
            for py_file in sorted(local_ops_root.rglob("*.py")):
                rel_file = py_file.relative_to(local_ops_root)
                if "__pycache__" in rel_file.parts:
                    continue
                if py_file.name.startswith("_") or py_file.name == "__init__.py":
                    continue

                module, err = import_module_from_file(py_file, "local_op")
                if module is None:
                    if not _is_expected_local_import_failure(err):
                        warnings.append(f"Could not load local operator module {rel_file.as_posix()}: {err}")
                    continue

                for class_name, cls in inspect.getmembers(module, inspect.isclass):
                    if class_name.startswith("_"):
                        continue
                    if cls.__module__ != module.__name__:
                        continue

                    resolved_category: str | None = None
                    for category, base_cls in base_cls_by_category.items():
                        try:
                            if issubclass(cls, base_cls) and cls is not base_cls:
                                resolved_category = category
                                break
                        except Exception:  # noqa: BLE001
                            continue

                    if resolved_category is None:
                        continue

                    spec_id = f"local::{module.__name__}.{class_name}"
                    if spec_id in specs[resolved_category]:
                        continue

                    specs[resolved_category][spec_id] = OperatorSpec(
                        id=spec_id,
                        name=class_name,
                        source=SOURCE_LOCAL,
                        category=resolved_category,
                        module=module.__name__,
                        class_name=class_name,
                    )

    for category, alias_map in LEGACY_OPERATOR_ALIASES.items():
        category_specs = specs.setdefault(category, {})
        for module_name, class_name in alias_map.values():
            spec_id = f"pymoo::{module_name}.{class_name}"
            if spec_id in category_specs:
                continue
            try:
                module = importlib.import_module(module_name)
                _ = getattr(module, class_name)
            except Exception:
                continue
            category_specs[spec_id] = OperatorSpec(
                id=spec_id,
                name=class_name,
                source="pymoo",
                category=category,
                module=module_name,
                class_name=class_name,
            )

    return specs


def discover_all_specs(base_dir: Path) -> tuple[
    dict[str, AlgorithmSpec],
    dict[str, ProblemSpec],
    dict[str, MetricSpec],
    list[str],
]:
    ensure_plugin_directories(base_dir)
    warnings: list[str] = []
    algorithm_specs = discover_algorithm_specs(base_dir, warnings)
    problem_specs = discover_problem_specs(base_dir, warnings)
    metric_specs = discover_metric_specs(base_dir, warnings)
    return algorithm_specs, problem_specs, metric_specs, warnings


class ExperimentBridge(QObject):
    """
    Bridge Class to manage experiment execution and UI communication.
    Follows the Bridge pattern to decouple execution logic from interface.

    Signals:
        progress (int, str): Emits execution progress (0-100) and status message.
        run_ready (object): Emits result payload from individual run.
        warning (str): Emits warning messages.
        error (str): Emits critical error messages.
        finished (object): Emits final experiment result.
    """
    progress = Signal(int, str)
    run_ready = Signal(object)
    partial_result = Signal(object)
    warning = Signal(str)
    error = Signal(str)
    finished = Signal(object)

    def __init__(
        self,
        config: dict[str, Any],
        algorithm_specs: dict[str, AlgorithmSpec],
        problem_specs: dict[str, ProblemSpec],
        metric_specs: dict[str, MetricSpec],
    ) -> None:
        super().__init__()
        self.config = config
        self.algorithm_specs = algorithm_specs
        self.problem_specs = problem_specs
        self.metric_specs = metric_specs
        self._cancelled = False

    def _save_results_to_history(self, payload: dict[str, Any]) -> None:
        """
        Saves the execution result to a persistent JSON history file.
        Appends the new result to 'test_module_results/results.json' inside the base package directory.
        """
        try:
            # 1. Prepare directory anchored to base_dir
            base_root = Path(__file__).resolve().parent
            results_dir = base_root / "test_module_results"
            results_dir.mkdir(parents=True, exist_ok=True)
            results_file = results_dir / "results.json"
            # 2. Prepare payload entry
            # Extract relevant info to save (full payload for reloading)
            now_dt = datetime.now().astimezone()
            execution_manifest = payload.get("execution_manifest")
            entry = {
                "timestamp_iso": now_dt.isoformat(),
                "timestamp_en_us": format_timestamp_en_us(now_dt),
                "timestamp_epoch": float(now_dt.timestamp()),
                "manifest_version": int(payload.get("experiment_manifest_version", EXPERIMENT_MANIFEST_VERSION)),
                "manifest_sha256": (
                    str(execution_manifest.get("manifest_sha256", ""))
                    if isinstance(execution_manifest, dict)
                    else ""
                ),
                "payload": payload
            }

            # 3. Load existing history
            history = []
            if results_file.exists():
                try:
                    text_content = results_file.read_text(encoding="utf-8")
                    if text_content.strip():
                        history = json.loads(text_content)
                        if not isinstance(history, list):
                            history = []
                except Exception as e:
                    print(f"Warning: Could not load existing results history: {e}")

            # 4. Append and Save
            class NumpyEncoder(json.JSONEncoder):
                def default(self, obj: Any) -> Any:
                    import numpy as np
                    if isinstance(obj, np.integer): return int(obj)
                    if isinstance(obj, np.floating): return float(obj)
                    if isinstance(obj, np.ndarray): return obj.tolist()
                    try:
                        return super().default(obj)
                    except TypeError:
                        return str(obj)

            history.append(entry)
            results_file.write_text(json.dumps(history, indent=2, ensure_ascii=False, cls=NumpyEncoder), encoding="utf-8")
            print(f"Result saved to {results_file}")

        except Exception as e:
            print(f"Error saving result history: {e}")

    @Slot()
    def cancel(self) -> None:
        self._cancelled = True
        self.warning.emit("Stop requested. Execution will stop after the current run.")

    @Slot()
    def run(self) -> None:
        try:
            payload = self._execute()
            self._save_results_to_history(payload)
            self.finished.emit(payload)
        except Exception as exc:  # noqa: BLE001
            self.error.emit(str(exc))

    @staticmethod
    def _resolve_jax_gpu() -> tuple[bool, str]:
        try:
            import jax
        except Exception as exc:  # noqa: BLE001
            return False, f"JAX unavailable: {exc}"

        try:
            devices = list(jax.devices())
        except Exception as exc:  # noqa: BLE001
            return False, f"Could not query JAX devices: {exc}"

        gpu_devices = [d for d in devices if str(getattr(d, "platform", "")).lower() in {"gpu", "cuda", "rocm"}]
        if not gpu_devices:
            return False, "No JAX GPU device detected."

        device_name = str(getattr(gpu_devices[0], "device_kind", gpu_devices[0]))
        return True, device_name

    @staticmethod
    def _try_create_jax_problem(
        base_problem: Any,
        gpu_dtype: str = "float64",
        problem_name: str = "",
    ) -> Any | None:
        if bool(getattr(base_problem, "_jax_wrapped", False)):
            return base_problem

        eval_f = getattr(base_problem, "_eval_F", None)

        try:
            import jax
            import jax.numpy as jnp
            from core.problem import Problem as PymooProblem
        except Exception:  # noqa: BLE001
            return None
        use_float64 = str(gpu_dtype).lower() == "float64"
        try:
            jax.config.update("jax_enable_x64", use_float64)
            jax.config.update("jax_disable_jit", False)
        except Exception:  # noqa: BLE001
            pass
        jax_dtype = jnp.float64 if use_float64 else jnp.float32
        if not callable(eval_f):
            return None

        class JaxWrappedProblem(PymooProblem):
            def __init__(self, base_problem: Any) -> None:
                self._base_problem = base_problem
                self._jax_wrapped = True
                super().__init__(
                    n_var=int(getattr(base_problem, "n_var", 1)),
                    n_obj=int(getattr(base_problem, "n_obj", 1)),
                    n_ieq_constr=int(getattr(base_problem, "n_ieq_constr", 0)),
                    n_eq_constr=int(getattr(base_problem, "n_eq_constr", 0)),
                    xl=getattr(base_problem, "xl", None),
                    xu=getattr(base_problem, "xu", None),
                )

            @partial(jax.jit, static_argnums=0)
            def _eval_F(self, x: Any) -> Any:
                return self._base_problem._eval_F(x)

            @partial(jax.jit, static_argnums=0)
            def _eval_G(self, x: Any) -> Any:
                eval_g = getattr(self._base_problem, "_eval_G", None)
                if callable(eval_g):
                    return eval_g(x)
                rows = int(x.shape[0])
                return jnp.zeros((rows, int(self.n_ieq_constr)), dtype=jax_dtype)

            @partial(jax.jit, static_argnums=0)
            def _eval_H(self, x: Any) -> Any:
                eval_h = getattr(self._base_problem, "_eval_H", None)
                if callable(eval_h):
                    return eval_h(x)
                rows = int(x.shape[0])
                return jnp.zeros((rows, int(self.n_eq_constr)), dtype=jax_dtype)

            def _evaluate(self, x: Any, out: dict[str, Any], *args: Any, **kwargs: Any) -> None:
                _x = jnp.array(x, dtype=jax_dtype)
                out["F"] = np.asarray(self._eval_F(_x))
                if int(self.n_ieq_constr) > 0:
                    out["G"] = np.asarray(self._eval_G(_x))
                if int(self.n_eq_constr) > 0:
                    out["H"] = np.asarray(self._eval_H(_x))

            def pareto_front(self, *args: Any, **kwargs: Any) -> Any:
                if hasattr(self._base_problem, "pareto_front"):
                    return self._base_problem.pareto_front(*args, **kwargs)
                return None

            def pareto_set(self, *args: Any, **kwargs: Any) -> Any:
                if hasattr(self._base_problem, "pareto_set"):
                    return self._base_problem.pareto_set(*args, **kwargs)
                return None

        wrapped = JaxWrappedProblem(base_problem)

        # Warm-up compile to ensure selected problem is really JAX-compatible.
        try:
            test_x = np.zeros((2, int(getattr(base_problem, "n_var", 1))), dtype=float)
            test_out: dict[str, Any] = {}
            wrapped._evaluate(test_x, test_out)
            test_f = np.asarray(test_out.get("F"))
            if test_f.ndim == 0 or test_f.size == 0:
                return None
        except Exception:  # noqa: BLE001
            return None

        return wrapped

    @staticmethod
    def _try_attach_joblib_runner(
        problem: Any,
        *,
        backend: str = "loky",
        n_jobs: int = -1,
    ) -> tuple[Any, str | None, str | None]:
        is_elementwise = bool(
            getattr(problem, "elementwise", False) or getattr(problem, "elementwise_evaluation", False)
        )
        if not is_elementwise:
            return problem, None, None

        try:
            from core.problem import LoopedElementwiseEvaluation
            runner = LoopedElementwiseEvaluation()
            setattr(problem, "elementwise_runner", runner)
            info = f"LoopedElementwiseEvaluation(n_jobs={n_jobs})"
            return problem, info, None
        except Exception as exc:  # noqa: BLE001
            return problem, None, f"Could not configure elementwise runner: {exc}"

    @staticmethod
    def _coerce_front_matrix(
        value: Any,
        *,
        n_obj_hint: int | None = None,
    ) -> np.ndarray | None:
        if value is None:
            return None

        if isinstance(value, dict):
            for key in ("F", "pf", "pareto_front", "front", "values", "y"):
                if key in value:
                    resolved = ExperimentBridge._coerce_front_matrix(
                        value.get(key),
                        n_obj_hint=n_obj_hint,
                    )
                    if resolved is not None:
                        return resolved
            return None

        # First try direct numeric coercion (covers lists/tuples that are already matrix-like).
        try:
            front = np.asarray(to_numpy(value), dtype=float)
        except Exception:  # noqa: BLE001
            front = None

        if front is not None:
            if front.ndim == 0 or front.size == 0:
                return None
            if front.ndim == 1:
                if n_obj_hint is not None and n_obj_hint > 1 and front.size % int(n_obj_hint) == 0:
                    front = front.reshape(-1, int(n_obj_hint))
                else:
                    front = front.reshape(-1, 1)
            elif front.ndim > 2:
                # Some backends return tuples like (X, F). When coerced as ndarray this becomes
                # a 3D tensor; prefer structural extraction before flattening.
                if isinstance(value, tuple):
                    if len(value) >= 2:
                        preferred = [value[1], value[0], *value[2:]]
                    else:
                        preferred = list(value)
                    for entry in preferred:
                        resolved = ExperimentBridge._coerce_front_matrix(
                            entry,
                            n_obj_hint=n_obj_hint,
                        )
                        if resolved is not None:
                            return resolved
                if front.shape[-1] <= 0:
                    return None
                front = front.reshape(-1, front.shape[-1])

            if front.ndim != 2 or front.size == 0:
                return None

            finite_mask = np.isfinite(front).all(axis=1)
            if finite_mask.ndim == 1:
                front = front[finite_mask]
            if front.size == 0:
                return None

            if n_obj_hint is not None and n_obj_hint > 0 and front.shape[1] > int(n_obj_hint):
                front = front[:, : int(n_obj_hint)]

            return front

        # Fallback for container-style outputs such as (X, F), dict/list wrappers, etc.
        if isinstance(value, (tuple, list)):
            if len(value) >= 2:
                preferred = [value[1], value[0], *value[2:]]
            else:
                preferred = list(value)
            for entry in preferred:
                resolved = ExperimentBridge._coerce_front_matrix(
                    entry,
                    n_obj_hint=n_obj_hint,
                )
                if resolved is not None:
                    return resolved
            return None

        return None

    @staticmethod
    def _coerce_front_image(
        value: Any,
        *,
        n_obj_hint: int | None = None,
    ) -> np.ndarray | None:
        # PlatEMO-style PF image often comes as {X, Y, Z} grids for visualization.
        # Convert coordinate grids to point cloud only when dimensionality matches n_obj.
        if value is None:
            return None

        grids: list[Any] | None = None
        if isinstance(value, dict):
            xyz_keys = [key for key in ("x", "y", "z", "X", "Y", "Z") if key in value]
            if xyz_keys:
                # Preserve x,y,z order when possible.
                ordered: list[str] = [key for key in ("x", "y", "z", "X", "Y", "Z") if key in value]
                grids = [value[key] for key in ordered]
            elif "PF" in value:
                return ExperimentBridge._coerce_front_image(value["PF"], n_obj_hint=n_obj_hint)
        elif isinstance(value, (tuple, list)):
            grids = list(value)

        if not grids or len(grids) < 2:
            return None

        arrays: list[np.ndarray] = []
        for entry in grids:
            try:
                arr = np.asarray(to_numpy(entry), dtype=float)
            except Exception:  # noqa: BLE001
                return None
            if arr.ndim != 2 or arr.size == 0:
                return None
            arrays.append(arr)

        base_shape = arrays[0].shape
        if any(arr.shape != base_shape for arr in arrays[1:]):
            return None

        n_dims = len(arrays)
        if n_obj_hint is not None and n_obj_hint > 0 and n_dims != int(n_obj_hint):
            # For true PF we avoid mapping a 3-grid feasible region to a 2D front.
            return None

        stacked = np.column_stack([arr.reshape(-1) for arr in arrays])
        if stacked.ndim != 2 or stacked.size == 0:
            return None
        finite_mask = np.isfinite(stacked).all(axis=1)
        if finite_mask.ndim == 1:
            stacked = stacked[finite_mask]
        if stacked.size == 0:
            return None
        return stacked

    @staticmethod
    def _is_default_optimum_placeholder(
        front: np.ndarray,
        *,
        n_obj_hint: int | None = None,
    ) -> bool:
        # PlatEMO base PROBLEM uses ones(1, M) when true PF is unknown.
        if not isinstance(front, np.ndarray) or front.ndim != 2 or front.size == 0:
            return True
        if front.shape[0] != 1:
            return False
        if n_obj_hint is not None and n_obj_hint > 0 and front.shape[1] != int(n_obj_hint):
            return False
        return bool(np.allclose(front, 1.0, atol=1e-12, rtol=0.0))

    @staticmethod
    def _resolve_pareto_front(
        problem: Any,
        problem_name: str,
        ref_dirs: np.ndarray,
        *,
        n_obj_hint: int | None = None,
    ) -> np.ndarray | None:
        call_kwargs: list[dict[str, Any]] = []
        if problem_name.startswith(("dtlz", "wfg")):
            # Prefer ref_dirs first for many-objective families to avoid expensive/default PF generators.
            if problem_name.startswith("wfg"):
                # WFG local PF generation is iterative and can stall the UI for many-objective cases.
                # Use a reduced iteration budget in the GUI path to keep test/experiment feedback responsive.
                call_kwargs.append(
                    {
                        "ref_dirs": ref_dirs,
                        "n_iterations": 10,
                        "points_each_iteration": max(50, min(100, int(ref_dirs.shape[0]))),
                    }
                )
            call_kwargs.append({"ref_dirs": ref_dirs})
        elif problem_name.startswith("zxh_cf"):
            # ZXH_CF local PF generation uses dense sampling + quadratic non-dominance filtering.
            # Use a smaller GUI PF budget for responsiveness in many-objective previews/metrics.
            call_kwargs.append(
                {
                    "n_pareto_points": max(80, min(100, int(ref_dirs.shape[0]))),
                }
            )
        pf_target = max(200, int(ref_dirs.shape[0]))
        call_kwargs.append({"n_pareto_points": pf_target})
        call_kwargs.append({"n_points": pf_target})
        call_kwargs.append({"n_samples": pf_target})
        call_kwargs.append({})

        true_method_calls: dict[str, list[dict[str, Any]]] = {
            "pareto_front": list(call_kwargs),
            "_calc_pareto_front": list(call_kwargs),
            "pf": list(call_kwargs),
            "GetOptimum": [{"N": pf_target}, {"n": pf_target}, {}],
            "get_optimum": [{"N": pf_target}, {"n": pf_target}, {}],
        }
        image_method_calls: dict[str, list[dict[str, Any]]] = {
            "GetPF": [{}],
            "get_pf": [{}],
        }

        for method_name in ("pareto_front", "_calc_pareto_front", "pf", "GetOptimum", "get_optimum"):
            fn = getattr(problem, method_name, None)
            if not callable(fn):
                continue
            for kwargs in true_method_calls.get(method_name, call_kwargs):
                try:
                    pf = fn(**kwargs)
                    if pf is None:
                        continue
                    matrix = ExperimentBridge._coerce_front_matrix(pf, n_obj_hint=n_obj_hint)
                    if matrix is not None and method_name in {"GetOptimum", "get_optimum"}:
                        if ExperimentBridge._is_default_optimum_placeholder(
                            matrix,
                            n_obj_hint=n_obj_hint,
                        ):
                            continue
                    if matrix is not None:
                        return matrix
                except Exception:  # noqa: BLE001
                    continue

        for method_name in ("GetPF", "get_pf"):
            fn = getattr(problem, method_name, None)
            if not callable(fn):
                continue
            for kwargs in image_method_calls.get(method_name, [{}]):
                try:
                    pf = fn(**kwargs)
                    if pf is None:
                        continue
                    matrix = ExperimentBridge._coerce_front_image(pf, n_obj_hint=n_obj_hint)
                    if matrix is not None:
                        return matrix
                except Exception:  # noqa: BLE001
                    continue

        for attr_name in ("pf", "pareto_front", "optimum", "true_pareto_front", "reference_front"):
            attr_value = getattr(problem, attr_name, None)
            if attr_value is None or callable(attr_value):
                continue
            matrix = ExperimentBridge._coerce_front_matrix(attr_value, n_obj_hint=n_obj_hint)
            if matrix is not None:
                if attr_name == "optimum" and ExperimentBridge._is_default_optimum_placeholder(
                    matrix,
                    n_obj_hint=n_obj_hint,
                ):
                    continue
                return matrix

        for attr_name in ("PF", "pf_image"):
            attr_value = getattr(problem, attr_name, None)
            if attr_value is None or callable(attr_value):
                continue
            matrix = ExperimentBridge._coerce_front_image(attr_value, n_obj_hint=n_obj_hint)
            if matrix is not None:
                return matrix

        return None

    @staticmethod
    def _resolve_pareto_set(problem: Any, problem_name: str, ref_dirs: np.ndarray) -> np.ndarray | None:
        if not (hasattr(problem, "pareto_set") or hasattr(problem, "_calc_pareto_set")):
            return None

        call_kwargs: list[dict[str, Any]] = []
        if problem_name.startswith(("dtlz", "wfg")):
            call_kwargs.append({"ref_dirs": ref_dirs})
        call_kwargs.append({"n_pareto_points": max(100, int(ref_dirs.shape[0]))})
        call_kwargs.append({})

        for method_name in ("pareto_set", "_calc_pareto_set"):
            fn = getattr(problem, method_name, None)
            if not callable(fn):
                continue
            for kwargs in call_kwargs:
                try:
                    ps = fn(**kwargs)
                    if ps is None:
                        continue
                    matrix = ExperimentBridge._coerce_front_matrix(ps)
                    if matrix is not None:
                        return matrix
                except Exception:  # noqa: BLE001
                    continue

        return None

    def _execute(self) -> dict[str, Any]:
        try:
            from core.callback import Callback
            from core.optimize import minimize
            from core.termination import get_termination
            from util.nds.non_dominated_sorting import NonDominatedSorting
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError("EmoPyLab core dependencies were not found.") from exc
        selected_algorithm_ids = [
            sid for sid in self.config.get("algorithm_ids", []) if sid in self.algorithm_specs
        ]
        compute_backend = str(self.config.get("compute_backend", "cpu")).lower()
        if compute_backend == "auto":
            compute_backend = "cpu"
        if compute_backend in {"jax"}:
            compute_backend = "gpu"
        if compute_backend not in {"cpu", "gpu", "mlx"}:
            compute_backend = "cpu"

        selected_metric_ids = [sid for sid in self.config.get("metric_ids", []) if sid in self.metric_specs]
        selected_metric_ids = [
            sid
            for sid in core_map_selected_ids_to_backend(
                selected_metric_ids,
                specs_by_id=self.metric_specs,
                backend=compute_backend if core_rollout_allows_domain("metrics") else "cpu",
                name_getter=lambda spec: spec.name,
                module_getter=lambda spec: spec.module,
                id_getter=lambda spec: spec.id,
            )
            if sid in self.metric_specs
        ]
        raw_problem_ids = self.config.get("problem_ids", [])
        selected_problem_ids: list[str] = []
        if isinstance(raw_problem_ids, (list, tuple)):
            for pid in raw_problem_ids:
                sid = str(pid)
                if sid in self.problem_specs and sid not in selected_problem_ids:
                    selected_problem_ids.append(sid)
        raw_problem_overrides = self.config.get("problem_overrides", {})
        problem_overrides: dict[str, dict[str, int]] = {}
        if isinstance(raw_problem_overrides, dict):
            for problem_id, raw_values in raw_problem_overrides.items():
                pid = str(problem_id)
                if pid not in self.problem_specs or not isinstance(raw_values, dict):
                    continue
                normalized: dict[str, int] = {}
                for key in ("pop_size", "n_obj", "n_var"):
                    if key in raw_values:
                        try:
                            normalized[key] = int(raw_values[key])
                        except Exception:  # noqa: BLE001
                            continue
                if normalized:
                    problem_overrides[pid] = normalized
        raw_algo_operator_overrides = self.config.get("algorithm_operator_overrides", {})
        algorithm_operator_overrides: dict[str, dict[str, Any]] = {}
        if isinstance(raw_algo_operator_overrides, dict):
            for algorithm_id, raw_values in raw_algo_operator_overrides.items():
                aid = str(algorithm_id)
                if aid not in self.algorithm_specs or not isinstance(raw_values, dict):
                    continue
                normalized_ops: dict[str, Any] = {}
                for key in ("crossover", "mutation", "selection", "sampling"):
                    if key in raw_values and raw_values[key] is not None:
                        normalized_ops[key] = str(raw_values[key])
                for key in ("crossover_eta", "crossover_prob", "mutation_eta", "mutation_prob"):
                    if key in raw_values and raw_values[key] is not None:
                        try:
                            normalized_ops[key] = float(raw_values[key])
                        except Exception:  # noqa: BLE001
                            continue
                if "selection_pressure" in raw_values and raw_values["selection_pressure"] is not None:
                    try:
                        normalized_ops["selection_pressure"] = int(raw_values["selection_pressure"])
                    except Exception:  # noqa: BLE001
                        pass
                if normalized_ops:
                    algorithm_operator_overrides[aid] = normalized_ops
        if not selected_problem_ids:
            fallback_problem_id = str(self.config.get("problem_id", ""))
            if fallback_problem_id in self.problem_specs:
                selected_problem_ids = [fallback_problem_id]
        raw_problem_entries = self.config.get("problem_entries", [])
        selected_problem_entries: list[dict[str, Any]] = []
        if isinstance(raw_problem_entries, (list, tuple)):
            for raw_entry in raw_problem_entries:
                if not isinstance(raw_entry, dict):
                    continue
                base_problem_id = str(raw_entry.get("problem_id", "")).strip()
                if base_problem_id not in self.problem_specs:
                    continue
                instance_id = str(raw_entry.get("instance_id", base_problem_id)).strip() or base_problem_id
                label = str(raw_entry.get("label", "")).strip()
                entry_override: dict[str, int] = {}
                for key in ("pop_size", "n_obj", "n_var"):
                    if key in raw_entry and raw_entry[key] is not None:
                        try:
                            entry_override[key] = int(raw_entry[key])
                        except Exception:  # noqa: BLE001
                            continue
                selected_problem_entries.append(
                    {
                        "problem_id": base_problem_id,
                        "instance_id": instance_id,
                        "label": label,
                        "override": entry_override,
                    }
                )
        if not selected_problem_entries:
            for problem_id in selected_problem_ids:
                selected_problem_entries.append(
                    {
                        "problem_id": problem_id,
                        "instance_id": problem_id,
                        "label": "",
                        "override": dict(problem_overrides.get(problem_id, {})),
                    }
                )
        if not selected_problem_ids and selected_problem_entries:
            selected_problem_ids = []
            for entry in selected_problem_entries:
                pid = str(entry.get("problem_id", "")).strip()
                if pid and pid not in selected_problem_ids:
                    selected_problem_ids.append(pid)
        single_problem_mode = bool(self.config.get("__single_problem_mode__", False))
        suppress_progress = bool(self.config.get("__suppress_progress__", False))
        exp_raw_first_metrics = bool(self.config.get("__exp_raw_first_metrics__", False))
        emit_partial_results = bool(self.config.get("__emit_partial_results__", True))

        def emit_progress(percent: int, message: str) -> None:
            if not suppress_progress:
                self.progress.emit(percent, message)

        if not selected_algorithm_ids:
            raise RuntimeError("No valid algorithms selected.")
        if not selected_metric_ids:
            raise RuntimeError("No valid metrics selected.")
        if not selected_problem_ids:
            raise RuntimeError("Selected problem is not available.")

        def _runtime_cfg_with_algo_overrides(base_cfg: dict[str, Any], algorithm_id: str) -> dict[str, Any]:
            cfg = dict(base_cfg)
            override = algorithm_operator_overrides.get(str(algorithm_id))
            if not isinstance(override, dict):
                return cfg
            for key in (
                "crossover",
                "mutation",
                "selection",
                "sampling",
                "crossover_eta",
                "crossover_prob",
                "mutation_eta",
                "mutation_prob",
                "selection_pressure",
            ):
                if key in override:
                    cfg[key] = override[key]
            return cfg

        if not single_problem_mode and len(selected_problem_entries) > 1:
            merged_results: dict[str, list[dict[str, Any]]] = {}
            merged_metrics: list[str] = []
            pareto_fronts_payload: dict[str, list[list[float]] | None] = {}
            backend_labels: set[str] = set()
            backend_codes: set[str] = set()
            child_manifest_hashes: list[str] = []
            profile_compare_enabled = False
            first_pareto_front = None
            done_problems = 0

            original_config = self.config
            try:
                for problem_entry in selected_problem_entries:
                    if self._cancelled:
                        break
                    problem_id = str(problem_entry.get("problem_id", "")).strip()
                    problem_instance_id = str(problem_entry.get("instance_id", problem_id)).strip() or problem_id
                    problem_label_override = str(problem_entry.get("label", "")).strip()
                    problem_spec = self.problem_specs.get(problem_id)
                    if problem_spec is None:
                        continue

                    sub_config = dict(original_config)
                    sub_config["problem_id"] = problem_id
                    sub_config["problem_ids"] = [problem_id]
                    sub_config["problem_instance_id"] = problem_instance_id
                    if problem_label_override:
                        sub_config["problem_label_override"] = problem_label_override
                    if problem_spec.source == "pymoo":
                        sub_config["n_obj"] = int(problem_spec.default_n_obj)
                        sub_config["n_var"] = int(problem_spec.default_n_var)
                    override = problem_entry.get("override")
                    if not isinstance(override, dict):
                        override = problem_overrides.get(problem_id)
                    if isinstance(override, dict):
                        for key in ("pop_size", "n_obj", "n_var"):
                            if key in override:
                                sub_config[key] = int(override[key])
                        sub_config["problem_overrides"] = {problem_id: {k: int(v) for k, v in override.items() if k in {"pop_size", "n_obj", "n_var"}}}
                    sub_config["__single_problem_mode__"] = True
                    sub_config["__suppress_progress__"] = True
                    self.config = sub_config

                    payload = self._execute()
                    if not merged_metrics:
                        merged_metrics = list(payload.get("metrics", []))

                    sub_results = payload.get("results", {})
                    if isinstance(sub_results, dict):
                        for algo_name, runs in sub_results.items():
                            target_runs = merged_results.setdefault(str(algo_name), [])
                            for raw_run in runs:
                                if not isinstance(raw_run, dict):
                                    continue
                                run_payload = dict(raw_run)
                                run_payload.setdefault("problem", problem_spec.label)
                                run_payload.setdefault("problem_id", problem_spec.id)
                                run_payload.setdefault("problem_name", problem_spec.name)
                                target_runs.append(run_payload)

                    pf_raw = payload.get("pareto_front")
                    if isinstance(pf_raw, list):
                        pareto_fronts_payload[problem_instance_id] = pf_raw
                        if first_pareto_front is None:
                            first_pareto_front = pf_raw
                    else:
                        pareto_fronts_payload[problem_instance_id] = None

                    backend_labels.add(str(payload.get("execution_backend_label", "CPU")))
                    backend_codes.add(str(payload.get("execution_backend", "cpu")))
                    profile_compare_enabled = profile_compare_enabled or bool(payload.get("profile_compare_enabled", False))
                    child_manifest = payload.get("execution_manifest")
                    if isinstance(child_manifest, dict):
                        child_hash = str(child_manifest.get("manifest_sha256", "")).strip()
                        if child_hash:
                            child_manifest_hashes.append(child_hash)

                    done_problems += 1
                    emit_progress(
                        int(100 * done_problems / max(1, len(selected_problem_entries))),
                        f"{(problem_label_override or problem_spec.name)} finished ({done_problems}/{len(selected_problem_entries)})",
                    )
            finally:
                self.config = original_config

            if len(backend_labels) == 1:
                execution_backend_label = next(iter(backend_labels))
                execution_backend_code = next(iter(backend_codes)) if len(backend_codes) == 1 else "mixed"
            elif len(backend_labels) > 1:
                execution_backend_label = "Mixed backends by problem/run (see Backend column)"
                execution_backend_code = "mixed"
            else:
                execution_backend_label = "CPU"
                execution_backend_code = "cpu"

            merged_seed_plan = {
                "mode": str(self.config.get("seed_mode", SEED_MODE_RANDOM)).strip().lower(),
                "deterministic": False,
                "base_seed": _normalize_seed_value(self.config.get("seed_base", 1), default=1),
                "step": _positive_int(self.config.get("seed_step", 1), 1, minimum=1),
                "provided_sequence": [],
                "seeds": [],
                "child_manifest_hashes": child_manifest_hashes,
            }
            execution_manifest = build_execution_manifest(
                config=self.config,
                seed_plan=merged_seed_plan,
                selected_problem_ids=[str(entry.get("instance_id", entry.get("problem_id", ""))) for entry in selected_problem_entries],
                selected_algorithm_ids=selected_algorithm_ids,
                selected_metric_ids=selected_metric_ids,
                execution_backend=execution_backend_code,
                execution_backend_label=execution_backend_label,
            )

            return {
                "config": self.config,
                "results": merged_results,
                "metrics": merged_metrics,
                "pareto_front": first_pareto_front,
                "pareto_fronts": pareto_fronts_payload,
                "execution_backend": execution_backend_code,
                "execution_backend_label": execution_backend_label,
                "profile_compare_enabled": profile_compare_enabled,
                "experiment_manifest_version": EXPERIMENT_MANIFEST_VERSION,
                "execution_manifest": execution_manifest,
                "cancelled": self._cancelled,
            }

        problem_entry = selected_problem_entries[0] if selected_problem_entries else {
            "problem_id": selected_problem_ids[0],
            "instance_id": selected_problem_ids[0],
            "label": "",
            "override": dict(problem_overrides.get(selected_problem_ids[0], {})),
        }
        problem_id = str(problem_entry.get("problem_id", ""))
        problem_instance_id = str(problem_entry.get("instance_id", problem_id)).strip() or problem_id
        problem_label_override = str(problem_entry.get("label", "")).strip()
        cfg_problem_instance_id = str(self.config.get("problem_instance_id", "")).strip()
        if cfg_problem_instance_id:
            problem_instance_id = cfg_problem_instance_id
        cfg_problem_label_override = str(self.config.get("problem_label_override", "")).strip()
        if cfg_problem_label_override:
            problem_label_override = cfg_problem_label_override

        problem_spec = self.problem_specs.get(problem_id)
        if problem_spec is None:
            raise RuntimeError("Selected problem is not available.")

        config_n_var = int(self.config.get("n_var", problem_spec.default_n_var))
        config_n_obj = int(self.config.get("n_obj", problem_spec.default_n_obj))
        pop_size = int(self.config.get("pop_size", 100))
        if problem_spec.source == "pymoo":
            config_n_var = int(problem_spec.default_n_var)
            config_n_obj = int(problem_spec.default_n_obj)
        problem_override = problem_entry.get("override")
        if not isinstance(problem_override, dict):
            problem_override = problem_overrides.get(problem_id, {})
        if isinstance(problem_override, dict):
            if "n_var" in problem_override:
                config_n_var = int(problem_override["n_var"])
            if "n_obj" in problem_override:
                config_n_obj = int(problem_override["n_obj"])
            if "pop_size" in problem_override:
                pop_size = int(problem_override["pop_size"])

        runtime_cfg = dict(self.config)
        runtime_cfg["n_var"] = config_n_var
        runtime_cfg["n_obj"] = config_n_obj
        runtime_cfg["pop_size"] = pop_size
        runtime_cfg["use_gpu"] = False

        gpu_dtype = str(self.config.get("gpu_dtype", "float64")).lower()
        if gpu_dtype not in {"float32", "float64"}:
            gpu_dtype = "float64"

        joblib_backend = str(self.config.get("joblib_backend", "loky")).strip().lower() or "loky"
        try:
            joblib_n_jobs = int(self.config.get("joblib_n_jobs", -1))
        except Exception:  # noqa: BLE001
            joblib_n_jobs = -1
        if joblib_n_jobs == 0:
            joblib_n_jobs = -1

        profile_compare_requested = bool(self.config.get("profile_compare", False))
        jax_runtime_info = detect_gpu_runtime()
        jax_runtime_ok = bool(jax_runtime_info.get("jax_ok"))
        jax_gpu_ok = bool(jax_runtime_info.get("cuda_ok"))
        if jax_gpu_ok:
            jax_gpu_info = str(jax_runtime_info.get("cuda_device_name") or "GPU device")
        elif jax_runtime_ok:
            jax_gpu_info = "No JAX GPU device detected"
        else:
            jax_gpu_info = str(jax_runtime_info.get("error") or "JAX runtime unavailable")
        if compute_backend == "gpu" and not jax_gpu_ok:
            self.warning.emit(f"JAX backend requested, but unavailable: {jax_gpu_info}")

        profile_compare_enabled = profile_compare_requested and jax_gpu_ok
        if profile_compare_requested and not profile_compare_enabled:
            self.warning.emit(
                "CPU vs GPU profiling requested, but unavailable for this machine/runtime. Profiling was disabled."
            )

        cpu_backend_label = "CPU (NumPy)"
        cpu_jax_backend_label = "CPU (JAX)"
        gpu_backend_label = f"JAX (GPU, {jax_gpu_info}, {gpu_dtype})"
        problem_rollout_enabled = core_rollout_allows_domain("problems")
        operator_rollout_enabled = core_rollout_allows_domain("operators")
        metric_rollout_enabled = core_rollout_allows_domain("metrics")
        if compute_backend == "gpu" and not problem_rollout_enabled:
            self.warning.emit(
                "GPU backend requested, but backend-aware rollout for problems is disabled. "
                "Using CPU problems for this run."
            )
        jax_fallback_notified = {"value": False}
        joblib_notify_state: dict[str, bool] = {"warned": False, "enabled": False}
        is_jax_problem_spec = (
            _looks_like_jax_identifier(problem_spec.name)
            or _looks_like_jax_identifier(problem_spec.module)
            or _looks_like_jax_identifier(problem_spec.id)
        )

        def build_problem(prefer_gpu: bool) -> tuple[Any, str, str]:
            runtime_cfg_problem = dict(runtime_cfg)
            requested_jax_cpu_problem = bool(
                compute_backend == "gpu"
                and not jax_gpu_ok
                and is_jax_problem_spec
                and jax_runtime_ok
            )
            runtime_cfg_problem["use_gpu"] = bool(
                prefer_gpu and jax_gpu_ok and problem_rollout_enabled
            )
            runtime_cfg_problem["array_backend"] = (
                "jax"
                if (runtime_cfg_problem["use_gpu"] or requested_jax_cpu_problem)
                else "numpy"
            )
            runtime_cfg_problem["gpu_dtype"] = gpu_dtype
            base_problem = problem_spec.factory(runtime_cfg_problem)
            if compute_backend == "mlx":
                return base_problem, "mlx", "MLX (Apple Silicon)"
            if prefer_gpu and jax_gpu_ok and problem_rollout_enabled:
                if is_jax_problem_spec:
                    return base_problem, "gpu", gpu_backend_label
                gpu_problem = self._try_create_jax_problem(
                    base_problem,
                    gpu_dtype=gpu_dtype,
                    problem_name=problem_spec.name,
                )
                if gpu_problem is not None:
                    return gpu_problem, "gpu", gpu_backend_label
                if not jax_fallback_notified["value"]:
                    self.warning.emit(
                        "Selected problem has no compatible JAX variant. Falling back to CPU."
                    )
                    jax_fallback_notified["value"] = True
            cpu_label = (
                cpu_jax_backend_label
                if requested_jax_cpu_problem
                else cpu_backend_label
            )
            cpu_problem, joblib_info, joblib_error = self._try_attach_joblib_runner(
                base_problem,
                backend=joblib_backend,
                n_jobs=joblib_n_jobs,
            )
            if joblib_info:
                joblib_notify_state["enabled"] = True
                return cpu_problem, "cpu", f"{cpu_label} + {joblib_info}"
            if joblib_error and not joblib_notify_state["warned"]:
                self.warning.emit(joblib_error)
                joblib_notify_state["warned"] = True
            return cpu_problem, "cpu", cpu_label

        try:
            probe_problem = problem_spec.factory(runtime_cfg)
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"Failed to initialize problem '{problem_spec.label}': {exc}") from exc

        effective_n_var = int(getattr(probe_problem, "n_var", config_n_var))
        effective_n_obj = int(getattr(probe_problem, "n_obj", config_n_obj))
        runtime_cfg["n_var"] = effective_n_var
        runtime_cfg["n_obj"] = effective_n_obj

        ref_dirs = build_reference_dirs(n_obj=effective_n_obj, target=max(12, pop_size))

        use_pf = bool(self.config.get("use_pf", True))
        problem_name = problem_spec.name.lower()
        pareto_front = None
        pareto_set = None
        if use_pf:
            pareto_front = self._resolve_pareto_front(
                probe_problem,
                problem_name,
                ref_dirs,
                n_obj_hint=effective_n_obj,
            )
            if pareto_front is None or pareto_front.size == 0:
                pareto_front = None
                problem_warn_label = problem_label_override or problem_spec.label
                self.warning.emit(
                    "Reference Pareto front is unavailable for "
                    f"'{problem_warn_label}' (M={effective_n_obj}, D={effective_n_var}). "
                    "PF-based metrics may be disabled."
                )
            pareto_set = self._resolve_pareto_set(probe_problem, problem_name, ref_dirs)
            if pareto_set is not None and pareto_set.size == 0:
                pareto_set = None

        # HV reference point - automatic strategy
        # If PF is available: ref_point = max(PF, axis=0) * 1.1
        # Otherwise: ref_point = [1.1] * n_obj
        if pareto_front is not None and pareto_front.shape[1] == effective_n_obj:
            ref_point = np.max(pareto_front, axis=0) * 1.1
            ref_source = "max(PF) * 1.1"
        else:
            ref_point = np.asarray([1.1] * effective_n_obj, dtype=float)
            ref_source = f"default [1.1] * {effective_n_obj}"
        ref_str = ", ".join(f"{v:.4f}" for v in ref_point)
        emit_progress(0, f"HV ref_point ({ref_source}): [{ref_str}]")

        metric_context = {
            "pareto_front": pareto_front,
            "pareto_set": pareto_set,
            "ref_point": ref_point,
            "ref_dirs": ref_dirs,
            "compute_backend_requested": compute_backend,
            "backend_aware_enabled": bool(core_backend_aware_loading_enabled()),
            "backend_rollout_stage": str(core_rollout_stage()),
            "backend_rollout": {
                "problems": bool(problem_rollout_enabled),
                "operators": bool(operator_rollout_enabled),
                "metrics": bool(metric_rollout_enabled),
            },
            "backend_code": "gpu"
            if (compute_backend == "gpu" and jax_gpu_ok and problem_rollout_enabled)
            else "cpu",
            "array_backend": "jax"
            if (compute_backend == "gpu" and jax_gpu_ok and problem_rollout_enabled)
            else "numpy",
            "use_gpu": bool(compute_backend == "gpu" and jax_gpu_ok and problem_rollout_enabled),
            "gpu_dtype": gpu_dtype,
            "n_obj": effective_n_obj,
            "n_var": effective_n_var,
            "pop_size": pop_size,
            "problem_name": problem_name,
            "problem": probe_problem,
            "hv_mc_samples": _positive_int(self.config.get("hv_mc_samples", 10000), 10000),
            "hv_mc_exclusive": _positive_int(self.config.get("hv_mc_exclusive", 1), 1),
            "current_population_F": None,
            "current_population_X": None,
            "current_population_G": None,
            "current_population_H": None,
            "current_population_CV": None,
            "current_population_feasible": None,
        }

        active_metrics: dict[str, Callable[[np.ndarray], float]] = {}
        if not exp_raw_first_metrics:
            for metric_id in selected_metric_ids:
                spec = self.metric_specs[metric_id]
                try:
                    active_metrics[spec.label] = spec.factory(metric_context)
                except Exception as exc:  # noqa: BLE001
                    self.warning.emit(f"Metric '{spec.label}' disabled: {exc}")

            if not active_metrics:
                raise RuntimeError("No metric is active after initialization.")

        valid_algorithms: list[AlgorithmSpec] = []
        for algorithm_id in selected_algorithm_ids:
            spec = self.algorithm_specs[algorithm_id]
            try:
                _ = spec.factory(_runtime_cfg_with_algo_overrides(runtime_cfg, spec.id))
                valid_algorithms.append(spec)
            except Exception as exc:  # noqa: BLE001
                self.warning.emit(f"Algorithm '{spec.label}' skipped: {exc}")

        if not valid_algorithms:
            raise RuntimeError("No selected algorithm could be initialized.")

        nds = NonDominatedSorting()

        def extract_front(values: Any) -> np.ndarray:
            if values is None:
                return np.empty((0, effective_n_obj), dtype=float)
            front = np.asarray(to_numpy(values), dtype=float)
            if front.ndim == 1:
                front = front.reshape(1, -1)
            if front.size == 0:
                return np.empty((0, effective_n_obj), dtype=float)
            try:
                nd_idx = nds.do(front, only_non_dominated_front=True)
                front = front[nd_idx]
            except Exception:  # noqa: BLE001
                pass
            return front

        def extract_matrix(values: Any, *, vector_as_row: bool = True, dtype: Any = float) -> np.ndarray | None:
            if values is None:
                return None
            matrix = np.asarray(to_numpy(values), dtype=dtype)
            if matrix.ndim == 0:
                matrix = matrix.reshape(1, 1)
            elif matrix.ndim == 1:
                matrix = matrix.reshape(1, -1) if vector_as_row else matrix.reshape(-1, 1)
            return matrix

        class MetricCallback(Callback):
            def __init__(
                self,
                metric_map: dict[str, Callable[[np.ndarray], float]],
                owner: ExperimentBridge,
                *,
                max_fe: int = 1,
                total_runs: int = 1,
                done_runs_ref: list[int] | None = None,
                emit_fn: Callable[[int, str], None] | None = None,
                algo_label: str = "",
                algo_id: str = "",
                problem_label: str = "",
                problem_id: str = "",
                problem_name: str = "",
                backend_label: str = "",
                backend_code: str = "",
                run_index: int = 1,
                n_runs: int = 1,
                emit_partial_results: bool = True,
                track_metric_history: bool = True,
            ) -> None:
                super().__init__()
                self.metric_map = metric_map
                self.owner = owner
                self.fe_history: list[int] = []
                self.history: dict[str, list[float]] = {name: [] for name in metric_map}
                # Real-time progress based on n_eval / max_fe
                self._max_fe = max(1, max_fe)
                self._total_runs = max(1, total_runs)
                self._done_runs_ref = done_runs_ref or [0]
                self._emit_fn = emit_fn
                self._algo_label = algo_label
                self._algo_id = algo_id
                self._problem_label = problem_label
                self._problem_id = problem_id
                self._problem_name = problem_name
                self._backend_label = backend_label
                self._backend_code = backend_code
                self._run_index = run_index
                self._n_runs = n_runs
                self._emit_partial_results = bool(emit_partial_results)
                self._track_metric_history = bool(track_metric_history)

            def notify(self, algorithm: Any) -> None:
                if self.owner._cancelled:
                    raise RuntimeError("__cancelled__")
                pop = getattr(algorithm, "pop", None)

                def _pop_get(field: str) -> Any:
                    if pop is None:
                        return None
                    try:
                        return pop.get(field)
                    except Exception:  # noqa: BLE001
                        return None

                n_eval = None
                try:
                    n_eval = int(algorithm.evaluator.n_eval)
                except Exception:  # noqa: BLE001
                    pass
                if n_eval is None:
                    n_eval = int(getattr(algorithm, "n_gen", len(self.fe_history) + 1))
                self.fe_history.append(n_eval)

                # Emit real-time progress: n_eval / maxFE inside the current run
                if self._emit_fn is not None:
                    run_frac = min(1.0, n_eval / self._max_fe)
                    base = self._done_runs_ref[0] / self._total_runs
                    slot = 1.0 / self._total_runs
                    percent = int(100 * (base + slot * run_frac))
                    percent = max(0, min(99, percent))  # never 100% until completion
                    self._emit_fn(
                        percent,
                        f"{self._algo_label} run {self._run_index}/{self._n_runs} - "
                        f"{n_eval}/{self._max_fe} evals",
                    )

                if not self._emit_partial_results:
                    return

                pop_f_raw = _pop_get("F")
                front = extract_front(pop_f_raw)
                if front.size == 0:
                    return

                metric_context["current_population_F"] = extract_matrix(pop_f_raw)
                metric_context["current_population_X"] = extract_matrix(_pop_get("X"))
                metric_context["current_population_G"] = extract_matrix(_pop_get("G"), vector_as_row=False)
                metric_context["current_population_H"] = extract_matrix(_pop_get("H"), vector_as_row=False)
                metric_context["current_population_CV"] = extract_matrix(_pop_get("CV"), vector_as_row=False)
                metric_context["current_population_feasible"] = extract_matrix(
                    _pop_get("feasible"),
                    vector_as_row=False,
                    dtype=bool,
                )

                metric_snapshot: dict[str, float] = {}
                if self._track_metric_history:
                    for metric_name, metric_fn in self.metric_map.items():
                        try:
                            value = float(metric_fn(front))
                        except Exception:  # noqa: BLE001
                            value = float("nan")
                        self.history[metric_name].append(value)
                        metric_snapshot[metric_name] = value

                # Emit partial result for real-time plot updates
                now_dt = datetime.now().astimezone()
                self.owner.partial_result.emit({
                    "algorithm": self._algo_label,
                    "algorithm_id": self._algo_id,
                    "problem": self._problem_label or self._problem_name,
                    "problem_id": self._problem_id,
                    "problem_name": self._problem_name,
                    "run_index": self._run_index,
                    "n_eval": n_eval,
                    "evaluations": n_eval,
                    "backend": self._backend_label,
                    "backend_code": self._backend_code,
                    "metrics": dict(metric_snapshot),
                    "front": front.tolist(),
                    "final_front": front.tolist(),
                    "reference_front": (
                        None
                        if pareto_front is None
                        else np.asarray(pareto_front, dtype=float).tolist()
                    ),
                    "history": {name: list(values) for name, values in self.history.items()},
                    "x_history": list(self.fe_history),
                    "generations": list(self.fe_history),
                    "timestamp_iso": now_dt.isoformat(),
                    "timestamp_en_us": format_timestamp_en_us(now_dt),
                    "timestamp_epoch": float(now_dt.timestamp()),
                    "is_partial": True,
                })

        n_runs = max(1, int(self.config.get("n_runs", 1)))
        max_fe = int(
            self.config.get("max_fe", self.config.get("maxFE", DEFAULT_MAX_FE))
        )
        if max_fe <= 0:
            max_fe = DEFAULT_MAX_FE
        runtime_cfg["max_fe"] = max_fe
        total_runs = n_runs * len(valid_algorithms)
        done_runs_ref: list[int] = [0]  # mutable reference for callback progress
        results: dict[str, list[dict[str, Any]]] = {spec.label: [] for spec in valid_algorithms}

        def run_trial(
            algo_spec: AlgorithmSpec,
            seed: int,
            prefer_gpu: bool,
            *,
            run_index: int = 1,
        ) -> dict[str, Any]:
            problem, trial_backend_code, trial_backend_label = build_problem(prefer_gpu)
            runtime_cfg_trial = dict(runtime_cfg)
            runtime_cfg_trial["use_gpu"] = trial_backend_code in {"gpu", "mlx"}
            runtime_cfg_trial["gpu_dtype"] = gpu_dtype
            runtime_cfg_trial["array_backend"] = "jax" if trial_backend_code == "gpu" else ("mlx" if trial_backend_code == "mlx" else "numpy")
            metric_context["backend_code"] = trial_backend_code
            metric_context["backend_label"] = trial_backend_label
            metric_context["array_backend"] = runtime_cfg_trial["array_backend"]
            metric_context["use_gpu"] = runtime_cfg_trial["use_gpu"]
            metric_context["gpu_dtype"] = gpu_dtype

            runtime_cfg_trial = _runtime_cfg_with_algo_overrides(runtime_cfg_trial, algo_spec.id)
            # CAUSA 2: Injeta seed no config para que instantiate_algorithm_class
            # a passe no construtor do algoritmo (se o __init__ aceitar 'seed')
            runtime_cfg_trial["seed"] = int(seed)
            algorithm = algo_spec.factory(runtime_cfg_trial)
            callback = MetricCallback(
                active_metrics,
                self,
                max_fe=max_fe,
                total_runs=total_runs,
                done_runs_ref=done_runs_ref,
                emit_fn=emit_progress,
                algo_label=algo_spec.label,
                algo_id=algo_spec.id,
                problem_label=(problem_label_override or problem_spec.label),
                problem_id=problem_instance_id,
                problem_name=problem_spec.name,
                backend_label=trial_backend_label,
                backend_code=trial_backend_code,
                run_index=run_index,
                n_runs=n_runs,
                emit_partial_results=emit_partial_results,
                track_metric_history=not exp_raw_first_metrics,
            )
            termination = get_termination("n_eval", max_fe)

            start = time.perf_counter()
            result = minimize(
                problem,
                algorithm,
                termination=termination,
                seed=seed,
                verbose=False,
                save_history=False,
                copy_algorithm=False,
                callback=callback,
            )
            elapsed = time.perf_counter() - start

            final_front = extract_front(result.F)
            final_pop = getattr(result, "pop", None)
            if final_pop is None:
                final_pop = getattr(getattr(result, "algorithm", None), "pop", None)

            def _final_pop_get(field: str) -> Any:
                if final_pop is None:
                    return None
                try:
                    return final_pop.get(field)
                except Exception:  # noqa: BLE001
                    return None

            final_pop_f_raw = _final_pop_get("F")
            if final_front.size == 0 and final_pop_f_raw is not None:
                final_front = extract_front(final_pop_f_raw)

            final_pop_F = extract_matrix(final_pop_f_raw) if final_pop_f_raw is not None else extract_matrix(final_front)
            final_pop_X = extract_matrix(_final_pop_get("X"))
            final_pop_G = extract_matrix(_final_pop_get("G"), vector_as_row=False)
            final_pop_H = extract_matrix(_final_pop_get("H"), vector_as_row=False)
            final_pop_CV = extract_matrix(_final_pop_get("CV"), vector_as_row=False)
            final_pop_feasible = extract_matrix(
                _final_pop_get("feasible"),
                vector_as_row=False,
                dtype=bool,
            )

            metric_context["current_population_F"] = final_pop_F
            metric_context["current_population_X"] = final_pop_X
            metric_context["current_population_G"] = final_pop_G
            metric_context["current_population_H"] = final_pop_H
            metric_context["current_population_CV"] = final_pop_CV
            metric_context["current_population_feasible"] = final_pop_feasible

            metric_values: dict[str, float] = {}
            if not exp_raw_first_metrics:
                for metric_name, metric_fn in active_metrics.items():
                    try:
                        metric_values[metric_name] = float(metric_fn(final_front))
                    except Exception:  # noqa: BLE001
                        metric_values[metric_name] = float("nan")

            evaluations = -1
            try:
                evaluations = int(result.algorithm.evaluator.n_eval)
            except Exception:  # noqa: BLE001
                pass

            return {
                "backend_code": trial_backend_code,
                "backend_label": trial_backend_label,
                "time_s": elapsed,
                "evaluations": evaluations,
                "metrics": metric_values,
                "history": callback.history,
                "x_history": callback.fe_history,
                "generations": callback.fe_history,
                "final_front": final_front,
                "final_population": {
                    "F": None if final_pop_F is None else np.asarray(final_pop_F).tolist(),
                    "X": None if final_pop_X is None else np.asarray(final_pop_X).tolist(),
                    "G": None if final_pop_G is None else np.asarray(final_pop_G).tolist(),
                    "H": None if final_pop_H is None else np.asarray(final_pop_H).tolist(),
                    "CV": None if final_pop_CV is None else np.asarray(final_pop_CV).tolist(),
                    "feasible": None if final_pop_feasible is None else np.asarray(final_pop_feasible).tolist(),
                },
                "block_profile": getattr(algorithm, "block_profile_snapshot", None),
            }

        primary_prefers_gpu = bool(
            compute_backend == "gpu" and jax_gpu_ok and problem_rollout_enabled
        )
        if profile_compare_enabled:
            primary_label = gpu_backend_label if primary_prefers_gpu else cpu_backend_label
            execution_backend_code = "profile"
            execution_backend_label = f"CPU/GPU profiling (primary result: {primary_label})"
        else:
            if compute_backend == "mlx":
                execution_backend_code = "mlx"
                execution_backend_label = "MLX (Apple Silicon)"
            else:
                execution_backend_code = "gpu" if primary_prefers_gpu else "cpu"
                if primary_prefers_gpu:
                    execution_backend_label = gpu_backend_label
                elif compute_backend == "gpu" and is_jax_problem_spec and jax_runtime_ok:
                    execution_backend_label = cpu_jax_backend_label
                else:
                    execution_backend_label = cpu_backend_label

        import concurrent.futures
        import threading

        lock = threading.Lock()
        recommended_workers, max_supported_workers = resolve_parallel_worker_limits()
        requested_workers = _positive_int(
            self.config.get("parallel_workers", recommended_workers),
            recommended_workers,
        )
        parallel_workers, worker_warnings = _resolve_parallel_workers_for_algorithms(
            requested_workers=requested_workers,
            max_supported_workers=max_supported_workers,
            selected_algorithms=valid_algorithms,
        )
        for worker_warning in worker_warnings:
            self.warning.emit(worker_warning)

        llm_algorithms = [spec.label for spec in valid_algorithms if _algorithm_uses_local_llm(spec)]
        if llm_algorithms:
            lmstudio_model = str(
                os.environ.get("EMOPYLAB_LARC_MODEL", DEFAULT_LOCAL_LMSTUDIO_MODEL)
            ).strip() or DEFAULT_LOCAL_LMSTUDIO_MODEL
            lmstudio_url = str(
                os.environ.get("EMOPYLAB_LMSTUDIO_URL") or os.environ.get("EMOPYLAB_OLLAMA_URL") or DEFAULT_LOCAL_LMSTUDIO_CHAT_URL
            ).strip()
            _assert_local_lmstudio_model_available(
                model=lmstudio_model,
                chat_url=lmstudio_url,
                timeout_s=6.0,
            )

        def run_single_trial(algo_spec: AlgorithmSpec, run_index: int) -> None:
            if self._cancelled:
                return

            seed = int(seed_lookup.get((algo_spec.id, int(run_index)), _random_seed()))

            try:
                if profile_compare_enabled:
                    cpu_trial = run_trial(algo_spec=algo_spec, seed=seed, prefer_gpu=False, run_index=run_index)
                    if self._cancelled:
                        return

                    gpu_trial = run_trial(algo_spec=algo_spec, seed=seed, prefer_gpu=True, run_index=run_index)

                    cpu_time = (
                        _float_or_none(cpu_trial.get("time_s")) if str(cpu_trial.get("backend_code")) == "cpu" else None
                    )
                    gpu_time = (
                        _float_or_none(gpu_trial.get("time_s")) if str(gpu_trial.get("backend_code")) == "gpu" else None
                    )

                    if primary_prefers_gpu and gpu_time is None:
                        self.warning.emit(
                            f"{algo_spec.label} trial {run_index}: GPU profile pass fell back to CPU. "
                            "Using CPU result for this trial."
                        )

                    primary_trial = gpu_trial if (primary_prefers_gpu and gpu_time is not None) else cpu_trial

                    speedup = None
                    if cpu_time is not None and gpu_time is not None and gpu_time > 0.0:
                        speedup = cpu_time / gpu_time
                else:
                    primary_trial = run_trial(
                        algo_spec=algo_spec,
                        seed=seed,
                        prefer_gpu=primary_prefers_gpu,
                        run_index=run_index,
                    )
                    cpu_time = None
                    gpu_time = None
                    speedup = None

                run_payload: dict[str, Any] = {
                    "problem": (problem_label_override or problem_spec.label),
                    "problem_id": problem_instance_id,
                    "base_problem_id": problem_spec.id,
                    "problem_name": problem_spec.name,
                    "algorithm": algo_spec.label,
                    "algorithm_id": algo_spec.id,
                    "run_index": run_index,
                    "seed": seed,
                    "n_obj": int(effective_n_obj),
                    "n_var": int(effective_n_var),
                    "backend": str(primary_trial.get("backend_label", cpu_backend_label)),
                    "backend_code": str(primary_trial.get("backend_code", "cpu")),
                    "time_s": _float_or_nan(primary_trial.get("time_s")),
                    "evaluations": int(primary_trial.get("evaluations", -1)),
                    "metrics": dict(primary_trial.get("metrics", {})),
                    "history": dict(primary_trial.get("history", {})),
                    "generations": list(primary_trial.get("generations", [])),
                    "final_front": np.asarray(primary_trial.get("final_front", []), dtype=float).tolist(),
                    "reference_front": (
                        None
                        if pareto_front is None
                        else np.asarray(pareto_front, dtype=float).tolist()
                    ),
                    "final_population": dict(primary_trial.get("final_population", {})),
                    "block_profile": primary_trial.get("block_profile"),
                    "profile_cpu_time_s": cpu_time,
                    "profile_gpu_time_s": gpu_time,
                    "profile_speedup_gpu_vs_cpu": speedup,
                }
                completed_dt = datetime.now().astimezone()
                run_payload["timestamp_iso"] = completed_dt.isoformat()
                run_payload["timestamp_en_us"] = format_timestamp_en_us(completed_dt)
                run_payload["timestamp_epoch"] = float(completed_dt.timestamp())
                with lock:
                    if not self._cancelled:
                        results[algo_spec.label].append(run_payload)
                        self.run_ready.emit(run_payload)
                        done_runs_ref[0] += 1
                        progress = int(100 * done_runs_ref[0] / max(1, total_runs))
                        emit_progress(progress, f"{algo_spec.label} trial {run_index}/{n_runs} finished")

            except RuntimeError as exc:
                if "__cancelled__" not in str(exc):
                    self.warning.emit(f"{algo_spec.label} trial {run_index} skipped: {exc}")
                    with lock:
                        done_runs_ref[0] += 1
                        progress = int(100 * done_runs_ref[0] / max(1, total_runs))
                        emit_progress(progress, f"{algo_spec.label} trial {run_index} skipped")
            except Exception as exc:  # noqa: BLE001
                self.warning.emit(f"{algo_spec.label} trial {run_index} skipped: {exc}")
                with lock:
                    done_runs_ref[0] += 1
                    progress = int(100 * done_runs_ref[0] / max(1, total_runs))
                    emit_progress(progress, f"{algo_spec.label} trial {run_index} skipped")

        tasks = [(algo, idx) for algo in valid_algorithms for idx in range(1, n_runs + 1)]
        seed_plan = build_seed_plan(self.config, len(tasks))
        seed_lookup: dict[tuple[str, int], int] = {}
        for idx, task in enumerate(tasks):
            algo_spec, run_idx = task
            seed_lookup[(algo_spec.id, int(run_idx))] = int(seed_plan["seeds"][idx])

        # -----------------------------------------------------------
        # Decide whether ProcessPoolExecutor is viable or whether the
        # ThreadPoolExecutor fallback is required for Qt signal closures.
        # ProcessPoolExecutor requires:
        #   1. parallel_workers > 1 (to avoid unnecessary overhead)
        #   2. no profile compare (which needs two calls per trial)
        #   3. a serializable worker (no closure/lambda/Qt objects)
        # -----------------------------------------------------------
        use_process_pool = bool(
            parallel_workers > 1
            and not profile_compare_enabled
        )

        if use_process_pool:
            # ===== ProcessPoolExecutor path (isolated subprocesses) =====
            try:
                from emopylab_process_worker import run_trial_in_process
            except ImportError:
                try:
                    from emopylab_process_worker import run_trial_in_process
                except ImportError:
                    use_process_pool = False
            emit_progress(0, f"Using ProcessPoolExecutor ({parallel_workers} workers)")
            project_root_str = str(Path(__file__).resolve().parent)

            # Extract module/class_name from each AlgorithmSpec.
            def _algo_info_for_process(algo_spec: AlgorithmSpec) -> dict[str, str]:
                """Extract serializable information from AlgorithmSpec."""
                mod_name = algo_spec.module

                # Strategy 1: extract class_name from the id.
                # Format: "source::module.ClassName"
                class_name = algo_spec.name  # fallback = display name
                spec_id = algo_spec.id
                if "::" in spec_id:
                    fqn = spec_id.split("::", 1)[1]  # "module.ClassName"
                    candidate = fqn.rsplit(".", 1)[-1]  # "ClassName"
                    if candidate:
                        class_name = candidate

                # Strategy 2: verify whether the class exists in the module.
                try:
                    mod = importlib.import_module(mod_name)
                    # Check whether the candidate extracted from the id exists.
                    if hasattr(mod, class_name):
                        return {"module": mod_name, "class_name": class_name}
                    # Fallback: normalized lookup.
                    for attr_name in dir(mod):
                        obj = getattr(mod, attr_name, None)
                        if inspect.isclass(obj) and (
                            _normalize_type_name_key(attr_name) == _normalize_type_name_key(algo_spec.name)
                        ):
                            class_name = attr_name
                            break
                except Exception:
                    pass
                return {"module": mod_name, "class_name": class_name}

            # Prepare serializable problem information.
            problem_class_name = problem_spec.name  # fallback = display name
            # Extract from id (format: "source::module.ClassName").
            if "::" in problem_spec.id:
                fqn = problem_spec.id.split("::", 1)[1]
                candidate = fqn.rsplit(".", 1)[-1]
                if candidate:
                    problem_class_name = candidate

            problem_info = {
                "source": problem_spec.source,
                "name": problem_spec.name,
                "module": problem_spec.module,
                "class_name": problem_class_name,
            }
            # Verify whether class_name exists in the module.
            try:
                prob_mod = importlib.import_module(problem_spec.module)
                if not hasattr(prob_mod, problem_class_name):
                    # Normalized lookup fallback.
                    for attr_name in dir(prob_mod):
                        obj = getattr(prob_mod, attr_name, None)
                        if inspect.isclass(obj) and (
                            _normalize_type_name_key(attr_name) == _normalize_type_name_key(problem_spec.name)
                        ):
                            problem_info["class_name"] = attr_name
                            break
            except Exception:
                pass

            problem_kwargs: dict[str, Any] = {}
            if effective_n_var > 0:
                problem_kwargs["n_var"] = int(effective_n_var)
            if effective_n_obj > 0:
                problem_kwargs["n_obj"] = int(effective_n_obj)

            process_futures: dict[concurrent.futures.Future, tuple[AlgorithmSpec, int, int]] = {}

            with concurrent.futures.ProcessPoolExecutor(max_workers=parallel_workers) as executor:
                for algo_spec, run_idx in tasks:
                    if self._cancelled:
                        break

                    seed = int(seed_lookup.get((algo_spec.id, int(run_idx)), _random_seed()))
                    algo_info = _algo_info_for_process(algo_spec)

                    # Prepare runtime_cfg for the worker without non-serializable objects.
                    worker_cfg = {
                        k: v for k, v in runtime_cfg.items()
                        if isinstance(v, (int, float, str, bool, list, dict, type(None)))
                    }
                    worker_cfg["use_gpu"] = execution_backend_code in {"gpu", "mlx"}
                    worker_cfg["gpu_dtype"] = gpu_dtype
                    worker_cfg["array_backend"] = "jax" if execution_backend_code == "gpu" else ("mlx" if execution_backend_code == "mlx" else "numpy")
                    worker_cfg["seed"] = int(seed)
                    ref_dirs_array = build_reference_dirs(
                        n_obj=int(effective_n_obj),
                        target=max(12, int(runtime_cfg.get("pop_size", 100))),
                    )
                    worker_cfg["ref_dirs"] = ref_dirs_array.tolist()

                    future = executor.submit(
                        run_trial_in_process,
                        algo_module=algo_info["module"],
                        algo_class_name=algo_info["class_name"],
                        algo_spec_id=algo_spec.id,
                        algo_spec_name=algo_spec.name,
                        problem_source=problem_info["source"],
                        problem_name=problem_info["name"],
                        problem_module=problem_info["module"],
                        problem_class_name=problem_info["class_name"],
                        problem_kwargs=problem_kwargs,
                        runtime_cfg=worker_cfg,
                        seed=seed,
                        max_fe=max_fe,
                        run_index=run_idx,
                        n_runs=n_runs,
                        project_root=project_root_str,
                    )
                    process_futures[future] = (algo_spec, run_idx, seed)

                for future in concurrent.futures.as_completed(process_futures):
                    if self._cancelled:
                        executor.shutdown(wait=False, cancel_futures=True)
                        break

                    algo_spec, run_idx, seed = process_futures[future]
                    try:
                        raw = future.result()
                    except Exception as exc:
                        self.warning.emit(f"{algo_spec.label} trial {run_idx} skipped: {exc}")
                        done_runs_ref[0] += 1
                        progress = int(100 * done_runs_ref[0] / max(1, total_runs))
                        emit_progress(progress, f"{algo_spec.label} trial {run_idx} skipped")
                        continue

                    # Compute metrics in the QThread (main process).
                    final_front_list = raw.get("final_front", [])
                    final_front = np.asarray(final_front_list, dtype=float) if final_front_list else np.empty((0, 0))
                    if final_front.ndim == 1 and final_front.size > 0:
                        final_front = final_front[None, :]

                    final_pop_data = raw.get("final_population", {})
                    final_pop_F = None
                    if final_pop_data.get("F") is not None:
                        final_pop_F = np.asarray(final_pop_data["F"], dtype=float)
                    elif final_front.size > 0:
                        final_pop_F = final_front

                    final_pop_X = None
                    if final_pop_data.get("X") is not None:
                        final_pop_X = np.asarray(final_pop_data["X"], dtype=float)

                    # Feed metric_context with population data.
                    metric_context["current_population_F"] = final_pop_F
                    metric_context["current_population_X"] = final_pop_X
                    for field in ("G", "H", "CV", "feasible"):
                        val = final_pop_data.get(field)
                        if val is not None:
                            metric_context[f"current_population_{field}"] = np.asarray(val)
                        else:
                            metric_context[f"current_population_{field}"] = None

                    metric_values: dict[str, float] = {}
                    for metric_name, metric_fn in active_metrics.items():
                        try:
                            metric_values[metric_name] = float(metric_fn(final_front))
                        except Exception:
                            metric_values[metric_name] = float("nan")

                    trial_backend_code = str(raw.get("backend_code", "cpu"))
                    trial_backend_label = str(raw.get("backend_label", cpu_backend_label))

                    run_payload: dict[str, Any] = {
                        "problem": (problem_label_override or problem_spec.label),
                        "problem_id": problem_instance_id,
                        "base_problem_id": problem_spec.id,
                        "problem_name": problem_spec.name,
                        "algorithm": algo_spec.label,
                        "algorithm_id": algo_spec.id,
                        "run_index": run_idx,
                        "seed": seed,
                        "n_obj": int(effective_n_obj),
                        "n_var": int(effective_n_var),
                        "backend": trial_backend_label,
                        "backend_code": trial_backend_code,
                        "time_s": _float_or_nan(raw.get("time_s")),
                        "evaluations": int(raw.get("evaluations", -1)),
                        "metrics": metric_values,
                        "history": {},
                        "generations": [],
                        "final_front": final_front.tolist() if final_front.size > 0 else [],
                        "reference_front": (
                            None
                            if pareto_front is None
                            else np.asarray(pareto_front, dtype=float).tolist()
                        ),
                        "final_population": final_pop_data,
                        "block_profile": None,
                        "profile_cpu_time_s": None,
                        "profile_gpu_time_s": None,
                        "profile_speedup_gpu_vs_cpu": None,
                    }
                    completed_dt = datetime.now().astimezone()
                    run_payload["timestamp_iso"] = completed_dt.isoformat()
                    run_payload["timestamp_en_us"] = format_timestamp_en_us(completed_dt)
                    run_payload["timestamp_epoch"] = float(completed_dt.timestamp())

                    if not self._cancelled:
                        results[algo_spec.label].append(run_payload)
                        self.run_ready.emit(run_payload)
                        done_runs_ref[0] += 1
                        progress = int(100 * done_runs_ref[0] / max(1, total_runs))
                        emit_progress(progress, f"{algo_spec.label} trial {run_idx}/{n_runs} finished")

        else:
            # ===== ThreadPoolExecutor path (safe fallback) =====
            # Used when parallel_workers == 1 or profile compare is enabled.
            # Keeps the closure with direct access to Qt signals and live metrics.
            with concurrent.futures.ThreadPoolExecutor(max_workers=parallel_workers) as executor:
                futures = [executor.submit(run_single_trial, algo, idx) for algo, idx in tasks]
                for future in concurrent.futures.as_completed(futures):
                    if self._cancelled:
                        executor.shutdown(wait=False, cancel_futures=True)
                        break

        if len(valid_algorithms) > 0 and len(results[valid_algorithms[0].label]) > 0:
            first_run = results[valid_algorithms[0].label][0]
            execution_backend_code = str(first_run.get("backend_code", execution_backend_code))
            execution_backend_label = str(first_run.get("backend", execution_backend_label))

        execution_manifest = build_execution_manifest(
            config=self.config,
            seed_plan=seed_plan,
            selected_problem_ids=[problem_instance_id],
            selected_algorithm_ids=selected_algorithm_ids,
            selected_metric_ids=selected_metric_ids,
            execution_backend=execution_backend_code,
            execution_backend_label=execution_backend_label,
        )

        return {
            "config": self.config,
            "results": results,
            "metrics": list(active_metrics.keys()),
            "pareto_front": None if pareto_front is None else pareto_front.tolist(),
            "pareto_fronts": {problem_instance_id: None if pareto_front is None else pareto_front.tolist()},
            "execution_backend": execution_backend_code,
            "execution_backend_label": execution_backend_label,
            "profile_compare_enabled": profile_compare_enabled,
            "experiment_manifest_version": EXPERIMENT_MANIFEST_VERSION,
            "execution_manifest": execution_manifest,
            "hv_ref_point": ref_point.tolist(),
            "hv_ref_source": ref_source,
            "cancelled": self._cancelled,
        }


class EmoPyLabMainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("EmoPyLab by METISBr")
        self.setMinimumSize(1200, 780)
        self.setWindowState(Qt.WindowState.WindowMaximized)
        self.base_dir = Path(__file__).resolve().parent
        self.plugin_dirs = ensure_plugin_directories(self.base_dir)
        (
            self.algorithm_specs,
            self.problem_specs,
            self.metric_specs,
            self.discovery_warnings,
        ) = discover_all_specs(self.base_dir)
        self.operator_specs = discover_operator_specs(self.discovery_warnings, self.base_dir)

        # Test Module state (1 algorithm, 1 problem, 1 run)
        self.test_worker_thread: QThread | None = None
        self.test_worker: ExperimentBridge | None = None
        self.test_results: dict[str, list[dict[str, Any]]] = {}
        self.test_metric_names: list[str] = []
        self.test_pareto_front: np.ndarray | None = None
        self.test_pareto_fronts: dict[str, np.ndarray | None] = {}
        self.test_execution_backend_label = "CPU"
        self.test_profile_compare_enabled = False

        # Experiment Module state (multi-algorithm, multi-problem, N runs)
        self.exp_worker_thread: QThread | None = None
        self.exp_worker: ExperimentBridge | None = None
        self.exp_results: dict[str, list[dict[str, Any]]] = {}
        self.exp_metric_names: list[str] = []
        self.exp_pareto_front: np.ndarray | None = None
        self.exp_pareto_fronts: dict[str, np.ndarray | None] = {}
        self.exp_execution_backend_label = "CPU"
        self.exp_profile_compare_enabled = False
        self.exp_problem_overrides: dict[str, dict[str, int]] = {}
        self._exp_problem_override_loading = False
        self.exp_problem_variants: dict[str, dict[str, Any]] = {}
        self._exp_problem_variant_counter = 0
        self.exp_algorithm_operator_overrides: dict[str, dict[str, Any]] = {}
        self._exp_algorithm_operator_loading = False
        self.exp_run_progress_row_by_problem: dict[str, int] = {}
        self.exp_run_progress_col_by_algorithm: dict[str, int] = {}
        self.exp_run_progress_metric_col_by_key: dict[tuple[str, str], int] = {}
        self.exp_run_progress_col_by_metric: dict[str, int] = {}
        self.exp_run_progress_counts: dict[tuple[str, str], int] = {}
        self.exp_run_progress_active_algo_by_problem: dict[str, str] = {}
        self.exp_run_progress_n_runs = 0
        self.exp_run_progress_row_meta: list[dict[str, Any]] = []
        self.exp_run_progress_algorithm_order: list[str] = []
        self.exp_run_progress_metric_order: list[str] = []
        self.exp_run_progress_metric_samples: dict[
            tuple[str, str],
            dict[str, list[float]],
        ] = {}
        self.exp_metric_runtime_cache: dict[
            tuple[str, str],
            tuple[dict[str, Any], Callable[[np.ndarray], float]],
        ] = {}
        self._exp_status_locked = False
        self._exp_tabs_enabled_snapshot: list[bool] | None = None
        self.exp_results_storage_path = self.base_dir / "experiment_module_results" / "latest_results.json"
        self.exp_results_storage_loaded_path: Path | None = None
        self.exp_results_storage_last_saved_path: Path | None = None

        # Legacy aliases for shared read-only access from results tab methods
        self.results: dict[str, list[dict[str, Any]]] = {}
        self.analysis_run_keys: set[str] = set()
        self.analysis_group_expanded: set[str] = set()
        self.analysis_group_user_touched = False
        self.analysis_row_meta: dict[int, dict[str, Any]] = {}
        self.metric_names: list[str] = []
        self.pareto_front: np.ndarray | None = None
        self.pareto_fronts: dict[str, np.ndarray | None] = {}
        self.execution_backend_label = "CPU"
        self.profile_compare_enabled = False
        self.stat_min_samples = 5
        self._last_stat_result: dict[str, Any] | None = None
        self.mcdm_last_decision: dict[str, Any] | None = None
        self.mcdm_decision_storage_path = self.base_dir / "mcdm_results" / "last_mcdm_decision.json"

        self.test_live_payload: dict[str, Any] | None = None
        self.exp_live_payload: dict[str, Any] | None = None
        self.gpu_runtime = detect_gpu_runtime()
        self.mlx_runtime = core_detect_mlx_runtime()
        self.algorithm_traits: dict[str, dict[str, Any]] = {}
        self.problem_traits: dict[str, dict[str, Any]] = {}
        self.problem_jax_compatibility: dict[str, bool] = {}
        self.algorithm_default_operator_labels_cache: dict[
            tuple[str, str, str, int, int, int],
            dict[str, str],
        ] = {}
        self._rebuild_trait_maps()

        self._local_llm_startup: LocalLLMStartupCoordinator | None = None
        self._local_llm_startup_state = "idle"
        self._local_llm_session_enabled = False
        self._local_llm_base_url = SPLASH_DEFAULT_BASE_URL
        self._local_llm_port = SPLASH_DEFAULT_MLX_PORT
        self._setup_menu()
        self._setup_ui()
        self._refresh_exp_results_storage_info()
        self._on_backend_mode_changed()
        self._on_exp_backend_mode_changed()
        self._append_registry_summary()
        for message in self.discovery_warnings:
            self._append_log(f"Registry warning: {message}")

        # Auto-load last experiment config (Phase 4)
        self._load_experiment_config_auto()
        # Auto-load persisted analysis history when available.
        self._auto_load_results_history_if_available()

    def _restore_maximized_window_mode(self) -> None:
        """Keep the main window maximized after module completion UI updates."""
        if not self.isVisible():
            return
        if self.isFullScreen() or self.isMinimized():
            return
        self.showMaximized()
        QTimer.singleShot(0, self.showMaximized)

    def _rebuild_trait_maps(self) -> None:
        self.algorithm_traits = {}
        for spec_id, spec in self.algorithm_specs.items():
            objective, encoding, difficulties, flags = infer_algorithm_traits(spec)
            self.algorithm_traits[spec_id] = {
                "objective": objective,
                "encoding": encoding,
                "difficulty": difficulties,
                "flags": flags,
            }

        self.problem_traits = {}
        for spec_id, spec in self.problem_specs.items():
            objective, encoding, difficulties = infer_problem_traits(spec)
            self.problem_traits[spec_id] = {
                "objective": objective,
                "encoding": encoding,
                "difficulty": difficulties,
            }

        # Recompute cache whenever problem registry changes.
        self.problem_jax_compatibility = {}

    def _is_jax_mode_active(self) -> bool:
        for combo_name in ("compute_backend_combo", "exp_compute_backend_combo"):
            combo = getattr(self, combo_name, None)
            if combo is None:
                continue
            token = str(combo.currentData() or "").strip().lower()
            if token in {"gpu", "jax"}:
                return True
        return False

    def _backend_aware_loading_enabled(self) -> bool:
        return bool(core_backend_aware_loading_enabled())

    def _backend_rollout_stage(self) -> str:
        return str(core_rollout_stage())

    def _rollout_allows_domain(self, domain: str) -> bool:
        if not self._backend_aware_loading_enabled():
            return False
        return bool(core_rollout_allows_domain(domain))

    def _test_backend_token(self) -> str:
        combo = getattr(self, "compute_backend_combo", None)
        if combo is None:
            return "cpu"
        token = str(combo.currentData() or "").strip().lower()
        return token if token else "cpu"

    def _exp_backend_token(self) -> str:
        combo = getattr(self, "exp_compute_backend_combo", None)
        if combo is None:
            return "cpu"
        token = str(combo.currentData() or "").strip().lower()
        return token if token else "cpu"

    def _test_backend_token_for(self, domain: str) -> str:
        if not self._rollout_allows_domain(domain):
            return "cpu"
        return self._test_backend_token()

    def _exp_backend_token_for(self, domain: str) -> str:
        if not self._rollout_allows_domain(domain):
            return "cpu"
        return self._exp_backend_token()

    def _test_backend_prefers_jax(self) -> bool:
        return self._test_backend_token() in {"gpu", "jax"}

    def _exp_backend_prefers_jax(self) -> bool:
        return self._exp_backend_token() in {"gpu", "jax"}

    def _test_backend_prefers_jax_for(self, domain: str) -> bool:
        return self._test_backend_token_for(domain) in {"gpu", "jax"}

    def _exp_backend_prefers_jax_for(self, domain: str) -> bool:
        return self._exp_backend_token_for(domain) in {"gpu", "jax"}

    def _is_jax_problem_spec(self, spec: ProblemSpec) -> bool:
        return (
            _looks_like_jax_identifier(spec.name)
            or _looks_like_jax_identifier(spec.module)
            or _looks_like_jax_identifier(spec.id)
        )

    def _is_problem_jax_compatible(self, spec: ProblemSpec) -> bool:
        cached = self.problem_jax_compatibility.get(spec.id)
        if cached is not None:
            return cached

        try:
            probe = spec.factory({"n_var": int(spec.default_n_var), "n_obj": int(spec.default_n_obj)})
        except Exception:
            probe = None

        compatible = False
        if probe is not None:
            if bool(getattr(probe, "_jax_wrapped", False)):
                compatible = True
            elif callable(getattr(probe, "_eval_F", None)):
                compatible = True
            elif callable(getattr(probe, "_evaluate", None)):
                try:
                    import warnings

                    n_var_probe = max(
                        1,
                        int(getattr(probe, "n_var", max(1, int(spec.default_n_var)))),
                    )
                    out_probe: dict[str, Any] = {}
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", RuntimeWarning)
                        probe._evaluate(np.zeros((2, n_var_probe), dtype=float), out_probe)
                    f_probe = out_probe.get("F")
                    if f_probe is not None:
                        f_arr = np.asarray(to_numpy(f_probe), dtype=float)
                        compatible = bool(f_arr.size > 0)
                except Exception:
                    compatible = False
            if not compatible and not self._is_jax_problem_spec(spec):
                wrapped = ExperimentBridge._try_create_jax_problem(
                    probe,
                    gpu_dtype="float64",
                    problem_name=spec.name,
                )
                compatible = wrapped is not None

        self.problem_jax_compatibility[spec.id] = bool(compatible)
        return bool(compatible)

    def _iter_visible_problem_specs(self, *, prefer_jax: bool | None = None) -> list[ProblemSpec]:
        sorted_specs = sorted(
            self.problem_specs.values(),
            key=lambda spec: _natural_lexicographic_key(spec.name),
        )
        if prefer_jax is None:
            prefer_jax = bool(
                self._is_jax_mode_active() and self._rollout_allows_domain("problems")
            )
        selected_specs = core_select_specs_for_backend(
            sorted_specs,
            prefer_jax=bool(prefer_jax),
            name_getter=lambda spec: spec.name,
            module_getter=lambda spec: spec.module,
            id_getter=lambda spec: spec.id,
        )
        if not prefer_jax:
            return selected_specs
        has_explicit_jax_specs = any(self._is_jax_problem_spec(spec) for spec in sorted_specs)
        if not has_explicit_jax_specs:
            return selected_specs
        # Avoid probing JAX compatibility for hundreds of problems on the UI thread.
        # Runtime execution still validates/falls back when a selected problem is not runnable.
        explicit_jax_specs = [spec for spec in selected_specs if self._is_jax_problem_spec(spec)]
        if explicit_jax_specs:
            return explicit_jax_specs
        cached_compatible = [
            spec for spec in selected_specs if self.problem_jax_compatibility.get(spec.id) is True
        ]
        return cached_compatible if cached_compatible else selected_specs

    def _is_jax_operator_spec(self, spec: OperatorSpec) -> bool:
        return (
            _looks_like_jax_identifier(spec.name)
            or _looks_like_jax_identifier(spec.module)
            or _looks_like_jax_identifier(spec.id)
        )

    def _normalize_operator_class_token(self, class_name: str) -> str:
        token = str(class_name).strip().lower()
        return _strip_jax_suffix(token).lower()

    def _iter_operator_specs_for_backend(
        self,
        operator_type: str,
        *,
        prefer_jax: bool,
    ) -> list[OperatorSpec]:
        all_specs = list(self.operator_specs.get(operator_type, {}).values())
        return core_select_specs_for_backend(
            all_specs,
            prefer_jax=bool(prefer_jax),
            name_getter=lambda spec: spec.name,
            module_getter=lambda spec: spec.module,
            id_getter=lambda spec: spec.id,
        )

    def _selected_test_algorithm_spec(self) -> AlgorithmSpec | None:
        item = self.algorithm_list.currentItem() if hasattr(self, "algorithm_list") else None
        if item is None:
            return None
        spec_id = item.data(Qt.ItemDataRole.UserRole)
        if isinstance(spec_id, str):
            return self.algorithm_specs.get(spec_id)
        return None

    def _selected_exp_algorithm_spec(self) -> AlgorithmSpec | None:
        if not hasattr(self, "exp_algorithm_list"):
            return None
        item = self.exp_algorithm_list.currentItem()
        if item is None:
            for i in range(self.exp_algorithm_list.count()):
                probe_item = self.exp_algorithm_list.item(i)
                if probe_item.checkState() == Qt.CheckState.Checked:
                    item = probe_item
                    break
        if item is None:
            return None
        spec_id = item.data(Qt.ItemDataRole.UserRole)
        if isinstance(spec_id, str):
            return self.algorithm_specs.get(spec_id)
        return None

    def _operator_probe_config(self, *, for_experiment: bool) -> dict[str, Any]:
        if for_experiment:
            n_obj = int(getattr(self, "exp_n_obj_spin", None).value()) if hasattr(self, "exp_n_obj_spin") else 2
            n_var = int(getattr(self, "exp_n_var_spin", None).value()) if hasattr(self, "exp_n_var_spin") else 30
            pop_size = (
                int(getattr(self, "exp_pop_size_spin", None).value())
                if hasattr(self, "exp_pop_size_spin")
                else DEFAULT_EXPERIMENT_POP_SIZE
            )
            token = self._exp_backend_token_for("operators")
        else:
            n_obj = int(getattr(self, "n_obj_spin", None).value()) if hasattr(self, "n_obj_spin") else 2
            n_var = int(getattr(self, "n_var_spin", None).value()) if hasattr(self, "n_var_spin") else 30
            pop_size = int(getattr(self, "pop_size_spin", None).value()) if hasattr(self, "pop_size_spin") else 100
            token = self._test_backend_token_for("operators")

        return {
            "n_obj": max(1, n_obj),
            "n_var": max(1, n_var),
            "pop_size": max(2, pop_size),
            "use_gpu": token in {"jax", "gpu", "mlx"},
            "array_backend": "jax" if token in {"jax", "gpu"} else ("mlx" if token == "mlx" else "numpy"),
            "gpu_dtype": "float32",
            # Keep the current semantics: "default" means do not override the algorithm proposal.
            "crossover": "default",
            "mutation": "default",
            "selection": "default",
            "sampling": "default",
        }

    @staticmethod
    def _get_algorithm_operator_attr(algorithm: Any, operator_type: str) -> Any:
        if algorithm is None:
            return None
        direct = getattr(algorithm, operator_type, None)
        if direct is not None:
            return direct
        mating = getattr(algorithm, "mating", None)
        if mating is not None:
            return getattr(mating, operator_type, None)
        return None

    def _algorithm_default_operator_labels(
        self,
        spec: AlgorithmSpec | None,
        *,
        prefer_jax: bool,
        for_experiment: bool,
    ) -> dict[str, str]:
        if spec is None:
            return {}

        backend_key = "jax" if prefer_jax else "cpu"
        probe_cfg = self._operator_probe_config(for_experiment=for_experiment)
        cache_key = (
            spec.id,
            backend_key,
            "exp" if for_experiment else "test",
            int(probe_cfg.get("n_obj", 2)),
            int(probe_cfg.get("n_var", 30)),
            int(probe_cfg.get("pop_size", 100)),
        )
        cached = self.algorithm_default_operator_labels_cache.get(cache_key)
        if cached is not None:
            return dict(cached)

        labels: dict[str, str] = {}
        try:
            algo = spec.factory(probe_cfg)
        except Exception:
            self.algorithm_default_operator_labels_cache[cache_key] = {}
            return {}

        for operator_type in ("crossover", "mutation", "selection", "sampling"):
            op_obj = self._get_algorithm_operator_attr(algo, operator_type)
            if op_obj is None:
                continue

            try:
                class_name = str(op_obj.__class__.__name__).strip()
            except Exception:  # noqa: BLE001
                class_name = ""
            if not class_name:
                continue

            default_name = class_name
            target_token = self._normalize_operator_class_token(class_name)
            for op_spec in self._iter_operator_specs_for_backend(operator_type, prefer_jax=prefer_jax):
                spec_token = self._normalize_operator_class_token(
                    self._operator_class_name(operator_type, op_spec.id)
                )
                if spec_token and spec_token == target_token:
                    default_name = op_spec.name
                    break

            labels[operator_type] = default_name

        self.algorithm_default_operator_labels_cache[cache_key] = dict(labels)
        return labels

    def _is_jax_metric_spec(self, spec: MetricSpec) -> bool:
        return (
            _looks_like_jax_identifier(spec.name)
            or _looks_like_jax_identifier(spec.module)
            or _looks_like_jax_identifier(spec.id)
        )

    def _iter_metric_specs_for_backend(self, *, prefer_jax: bool) -> list[MetricSpec]:
        sorted_specs = sorted(
            self.metric_specs.values(),
            key=lambda spec: _natural_lexicographic_key(spec.name),
        )
        return core_select_specs_for_backend(
            sorted_specs,
            prefer_jax=bool(prefer_jax),
            name_getter=lambda spec: spec.name,
            module_getter=lambda spec: spec.module,
            id_getter=lambda spec: spec.id,
        )

    def _map_metric_ids_to_backend(
        self,
        selected_ids: set[str] | list[str],
        *,
        prefer_jax: bool,
    ) -> list[str]:
        normalized_ids = [str(sid) for sid in selected_ids if str(sid) in self.metric_specs]
        if not normalized_ids:
            return []
        mapped = core_map_selected_ids_to_backend(
            normalized_ids,
            specs_by_id=self.metric_specs,
            prefer_jax=bool(prefer_jax),
            name_getter=lambda spec: spec.name,
            module_getter=lambda spec: spec.module,
            id_getter=lambda spec: spec.id,
        )
        return [sid for sid in mapped if sid in self.metric_specs]

    def _is_default_metric_checked(self, spec: MetricSpec) -> bool:
        metric_name_norm = re.sub(r"[^a-z0-9]+", "", spec.name.lower())
        metric_id_norm = re.sub(r"[^a-z0-9]+", "", spec.id.lower())
        # Default metric: Fast-MC Hypervolume (hv_fast / hv_fast_mc / fast_mc)
        is_hv_fast = (
            "hvfast" in metric_name_norm
            or "fastmc" in metric_name_norm
            or "hvfastmc" in metric_name_norm
            or "hvfast" in metric_id_norm
            or "fastmc" in metric_id_norm
            or "hvfastmc" in metric_id_norm
            or metric_name_norm == "hvfast"
        )
        return bool(is_hv_fast)
    def _normalize_operator_value(self, operator_type: str, value: Any) -> Any:
        if not isinstance(value, str):
            return value
        cleaned = value.strip()
        if cleaned in {"default", "none"}:
            return cleaned
        resolved = _resolve_operator_module_class(operator_type, cleaned)
        if resolved is None:
            return cleaned
        module_name, class_name = resolved
        source = SOURCE_PYMOO
        if "::" in cleaned:
            raw_source = cleaned.split("::", 1)[0].strip().lower()
            source = SOURCE_LOCAL if _is_custom_source(raw_source) else raw_source or SOURCE_PYMOO
        return f"{source}::{module_name}.{class_name}"

    def _operator_class_name(self, operator_type: str, value: Any) -> str:
        if not isinstance(value, str):
            return ""
        resolved = _resolve_operator_module_class(operator_type, value)
        if resolved is None:
            return value.strip().lower()
        return resolved[1].lower()

    def _populate_operator_combo(
        self,
        combo: QComboBox,
        operator_type: str,
        *,
        include_none: bool = False,
        prefer_jax: bool = False,
        default_label: str = "Default (EmoPyLab)",
    ) -> None:
        current = self._normalize_operator_value(operator_type, combo.currentData())
        requested_class = self._normalize_operator_class_token(
            self._operator_class_name(operator_type, current)
        )
        combo.blockSignals(True)
        combo.clear()
        combo.addItem(default_label, "default")

        all_specs = self._iter_operator_specs_for_backend(operator_type, prefer_jax=prefer_jax)
        dedup_specs: list[OperatorSpec] = []
        seen_tokens: set[str] = set()
        for spec in sorted(all_specs, key=lambda s: (s.name.lower(), s.module.lower(), s.id.lower())):
            class_token = self._normalize_operator_class_token(
                self._operator_class_name(operator_type, spec.id)
            )
            dedup_key = class_token or str(spec.id).strip().lower()
            if dedup_key in seen_tokens:
                continue
            seen_tokens.add(dedup_key)
            dedup_specs.append(spec)

        specs = dedup_specs
        name_counts: dict[str, int] = {}
        for spec in specs:
            name_counts[spec.name] = name_counts.get(spec.name, 0) + 1

        for spec in sorted(specs, key=lambda s: (s.name.lower(), s.module.lower())):
            label = spec.name
            if name_counts.get(spec.name, 0) > 1:
                label = f"{spec.name} ({spec.module.rsplit('.', 1)[-1]})"
            combo.addItem(label, spec.id)

        if include_none:
            combo.addItem(f"None (no {operator_type})", "none")

        target = current if combo.findData(current) >= 0 else None
        if target is None and requested_class:
            for spec in specs:
                spec_class = self._normalize_operator_class_token(
                    self._operator_class_name(operator_type, spec.id)
                )
                if spec_class == requested_class:
                    target = spec.id
                    break
        if target is None:
            target = "default"

        combo.setCurrentIndex(max(0, combo.findData(target)))
        combo.blockSignals(False)

    def _populate_all_operator_combos(self) -> None:
        test_pref_jax = self._test_backend_prefers_jax_for("operators")
        exp_pref_jax = self._exp_backend_prefers_jax_for("operators")
        test_default_labels = self._algorithm_default_operator_labels(
            self._selected_test_algorithm_spec(),
            prefer_jax=test_pref_jax,
            for_experiment=False,
        )
        exp_default_labels = self._algorithm_default_operator_labels(
            self._selected_exp_algorithm_spec(),
            prefer_jax=exp_pref_jax,
            for_experiment=True,
        )

        def _default_combo_label(operator_type: str, labels: dict[str, str]) -> str:
            name = str(labels.get(operator_type, "")).strip()
            if not name:
                return "Default (algorithm)"
            return f"Default ({name})"

        if hasattr(self, "crossover_combo"):
            self._populate_operator_combo(
                self.crossover_combo,
                "crossover",
                include_none=True,
                prefer_jax=test_pref_jax,
                default_label=_default_combo_label("crossover", test_default_labels),
            )
        if hasattr(self, "mutation_combo"):
            self._populate_operator_combo(
                self.mutation_combo,
                "mutation",
                include_none=True,
                prefer_jax=test_pref_jax,
                default_label=_default_combo_label("mutation", test_default_labels),
            )
        if hasattr(self, "selection_combo"):
            self._populate_operator_combo(
                self.selection_combo,
                "selection",
                include_none=False,
                prefer_jax=test_pref_jax,
                default_label=_default_combo_label("selection", test_default_labels),
            )
        if hasattr(self, "sampling_combo"):
            self._populate_operator_combo(
                self.sampling_combo,
                "sampling",
                include_none=False,
                prefer_jax=test_pref_jax,
                default_label=_default_combo_label("sampling", test_default_labels),
            )

        if hasattr(self, "exp_crossover_combo"):
            self._populate_operator_combo(
                self.exp_crossover_combo,
                "crossover",
                include_none=True,
                prefer_jax=exp_pref_jax,
                default_label=_default_combo_label("crossover", exp_default_labels),
            )
        if hasattr(self, "exp_mutation_combo"):
            self._populate_operator_combo(
                self.exp_mutation_combo,
                "mutation",
                include_none=True,
                prefer_jax=exp_pref_jax,
                default_label=_default_combo_label("mutation", exp_default_labels),
            )
        if hasattr(self, "exp_selection_combo"):
            self._populate_operator_combo(
                self.exp_selection_combo,
                "selection",
                include_none=False,
                prefer_jax=exp_pref_jax,
                default_label=_default_combo_label("selection", exp_default_labels),
            )
        if hasattr(self, "exp_sampling_combo"):
            self._populate_operator_combo(
                self.exp_sampling_combo,
                "sampling",
                include_none=False,
                prefer_jax=exp_pref_jax,
                default_label=_default_combo_label("sampling", exp_default_labels),
            )

    def _setup_menu(self) -> None:
        file_menu = self.menuBar().addMenu("File")

        act_save_cfg = QAction("Save test configuration", self)
        act_save_cfg.triggered.connect(self._save_config_to_file)
        file_menu.addAction(act_save_cfg)

        act_load_cfg = QAction("Load test configuration", self)
        act_load_cfg.triggered.connect(self._load_config_from_file)
        file_menu.addAction(act_load_cfg)

        file_menu.addSeparator()

        act_save_exp = QAction("Save experiment configuration", self)
        act_save_exp.triggered.connect(self._save_experiment_config_to_file)
        file_menu.addAction(act_save_exp)

        act_load_exp = QAction("Load experiment configuration", self)
        act_load_exp.triggered.connect(self._load_experiment_config_from_file)
        file_menu.addAction(act_load_exp)

        file_menu.addSeparator()

        act_reload = QAction("Reload registries", self)
        act_reload.triggered.connect(self._reload_registries)
        file_menu.addAction(act_reload)

        file_menu.addSeparator()

        act_quit = QAction("Exit", self)
        act_quit.triggered.connect(self.close)
        file_menu.addAction(act_quit)

        view_menu = self.menuBar().addMenu("View")
        act_light_mode = QAction(app_icon("light_mode"), "Light Theme (Default / Paper Prints)", self)
        act_light_mode.setCheckable(True)
        act_light_mode.setChecked(StylesAppStyles.current_theme == "light")
        act_light_mode.triggered.connect(lambda: self._set_workspace_theme("light"))

        act_dark_mode = QAction(app_icon("dark_mode"), "Dark Theme", self)
        act_dark_mode.setCheckable(True)
        act_dark_mode.setChecked(StylesAppStyles.current_theme == "dark")
        act_dark_mode.triggered.connect(lambda: self._set_workspace_theme("dark"))

        self._act_light_theme = act_light_mode
        self._act_dark_theme = act_dark_mode

        view_menu.addAction(act_light_mode)
        view_menu.addAction(act_dark_mode)

        help_menu = self.menuBar().addMenu("Help")

        act_guide = QAction("Getting started", self)
        act_guide.triggered.connect(self._show_getting_started_dialog)
        help_menu.addAction(act_guide)

        act_group = QAction("METISBr research group — metisbr.com", self)
        act_group.triggered.connect(
            lambda: QDesktopServices.openUrl(QUrl("https://metisbr.com"))
        )
        help_menu.addAction(act_group)

        help_menu.addSeparator()

        act_about = QAction("About EmoPyLab", self)
        act_about.triggered.connect(self._show_about_dialog)
        help_menu.addAction(act_about)

    @Slot()
    def _set_workspace_theme(self, mode: str) -> None:
        """Switch application theme between Dark (default) and Light dynamically."""
        target_mode = "dark" if mode.lower() == "dark" else "light"
        StylesAppStyles.set_theme_mode(target_mode)
        app = QApplication.instance()
        if app is not None:
            theme_xml = "dark_amber.xml" if target_mode == "dark" else "light_amber.xml"
            apply_stylesheet(
                app,
                theme=theme_xml,
                invert_secondary=(target_mode == "light"),
                extra=StylesAppStyles.get_qt_material_theme(),
            )
            app.setStyleSheet(StylesAppStyles.get_stylesheet())
        if hasattr(self, "_act_dark_theme"):
            self._act_dark_theme.setChecked(target_mode == "dark")
        if hasattr(self, "_act_light_theme"):
            self._act_light_theme.setChecked(target_mode == "light")
        if hasattr(self, "primary_workflow_tab_bar"):
            self.primary_workflow_tab_bar.update()
        if hasattr(self, "results_view_tabs"):
            self.results_view_tabs.setStyleSheet(f"""
                QTabWidget#resultsViewTabs::pane {{
                    border: 1px solid {AppStyles.BORDER_LIGHT};
                    border-radius: 8px;
                    background-color: {AppStyles.PANEL};
                    padding: 6px;
                }}
                QTabBar::tab {{
                    background: {AppStyles.PANEL_ALT};
                    color: {AppStyles.TEXT_SECONDARY};
                    border: 1px solid {AppStyles.BORDER_LIGHT};
                    border-bottom: 0px;
                    border-top-left-radius: 6px;
                    border-top-right-radius: 6px;
                    min-width: 140px;
                    padding: 7px 12px;
                    margin-right: 4px;
                    font-weight: 600;
                }}
                QTabBar::tab:selected {{
                    background: {AppStyles.PANEL};
                    color: {AppStyles.PRIMARY_LIGHT};
                    border-color: {AppStyles.BORDER_LIGHT};
                    border-bottom: 2px solid {AppStyles.PRIMARY_LIGHT};
                }}
                QTabBar::tab:hover:!selected {{
                    background: {AppStyles.PRIMARY_SUBTLE};
                    color: {AppStyles.PRIMARY_LIGHT};
                }}
            """)
        self._append_log(f"Workspace theme changed to {target_mode.upper()} mode.")

    def _show_getting_started_dialog(self) -> None:
        """Concise workflow orientation for first-time users (Help menu)."""
        QMessageBox.information(
            self,
            "Getting started with EmoPyLab",
            (
                "<b>A typical research session moves left to right across the tabs:</b>"
                "<ol>"
                "<li><b>Quick Test</b> — pair a single algorithm with a single problem "
                "for one exploratory run. Ideal for sanity-checking a configuration "
                "before committing to a full study.</li>"
                "<li><b>Experiment</b> — assemble the full study: several algorithms, "
                "several problems, multiple independent runs, matched seeds, and "
                "per-problem overrides. This is where publication-grade data is produced.</li>"
                "<li><b>Results</b> — inspect metric tables, statistical tests, and "
                "Pareto-front visualizations; export whatever the manuscript needs.</li>"
                "<li><b>AI Agent</b> — draft new algorithms, problems, or metrics with "
                "LLM assistance, then refine them by hand.</li>"
                "<li><b>Extensibility</b> — register your own plugins so they appear "
                "alongside the built-in catalog.</li>"
                "</ol>"
                "<p>Configurations can be saved and restored at any point from the "
                "<b>File</b> menu, so long studies survive interruptions.</p>"
                "<p>Questions, benchmarks, and collaborations: "
                '<a href="https://metisbr.com">METISBr research group</a>.</p>'
            ),
        )

    def _show_about_dialog(self) -> None:
        """Version, authorship, and citation pointers (Help menu)."""
        try:
            from version import __version__ as _app_version
        except Exception:  # pragma: no cover - version module is optional
            _app_version = "unknown"
        QMessageBox.about(
            self,
            "About EmoPyLab",
            (
                f"<b>EmoPyLab</b> — release {_app_version}<br>"
                "An open-source visual analytics framework for multi- and "
                "many-objective optimization.<br><br>"
                "Developed by the <a href='https://metisbr.com'>METISBr research "
                "group</a> (Multi-Objective and Many-Objective Optimization "
                "Research), Federal University of Ouro Preto (UFOP), Brazil.<br><br>"
                "If EmoPyLab contributes to your published work, please cite the "
                "arXiv preprint listed in the README "
                "(<i>EmoPyLab: An Open-Source Visual Analytics Framework for "
                "Multi-Objective Optimization using LLM-Based Code Generation "
                "and MCDM</i>).<br><br>"
                'Website: <a href="https://metisbr.com">metisbr.com</a>'
            ),
        )

    def _setup_ui(self) -> None:
        central = QWidget(self)
        central.setObjectName("appRoot")
        root = QVBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        self.tabs = QTabWidget()
        self.tabs.setObjectName("primaryWorkflowTabs")
        self.tabs.setTabBar(PrimaryWorkflowTabBar(self.tabs))
        self.tabs.setTabPosition(QTabWidget.TabPosition.North)
        self.tabs.setIconSize(QSize(28, 28))
        self.tabs.setDocumentMode(True)
        self.tabs.setAccessibleName("EmoPyLab primary workflow tabs")
        self.tabs.setAccessibleDescription(
            "Navigate across Quick Test, Experiment, Results, AI Agent, and Extensibility."
        )

        root.addWidget(self.tabs, 1)

        self.config_tab = self._build_config_tab()
        self.experiment_tab = self._build_experiment_tab()
        self.results_tab = self._build_results_tab()
        self.llm_tab = self._build_llm_tab()
        self.extensibility_tab = self._build_extensibility_tab()

        self.tabs.addTab(self.config_tab, _navigation_icon("science"), "Quick Test")
        self.tabs.addTab(self.experiment_tab, _navigation_icon("biotech"), "Experiment")
        self.tabs.addTab(self.results_tab, _navigation_icon("analytics"), "Results")
        self.tabs.addTab(self.llm_tab, _navigation_icon("smart_toy"), "AI Agent")
        self.tabs.addTab(self.extensibility_tab, _navigation_icon("code"), "Extensibility")

        # Workflow orientation: one-line tooltips so a newcomer can infer the
        # intended left-to-right research pipeline without opening the docs.
        self.tabs.setTabToolTip(
            0,
            "Run one algorithm on one problem — a fast sanity check "
            "before committing to a full experiment.",
        )
        self.tabs.setTabToolTip(
            1,
            "Design the full study: multiple algorithms, problems, and "
            "independent runs with matched seeds and per-problem overrides.",
        )
        self.tabs.setTabToolTip(
            2,
            "Analyze completed runs: metric tables, statistical tests, "
            "and Pareto-front visualizations, ready for export.",
        )
        self.tabs.setTabToolTip(
            3,
            "Draft new algorithms, problems, or metrics with LLM assistance, "
            "then refine the generated code by hand.",
        )
        self.tabs.setTabToolTip(
            4,
            "Register and manage your own plugins alongside the built-in catalog.",
        )

        self.setCentralWidget(central)

        # Build clean, always-visible, modern dark footer status bar with accelerator hardware badge
        status_bar = self.statusBar()
        status_bar.setVisible(True)
        status_bar.setMinimumHeight(32)
        status_bar.setStyleSheet("""
            QStatusBar {
                background-color: #0F172A;
                color: #94A3B8;
                border-top: 1px solid #334155;
                font-size: 11px;
                padding: 2px 8px;
            }
            QStatusBar::item { border: None; }
            QLabel { color: #94A3B8; font-size: 11px; }
            QLabel a { color: #38BDF8; text-decoration: none; font-weight: 600; }
            QLabel a:hover { color: #7DD3FC; text-decoration: underline; }
        """)

        # Hardware accelerator info badge
        from core.tensor.backend import get_backend_type, init_tensor_backend
        dev_info = init_tensor_backend()
        backend_str = dev_info.get("backend", "numpy").upper()
        platform_str = dev_info.get("platform", "cpu").upper()
        gpu_vendor = dev_info.get("gpu_vendor")
        gpu_name = dev_info.get("gpu_name")
        
        if dev_info.get("is_gpu"):
            if "apple" in platform_str.lower() or backend_str == "MLX" or gpu_vendor == "Apple":
                hw_badge_text = "⚡ NPU/GPU: Apple Silicon Neural Engine (MLX Active)"
                hw_color = "#10B981"  # Emerald
            elif gpu_vendor == "NVIDIA" or "cuda" in platform_str.lower():
                name_tag = gpu_name or f"NVIDIA CUDA ({platform_str})"
                hw_badge_text = f"⚡ GPU: {name_tag} (JAX Tensor Core)"
                hw_color = "#10B981"  # Emerald
            elif gpu_vendor == "AMD" or "rocm" in platform_str.lower():
                name_tag = gpu_name or f"AMD ROCm ({platform_str})"
                hw_badge_text = f"⚡ GPU: {name_tag} (JAX Accelerated)"
                hw_color = "#10B981"  # Emerald
            else:
                hw_badge_text = f"⚡ GPU: {backend_str} ({platform_str} Accelerated)"
                hw_color = "#10B981"  # Emerald
        else:
            cpu_count = dev_info.get("device_count", os.cpu_count() or 1)
            hw_badge_text = f"⚙️ CPU SIMD: NumPy Fallback ({cpu_count} Threads / Vectorized)"
            hw_color = "#38BDF8"  # Cyan/Blue
        hw_label = QLabel(f'<span style="color: {hw_color}; font-weight: 600;">{hw_badge_text}</span>')
        hw_label.setStyleSheet("padding-left: 6px; padding-right: 12px;")
        status_bar.addWidget(hw_label)

        # Version & Research Group Attribution
        ver_label = QLabel('<span style="color: #64748B;">EmoPyLab v1.0.1 • </span>')
        status_bar.addPermanentWidget(ver_label)

        metis_link = QLabel(
            '<a href="https://metisbr.com">'
            "METISBr Research Group (UFOP) &mdash; metisbr.com</a>"
        )
        metis_link.setOpenExternalLinks(True)
        metis_link.setStyleSheet("padding-right: 8px;")
        metis_link.setToolTip(
            "Multi-Objective and Many-Objective Evolutionary Optimization Research Group (UFOP, Brazil).\n"
            "Click to open metisbr.com"
        )
        status_bar.addPermanentWidget(metis_link)
        # Skill: Entrance animations
        self.fade_anim = QPropertyAnimation(self, b"windowOpacity")
        self.fade_anim.setDuration(500)
        self.fade_anim.setStartValue(0)
        self.fade_anim.setEndValue(1)
        self.fade_anim.setEasingCurve(QEasingCurve.Type.OutCubic)
        self.fade_anim.start()

    def _build_llm_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)
        main_row = QHBoxLayout()
        main_row.setSpacing(8)
        left_col = QVBoxLayout()
        left_col.setSpacing(8)
        right_col = QVBoxLayout()
        right_col.setSpacing(8)
        main_row.addLayout(left_col, 3)
        main_row.addLayout(right_col, 2)
        layout.addLayout(main_row, 1)

        self.llm_hint_label = QLabel()
        self.llm_hint_label.setWordWrap(True)
        self.llm_hint_label.setStyleSheet(f"color: {AppStyles.TEXT_MUTED};")
        left_col.addWidget(self.llm_hint_label)

        form = QFormLayout()
        self.llm_problem_name_edit = QLineEdit()
        self.llm_problem_name_edit.setStyleSheet(AppStyles.search_input_style())
        self.llm_problem_name_edit.setPlaceholderText("Example: WeldedBeamPrompt")
        form.addRow("Base name:", self.llm_problem_name_edit)

        self.llm_artifact_type_combo = QComboBox()
        self.llm_artifact_type_combo.addItem("Problem", "problem")
        self.llm_artifact_type_combo.addItem("Metric", "metric")
        self.llm_artifact_type_combo.currentIndexChanged.connect(self._on_llm_artifact_type_changed)
        form.addRow("Artifact type:", self.llm_artifact_type_combo)

        # LLM generation routes directly through local Qwen (models/qwen2.5-0.5b-instruct-q4_k_m.gguf).
        self.llm_provider_combo = None
        self.llm_api_key_edit = None

        self.llm_n_var_spin = QSpinBox()
        self.llm_n_var_spin.setRange(1, 10_000)
        self.llm_n_var_spin.setValue(30)
        self.llm_n_var_spin.valueChanged.connect(self._on_llm_n_var_hidden_changed)

        self.llm_n_obj_spin = QSpinBox()
        self.llm_n_obj_spin.setRange(1, 20)
        self.llm_n_obj_spin.setValue(2)
        self.llm_n_obj_spin.valueChanged.connect(self._on_llm_n_obj_hidden_changed)
        left_col.addLayout(form)

        self.llm_problem_scope_box = QGroupBox("Objective Scope")
        llm_problem_scope_layout = QVBoxLayout(self.llm_problem_scope_box)
        llm_problem_scope_layout.setSpacing(4)
        scope_row = QFormLayout()
        scope_row.setContentsMargins(0, 0, 0, 0)
        self.llm_problem_scope_combo = QComboBox()
        self.llm_problem_scope_combo.addItem("Single (n_obj = 1)", "single")
        self.llm_problem_scope_combo.addItem("Multi/Many (m >= 2)", "multi_plus")
        idx_multi = self.llm_problem_scope_combo.findData("multi_plus")
        if idx_multi >= 0:
            self.llm_problem_scope_combo.setCurrentIndex(idx_multi)
        self.llm_problem_scope_combo.currentIndexChanged.connect(self._on_llm_problem_scope_tab_changed)
        scope_row.addRow("Scope:", self.llm_problem_scope_combo)
        llm_problem_scope_layout.addLayout(scope_row)
        self.llm_problem_scope_summary_label = QLabel(
            "Describe the problem in natural language. EmoPyLab refines the prompt for pymoo/EmoPyLab compatibility."
        )
        self.llm_problem_scope_summary_label.setWordWrap(True)
        self.llm_problem_scope_summary_label.setStyleSheet(f"color: {AppStyles.TEXT_MUTED};")
        llm_problem_scope_layout.addWidget(self.llm_problem_scope_summary_label)
        left_col.addWidget(self.llm_problem_scope_box)

        self.llm_metric_mode_box = QGroupBox("Metric Generation Mode")
        llm_metric_mode_layout = QVBoxLayout(self.llm_metric_mode_box)
        llm_metric_mode_layout.setSpacing(4)
        metric_mode_row = QFormLayout()
        metric_mode_row.setContentsMargins(0, 0, 0, 0)
        self.llm_metric_mode_combo = QComboBox()
        self.llm_metric_mode_combo.addItem("Canonical wrapper (project-compatible)", "canonical_wrapper")
        self.llm_metric_mode_combo.addItem("GitHub-converted implementation", "github_converted")
        self.llm_metric_mode_combo.currentIndexChanged.connect(self._refresh_llm_metric_mode_summary)
        metric_mode_row.addRow("Mode:", self.llm_metric_mode_combo)
        llm_metric_mode_layout.addLayout(metric_mode_row)
        self.llm_metric_mode_summary_label = QLabel(
            "Canonical wrapper mode uses project-compatible wrappers whenever they are applicable."
        )
        self.llm_metric_mode_summary_label.setWordWrap(True)
        self.llm_metric_mode_summary_label.setStyleSheet(f"color: {AppStyles.TEXT_MUTED};")
        llm_metric_mode_layout.addWidget(self.llm_metric_mode_summary_label)
        self.llm_metric_mode_box.setVisible(False)
        left_col.addWidget(self.llm_metric_mode_box)

        self.llm_prompt_edit = QPlainTextEdit()
        self.llm_prompt_edit.setPlaceholderText(
            "Describe the optimization problem in natural language: objectives, constraints, ranges, and expected behavior..."
        )
        self.llm_prompt_edit.setMaximumHeight(140)
        left_col.addWidget(self.llm_prompt_edit)

        self.llm_spec_first_check = QCheckBox("Spec-first mode (builds a specification before code generation; slower)")
        self.llm_spec_first_check.setChecked(True)
        left_col.addWidget(self.llm_spec_first_check)

        action_row = QHBoxLayout()
        self.llm_generate_btn = QPushButton("Generate")
        self.llm_generate_btn.setStyleSheet(AppStyles.primary_button_style())
        self.llm_generate_btn.clicked.connect(self._on_llm_generate_clicked)
        action_row.addWidget(self.llm_generate_btn)

        self.llm_validate_btn = QPushButton("Validate")
        self.llm_validate_btn.setStyleSheet(AppStyles.secondary_button_style())
        self.llm_validate_btn.clicked.connect(self._on_llm_validate_clicked)
        action_row.addWidget(self.llm_validate_btn)

        self.llm_copy_validation_btn = QPushButton("Copy JSON")
        self.llm_copy_validation_btn.setStyleSheet(AppStyles.secondary_button_style())
        self.llm_copy_validation_btn.clicked.connect(self._on_llm_copy_validation_report_clicked)
        action_row.addWidget(self.llm_copy_validation_btn)

        self.llm_save_validation_btn = QPushButton("Save JSON...")
        self.llm_save_validation_btn.setStyleSheet(AppStyles.secondary_button_style())
        self.llm_save_validation_btn.clicked.connect(self._on_llm_save_validation_report_clicked)
        action_row.addWidget(self.llm_save_validation_btn)

        self.llm_save_key_btn = None
        self.llm_load_key_btn = None
        self.llm_get_key_btn = None
        self.llm_validate_key_btn = None

        self.llm_save_btn = QPushButton("Save + Reload")
        self.llm_save_btn.clicked.connect(self._on_llm_save_clicked)
        self.llm_save_btn.setStyleSheet(AppStyles.primary_button_style())
        action_row.addWidget(self.llm_save_btn)
        self.llm_delete_btn = QPushButton("Delete Generated Plugin")
        self.llm_delete_btn.setIcon(app_icon("delete"))
        self.llm_delete_btn.setStyleSheet(AppStyles.danger_button_style())
        self.llm_delete_btn.clicked.connect(self._on_llm_delete_generated_clicked)
        action_row.addWidget(self.llm_delete_btn)
        action_row.addStretch(1)
        left_col.addLayout(action_row)

        llm_progress_row = QHBoxLayout()
        llm_progress_row.setSpacing(8)
        self.llm_progress = QProgressBar()
        self.llm_progress.setVisible(True)
        self.llm_progress.setTextVisible(False)
        self.llm_progress.setRange(0, 100)
        self.llm_progress.setValue(0)
        llm_progress_row.addWidget(self.llm_progress, 1)
        self.llm_elapsed_label = QLabel("00:00:00")
        self.llm_elapsed_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self.llm_elapsed_label.setMinimumWidth(72)
        self.llm_elapsed_label.setStyleSheet(f"color: {AppStyles.TEXT_MUTED}; font-weight: 600;")
        self.llm_elapsed_label.setToolTip("LLM API call stopwatch (HH:MM:SS).")
        llm_progress_row.addWidget(self.llm_elapsed_label, 0)
        left_col.addLayout(llm_progress_row)

        self.llm_validation_box = QGroupBox("Validation Confidence")
        llm_val_layout = QVBoxLayout(self.llm_validation_box)
        llm_val_layout.setSpacing(4)
        self.llm_validation_summary_label = QLabel("No generated bundle yet.")
        self.llm_validation_summary_label.setWordWrap(True)
        self.llm_validation_summary_label.setStyleSheet(f"color: {AppStyles.TEXT_MUTED};")
        llm_val_layout.addWidget(self.llm_validation_summary_label)
        self.llm_validation_badge_label = QLabel("Status: n/a")
        self.llm_validation_badge_label.setStyleSheet(f"color: {AppStyles.TEXT_MUTED};")
        llm_val_layout.addWidget(self.llm_validation_badge_label)
        self.llm_validation_detail_edit = QPlainTextEdit()
        self.llm_validation_detail_edit.setReadOnly(True)
        self.llm_validation_detail_edit.setMaximumHeight(96)
        self.llm_validation_detail_edit.setPlaceholderText("Validation details will appear here.")
        self.llm_validation_detail_edit.setStyleSheet(AppStyles.log_view_style())
        llm_val_layout.addWidget(self.llm_validation_detail_edit)
        self.llm_autosave_validation_check = QCheckBox("Auto-save validation JSON with Save + Reload")
        self.llm_autosave_validation_check.setChecked(True)
        llm_val_layout.addWidget(self.llm_autosave_validation_check)
        self.llm_autosave_manifest_check = QCheckBox("Auto-save manifest JSON with Save + Reload")
        self.llm_autosave_manifest_check.setChecked(True)
        llm_val_layout.addWidget(self.llm_autosave_manifest_check)
        self.llm_validation_timestamp_check = QCheckBox("Timestamp validation and manifest sidecar files")
        self.llm_validation_timestamp_check.setChecked(False)
        llm_val_layout.addWidget(self.llm_validation_timestamp_check)
        left_col.addWidget(self.llm_validation_box)

        self.llm_history_box = QGroupBox("Recent Bundles")
        llm_hist_layout = QVBoxLayout(self.llm_history_box)
        llm_hist_layout.setSpacing(4)
        self.llm_history_list = QListWidget()
        self.llm_history_list.setMaximumHeight(110)
        self.llm_history_list.itemSelectionChanged.connect(self._on_llm_history_selection_changed)
        llm_hist_layout.addWidget(self.llm_history_list)
        left_col.addWidget(self.llm_history_box)

        self.llm_survey_bridge_box = QGroupBox("Survey -> Problem Plugin")
        llm_survey_bridge_layout = QVBoxLayout(self.llm_survey_bridge_box)
        llm_survey_bridge_layout.setSpacing(4)
        self.llm_survey_entry_combo = QComboBox()
        self.llm_survey_entry_combo.setToolTip("Select a benchmark entry from the latest benchmark survey report.")
        self.llm_survey_entry_combo.currentIndexChanged.connect(self._on_llm_survey_entry_selection_changed)
        llm_survey_bridge_layout.addWidget(self.llm_survey_entry_combo)
        llm_survey_bridge_btn_row = QHBoxLayout()
        llm_survey_bridge_btn_row.setSpacing(6)
        self.llm_survey_use_prompt_btn = QPushButton("Use Selected in Problem Prompt")
        self.llm_survey_use_prompt_btn.setStyleSheet(AppStyles.secondary_button_style())
        self.llm_survey_use_prompt_btn.clicked.connect(self._on_llm_survey_use_selected_prompt_clicked)
        llm_survey_bridge_btn_row.addWidget(self.llm_survey_use_prompt_btn, 1)
        self.llm_survey_generate_problem_btn = QPushButton("Generate Problem From Selected")
        self.llm_survey_generate_problem_btn.setStyleSheet(AppStyles.primary_button_style())
        self.llm_survey_generate_problem_btn.clicked.connect(self._on_llm_survey_generate_problem_from_selected_clicked)
        llm_survey_bridge_btn_row.addWidget(self.llm_survey_generate_problem_btn, 1)
        llm_survey_bridge_layout.addLayout(llm_survey_bridge_btn_row)
        self.llm_survey_bridge_info_label = QLabel("Run 'Benchmark Survey (Web)' first, then select one entry to generate a single Problem plugin.")
        self.llm_survey_bridge_info_label.setWordWrap(True)
        self.llm_survey_bridge_info_label.setStyleSheet(f"color: {AppStyles.TEXT_MUTED};")
        llm_survey_bridge_layout.addWidget(self.llm_survey_bridge_info_label)
        self.llm_survey_bridge_box.setVisible(False)
        left_col.addWidget(self.llm_survey_bridge_box)

        code_box = QGroupBox("Generated Artifact Preview")
        code_box_layout = QVBoxLayout(code_box)
        code_box_layout.setContentsMargins(6, 6, 6, 6)
        self.llm_code_edit = QPlainTextEdit()
        self.llm_code_edit.setReadOnly(True)
        self.llm_code_edit.setPlaceholderText("Generated CPU and JAX code previews will appear here.")
        self.llm_code_edit.setStyleSheet(AppStyles.log_view_style())
        code_box_layout.addWidget(self.llm_code_edit)
        right_col.addWidget(code_box, 1)

        raw_box = QGroupBox("API raw output")
        raw_box_layout = QVBoxLayout(raw_box)
        raw_box_layout.setContentsMargins(6, 6, 6, 6)
        self.llm_raw_response_edit = QPlainTextEdit()
        self.llm_raw_response_edit.setReadOnly(True)
        self.llm_raw_response_edit.setPlaceholderText("Raw API response will appear here after a remote generation call.")
        self.llm_raw_response_edit.setStyleSheet(AppStyles.log_view_style())
        raw_box_layout.addWidget(self.llm_raw_response_edit)
        # Keep internal raw-response capture available, but suppress the panel in the current UI/UX.
        raw_box.setVisible(False)
        right_col.addWidget(raw_box, 1)

        local_ai_row = QHBoxLayout()
        local_ai_row.setSpacing(8)
        self.llm_status_label = QLabel("Local AI is off for this session.")
        self.llm_status_label.setWordWrap(True)
        self.llm_status_label.setStyleSheet(f"color: {AppStyles.TEXT_MUTED};")
        local_ai_row.addWidget(self.llm_status_label, 1)
        self.llm_local_ai_progress = QProgressBar()
        self.llm_local_ai_progress.setRange(0, 1)
        self.llm_local_ai_progress.setTextVisible(False)
        self.llm_local_ai_progress.setFixedHeight(8)
        self.llm_local_ai_progress.setMaximumWidth(132)
        self.llm_local_ai_progress.setVisible(False)
        self.llm_local_ai_progress.setAccessibleName("Local AI loading progress")
        self.llm_local_ai_progress.setToolTip("Local AI is loading in the background. EmoPyLab remains available.")
        local_ai_row.addWidget(self.llm_local_ai_progress, 0)
        self.llm_start_local_btn = QPushButton("Start Local AI")
        self.llm_start_local_btn.setIcon(app_icon("smart_toy"))
        self.llm_start_local_btn.setStyleSheet(AppStyles.secondary_button_style())
        self.llm_start_local_btn.setToolTip(
            "Start MLX-LM for this session. This is optional and does not affect optimization workflows."
        )
        self.llm_start_local_btn.clicked.connect(self._on_start_local_llm_clicked)
        local_ai_row.addWidget(self.llm_start_local_btn, 0)

        self.llm_stop_local_btn = QPushButton("Stop Local AI")
        self.llm_stop_local_btn.setIcon(app_icon("stop"))
        self.llm_stop_local_btn.setStyleSheet(AppStyles.danger_button_style())
        self.llm_stop_local_btn.setToolTip("Terminate the running local MLX-LM process and free memory/port.")
        self.llm_stop_local_btn.setVisible(False)
        self.llm_stop_local_btn.clicked.connect(self._on_stop_local_llm_clicked)
        local_ai_row.addWidget(self.llm_stop_local_btn, 0)

        left_col.addLayout(local_ai_row)

        self.llm_generated_bundle: dict[str, Any] | None = None
        self.llm_generated_bundle_validated = False
        self.llm_api_key_source = "local_qwen"
        self.llm_generation_history: list[dict[str, Any]] = []
        self.llm_last_survey_bundle: dict[str, Any] | None = None
        self._llm_survey_bridge_entries: list[dict[str, Any]] = []
        self.llm_progress_loop_timer = QTimer(self)
        self.llm_progress_loop_timer.setInterval(90)
        self.llm_progress_loop_timer.timeout.connect(self._on_llm_progress_loop_tick)
        self.llm_elapsed_timer = QTimer(self)
        self.llm_elapsed_timer.setInterval(1000)
        self.llm_elapsed_timer.timeout.connect(self._on_llm_elapsed_tick)
        self._llm_progress_loop_value = 0
        self._llm_progress_loop_direction = 1
        self._llm_busy_started_at: float | None = None
        self._llm_busy_active = False
        self._llm_task_thread: QThread | None = None
        self._llm_task_bridge: LLMTaskBridge | None = None
        self._llm_stream_preview_text = ""
        self._llm_stream_preview_phase = ""
        self._llm_stream_preview_started = False
        self._refresh_llm_metric_mode_summary()
        self.llm_raw_response_edit.setPlainText(self._render_llm_api_raw_text(None))
        self._update_llm_validation_confidence_panel(None)
        self._refresh_llm_survey_bridge_controls()

        self._on_llm_artifact_type_changed()

        return tab

    def configure_local_llm_session(self, *, enabled: bool, base_url: str, port: int) -> None:
        """Apply the splash choice without persisting an always-start preference."""
        self._local_llm_session_enabled = bool(enabled)
        self._local_llm_base_url = str(base_url).rstrip("/") or SPLASH_DEFAULT_BASE_URL
        self._local_llm_port = int(port)
        if not self._local_llm_session_enabled:
            self._apply_local_llm_startup_update(
                {
                    "state": "idle",
                    "message": "Local AI is off for this session. Start it any time from this tab.",
                }
            )
            return
        QTimer.singleShot(0, self.start_local_llm_startup)

    def start_local_llm_startup(self) -> None:
        """Start MLX-LM only after an explicit splash or AI Agent choice."""
        if self._local_llm_startup is not None and self._local_llm_startup.running:
            return

        self._local_llm_session_enabled = True
        self._local_llm_startup_state = "working"
        self._apply_local_llm_startup_update(
            {
                "state": "working",
                "message": "Preparing local AI in the background…",
                "base_url": self._local_llm_base_url,
            }
        )
        self._local_llm_startup = LocalLLMStartupCoordinator(
            base_url=self._local_llm_base_url,
            port=self._local_llm_port,
            status_callback=self._queue_local_llm_startup_update,
            readiness_timeout=120.0,
            auto_start=True,
        )
        self._local_llm_startup.start()

    @Slot()
    def _on_start_local_llm_clicked(self) -> None:
        """Allow a skipped session to enable MLX-LM without restarting EmoPyLab."""
        if self._local_llm_startup is not None and self._local_llm_startup.running:
            return
        self.start_local_llm_startup()

    def _queue_local_llm_startup_update(self, update: dict[str, Any]) -> None:
        """Cross from the coordinator thread to the Qt GUI thread safely."""
        QTimer.singleShot(
            0,
            self,
            lambda payload=dict(update): self._apply_local_llm_startup_update(payload),
        )

    def _apply_local_llm_startup_update(self, update: dict[str, Any]) -> None:
        """Show truthful local-AI boot state without blocking any research workflow."""
        if not isinstance(update, dict) or getattr(self, "llm_status_label", None) is None:
            return
        state = str(update.get("state", "working")).strip().lower()
        message = str(update.get("message", "Checking local AI…")).strip()
        if not message:
            return

        color_by_state = {
            "success": AppStyles.SUCCESS,
            "warning": AppStyles.WARNING,
            "danger": AppStyles.ERROR,
            "working": AppStyles.INFO,
            "idle": AppStyles.TEXT_MUTED,
        }
        self._local_llm_startup_state = state
        self.llm_status_label.setText(message)
        self.llm_status_label.setStyleSheet(f"color: {color_by_state.get(state, AppStyles.TEXT_MUTED)};")
        self.llm_status_label.setToolTip(
            "Local AI startup does not block Quick Test, Experiment, Results, or Extensibility."
        )
        button = getattr(self, "llm_start_local_btn", None)
        stop_btn = getattr(self, "llm_stop_local_btn", None)
        progress = getattr(self, "llm_local_ai_progress", None)
        loading = state == "working" and str(update.get("stage", "")) in {"runtime", "starting", "loading"}
        if progress is not None:
            progress.setHidden(not loading)
            if loading:
                progress.setRange(0, 0)
                progress.show()
            else:
                progress.setRange(0, 1)
                progress.setValue(0)
        if button is not None:
            if bool(update.get("ready")):
                button.setText("Local AI Ready")
                button.setEnabled(False)
                if stop_btn is not None:
                    stop_btn.setVisible(True)
                    stop_btn.setEnabled(True)
            elif bool(update.get("terminal")) and state in {"idle", "warning", "danger"}:
                button.setText("Start Local AI")
                button.setEnabled(True)
                if stop_btn is not None:
                    stop_btn.setVisible(False)
            elif loading:
                button.setText("Starting Local AI…")
                button.setEnabled(False)
                if stop_btn is not None:
                    stop_btn.setVisible(False)

    @Slot()
    def _on_stop_local_llm_clicked(self) -> None:
        """Stop local MLX-LM server safely across platforms."""
        from core.llm.local_llm import terminate_local_server
        try:
            if self._local_llm_startup is not None:
                self._local_llm_startup.cancel_waiting()
            port = getattr(self, "_local_llm_port", 8080)
            terminate_local_server(port)
            self._apply_local_llm_startup_update({
                "state": "idle",
                "message": "Local AI stopped. Port freed.",
                "terminal": True,
                "ready": False,
            })
        except Exception as exc:
            self._apply_local_llm_startup_update({
                "state": "warning",
                "message": f"Could not stop local AI: {exc}",
                "terminal": True,
                "ready": False,
            })

    def _llm_provider_token(self) -> str:
        return CoreLLMFormulationService.DEFAULT_PROVIDER

    def _llm_artifact_token(self) -> str:
        token = self.llm_artifact_type_combo.currentData()
        return str(token or "problem").strip().lower()

    def _llm_metric_generation_mode_token(self) -> str:
        # Metric generation is now fixed to GitHub-converted mode (no UI strategy toggle).
        return "github_converted"

    def _llm_base_name(self) -> str:
        raw = self.llm_problem_name_edit.text().strip()
        if raw:
            return raw
        token = self._llm_artifact_token()
        if token == "metric":
            return "generated_metric"
        return "GeneratedProblem"

    def _llm_provider_token(self) -> str:
        return CoreLLMFormulationService.DEFAULT_PROVIDER

    def _llm_model_name(self) -> str:
        return CoreLLMFormulationService.DEFAULT_LOCAL_MODEL
    @staticmethod
    def _llm_problem_scope_token_from_n_obj(n_obj: int) -> str:
        try:
            val = int(n_obj)
        except Exception:
            val = 2
        if val <= 1:
            return "single"
        return "multi_plus"

    @staticmethod
    def _llm_problem_scope_default_n_obj(scope_token: str) -> int:
        token = str(scope_token or "").strip().lower()
        if token == "single":
            return 1
        return 2

    def _llm_problem_prompt_policy_summary_text(self) -> str:
        n_var = int(self.llm_n_var_spin.value()) if getattr(self, "llm_n_var_spin", None) is not None else 30
        n_obj = int(self.llm_n_obj_spin.value()) if getattr(self, "llm_n_obj_spin", None) is not None else 2
        scope_token = self._llm_problem_scope_token_from_n_obj(n_obj)
        if scope_token == "single":
            scope_desc = "Single-objective"
        else:
            scope_desc = "Multi/Many-objective (m >= 2)"
        return (
            f"{scope_desc} preset selected (internal dimensions: n_var={n_var}, n_obj={n_obj}). "
            "Write the request in natural language; EmoPyLab refines the prompt for pymoo/EmoPyLab compatibility."
        )

    def _refresh_llm_problem_scope_summary(self) -> None:
        if getattr(self, "llm_problem_scope_summary_label", None) is None:
            return
        self.llm_problem_scope_summary_label.setText(self._llm_problem_prompt_policy_summary_text())

    @Slot()
    def _refresh_llm_metric_mode_summary(self) -> None:
        if getattr(self, "llm_metric_mode_summary_label", None) is None:
            return
        self.llm_metric_mode_summary_label.setText(
            "Metric generation runs on the local Qwen model, then validates CPU/JAX runtime parity."
        )
        if getattr(self, "llm_problem_scope_combo", None) is None or getattr(self, "llm_n_obj_spin", None) is None:
            return
        target_token = self._llm_problem_scope_token_from_n_obj(int(self.llm_n_obj_spin.value()))
        idx = self.llm_problem_scope_combo.findData(target_token)
        if idx >= 0:
            prev = self.llm_problem_scope_combo.blockSignals(True)
            try:
                self.llm_problem_scope_combo.setCurrentIndex(idx)
            finally:
                self.llm_problem_scope_combo.blockSignals(prev)
        self._refresh_llm_problem_scope_summary()

    @Slot(int)
    def _on_llm_problem_scope_tab_changed(self, index: int) -> None:
        if index < 0 or getattr(self, "llm_n_obj_spin", None) is None:
            return
        token = ""
        if getattr(self, "llm_problem_scope_combo", None) is not None:
            token = str(self.llm_problem_scope_combo.itemData(index) or "").strip().lower()
        preset_n_obj = int(self._llm_problem_scope_default_n_obj(token))
        if int(self.llm_n_obj_spin.value()) != preset_n_obj:
            self.llm_n_obj_spin.setValue(preset_n_obj)
        else:
            self._refresh_llm_problem_scope_summary()

    @Slot(int)
    def _on_llm_n_obj_hidden_changed(self, _value: int) -> None:
        self._sync_llm_problem_scope_tab_from_n_obj()

    @Slot(int)
    def _on_llm_n_var_hidden_changed(self, _value: int) -> None:
        self._refresh_llm_problem_scope_summary()

    def _llm_hint_text(self) -> str:
        token = self._llm_artifact_token()
        if token == "metric":
            return (
                "Generate a local metric plugin and the matching _JAX variant (create_metric(context)) via local Qwen "
                f"({CoreLLMFormulationService.DEFAULT_LOCAL_MODEL}). The workflow validates both modules before saving to metrics/. "
                "Use precise names when possible (HV/IGD/GD/DeltaP), and describe semantics and inputs explicitly. "
                "Scope: one plugin per request."
            )
        if token == "benchmark_survey":
            return (
                "Benchmark survey extraction mode. "
                "Build a source-linked inventory of benchmark functions and reported dimensions."
            )
        return (
            "Generate one pymoo Problem plugin (CPU + _JAX) from natural language via local Qwen "
            f"({CoreLLMFormulationService.DEFAULT_LOCAL_MODEL}), with code aligned to EmoPyLab conventions. "
            "Use objective scope to set the n_obj preset. The internal prompt reinforces vectorized `Problem` code, "
            "`out['F']`, constraints in `out['G']`/`out['H']`, CPU/JAX parity, and no `if __name__ == '__main__':` block. "
            "Scope: one Problem plugin per request."
        )

    @staticmethod
    def _llm_problem_prompt_placeholder_example() -> str:
        return (
            "Implement the IGD indicator using the p-norm for an arbitrary number of objectives."
        )

    def _llm_is_survey_mode(self) -> bool:
        return self._llm_artifact_token() == "benchmark_survey"

    def _sync_llm_mode_control_states(self) -> None:
        if not getattr(self, "llm_artifact_type_combo", None):
            return
        is_survey = self._llm_is_survey_mode()
        is_problem = self._llm_artifact_token() == "problem"
        busy = bool(getattr(self, "_llm_busy_active", False))
        plugin_only_controls = [
            getattr(self, "llm_validate_btn", None),
            getattr(self, "llm_copy_validation_btn", None),
            getattr(self, "llm_save_validation_btn", None),
            getattr(self, "llm_save_btn", None),
            getattr(self, "llm_delete_btn", None),
            getattr(self, "llm_autosave_validation_check", None),
            getattr(self, "llm_autosave_manifest_check", None),
            getattr(self, "llm_validation_timestamp_check", None),
        ]
        for widget in plugin_only_controls:
            if widget is None:
                continue
            widget.setEnabled((not busy) and (not is_survey))
        if getattr(self, "llm_validation_box", None) is not None:
            self.llm_validation_box.setVisible(not is_survey)
        if getattr(self, "llm_spec_first_check", None) is not None:
            self.llm_spec_first_check.setEnabled((not busy) and (not is_survey))
            self.llm_spec_first_check.setVisible(not is_survey)
        if getattr(self, "llm_problem_scope_box", None) is not None:
            self.llm_problem_scope_box.setVisible(is_problem)
            self.llm_problem_scope_box.setEnabled((not busy) and is_problem)
        if getattr(self, "llm_metric_mode_box", None) is not None:
            is_metric = self._llm_artifact_token() == "metric"
            # Strategy toggle removed from UX; metric generation is fixed to GitHub-only conversion.
            self.llm_metric_mode_box.setVisible(False)
            self.llm_metric_mode_box.setEnabled(False if is_metric else False)
        self._sync_llm_problem_scope_tab_from_n_obj()

    @staticmethod
    def _llm_is_survey_bundle(bundle: dict[str, Any] | None) -> bool:
        if not isinstance(bundle, dict):
            return False
        return str(bundle.get("artifact_type", "")).strip().lower() == "benchmark_survey"

    @staticmethod
    def _llm_survey_bundle_entries(bundle: dict[str, Any] | None) -> list[dict[str, Any]]:
        if not isinstance(bundle, dict):
            return []
        report = bundle.get("survey_report", {})
        if not isinstance(report, dict):
            return []
        entries = report.get("benchmark_entries", [])
        if not isinstance(entries, list):
            return []
        cleaned: list[dict[str, Any]] = []
        for item in entries:
            if isinstance(item, dict):
                cleaned.append(item)
        return cleaned

    @staticmethod
    def _llm_survey_entry_display_label(entry: dict[str, Any], idx: int) -> str:
        name = str(entry.get("name", "unknown")).strip() or "unknown"
        family = str(entry.get("family", "") or "").strip()
        n_var = entry.get("n_var", None)
        n_obj = entry.get("n_obj", None)
        confidence = str(entry.get("confidence", "n/a") or "n/a").strip()
        return (
            f"{idx}. {name}"
            + (f" [{family}]" if family else "")
            + f" | n_var={n_var if n_var not in (None, '') else 'null'}"
            + f" | n_obj={n_obj if n_obj not in (None, '') else 'null'}"
            + f" | conf={confidence}"
        )

    @staticmethod
    def _llm_parse_single_positive_int(value: Any) -> int | None:
        if isinstance(value, bool):
            return None
        if isinstance(value, (int, np.integer)):
            iv = int(value)
            return iv if iv > 0 else None
        if isinstance(value, (float, np.floating)):
            fv = float(value)
            if np.isfinite(fv) and abs(fv - round(fv)) < 1e-9 and fv > 0:
                return int(round(fv))
            return None
        txt = str(value or "").strip()
        if not txt:
            return None
        ints = [int(x) for x in re.findall(r"\d+", txt)]
        if not ints:
            return None
        uniq = list(dict.fromkeys(ints))
        if len(uniq) == 1 and uniq[0] > 0:
            return uniq[0]
        return None

    @staticmethod
    def _llm_sanitize_problem_base_name(raw: str) -> str:
        text = re.sub(r"[^0-9A-Za-z]+", " ", str(raw or "")).strip()
        parts = [p for p in text.split() if p]
        if not parts:
            name = "GeneratedSurveyProblem"
        else:
            name = "".join(p[:1].upper() + p[1:] for p in parts)
        if name and name[0].isdigit():
            name = f"B{name}"
        if not name.endswith("Prompt"):
            name = f"{name}Prompt"
        return name

    def _llm_problem_base_name_from_survey_entry(self, entry: dict[str, Any]) -> str:
        preferred = (
            str(entry.get("name", "") or "").strip()
            or str(entry.get("family", "") or "").strip()
            or "GeneratedSurveyProblem"
        )
        return self._llm_sanitize_problem_base_name(preferred)

    @staticmethod
    def _llm_problem_prompt_from_survey_entry(entry: dict[str, Any]) -> str:
        name = str(entry.get("name", "unknown")).strip() or "unknown"
        family = str(entry.get("family", "") or "").strip()
        source_title = str(entry.get("source_title", "") or "").strip()
        source_url = str(entry.get("source_url", "") or "").strip()
        n_var = str(entry.get("n_var", "") or "").strip()
        n_obj = str(entry.get("n_obj", "") or "").strip()
        dim_stmt = str(entry.get("dimension_statement", "") or "").strip()
        evidence = str(entry.get("evidence", "") or "").strip()
        notes = str(entry.get("notes", "") or "").strip()
        lines = [
            "Generate one pymoo Problem plugin and a matching _JAX variant for the benchmark entry below.",
            "This is a single-problem generation request (not a list/catalog request).",
            "",
            f"Benchmark name: {name}",
            f"Family: {family or 'n/a'}",
            f"Reported n_var: {n_var or 'unspecified/variable'}",
            f"Reported n_obj: {n_obj or 'unspecified'}",
            f"Dimension statement: {dim_stmt or 'n/a'}",
            f"Evidence note: {evidence or 'n/a'}",
            f"Source title: {source_title or 'n/a'}",
            f"Source URL: {source_url or 'n/a'}",
            f"Notes: {notes or 'n/a'}",
            "",
            "Requirements:",
            "- Generate exactly one Problem plugin pair (CPU and JAX) compatible with EmoPyLab.",
            "- Use `from pymoo.core.problem import Problem` and vectorized `_evaluate(self, X, out, *args, **kwargs)`.",
            "- Set `out['F']` with shape (N, n_obj) and define valid `xl` / `xu` bounds.",
            "- If constrained, use pymoo constraint API (`n_ieq_constr` / `n_eq_constr`) and write `out['G']` / `out['H']`.",
            "- If n_var varies by benchmark instance/suite, expose n_var as a constructor parameter and document supported values in the class docstring.",
            "- If the benchmark depends on an external package/dataset/suite (e.g., COCO), create a wrapper integration and state the dependency/import clearly.",
            "- Use minimization convention and vectorized evaluation where possible; keep CPU/JAX parity.",
            "- Preserve the benchmark semantics from the source evidence; do not invent formulas not supported by the source context.",
            "- No markdown fences, no `if __name__ == '__main__':`, and no test harness in the returned modules.",
        ]
        return "\n".join(lines).strip() + "\n"

    def _llm_selected_survey_bridge_entry(self) -> dict[str, Any] | None:
        combo = getattr(self, "llm_survey_entry_combo", None)
        if combo is None:
            return None
        data = combo.currentData()
        return data if isinstance(data, dict) else None

    def _update_llm_survey_bridge_info_from_selected(self) -> None:
        label = getattr(self, "llm_survey_bridge_info_label", None)
        if label is None:
            return
        entry = self._llm_selected_survey_bridge_entry()
        if not isinstance(entry, dict):
            if not self._llm_survey_bridge_entries:
                label.setText("Run 'Benchmark Survey (Web)' first, then select one entry to generate a single Problem plugin.")
            else:
                label.setText("Select a benchmark entry to build a single-problem generation prompt.")
            return
        source_url = str(entry.get("source_url", "") or "").strip()
        dim_stmt = str(entry.get("dimension_statement", "") or "").strip()
        n_var = entry.get("n_var", None)
        n_obj = entry.get("n_obj", None)
        label_lines = [
            f"Selected benchmark -> n_var={n_var if n_var not in (None, '') else 'null'} | n_obj={n_obj if n_obj not in (None, '') else 'null'}",
        ]
        if dim_stmt:
            label_lines.append(f"Dims: {dim_stmt}")
        if source_url:
            label_lines.append(f"Source: {source_url}")
        label.setText("\n".join(label_lines))

    def _refresh_llm_survey_bridge_controls(self) -> None:
        combo = getattr(self, "llm_survey_entry_combo", None)
        if combo is None:
            return
        busy = bool(getattr(self, "_llm_busy_active", False))
        current_bundle = getattr(self, "llm_generated_bundle", None)
        source_bundle = current_bundle if self._llm_is_survey_bundle(current_bundle) else getattr(self, "llm_last_survey_bundle", None)
        entries = self._llm_survey_bundle_entries(source_bundle)
        self._llm_survey_bridge_entries = entries

        prev_url = ""
        prev_name = ""
        prev = combo.currentData()
        if isinstance(prev, dict):
            prev_url = str(prev.get("source_url", "") or "").strip()
            prev_name = str(prev.get("name", "") or "").strip()

        combo.blockSignals(True)
        combo.clear()
        for idx, entry in enumerate(entries, start=1):
            combo.addItem(self._llm_survey_entry_display_label(entry, idx), entry)
        if entries:
            selected_index = 0
            if prev_name or prev_url:
                for idx, entry in enumerate(entries):
                    if (
                        str(entry.get("name", "") or "").strip() == prev_name
                        and str(entry.get("source_url", "") or "").strip() == prev_url
                    ):
                        selected_index = idx
                        break
            combo.setCurrentIndex(selected_index)
        combo.blockSignals(False)

        can_use = bool(entries) and (not busy)
        for widget in (
            getattr(self, "llm_survey_entry_combo", None),
            getattr(self, "llm_survey_use_prompt_btn", None),
            getattr(self, "llm_survey_generate_problem_btn", None),
        ):
            if widget is not None:
                widget.setEnabled(can_use)
        if getattr(self, "llm_survey_bridge_box", None) is not None:
            self.llm_survey_bridge_box.setEnabled(not busy)
        self._update_llm_survey_bridge_info_from_selected()

    def _llm_apply_survey_entry_to_problem_form(self, *, generate_now: bool) -> None:
        entry = self._llm_selected_survey_bridge_entry()
        if not isinstance(entry, dict):
            QMessageBox.information(
                self,
                "Survey -> Problem Plugin",
                "No survey benchmark entry is selected. Run a benchmark survey and select one entry first.",
            )
            self.llm_status_label.setText("Survey bridge: no benchmark entry selected.")
            self.llm_status_label.setStyleSheet(f"color: {AppStyles.WARNING};")
            return
        if self._llm_task_running():
            self.llm_status_label.setText("Llm Agent is busy. Wait for the current API call to finish.")
            self.llm_status_label.setStyleSheet(f"color: {AppStyles.WARNING};")
            return

        prompt = self._llm_problem_prompt_from_survey_entry(entry)
        base_name = self._llm_problem_base_name_from_survey_entry(entry)
        n_var_guess = self._llm_parse_single_positive_int(entry.get("n_var"))
        n_obj_guess = self._llm_parse_single_positive_int(entry.get("n_obj"))

        try:
            idx_problem = self.llm_artifact_type_combo.findData("problem")
            if idx_problem >= 0:
                self.llm_artifact_type_combo.setCurrentIndex(idx_problem)
        except Exception:
            pass
        self.llm_problem_name_edit.setText(base_name)
        self.llm_prompt_edit.setPlainText(prompt)
        if n_var_guess is not None:
            self.llm_n_var_spin.setValue(max(self.llm_n_var_spin.minimum(), min(self.llm_n_var_spin.maximum(), int(n_var_guess))))
        if n_obj_guess is not None:
            self.llm_n_obj_spin.setValue(max(self.llm_n_obj_spin.minimum(), min(self.llm_n_obj_spin.maximum(), int(n_obj_guess))))

        msg = f"Survey entry loaded into Problem prompt: {str(entry.get('name', 'unknown')).strip() or 'unknown'}."
        if n_var_guess is not None or n_obj_guess is not None:
            msg += (
                " "
                f"(n_var={n_var_guess if n_var_guess is not None else 'unchanged'}, "
                f"n_obj={n_obj_guess if n_obj_guess is not None else 'unchanged'})"
            )
        if generate_now:
            msg += " Starting generation..."
        self.llm_status_label.setText(msg)
        self.llm_status_label.setStyleSheet(f"color: {AppStyles.INFO};")

        if generate_now:
            QTimer.singleShot(0, self._on_llm_generate_clicked)

    @Slot()
    def _on_llm_survey_entry_selection_changed(self) -> None:
        self._update_llm_survey_bridge_info_from_selected()

    @Slot()
    def _on_llm_survey_use_selected_prompt_clicked(self) -> None:
        self._llm_apply_survey_entry_to_problem_form(generate_now=False)

    @Slot()
    def _on_llm_survey_generate_problem_from_selected_clicked(self) -> None:
        self._llm_apply_survey_entry_to_problem_form(generate_now=True)

    @staticmethod
    def _llm_is_within_directory(path: Path, root: Path) -> bool:
        try:
            path_r = Path(path).resolve(strict=False)
            root_r = Path(root).resolve(strict=False)
            return os.path.commonpath([str(path_r), str(root_r)]) == str(root_r)
        except Exception:
            return False

    def _llm_collect_saved_artifact_paths(self, bundle: dict[str, Any]) -> dict[str, Path]:
        out: dict[str, Path] = {}
        if not isinstance(bundle, dict):
            return out

        tracked = bundle.get("_saved_artifact_paths", {})
        if isinstance(tracked, dict):
            for key in ("cpu", "jax", "validation_json", "manifest_json"):
                raw = tracked.get(key)
                if not raw:
                    continue
                try:
                    out[key] = Path(str(raw))
                except Exception:
                    continue
            if out:
                return out

        # Fallback derivation for current-session bundles (CPU/JAX only) when tracked paths are absent.
        artifact = str(bundle.get("artifact_type", "")).strip().lower()
        cpu_file = str(bundle.get("cpu_file", "") or "").strip()
        jax_file = str(bundle.get("jax_file", "") or "").strip()
        if not cpu_file and not jax_file:
            return out
        if artifact == "metric":
            root = Path(self.base_dir) / "metrics"
        elif artifact == "problem":
            try:
                root = CoreLLMFormulationService._problem_target_dir(Path(self.base_dir), int(bundle.get("n_obj", 2) or 2))
            except Exception:
                root = Path(self.base_dir) / "problems"
        else:
            return out
        if cpu_file:
            out["cpu"] = root / cpu_file
        if jax_file:
            out["jax"] = root / jax_file
        return out

    def _llm_is_safe_generated_plugin_path(self, path: Path) -> bool:
        p = Path(path)
        if p.suffix.lower() not in {".py", ".json"}:
            return False
        base = Path(self.base_dir)
        allowed_roots = [
            base / "metrics",
            base / "problems",
        ]
        if not self._llm_is_within_directory(p, base):
            return False
        return any(self._llm_is_within_directory(p, root) for root in allowed_roots)

    @Slot()
    def _on_llm_delete_generated_clicked(self) -> None:
        bundle = getattr(self, "llm_generated_bundle", None)
        if not isinstance(bundle, dict):
            QMessageBox.warning(self, "Delete Generated Plugin", "No generated plugin bundle is loaded.")
            return
        if self._llm_is_survey_bundle(bundle):
            QMessageBox.information(self, "Delete Generated Plugin", "Benchmark survey reports are not plugin modules.")
            self.llm_status_label.setText("Delete not applicable: benchmark survey report.")
            self.llm_status_label.setStyleSheet(f"color: {AppStyles.WARNING};")
            return
        if self._llm_task_running():
            QMessageBox.information(self, "Delete Generated Plugin", "Wait for the current LLM task to finish before deleting files.")
            self.llm_status_label.setText("Delete blocked: Llm Agent is busy.")
            self.llm_status_label.setStyleSheet(f"color: {AppStyles.WARNING};")
            return
        if getattr(self, "test_worker_thread", None) is not None or getattr(self, "exp_worker_thread", None) is not None:
            QMessageBox.information(
                self,
                "Delete Generated Plugin",
                "Stop Test/Experiment execution before deleting generated plugin files.",
            )
            self.llm_status_label.setText("Delete blocked: stop Test/Experiment execution first.")
            self.llm_status_label.setStyleSheet(f"color: {AppStyles.WARNING};")
            return

        path_map = self._llm_collect_saved_artifact_paths(bundle)
        if not path_map:
            QMessageBox.information(
                self,
                "Delete Generated Plugin",
                "No saved file paths are tracked for this bundle yet.\n\nUse 'Save + Reload' first, then delete from this panel.",
            )
            self.llm_status_label.setText("Delete aborted: no tracked saved paths for current bundle.")
            self.llm_status_label.setStyleSheet(f"color: {AppStyles.WARNING};")
            return

        unsafe = [p for p in path_map.values() if not self._llm_is_safe_generated_plugin_path(p)]
        if unsafe:
            QMessageBox.warning(
                self,
                "Delete Generated Plugin",
                "Deletion aborted because one or more tracked paths are outside safe plugin directories:\n\n"
                + "\n".join(str(p) for p in unsafe[:8]),
            )
            self.llm_status_label.setText("Delete aborted: unsafe path detected.")
            self.llm_status_label.setStyleSheet(f"color: {AppStyles.ERROR};")
            return

        existing = {k: p for k, p in path_map.items() if p.exists()}
        missing = {k: p for k, p in path_map.items() if not p.exists()}
        rel_lines: list[str] = []
        for key, p in path_map.items():
            try:
                rel = p.relative_to(self.base_dir)
            except Exception:
                rel = p
            rel_lines.append(f"- {key}: {rel}" + ("" if p.exists() else " (missing)"))

        if not existing:
            QMessageBox.information(
                self,
                "Delete Generated Plugin",
                "Tracked files were not found on disk (nothing to delete).\n\n" + "\n".join(rel_lines),
            )
            self.llm_status_label.setText("Delete skipped: tracked files already absent.")
            self.llm_status_label.setStyleSheet(f"color: {AppStyles.WARNING};")
            return

        confirm = QMessageBox.question(
            self,
            "Delete Generated Plugin",
            "Delete the tracked generated plugin files?\n\n"
            + "\n".join(rel_lines)
            + "\n\nThis will also reload registries.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if confirm != QMessageBox.StandardButton.Yes:
            self.llm_status_label.setText("Delete generated plugin canceled.")
            self.llm_status_label.setStyleSheet(f"color: {AppStyles.TEXT_MUTED};")
            return

        removed: dict[str, Path] = {}
        delete_errors: list[str] = []
        for key, p in existing.items():
            try:
                p.unlink()
                removed[key] = p
            except Exception as exc:  # noqa: BLE001
                delete_errors.append(f"{key}: {exc}")

        if removed:
            try:
                self._reload_registries()
            except Exception as exc:  # noqa: BLE001
                delete_errors.append(f"reload_registries: {exc}")

        tracked_payload = dict(bundle.get("_saved_artifact_paths", {})) if isinstance(bundle.get("_saved_artifact_paths", {}), dict) else {}
        bundle["_saved_with_llm_agent"] = True
        bundle["_saved_artifact_paths"] = tracked_payload if tracked_payload else {k: str(v) for k, v in path_map.items()}
        bundle["_deleted_artifact_paths"] = {k: str(v) for k, v in removed.items()}

        removed_lines: list[str] = []
        for key, p in removed.items():
            try:
                rel = p.relative_to(self.base_dir)
            except Exception:
                rel = p
            removed_lines.append(f"- {key}: {rel}")

        if delete_errors:
            self.llm_status_label.setText(f"Delete completed with warnings ({len(removed)} removed, {len(delete_errors)} issue(s)).")
            self.llm_status_label.setStyleSheet(f"color: {AppStyles.WARNING};")
            QMessageBox.warning(
                self,
                "Delete Generated Plugin",
                "Delete finished with warnings.\n\nRemoved files:\n"
                + ("\n".join(removed_lines) if removed_lines else "(none)")
                + "\n\nWarnings:\n"
                + "\n".join(f"- {msg}" for msg in delete_errors[:8]),
            )
        else:
            self.llm_status_label.setText(f"Deleted generated plugin files ({len(removed)} file(s)); registries reloaded.")
            self.llm_status_label.setStyleSheet(f"color: {AppStyles.SUCCESS};")
            QMessageBox.information(
                self,
                "Delete Generated Plugin",
                "Generated plugin files deleted successfully and registries reloaded.\n\n"
                + "\n".join(removed_lines),
            )

    def _llm_request_intent_analysis(self, *, prompt: str, artifact_type: str, base_name: str) -> dict[str, Any]:
        try:
            analysis = CoreLLMFormulationService.analyze_request_intent(
                prompt=str(prompt or ""),
                artifact_type=str(artifact_type or "problem"),
                base_name=str(base_name or ""),
            )
        except Exception as exc:  # noqa: BLE001
            return {
                "task_kind": "plugin_generation",
                "fit_for_llm_agent": True,
                "unsupported_for_generation": False,
                "needs_external_sources": False,
                "reasons": [f"Intent analysis unavailable: {exc}"],
            }
        return analysis if isinstance(analysis, dict) else {
            "task_kind": "plugin_generation",
            "fit_for_llm_agent": True,
            "unsupported_for_generation": False,
            "needs_external_sources": False,
            "reasons": [],
        }

    def _llm_block_generate_for_out_of_scope_request(self, *, prompt: str, artifact_type: str, base_name: str) -> bool:
        analysis = self._llm_request_intent_analysis(prompt=prompt, artifact_type=artifact_type, base_name=base_name)
        if not bool(analysis.get("unsupported_for_generation", False)):
            return False
        reasons = [str(x) for x in (analysis.get("reasons", []) or []) if str(x).strip()]
        reason_text = "\n".join(f"- {msg}" for msg in reasons[:3]) if reasons else "- Prompt classified outside plugin-generation scope."
        task_kind = str(analysis.get("task_kind", "benchmark_survey")).replace("_", " ")
        msg = (
            "This request was classified as a benchmark survey/research task, not a single plugin generation task.\n\n"
            f"Detected task kind: {task_kind}\n"
            f"{reason_text}\n\n"
            "Llm Agent (this tab) now generates one Problem plugin at a time.\n"
            "Use a prompt for one specific benchmark/problem (not a conference-wide inventory/listing)."
        )
        QMessageBox.information(self, "LLM Agent Scope", msg)
        self.llm_status_label.setText("Generation blocked: prompt requires benchmark survey/research (outside Llm Agent plugin-generation scope).")
        self.llm_status_label.setStyleSheet(f"color: {AppStyles.WARNING};")
        return True

    def _render_llm_bundle_preview(self, bundle: dict[str, Any]) -> str:
        if str(bundle.get("artifact_type", "")).strip().lower() == "benchmark_survey":
            report = bundle.get("survey_report", {}) if isinstance(bundle.get("survey_report", {}), dict) else {}
            entries = report.get("benchmark_entries", []) if isinstance(report.get("benchmark_entries", []), list) else []
            warnings = report.get("warnings", []) if isinstance(report.get("warnings", []), list) else []
            missing = report.get("missing_information", []) if isinstance(report.get("missing_information", []), list) else []
            lines = [
                f"# Benchmark Survey Report ({bundle.get('base_name', 'benchmark_survey')})",
                f"Provider: {bundle.get('provider', 'n/a')} | Model: {bundle.get('model', 'n/a')}",
                f"Search results: {report.get('search_result_count', 'n/a')} | Fetched sources: {report.get('fetched_source_count', 'n/a')}",
                "",
                "## Summary",
                str(report.get("summary", "No survey summary available.")).strip() or "No survey summary available.",
                "",
                "## Scope Assessment",
                str(report.get("scope_assessment", "n/a")).strip() or "n/a",
                "",
                f"## Benchmark Entries ({len(entries)})",
            ]
            for idx, item in enumerate(entries[:30], start=1):
                if not isinstance(item, dict):
                    lines.append(f"{idx}. {item}")
                    continue
                name = str(item.get("name", "unknown")).strip() or "unknown"
                family = str(item.get("family", "") or "").strip()
                n_var = item.get("n_var", None)
                n_obj = item.get("n_obj", None)
                source_url = str(item.get("source_url", "") or "").strip()
                confidence = str(item.get("confidence", "n/a") or "n/a").strip()
                lines.append(
                    f"{idx}. {name}"
                    + (f" [{family}]" if family else "")
                    + f" | n_var={n_var if n_var not in (None, '') else 'null'}"
                    + f" | n_obj={n_obj if n_obj not in (None, '') else 'null'}"
                    + f" | confidence={confidence}"
                )
                dim_stmt = str(item.get("dimension_statement", "") or "").strip()
                if dim_stmt:
                    lines.append(f"   dims: {dim_stmt}")
                ev = str(item.get("evidence", "") or "").strip()
                if ev:
                    lines.append(f"   evidence: {ev}")
                if source_url:
                    lines.append(f"   source: {source_url}")
            if len(entries) > 30:
                lines.append(f"... {len(entries)-30} more benchmark entrie(s)")
            if warnings:
                lines.extend(["", "## Warnings"])
                lines.extend(f"- {str(w)}" for w in warnings[:10])
            if missing:
                lines.extend(["", "## Missing Information"])
                lines.extend(f"- {str(x)}" for x in missing[:10])
            return "\n".join(lines).rstrip() + "\n"
        cpu_file = str(bundle.get("cpu_file", "artifact.py"))
        jax_file = str(bundle.get("jax_file", "artifact_JAX.py"))
        cpu_code = str(bundle.get("cpu_code", "")).rstrip()
        jax_code = str(bundle.get("jax_code", "")).rstrip()
        header = [f"# Made by EmoPyLab {datetime.now().year}.", ""]
        return (
            "\n".join(header)
            + f"# ===== CPU ({cpu_file}) =====\n"
            + cpu_code
            + "\n\n"
            + f"# ===== JAX ({jax_file}) =====\n"
            + jax_code
            + "\n"
        )

    @staticmethod
    def _render_llm_api_raw_text(bundle: dict[str, Any] | None) -> str:
        if not isinstance(bundle, dict):
            return "No API response available."
        if str(bundle.get("artifact_type", "")).strip().lower() == "benchmark_survey":
            parts: list[str] = []
            search_debug = str(bundle.get("_search_debug_text", "") or "").strip()
            if search_debug:
                parts.append("# ===== OPENAI WEB_SEARCH TOOL DEBUG =====")
                parts.append(search_debug)
            raw_text = str(bundle.get("_api_raw_text", "") or "").strip()
            if raw_text:
                parts.append("# ===== LLM EXTRACTION RAW RESPONSE =====")
                parts.append(raw_text)
            if not parts:
                return "No benchmark survey raw output captured."
            return "\n\n".join(parts) + "\n"
        parts: list[str] = []
        spec_report = bundle.get("_spec_report", {})
        if isinstance(spec_report, dict):
            spec_call_debug = spec_report.get("_api_call_debug", {})
            if isinstance(spec_call_debug, dict) and spec_call_debug:
                parts.append("# ===== SPEC-FIRST API CALL DEBUG =====")
                try:
                    parts.append(json.dumps(spec_call_debug, indent=2, ensure_ascii=False))
                except Exception:
                    parts.append(str(spec_call_debug))
            raw_spec = str(spec_report.get("_raw_text", "") or "").strip()
            if raw_spec:
                parts.append("# ===== SPEC-FIRST API RAW RESPONSE =====")
                parts.append(raw_spec)
        gen_call_debug = bundle.get("_api_call_debug", {})
        if isinstance(gen_call_debug, dict) and gen_call_debug:
            parts.append("# ===== GENERATION API CALL DEBUG =====")
            try:
                parts.append(json.dumps(gen_call_debug, indent=2, ensure_ascii=False))
            except Exception:
                parts.append(str(gen_call_debug))
        raw_text = str(bundle.get("_api_raw_text", "") or "").strip()
        if raw_text:
            parts.append("# ===== GENERATION API RAW RESPONSE =====")
            parts.append(raw_text)
        repair_report = bundle.get("_repair_report", {})
        if isinstance(repair_report, dict):
            attempts = repair_report.get("attempts", [])
            if isinstance(attempts, list):
                for attempt in attempts:
                    if not isinstance(attempt, dict):
                        continue
                    attempt_debug = attempt.get("api_call_debug", {})
                    if not isinstance(attempt_debug, dict) or not attempt_debug:
                        continue
                    attempt_id = int(attempt.get("attempt", 0) or 0)
                    parts.append(f"# ===== REPAIR API CALL DEBUG (attempt {attempt_id}) =====")
                    try:
                        parts.append(json.dumps(attempt_debug, indent=2, ensure_ascii=False))
                    except Exception:
                        parts.append(str(attempt_debug))
        if not parts:
            return "No API raw response captured (local template or generated bundle loaded without API payload)."
        return "\n\n".join(parts) + "\n"

    @staticmethod
    def _llm_stream_phase_label(phase: str) -> str:
        p = str(phase or "").strip().lower()
        if p == "spec_first":
            return "Spec-first"
        if p == "generation":
            return "Generation"
        if p.startswith("repair_"):
            suffix = p.split("_", 1)[1] if "_" in p else ""
            return f"Repair {suffix}".strip()
        if p == "repair":
            return "Repair"
        return p.replace("_", " ").title() or "LLM"

    def _llm_reset_stream_preview(self) -> None:
        self._llm_stream_preview_text = ""
        self._llm_stream_preview_phase = ""
        self._llm_stream_preview_started = False

    def _llm_append_stream_preview_text(self, text: str) -> None:
        chunk = str(text or "")
        if not chunk:
            return
        current = str(getattr(self, "_llm_stream_preview_text", "") or "")
        updated = current + chunk
        self._llm_stream_preview_text = updated
        self.llm_code_edit.setPlainText(updated)
        try:
            sb = self.llm_code_edit.verticalScrollBar()
            if sb is not None:
                sb.setValue(sb.maximum())
        except Exception:
            pass

    @Slot(object)
    def _on_llm_generate_partial_update(self, payload_obj: object) -> None:
        payload = payload_obj if isinstance(payload_obj, dict) else {}
        if not payload:
            return
        if str(payload.get("kind", "")) != "llm_stream":
            return
        phase = str(payload.get("phase", "") or "").strip().lower()
        event = str(payload.get("event", "") or "").strip().lower()
        if not getattr(self, "_llm_stream_preview_started", False):
            self._llm_reset_stream_preview()
            self._llm_stream_preview_started = True
            self._llm_append_stream_preview_text(
                "# Llm Agent streaming response (partial)\n"
                "# Final generated code will replace this preview when the call finishes.\n\n"
            )
        phase_label = self._llm_stream_phase_label(phase)
        current_phase = str(getattr(self, "_llm_stream_preview_phase", "") or "")
        if event == "stage_start":
            if current_phase != phase:
                self._llm_stream_preview_phase = phase
                self._llm_append_stream_preview_text(f"\n# ----- {phase_label} -----\n")
            message = str(payload.get("message", "") or "").strip()
            if message:
                self._llm_append_stream_preview_text(f"# {message}\n")
            return
        if event == "web_search":
            if current_phase != phase:
                self._llm_stream_preview_phase = phase
                self._llm_append_stream_preview_text(f"\n# ----- {phase_label} -----\n")
            message = str(payload.get("message", "") or "").strip()
            if message:
                self._llm_append_stream_preview_text(f"# {message}\n")
            return
        if event == "stage_end":
            message = str(payload.get("message", "") or "").strip()
            if message:
                self._llm_append_stream_preview_text(f"\n# {message}\n")
            return
        if event == "text_delta":
            if current_phase != phase:
                self._llm_stream_preview_phase = phase
                self._llm_append_stream_preview_text(f"\n# ----- {phase_label} -----\n")
            self._llm_append_stream_preview_text(str(payload.get("text", "") or ""))
            return

    @staticmethod
    def _llm_validation_confidence_score(report: dict[str, Any], bundle: dict[str, Any] | None = None) -> int:
        if not isinstance(report, dict) or not report:
            return 0
        score = 100.0
        ok = bool(report.get("ok", False))
        issues = report.get("issues", [])
        runtime = report.get("runtime", {})
        if not ok:
            score -= 35.0
        if isinstance(issues, list):
            score -= min(35.0, 7.0 * len(issues))
        if isinstance(runtime, dict):
            try:
                rel = float(runtime.get("cpu_jax_rel_err", 0.0))
                if np.isfinite(rel):
                    score -= min(15.0, max(0.0, rel * 200.0))
            except Exception:
                pass
            if "oracle_skipped_reason" not in runtime:
                try:
                    oracle_rel = float(runtime.get("oracle_rel_err", 0.0))
                    if np.isfinite(oracle_rel):
                        score -= min(15.0, max(0.0, oracle_rel * 100.0))
                except Exception:
                    pass
        if isinstance(bundle, dict):
            try:
                repairs = int(bundle.get("_repair_attempts", 0) or 0)
            except Exception:
                repairs = 0
            score -= min(10.0, 3.0 * max(0, repairs))
        return int(max(0, min(100, round(score))))

    def _update_llm_validation_confidence_panel(self, bundle: dict[str, Any] | None) -> None:
        if not getattr(self, "llm_validation_summary_label", None):
            return
        if not isinstance(bundle, dict):
            self.llm_validation_summary_label.setText("No generated bundle yet.")
            self.llm_validation_summary_label.setStyleSheet(f"color: {AppStyles.TEXT_MUTED};")
            if getattr(self, "llm_validation_badge_label", None):
                self.llm_validation_badge_label.setText("Status: n/a")
                self.llm_validation_badge_label.setStyleSheet(f"color: {AppStyles.TEXT_MUTED};")
            self.llm_validation_detail_edit.setPlainText("")
            return

        report = bundle.get("_validation_report", {})
        if not isinstance(report, dict) or not report:
            self.llm_validation_summary_label.setText("No validation report is available yet.")
            self.llm_validation_summary_label.setStyleSheet(f"color: {AppStyles.WARNING};")
            if getattr(self, "llm_validation_badge_label", None):
                self.llm_validation_badge_label.setText("Status: validation not run")
            self.llm_validation_badge_label.setStyleSheet(f"color: {AppStyles.WARNING};")
            self.llm_validation_detail_edit.setPlainText("")
            return
        if str(bundle.get("artifact_type", "")).strip().lower() == "benchmark_survey":
            report = bundle.get("survey_report", {}) if isinstance(bundle.get("survey_report", {}), dict) else {}
            entries = report.get("benchmark_entries", []) if isinstance(report.get("benchmark_entries", []), list) else []
            sources = bundle.get("web_sources", []) if isinstance(bundle.get("web_sources", []), list) else []
            self.llm_validation_summary_label.setText(
                f"Benchmark survey mode (plugin validation N/A). Sources fetched: {len(sources)}. Benchmark entries: {len(entries)}."
            )
            self.llm_validation_summary_label.setStyleSheet(f"color: {AppStyles.INFO}; font-weight: 600;")
            if getattr(self, "llm_validation_badge_label", None):
                self.llm_validation_badge_label.setText("Mode: Benchmark Survey (Web)")
                self.llm_validation_badge_label.setStyleSheet(f"color: {AppStyles.INFO}; font-weight: 600;")
            details: list[str] = []
            if report.get("summary"):
                details.append(f"Summary: {report.get('summary')}")
            if report.get("scope_assessment"):
                details.append(f"Scope: {report.get('scope_assessment')}")
            details.append(f"Search queries: {report.get('search_queries', bundle.get('_search_queries', []))}")
            details.append(f"Search result count: {report.get('search_result_count', 'n/a')}")
            details.append(f"Fetched source count: {report.get('fetched_source_count', len(sources))}")
            warnings = report.get("warnings", []) if isinstance(report.get("warnings", []), list) else []
            if warnings:
                details.append("Warnings:")
                details.extend(f"- {str(w)}" for w in warnings[:8])
            missing = report.get("missing_information", []) if isinstance(report.get("missing_information", []), list) else []
            if missing:
                details.append("Missing information:")
                details.extend(f"- {str(m)}" for m in missing[:8])
            self.llm_validation_detail_edit.setPlainText("\n".join(details))
            return

        ok = bool(report.get("ok", False))
        score = self._llm_validation_confidence_score(report, bundle)
        runtime = report.get("runtime", {}) if isinstance(report.get("runtime", {}), dict) else {}
        issues = report.get("issues", []) if isinstance(report.get("issues", []), list) else []
        status_txt = "Accepted" if ok else "Rejected"
        color = AppStyles.SUCCESS if ok else AppStyles.ERROR
        self.llm_validation_summary_label.setText(
            f"{status_txt} (confidence {score}/100). "
            f"Issues: {len(issues)}. "
            f"CPU/JAX parity rel: {runtime.get('cpu_jax_rel_err', 'n/a')}."
        )
        self.llm_validation_summary_label.setStyleSheet(f"color: {color}; font-weight: 600;")
        if getattr(self, "llm_validation_badge_label", None):
            mode_label = self._llm_acceptance_mode_label(bundle)
            self.llm_validation_badge_label.setText(f"Mode: {mode_label}")
            self.llm_validation_badge_label.setStyleSheet(f"color: {self._llm_acceptance_mode_color(bundle)}; font-weight: 600;")

        details: list[str] = []
        artifact = str(report.get("artifact_type", "")).strip() or str(bundle.get("artifact_type", ""))
        if artifact:
            details.append(f"Artifact: {artifact}")
        kind_hint = runtime.get("metric_kind_hint")
        if kind_hint:
            details.append(f"Metric kind hint: {kind_hint}")
        if "cpu_value" in runtime or "jax_value" in runtime:
            details.append(f"CPU value: {runtime.get('cpu_value', 'n/a')}")
            details.append(f"JAX value: {runtime.get('jax_value', 'n/a')}")
        if "cpu_jax_abs_err" in runtime or "cpu_jax_rel_err" in runtime:
            details.append(
                f"CPU/JAX parity: abs={runtime.get('cpu_jax_abs_err', 'n/a')} rel={runtime.get('cpu_jax_rel_err', 'n/a')}"
            )
        if "oracle_skipped_reason" in runtime:
            details.append(f"Oracle sanity: skipped ({runtime.get('oracle_skipped_reason')})")
        elif "oracle_value" in runtime:
            details.append(
                f"Oracle sanity: value={runtime.get('oracle_value', 'n/a')} "
                f"abs={runtime.get('oracle_abs_err', 'n/a')} rel={runtime.get('oracle_rel_err', 'n/a')}"
            )
        repairs = bundle.get("_repair_attempts")
        if repairs:
            details.append(f"Auto-repair attempts used: {repairs}")
        spec_report = bundle.get("_spec_report", {})
        if isinstance(spec_report, dict) and spec_report:
            details.append(f"Spec-first: {bool(bundle.get('_spec_first', False))}")
            if spec_report.get("mode"):
                details.append(f"Spec mode: {spec_report.get('mode')}")
            if spec_report.get("summary"):
                details.append(f"Spec summary: {spec_report.get('summary')}")
            ambiguity = spec_report.get("ambiguity_notes", [])
            if isinstance(ambiguity, list) and ambiguity:
                details.append("Spec ambiguity notes:")
                details.extend(f"- {str(msg)}" for msg in ambiguity[:5])
        repair_report = bundle.get("_repair_report", {})
        if isinstance(repair_report, dict) and repair_report:
            details.append(f"Repair report: final_status={repair_report.get('final_status', 'unknown')}")
        if issues:
            details.append("Issues:")
            details.extend(f"- {msg}" for msg in issues[:10])
            if len(issues) > 10:
                details.append(f"... {len(issues)-10} more issue(s)")
        else:
            details.append("Issues: none")
        self.llm_validation_detail_edit.setPlainText("\n".join(details))

    def _llm_validation_dialog_summary(self, bundle: dict[str, Any] | None) -> str:
        if not isinstance(bundle, dict):
            return "Validation report unavailable."
        report = bundle.get("_validation_report", {})
        if not isinstance(report, dict) or not report:
            return "Validation report unavailable."
        runtime = report.get("runtime", {}) if isinstance(report.get("runtime", {}), dict) else {}
        issues = report.get("issues", []) if isinstance(report.get("issues", []), list) else []
        score = self._llm_validation_confidence_score(report, bundle)
        lines = [
            f"Validation confidence: {score}/100",
            f"Accepted: {bool(report.get('ok', False))}",
            f"Issues: {len(issues)}",
        ]
        if "metric_kind_hint" in runtime and runtime.get("metric_kind_hint"):
            lines.append(f"Metric kind hint: {runtime.get('metric_kind_hint')}")
        if "cpu_jax_abs_err" in runtime or "cpu_jax_rel_err" in runtime:
            lines.append(
                f"CPU/JAX parity: abs={runtime.get('cpu_jax_abs_err', 'n/a')} rel={runtime.get('cpu_jax_rel_err', 'n/a')}"
            )
        if "oracle_skipped_reason" in runtime:
            lines.append(f"Oracle sanity: skipped ({runtime.get('oracle_skipped_reason')})")
        elif "oracle_abs_err" in runtime or "oracle_rel_err" in runtime:
            lines.append(
                f"Oracle sanity: abs={runtime.get('oracle_abs_err', 'n/a')} rel={runtime.get('oracle_rel_err', 'n/a')}"
            )
        repairs = bundle.get("_repair_attempts")
        if repairs:
            lines.append(f"Auto-repair attempts used: {repairs}")
        if "_spec_first" in bundle:
            lines.append(f"Spec-first mode: {bool(bundle.get('_spec_first', False))}")
        spec_report = bundle.get("_spec_report", {})
        if isinstance(spec_report, dict) and spec_report:
            if spec_report.get("mode"):
                lines.append(f"Spec mode: {spec_report.get('mode')}")
            if spec_report.get("summary"):
                lines.append(f"Spec summary: {spec_report.get('summary')}")
        repair_report = bundle.get("_repair_report", {})
        if isinstance(repair_report, dict) and repair_report:
            lines.append(f"Repair final status: {repair_report.get('final_status', 'unknown')}")
        if issues:
            lines.append("")
            lines.append("Top issues:")
            lines.extend(f"- {msg}" for msg in issues[:5])
            if len(issues) > 5:
                lines.append(f"... {len(issues) - 5} more issue(s)")
        return "\n".join(lines)

    @staticmethod
    def _llm_json_default_serializer(obj: Any) -> Any:
        try:
            import numpy as _np

            if isinstance(obj, (_np.floating,)):
                return float(obj)
            if isinstance(obj, (_np.integer,)):
                return int(obj)
            if isinstance(obj, (_np.bool_,)):
                return bool(obj)
            if isinstance(obj, _np.ndarray):
                return obj.tolist()
        except Exception:
            pass
        return str(obj)

    def _llm_validation_report_json_text(self, bundle: dict[str, Any] | None) -> str:
        if not isinstance(bundle, dict):
            payload = {"error": "No generated bundle available."}
        elif str(bundle.get("artifact_type", "")).strip().lower() == "benchmark_survey":
            payload = {
                "artifact_type": "benchmark_survey",
                "base_name": bundle.get("base_name"),
                "provider": bundle.get("provider"),
                "model": bundle.get("model"),
                "request_intent": bundle.get("_request_intent", {}),
                "search_queries": bundle.get("_search_queries", []),
                "survey_report": bundle.get("survey_report", {}),
                "web_sources": bundle.get("web_sources", []),
            }
        else:
            payload = {
                "artifact_type": bundle.get("artifact_type"),
                "base_name": bundle.get("base_name"),
                "provider": bundle.get("provider"),
                "model": bundle.get("model"),
                "cpu_file": bundle.get("cpu_file"),
                "jax_file": bundle.get("jax_file"),
                "n_var": bundle.get("n_var"),
                "n_obj": bundle.get("n_obj"),
                "repair_attempts": bundle.get("_repair_attempts", 0),
                "spec_first": bool(bundle.get("_spec_first", False)),
                "spec_report": bundle.get("_spec_report", {}),
                "repair_report": bundle.get("_repair_report", {}),
                "validation_report": bundle.get("_validation_report", {}),
            }
        return json.dumps(payload, indent=2, ensure_ascii=False, default=self._llm_json_default_serializer)

    @staticmethod
    def _llm_now_timestamp_token() -> str:
        return datetime.now().strftime("%Y%m%d_%H%M%S")

    @staticmethod
    def _llm_acceptance_mode(bundle: dict[str, Any] | None) -> str:
        if not isinstance(bundle, dict):
            return "none"
        report = bundle.get("_validation_report", {})
        if not isinstance(report, dict):
            return "none"
        if not bool(report.get("ok", False)):
            return "rejected"
        if bundle.get("_repair_attempts"):
            return "repaired_validated"
        runtime = report.get("runtime", {}) if isinstance(report.get("runtime", {}), dict) else {}
        if str(runtime.get("metric_kind_hint", "")) == "hv_mc" and str(bundle.get("base_name", "")).upper().find("HV") >= 0:
            if "Monte Carlo Hypervolume (project-compatible" in str(bundle.get("cpu_code", "")):
                return "canonical_fallback"
        prompt = str(bundle.get("_prompt", "") or "")
        if str(bundle.get("artifact_type", "")).strip().lower() == "problem":
            cpu_code = str(bundle.get("cpu_code", ""))
            if "_CanonicalProblem" in cpu_code and "from problems." in cpu_code and "variant" not in prompt.lower():
                return "canonical_fallback"
        if "from metrics.community_metrics import _metric_" in str(bundle.get("cpu_code", "")) and "variant" not in prompt.lower():
            return "canonical_fallback"
        return "validated"

    @classmethod
    def _llm_acceptance_mode_label(cls, bundle: dict[str, Any] | None) -> str:
        mapping = {
            "none": "Status: n/a",
            "rejected": "Rejected",
            "repaired_validated": "Accepted (auto-repaired)",
            "canonical_fallback": "Accepted (canonical fallback)",
            "validated": "Accepted (LLM-generated and validated)",
        }
        return mapping.get(cls._llm_acceptance_mode(bundle), "Accepted")

    @classmethod
    def _llm_acceptance_mode_color(cls, bundle: dict[str, Any] | None) -> str:
        mode = cls._llm_acceptance_mode(bundle)
        if mode == "rejected":
            return AppStyles.ERROR
        if mode == "canonical_fallback":
            return AppStyles.INFO
        if mode == "repaired_validated":
            return AppStyles.WARNING
        if mode == "validated":
            return AppStyles.SUCCESS
        return AppStyles.TEXT_MUTED

    def _llm_build_manifest_payload(self, bundle: dict[str, Any], *, cpu_path: Path | None = None, jax_path: Path | None = None) -> dict[str, Any]:
        report = bundle.get("_validation_report", {}) if isinstance(bundle.get("_validation_report", {}), dict) else {}
        payload = {
            "manifest_type": "llm_agent_artifact_manifest",
            "generated_at": datetime.now().isoformat(),
            "artifact_type": bundle.get("artifact_type"),
            "base_name": bundle.get("base_name"),
            "provider": bundle.get("provider"),
            "model": bundle.get("model"),
            "n_var": bundle.get("n_var"),
            "n_obj": bundle.get("n_obj"),
            "acceptance_mode": self._llm_acceptance_mode(bundle),
            "repair_attempts": bundle.get("_repair_attempts", 0),
            "spec_first": bool(bundle.get("_spec_first", False)),
            "prompt": bundle.get("_prompt", ""),
            "spec_report": bundle.get("_spec_report", {}),
            "repair_report": bundle.get("_repair_report", {}),
            "cpu_file": str(cpu_path) if cpu_path else bundle.get("cpu_file"),
            "jax_file": str(jax_path) if jax_path else bundle.get("jax_file"),
            "cpu_sha256": hashlib.sha256(str(bundle.get("cpu_code", "")).encode("utf-8")).hexdigest(),
            "jax_sha256": hashlib.sha256(str(bundle.get("jax_code", "")).encode("utf-8")).hexdigest(),
            "validation_report": report,
        }
        return payload

    def _llm_save_sidecar_files(self, bundle: dict[str, Any], *, cpu_path: Path, jax_path: Path) -> tuple[dict[str, Path], list[str]]:
        saved: dict[str, Path] = {}
        errors: list[str] = []
        use_timestamp = bool(getattr(self, "llm_validation_timestamp_check", None) and self.llm_validation_timestamp_check.isChecked())
        stamp = f"_{self._llm_now_timestamp_token()}" if use_timestamp else ""

        if getattr(self, "llm_autosave_validation_check", None) is not None and self.llm_autosave_validation_check.isChecked():
            try:
                validation_json_text = self._llm_validation_report_json_text(bundle)
                vpath = cpu_path.with_name(f"{cpu_path.stem}_validation{stamp}.json")
                vpath.write_text(validation_json_text, encoding="utf-8")
                saved["validation_json"] = vpath
            except Exception as exc:  # noqa: BLE001
                errors.append(f"validation_json: {exc}")

        if getattr(self, "llm_autosave_manifest_check", None) is not None and self.llm_autosave_manifest_check.isChecked():
            try:
                manifest_payload = self._llm_build_manifest_payload(bundle, cpu_path=cpu_path, jax_path=jax_path)
                mpath = cpu_path.with_name(f"{cpu_path.stem}_manifest{stamp}.json")
                mpath.write_text(
                    json.dumps(manifest_payload, indent=2, ensure_ascii=False, default=self._llm_json_default_serializer),
                    encoding="utf-8",
                )
                saved["manifest_json"] = mpath
            except Exception as exc:  # noqa: BLE001
                errors.append(f"manifest_json: {exc}")
        return saved, errors

    @staticmethod
    def _llm_categorize_exception_message(detail: str) -> str:
        text = str(detail or "").lower()
        if "benchmark survey/research request" in text or "classified as a benchmark survey" in text:
            return "request_scope"
        if "privacy" in text and "policy" in text:
            return "privacy_policy"
        if "rate-limit" in text or "rate limit" in text or "http 429" in text:
            return "rate_limit"
        if "api key" in text or "http 401" in text or "http 403" in text:
            return "auth"
        if "parity check failed" in text:
            return "parity_validation"
        if "oracle" in text and "failed" in text:
            return "oracle_validation"
        if "validation" in text and "failed" in text:
            return "validation"
        return "generation"

    @staticmethod
    def _llm_categorize_validation_issues(issues: list[str] | None) -> str:
        msgs = [str(x).lower() for x in (issues or [])]
        if not msgs:
            return "validation"
        if any("parity" in m for m in msgs):
            return "parity_validation"
        if any("oracle" in m for m in msgs):
            return "oracle_validation"
        if any("compile error" in m or "unsupported ast" in m or "unsupported import" in m or "unsupported call" in m for m in msgs):
            return "static_validation"
        if any("runtime" in m for m in msgs):
            return "runtime_validation"
        if any("shape" in m for m in msgs):
            return "shape_validation"
        return "validation"

    def _llm_push_history_bundle(self, bundle: dict[str, Any]) -> None:
        if not isinstance(bundle, dict):
            return
        history = getattr(self, "llm_generation_history", None)
        if not isinstance(history, list):
            self.llm_generation_history = []
            history = self.llm_generation_history
        snapshot = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "bundle": dict(bundle),
        }
        history.insert(0, snapshot)
        del history[12:]
        if getattr(self, "llm_history_list", None) is None:
            return
        self.llm_history_list.blockSignals(True)
        self.llm_history_list.clear()
        for idx, entry in enumerate(history):
            b = entry.get("bundle", {}) if isinstance(entry, dict) else {}
            if str(b.get("artifact_type", "")).strip().lower() == "benchmark_survey":
                source_count = 0
                report = b.get("survey_report", {}) if isinstance(b.get("survey_report", {}), dict) else {}
                try:
                    source_count = int(report.get("fetched_source_count", len(b.get("web_sources", []) or [])))
                except Exception:
                    source_count = 0
                label = (
                    f"{entry.get('timestamp', '')} | survey | "
                    f"{b.get('base_name', '?')} | sources={source_count}"
                )
                item = QListWidgetItem(label)
                item.setData(Qt.ItemDataRole.UserRole, idx)
                self.llm_history_list.addItem(item)
                continue
            label = (
                f"{entry.get('timestamp', '')} | "
                f"{str(b.get('artifact_type', '?')).lower()} | "
                f"{b.get('base_name', '?')} | "
                f"{self._llm_acceptance_mode_label(b)}"
            )
            item = QListWidgetItem(label)
            item.setData(Qt.ItemDataRole.UserRole, idx)
            self.llm_history_list.addItem(item)
        self.llm_history_list.blockSignals(False)

    @Slot()
    def _on_llm_history_selection_changed(self) -> None:
        if getattr(self, "llm_history_list", None) is None:
            return
        item = self.llm_history_list.currentItem()
        if item is None:
            return
        idx = item.data(Qt.ItemDataRole.UserRole)
        if not isinstance(idx, int):
            return
        history = getattr(self, "llm_generation_history", [])
        if not (0 <= idx < len(history)):
            return
        entry = history[idx]
        bundle = entry.get("bundle") if isinstance(entry, dict) else None
        if not isinstance(bundle, dict):
            return
        self.llm_generated_bundle = bundle
        self.llm_generated_bundle_validated = bool(((bundle.get("_validation_report") or {}).get("ok", False)))
        self.llm_code_edit.setPlainText(self._render_llm_bundle_preview(bundle))
        if getattr(self, "llm_raw_response_edit", None) is not None:
            self.llm_raw_response_edit.setPlainText(self._render_llm_api_raw_text(bundle))
        self._update_llm_validation_confidence_panel(bundle)
        if self._llm_is_survey_bundle(bundle):
            self.llm_last_survey_bundle = dict(bundle)
        self._refresh_llm_survey_bridge_controls()
        self.llm_status_label.setText(f"Loaded bundle snapshot from history: {entry.get('timestamp', 'unknown')}.")
        self.llm_status_label.setStyleSheet(f"color: {AppStyles.INFO};")

    @staticmethod
    def _llm_format_elapsed_hms(elapsed_s: int) -> str:
        total = max(0, int(elapsed_s))
        hours = total // 3600
        minutes = (total % 3600) // 60
        seconds = total % 60
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

    def _llm_set_elapsed_display(self, elapsed_s: int) -> None:
        if getattr(self, "llm_elapsed_label", None) is None:
            return
        self.llm_elapsed_label.setText(self._llm_format_elapsed_hms(elapsed_s))

    @Slot()
    def _on_llm_progress_loop_tick(self) -> None:
        if getattr(self, "llm_progress", None) is None:
            return
        value = int(getattr(self, "_llm_progress_loop_value", 0))
        direction = int(getattr(self, "_llm_progress_loop_direction", 1) or 1)
        value += 3 * direction
        if value >= 100:
            value = 100
            direction = -1
        elif value <= 0:
            value = 0
            direction = 1
        self._llm_progress_loop_value = value
        self._llm_progress_loop_direction = direction
        self.llm_progress.setValue(value)

    @Slot()
    def _on_llm_elapsed_tick(self) -> None:
        started_at = getattr(self, "_llm_busy_started_at", None)
        if started_at is None:
            return
        elapsed_s = int(max(0.0, time.perf_counter() - float(started_at)))
        self._llm_set_elapsed_display(elapsed_s)

    def _llm_task_running(self) -> bool:
        thread = getattr(self, "_llm_task_thread", None)
        return bool(thread is not None and thread.isRunning())

    def _llm_start_background_task(
        self,
        *,
        task_fn: Callable[..., Any],
        result_slot: Callable[[object], None],
        error_slot: Callable[[str], None],
        busy_status: str,
        partial_slot: Callable[[object], None] | None = None,
    ) -> bool:
        if self._llm_task_running():
            self.llm_status_label.setText("Llm Agent is already processing another API call.")
            self.llm_status_label.setStyleSheet(f"color: {AppStyles.WARNING};")
            return False

        thread = QThread(self)
        bridge = LLMTaskBridge(task_fn, use_partial_callback=(partial_slot is not None))
        bridge.moveToThread(thread)
        thread.started.connect(bridge.run)
        bridge.result_ready.connect(result_slot)
        if partial_slot is not None:
            bridge.partial_update.connect(partial_slot)
        bridge.error.connect(error_slot)
        bridge.done.connect(thread.quit)
        bridge.done.connect(bridge.deleteLater)
        thread.finished.connect(self._on_llm_background_task_finished)
        thread.finished.connect(thread.deleteLater)

        self._llm_task_thread = thread
        self._llm_task_bridge = bridge
        self._set_llm_busy(True, status=busy_status)
        thread.start()
        return True

    @Slot()
    def _on_llm_background_task_finished(self) -> None:
        self._set_llm_busy(False)
        self._llm_task_thread = None
        self._llm_task_bridge = None

    def _set_llm_busy(self, busy: bool, *, status: str | None = None) -> None:
        controls = [
            self.llm_problem_name_edit,
            self.llm_artifact_type_combo,
            self.llm_n_var_spin,
            self.llm_n_obj_spin,
            self.llm_prompt_edit,
            self.llm_spec_first_check,
            self.llm_generate_btn,
            self.llm_validate_btn,
            self.llm_copy_validation_btn,
            self.llm_save_btn,
            getattr(self, "llm_delete_btn", None),
            getattr(self, "llm_metric_mode_combo", None),
            self.llm_autosave_validation_check,
            getattr(self, "llm_survey_entry_combo", None),
            getattr(self, "llm_survey_use_prompt_btn", None),
            getattr(self, "llm_survey_generate_problem_btn", None),
        ]
        for widget in controls:
            if widget is None:
                continue
            widget.setEnabled(not busy)
        if busy:
            self._llm_busy_active = True
            self._llm_progress_loop_value = 0
            self._llm_progress_loop_direction = 1
            self.llm_progress.setRange(0, 100)
            self.llm_progress.setValue(0)
            self._llm_busy_started_at = time.perf_counter()
            self._llm_set_elapsed_display(0)
            if getattr(self, "llm_progress_loop_timer", None) is not None:
                self.llm_progress_loop_timer.start()
            if getattr(self, "llm_elapsed_timer", None) is not None:
                self.llm_elapsed_timer.start()
            self.llm_progress.setTextVisible(False)
        else:
            self._llm_busy_active = False
            if getattr(self, "llm_progress_loop_timer", None) is not None:
                self.llm_progress_loop_timer.stop()
            if getattr(self, "llm_elapsed_timer", None) is not None:
                self.llm_elapsed_timer.stop()
            self.llm_progress.setTextVisible(False)
        if status:
            self.llm_status_label.setText(status)
            self.llm_status_label.setStyleSheet(f"color: {AppStyles.INFO};")
        self._refresh_llm_survey_bridge_controls()
        self._sync_llm_mode_control_states()

    @Slot()
    def _on_llm_artifact_type_changed(self) -> None:
        token = self._llm_artifact_token()
        is_metric = token == "metric"
        is_survey = token == "benchmark_survey"
        is_problem = token == "problem"
        self.llm_hint_label.setText(self._llm_hint_text())
        self.llm_problem_name_edit.setPlaceholderText(
            "Example: DeltaPromptMetric"
            if is_metric
            else ("Example: GECCO2025_MO_BenchmarkSurvey" if is_survey else "Example: WeldedBeamPrompt")
        )
        self.llm_n_var_spin.setEnabled(not is_metric)
        self.llm_n_obj_spin.setEnabled(not is_metric)
        self.llm_prompt_edit.setPlaceholderText(
            (
                "Describe the metric/indicator semantics and expected front-based computation..."
                if is_metric
                else (
                    "Describe the benchmark survey scope (conference/year/topic) and required extracted fields (e.g., benchmark names and dimensions with sources)..."
                    if is_survey
                    else "Describe the optimization problem in natural language (objectives, constraints, ranges)..."
                )
            )
        )
        if is_problem:
            current_prompt = self.llm_prompt_edit.toPlainText().strip()
            if not current_prompt:
                self.llm_prompt_edit.setPlainText(self._llm_problem_prompt_placeholder_example())
            self._sync_llm_problem_scope_tab_from_n_obj()
            self._refresh_llm_problem_scope_summary()
        if is_metric:
            self._refresh_llm_metric_mode_summary()
        self.llm_generated_bundle = None
        self.llm_generated_bundle_validated = False
        if getattr(self, "llm_raw_response_edit", None) is not None:
            self.llm_raw_response_edit.setPlainText(self._render_llm_api_raw_text(None))
        if getattr(self, "llm_code_edit", None) is not None:
            self.llm_code_edit.setPlainText("")
        self._update_llm_validation_confidence_panel(None)
        self._refresh_llm_survey_bridge_controls()
        self._sync_llm_mode_control_states()

    @Slot()
    def _on_llm_copy_validation_report_clicked(self) -> None:
        bundle = getattr(self, "llm_generated_bundle", None)
        if not bundle:
            QMessageBox.warning(self, "Llm Agent Validation JSON", "No generated artifact bundle is available.")
            return
        if str(bundle.get("artifact_type", "")).strip().lower() == "benchmark_survey":
            QMessageBox.information(
                self,
                "Llm Agent Validation JSON",
                "Survey reports do not have plugin validation JSON. Use the raw output / preview panels for the benchmark survey report.",
            )
            self.llm_status_label.setText("Validation JSON copy not applicable for benchmark survey reports.")
            self.llm_status_label.setStyleSheet(f"color: {AppStyles.WARNING};")
            return
        try:
            text = self._llm_validation_report_json_text(bundle)
            QApplication.clipboard().setText(text)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.warning(self, "Llm Agent Validation JSON", f"Could not copy validation JSON:\n{exc}")
            self.llm_status_label.setText("Failed to copy validation JSON.")
            self.llm_status_label.setStyleSheet(f"color: {AppStyles.ERROR};")
            return
        self.llm_status_label.setText("Validation JSON copied to clipboard.")
        self.llm_status_label.setStyleSheet(f"color: {AppStyles.INFO};")
        QMessageBox.information(
            self,
            "Llm Agent Validation JSON",
            "Validation report JSON copied to clipboard.",
        )

    @Slot()
    def _on_llm_save_validation_report_clicked(self) -> None:
        bundle = getattr(self, "llm_generated_bundle", None)
        if not bundle:
            QMessageBox.warning(self, "Llm Agent Validation JSON", "No generated artifact bundle is available.")
            return
        if str(bundle.get("artifact_type", "")).strip().lower() == "benchmark_survey":
            QMessageBox.information(
                self,
                "Llm Agent Validation JSON",
                "Survey reports do not have plugin validation JSON. Validation JSON save is not applicable in Benchmark Survey mode.",
            )
            self.llm_status_label.setText("Validation JSON save not applicable for benchmark survey reports.")
            self.llm_status_label.setStyleSheet(f"color: {AppStyles.WARNING};")
            return
        try:
            text = self._llm_validation_report_json_text(bundle)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.warning(self, "Llm Agent Validation JSON", f"Could not build validation JSON:\n{exc}")
            self.llm_status_label.setText("Failed to build validation JSON.")
            self.llm_status_label.setStyleSheet(f"color: {AppStyles.ERROR};")
            return

        base_name = str(bundle.get("base_name", "artifact")).strip() or "artifact"
        suggested = f"{base_name}_validation.json"
        start_dir = str(self.base_dir)
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Validation JSON",
            str(Path(start_dir) / suggested),
            "JSON files (*.json);;All files (*)",
        )
        if not path:
            self.llm_status_label.setText("Validation JSON save canceled.")
            self.llm_status_label.setStyleSheet(f"color: {AppStyles.TEXT_MUTED};")
            return
        try:
            Path(path).write_text(text, encoding="utf-8")
        except Exception as exc:  # noqa: BLE001
            QMessageBox.warning(self, "Llm Agent Validation JSON", f"Could not save validation JSON:\n{exc}")
            self.llm_status_label.setText("Failed to save validation JSON.")
            self.llm_status_label.setStyleSheet(f"color: {AppStyles.ERROR};")
            return

        try:
            rel = Path(path).relative_to(self.base_dir)
        except Exception:
            rel = Path(path)
        self.llm_status_label.setText(f"Validation JSON saved to {rel}.")
        self.llm_status_label.setStyleSheet(f"color: {AppStyles.SUCCESS};")
        QMessageBox.information(self, "Llm Agent Validation JSON", f"Validation report saved to:\n{rel}")

    def _build_extensibility_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        hint = QLabel(
            "Create plugins in-app (problem/algorithm/metric/operator), validate quickly, save into local folders, and hot-reload registries."
        )
        hint.setWordWrap(True)
        hint.setStyleSheet(f"color: {AppStyles.TEXT_MUTED};")
        layout.addWidget(hint)

        form = QFormLayout()
        self.ext_type_combo = QComboBox()
        self.ext_type_combo.addItem("Problem", "problems")
        self.ext_type_combo.addItem("Algorithm", "algorithms")
        self.ext_type_combo.addItem("Metric/Indicator", "metrics")
        self.ext_type_combo.addItem("Operator", "operators")
        form.addRow("Plugin type:", self.ext_type_combo)

        self.ext_module_name_edit = QLineEdit()
        self.ext_module_name_edit.setStyleSheet(AppStyles.search_input_style())
        self.ext_module_name_edit.setPlaceholderText("example_plugin")
        form.addRow("Module file:", self.ext_module_name_edit)
        layout.addLayout(form)

        action_row = QHBoxLayout()
        self.ext_template_btn = QPushButton("Load Template")
        self.ext_template_btn.setStyleSheet(AppStyles.secondary_button_style())
        self.ext_template_btn.clicked.connect(self._on_ext_load_template)
        action_row.addWidget(self.ext_template_btn)

        self.ext_validate_btn = QPushButton("Validate")
        self.ext_validate_btn.setStyleSheet(AppStyles.secondary_button_style())
        self.ext_validate_btn.clicked.connect(self._on_ext_validate)
        action_row.addWidget(self.ext_validate_btn)

        self.ext_save_btn = QPushButton("Save + Reload")
        self.ext_save_btn.setStyleSheet(AppStyles.primary_button_style())
        self.ext_save_btn.clicked.connect(self._on_ext_save_reload)
        action_row.addWidget(self.ext_save_btn)
        action_row.addStretch(1)
        layout.addLayout(action_row)

        self.ext_code_edit = QPlainTextEdit()
        self.ext_code_edit.setPlaceholderText("Write plugin code here...")
        self.ext_code_edit.setStyleSheet(AppStyles.log_view_style())
        layout.addWidget(self.ext_code_edit, 1)

        self.ext_status_label = QLabel("Ready.")
        self.ext_status_label.setStyleSheet(f"color: {AppStyles.TEXT_MUTED};")
        layout.addWidget(self.ext_status_label)

        return tab

    @Slot()
    def _on_llm_generate_clicked(self) -> None:
        base_name = self._llm_base_name()
        artifact_type = self._llm_artifact_token()
        prompt = self.llm_prompt_edit.toPlainText().strip()
        provider = self._llm_provider_token()
        n_var = int(self.llm_n_var_spin.value())
        n_obj = int(self.llm_n_obj_spin.value())
        metric_generation_mode = self._llm_metric_generation_mode_token()
        spec_first = bool(getattr(self, "llm_spec_first_check", None) and self.llm_spec_first_check.isChecked())
        api_key = None
        self.llm_generated_bundle = None
        self.llm_generated_bundle_validated = False
        self._llm_reset_stream_preview()
        _ = self._llm_start_background_task(
            task_fn=lambda partial_emit: self._llm_generate_and_validate_bundle_task(
                prompt=prompt,
                artifact_type=artifact_type,
                base_name=base_name,
                n_var=n_var,
                n_obj=n_obj,
                provider=provider,
                api_key=api_key,
                spec_first=spec_first,
                metric_generation_mode=metric_generation_mode,
                partial_emit=partial_emit,
            ),
            result_slot=self._on_llm_generate_task_result,
            error_slot=self._on_llm_generate_task_error,
            busy_status="Processing request with local Qwen model (models/qwen2.5-0.5b-instruct-q4_k_m.gguf)...",
            partial_slot=self._on_llm_generate_partial_update,
        )

    @staticmethod
    def _llm_generate_and_validate_bundle_task(
        *,
        prompt: str,
        artifact_type: str,
        base_name: str,
        n_var: int,
        n_obj: int,
        provider: str,
        api_key: str,
        spec_first: bool,
        metric_generation_mode: str = "canonical_wrapper",
        partial_emit: Callable[[object], None] | None = None,
    ) -> dict[str, Any]:
        bundle = CoreLLMFormulationService.generate_artifact_bundle(
            prompt,
            artifact_type=artifact_type,
            base_name=base_name,
            n_var=int(n_var),
            n_obj=int(n_obj),
            provider=provider,
            api_key=api_key,
            spec_first=spec_first,
            metric_generation_mode=metric_generation_mode,
            stream_event_cb=partial_emit if callable(partial_emit) else None,
        )
        ok, issues = CoreLLMFormulationService.validate_artifact_bundle(bundle)
        issues_list = issues if isinstance(issues, list) else [str(issues)]
        return {
            "bundle": bundle,
            "ok": bool(ok),
            "issues": issues_list,
        }


    @Slot(object)
    def _on_llm_benchmark_survey_task_result(self, payload_obj: object) -> None:
        bundle = payload_obj if isinstance(payload_obj, dict) else None
        if not isinstance(bundle, dict):
            self._on_llm_benchmark_survey_task_error("invalid worker payload (missing survey report)")
            return
        self.llm_generated_bundle = bundle
        self.llm_generated_bundle_validated = False
        self.llm_last_survey_bundle = dict(bundle)
        self.llm_code_edit.setPlainText(self._render_llm_bundle_preview(bundle))
        if getattr(self, "llm_raw_response_edit", None) is not None:
            self.llm_raw_response_edit.setPlainText(self._render_llm_api_raw_text(bundle))
        self._update_llm_validation_confidence_panel(bundle)
        self._refresh_llm_survey_bridge_controls()
        self._llm_push_history_bundle(bundle)
        report = bundle.get("survey_report", {}) if isinstance(bundle.get("survey_report", {}), dict) else {}
        try:
            entries_count = int(len(report.get("benchmark_entries", []) or []))
        except Exception:
            entries_count = 0
        try:
            source_count = int(report.get("fetched_source_count", len(bundle.get("web_sources", []) or [])))
        except Exception:
            source_count = 0
        model_used = str(bundle.get("model", "")).strip()
        status = f"Benchmark survey report generated: {entries_count} benchmark entrie(s) from {source_count} source(s)."
        if model_used:
            status += f" Model: {model_used}."
        self.llm_status_label.setText(status)
        self.llm_status_label.setStyleSheet(f"color: {AppStyles.SUCCESS};")

    @Slot(str)
    def _on_llm_benchmark_survey_task_error(self, detail: str) -> None:
        QMessageBox.warning(self, "Benchmark Survey (Web)", f"Benchmark survey failed:\n{detail}")
        category = self._llm_categorize_exception_message(detail)
        if category == "auth":
            self.llm_status_label.setText("Benchmark survey failed: local LLM/auth issue.")
        elif category == "rate_limit":
            self.llm_status_label.setText("Benchmark survey failed: local LLM rate limit / temporary block.")
        else:
            self.llm_status_label.setText(f"Benchmark survey failed ({category}). Review search/extraction details and narrow the prompt.")
        self.llm_status_label.setStyleSheet(f"color: {AppStyles.ERROR};")

    @Slot(object)
    def _on_llm_generate_task_result(self, payload_obj: object) -> None:
        payload = payload_obj if isinstance(payload_obj, dict) else {}
        bundle = payload.get("bundle")
        ok = bool(payload.get("ok", False))
        issues = payload.get("issues", [])
        if not isinstance(bundle, dict):
            self._on_llm_generate_task_error("invalid worker payload (missing generated bundle)")
            return
        if not isinstance(issues, list):
            issues = [str(issues)]

        self._llm_reset_stream_preview()
        self.llm_generated_bundle = bundle
        self.llm_generated_bundle_validated = bool(ok)
        self.llm_code_edit.setPlainText(self._render_llm_bundle_preview(bundle))
        if getattr(self, "llm_raw_response_edit", None) is not None:
            self.llm_raw_response_edit.setPlainText(self._render_llm_api_raw_text(bundle))
        self._update_llm_validation_confidence_panel(bundle)
        self._refresh_llm_survey_bridge_controls()
        self._llm_push_history_bundle(bundle)
        model_used = str(bundle.get("model", "")).strip()
        if ok:
            status = f"Generated and validated via local Qwen ({CoreLLMFormulationService.DEFAULT_LOCAL_MODEL})."
            self.llm_status_label.setText(status)
            self.llm_status_label.setStyleSheet(f"color: {AppStyles.SUCCESS};")
            return
        val_category = self._llm_categorize_validation_issues(issues)
        self.llm_status_label.setText(
            f"Generated bundle, but {val_category} failed. Review details and regenerate."
        )
        self.llm_status_label.setStyleSheet(f"color: {AppStyles.WARNING};")
        QMessageBox.warning(
            self,
            "LLM Validation",
            self._llm_validation_dialog_summary(bundle),
        )

    @Slot(str)
    def _on_llm_generate_task_error(self, detail: str) -> None:
        self._llm_reset_stream_preview()
        QMessageBox.warning(self, "LLM Generate", f"Code generation failed:\n{detail}")
        category = self._llm_categorize_exception_message(detail)
        if category == "request_scope":
            self.llm_status_label.setText(
                "Generation blocked: benchmark survey/research request detected. Ask for one specific problem/benchmark at a time."
            )
            self.llm_status_label.setStyleSheet(f"color: {AppStyles.WARNING};")
            return
        self.llm_status_label.setText(f"Generation failed ({category}). Check local LLM status / logs.")
        self.llm_status_label.setStyleSheet(f"color: {AppStyles.ERROR};")

    @Slot()
    def _on_llm_validate_clicked(self) -> None:
        bundle = getattr(self, "llm_generated_bundle", None)
        if not bundle:
            QMessageBox.warning(self, "LLM Validation", "No generated artifact bundle is available to validate.")
            return
        if str(bundle.get("artifact_type", "")).strip().lower() == "benchmark_survey":
            QMessageBox.information(
                self,
                "LLM Validation",
                "Benchmark survey reports do not run CPU/JAX plugin validation. Validation is not applicable.",
            )
            self.llm_status_label.setText("Validation not applicable: benchmark survey report (no plugin code).")
            self.llm_status_label.setStyleSheet(f"color: {AppStyles.WARNING};")
            return
        ok, issues = CoreLLMFormulationService.validate_artifact_bundle(bundle)
        self.llm_generated_bundle_validated = bool(ok)
        self._update_llm_validation_confidence_panel(bundle)
        if ok:
            self.llm_status_label.setText("Validation successful for CPU and JAX modules.")
            self.llm_status_label.setStyleSheet(f"color: {AppStyles.SUCCESS};")
            QMessageBox.information(
                self,
                "LLM Validation",
                "Validation successful for CPU and JAX modules.\n\n" + self._llm_validation_dialog_summary(bundle),
            )
            return
        val_category = self._llm_categorize_validation_issues(issues)
        self.llm_status_label.setText(f"Validation failed ({val_category}). See details in dialog.")
        self.llm_status_label.setStyleSheet(f"color: {AppStyles.ERROR};")
        QMessageBox.warning(self, "LLM Validation", self._llm_validation_dialog_summary(bundle))

    @Slot()
    def _on_llm_save_clicked(self) -> None:
        bundle = getattr(self, "llm_generated_bundle", None)
        if not bundle:
            QMessageBox.warning(self, "LLM Save", "Generate an artifact bundle before saving.")
            return
        if str(bundle.get("artifact_type", "")).strip().lower() == "benchmark_survey":
            QMessageBox.information(
                self,
                "LLM Save",
                "Benchmark survey reports are not plugin modules and cannot be saved with Save + Reload.",
            )
            self.llm_status_label.setText("Save + Reload not applicable: benchmark survey report.")
            self.llm_status_label.setStyleSheet(f"color: {AppStyles.WARNING};")
            return
        ok, issues = CoreLLMFormulationService.validate_artifact_bundle(bundle)
        self.llm_generated_bundle_validated = bool(ok)
        self._update_llm_validation_confidence_panel(bundle)
        if not ok:
            val_category = self._llm_categorize_validation_issues(issues)
            QMessageBox.warning(
                self,
                "LLM Save",
                f"Generated bundle is invalid ({val_category}).\n\n" + self._llm_validation_dialog_summary(bundle),
            )
            self.llm_status_label.setText(f"Save blocked: {val_category} failed.")
            self.llm_status_label.setStyleSheet(f"color: {AppStyles.ERROR};")
            return
        try:
            cpu_path, jax_path = CoreLLMFormulationService.save_artifact_bundle(
                base_dir=self.base_dir,
                bundle=bundle,
            )
        except Exception as exc:  # noqa: BLE001
            category = self._llm_categorize_exception_message(str(exc))
            self.llm_status_label.setText(f"Save failed ({category}).")
            self.llm_status_label.setStyleSheet(f"color: {AppStyles.ERROR};")
            QMessageBox.warning(self, "LLM Save", f"Could not save generated bundle ({category}):\n{exc}")
            return
        sidecars_saved, sidecar_errors = self._llm_save_sidecar_files(bundle, cpu_path=cpu_path, jax_path=jax_path)
        tracked_saved_paths: dict[str, str] = {
            "cpu": str(cpu_path),
            "jax": str(jax_path),
        }
        if "validation_json" in sidecars_saved:
            tracked_saved_paths["validation_json"] = str(sidecars_saved["validation_json"])
        if "manifest_json" in sidecars_saved:
            tracked_saved_paths["manifest_json"] = str(sidecars_saved["manifest_json"])
        bundle["_saved_with_llm_agent"] = True
        bundle["_saved_at"] = datetime.now().isoformat()
        bundle["_saved_artifact_paths"] = tracked_saved_paths
        self._reload_registries()
        try:
            cpu_rel = cpu_path.relative_to(self.base_dir)
        except Exception:
            cpu_rel = cpu_path
        try:
            jax_rel = jax_path.relative_to(self.base_dir)
        except Exception:
            jax_rel = jax_path
        validation_json_rel = None
        manifest_json_rel = None
        if "validation_json" in sidecars_saved:
            try:
                validation_json_rel = sidecars_saved["validation_json"].relative_to(self.base_dir)
            except Exception:
                validation_json_rel = sidecars_saved["validation_json"]
        if "manifest_json" in sidecars_saved:
            try:
                manifest_json_rel = sidecars_saved["manifest_json"].relative_to(self.base_dir)
            except Exception:
                manifest_json_rel = sidecars_saved["manifest_json"]
        if sidecar_errors:
            self.llm_status_label.setText(
                f"Saved CPU/JAX modules with sidecar warnings: {'; '.join(sidecar_errors[:2])}"
            )
            self.llm_status_label.setStyleSheet(f"color: {AppStyles.WARNING};")
        elif validation_json_rel is not None or manifest_json_rel is not None:
            self.llm_status_label.setText(
                f"Saved CPU/JAX modules with sidecars; registries reloaded."
            )
            self.llm_status_label.setStyleSheet(f"color: {AppStyles.SUCCESS};")
        else:
            self.llm_status_label.setText(f"Saved CPU/JAX modules to {cpu_rel} and {jax_rel}; registries reloaded.")
            self.llm_status_label.setStyleSheet(f"color: {AppStyles.SUCCESS};")
        validation_json_line = f"\nValidation JSON: {validation_json_rel}\n" if validation_json_rel is not None else ""
        manifest_json_line = f"Manifest JSON: {manifest_json_rel}\n" if manifest_json_rel is not None else ""
        sidecar_warn_line = f"\nSidecar warnings: {'; '.join(sidecar_errors)}\n" if sidecar_errors else ""
        QMessageBox.information(
            self,
            "Llm Agent Save",
            f"CPU/JAX modules saved successfully and registries reloaded.\n\n"
            f"CPU: {cpu_rel}\nJAX: {jax_rel}{validation_json_line}\n"
            f"{manifest_json_line}"
            f"{sidecar_warn_line}"
            f"{self._llm_validation_dialog_summary(bundle)}",
        )
        self._llm_push_history_bundle(bundle)

    @staticmethod
    def _ext_template_for_type(kind: str) -> str:
        k = str(kind).strip().lower()
        if k == "problems":
            return (
                "import numpy as np\n"
                "from core.problem import Problem\n\n"
                "class MyProblem(Problem):\n"
                "    def __init__(self):\n"
                "        super().__init__(n_var=10, n_obj=2, xl=0.0, xu=1.0)\n\n"
                "    def _evaluate(self, X, out, *args, **kwargs):\n"
                "        f1 = X[:, 0]\n"
                "        g = 1.0 + 9.0 * np.mean(X[:, 1:], axis=1)\n"
                "        f2 = g * (1.0 - np.sqrt(f1 / g))\n"
                "        out['F'] = np.column_stack([f1, f2])\n"
            )
        elif k == "algorithms":
            return (
                "from algorithms.moo.nsga2 import NSGA2\n\n"
                "def create_algorithm(config):\n"
                "    pop_size = int(config.get('pop_size', 100))\n"
                "    return NSGA2(pop_size=pop_size)\n"
            )
        elif k == "operators":
            return (
                "from core.mutation import Mutation\n"
                "import numpy as np\n\n"
                "class MyMutation(Mutation):\n"
                "    def _do(self, problem, X, **kwargs):\n"
                "        rng = np.random.default_rng()\n"
                "        scale = 0.05 * (problem.xu - problem.xl)\n"
                "        noise = rng.normal(0, scale, size=X.shape)\n"
                "        return np.clip(X + noise, problem.xl, problem.xu)\n"
            )
        return ""
    def _on_ext_load_template(self) -> None:
        kind = str(self.ext_type_combo.currentData() or "metrics")
        self.ext_code_edit.setPlainText(self._ext_template_for_type(kind))
        if not self.ext_module_name_edit.text().strip():
            self.ext_module_name_edit.setText(f"my_{kind[:-1] if kind.endswith('s') else kind}")
        self.ext_status_label.setText("Template loaded.")
        self.ext_status_label.setStyleSheet(f"color: {AppStyles.INFO};")

    @Slot()
    def _on_ext_validate(self) -> None:
        code = self.ext_code_edit.toPlainText().strip()
        if not code:
            QMessageBox.warning(self, "Extensibility", "Code editor is empty.")
            return
        try:
            ast.parse(code)
            compile(code, "<ext_plugin>", "exec")
        except Exception as exc:  # noqa: BLE001
            self.ext_status_label.setText("Validation failed.")
            self.ext_status_label.setStyleSheet(f"color: {AppStyles.ERROR};")
            QMessageBox.warning(self, "Extensibility", str(exc))
            return
        self.ext_status_label.setText("Validation successful.")
        self.ext_status_label.setStyleSheet(f"color: {AppStyles.SUCCESS};")

    @Slot()
    def _on_ext_save_reload(self) -> None:
        kind = str(self.ext_type_combo.currentData() or "metrics")
        module_name = self.ext_module_name_edit.text().strip()
        code = self.ext_code_edit.toPlainText().strip()
        if not module_name:
            QMessageBox.warning(self, "Extensibility", "Provide a module file name.")
            return
        if not code:
            QMessageBox.warning(self, "Extensibility", "Code editor is empty.")
            return
        try:
            ast.parse(code)
            compile(code, "<ext_plugin>", "exec")
        except Exception as exc:  # noqa: BLE001
            QMessageBox.warning(self, "Extensibility", f"Invalid code: {exc}")
            return

        safe_name = re.sub(r"[^a-zA-Z0-9_]+", "_", module_name).strip("_") or "plugin"
        target_dir = self.base_dir / kind
        target_dir.mkdir(parents=True, exist_ok=True)
        target_file = target_dir / f"{safe_name}.py"
        target_file.write_text(code + "\n", encoding="utf-8")
        self._reload_registries()
        self.ext_status_label.setText(f"Saved to {kind}/{safe_name}.py and registries reloaded.")
        self.ext_status_label.setStyleSheet(f"color: {AppStyles.SUCCESS};")

    def _build_config_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        subtitle = QLabel(
            "Run a focused first test: choose one algorithm, one problem, and review the essential parameters before execution."
        )
        subtitle.setWordWrap(True)
        subtitle.setStyleSheet(AppStyles.section_subtitle_style())
        layout.addWidget(subtitle)

        split = QSplitter(Qt.Orientation.Horizontal)
        layout.addWidget(split, 1)

        left_panel = QWidget()
        self.exp_left_panel = left_panel
        left_layout = QVBoxLayout(left_panel)
        left_layout.setSpacing(8)

        selection_box = QGroupBox("1. Choose a Scenario")
        selection_layout = QVBoxLayout(selection_box)
        selection_layout.setSpacing(6)

        # Objectives and Variable Type filters removed from the UI (not useful
        # for this project's workflow). Kept as empty maps so catalog filtering
        # falls through to "no filter" without changing _apply_catalog_filters.
        self.objective_filter_buttons: dict[str, QPushButton] = {}
        self.encoding_filter_buttons: dict[str, QPushButton] = {}

        # Special difficulties filter removed from the UI (currently unused).
        self.difficulty_filter_buttons: dict[str, QPushButton] = {}

        algo_header = QHBoxLayout()
        algo_title = QLabel("Algorithms")
        algo_title.setStyleSheet(f"color: {AppStyles.ACCENT_BLUE}; font-weight: 700;")
        self.algorithm_filtered_count_label = QLabel("0 / 0")
        self.algorithm_filtered_count_label.setStyleSheet(AppStyles.count_badge_style())
        algo_header.addWidget(algo_title)
        algo_header.addStretch(1)
        algo_header.addWidget(self.algorithm_filtered_count_label)
        selection_layout.addLayout(algo_header)

        self.algorithm_filter = QLineEdit()
        self.algorithm_filter.setStyleSheet(AppStyles.search_input_style())
        self.algorithm_filter.setPlaceholderText("Search algorithms...")
        self.algorithm_filter.setAccessibleName("Quick test algorithm filter")
        self.algorithm_filter.setAccessibleDescription("Filter the list before selecting an algorithm.")
        self.algorithm_filter.textChanged.connect(self._apply_catalog_filters)
        selection_layout.addWidget(self.algorithm_filter)

        self.algorithm_list = QListWidget()
        self.algorithm_list.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.algorithm_list.setMidLineWidth(80)
        self.algorithm_list.setStyleSheet(AppStyles.algorithm_list_style())
        self.algorithm_list.currentItemChanged.connect(self._on_algorithm_selected)
        selection_layout.addWidget(self.algorithm_list)

        problem_header = QHBoxLayout()
        problem_title = QLabel("Problems")
        problem_title.setStyleSheet(f"color: {AppStyles.ACCENT_PROBLEM}; font-weight: 700;")
        self.problem_filtered_count_label = QLabel("0 / 0")
        self.problem_filtered_count_label.setStyleSheet(AppStyles.count_badge_style())
        problem_header.addWidget(problem_title)
        problem_header.addStretch(1)
        problem_header.addWidget(self.problem_filtered_count_label)
        selection_layout.addLayout(problem_header)

        self.problem_filter = QLineEdit()
        self.problem_filter.setStyleSheet(AppStyles.search_input_style())
        self.problem_filter.setPlaceholderText("Search problems...")
        self.problem_filter.setAccessibleName("Quick test problem filter")
        self.problem_filter.setAccessibleDescription("Filter the list before selecting a problem.")
        self.problem_filter.textChanged.connect(self._apply_catalog_filters)
        selection_layout.addWidget(self.problem_filter)

        self.problem_catalog_list = QListWidget()
        self.problem_catalog_list.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.problem_catalog_list.setMidLineWidth(80)
        self.problem_catalog_list.setStyleSheet(AppStyles.problem_list_style())
        self.problem_catalog_list.currentItemChanged.connect(self._on_problem_catalog_selected)
        selection_layout.addWidget(self.problem_catalog_list)

        left_layout.addWidget(selection_box, 1)

        # Wrap left panel in scroll area for small screens
        left_panel.setMinimumWidth(320)
        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setWidget(left_panel)
        left_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        left_scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        left_scroll.setMinimumWidth(320)
        left_scroll.setMaximumWidth(420)
        split.addWidget(left_scroll)

        param_panel = QWidget()
        param_layout = QVBoxLayout(param_panel)
        param_layout.setSpacing(8)

        param_box = QGroupBox("2. Review Parameters")
        param_box_layout = QVBoxLayout(param_box)
        param_box_layout.setSpacing(8)

        self.selected_algorithm_label = QLabel("No algorithm selected")
        self.selected_algorithm_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.selected_algorithm_label.setStyleSheet(AppStyles.selection_card_style(AppStyles.SELECTION_BLUE))
        param_box_layout.addWidget(self.selected_algorithm_label)

        self.selected_problem_label = QLabel("No problem selected")
        self.selected_problem_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.selected_problem_label.setStyleSheet(AppStyles.selection_card_style(AppStyles.SELECTION_PROBLEM))
        param_box_layout.addWidget(self.selected_problem_label)

        self.problem_combo = QComboBox()
        self.problem_combo.currentIndexChanged.connect(self._on_problem_changed)
        self.problem_combo.setVisible(False)
        param_box_layout.addWidget(self.problem_combo)

        basic_form = QFormLayout()

        self.pop_size_spin = QSpinBox()
        self.pop_size_spin.setRange(5, 5000)
        self.pop_size_spin.setValue(100)
        basic_form.addRow("Population (N):", self.pop_size_spin)

        self.n_obj_spin = QSpinBox()
        self.n_obj_spin.setRange(1, 20)
        self.n_obj_spin.valueChanged.connect(self._on_test_n_obj_changed)
        basic_form.addRow("Objectives (M):", self.n_obj_spin)

        self.n_var_spin = QSpinBox()
        self.n_var_spin.setRange(1, 5000)
        basic_form.addRow("Decision Variables (D):", self.n_var_spin)

        self.max_fe_spin = QSpinBox()
        self.max_fe_spin.setRange(100, 100_000_000)
        self.max_fe_spin.setSingleStep(1000)
        self.max_fe_spin.setValue(DEFAULT_MAX_FE)
        # Skill: Icon for maxFE (n_eval) termination criterion
        max_fe_label = QLabel("MaxFE:")
        max_fe_label.setStyleSheet(f"color: {AppStyles.TEXT}; font-weight: 600;")
        max_fe_label.setToolTip("Maximum number of function evaluations (stopping criterion)")
        self.max_fe_spin.setToolTip("Maximum number of function evaluations (n_eval)")
        basic_form.addRow(max_fe_label, self.max_fe_spin)

        # Algorithm operators - using EmoPyLab defaults when set to "default"
        # --- Crossover operator + parameters ---
        self.crossover_combo = QComboBox()
        self.crossover_combo.addItem("Default (EmoPyLab)", "default")
        self.crossover_combo.addItem("SBX (Simulated Binary)", "sbx")
        self.crossover_combo.addItem("PMX (Partial Match)", "pmx")
        self.crossover_combo.addItem("OX (Order)", "ox")
        self.crossover_combo.addItem("UX (Uniform)", "ux")
        self.crossover_combo.addItem("DEX (Discrete Exchange)", "dex")
        self.crossover_combo.addItem("None (no crossover)", "none")
        self.crossover_combo.setCurrentIndex(0)
        basic_form.addRow("Crossover:", self.crossover_combo)

        # Crossover parameter widgets (visible only when relevant operator is selected)
        self.crossover_eta_spin = QDoubleSpinBox()
        self.crossover_eta_spin.setRange(1.0, 100.0)
        self.crossover_eta_spin.setSingleStep(1.0)
        self.crossover_eta_spin.setDecimals(1)
        self.crossover_eta_spin.setValue(15.0)
        self.crossover_eta_spin.setToolTip("Distribution index (eta) for SBX. Higher = more similar to parents")
        self.crossover_eta_label = QLabel("  eta:")
        self.crossover_eta_label.setStyleSheet(f"color: {AppStyles.TEXT_MUTED}; font-size: 11px;")
        basic_form.addRow(self.crossover_eta_label, self.crossover_eta_spin)

        self.crossover_prob_spin = QDoubleSpinBox()
        self.crossover_prob_spin.setRange(0.0, 1.0)
        self.crossover_prob_spin.setSingleStep(0.05)
        self.crossover_prob_spin.setDecimals(2)
        self.crossover_prob_spin.setValue(0.9)
        self.crossover_prob_spin.setToolTip("Crossover probability (0.0-1.0)")
        self.crossover_prob_label = QLabel("  Prob:")
        self.crossover_prob_label.setStyleSheet(f"color: {AppStyles.TEXT_MUTED}; font-size: 11px;")
        basic_form.addRow(self.crossover_prob_label, self.crossover_prob_spin)

        self.crossover_combo.currentIndexChanged.connect(self._on_crossover_changed)

        # --- Mutation operator + parameters ---
        self.mutation_combo = QComboBox()
        self.mutation_combo.addItem("Default (EmoPyLab)", "default")
        self.mutation_combo.addItem("PM (Polynomial)", "pm")
        self.mutation_combo.addItem("SBM (Single Bit)", "sbm")
        self.mutation_combo.addItem("Bitflip", "bitflip")
        self.mutation_combo.addItem("None (no mutation)", "none")
        self.mutation_combo.setCurrentIndex(0)
        basic_form.addRow("Mutation:", self.mutation_combo)

        # Mutation parameter widgets
        self.mutation_eta_spin = QDoubleSpinBox()
        self.mutation_eta_spin.setRange(1.0, 100.0)
        self.mutation_eta_spin.setSingleStep(1.0)
        self.mutation_eta_spin.setDecimals(1)
        self.mutation_eta_spin.setValue(20.0)
        self.mutation_eta_spin.setToolTip("Distribution index (eta) for PM. Higher = smaller mutations")
        self.mutation_eta_label = QLabel("  eta:")
        self.mutation_eta_label.setStyleSheet(f"color: {AppStyles.TEXT_MUTED}; font-size: 11px;")
        basic_form.addRow(self.mutation_eta_label, self.mutation_eta_spin)

        self.mutation_prob_spin = QDoubleSpinBox()
        self.mutation_prob_spin.setRange(0.0, 1.0)
        self.mutation_prob_spin.setSingleStep(0.01)
        self.mutation_prob_spin.setDecimals(3)
        self.mutation_prob_spin.setValue(0.0)
        self.mutation_prob_spin.setToolTip("Mutation probability (0 = auto: 1/D)")
        self.mutation_prob_label = QLabel("  Prob (0=auto):")
        self.mutation_prob_label.setStyleSheet(f"color: {AppStyles.TEXT_MUTED}; font-size: 11px;")
        basic_form.addRow(self.mutation_prob_label, self.mutation_prob_spin)

        self.mutation_combo.currentIndexChanged.connect(self._on_mutation_changed)

        # --- Selection operator + parameters ---
        self.selection_combo = QComboBox()
        self.selection_combo.addItem("Default (EmoPyLab)", "default")
        self.selection_combo.addItem("Tournament", "tournament")
        self.selection_combo.addItem("Random", "random")
        self.selection_combo.addItem("Best", "best")
        self.selection_combo.addItem("Random Binary", "random_binary")
        self.selection_combo.setCurrentIndex(0)
        basic_form.addRow("Selection:", self.selection_combo)

        self.sampling_combo = QComboBox()
        self.sampling_combo.addItem("Default (EmoPyLab)", "default")
        basic_form.addRow("Sampling:", self.sampling_combo)

        # Selection parameter widgets
        self.selection_pressure_spin = QSpinBox()
        self.selection_pressure_spin.setRange(1, 20)
        self.selection_pressure_spin.setValue(2)
        self.selection_pressure_spin.setToolTip("Tournament size / selection pressure (1-20)")
        self.selection_pressure_label = QLabel("  Pressure:")
        self.selection_pressure_label.setStyleSheet(f"color: {AppStyles.TEXT_MUTED}; font-size: 11px;")
        basic_form.addRow(self.selection_pressure_label, self.selection_pressure_spin)

        self.selection_combo.currentIndexChanged.connect(self._on_selection_changed)
        self._populate_all_operator_combos()

        # Initial visibility - hide all operator params (default selected)
        self._on_crossover_changed()
        self._on_mutation_changed()
        self._on_selection_changed()

        # UI/UX simplification: hide operator controls and always use algorithm defaults.
        for widget in (
            self.crossover_combo,
            self.crossover_eta_label,
            self.crossover_eta_spin,
            self.crossover_prob_label,
            self.crossover_prob_spin,
            self.mutation_combo,
            self.mutation_eta_label,
            self.mutation_eta_spin,
            self.mutation_prob_label,
            self.mutation_prob_spin,
            self.selection_combo,
            self.sampling_combo,
            self.selection_pressure_label,
            self.selection_pressure_spin,
        ):
            widget.setVisible(False)
        for field in (
            self.crossover_combo,
            self.crossover_eta_spin,
            self.crossover_prob_spin,
            self.mutation_combo,
            self.mutation_eta_spin,
            self.mutation_prob_spin,
            self.selection_combo,
            self.sampling_combo,
            self.selection_pressure_spin,
        ):
            row_label = basic_form.labelForField(field)
            if row_label is not None:
                row_label.setVisible(False)

        self.stop_criterion_label = QLabel(
            "Stops when MaxFE is reached."
        )
        self.stop_criterion_label.setStyleSheet(f"color: {AppStyles.TEXT_MUTED};")
        self.stop_criterion_label.setWordWrap(True)
        basic_form.addRow("Termination:", self.stop_criterion_label)

        self.n_runs_spin = QSpinBox()
        self.n_runs_spin.setRange(1, 1)
        self.n_runs_spin.setValue(1)
        self.n_runs_spin.setVisible(False)

        self.seed_mode_combo = QComboBox()
        self.seed_mode_combo.addItem("Random", SEED_MODE_RANDOM)
        self.seed_mode_combo.addItem("Fixed", SEED_MODE_FIXED)
        basic_form.addRow("Seed:", self.seed_mode_combo)

        self.seed_spin = QSpinBox()
        self.seed_spin.setRange(1, 2_147_483_647)
        self.seed_spin.setValue(1)
        basic_form.addRow("Seed Value:", self.seed_spin)

        self.seed_mode_hint = QLabel("Random mode generates a new seed for each execution.")
        self.seed_mode_hint.setStyleSheet(f"color: {AppStyles.TEXT_MUTED};")
        self.seed_mode_hint.setWordWrap(True)
        basic_form.addRow("", self.seed_mode_hint)
        self.seed_mode_combo.currentIndexChanged.connect(self._on_test_seed_mode_changed)
        self._on_test_seed_mode_changed()

        param_box_layout.addLayout(basic_form)

        backend_form = QFormLayout()

        default_workers, max_workers = resolve_parallel_worker_limits()

        # Test Module always uses one worker (one algorithm, one problem, one run).
        self.parallel_workers_spin = QSpinBox()
        self.parallel_workers_spin.setRange(1, 1)
        self.parallel_workers_spin.setValue(1)
        self.parallel_workers_spin.setVisible(False)

        self.compute_backend_combo = QComboBox()
        # MLX is offered and selected by default only when actually detected
        # (Apple Silicon + a working mlx runtime); otherwise the default is CPU.
        mlx_detected = bool(self.mlx_runtime.get("mlx_ok"))
        for key, label in BACKEND_OPTIONS.items():
            if key == "mlx" and not mlx_detected:
                continue
            self.compute_backend_combo.addItem(label, key)
        if mlx_detected and self.compute_backend_combo.findData("mlx") != -1:
            self.compute_backend_combo.setCurrentIndex(self.compute_backend_combo.findData("mlx"))
        else:
            self.compute_backend_combo.setCurrentIndex(self.compute_backend_combo.findData("cpu"))
        self.compute_backend_combo.currentIndexChanged.connect(self._on_backend_mode_changed)
        backend_form.addRow("Compute Backend:", self.compute_backend_combo)

        # UI/UX simplification: keep only one GPU dtype internally (float32/speed)
        # and hide status/profiling controls from the Test Module UI.
        self.gpu_dtype_combo = QComboBox()
        self.gpu_dtype_combo.addItem("float64 (speed)", "float64")
        self.gpu_dtype_combo.setCurrentIndex(0)
        self.gpu_dtype_combo.setVisible(False)

        self.gpu_status_label = QLabel(build_gpu_status_text(self.gpu_runtime))
        self.gpu_status_label.setWordWrap(True)
        self.gpu_status_label.setStyleSheet(f"color: {AppStyles.TEXT_MUTED};")
        self.gpu_status_label.setVisible(False)

        self.profile_compare_check = QCheckBox("Enable CPU vs JAX(GPU) profiling (same seed)")
        self.profile_compare_check.setChecked(False)
        self.profile_compare_check.setVisible(False)

        self.profile_status_label = QLabel("")
        self.profile_status_label.setStyleSheet(f"color: {AppStyles.TEXT_MUTED};")
        self.profile_status_label.setWordWrap(True)
        self.profile_status_label.setVisible(False)
        param_box_layout.addLayout(backend_form)

        self.gpu_scope_label = QLabel(
            "JAX mode uses the native pymoo+JAX pattern. "
            "Only problems with JAX-compatible _eval_F implementations can run in this mode."
        )
        self.gpu_scope_label.setStyleSheet(f"color: {AppStyles.TEXT_MUTED};")
        self.gpu_scope_label.setWordWrap(True)
        param_box_layout.addWidget(self.gpu_scope_label)

        ref_point_header = QHBoxLayout()
        ref_point_title = QLabel("HV Reference Point:")
        ref_point_title.setStyleSheet("font-weight: 600;")
        ref_point_header.addWidget(ref_point_title)
        ref_point_header.addStretch()
        self.ref_point_label = QLabel("Computed automatically during execution")
        self.ref_point_label.setStyleSheet(f"color: {AppStyles.TEXT_MUTED}; font-style: italic;")
        self.ref_point_label.setWordWrap(True)
        ref_point_header.addWidget(self.ref_point_label)
        param_box_layout.addLayout(ref_point_header)

        self.use_pf_check = QCheckBox("Use the true Pareto front when available")
        self.use_pf_check.setChecked(True)
        param_box_layout.addWidget(self.use_pf_check)

        self.experiment_problem_list = QListWidget()
        self.experiment_problem_list.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        self.experiment_problem_list.setMinimumHeight(110)
        self.experiment_problem_label = QLabel("Experiment problem set:")
        param_box_layout.addWidget(self.experiment_problem_label)
        param_box_layout.addWidget(self.experiment_problem_list)

        self.problem_scope_hint = QLabel("Checked items above are used by the Experiments tab.")
        self.problem_scope_hint.setStyleSheet(f"color: {AppStyles.TEXT_MUTED};")
        self.problem_scope_hint.setWordWrap(True)
        param_box_layout.addWidget(self.problem_scope_hint)

        self.problem_count_label = QLabel("")
        self.problem_count_label.setStyleSheet(f"color: {AppStyles.TEXT_MUTED};")
        self.problem_count_label.setWordWrap(True)
        param_box_layout.addWidget(self.problem_count_label)

        # Test Module operates with one algorithm and one problem only.
        self.experiment_problem_label.setVisible(False)
        self.experiment_problem_list.setVisible(False)
        self.problem_scope_hint.setVisible(False)

        self.quick_start_summary_label = QLabel("")
        self.quick_start_summary_label.setWordWrap(True)
        self.quick_start_summary_label.setStyleSheet(AppStyles.guided_panel_style())
        param_layout.addWidget(param_box, 1)
        param_layout.addWidget(self.quick_start_summary_label)

        # Wrap parameter panel in scroll area for small screens
        param_panel.setMinimumWidth(300)
        param_scroll = QScrollArea()
        param_scroll.setWidgetResizable(True)
        param_scroll.setWidget(param_panel)
        param_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        param_scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        param_scroll.setFixedWidth(320)
        split.addWidget(param_scroll)

        result_panel = QWidget()
        result_layout = QVBoxLayout(result_panel)
        result_layout.setSpacing(8)

        result_box = QGroupBox("3. Run and Inspect")
        result_box_layout = QVBoxLayout(result_box)
        result_box_layout.setSpacing(8)

        result_row = QHBoxLayout()
        self.test_mode_combo = QComboBox()
        self.test_mode_combo.addItem("Population (Objectives)", "population")
        self.test_mode_combo.addItem("Convergence (Metric)", "convergence")
        self.test_mode_combo.currentIndexChanged.connect(self._refresh_test_result_chart)
        result_row.addWidget(self.test_mode_combo)

        self.test_metric_combo = QComboBox()
        self.test_metric_combo.currentTextChanged.connect(self._refresh_test_result_chart)
        result_row.addWidget(self.test_metric_combo)

        self.test_problem_combo = QComboBox()
        self.test_problem_combo.currentIndexChanged.connect(self._on_test_problem_changed)
        result_row.addWidget(self.test_problem_combo)

        self.test_algo_combo = QComboBox()
        self.test_algo_combo.currentTextChanged.connect(self._on_test_algo_changed)
        result_row.addWidget(self.test_algo_combo)

        self.test_run_spin = QSpinBox()
        self.test_run_spin.setRange(1, 1)
        self.test_run_spin.valueChanged.connect(self._refresh_test_result_chart)
        self.test_run_spin.setVisible(False)

        self.test_anchor_origin = QCheckBox("Anchor to Origin")
        self.test_anchor_origin.setChecked(True)
        self.test_anchor_origin.toggled.connect(self._refresh_test_result_chart)
        result_row.addWidget(self.test_anchor_origin)

        result_box_layout.addLayout(result_row)

        self.test_chart_view = QChartView()
        self.test_chart_view.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.test_chart_view.setRubberBand(QChartView.RubberBand.RectangleRubberBand)
        self.test_chart_view.setAccessibleName("Quick test visual analysis chart")
        self.test_chart_view.setToolTip("Drag to zoom. Double-click to reset the chart view.")
        self.test_chart_view.setMinimumHeight(180)

        self.test_3d_container = QWidget()
        self.test_3d_layout = QVBoxLayout(self.test_3d_container)
        self.test_3d_layout.setContentsMargins(0, 0, 0, 0)
        self.test_3d_container.setMinimumHeight(180)

        self.test_chart_stack = QStackedWidget()
        self.test_chart_stack.addWidget(self.test_chart_view)       # page 0
        self.test_chart_stack.addWidget(self.test_3d_container)     # page 1
        result_box_layout.addWidget(self.test_chart_stack, 1)

        self.test_result_summary = QPlainTextEdit()
        self.test_result_summary.setReadOnly(True)
        self.test_result_summary.setPlaceholderText("The test summary will appear here after execution.")
        self.test_result_summary.setMaximumHeight(65)
        self.test_result_summary.setStyleSheet(AppStyles.log_view_style())
        result_box_layout.addWidget(self.test_result_summary)

        result_layout.addWidget(result_box, 1)

        row_actions = QHBoxLayout()
        self.btn_start = QPushButton("Run Test")
        self.btn_start.setAccessibleName("Run Test")
        self.btn_start.setAccessibleDescription("Run one test with the selected algorithm, problem, and parameters.")
        self.btn_start.setIcon(app_icon("play_arrow")) # Skill: Icon e.g. play_arrow
        self.btn_start.setStyleSheet(AppStyles.primary_button_style())
        self.btn_start.clicked.connect(self._start_test)
        row_actions.addWidget(self.btn_start)

        self.btn_stop = QPushButton("Stop")
        self.btn_stop.setAccessibleName("Stop Test")
        self.btn_stop.setAccessibleDescription("Request cancellation of the running test.")
        self.btn_stop.setIcon(app_icon("stop"))
        self.btn_stop.setEnabled(False)
        self.btn_stop.setStyleSheet(AppStyles.secondary_button_style())
        self.btn_stop.clicked.connect(self._stop_test)
        row_actions.addWidget(self.btn_stop)

        self.btn_save_cfg = QPushButton("Save")
        self.btn_save_cfg.setIcon(app_icon("save"))
        self.btn_save_cfg.setStyleSheet(AppStyles.secondary_button_style())
        self.btn_save_cfg.clicked.connect(self._save_config_to_file)
        row_actions.addWidget(self.btn_save_cfg)

        self.btn_load_cfg = QPushButton("Load")
        self.btn_load_cfg.setIcon(app_icon("folder_open"))
        self.btn_load_cfg.setStyleSheet(AppStyles.secondary_button_style())
        self.btn_load_cfg.clicked.connect(self._load_config_from_file)
        row_actions.addWidget(self.btn_load_cfg)

        result_layout.addLayout(row_actions)

        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.setFixedHeight(14)
        result_layout.addWidget(self.progress)

        split.addWidget(result_panel)

        side_panel = QWidget()
        side_layout = QVBoxLayout(side_panel)
        side_layout.setSpacing(8)

        metric_box = QGroupBox()
        metric_box.setTitle("Metrics")
        metric_layout = QVBoxLayout(metric_box)
        self.metric_filter = QLineEdit()
        self.metric_filter.setStyleSheet(AppStyles.search_input_style())
        self.metric_filter.setPlaceholderText("Search metrics...")
        self.metric_filter.setAccessibleName("Quick test metric filter")
        self.metric_filter.setAccessibleDescription("Filter the metrics shown beside the quick test results.")
        self.metric_filter.textChanged.connect(self._filter_metric_items)
        metric_layout.addWidget(self.metric_filter)

        self.metric_list = QListWidget()
        self.metric_list.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        self.metric_list.setMinimumHeight(200)
        self.metric_list.setStyleSheet(AppStyles.metric_list_style())
        metric_layout.addWidget(self.metric_list, 1)
        side_layout.addWidget(metric_box, 1)

        log_box = QGroupBox("Technical Log")
        log_layout = QVBoxLayout(log_box)
        self.log_box = QPlainTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setPlaceholderText("Execution logs will appear here.")
        self.log_box.setStyleSheet(AppStyles.log_view_style())
        log_layout.addWidget(self.log_box, 1)
        side_layout.addWidget(log_box, 1)

        # Wrap side panel in scroll area for small screens
        side_scroll = QScrollArea()
        side_scroll.setWidgetResizable(True)
        side_scroll.setWidget(side_panel)
        side_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        side_scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        split.addWidget(side_scroll)
        split.setSizes([340, 300, 760, 300])

        self._set_empty_test_result_chart()
        self._populate_problem_combo()
        self._populate_algorithm_list()
        self._populate_metric_list()
        self._refresh_test_result_controls()
        self.pop_size_spin.valueChanged.connect(lambda _value: self._update_selection_cards())
        self.n_obj_spin.valueChanged.connect(lambda _value: self._update_selection_cards())
        self.n_var_spin.valueChanged.connect(lambda _value: self._update_selection_cards())
        self.max_fe_spin.valueChanged.connect(lambda _value: self._update_selection_cards())
        self._update_selection_cards()

        return tab

    def _build_experiment_tab(self) -> QWidget:
        """Build the Experiment Module tab with multi-select algorithms, problems, and n_runs."""
        tab = QWidget()
        main_layout = QVBoxLayout(tab)
        main_layout.setContentsMargins(12, 12, 12, 12)
        main_layout.setSpacing(6)

        subtitle = QLabel(
            "Build reproducible batch comparisons: select algorithms and problems, then review the workload before execution."
        )
        subtitle.setWordWrap(True)
        subtitle.setStyleSheet(AppStyles.section_subtitle_style())
        main_layout.addWidget(subtitle)

        split = QSplitter(Qt.Orientation.Horizontal)
        main_layout.addWidget(split, 1)

        # --- Left panel: Algorithm & Problem selection ---
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setSpacing(6)

        # Algorithm multi-select
        algo_box = QGroupBox("1. Algorithms")
        algo_box_layout = QVBoxLayout(algo_box)
        algo_box_layout.setSpacing(4)

        self.exp_algorithm_filter = QLineEdit()
        self.exp_algorithm_filter.setStyleSheet(AppStyles.search_input_style())
        self.exp_algorithm_filter.setPlaceholderText("Search algorithms...")
        self.exp_algorithm_filter.setAccessibleName("Experiment algorithm filter")
        self.exp_algorithm_filter.setAccessibleDescription("Filter the algorithm list for batch experiments.")
        self.exp_algorithm_filter.textChanged.connect(self._apply_exp_list_filters)
        algo_box_layout.addWidget(self.exp_algorithm_filter)

        self.exp_algorithm_list = QListWidget()
        self.exp_algorithm_list.setSelectionMode(QAbstractItemView.SelectionMode.MultiSelection)
        self.exp_algorithm_list.setMinimumHeight(140)
        self.exp_algorithm_list.setStyleSheet(AppStyles.algorithm_list_style())
        algo_box_layout.addWidget(self.exp_algorithm_list)

        self.exp_algo_counter = QLabel("0 selected")
        self.exp_algo_counter.setStyleSheet(f"color: {AppStyles.TEXT_MUTED}; font-size: 11px;")
        algo_box_layout.addWidget(self.exp_algo_counter)

        self.exp_remove_algorithms_btn = QPushButton("Remove Selected")
        self.exp_remove_algorithms_btn.setIcon(app_icon("remove"))
        self.exp_remove_algorithms_btn.setStyleSheet(AppStyles.secondary_button_style())
        self.exp_remove_algorithms_btn.clicked.connect(self._remove_selected_exp_algorithms)
        algo_box_layout.addWidget(self.exp_remove_algorithms_btn)

        left_layout.addWidget(algo_box, 3)

        # Problem multi-select
        prob_box = QGroupBox("2. Problems")
        prob_box_layout = QVBoxLayout(prob_box)
        prob_box_layout.setSpacing(4)

        self.exp_problem_filter = QLineEdit()
        self.exp_problem_filter.setStyleSheet(AppStyles.search_input_style())
        self.exp_problem_filter.setPlaceholderText("Search problems...")
        self.exp_problem_filter.setAccessibleName("Experiment problem filter")
        self.exp_problem_filter.setAccessibleDescription("Filter the problem list for batch experiments.")
        self.exp_problem_filter.textChanged.connect(self._apply_exp_list_filters)
        prob_box_layout.addWidget(self.exp_problem_filter)

        self.exp_problem_list = QListWidget()
        self.exp_problem_list.setSelectionMode(QAbstractItemView.SelectionMode.MultiSelection)
        self.exp_problem_list.setMinimumHeight(140)
        self.exp_problem_list.setStyleSheet(AppStyles.problem_list_style())
        prob_box_layout.addWidget(self.exp_problem_list)

        self.exp_prob_counter = QLabel("0 selected")
        self.exp_prob_counter.setStyleSheet(f"color: {AppStyles.TEXT_MUTED}; font-size: 11px;")
        prob_box_layout.addWidget(self.exp_prob_counter)

        self.exp_remove_problems_btn = QPushButton("Remove Selected")
        self.exp_remove_problems_btn.setIcon(app_icon("remove"))
        self.exp_remove_problems_btn.setStyleSheet(AppStyles.secondary_button_style())
        self.exp_remove_problems_btn.clicked.connect(self._remove_selected_exp_problems)
        prob_box_layout.addWidget(self.exp_remove_problems_btn)

        left_layout.addWidget(prob_box, 3)

        # Metric multi-select
        metric_box = QGroupBox("3. Primary Metric")
        metric_box_layout = QVBoxLayout(metric_box)
        metric_box_layout.setSpacing(4)

        self.exp_metric_filter = QLineEdit()
        self.exp_metric_filter.setStyleSheet(AppStyles.search_input_style())
        self.exp_metric_filter.setPlaceholderText("Search metrics...")
        self.exp_metric_filter.setAccessibleName("Experiment metric filter")
        self.exp_metric_filter.setAccessibleDescription("Filter the metric used to summarize results.")
        self.exp_metric_filter.textChanged.connect(self._apply_exp_list_filters)
        metric_box_layout.addWidget(self.exp_metric_filter)

        self.exp_metric_list = QListWidget()
        self.exp_metric_list.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        self.exp_metric_list.setMinimumHeight(80)
        self.exp_metric_list.setStyleSheet(AppStyles.metric_list_style())
        metric_box_layout.addWidget(self.exp_metric_list)

        left_layout.addWidget(metric_box, 1)

        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        left_scroll.setWidget(left_panel)
        split.addWidget(left_scroll)

        # --- Center panel: parameters ---
        center_panel = QWidget()
        self.exp_center_panel = center_panel
        center_layout = QVBoxLayout(center_panel)
        center_layout.setSpacing(6)

        param_box = QGroupBox("4. Global Parameters")
        param_form = QFormLayout(param_box)
        param_form.setSpacing(6)

        # Number of runs (experiment-specific!)
        self.exp_n_runs_spin = QSpinBox()
        self.exp_n_runs_spin.setRange(1, 100)
        self.exp_n_runs_spin.setValue(DEFAULT_EXPERIMENT_N_RUNS)
        self.exp_n_runs_spin.setToolTip(
            f"Number of independent runs per algorithm (default: {DEFAULT_EXPERIMENT_N_RUNS})"
        )
        exp_n_runs_label = QLabel("Runs per Algorithm:")
        exp_n_runs_label.setStyleSheet(f"color: {AppStyles.TEXT}; font-weight: 700;")
        param_form.addRow(exp_n_runs_label, self.exp_n_runs_spin)

        self.exp_pop_size_spin = QSpinBox()
        self.exp_pop_size_spin.setRange(10, 10_000)
        self.exp_pop_size_spin.setSingleStep(10)
        self.exp_pop_size_spin.setValue(DEFAULT_EXPERIMENT_POP_SIZE)
        self.exp_pop_size_spin.valueChanged.connect(self._on_exp_global_problem_defaults_changed)
        param_form.addRow("Population (N):", self.exp_pop_size_spin)

        self.exp_n_obj_spin = QSpinBox()
        self.exp_n_obj_spin.setRange(1, 20)
        self.exp_n_obj_spin.setValue(2)
        self.exp_n_obj_spin.setToolTip("Number of objectives (M).")
        self.exp_n_obj_spin.valueChanged.connect(self._on_exp_global_problem_defaults_changed)

        self.exp_n_var_spin = QSpinBox()
        self.exp_n_var_spin.setRange(1, 10_000)
        self.exp_n_var_spin.setValue(30)
        self.exp_n_var_spin.setToolTip(
            "Decision variables (D). For DTLZ problems, D is auto-computed from M (D = M + k - 1)."
        )
        self.exp_n_var_spin.valueChanged.connect(self._on_exp_global_problem_defaults_changed)

        self.exp_max_fe_spin = QSpinBox()
        self.exp_max_fe_spin.setRange(100, 100_000_000)
        self.exp_max_fe_spin.setSingleStep(1000)
        self.exp_max_fe_spin.setValue(DEFAULT_MAX_FE)
        self.exp_max_fe_spin.valueChanged.connect(self._on_exp_global_problem_defaults_changed)
        param_form.addRow("MaxFE:", self.exp_max_fe_spin)

        self.exp_problem_dims_hint = QLabel("M and D are configured per problem below.")
        self.exp_problem_dims_hint.setStyleSheet(f"color: {AppStyles.TEXT_MUTED};")
        self.exp_problem_dims_hint.setWordWrap(True)
        param_form.addRow("", self.exp_problem_dims_hint)

        # Seed mode
        self.exp_seed_mode_combo = QComboBox()
        self.exp_seed_mode_combo.addItem("Random", SEED_MODE_RANDOM)
        self.exp_seed_mode_combo.addItem("Fixed", SEED_MODE_FIXED)
        self.exp_seed_mode_combo.addItem("Sequence", SEED_MODE_SEQUENCE)
        param_form.addRow("Seed:", self.exp_seed_mode_combo)

        self.exp_seed_base_spin = QSpinBox()
        self.exp_seed_base_spin.setRange(1, 2_147_483_647)
        self.exp_seed_base_spin.setValue(1)
        param_form.addRow("Base Seed:", self.exp_seed_base_spin)

        self.exp_seed_step_spin = QSpinBox()
        self.exp_seed_step_spin.setRange(1, 100_000)
        self.exp_seed_step_spin.setValue(1)
        param_form.addRow("Seed Step:", self.exp_seed_step_spin)

        self.exp_seed_label = QLabel("")
        self.exp_seed_label.setStyleSheet(f"color: {AppStyles.TEXT_MUTED};")
        self.exp_seed_label.setWordWrap(True)
        param_form.addRow("", self.exp_seed_label)
        self.exp_seed_mode_combo.currentIndexChanged.connect(self._on_exp_seed_mode_changed)
        self.exp_seed_base_spin.valueChanged.connect(self._on_exp_seed_mode_changed)
        self.exp_seed_step_spin.valueChanged.connect(self._on_exp_seed_mode_changed)
        self._on_exp_seed_mode_changed()

        center_layout.addWidget(param_box)

        # Per-problem overrides (N/M/D only), inspired by benchmark parameter cards.
        prob_cfg_box = QGroupBox("5. Per-Problem Settings")
        prob_cfg_form = QFormLayout(prob_cfg_box)
        prob_cfg_form.setSpacing(6)

        self.exp_problem_cfg_combo = QComboBox()
        self.exp_problem_cfg_combo.currentIndexChanged.connect(self._on_exp_problem_cfg_target_changed)
        prob_cfg_form.addRow("Problem:", self.exp_problem_cfg_combo)

        self.exp_problem_pop_size_spin = QSpinBox()
        self.exp_problem_pop_size_spin.setRange(10, 10_000)
        self.exp_problem_pop_size_spin.setSingleStep(10)
        self.exp_problem_pop_size_spin.valueChanged.connect(self._on_exp_problem_override_value_changed)
        prob_cfg_form.addRow("Population (N):", self.exp_problem_pop_size_spin)

        self.exp_problem_n_obj_spin = QSpinBox()
        self.exp_problem_n_obj_spin.setRange(1, 20)
        self.exp_problem_n_obj_spin.valueChanged.connect(self._on_exp_problem_override_n_obj_changed)
        prob_cfg_form.addRow("Objectives (M):", self.exp_problem_n_obj_spin)

        self.exp_problem_n_var_spin = QSpinBox()
        self.exp_problem_n_var_spin.setRange(1, 10_000)
        self.exp_problem_n_var_spin.valueChanged.connect(self._on_exp_problem_override_value_changed)
        prob_cfg_form.addRow("Decision Variables (D):", self.exp_problem_n_var_spin)

        self.exp_problem_max_fe_spin = QSpinBox()
        self.exp_problem_max_fe_spin.setRange(100, 100_000_000)
        self.exp_problem_max_fe_spin.setSingleStep(1000)
        self.exp_problem_max_fe_spin.valueChanged.connect(self._on_exp_problem_override_value_changed)
        # maxFE is global for the experiment; keep widget only for backward-compat code paths.
        self.exp_problem_max_fe_spin.setVisible(False)

        self.exp_problem_cfg_hint = QLabel("Select one or more problems, then adjust each one here when needed.")
        self.exp_problem_cfg_hint.setStyleSheet(f"color: {AppStyles.TEXT_MUTED};")
        self.exp_problem_cfg_hint.setWordWrap(True)
        prob_cfg_form.addRow(self.exp_problem_cfg_hint)

        self.exp_problem_cfg_reset_btn = QPushButton("Reset Problem")
        self.exp_problem_cfg_reset_btn.setIcon(app_icon("restart_alt"))
        self.exp_problem_cfg_reset_btn.clicked.connect(self._reset_exp_problem_override)
        self.exp_problem_cfg_duplicate_btn = QPushButton("Duplicate Variant")
        self.exp_problem_cfg_duplicate_btn.setIcon(app_icon("content_copy"))
        self.exp_problem_cfg_duplicate_btn.clicked.connect(self._duplicate_exp_problem_variant)
        self.exp_problem_cfg_remove_variant_btn = QPushButton("Remove Variant")
        self.exp_problem_cfg_remove_variant_btn.setIcon(app_icon("delete"))
        self.exp_problem_cfg_remove_variant_btn.clicked.connect(self._remove_exp_problem_variant)
        prob_cfg_btn_row = QWidget()
        prob_cfg_btn_layout = QHBoxLayout(prob_cfg_btn_row)
        prob_cfg_btn_layout.setContentsMargins(0, 0, 0, 0)
        prob_cfg_btn_layout.setSpacing(6)
        prob_cfg_btn_layout.addWidget(self.exp_problem_cfg_duplicate_btn)
        prob_cfg_btn_layout.addWidget(self.exp_problem_cfg_remove_variant_btn)
        prob_cfg_form.addRow(prob_cfg_btn_row)
        prob_cfg_form.addRow(self.exp_problem_cfg_reset_btn)

        center_layout.addWidget(prob_cfg_box)

        # Operators
        op_box = QGroupBox("Operators")
        op_form = QFormLayout(op_box)
        op_form.setSpacing(6)

        self.exp_operator_algo_combo = QComboBox()
        self.exp_operator_algo_combo.currentIndexChanged.connect(self._on_exp_operator_target_changed)
        op_form.addRow("Algorithm:", self.exp_operator_algo_combo)

        # Crossover
        self.exp_crossover_combo = QComboBox()
        self.exp_crossover_combo.addItem("Default (EmoPyLab)", "default")
        self.exp_crossover_combo.addItem("SBX (Simulated Binary)", "sbx")
        self.exp_crossover_combo.addItem("PMX (Partial Match)", "pmx")
        self.exp_crossover_combo.addItem("OX (Order)", "ox")
        self.exp_crossover_combo.addItem("UX (Uniform)", "ux")
        self.exp_crossover_combo.addItem("DEX (Discrete Exchange)", "dex")
        self.exp_crossover_combo.addItem("None (no crossover)", "none")
        op_form.addRow("Crossover:", self.exp_crossover_combo)

        self.exp_crossover_eta_spin = QDoubleSpinBox()
        self.exp_crossover_eta_spin.setRange(1.0, 100.0)
        self.exp_crossover_eta_spin.setSingleStep(1.0)
        self.exp_crossover_eta_spin.setDecimals(1)
        self.exp_crossover_eta_spin.setValue(15.0)
        self.exp_crossover_eta_label = QLabel("  eta:")
        self.exp_crossover_eta_label.setStyleSheet(f"color: {AppStyles.TEXT_MUTED}; font-size: 11px;")
        op_form.addRow(self.exp_crossover_eta_label, self.exp_crossover_eta_spin)

        self.exp_crossover_prob_spin = QDoubleSpinBox()
        self.exp_crossover_prob_spin.setRange(0.0, 1.0)
        self.exp_crossover_prob_spin.setSingleStep(0.05)
        self.exp_crossover_prob_spin.setDecimals(2)
        self.exp_crossover_prob_spin.setValue(0.9)
        self.exp_crossover_prob_label = QLabel("  Prob:")
        self.exp_crossover_prob_label.setStyleSheet(f"color: {AppStyles.TEXT_MUTED}; font-size: 11px;")
        op_form.addRow(self.exp_crossover_prob_label, self.exp_crossover_prob_spin)

        self.exp_crossover_combo.currentIndexChanged.connect(self._on_exp_crossover_changed)

        # Mutation
        self.exp_mutation_combo = QComboBox()
        self.exp_mutation_combo.addItem("Default (EmoPyLab)", "default")
        self.exp_mutation_combo.addItem("PM (Polynomial)", "pm")
        self.exp_mutation_combo.addItem("SBM (Single Bit)", "sbm")
        self.exp_mutation_combo.addItem("Bitflip", "bitflip")
        self.exp_mutation_combo.addItem("None (no mutation)", "none")
        op_form.addRow("Mutation:", self.exp_mutation_combo)

        self.exp_mutation_eta_spin = QDoubleSpinBox()
        self.exp_mutation_eta_spin.setRange(1.0, 100.0)
        self.exp_mutation_eta_spin.setSingleStep(1.0)
        self.exp_mutation_eta_spin.setDecimals(1)
        self.exp_mutation_eta_spin.setValue(20.0)
        self.exp_mutation_eta_label = QLabel("  eta:")
        self.exp_mutation_eta_label.setStyleSheet(f"color: {AppStyles.TEXT_MUTED}; font-size: 11px;")
        op_form.addRow(self.exp_mutation_eta_label, self.exp_mutation_eta_spin)

        self.exp_mutation_prob_spin = QDoubleSpinBox()
        self.exp_mutation_prob_spin.setRange(0.0, 1.0)
        self.exp_mutation_prob_spin.setSingleStep(0.01)
        self.exp_mutation_prob_spin.setDecimals(3)
        self.exp_mutation_prob_spin.setValue(0.0)
        self.exp_mutation_prob_label = QLabel("  Prob (0=auto):")
        self.exp_mutation_prob_label.setStyleSheet(f"color: {AppStyles.TEXT_MUTED}; font-size: 11px;")
        op_form.addRow(self.exp_mutation_prob_label, self.exp_mutation_prob_spin)

        self.exp_mutation_combo.currentIndexChanged.connect(self._on_exp_mutation_changed)

        # Selection
        self.exp_selection_combo = QComboBox()
        self.exp_selection_combo.addItem("Default (EmoPyLab)", "default")
        self.exp_selection_combo.addItem("Tournament", "tournament")
        self.exp_selection_combo.addItem("Random", "random")
        self.exp_selection_combo.addItem("Best", "best")
        self.exp_selection_combo.addItem("Random Binary", "random_binary")
        op_form.addRow("Selection:", self.exp_selection_combo)

        self.exp_sampling_combo = QComboBox()
        self.exp_sampling_combo.addItem("Default (EmoPyLab)", "default")
        op_form.addRow("Sampling:", self.exp_sampling_combo)

        self.exp_selection_pressure_spin = QSpinBox()
        self.exp_selection_pressure_spin.setRange(1, 20)
        self.exp_selection_pressure_spin.setValue(2)
        self.exp_selection_pressure_label = QLabel("  Pressure:")
        self.exp_selection_pressure_label.setStyleSheet(f"color: {AppStyles.TEXT_MUTED}; font-size: 11px;")
        op_form.addRow(self.exp_selection_pressure_label, self.exp_selection_pressure_spin)

        self.exp_selection_combo.currentIndexChanged.connect(self._on_exp_selection_changed)
        self.exp_selection_combo.currentIndexChanged.connect(self._on_exp_algorithm_operator_value_changed)
        self.exp_sampling_combo.currentIndexChanged.connect(self._on_exp_algorithm_operator_value_changed)
        self.exp_crossover_combo.currentIndexChanged.connect(self._on_exp_algorithm_operator_value_changed)
        self.exp_mutation_combo.currentIndexChanged.connect(self._on_exp_algorithm_operator_value_changed)
        self.exp_crossover_eta_spin.valueChanged.connect(self._on_exp_algorithm_operator_value_changed)
        self.exp_crossover_prob_spin.valueChanged.connect(self._on_exp_algorithm_operator_value_changed)
        self.exp_mutation_eta_spin.valueChanged.connect(self._on_exp_algorithm_operator_value_changed)
        self.exp_mutation_prob_spin.valueChanged.connect(self._on_exp_algorithm_operator_value_changed)
        self.exp_selection_pressure_spin.valueChanged.connect(self._on_exp_algorithm_operator_value_changed)
        self._populate_all_operator_combos()

        # UI/UX simplification: operator configuration is hidden; execution uses algorithm defaults.
        op_box.setVisible(False)
        center_layout.addWidget(op_box)

        # Backend
        backend_box = QGroupBox("6. Compute")
        backend_form = QFormLayout(backend_box)
        backend_form.setSpacing(6)

        default_workers, max_workers = resolve_parallel_worker_limits()

        self.exp_parallel_workers_spin = QSpinBox()
        self.exp_parallel_workers_spin.setRange(1, max_workers)
        self.exp_parallel_workers_spin.setValue(default_workers)
        self.exp_parallel_workers_spin.setToolTip(
            "Parallel workers for multi-run executions (processes when supported). "
            f"Recommended: {default_workers}. Max on this machine: {max_workers}. "
            f"When LARC_NSGA3 is selected, workers are capped at {LLM_LOCAL_MAX_PARALLEL_WORKERS} for memory safety."
        )
        backend_form.addRow("Parallel Workers:", self.exp_parallel_workers_spin)

        self.exp_compute_backend_combo = QComboBox()
        self.exp_compute_backend_combo.addItem(BACKEND_OPTIONS["cpu"], "cpu")
        self.exp_compute_backend_combo.addItem(BACKEND_OPTIONS["gpu"], "gpu")
        # MLX offered and default only when actually detected; CPU default otherwise.
        if bool(self.mlx_runtime.get("mlx_ok")):
            self.exp_compute_backend_combo.addItem(BACKEND_OPTIONS["mlx"], "mlx")
            self.exp_compute_backend_combo.setCurrentIndex(self.exp_compute_backend_combo.findData("mlx"))
        else:
            self.exp_compute_backend_combo.setCurrentIndex(self.exp_compute_backend_combo.findData("cpu"))
        self.exp_compute_backend_combo.currentIndexChanged.connect(self._on_exp_backend_mode_changed)
        backend_form.addRow("Backend:", self.exp_compute_backend_combo)

        # UI/UX simplification: fixed float32 (speed) for experiment backend too.
        self.exp_gpu_dtype_combo = QComboBox()
        self.exp_gpu_dtype_combo.addItem("float32", "float32")
        self.exp_gpu_dtype_combo.setCurrentIndex(0)
        self.exp_gpu_dtype_combo.setVisible(False)

        self.exp_use_pf_check = QCheckBox("Use the true Pareto front when available")
        self.exp_use_pf_check.setChecked(True)
        backend_form.addRow(self.exp_use_pf_check)

        self.exp_selected_algorithms_summary_label = QLabel("No algorithm selected")
        self.exp_selected_algorithms_summary_label.setStyleSheet(f"color: {AppStyles.TEXT_MUTED};")
        self.exp_selected_algorithms_summary_label.setWordWrap(True)

        center_layout.addWidget(backend_box)
        exp_algo_summary_box = QGroupBox("Algorithm Summary")
        exp_algo_summary_layout = QVBoxLayout(exp_algo_summary_box)
        exp_algo_summary_layout.setContentsMargins(8, 8, 8, 8)
        exp_algo_summary_layout.addWidget(self.exp_selected_algorithms_summary_label)
        center_layout.addWidget(exp_algo_summary_box)
        center_layout.addStretch(1)

        center_scroll = QScrollArea()
        center_scroll.setWidgetResizable(True)
        center_scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        center_scroll.setWidget(center_panel)
        split.addWidget(center_scroll)

        # --- Right panel: actions + progress + log ---
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setSpacing(6)

        action_box = QGroupBox("7. Review and Run")
        action_layout = QVBoxLayout(action_box)
        action_layout.setSpacing(6)

        self.exp_workload_summary_label = QLabel(
            "Select or review at least one algorithm, one problem, and one metric before running the experiment."
        )
        self.exp_workload_summary_label.setWordWrap(True)
        self.exp_workload_summary_label.setMinimumHeight(52)
        self.exp_workload_summary_label.setStyleSheet(AppStyles.metric_tile_style(AppStyles.PRIMARY))
        action_layout.addWidget(self.exp_workload_summary_label)

        btn_row = QHBoxLayout()
        self.exp_btn_start = QPushButton("Run Experiment")
        self.exp_btn_start.setAccessibleName("Run Experiment")
        self.exp_btn_start.setAccessibleDescription("Run the configured batch experiment.")
        self.exp_btn_start.setIcon(app_icon("play_arrow"))
        self.exp_btn_start.setStyleSheet(AppStyles.primary_button_style())
        self.exp_btn_start.clicked.connect(self._start_experiment)
        btn_row.addWidget(self.exp_btn_start)

        self.exp_btn_stop = QPushButton("Stop")
        self.exp_btn_stop.setAccessibleName("Stop Experiment")
        self.exp_btn_stop.setAccessibleDescription("Request cancellation of the running experiment.")
        self.exp_btn_stop.setIcon(app_icon("stop"))
        self.exp_btn_stop.setEnabled(False)
        self.exp_btn_stop.setStyleSheet(AppStyles.secondary_button_style())
        self.exp_btn_stop.clicked.connect(self._stop_experiment)
        btn_row.addWidget(self.exp_btn_stop)
        action_layout.addLayout(btn_row)

        self.exp_progress = QProgressBar()
        self.exp_progress.setRange(0, 100)
        self.exp_progress.setValue(0)
        action_layout.addWidget(self.exp_progress)

        self.exp_status_label = QLabel("Ready - review the configuration, then run the experiment.")
        self.exp_status_label.setStyleSheet(AppStyles.panel_hint_style())
        self.exp_status_label.setWordWrap(True)
        action_layout.addWidget(self.exp_status_label)

        right_layout.addWidget(action_box)

        results_store_box = QGroupBox("Raw Results")
        self.exp_results_store_box = results_store_box
        results_store_layout = QVBoxLayout(results_store_box)
        results_store_layout.setSpacing(4)

        self.exp_results_path_edit = QLineEdit(str(self.exp_results_storage_path))
        self.exp_results_path_edit.setPlaceholderText("JSON file used to save or load raw experiment results")
        self.exp_results_path_edit.setToolTip(
            "Raw experiment run payloads are saved here after each experiment.\n"
            "These saved raw runs are reused to recalculate summary metrics when you change the metric."
        )
        self.exp_results_path_edit.editingFinished.connect(self._on_exp_results_path_editing_finished)

        exp_results_path_row = QHBoxLayout()
        exp_results_path_row.addWidget(self.exp_results_path_edit, 1)

        self.exp_results_path_browse_btn = QPushButton("Browse...")
        self.exp_results_path_browse_btn.setIcon(app_icon("folder_open"))
        self.exp_results_path_browse_btn.setStyleSheet(AppStyles.secondary_button_style())
        self.exp_results_path_browse_btn.clicked.connect(self._browse_exp_results_path)
        exp_results_path_row.addWidget(self.exp_results_path_browse_btn)

        self.exp_results_path_load_btn = QPushButton("Load")
        self.exp_results_path_load_btn.setIcon(app_icon("upload_file"))
        self.exp_results_path_load_btn.setStyleSheet(AppStyles.secondary_button_style())
        self.exp_results_path_load_btn.clicked.connect(self._load_exp_results_from_current_path)
        exp_results_path_row.addWidget(self.exp_results_path_load_btn)

        results_store_layout.addLayout(exp_results_path_row)

        self.exp_results_storage_info_label = QLabel("")
        self.exp_results_storage_info_label.setStyleSheet(f"color: {AppStyles.TEXT_MUTED};")
        self.exp_results_storage_info_label.setWordWrap(True)
        results_store_layout.addWidget(self.exp_results_storage_info_label)

        self.exp_send_to_analysis_btn = QPushButton("Open in Results")
        self.exp_send_to_analysis_btn.setIcon(app_icon("analytics"))
        self.exp_send_to_analysis_btn.setStyleSheet(AppStyles.secondary_button_style())
        self.exp_send_to_analysis_btn.setToolTip(
            "Send the latest Experiment results (in-memory or current raw-results file) to the Analysis & MCDM workspace."
        )
        self.exp_send_to_analysis_btn.clicked.connect(self._send_latest_experiment_to_analysis)
        results_store_layout.addWidget(self.exp_send_to_analysis_btn)

        right_layout.addWidget(results_store_box)

        exp_config_box = QGroupBox("Experiment Configuration")
        self.exp_config_box = exp_config_box
        exp_config_layout = QVBoxLayout(exp_config_box)
        exp_config_layout.setSpacing(4)

        self.exp_config_autosave_path_edit = QLineEdit(str(self.base_dir / EXP_CONFIG_FILENAME))
        self.exp_config_autosave_path_edit.setReadOnly(True)
        self.exp_config_autosave_path_edit.setToolTip(
            "Last experiment configuration is auto-saved to this file when an experiment starts\n"
            "and auto-loaded when the application opens."
        )
        exp_config_layout.addWidget(self.exp_config_autosave_path_edit)

        self.exp_config_autosave_info_label = QLabel(
            "Auto-save: enabled when an experiment starts. Auto-load: enabled at application startup."
        )
        self.exp_config_autosave_info_label.setStyleSheet(f"color: {AppStyles.TEXT_MUTED};")
        self.exp_config_autosave_info_label.setWordWrap(True)
        exp_config_layout.addWidget(self.exp_config_autosave_info_label)

        exp_config_btn_row = QHBoxLayout()
        self.exp_config_save_btn = QPushButton("Save Config...")
        self.exp_config_save_btn.setIcon(app_icon("save"))
        self.exp_config_save_btn.setStyleSheet(AppStyles.secondary_button_style())
        self.exp_config_save_btn.clicked.connect(self._save_experiment_config_to_file)
        exp_config_btn_row.addWidget(self.exp_config_save_btn)

        self.exp_config_load_btn = QPushButton("Load Config...")
        self.exp_config_load_btn.setIcon(app_icon("folder_open"))
        self.exp_config_load_btn.setStyleSheet(AppStyles.secondary_button_style())
        self.exp_config_load_btn.clicked.connect(self._load_experiment_config_from_file)
        exp_config_btn_row.addWidget(self.exp_config_load_btn)

        self.exp_config_load_last_btn = QPushButton("Load Last")
        self.exp_config_load_last_btn.setIcon(app_icon("history"))
        self.exp_config_load_last_btn.setStyleSheet(AppStyles.secondary_button_style())
        self.exp_config_load_last_btn.clicked.connect(self._load_last_experiment_config_from_ui)
        exp_config_btn_row.addWidget(self.exp_config_load_last_btn)
        exp_config_btn_row.addStretch(1)
        exp_config_layout.addLayout(exp_config_btn_row)

        right_layout.addWidget(exp_config_box)

        progress_table_box = QGroupBox("Summary Table")
        progress_table_layout = QVBoxLayout(progress_table_box)
        progress_table_layout.setSpacing(4)

        self.exp_run_progress_hint = QLabel(
            "Single-metric summary: Problem | M | D | Run | Algorithms. The table updates during execution."
        )
        self.exp_run_progress_hint.setStyleSheet(f"color: {AppStyles.TEXT_MUTED};")
        self.exp_run_progress_hint.setWordWrap(True)
        progress_table_layout.addWidget(self.exp_run_progress_hint)

        self.exp_run_progress_table = QTableWidget()
        self.exp_run_progress_table.setObjectName("expRunProgressTable")
        self.exp_run_progress_table.setColumnCount(0)
        self.exp_run_progress_table.setRowCount(0)
        self.exp_run_progress_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.exp_run_progress_table.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        self.exp_run_progress_table.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.exp_run_progress_table.setAlternatingRowColors(True)
        self.exp_run_progress_table.verticalHeader().setVisible(True)
        self.exp_run_progress_table.verticalHeader().setHighlightSections(False)
        self.exp_run_progress_table.horizontalHeader().setStretchLastSection(True)
        self.exp_run_progress_table.horizontalHeader().setHighlightSections(False)
        self.exp_run_progress_table.setStyleSheet(AppStyles.table_style("expRunProgressTable"))
        progress_table_layout.addWidget(self.exp_run_progress_table, 1)

        exp_export_row = QHBoxLayout()
        self.exp_save_progress_csv_btn = QPushButton("Save CSV")
        self.exp_save_progress_csv_btn.setIcon(app_icon("table_view"))
        self.exp_save_progress_csv_btn.setStyleSheet(AppStyles.secondary_button_style())
        self.exp_save_progress_csv_btn.clicked.connect(self._save_exp_progress_table_csv)
        self.exp_save_progress_csv_btn.setEnabled(False)
        exp_export_row.addWidget(self.exp_save_progress_csv_btn)

        self.exp_save_progress_latex_btn = QPushButton("Save LaTeX")
        self.exp_save_progress_latex_btn.setIcon(app_icon("code"))
        self.exp_save_progress_latex_btn.setStyleSheet(AppStyles.secondary_button_style())
        self.exp_save_progress_latex_btn.clicked.connect(self._save_exp_progress_table_latex)
        self.exp_save_progress_latex_btn.setEnabled(False)
        exp_export_row.addWidget(self.exp_save_progress_latex_btn)
        exp_export_row.addStretch(1)
        progress_table_layout.addLayout(exp_export_row)

        # Experiment execution log
        self.exp_log_box = QPlainTextEdit()
        self.exp_log_box.setReadOnly(True)
        self.exp_log_box.setMaximumBlockCount(2000)
        self.exp_log_box.setFont(QFont(StylesAppStyles.typography.font_family_mono, 10))
        self.exp_log_box.setPlaceholderText("Experiment logs will appear here.")
        self.exp_log_box.setMinimumHeight(140)
        self.exp_log_box.setStyleSheet(AppStyles.log_view_style())

        right_layout.addWidget(progress_table_box, 3)
        exp_log_group = QGroupBox("Technical Log")
        exp_log_layout = QVBoxLayout(exp_log_group)
        exp_log_layout.setContentsMargins(6, 6, 6, 6)
        exp_log_layout.addWidget(self.exp_log_box)
        right_layout.addWidget(exp_log_group, 1)

        right_scroll = QScrollArea()
        right_scroll.setWidgetResizable(True)
        right_scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        right_scroll.setWidget(right_panel)
        split.addWidget(right_scroll)
        
        # Apply responsiveness limits and proportions
        left_scroll.setMinimumWidth(260)
        center_scroll.setMinimumWidth(260)
        right_scroll.setMinimumWidth(380)
        
        split.setChildrenCollapsible(False)
        split.setSizes([320, 300, 380])
        split.setStretchFactor(0, 1)
        split.setStretchFactor(1, 1)
        split.setStretchFactor(2, 4)

        # Initial operator param visibility
        self._on_exp_crossover_changed()
        self._on_exp_mutation_changed()
        self._on_exp_selection_changed()

        # Populate lists
        self._populate_exp_lists()
        self.exp_n_runs_spin.valueChanged.connect(lambda _value: self._refresh_exp_workload_summary())
        self.exp_max_fe_spin.valueChanged.connect(lambda _value: self._refresh_exp_workload_summary())
        self._refresh_exp_workload_summary()

        return tab

    def _populate_exp_lists(self) -> None:
        """Populate experiment module algorithm, problem, and metric lists with checkboxes."""
        self.exp_problem_variants = {
            key: dict(value)
            for key, value in self.exp_problem_variants.items()
            if isinstance(value, dict)
            and str(value.get("problem_id", "")).strip() in self.problem_specs
        }
        self.exp_problem_overrides = {
            pid: dict(values)
            for pid, values in self.exp_problem_overrides.items()
            if isinstance(values, dict)
            and (
                pid in self.problem_specs
                or (pid in self.exp_problem_variants)
            )
        }
        self.exp_algorithm_operator_overrides = {
            aid: dict(values)
            for aid, values in self.exp_algorithm_operator_overrides.items()
            if aid in self.algorithm_specs and isinstance(values, dict)
        }
        previous_metric_ids = set(self._exp_checked_ids(self.exp_metric_list))

        # Disconnect old signals to prevent duplicate connections on reload
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            try:
                self.exp_algorithm_list.itemChanged.disconnect(self._on_exp_algo_check_changed)
            except (RuntimeError, TypeError):
                pass
            try:
                self.exp_problem_list.itemChanged.disconnect(self._on_exp_prob_check_changed)
            except (RuntimeError, TypeError):
                pass
            try:
                self.exp_algorithm_list.itemSelectionChanged.disconnect(self._on_exp_algo_selection_changed)
            except (RuntimeError, TypeError):
                pass
            try:
                self.exp_problem_list.itemSelectionChanged.disconnect(self._on_exp_prob_selection_changed)
            except (RuntimeError, TypeError):
                pass
            try:
                self.exp_problem_list.currentItemChanged.disconnect(self._on_exp_problem_current_changed)
            except (RuntimeError, TypeError):
                pass
            try:
                self.exp_metric_list.itemChanged.disconnect(self._on_exp_metric_check_changed)
            except (RuntimeError, TypeError):
                pass

        # Algorithms
        self.exp_algorithm_list.clear()
        for algo_id, spec in sorted(
            self.algorithm_specs.items(),
            key=lambda x: _natural_lexicographic_key(x[1].name),
        ):
            item = QListWidgetItem(spec.name)
            item.setData(Qt.ItemDataRole.UserRole, algo_id)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Unchecked)
            self.exp_algorithm_list.addItem(item)
        self.exp_algorithm_list.itemChanged.connect(self._on_exp_algo_check_changed)
        self.exp_algorithm_list.itemSelectionChanged.connect(self._on_exp_algo_selection_changed)

        # Problems
        self.exp_problem_list.clear()
        visible_problem_ids = {
            spec.id
            for spec in self._iter_visible_problem_specs(
                prefer_jax=self._exp_backend_prefers_jax_for("problems")
            )
        }
        for prob_id, spec in sorted(
            self.problem_specs.items(),
            key=lambda x: _natural_lexicographic_key(x[1].name),
        ):
            if visible_problem_ids and prob_id not in visible_problem_ids:
                continue
            item = QListWidgetItem(spec.name)
            item.setData(Qt.ItemDataRole.UserRole, prob_id)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Unchecked)
            self.exp_problem_list.addItem(item)
        self.exp_problem_list.itemChanged.connect(self._on_exp_prob_check_changed)
        self.exp_problem_list.itemSelectionChanged.connect(self._on_exp_prob_selection_changed)
        self.exp_problem_list.currentItemChanged.connect(self._on_exp_problem_current_changed)

        # Default startup selections for experiment module.
        default_exp_algo_item: QListWidgetItem | None = None
        for i in range(self.exp_algorithm_list.count()):
            item = self.exp_algorithm_list.item(i)
            algo_id = item.data(Qt.ItemDataRole.UserRole)
            spec = self.algorithm_specs.get(algo_id) if isinstance(algo_id, str) else None
            if spec is not None and core_normalize_backend_token(spec.name) == "nsga3":
                default_exp_algo_item = item
                break
        if default_exp_algo_item is None and self.exp_algorithm_list.count() > 0:
            default_exp_algo_item = self.exp_algorithm_list.item(0)
        if default_exp_algo_item is not None:
            default_exp_algo_item.setCheckState(Qt.CheckState.Checked)
            self.exp_algorithm_list.setCurrentItem(default_exp_algo_item)

        default_exp_prob_item: QListWidgetItem | None = None
        for i in range(self.exp_problem_list.count()):
            item = self.exp_problem_list.item(i)
            prob_id = item.data(Qt.ItemDataRole.UserRole)
            spec = self.problem_specs.get(prob_id) if isinstance(prob_id, str) else None
            if spec is not None and core_normalize_backend_token(spec.name) == "zdt1":
                default_exp_prob_item = item
                break
        if default_exp_prob_item is None and self.exp_problem_list.count() > 0:
            default_exp_prob_item = self.exp_problem_list.item(0)
        if default_exp_prob_item is not None:
            default_exp_prob_item.setCheckState(Qt.CheckState.Checked)
            self.exp_problem_list.setCurrentItem(default_exp_prob_item)

        # Metrics
        self._populate_exp_metric_list(
            checked_ids=previous_metric_ids if previous_metric_ids else None
        )
        self.exp_metric_list.itemChanged.connect(self._on_exp_metric_check_changed)

        self.exp_algo_counter.setText(f"{self._count_checked_exp_items(self.exp_algorithm_list)} selected")
        self.exp_prob_counter.setText(f"{self._count_checked_exp_items(self.exp_problem_list)} selected")
        self._refresh_exp_problem_cfg_targets()
        self._refresh_exp_operator_cfg_targets()
        self._refresh_exp_selected_algorithms_summary()
        self._refresh_exp_workload_summary()
        self._apply_exp_list_filters()

    def _populate_exp_metric_list(self, checked_ids: set[str] | None = None) -> None:
        prefer_jax = self._exp_backend_prefers_jax_for("metrics")
        mapped_ids = set(
            self._map_metric_ids_to_backend(checked_ids or set(), prefer_jax=prefer_jax)
        )
        visible_specs = self._iter_metric_specs_for_backend(prefer_jax=prefer_jax)

        self.exp_metric_list.blockSignals(True)
        self.exp_metric_list.clear()
        any_metric_checked = False
        for spec in visible_specs:
            item = QListWidgetItem(spec.label)
            item.setData(Qt.ItemDataRole.UserRole, spec.id)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            is_checked = spec.id in mapped_ids if mapped_ids else self._is_default_metric_checked(spec)
            item.setCheckState(Qt.CheckState.Checked if is_checked else Qt.CheckState.Unchecked)
            any_metric_checked = any_metric_checked or is_checked
            self.exp_metric_list.addItem(item)
        if not any_metric_checked and self.exp_metric_list.count() > 0:
            preferred_idx = -1
            for i in range(self.exp_metric_list.count()):
                item = self.exp_metric_list.item(i)
                metric_id = item.data(Qt.ItemDataRole.UserRole)
                spec = self.metric_specs.get(metric_id) if isinstance(metric_id, str) else None
                if spec is not None and self._is_default_metric_checked(spec):
                    preferred_idx = i
                    break
            if preferred_idx < 0:
                preferred_idx = 0
            self.exp_metric_list.item(preferred_idx).setCheckState(Qt.CheckState.Checked)
        # Experiment Module: enforce exactly one checked metric in the UI.
        first_checked_seen = False
        for i in range(self.exp_metric_list.count()):
            item = self.exp_metric_list.item(i)
            if item.checkState() == Qt.CheckState.Checked:
                if not first_checked_seen:
                    first_checked_seen = True
                else:
                    item.setCheckState(Qt.CheckState.Unchecked)
        self.exp_metric_list.blockSignals(False)

    def _refresh_exp_problem_list_for_backend(self) -> None:
        if not hasattr(self, "exp_problem_list"):
            return

        checked_ids = set(self._exp_checked_ids(self.exp_problem_list))
        current_id = None
        current_item = self.exp_problem_list.currentItem()
        if current_item is not None:
            value = current_item.data(Qt.ItemDataRole.UserRole)
            if isinstance(value, str):
                current_id = value

        visible_specs = self._iter_visible_problem_specs(
            prefer_jax=self._exp_backend_prefers_jax_for("problems")
        )
        visible_ids = {spec.id for spec in visible_specs}

        self.exp_problem_list.blockSignals(True)
        self.exp_problem_list.clear()
        target_item: QListWidgetItem | None = None
        for spec in visible_specs:
            item = QListWidgetItem(spec.name)
            item.setData(Qt.ItemDataRole.UserRole, spec.id)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Checked if spec.id in checked_ids else Qt.CheckState.Unchecked)
            self.exp_problem_list.addItem(item)
            if current_id is not None and spec.id == current_id:
                target_item = item

        if target_item is None and self.exp_problem_list.count() > 0:
            target_item = self.exp_problem_list.item(0)
        if target_item is not None:
            self.exp_problem_list.setCurrentItem(target_item)

        self.exp_problem_list.blockSignals(False)
        if checked_ids and not checked_ids.intersection(visible_ids):
            self.exp_prob_counter.setText("0 selected")
        else:
            self.exp_prob_counter.setText(f"{self._count_checked_exp_items(self.exp_problem_list)} selected")
        self._refresh_exp_problem_cfg_targets()
        self._refresh_exp_workload_summary()
        self._apply_exp_list_filters()

    def _clear_checked_exp_items(self, list_widget: QListWidget) -> int:
        """Clear current selection in an experiment list (fallback: unchecked items)."""
        changed = 0
        list_widget.blockSignals(True)
        selected_items = list_widget.selectedItems()
        if selected_items:
            for item in selected_items:
                item.setSelected(False)
                if item.flags() & Qt.ItemFlag.ItemIsUserCheckable:
                    item.setCheckState(Qt.CheckState.Unchecked)
                changed += 1
        else:
            for i in range(list_widget.count()):
                item = list_widget.item(i)
                if item.checkState() == Qt.CheckState.Checked:
                    item.setCheckState(Qt.CheckState.Unchecked)
                    changed += 1
        list_widget.clearSelection()
        list_widget.setCurrentRow(-1)
        list_widget.blockSignals(False)
        return changed

    def _count_checked_exp_items(self, list_widget: QListWidget) -> int:
        """Count effective selected items (union of selected rows and checked boxes)."""
        return len(self._exp_checked_ids(list_widget))

    def _sync_exp_checks_from_selection(self, list_widget: QListWidget) -> None:
        """Mirror selection state into checkbox state for clearer visual feedback."""
        list_widget.blockSignals(True)
        for i in range(list_widget.count()):
            item = list_widget.item(i)
            if item.flags() & Qt.ItemFlag.ItemIsUserCheckable:
                item.setCheckState(Qt.CheckState.Checked if item.isSelected() else Qt.CheckState.Unchecked)
        list_widget.blockSignals(False)

    @Slot()
    def _remove_selected_exp_algorithms(self) -> None:
        """Remove selected algorithms from the current experiment selection."""
        removed = self._clear_checked_exp_items(self.exp_algorithm_list)
        self.exp_algo_counter.setText(f"{self._count_checked_exp_items(self.exp_algorithm_list)} selected")
        self._refresh_exp_operator_cfg_targets()
        self._refresh_exp_selected_algorithms_summary()
        if removed > 0:
            self._exp_append_log(f"Removed {removed} selected algorithm(s) from Experiment selection.")
        else:
            self._exp_append_log("No selected algorithm to remove.")

    @Slot()
    def _remove_selected_exp_problems(self) -> None:
        """Remove selected problems from the current experiment selection."""
        removed = self._clear_checked_exp_items(self.exp_problem_list)
        self.exp_prob_counter.setText(f"{self._count_checked_exp_items(self.exp_problem_list)} selected")
        self._refresh_exp_problem_cfg_targets()
        self._refresh_exp_workload_summary()
        if removed > 0:
            self._exp_append_log(f"Removed {removed} selected problem(s) from Experiment selection.")
        else:
            self._exp_append_log("No selected problem to remove.")

    @Slot(QListWidgetItem)
    def _on_exp_algo_check_changed(self, item: QListWidgetItem) -> None:
        """Update selected algorithm counter in experiment module."""
        _ = item
        if item.checkState() != Qt.CheckState.Checked and item.isSelected():
            self.exp_algorithm_list.blockSignals(True)
            item.setSelected(False)
            self.exp_algorithm_list.blockSignals(False)
        checked = self._count_checked_exp_items(self.exp_algorithm_list)
        self.exp_algo_counter.setText(f"{checked} selected")
        self._refresh_exp_operator_cfg_targets()
        self._refresh_exp_selected_algorithms_summary()

    @Slot(QListWidgetItem)
    def _on_exp_prob_check_changed(self, item: QListWidgetItem) -> None:
        """Update selected problem counter in experiment module."""
        _ = item
        if item.checkState() != Qt.CheckState.Checked and item.isSelected():
            self.exp_problem_list.blockSignals(True)
            item.setSelected(False)
            self.exp_problem_list.blockSignals(False)
        checked = self._count_checked_exp_items(self.exp_problem_list)
        self.exp_prob_counter.setText(f"{checked} selected")
        self._refresh_exp_problem_cfg_targets()
        self._refresh_exp_workload_summary()

    @Slot(QListWidgetItem)
    def _on_exp_metric_check_changed(self, item: QListWidgetItem) -> None:
        """Experiment metrics are single-select by checkbox (one active metric for summary table)."""
        if not hasattr(self, "exp_metric_list"):
            return
        self.exp_metric_list.blockSignals(True)
        try:
            if item.checkState() == Qt.CheckState.Checked:
                for i in range(self.exp_metric_list.count()):
                    other = self.exp_metric_list.item(i)
                    if other is item:
                        continue
                    if other.checkState() == Qt.CheckState.Checked:
                        other.setCheckState(Qt.CheckState.Unchecked)
            else:
                has_any_checked = any(
                    self.exp_metric_list.item(i).checkState() == Qt.CheckState.Checked
                    for i in range(self.exp_metric_list.count())
                )
                if not has_any_checked:
                    item.setCheckState(Qt.CheckState.Checked)
        finally:
            self.exp_metric_list.blockSignals(False)
        self._refresh_exp_workload_summary()
        self._recompute_exp_metric_samples_from_results()

    def _exp_selected_metric_spec_for_table(self) -> MetricSpec | None:
        if not hasattr(self, "exp_metric_list"):
            return None
        metric_ids = self._exp_checked_ids(self.exp_metric_list)
        metric_ids = self._map_metric_ids_to_backend(
            metric_ids[:1],
            prefer_jax=self._exp_backend_prefers_jax_for("metrics"),
        )
        if not metric_ids:
            return None
        return self.metric_specs.get(str(metric_ids[0]))

    @staticmethod
    def _exp_run_payload_matrix(
        run_payload: dict[str, Any],
        field: str,
        *,
        vector_as_row: bool = True,
        dtype: Any = float,
    ) -> np.ndarray | None:
        raw_pop = run_payload.get("final_population")
        if not isinstance(raw_pop, dict):
            return None
        raw = raw_pop.get(field)
        if raw is None:
            return None
        try:
            matrix = np.asarray(to_numpy(raw), dtype=dtype)
        except Exception:  # noqa: BLE001
            return None
        if matrix.ndim == 0:
            matrix = matrix.reshape(1, 1)
        elif matrix.ndim == 1:
            matrix = matrix.reshape(1, -1) if vector_as_row else matrix.reshape(-1, 1)
        return matrix

    def _exp_metric_runtime_context_and_fn(
        self,
        metric_spec: MetricSpec,
        run_payload: dict[str, Any],
    ) -> tuple[dict[str, Any], Callable[[np.ndarray], float]] | None:
        problem_instance_id = str(run_payload.get("problem_id", "")).strip()
        if not problem_instance_id:
            return None
        cache_key = (problem_instance_id, metric_spec.id)
        cached = self.exp_metric_runtime_cache.get(cache_key)
        if cached is not None:
            return cached

        base_problem_id = str(
            run_payload.get("base_problem_id", run_payload.get("problem_id", ""))
        ).strip()
        problem_spec = self.problem_specs.get(base_problem_id)
        if problem_spec is None:
            return None

        try:
            n_obj = max(1, int(run_payload.get("n_obj", problem_spec.default_n_obj)))
        except Exception:  # noqa: BLE001
            n_obj = int(problem_spec.default_n_obj)
        try:
            n_var = max(1, int(run_payload.get("n_var", problem_spec.default_n_var)))
        except Exception:  # noqa: BLE001
            n_var = int(problem_spec.default_n_var)

        pop_F = self._exp_run_payload_matrix(run_payload, "F")
        pop_size_guess = int(pop_F.shape[0]) if isinstance(pop_F, np.ndarray) and pop_F.ndim == 2 else 100
        runtime_cfg_problem = {
            "n_obj": int(n_obj),
            "n_var": int(n_var),
            "pop_size": int(pop_size_guess),
            "use_gpu": False,
            "array_backend": "numpy",
            "gpu_dtype": "float32",
        }
        try:
            probe_problem = problem_spec.factory(runtime_cfg_problem)
        except Exception:
            probe_problem = None

        ref_dirs = build_reference_dirs(n_obj=n_obj, target=max(12, pop_size_guess))
        pareto_front = ExperimentBridge._coerce_front_matrix(
            run_payload.get("reference_front"),
            n_obj_hint=n_obj,
        )
        pareto_set = None
        problem_name = str(run_payload.get("problem_name", problem_spec.name)).lower()
        if pareto_front is None and probe_problem is not None:
            try:
                pareto_front = ExperimentBridge._resolve_pareto_front(
                    probe_problem,
                    problem_name,
                    ref_dirs,
                    n_obj_hint=n_obj,
                )
                if pareto_front is not None and pareto_front.size == 0:
                    pareto_front = None
            except Exception:  # noqa: BLE001
                pareto_front = None
            try:
                pareto_set = ExperimentBridge._resolve_pareto_set(probe_problem, problem_name, ref_dirs)
                if pareto_set is not None and pareto_set.size == 0:
                    pareto_set = None
            except Exception:  # noqa: BLE001
                pareto_set = None

        if pareto_front is None:
            pf_cached = self.exp_pareto_fronts.get(problem_instance_id)
            if pf_cached is not None:
                try:
                    pf_arr = np.asarray(to_numpy(pf_cached), dtype=float)
                    pareto_front = pf_arr if pf_arr.ndim == 2 and pf_arr.size else None
                except Exception:  # noqa: BLE001
                    pareto_front = None

        if pareto_front is not None and pareto_front.ndim == 2 and pareto_front.shape[1] == n_obj:
            ref_point = np.max(pareto_front, axis=0) * 1.1
        else:
            ref_point = np.asarray([1.1] * int(n_obj), dtype=float)

        backend_code = str(run_payload.get("backend_code", "cpu")).lower()
        use_gpu = backend_code in {"gpu", "mlx"}
        context: dict[str, Any] = {
            "pareto_front": pareto_front,
            "pareto_set": pareto_set,
            "ref_point": ref_point,
            "ref_dirs": ref_dirs,
            "compute_backend_requested": backend_code,
            "backend_aware_enabled": bool(core_backend_aware_loading_enabled()),
            "backend_rollout_stage": str(core_rollout_stage()),
            "backend_rollout": {
                "problems": bool(core_rollout_allows_domain("problems")),
                "operators": bool(core_rollout_allows_domain("operators")),
                "metrics": bool(core_rollout_allows_domain("metrics")),
            },
            "backend_code": backend_code,
            "backend_label": str(run_payload.get("backend", "CPU")),
            "array_backend": "jax" if backend_code == "gpu" else ("mlx" if backend_code == "mlx" else "numpy"),
            "use_gpu": bool(use_gpu),
            "gpu_dtype": "float32",
            "n_obj": int(n_obj),
            "n_var": int(n_var),
            "pop_size": int(pop_size_guess),
            "problem_name": problem_name,
            "problem": probe_problem,
            "hv_mc_samples": 10000,
            "hv_mc_exclusive": 1,
            "current_population_F": None,
            "current_population_X": None,
            "current_population_G": None,
            "current_population_H": None,
            "current_population_CV": None,
            "current_population_feasible": None,
        }
        try:
            metric_fn = metric_spec.factory(context)
        except Exception:
            return None
        self.exp_metric_runtime_cache[cache_key] = (context, metric_fn)
        return context, metric_fn

    def _exp_compute_metric_for_run(self, metric_spec: MetricSpec, run_payload: dict[str, Any]) -> float:
        existing_metrics = run_payload.get("metrics")
        if isinstance(existing_metrics, dict) and metric_spec.label in existing_metrics:
            try:
                return float(existing_metrics[metric_spec.label])
            except Exception:  # noqa: BLE001
                pass

        ctx_and_fn = self._exp_metric_runtime_context_and_fn(metric_spec, run_payload)
        if ctx_and_fn is None:
            return float("nan")
        metric_context, metric_fn = ctx_and_fn
        front_raw = run_payload.get("final_front", [])
        try:
            front = np.asarray(to_numpy(front_raw), dtype=float)
        except Exception:  # noqa: BLE001
            return float("nan")
        if front.ndim == 1:
            front = front.reshape(1, -1)
        if front.ndim != 2 or front.size == 0:
            return float("nan")

        metric_context["current_population_F"] = self._exp_run_payload_matrix(run_payload, "F")
        if metric_context["current_population_F"] is None:
            metric_context["current_population_F"] = front
        metric_context["current_population_X"] = self._exp_run_payload_matrix(run_payload, "X")
        metric_context["current_population_G"] = self._exp_run_payload_matrix(
            run_payload, "G", vector_as_row=False
        )
        metric_context["current_population_H"] = self._exp_run_payload_matrix(
            run_payload, "H", vector_as_row=False
        )
        metric_context["current_population_CV"] = self._exp_run_payload_matrix(
            run_payload, "CV", vector_as_row=False
        )
        metric_context["current_population_feasible"] = self._exp_run_payload_matrix(
            run_payload, "feasible", vector_as_row=False, dtype=bool
        )
        try:
            return float(metric_fn(front))
        except Exception:  # noqa: BLE001
            return float("nan")

    def _recompute_exp_metric_samples_from_results(self) -> None:
        """Recompute summary metric samples from stored raw runs (no re-execution)."""
        metric_spec = self._exp_selected_metric_spec_for_table()
        self.exp_run_progress_metric_samples.clear()
        self.exp_run_progress_counts.clear()
        self.exp_metric_runtime_cache.clear()
        self.exp_run_progress_metric_order = [metric_spec.label] if metric_spec is not None else []

        for runs in self.exp_results.values():
            if not isinstance(runs, list):
                continue
            for raw_run in runs:
                if not isinstance(raw_run, dict):
                    continue
                problem_id = str(raw_run.get("problem_id", "")).strip()
                algorithm_id = str(raw_run.get("algorithm_id", "")).strip()
                if not problem_id or not algorithm_id:
                    continue
                key = (problem_id, algorithm_id)
                self.exp_run_progress_counts[key] = int(self.exp_run_progress_counts.get(key, 0)) + 1
                if metric_spec is None:
                    continue
                value = self._exp_compute_metric_for_run(metric_spec, raw_run)
                self.exp_run_progress_metric_samples.setdefault(key, {}).setdefault(metric_spec.label, []).append(value)

        self._refresh_exp_run_progress_table_display()

    def _refresh_exp_run_progress_table_display(self) -> None:
        if not hasattr(self, "exp_run_progress_table"):
            return
        table = self.exp_run_progress_table
        if table.rowCount() <= 0 or table.columnCount() <= 0:
            return
        updates_enabled = table.updatesEnabled()
        table.setUpdatesEnabled(False)
        try:
            for meta in self.exp_run_progress_row_meta:
                row = int(meta.get("row", -1))
                if row < 0:
                    continue
                problem_id = str(meta.get("instance_id", ""))
                run_item = table.item(row, 3)
                if run_item is None:
                    run_item = self._make_table_item(
                        self._exp_problem_runs_cell_text(problem_id),
                        align=Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
                    )
                    table.setItem(row, 3, run_item)
                run_item.setText(self._exp_problem_runs_cell_text(problem_id))
                run_item.setTextAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
                run_item.setToolTip(self._exp_problem_runs_cell_tooltip(problem_id))
                try:
                    active_algo = self.exp_run_progress_active_algo_by_problem.get(problem_id)
                    if not active_algo and self.exp_run_progress_algorithm_order:
                        active_algo = self.exp_run_progress_algorithm_order[0]
                    active_count = int(
                        self.exp_run_progress_counts.get((problem_id, str(active_algo or "")), 0)
                    )
                    row_ratio = max(0.0, min(1.0, active_count / max(1, int(self.exp_run_progress_n_runs))))
                    gray_base = QColor(getattr(AppStyles, "PANEL_ALT", "#E5E7EB"))
                    alpha_row = 30 + int(60 * row_ratio)
                    run_item.setBackground(QBrush(QColor(gray_base.red(), gray_base.green(), gray_base.blue(), alpha_row)))
                except Exception:  # noqa: BLE001
                    pass
                for algorithm_id in self.exp_run_progress_algorithm_order:
                    col = self.exp_run_progress_col_by_algorithm.get(algorithm_id)
                    if col is None:
                        continue
                    item = table.item(row, col)
                    if item is None:
                        item = self._make_table_item(
                            self._exp_progress_cell_text(problem_id, algorithm_id),
                            align=Qt.AlignmentFlag.AlignCenter,
                        )
                        table.setItem(row, col, item)
                    key = (problem_id, algorithm_id)
                    count = int(self.exp_run_progress_counts.get(key, 0))
                    cell_text = self._exp_progress_cell_text(problem_id, algorithm_id)
                    item.setText(cell_text)
                    item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                    active_metric_label = self._exp_progress_active_metric_label()
                    metric_brief = self._metric_label_brief(active_metric_label) if active_metric_label else "Metric"
                    item.setToolTip(
                        f"{metric_brief}: {cell_text}\n{count} / {max(1, self.exp_run_progress_n_runs)} completed"
                    )
                    try:
                        ratio = max(0.0, min(1.0, count / max(1, self.exp_run_progress_n_runs)))
                        gray_base = QColor(getattr(AppStyles, "PANEL_ALT", "#E5E7EB"))
                        alpha = 35 + int(70 * ratio)
                        item.setBackground(QBrush(QColor(gray_base.red(), gray_base.green(), gray_base.blue(), alpha)))
                    except Exception:  # noqa: BLE001
                        pass
                self._apply_exp_row_winner_bold(row, problem_id)
        finally:
            table.setUpdatesEnabled(updates_enabled)

    @Slot()
    def _on_exp_algo_selection_changed(self) -> None:
        self._sync_exp_checks_from_selection(self.exp_algorithm_list)
        self.exp_algo_counter.setText(f"{self._count_checked_exp_items(self.exp_algorithm_list)} selected")
        self._refresh_exp_operator_cfg_targets()
        self._refresh_exp_selected_algorithms_summary()
        self._populate_all_operator_combos()

    @Slot()
    def _on_exp_prob_selection_changed(self) -> None:
        self._sync_exp_checks_from_selection(self.exp_problem_list)
        self.exp_prob_counter.setText(f"{self._count_checked_exp_items(self.exp_problem_list)} selected")
        self._refresh_exp_problem_cfg_targets()
        self._refresh_exp_workload_summary()

    @Slot(QListWidgetItem, QListWidgetItem)
    def _on_exp_problem_current_changed(
        self,
        current: QListWidgetItem | None,
        _previous: QListWidgetItem | None,
    ) -> None:
        if current is None:
            return
        problem_id = current.data(Qt.ItemDataRole.UserRole)
        if not isinstance(problem_id, str):
            return
        idx = self.exp_problem_cfg_combo.findData(problem_id)
        if idx >= 0 and idx != self.exp_problem_cfg_combo.currentIndex():
            self.exp_problem_cfg_combo.setCurrentIndex(idx)

    def _exp_problem_target_base_id(self, problem_key: str | None) -> str | None:
        if not isinstance(problem_key, str):
            return None
        if problem_key in self.problem_specs:
            return problem_key
        entry = self.exp_problem_variants.get(problem_key)
        if isinstance(entry, dict):
            base_id = str(entry.get("problem_id", "")).strip()
            if base_id in self.problem_specs:
                return base_id
        return None

    def _exp_problem_target_label(self, problem_key: str | None) -> str:
        if not isinstance(problem_key, str):
            return ""
        if problem_key in self.problem_specs:
            spec = self.problem_specs.get(problem_key)
            return spec.label if spec is not None else problem_key
        entry = self.exp_problem_variants.get(problem_key)
        if isinstance(entry, dict):
            label = str(entry.get("label", "")).strip()
            if label:
                return label
            base_id = str(entry.get("problem_id", "")).strip()
            if base_id in self.problem_specs:
                spec = self.problem_specs[base_id]
                return f"{spec.label} (variant)"
        return problem_key

    def _exp_problem_base_values(self, problem_id: str | None = None) -> dict[str, int]:
        values = {
            "pop_size": int(self.exp_pop_size_spin.value()),
            "n_obj": int(self.exp_n_obj_spin.value()),
            "n_var": int(self.exp_n_var_spin.value()),
        }
        base_problem_id = self._exp_problem_target_base_id(problem_id)
        if isinstance(base_problem_id, str):
            spec = self.problem_specs.get(base_problem_id)
            if spec is not None:
                # M/D are always resolved per problem; use catalog defaults as baseline.
                values["n_obj"] = max(1, int(getattr(spec, "default_n_obj", values["n_obj"])))
                values["n_var"] = max(1, int(getattr(spec, "default_n_var", values["n_var"])))
        return values

    def _exp_problem_effective_values(self, problem_id: str) -> dict[str, int]:
        values = self._exp_problem_base_values(problem_id)
        override = self.exp_problem_overrides.get(problem_id, {})
        if isinstance(override, dict):
            for key in ("pop_size", "n_obj", "n_var"):
                if key in override:
                    try:
                        values[key] = int(override[key])
                    except Exception:  # noqa: BLE001
                        pass
        return values

    def _set_exp_problem_cfg_enabled(self, enabled: bool) -> None:
        for widget in (
            self.exp_problem_cfg_combo,
            self.exp_problem_pop_size_spin,
            self.exp_problem_n_obj_spin,
            self.exp_problem_n_var_spin,
            self.exp_problem_cfg_reset_btn,
            getattr(self, "exp_problem_cfg_duplicate_btn", None),
            getattr(self, "exp_problem_cfg_remove_variant_btn", None),
        ):
            if widget is not None:
                widget.setEnabled(enabled)

    def _load_exp_problem_cfg_for(self, problem_id: str | None) -> None:
        if not isinstance(problem_id, str):
            return
        values = self._exp_problem_effective_values(problem_id)
        self._exp_problem_override_loading = True
        self.exp_problem_pop_size_spin.setValue(values["pop_size"])
        self.exp_problem_n_obj_spin.setValue(values["n_obj"])
        self.exp_problem_n_var_spin.setValue(values["n_var"])

        is_fixed_obj = False
        base_id = self._exp_problem_target_base_id(problem_id)
        if isinstance(base_id, str) and base_id in self.problem_specs:
            spec = self.problem_specs[base_id]
            name_lower = spec.name.lower()
            is_fixed_obj = name_lower.startswith("zdt") or name_lower.startswith("imop")

        self.exp_problem_n_obj_spin.setEnabled(not is_fixed_obj)
        self.exp_problem_n_var_spin.setEnabled(not is_fixed_obj)
        if is_fixed_obj:
            self.exp_problem_n_obj_spin.setToolTip("The number of objectives is fixed and cannot be edited for this problem.")
            self.exp_problem_n_var_spin.setToolTip("The number of variables is fixed and cannot be edited for this problem.")
        else:
            self.exp_problem_n_obj_spin.setToolTip("Number of objectives (M).")
            self.exp_problem_n_var_spin.setToolTip("Number of variables (D).")

        self._exp_problem_override_loading = False
        if hasattr(self, "exp_problem_cfg_remove_variant_btn"):
            self.exp_problem_cfg_remove_variant_btn.setEnabled(problem_id in self.exp_problem_variants)

    def _refresh_exp_problem_cfg_targets(self) -> None:
        selected_problem_ids = self._exp_checked_ids(self.exp_problem_list)
        selected_set = set(selected_problem_ids)
        current_id = self.exp_problem_cfg_combo.currentData()

        self.exp_problem_cfg_combo.blockSignals(True)
        self.exp_problem_cfg_combo.clear()
        for i in range(self.exp_problem_list.count()):
            item = self.exp_problem_list.item(i)
            problem_id = item.data(Qt.ItemDataRole.UserRole)
            if not isinstance(problem_id, str) or problem_id not in selected_set:
                continue
            self.exp_problem_cfg_combo.addItem(self._exp_problem_target_label(problem_id), problem_id)
            for variant_key, variant_entry in self.exp_problem_variants.items():
                if not isinstance(variant_entry, dict):
                    continue
                if str(variant_entry.get("problem_id", "")).strip() != problem_id:
                    continue
                self.exp_problem_cfg_combo.addItem(self._exp_problem_target_label(variant_key), variant_key)

        next_index = -1
        if self.exp_problem_cfg_combo.count() > 0:
            if isinstance(current_id, str):
                next_index = self.exp_problem_cfg_combo.findData(current_id)
            if next_index < 0:
                current_item = self.exp_problem_list.currentItem()
                if current_item is not None:
                    cid = current_item.data(Qt.ItemDataRole.UserRole)
                    if isinstance(cid, str):
                        next_index = self.exp_problem_cfg_combo.findData(cid)
            if next_index < 0:
                next_index = 0
            self.exp_problem_cfg_combo.setCurrentIndex(next_index)
        self.exp_problem_cfg_combo.blockSignals(False)

        has_targets = self.exp_problem_cfg_combo.count() > 0
        self._set_exp_problem_cfg_enabled(has_targets)
        if has_targets:
            self._load_exp_problem_cfg_for(self.exp_problem_cfg_combo.currentData())
        else:
            self._exp_problem_override_loading = True
            self.exp_problem_pop_size_spin.setValue(self.exp_pop_size_spin.value())
            self.exp_problem_n_obj_spin.setValue(self.exp_n_obj_spin.value())
            self.exp_problem_n_var_spin.setValue(self.exp_n_var_spin.value())
            self.exp_problem_n_var_spin.setEnabled(False)
            self._exp_problem_override_loading = False
            if hasattr(self, "exp_problem_cfg_remove_variant_btn"):
                self.exp_problem_cfg_remove_variant_btn.setEnabled(False)

    def _store_exp_problem_override_for_current(self) -> None:
        if self._exp_problem_override_loading:
            return
        problem_id = self.exp_problem_cfg_combo.currentData()
        if not isinstance(problem_id, str):
            return

        values = {
            "pop_size": int(self.exp_problem_pop_size_spin.value()),
            "n_obj": int(self.exp_problem_n_obj_spin.value()),
            "n_var": int(self.exp_problem_n_var_spin.value()),
        }

        base = self._exp_problem_base_values(problem_id)
        diff: dict[str, int] = {}
        for key in ("pop_size", "n_obj", "n_var"):
            if int(values[key]) != int(base[key]):
                diff[key] = int(values[key])

        if diff:
            self.exp_problem_overrides[problem_id] = diff
        else:
            self.exp_problem_overrides.pop(problem_id, None)

    @Slot()
    def _on_exp_problem_cfg_target_changed(self) -> None:
        problem_id = self.exp_problem_cfg_combo.currentData()
        self._load_exp_problem_cfg_for(problem_id)

    @Slot()
    def _on_exp_problem_override_value_changed(self) -> None:
        self._store_exp_problem_override_for_current()

    @Slot()
    def _on_exp_problem_override_n_obj_changed(self) -> None:
        if self._exp_problem_override_loading:
            return
        self._store_exp_problem_override_for_current()

    @Slot()
    def _reset_exp_problem_override(self) -> None:
        problem_id = self.exp_problem_cfg_combo.currentData()
        if not isinstance(problem_id, str):
            return
        self.exp_problem_overrides.pop(problem_id, None)
        self._load_exp_problem_cfg_for(problem_id)
        self._exp_append_log(f"Reset per-problem settings for '{problem_id}'.")

    @Slot()
    def _duplicate_exp_problem_variant(self) -> None:
        current_key = self.exp_problem_cfg_combo.currentData()
        base_problem_id = self._exp_problem_target_base_id(current_key)
        if not isinstance(base_problem_id, str):
            return
        spec = self.problem_specs.get(base_problem_id)
        if spec is None:
            return

        self._store_exp_problem_override_for_current()
        self._exp_problem_variant_counter += 1
        variant_idx = self._exp_problem_variant_counter
        variant_key = f"{base_problem_id}::v{variant_idx}"
        while variant_key in self.problem_specs or variant_key in self.exp_problem_variants:
            self._exp_problem_variant_counter += 1
            variant_idx = self._exp_problem_variant_counter
            variant_key = f"{base_problem_id}::v{variant_idx}"

        variant_label = f"{spec.label} [variant {variant_idx}]"
        self.exp_problem_variants[variant_key] = {
            "problem_id": base_problem_id,
            "label": variant_label,
        }
        current_values = self._exp_problem_effective_values(str(current_key))
        base_values = self._exp_problem_base_values(variant_key)
        diff: dict[str, int] = {}
        for key in ("pop_size", "n_obj", "n_var"):
            if int(current_values[key]) != int(base_values[key]):
                diff[key] = int(current_values[key])
        if diff:
            self.exp_problem_overrides[variant_key] = diff

        self._refresh_exp_problem_cfg_targets()
        idx = self.exp_problem_cfg_combo.findData(variant_key)
        if idx >= 0:
            self.exp_problem_cfg_combo.setCurrentIndex(idx)
        self._exp_append_log(f"Added problem variant: {variant_label}")

    @Slot()
    def _remove_exp_problem_variant(self) -> None:
        current_key = self.exp_problem_cfg_combo.currentData()
        if not isinstance(current_key, str) or current_key not in self.exp_problem_variants:
            return
        label = self._exp_problem_target_label(current_key) or current_key
        self.exp_problem_variants.pop(current_key, None)
        self.exp_problem_overrides.pop(current_key, None)
        self._refresh_exp_problem_cfg_targets()
        self._exp_append_log(f"Removed problem variant: {label}")

    @Slot()
    def _on_exp_global_problem_defaults_changed(self) -> None:
        problem_id = self.exp_problem_cfg_combo.currentData()
        if not isinstance(problem_id, str):
            return
        if problem_id not in self.exp_problem_overrides:
            self._load_exp_problem_cfg_for(problem_id)

    def _exp_algorithm_operator_base_values(self) -> dict[str, Any]:
        return {
            "crossover": "default",
            "mutation": "default",
            "selection": "default",
            "sampling": "default",
            "crossover_eta": 15.0,
            "crossover_prob": 0.9,
            "mutation_eta": 20.0,
            "mutation_prob": 0.0,
            "selection_pressure": 2,
        }

    def _exp_algorithm_operator_effective_values(self, algorithm_id: str) -> dict[str, Any]:
        values = dict(self._exp_algorithm_operator_base_values())
        override = self.exp_algorithm_operator_overrides.get(algorithm_id, {})
        if isinstance(override, dict):
            for key, default_value in values.items():
                if key not in override:
                    continue
                try:
                    if isinstance(default_value, str):
                        values[key] = str(override[key])
                    elif isinstance(default_value, int):
                        values[key] = int(override[key])
                    else:
                        values[key] = float(override[key])
                except Exception:  # noqa: BLE001
                    continue
        return values

    def _set_exp_algorithm_operator_cfg_enabled(self, enabled: bool) -> None:
        for widget in (
            getattr(self, "exp_operator_algo_combo", None),
            getattr(self, "exp_crossover_combo", None),
            getattr(self, "exp_mutation_combo", None),
            getattr(self, "exp_selection_combo", None),
            getattr(self, "exp_sampling_combo", None),
            getattr(self, "exp_crossover_eta_spin", None),
            getattr(self, "exp_crossover_prob_spin", None),
            getattr(self, "exp_mutation_eta_spin", None),
            getattr(self, "exp_mutation_prob_spin", None),
            getattr(self, "exp_selection_pressure_spin", None),
        ):
            if widget is not None:
                widget.setEnabled(enabled)

    def _load_exp_algorithm_operator_cfg_for(self, algorithm_id: str | None) -> None:
        if not isinstance(algorithm_id, str):
            return
        values = self._exp_algorithm_operator_effective_values(algorithm_id)
        self._exp_algorithm_operator_loading = True

        for combo, key in [
            (self.exp_crossover_combo, "crossover"),
            (self.exp_mutation_combo, "mutation"),
            (self.exp_selection_combo, "selection"),
            (self.exp_sampling_combo, "sampling"),
        ]:
            wanted = self._normalize_operator_value(key, values[key])
            idx = combo.findData(wanted)
            if idx < 0:
                idx = combo.findData("default")
            if idx >= 0:
                combo.setCurrentIndex(idx)

        self.exp_crossover_eta_spin.setValue(float(values["crossover_eta"]))
        self.exp_crossover_prob_spin.setValue(float(values["crossover_prob"]))
        self.exp_mutation_eta_spin.setValue(float(values["mutation_eta"]))
        self.exp_mutation_prob_spin.setValue(float(values["mutation_prob"]))
        self.exp_selection_pressure_spin.setValue(int(values["selection_pressure"]))

        self._exp_algorithm_operator_loading = False
        self._on_exp_crossover_changed()
        self._on_exp_mutation_changed()
        self._on_exp_selection_changed()

    def _refresh_exp_operator_cfg_targets(self) -> None:
        if not hasattr(self, "exp_operator_algo_combo") or not hasattr(self, "exp_algorithm_list"):
            return
        selected_algorithm_ids = self._exp_checked_ids(self.exp_algorithm_list)
        selected_set = set(selected_algorithm_ids)
        current_id = self.exp_operator_algo_combo.currentData()

        # Prune overrides for removed algorithms.
        self.exp_algorithm_operator_overrides = {
            aid: dict(values)
            for aid, values in self.exp_algorithm_operator_overrides.items()
            if aid in self.algorithm_specs and aid in selected_set and isinstance(values, dict)
        }

        self.exp_operator_algo_combo.blockSignals(True)
        self.exp_operator_algo_combo.clear()
        for i in range(self.exp_algorithm_list.count()):
            item = self.exp_algorithm_list.item(i)
            algorithm_id = item.data(Qt.ItemDataRole.UserRole)
            if not isinstance(algorithm_id, str) or algorithm_id not in selected_set:
                continue
            spec = self.algorithm_specs.get(algorithm_id)
            label = spec.label if spec is not None else item.text()
            self.exp_operator_algo_combo.addItem(label, algorithm_id)

        next_index = -1
        if self.exp_operator_algo_combo.count() > 0:
            if isinstance(current_id, str):
                next_index = self.exp_operator_algo_combo.findData(current_id)
            if next_index < 0:
                current_item = self.exp_algorithm_list.currentItem()
                if current_item is not None:
                    cid = current_item.data(Qt.ItemDataRole.UserRole)
                    if isinstance(cid, str):
                        next_index = self.exp_operator_algo_combo.findData(cid)
            if next_index < 0:
                next_index = 0
            self.exp_operator_algo_combo.setCurrentIndex(next_index)
        self.exp_operator_algo_combo.blockSignals(False)

        has_targets = self.exp_operator_algo_combo.count() > 0
        self._set_exp_algorithm_operator_cfg_enabled(has_targets)
        if has_targets:
            self._load_exp_algorithm_operator_cfg_for(self.exp_operator_algo_combo.currentData())

    def _store_exp_algorithm_operator_override_for_current(self) -> None:
        if self._exp_algorithm_operator_loading:
            return
        algorithm_id = self.exp_operator_algo_combo.currentData() if hasattr(self, "exp_operator_algo_combo") else None
        if not isinstance(algorithm_id, str):
            return

        values: dict[str, Any] = {
            "crossover": str(self.exp_crossover_combo.currentData() or "default"),
            "mutation": str(self.exp_mutation_combo.currentData() or "default"),
            "selection": str(self.exp_selection_combo.currentData() or "default"),
            "sampling": str(self.exp_sampling_combo.currentData() or "default"),
            "crossover_eta": float(self.exp_crossover_eta_spin.value()),
            "crossover_prob": float(self.exp_crossover_prob_spin.value()),
            "mutation_eta": float(self.exp_mutation_eta_spin.value()),
            "mutation_prob": float(self.exp_mutation_prob_spin.value()),
            "selection_pressure": int(self.exp_selection_pressure_spin.value()),
        }
        base = self._exp_algorithm_operator_base_values()
        diff: dict[str, Any] = {}
        for key, base_value in base.items():
            value = values[key]
            if isinstance(base_value, float):
                if abs(float(value) - float(base_value)) > 1e-12:
                    diff[key] = float(value)
            elif value != base_value:
                diff[key] = value

        if diff:
            self.exp_algorithm_operator_overrides[algorithm_id] = diff
        else:
            self.exp_algorithm_operator_overrides.pop(algorithm_id, None)

    @Slot()
    def _on_exp_operator_target_changed(self) -> None:
        algorithm_id = self.exp_operator_algo_combo.currentData() if hasattr(self, "exp_operator_algo_combo") else None
        self._load_exp_algorithm_operator_cfg_for(algorithm_id)

    @Slot()
    def _on_exp_algorithm_operator_value_changed(self) -> None:
        self._store_exp_algorithm_operator_override_for_current()

    @Slot()
    def _on_exp_crossover_changed(self) -> None:
        """Show/hide experiment crossover param widgets."""
        class_name = self._operator_class_name("crossover", self.exp_crossover_combo.currentData())
        show_sbx = class_name in {"sbx", "simulatedbinarycrossover"}
        self.exp_crossover_eta_label.setVisible(show_sbx)
        self.exp_crossover_eta_spin.setVisible(show_sbx)
        self.exp_crossover_prob_label.setVisible(show_sbx)
        self.exp_crossover_prob_spin.setVisible(show_sbx)

    @Slot()
    def _on_exp_mutation_changed(self) -> None:
        """Show/hide experiment mutation param widgets."""
        class_name = self._operator_class_name("mutation", self.exp_mutation_combo.currentData())
        show_pm = class_name in {"pm", "polynomialmutation"}
        show_prob = class_name in {
            "pm",
            "polynomialmutation",
            "bfm",
            "bitflipmutation",
            "gaussianmutation",
            "gm",
        }
        self.exp_mutation_eta_label.setVisible(show_pm)
        self.exp_mutation_eta_spin.setVisible(show_pm)
        self.exp_mutation_prob_label.setVisible(show_prob)
        self.exp_mutation_prob_spin.setVisible(show_prob)

    @Slot()
    def _on_exp_selection_changed(self) -> None:
        """Show/hide experiment selection param widgets."""
        class_name = self._operator_class_name("selection", self.exp_selection_combo.currentData())
        show_pressure = class_name == "tournamentselection"
        self.exp_selection_pressure_label.setVisible(show_pressure)
        self.exp_selection_pressure_spin.setVisible(show_pressure)

    def _exp_append_log(self, text: str) -> None:
        """Append timestamped message to the experiment log."""
        if not getattr(self, "exp_log_box", None):
            return
        timestamp = format_timestamp_en_us()
        self.exp_log_box.appendPlainText(f"[{timestamp}] {text}")

    @staticmethod
    def _sanitize_exp_status_message(text: str) -> str:
        value = str(text)
        replacements = {
            "\u00c3\u00a2\u00e2\u201a\u00ac\u00e2\u20ac\u009d": "-",
            "\u00c3\u00a2\u00e2\u201a\u00ac\u00e2\u20ac\u0153": "-",
            "\u00e2\u20ac\u201d": "-",
            "\u00e2\u20ac\u201c": "-",
            "\u00c3\u00a2\u00e2\u201a\u00ac\u00c2\u00a6": "...",
            "\u00e2\u20ac\u00a6": "...",
            "\u00c3\u0192\u00e2\u20ac\u201d": "x",
            "\u00c4\u201a\u00e2\u20ac\u201d": "x",
            " x ": " x ",
        }
        for old, new in replacements.items():
            value = value.replace(old, new)
        value = re.sub(r"algo\(s\)", "algorithm(s)", value, flags=re.IGNORECASE)
        value = re.sub(r"prob\(s\)", "problem(s)", value, flags=re.IGNORECASE)
        return value

    def _normalize_exp_results_storage_path(self, raw_path: str | Path | None) -> Path:
        text = str(raw_path or "").strip()
        if not text:
            return Path(self.base_dir) / "experiment_module_results" / "latest_results.json"
        path = Path(text).expanduser()
        if not path.is_absolute():
            path = (Path(self.base_dir) / path).resolve()
        if not str(path.suffix).strip():
            path = path.with_suffix(".json")
        return path

    def _refresh_exp_results_storage_info(self, note: str | None = None) -> None:
        if not hasattr(self, "exp_results_storage_info_label"):
            return
        save_path = self._normalize_exp_results_storage_path(
            getattr(self, "exp_results_storage_path", None)
        )
        self.exp_results_storage_path = save_path
        if hasattr(self, "exp_results_path_edit"):
            current_text = self.exp_results_path_edit.text().strip()
            normalized_text = str(save_path)
            if current_text != normalized_text:
                self.exp_results_path_edit.blockSignals(True)
                self.exp_results_path_edit.setText(normalized_text)
                self.exp_results_path_edit.blockSignals(False)
        source_path = self.exp_results_storage_loaded_path
        if source_path is not None:
            source_text = f"Metric recalculation source: loaded file ({source_path.name})."
        elif self.exp_results:
            source_text = "Metric recalculation source: latest in-memory experiment results."
        else:
            source_text = "Metric recalculation source: no results loaded yet."
        lines = [
            "Raw experiment payloads are saved here and reused when you change the summary metric.",
            f"Output file: {save_path}",
            source_text,
        ]
        if self.exp_results_storage_last_saved_path is not None:
            lines.append(f"Last saved: {self.exp_results_storage_last_saved_path}")
        if note:
            lines.append(str(note))
        self.exp_results_storage_info_label.setText("\n".join(lines))
        self.exp_results_storage_info_label.setToolTip("\n".join(lines))

    def _apply_exp_results_storage_path(self, raw_path: str | Path | None, *, load_if_exists: bool) -> None:
        path = self._normalize_exp_results_storage_path(raw_path)
        self.exp_results_storage_path = path
        self._refresh_exp_results_storage_info(
            "The existing file will be reused for metric recalculation."
            if (load_if_exists and path.exists())
            else "Future experiment results will be saved to this path."
        )
        if load_if_exists and path.exists() and path.is_file():
            self._load_exp_results_from_path(path, interactive=False)

    @staticmethod
    def _json_write_with_numpy(path: Path, data: Any) -> None:
        class _NumpyEncoder(json.JSONEncoder):
            def default(self, obj: Any) -> Any:
                if isinstance(obj, np.integer):
                    return int(obj)
                if isinstance(obj, np.floating):
                    return float(obj)
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                return super().default(obj)

        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(data, indent=2, ensure_ascii=False, cls=_NumpyEncoder),
            encoding="utf-8",
        )

    def _save_exp_results_payload_to_path(self, payload: dict[str, Any], path: Path) -> tuple[bool, str]:
        try:
            now_dt = datetime.now().astimezone()
            wrapper = {
                "kind": "emopylab_experiment_results",
                "schema_version": 1,
                "saved_at_iso": now_dt.isoformat(),
                "saved_at_en_us": format_timestamp_en_us(now_dt),
                "payload": payload,
            }
            self._json_write_with_numpy(path, wrapper)
            self.exp_results_storage_last_saved_path = path
            return True, f"Raw results saved: {path}"
        except Exception as exc:  # noqa: BLE001
            return False, f"Warning: could not save raw experiment results - {exc}"

    def _decode_exp_results_payload_file(self, path: Path) -> dict[str, Any]:
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        if isinstance(data, dict) and isinstance(data.get("payload"), dict):
            payload = dict(data.get("payload", {}))
        elif isinstance(data, dict):
            payload = dict(data)
        else:
            raise ValueError("Invalid JSON structure. Expected an object with experiment payload.")
        if not isinstance(payload.get("results"), dict):
            raise ValueError("The selected file does not contain experiment raw results ('results').")
        return payload

    def _apply_exp_results_payload(
        self,
        payload: dict[str, Any],
        *,
        rebuild_table_from_payload_config: bool,
        loaded_from_path: Path | None = None,
    ) -> None:
        self.exp_results = dict(payload.get("results", {}))
        self.exp_metric_names = list(payload.get("metrics", []))
        self.exp_execution_backend_label = str(
            payload.get("execution_backend_label", self.exp_execution_backend_label)
        )
        self.exp_profile_compare_enabled = bool(payload.get("profile_compare_enabled", False))

        self.exp_pareto_fronts = {}
        pf_map_raw = payload.get("pareto_fronts")
        if isinstance(pf_map_raw, dict):
            for problem_id, pf_raw in pf_map_raw.items():
                key = str(problem_id)
                if pf_raw is None:
                    self.exp_pareto_fronts[key] = None
                else:
                    try:
                        self.exp_pareto_fronts[key] = np.asarray(pf_raw, dtype=float)
                    except Exception:  # noqa: BLE001
                        self.exp_pareto_fronts[key] = None

        pf_raw = payload.get("pareto_front")
        try:
            self.exp_pareto_front = None if pf_raw is None else np.asarray(pf_raw, dtype=float)
        except Exception:  # noqa: BLE001
            self.exp_pareto_front = None
        if self.exp_pareto_front is None:
            for value in self.exp_pareto_fronts.values():
                if value is not None:
                    self.exp_pareto_front = value
                    break

        if rebuild_table_from_payload_config:
            cfg = payload.get("config")
            if isinstance(cfg, dict):
                self._build_exp_run_progress_table(cfg)

        self.exp_metric_runtime_cache.clear()
        self._recompute_exp_metric_samples_from_results()
        self.exp_results_storage_loaded_path = loaded_from_path
        self._refresh_exp_results_storage_info(
            None
            if loaded_from_path is None
            else f"Loaded raw results file for metric reuse: {loaded_from_path.name}"
        )

    def _load_exp_results_from_path(self, path: Path, *, interactive: bool) -> bool:
        if self.exp_worker_thread is not None:
            message = "Cannot load experiment results while an experiment is running."
            if interactive:
                QMessageBox.information(self, "Busy", message)
            self._refresh_exp_results_storage_info(message)
            return False
        try:
            payload = self._decode_exp_results_payload_file(path)
            self._apply_exp_results_payload(
                payload,
                rebuild_table_from_payload_config=True,
                loaded_from_path=path,
            )
            self._sync_payload_to_analysis_workspace(payload)
            self.exp_status_label.setText(
                self._sanitize_exp_status_message(
                    f"Loaded raw experiment results for metric reuse: {path.name}"
                )
            )
            self.exp_status_label.setStyleSheet(f"color: {AppStyles.INFO};")
            self._exp_append_log(f"Experiment raw results loaded: {path}")
            return True
        except Exception as exc:  # noqa: BLE001
            message = f"Could not load raw experiment results:\n{exc}"
            self._exp_append_log(f"Warning: {self._sanitize_exp_status_message(str(exc))}")
            self._refresh_exp_results_storage_info(f"Load failed: {exc}")
            if interactive:
                QMessageBox.warning(self, "Load results", message)
            return False

    @Slot()
    def _on_exp_results_path_editing_finished(self) -> None:
        if not hasattr(self, "exp_results_path_edit"):
            return
        raw_text = self.exp_results_path_edit.text().strip()
        self._apply_exp_results_storage_path(raw_text, load_if_exists=True)

    @Slot()
    def _browse_exp_results_path(self) -> None:
        current = self._normalize_exp_results_storage_path(getattr(self, "exp_results_storage_path", None))
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Select raw experiment results file",
            str(current),
            "JSON files (*.json);;All files (*.*)",
        )
        if not path:
            return
        self._apply_exp_results_storage_path(path, load_if_exists=True)

    @Slot()
    def _load_exp_results_from_current_path(self) -> None:
        path = self._normalize_exp_results_storage_path(
            self.exp_results_path_edit.text() if hasattr(self, "exp_results_path_edit") else self.exp_results_storage_path
        )
        self.exp_results_storage_path = path
        self._refresh_exp_results_storage_info()
        if not path.exists():
            QMessageBox.information(
                self,
                "Load results",
                f"Raw experiment results file does not exist yet:\n{path}",
            )
            return
        self._load_exp_results_from_path(path, interactive=True)

    @Slot()
    def _send_latest_experiment_to_analysis(self) -> None:
        if self.exp_worker_thread is not None:
            QMessageBox.information(
                self,
                "Busy",
                "Wait for the current experiment to finish before sending results to Analysis & MCDM.",
            )
            return

        has_local_results = bool(getattr(self, "exp_results", {}))
        if not has_local_results:
            path = self._normalize_exp_results_storage_path(
                self.exp_results_path_edit.text()
                if hasattr(self, "exp_results_path_edit")
                else getattr(self, "exp_results_storage_path", None)
            )
            self.exp_results_storage_path = path
            self._refresh_exp_results_storage_info()
            if not path.exists():
                QMessageBox.information(
                    self,
                    "Send to Analysis & MCDM",
                    f"No in-memory experiment results are available and the raw results file does not exist yet:\n{path}",
                )
                return
            ok = self._load_exp_results_from_path(path, interactive=True)
            if ok and hasattr(self, "tabs") and hasattr(self, "results_tab"):
                self.tabs.setCurrentWidget(self.results_tab)
            return

        payload = {
            "results": {
                str(algo): [dict(run) for run in runs if isinstance(run, dict)]
                for algo, runs in self.exp_results.items()
                if isinstance(runs, list)
            },
            "metrics": list(getattr(self, "exp_metric_names", []) or []),
            "pareto_front": getattr(self, "exp_pareto_front", None),
            "pareto_fronts": dict(getattr(self, "exp_pareto_fronts", {}) or {}),
            "execution_backend_label": str(getattr(self, "exp_execution_backend_label", "CPU")),
            "profile_compare_enabled": bool(getattr(self, "exp_profile_compare_enabled", False)),
        }
        added_runs = self._sync_payload_to_analysis_workspace(payload)
        self.tabs.setCurrentWidget(self.results_tab)
        if added_runs > 0:
            self.exp_status_label.setText(
                self._sanitize_exp_status_message(
                    f"Sent latest Experiment results to Analysis & MCDM ({added_runs} new run(s))."
                )
            )
            self.exp_status_label.setStyleSheet(f"color: {AppStyles.INFO};")
        else:
            self.exp_status_label.setText(
                self._sanitize_exp_status_message(
                    "Analysis & MCDM is already synchronized with the latest Experiment results."
                )
            )
            self.exp_status_label.setStyleSheet(f"color: {AppStyles.INFO};")

    def _clear_exp_run_progress_table(self) -> None:
        if not hasattr(self, "exp_run_progress_table"):
            return
        self.exp_run_progress_row_by_problem.clear()
        self.exp_run_progress_col_by_algorithm.clear()
        self.exp_run_progress_metric_col_by_key.clear()
        self.exp_run_progress_col_by_metric.clear()
        self.exp_run_progress_counts.clear()
        self.exp_run_progress_active_algo_by_problem.clear()
        self.exp_run_progress_n_runs = 0
        self.exp_run_progress_row_meta = []
        self.exp_run_progress_algorithm_order = []
        self.exp_run_progress_metric_order = []
        self.exp_run_progress_metric_samples.clear()
        self.exp_metric_runtime_cache.clear()
        self.exp_run_progress_table.clear()
        self.exp_run_progress_table.setRowCount(0)
        self.exp_run_progress_table.setColumnCount(0)
        if hasattr(self, "exp_save_progress_csv_btn"):
            self.exp_save_progress_csv_btn.setEnabled(False)
        if hasattr(self, "exp_save_progress_latex_btn"):
            self.exp_save_progress_latex_btn.setEnabled(False)

    @staticmethod
    def _make_table_item(
        text: str,
        *,
        align: Qt.AlignmentFlag = Qt.AlignmentFlag.AlignCenter,
    ) -> QTableWidgetItem:
        item = QTableWidgetItem(text)
        item.setTextAlignment(align)
        item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
        return item

    @staticmethod
    def _metric_label_brief(metric_label: str) -> str:
        text = str(metric_label).strip()
        text = re.sub(r"\s*\[[^\]]+\]\s*$", "", text)
        return text or str(metric_label)

    @staticmethod
    def _format_scientific_compact(value: float, *, sig_digits: int = 3) -> str:
        try:
            x = float(value)
        except Exception:  # noqa: BLE001
            return "nan"
        if not math.isfinite(x):
            return "nan" if math.isnan(x) else ("inf" if x > 0 else "-inf")
        if x == 0.0:
            return "0"
        sig = max(1, int(sig_digits))
        s = f"{x:.{sig - 1}e}"
        mantissa, exp = s.split("e")
        mantissa = mantissa.rstrip("0").rstrip(".")
        exp_i = int(exp)
        return f"{mantissa}e{exp_i}"

    @staticmethod
    def _format_mean_std(values: list[float]) -> str:
        finite = [float(v) for v in values if isinstance(v, (int, float)) and math.isfinite(float(v))]
        if not finite:
            return "nan +/- nan"
        arr = np.asarray(finite, dtype=float)
        mean = float(np.mean(arr))
        std = float(np.std(arr, ddof=1 if arr.size > 1 else 0))
        return (
            f"{EmoPyLabMainWindow._format_scientific_compact(mean, sig_digits=4)} +/- "
            f"{EmoPyLabMainWindow._format_scientific_compact(std, sig_digits=3)}"
        )

    def _exp_progress_active_metric_label(self) -> str | None:
        if not self.exp_run_progress_metric_order:
            return None
        return str(self.exp_run_progress_metric_order[0])

    def _exp_problem_runs_cell_text(self, problem_id: str) -> str:
        total = max(1, int(self.exp_run_progress_n_runs))
        active_algo = self.exp_run_progress_active_algo_by_problem.get(problem_id)
        if not active_algo and self.exp_run_progress_algorithm_order:
            active_algo = self.exp_run_progress_algorithm_order[0]
        completed = int(self.exp_run_progress_counts.get((problem_id, str(active_algo or "")), 0))
        return f"{completed}/{total}"

    def _exp_problem_runs_cell_tooltip(self, problem_id: str) -> str:
        per_algo_total = max(1, int(self.exp_run_progress_n_runs))
        active_algo = self.exp_run_progress_active_algo_by_problem.get(problem_id)
        if not active_algo and self.exp_run_progress_algorithm_order:
            active_algo = self.exp_run_progress_algorithm_order[0]
        active_name = ""
        if active_algo:
            spec = self.algorithm_specs.get(str(active_algo))
            active_name = spec.name if spec is not None else str(active_algo)
        active_count = int(self.exp_run_progress_counts.get((problem_id, str(active_algo or "")), 0))
        lines: list[str] = []
        for algorithm_id in self.exp_run_progress_algorithm_order:
            count = int(self.exp_run_progress_counts.get((problem_id, algorithm_id), 0))
            spec = self.algorithm_specs.get(algorithm_id)
            algo_name = spec.name if spec is not None else algorithm_id
            lines.append(f"{algo_name}: {count}/{per_algo_total}")
        title = f"Run (current algorithm): {active_count}/{per_algo_total}"
        if active_name:
            title += f" [{active_name}]"
        return title + (("\n" + "\n".join(lines)) if lines else "")

    def _exp_progress_cell_text(self, problem_id: str, algorithm_id: str) -> str:
        metric_label = self._exp_progress_active_metric_label()
        if metric_label:
            value = self._exp_progress_metric_cell_text(problem_id, algorithm_id, metric_label)
            return value if value.strip() else "-"
        count = int(self.exp_run_progress_counts.get((problem_id, algorithm_id), 0))
        total = max(1, int(self.exp_run_progress_n_runs))
        return "-" if count <= 0 else f"{count}/{total}"

    def _exp_progress_metric_cell_text(self, problem_id: str, algorithm_id: str, metric_label: str) -> str:
        samples_by_metric = self.exp_run_progress_metric_samples.get((problem_id, algorithm_id), {})
        values = list(samples_by_metric.get(metric_label, []))
        if not values:
            return "-"
        return self._format_mean_std(values)

    def _exp_progress_winner_algorithm_ids(self, problem_id: str) -> set[str]:
        metric_label = self._exp_progress_active_metric_label()
        if not metric_label:
            return set()
        higher_better = self._metric_higher_better(metric_label)
        means: dict[str, float] = {}
        for algorithm_id in self.exp_run_progress_algorithm_order:
            samples_by_metric = self.exp_run_progress_metric_samples.get((problem_id, algorithm_id), {})
            values = [
                float(v)
                for v in samples_by_metric.get(metric_label, [])
                if isinstance(v, (int, float)) and math.isfinite(float(v))
            ]
            if not values:
                continue
            means[algorithm_id] = float(np.mean(np.asarray(values, dtype=float)))
        if not means:
            return set()
        best_value = max(means.values()) if higher_better else min(means.values())
        winners: set[str] = set()
        for algorithm_id, value in means.items():
            if math.isclose(float(value), float(best_value), rel_tol=1e-12, abs_tol=1e-15):
                winners.add(algorithm_id)
        return winners

    def _apply_exp_row_winner_bold(self, row: int, problem_id: str) -> None:
        if not hasattr(self, "exp_run_progress_table"):
            return
        winners = self._exp_progress_winner_algorithm_ids(problem_id)
        for algorithm_id in self.exp_run_progress_algorithm_order:
            col = self.exp_run_progress_col_by_algorithm.get(algorithm_id)
            if col is None:
                continue
            item = self.exp_run_progress_table.item(row, col)
            if item is None:
                continue
            text = item.text().strip()
            font = item.font()
            font.setBold(bool(algorithm_id in winners and text not in {"", "-"}))
            item.setFont(font)

    def _build_exp_run_progress_table(self, config: dict[str, Any]) -> None:
        if not hasattr(self, "exp_run_progress_table"):
            return
        self._clear_exp_run_progress_table()

        if not isinstance(config, dict):
            return

        raw_problem_entries = config.get("problem_entries", [])
        problem_entries: list[dict[str, Any]] = []
        if isinstance(raw_problem_entries, list):
            for entry in raw_problem_entries:
                if isinstance(entry, dict):
                    problem_entries.append(dict(entry))
        if not problem_entries:
            for problem_id in config.get("problem_ids", []) if isinstance(config.get("problem_ids"), list) else []:
                pid = str(problem_id)
                spec = self.problem_specs.get(pid)
                if spec is None:
                    continue
                problem_entries.append(
                    {
                        "instance_id": pid,
                        "problem_id": pid,
                        "label": spec.label,
                        "n_obj": int(getattr(spec, "default_n_obj", 1)),
                        "n_var": int(getattr(spec, "default_n_var", 1)),
                    }
                )

        algorithm_ids = [
            str(aid)
            for aid in (config.get("algorithm_ids", []) if isinstance(config.get("algorithm_ids"), list) else [])
            if str(aid) in self.algorithm_specs
        ]
        if not problem_entries or not algorithm_ids:
            return

        self.exp_run_progress_n_runs = max(1, int(config.get("n_runs", 1)))
        self.exp_run_progress_row_meta = []
        self.exp_run_progress_algorithm_order = list(algorithm_ids)
        self.exp_run_progress_metric_order = []
        metric_ids = config.get("metric_ids", [])
        if isinstance(metric_ids, list):
            for metric_id in metric_ids[:1]:
                spec = self.metric_specs.get(str(metric_id))
                if spec is not None:
                    self.exp_run_progress_metric_order.append(spec.label)

        table = self.exp_run_progress_table
        table.blockSignals(True)
        table.setRowCount(len(problem_entries))
        metric_labels = list(self.exp_run_progress_metric_order[:1])
        table.setColumnCount(4 + len(algorithm_ids))
        header_labels = ["Problem", "M", "D", "Run"]
        for algorithm_id in algorithm_ids:
            algo_name = self.algorithm_specs[algorithm_id].name
            header_labels.append(algo_name)
        table.setHorizontalHeaderLabels(header_labels)

        vertical_labels: list[str] = []
        for row, entry in enumerate(problem_entries):
            instance_id = str(entry.get("instance_id", entry.get("problem_id", ""))).strip()
            base_problem_id = str(entry.get("problem_id", "")).strip()
            label = str(entry.get("label", "")).strip()
            if not label:
                spec = self.problem_specs.get(base_problem_id)
                label = spec.label if spec is not None else (instance_id or base_problem_id or f"Problem {row+1}")
            vertical_labels.append(label)
            self.exp_run_progress_row_by_problem[instance_id or base_problem_id or str(row)] = row

            try:
                m_val = int(entry.get("n_obj", 0))
            except Exception:  # noqa: BLE001
                m_val = 0
            try:
                d_val = int(entry.get("n_var", 0))
            except Exception:  # noqa: BLE001
                d_val = 0
            table.setItem(row, 1, self._make_table_item(str(m_val)))
            table.setItem(row, 2, self._make_table_item(str(d_val)))
            base_label = (
                self.problem_specs.get(base_problem_id).label
                if base_problem_id in self.problem_specs
                else (label or base_problem_id)
            )
            self.exp_run_progress_row_meta.append(
                {
                    "row": row,
                    "instance_id": instance_id or base_problem_id or str(row),
                    "base_problem_id": base_problem_id,
                    "base_label": base_label,
                    "display_label": label,
                    "n_obj": m_val,
                    "n_var": d_val,
                }
            )

        table.setVerticalHeaderLabels(vertical_labels)
        table.verticalHeader().setVisible(False)

        for row, meta in enumerate(self.exp_run_progress_row_meta):
            display_label = str(meta.get("display_label", "")).strip() or str(meta.get("base_label", "")).strip()
            table.setItem(row, 0, self._make_table_item(display_label, align=Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter))
            pid = str(meta.get("instance_id", ""))
            run_item = self._make_table_item(
                self._exp_problem_runs_cell_text(pid),
                align=Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
            )
            run_item.setToolTip(self._exp_problem_runs_cell_tooltip(pid))
            table.setItem(row, 3, run_item)

        for algorithm_offset, algorithm_id in enumerate(algorithm_ids):
            col = 4 + algorithm_offset
            self.exp_run_progress_col_by_algorithm[algorithm_id] = col
            for row in range(len(problem_entries)):
                problem_meta = self.exp_run_progress_row_meta[row]
                pid = str(problem_meta.get("instance_id", ""))
                cell = self._make_table_item(
                    self._exp_progress_cell_text(pid, algorithm_id),
                    align=Qt.AlignmentFlag.AlignCenter,
                )
                metric_tooltip = self._metric_label_brief(metric_labels[0]) if metric_labels else "Metric"
                cell.setToolTip(f"{metric_tooltip}: pending (0 / {self.exp_run_progress_n_runs} completed)")
                table.setItem(row, col, cell)

        for row, meta in enumerate(self.exp_run_progress_row_meta):
            pid = str(meta.get("instance_id", ""))
            self._apply_exp_row_winner_bold(row, pid)

        table.resizeColumnsToContents()
        try:
            run_col = 3
            run_text_px = table.fontMetrics().horizontalAdvance("999/999")
            header_text_px = table.horizontalHeader().fontMetrics().horizontalAdvance("Run")
            target_run_width = max(72, run_text_px, header_text_px) + 12
            table.setColumnWidth(run_col, target_run_width)
        except Exception:  # noqa: BLE001
            pass
        for algorithm_id in self.exp_run_progress_algorithm_order:
            col = self.exp_run_progress_col_by_algorithm.get(algorithm_id)
            if col is None:
                continue
            try:
                table.setColumnWidth(col, max(table.columnWidth(col), 150))
            except Exception:  # noqa: BLE001
                pass
        table.blockSignals(False)
        if hasattr(self, "exp_save_progress_csv_btn"):
            self.exp_save_progress_csv_btn.setEnabled(True)
        if hasattr(self, "exp_save_progress_latex_btn"):
            self.exp_save_progress_latex_btn.setEnabled(True)

    def _update_exp_run_progress_table(self, run_payload: dict[str, Any]) -> None:
        if not isinstance(run_payload, dict) or not hasattr(self, "exp_run_progress_table"):
            return

        problem_id = str(run_payload.get("problem_id", "")).strip()
        algorithm_id = str(run_payload.get("algorithm_id", "")).strip()
        if not problem_id or not algorithm_id:
            return

        row = self.exp_run_progress_row_by_problem.get(problem_id)
        col = self.exp_run_progress_col_by_algorithm.get(algorithm_id)
        if row is None or col is None:
            return

        key = (problem_id, algorithm_id)
        count = int(self.exp_run_progress_counts.get(key, 0)) + 1
        self.exp_run_progress_counts[key] = count
        self.exp_run_progress_active_algo_by_problem[problem_id] = algorithm_id
        active_metric_spec = self._exp_selected_metric_spec_for_table()
        active_metric_label = active_metric_spec.label if active_metric_spec is not None else None
        if active_metric_label:
            self.exp_run_progress_metric_order = [active_metric_label]
        else:
            self.exp_run_progress_metric_order = []
        metric_map = run_payload.get("metrics", {})
        appended_active_metric = False
        if isinstance(metric_map, dict):
            samples_by_metric = self.exp_run_progress_metric_samples.setdefault(key, {})
            for metric_label, raw_value in metric_map.items():
                label = str(metric_label)
                if active_metric_label and label != active_metric_label:
                    continue
                try:
                    value = float(raw_value)
                except Exception:  # noqa: BLE001
                    continue
                samples_by_metric.setdefault(label, []).append(value)
                if active_metric_label and label == active_metric_label:
                    appended_active_metric = True
        if active_metric_spec is not None and not appended_active_metric:
            value = self._exp_compute_metric_for_run(active_metric_spec, run_payload)
            self.exp_run_progress_metric_samples.setdefault(key, {}).setdefault(active_metric_spec.label, []).append(value)

        item = self.exp_run_progress_table.item(row, col)
        cell_text = self._exp_progress_cell_text(problem_id, algorithm_id)
        if item is None:
            item = self._make_table_item(
                cell_text,
                align=Qt.AlignmentFlag.AlignCenter,
            )
            self.exp_run_progress_table.setItem(row, col, item)
        else:
            item.setText(cell_text)
            item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
        active_metric_label = self._exp_progress_active_metric_label()
        metric_brief = self._metric_label_brief(active_metric_label) if active_metric_label else "Metric"
        item.setToolTip(f"{metric_brief}: {cell_text}\n{count} / {max(1, self.exp_run_progress_n_runs)} completed")

        # Light visual progress cue without extra widgets.
        try:
            ratio = max(0.0, min(1.0, count / max(1, self.exp_run_progress_n_runs)))
            gray_base = QColor(getattr(AppStyles, "PANEL_ALT", "#E5E7EB"))
            alpha = 35 + int(70 * ratio)
            item.setBackground(QBrush(QColor(gray_base.red(), gray_base.green(), gray_base.blue(), alpha)))
        except Exception:  # noqa: BLE001
            pass
        run_item = self.exp_run_progress_table.item(row, 3)
        if run_item is None:
            run_item = self._make_table_item(
                self._exp_problem_runs_cell_text(problem_id),
                align=Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
            )
            self.exp_run_progress_table.setItem(row, 3, run_item)
        else:
            run_item.setText(self._exp_problem_runs_cell_text(problem_id))
            run_item.setTextAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        run_item.setToolTip(self._exp_problem_runs_cell_tooltip(problem_id))
        try:
            row_ratio = max(0.0, min(1.0, count / max(1, int(self.exp_run_progress_n_runs))))
            gray_base = QColor(getattr(AppStyles, "PANEL_ALT", "#E5E7EB"))
            alpha_row = 30 + int(60 * row_ratio)
            run_item.setBackground(QBrush(QColor(gray_base.red(), gray_base.green(), gray_base.blue(), alpha_row)))
        except Exception:  # noqa: BLE001
            pass
        self._apply_exp_row_winner_bold(row, problem_id)
        try:
            self.exp_run_progress_table.resizeRowToContents(row)
        except Exception:  # noqa: BLE001
            pass

    @staticmethod
    def _latex_escape(text: str) -> str:
        value = str(text)
        replacements = {
            "\\": r"\textbackslash{}",
            "&": r"\&",
            "%": r"\%",
            "$": r"\$",
            "#": r"\#",
            "_": r"\_",
            "{": r"\{",
            "}": r"\}",
            "~": r"\textasciitilde{}",
            "^": r"\textasciicircum{}",
        }
        for old, new in replacements.items():
            value = value.replace(old, new)
        return value

    def _latex_cell(self, text: str) -> str:
        raw = str(text)
        parts = [self._latex_escape(p) for p in raw.splitlines() if p is not None]
        if not parts:
            return ""
        if len(parts) == 1:
            return parts[0]
        return r"\shortstack[l]{" + r" \\ ".join(parts) + "}"

    def _exp_progress_table_export_matrix(self) -> tuple[list[str], list[list[str]], list[dict[str, Any]]]:
        headers = ["Problem", "M", "D", "Run"]
        for aid in self.exp_run_progress_algorithm_order:
            algo_name = self.algorithm_specs.get(aid).name if aid in self.algorithm_specs else aid
            headers.append(algo_name)

        # Group by base problem label, preserving table row order inside each group.
        grouped: dict[str, list[dict[str, Any]]] = {}
        group_order: list[str] = []
        row_meta_sorted = sorted(self.exp_run_progress_row_meta, key=lambda m: int(m.get("row", 0)))
        for meta in row_meta_sorted:
            group_key = str(meta.get("base_label", "")).strip() or str(meta.get("display_label", "")).strip()
            if group_key not in grouped:
                grouped[group_key] = []
                group_order.append(group_key)
            grouped[group_key].append(meta)

        rows: list[list[str]] = []
        ordered_meta: list[dict[str, Any]] = []
        for group_key in group_order:
            metas = grouped.get(group_key, [])
            metas = sorted(
                metas,
                key=lambda m: (
                    int(m.get("n_obj", 0)),
                    int(m.get("n_var", 0)),
                    str(m.get("display_label", "")).lower(),
                ),
            )
            for idx, meta in enumerate(metas):
                row_idx = int(meta.get("row", 0))
                problem_label_cell = group_key if idx == 0 else ""
                m_val = str(meta.get("n_obj", ""))
                d_val = str(meta.get("n_var", ""))
                pid = str(meta.get("instance_id", ""))
                row_values = [problem_label_cell, m_val, d_val, self._exp_problem_runs_cell_text(pid)]
                for algorithm_id in self.exp_run_progress_algorithm_order:
                    col = self.exp_run_progress_col_by_algorithm.get(algorithm_id)
                    item = self.exp_run_progress_table.item(row_idx, col) if col is not None else None
                    row_values.append(item.text().strip() if item is not None else "-")
                rows.append(row_values)
                ordered_meta.append(meta)
        return headers, rows, ordered_meta

    @Slot()
    def _save_exp_progress_table_csv(self) -> None:
        if not self.exp_run_progress_row_meta or not self.exp_run_progress_algorithm_order:
            QMessageBox.information(self, "Export", "No experiment progress table data to export.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Experiment Progress Table (CSV)",
            str(self.base_dir / "experiment_progress_table.csv"),
            "CSV files (*.csv);;All files (*.*)",
        )
        if not path:
            return
        headers, rows, _ = self._exp_progress_table_export_matrix()
        try:
            with open(path, "w", encoding="utf-8-sig", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(headers)
                writer.writerows(rows)
            self.exp_status_label.setText(f"Saved CSV: {Path(path).name}")
            self.exp_status_label.setStyleSheet(f"color: {AppStyles.SUCCESS};")
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Export failed", f"Could not save CSV:\n{exc}")

    @Slot()
    def _save_exp_progress_table_latex(self) -> None:
        if not self.exp_run_progress_row_meta or not self.exp_run_progress_algorithm_order:
            QMessageBox.information(self, "Export", "No experiment progress table data to export.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Experiment Progress Table (LaTeX)",
            str(self.base_dir / "experiment_progress_table.tex"),
            "TeX files (*.tex);;Text files (*.txt);;All files (*.*)",
        )
        if not path:
            return
        headers, rows, ordered_meta = self._exp_progress_table_export_matrix()
        align = "lrr" + ("l" * max(0, len(headers) - 3))
        lines: list[str] = []
        lines.append("% Generated by EmoPyLab (Experiment Module summary table)")
        metric_label = self._exp_progress_active_metric_label()
        metric_brief = self._metric_label_brief(metric_label) if metric_label else "N/A"
        lines.append(
            f"% Metric: {metric_brief}; cells store mean +/- std over completed runs (n_runs={max(1, self.exp_run_progress_n_runs)})"
        )
        lines.append(r"\begin{tabular}{" + align + "}")
        lines.append(r"\hline")
        lines.append(" & ".join(self._latex_escape(h) for h in headers) + r" \\")
        lines.append(r"\hline")
        algo_col_start = 4
        algo_col_end = algo_col_start + len(self.exp_run_progress_algorithm_order)
        for row_idx, row in enumerate(rows):
            meta = ordered_meta[row_idx] if row_idx < len(ordered_meta) else {}
            problem_instance_id = str(meta.get("instance_id", "")).strip()
            winner_algorithms = (
                self._exp_progress_winner_algorithm_ids(problem_instance_id) if problem_instance_id else set()
            )
            row_cells: list[str] = []
            for col_idx, cell_text in enumerate(row):
                latex_cell = self._latex_cell(cell_text)
                if (
                    algo_col_start <= col_idx < algo_col_end
                    and (col_idx - algo_col_start) < len(self.exp_run_progress_algorithm_order)
                ):
                    algorithm_id = self.exp_run_progress_algorithm_order[col_idx - algo_col_start]
                    if algorithm_id in winner_algorithms and str(cell_text).strip() not in {"", "-"}:
                        latex_cell = r"\textbf{" + latex_cell + "}"
                row_cells.append(latex_cell)
            lines.append(" & ".join(row_cells) + r" \\")
        lines.append(r"\hline")
        lines.append(r"\end{tabular}")
        try:
            Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")
            self.exp_status_label.setText(f"Saved LaTeX table: {Path(path).name}")
            self.exp_status_label.setStyleSheet(f"color: {AppStyles.SUCCESS};")
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Export failed", f"Could not save LaTeX table:\n{exc}")

    def _exp_checked_ids(self, list_widget: QListWidget) -> list[str]:
        """Return effective selected IDs (union of selected rows and checked boxes)."""
        selected_set: set[str] = set()
        for item in list_widget.selectedItems():
            item_id = item.data(Qt.ItemDataRole.UserRole)
            if isinstance(item_id, str):
                selected_set.add(item_id)

        ids: list[str] = []
        for i in range(list_widget.count()):
            item = list_widget.item(i)
            item_id = item.data(Qt.ItemDataRole.UserRole)
            if not isinstance(item_id, str):
                continue
            if item_id in selected_set or item.checkState() == Qt.CheckState.Checked:
                ids.append(item_id)
        return ids

    def _refresh_exp_selected_algorithms_summary(self) -> None:
        if not hasattr(self, "exp_selected_algorithms_summary_label") or not hasattr(self, "exp_algorithm_list"):
            return
        algo_ids = self._exp_checked_ids(self.exp_algorithm_list)
        labels: list[str] = []
        for aid in algo_ids:
            spec = self.algorithm_specs.get(aid)
            if spec is not None:
                labels.append(spec.name)
        if not labels:
            self.exp_selected_algorithms_summary_label.setText("No algorithm selected")
            self._refresh_exp_workload_summary()
            return
        preview_limit = 5
        shown = labels[:preview_limit]
        suffix = ""
        if len(labels) > preview_limit:
            suffix = f" ... (+{len(labels) - preview_limit})"
        self.exp_selected_algorithms_summary_label.setText(f"{len(labels)} selected: " + ", ".join(shown) + suffix)
        self._refresh_exp_workload_summary()

    def _refresh_exp_workload_summary(self) -> None:
        if not hasattr(self, "exp_workload_summary_label"):
            return

        algo_count = self._count_checked_exp_items(self.exp_algorithm_list) if hasattr(self, "exp_algorithm_list") else 0
        problem_count = self._count_checked_exp_items(self.exp_problem_list) if hasattr(self, "exp_problem_list") else 0
        metric_count = self._count_checked_exp_items(self.exp_metric_list) if hasattr(self, "exp_metric_list") else 0
        runs = (
            int(self.exp_n_runs_spin.value())
            if hasattr(self, "exp_n_runs_spin")
            else DEFAULT_EXPERIMENT_N_RUNS
        )
        workload = algo_count * problem_count * max(1, runs)
        metric_label = "metric" if metric_count == 1 else "metrics"
        ready = algo_count > 0 and problem_count > 0 and metric_count > 0

        if ready:
            self.exp_workload_summary_label.setText(
                f"{algo_count} algorithm(s) x {problem_count} problem(s) x "
                f"{runs} run(s) = {workload} task(s). "
                f"{metric_count} {metric_label}; MaxFE={self.exp_max_fe_spin.value()}."
            )
        else:
            self.exp_workload_summary_label.setText(
                "Select at least one algorithm, one problem, and one metric to enable execution."
            )

        if hasattr(self, "exp_btn_start") and self.exp_worker_thread is None:
            self.exp_btn_start.setEnabled(ready)

    def _set_experiment_ui_controls_enabled(self, enabled: bool) -> None:
        enabled = bool(enabled)
        for name in (
            "exp_left_panel",
            "exp_center_panel",
            "exp_results_store_box",
            "exp_config_box",
        ):
            widget = getattr(self, name, None)
            if widget is not None:
                widget.setEnabled(enabled)

        if not enabled:
            for name in (
                "exp_algorithm_filter",
                "exp_problem_filter",
                "exp_metric_filter",
                "exp_results_path_edit",
            ):
                widget = getattr(self, name, None)
                try:
                    if widget is not None and widget.hasFocus():
                        widget.clearFocus()
                except Exception:  # noqa: BLE001
                    pass

        can_export = (
            enabled
            and bool(getattr(self, "exp_run_progress_row_meta", []))
            and bool(getattr(self, "exp_run_progress_algorithm_order", []))
        )
        if hasattr(self, "exp_save_progress_csv_btn"):
            self.exp_save_progress_csv_btn.setEnabled(can_export)
        if hasattr(self, "exp_save_progress_latex_btn"):
            self.exp_save_progress_latex_btn.setEnabled(can_export)

        tabs = getattr(self, "tabs", None)
        exp_tab = getattr(self, "experiment_tab", None)
        if tabs is None or exp_tab is None:
            return
        try:
            if not enabled:
                if self._exp_tabs_enabled_snapshot is None or len(self._exp_tabs_enabled_snapshot) != tabs.count():
                    self._exp_tabs_enabled_snapshot = [tabs.isTabEnabled(i) for i in range(tabs.count())]
                for i in range(tabs.count()):
                    tabs.setTabEnabled(i, tabs.widget(i) is exp_tab)
            else:
                snapshot = self._exp_tabs_enabled_snapshot
                if isinstance(snapshot, list) and len(snapshot) == tabs.count():
                    for i, was_enabled in enumerate(snapshot):
                        tabs.setTabEnabled(i, bool(was_enabled))
                else:
                    for i in range(tabs.count()):
                        tabs.setTabEnabled(i, True)
                self._exp_tabs_enabled_snapshot = None
        except Exception:  # noqa: BLE001
            pass

    def _collect_experiment_config(self) -> dict[str, Any] | None:
        """Collect configuration for Experiment Module (multi-algo, multi-problem, n_runs)."""
        algorithm_ids = self._exp_checked_ids(self.exp_algorithm_list)
        if not algorithm_ids:
            QMessageBox.warning(self, "Validation", "Select at least one algorithm.")
            return None

        problem_ids = self._exp_checked_ids(self.exp_problem_list)
        if not problem_ids:
            QMessageBox.warning(self, "Validation", "Select at least one problem.")
            return None

        metric_ids = self._exp_checked_ids(self.exp_metric_list)
        metric_ids = self._map_metric_ids_to_backend(
            metric_ids,
            prefer_jax=self._exp_backend_prefers_jax_for("metrics"),
        )
        if len(metric_ids) > 1:
            metric_ids = metric_ids[:1]
        if not metric_ids:
            QMessageBox.warning(self, "Validation", "Select at least one metric.")
            return None

        # Ensure currently edited per-problem form is persisted before collecting config.
        self._store_exp_problem_override_for_current()
        # Operator UI is hidden; force defaults per algorithm.
        self.exp_algorithm_operator_overrides = {}

        # Ref point: auto-computed (automatic strategy)
        ref_point = None

        # Store effective per-problem settings (base + variants).
        problem_overrides: dict[str, dict[str, int]] = {}
        problem_entries: list[dict[str, Any]] = []
        for problem_id in problem_ids:
            effective = self._exp_problem_effective_values(problem_id)
            problem_overrides[problem_id] = {
                "pop_size": int(effective["pop_size"]),
                "n_obj": int(effective["n_obj"]),
                "n_var": int(effective["n_var"]),
            }
            problem_entries.append({
                "instance_id": str(problem_id),
                "problem_id": str(problem_id),
                "label": self._exp_problem_target_label(problem_id),
                "pop_size": int(effective["pop_size"]),
                "n_obj": int(effective["n_obj"]),
                "n_var": int(effective["n_var"]),
            })
            for variant_key, variant_entry in self.exp_problem_variants.items():
                if not isinstance(variant_entry, dict):
                    continue
                if str(variant_entry.get("problem_id", "")).strip() != str(problem_id):
                    continue
                veffective = self._exp_problem_effective_values(variant_key)
                problem_overrides[variant_key] = {
                    "pop_size": int(veffective["pop_size"]),
                    "n_obj": int(veffective["n_obj"]),
                    "n_var": int(veffective["n_var"]),
                }
                problem_entries.append({
                    "instance_id": str(variant_key),
                    "problem_id": str(problem_id),
                    "label": self._exp_problem_target_label(variant_key),
                    "pop_size": int(veffective["pop_size"]),
                    "n_obj": int(veffective["n_obj"]),
                    "n_var": int(veffective["n_var"]),
                })

        first_problem_values = (problem_entries[0] if problem_entries else {
            "pop_size": int(self.exp_pop_size_spin.value()),
            "n_obj": int(self.exp_n_obj_spin.value()),
            "n_var": int(self.exp_n_var_spin.value()),
        })

        exp_seed_mode = str(self.exp_seed_mode_combo.currentData() or SEED_MODE_RANDOM).strip().lower()
        if exp_seed_mode not in {SEED_MODE_RANDOM, SEED_MODE_FIXED, SEED_MODE_SEQUENCE}:
            exp_seed_mode = SEED_MODE_RANDOM
        raw_results_path = self._normalize_exp_results_storage_path(
            self.exp_results_path_edit.text() if hasattr(self, "exp_results_path_edit") else self.exp_results_storage_path
        )
        self.exp_results_storage_path = raw_results_path

        return {
            "problem_id": problem_ids[0],
            "problem_ids": problem_ids,
            "problem_entries": problem_entries,
            "n_var": int(first_problem_values["n_var"]),
            "n_obj": int(first_problem_values["n_obj"]),
            "pop_size": int(first_problem_values["pop_size"]),
            "max_fe": int(self.exp_max_fe_spin.value()),
            "maxFE": int(self.exp_max_fe_spin.value()),
            "n_runs": self.exp_n_runs_spin.value(),
            "seed_mode": exp_seed_mode,
            "seed_base": int(self.exp_seed_base_spin.value()),
            "seed_step": int(self.exp_seed_step_spin.value()),
            "algorithm_ids": algorithm_ids,
            "metric_ids": metric_ids,
            "ref_point": ref_point,
            "use_pf": self.exp_use_pf_check.isChecked(),
            "compute_backend": self.exp_compute_backend_combo.currentData(),
            "gpu_dtype": "float32",
            "joblib_backend": "loky",
            "joblib_n_jobs": -1,
            "profile_compare": False,
            "stat_test_method": self.stat_method_combo.currentData(),
            "stat_test_reference": self.stat_reference_combo.currentText(),
            "stat_test_alpha": self._analysis_stat_alpha(),
            # Operators are intentionally hidden in UI; always use algorithm defaults.
            "crossover": "default",
            "mutation": "default",
            "selection": "default",
            "sampling": "default",
            "crossover_eta": 15.0,
            "crossover_prob": 0.9,
            "mutation_eta": 20.0,
            "mutation_prob": None,
            "selection_pressure": 2,
            "parallel_workers": self.exp_parallel_workers_spin.value(),
            # Experiment Module optimization path: store raw runs first, compute summary metrics in UI from saved payloads.
            # This avoids per-generation metric overhead and heavy partial_result traffic, which the Experiment UI does not use.
            "__exp_raw_first_metrics__": True,
            "__emit_partial_results__": False,
            "experiment_results_path": str(raw_results_path),
            "problem_overrides": problem_overrides,
            "problem_variants": {
                key: dict(value)
                for key, value in self.exp_problem_variants.items()
                if isinstance(value, dict)
                and str(value.get("problem_id", "")).strip() in set(problem_ids)
            },
            "algorithm_operator_overrides": {},
        }

    @Slot()
    def _start_experiment(self) -> None:
        """Start Experiment Module execution (multi-algo, multi-problem, n_runs)."""
        if self.exp_worker_thread is not None:
            QMessageBox.information(self, "Execution", "An experiment is already running.")
            return

        config = self._collect_experiment_config()
        if config is None:
            return

        # Auto-save last experiment config for persistence
        self._save_experiment_config_auto(config)

        # Reset experiment state
        self.exp_results.clear()
        self.exp_metric_names.clear()
        self.exp_execution_backend_label = ""
        self.exp_profile_compare_enabled = False
        self.exp_pareto_front = None
        self.exp_pareto_fronts.clear()
        self.exp_metric_runtime_cache.clear()
        self.exp_results_storage_loaded_path = None
        self._refresh_exp_results_storage_info("Running experiment. New raw results will overwrite/update the configured file on completion.")

        # Clear experiment log
        self.exp_log_box.clear()
        self._build_exp_run_progress_table(config)
        self._set_experiment_ui_controls_enabled(False)

        self.exp_btn_start.setEnabled(False)
        self.exp_btn_stop.setEnabled(True)
        self.exp_btn_stop.setFocus(Qt.FocusReason.OtherFocusReason)
        self.exp_progress.setValue(0)
        self._exp_status_locked = False

        n_algos = len(config["algorithm_ids"])
        problem_entries = config.get("problem_entries", [])
        if isinstance(problem_entries, list) and problem_entries:
            n_probs = len(problem_entries)
        else:
            n_probs = len(config["problem_ids"])
        n_runs = config["n_runs"]
        self._exp_append_log(
            f"Starting experiment: {n_algos} algorithm(s) x {n_probs} problem(s) x {n_runs} run(s) "
            f"[seed_mode={config.get('seed_mode', SEED_MODE_RANDOM)}]"
        )
        self.exp_status_label.setText(
            f"Running: {n_algos} algorithm(s) x {n_probs} problem(s) x {n_runs} run(s)..."
        )
        self.exp_status_label.setStyleSheet(f"color: {AppStyles.INFO};")

        self.exp_worker_thread = QThread(self)
        self.exp_worker = ExperimentBridge(
            config=config,
            algorithm_specs=dict(self.algorithm_specs),
            problem_specs=dict(self.problem_specs),
            metric_specs=dict(self.metric_specs),
        )
        self.exp_worker.moveToThread(self.exp_worker_thread)

        # Connect signals
        self.exp_worker_thread.started.connect(self.exp_worker.run)
        self.exp_worker.progress.connect(self._on_exp_progress)
        self.exp_worker.run_ready.connect(self._on_exp_run_ready)
        self.exp_worker.partial_result.connect(self._on_exp_partial_result)
        self.exp_worker.warning.connect(self._on_exp_warning)
        self.exp_worker.error.connect(self._on_exp_error)
        self.exp_worker.finished.connect(self._on_exp_finished)

        # Thread lifecycle: quit thread when worker finishes or errors
        self.exp_worker.finished.connect(self.exp_worker_thread.quit)
        self.exp_worker.error.connect(self.exp_worker_thread.quit)
        self.exp_worker_thread.finished.connect(self._on_exp_thread_finished)

        self.exp_worker_thread.start()

    @Slot()
    def _stop_experiment(self) -> None:
        """Stop Experiment Module execution."""
        if self.exp_worker is None:
            return
        self.exp_worker.cancel()
        self.exp_btn_stop.setEnabled(False)
        self._exp_append_log("Experiment stop requested...")

    @Slot(int, str)
    def _on_exp_progress(self, percent: int, message: str) -> None:
        if getattr(self, "_exp_status_locked", False):
            return
        self.exp_progress.setValue(percent)
        clean_message = self._sanitize_exp_status_message(message)
        self.exp_status_label.setText(clean_message)
        self._exp_append_log(clean_message)

    @Slot(object)
    def _on_exp_partial_result(self, payload: dict[str, Any]) -> None:
        """Handle real-time updates from the experiment worker."""
        self.exp_live_payload = payload
        # Optionally update a plot in experiment tab if we had one.
        # For now, we mainly update the Test Module result display if it corresponds to the current run.
        # But usually Test Module and Experiment Module are distinct.

    @Slot(object)
    def _on_exp_run_ready(self, payload: dict[str, Any]) -> None:
        payload = self._ensure_run_dimensions(payload)
        algo = payload.get("algorithm", "-")
        prob = payload.get("problem", "-")
        run_idx = payload.get("run_index", "-")
        n_obj = payload.get("n_obj", "-")
        n_var = payload.get("n_var", "-")
        self._exp_append_log(f"Run ready: {algo} on {prob} - run #{run_idx} (M={n_obj}, D={n_var})")

        algo_name = str(algo)
        self.exp_results.setdefault(algo_name, []).append(dict(payload))
        self._update_exp_run_progress_table(payload)

    @Slot(str)
    def _on_exp_warning(self, message: str) -> None:
        text = str(message)
        is_jax_gpu_fallback_notice = (
            (
                "GPU requested, but unavailable:" in text
                or "JAX backend requested, but unavailable:" in text
            )
            and "No JAX GPU device detected" in text
        )
        if is_jax_gpu_fallback_notice:
            info_text = (
                "JAX GPU not detected. Using JAX on CPU (CPU (JAX)) "
                "for JAX-compatible problems when available."
            )
            self._exp_append_log(f"Info: {info_text}")
            self.exp_status_label.setText(self._sanitize_exp_status_message(info_text[:180]))
            self.exp_status_label.setStyleSheet(f"color: {AppStyles.INFO};")
            self.exp_status_label.setToolTip(text)
            return

        self._exp_append_log(f"Warning: {text}")
        self.exp_status_label.setText(self._sanitize_exp_status_message(f"Warning: {text[:160]}"))
        self.exp_status_label.setStyleSheet(f"color: {AppStyles.WARNING};")
        self.exp_status_label.setToolTip(text)

    @Slot(str)
    def _on_exp_error(self, message: str) -> None:
        self._exp_status_locked = True
        self._exp_append_log(f"ERROR: {message}")
        self.exp_status_label.setText(self._sanitize_exp_status_message(f"Error: {message[:120]}"))
        self.exp_status_label.setStyleSheet(f"color: {AppStyles.ERROR};")
        QMessageBox.critical(self, "Experiment failed", message)
        self._restore_maximized_window_mode()

    @Slot(object)
    def _on_exp_finished(self, payload: dict[str, Any]) -> None:
        self._exp_status_locked = True
        self.exp_live_payload = None # Clear live data
        self._apply_exp_results_payload(payload, rebuild_table_from_payload_config=False, loaded_from_path=None)
        self._sync_payload_to_analysis_workspace(payload)

        # Accumulate new results for the Analysis tab state

        # Keep compatibility with local states (optional; keep latest for module-specific views)
        # Keep experiment-local results only (no automatic Analysis tab sync).

        cancelled = bool(payload.get("cancelled", False))
        n_total_runs = sum(len(runs) for runs in self.exp_results.values())
        status_suffix = " (cancelled)" if cancelled else ""
        self._exp_append_log(
            f"Experiment finished{status_suffix} - {len(self.exp_results)} algorithm(s), "
            f"{n_total_runs} total run(s). Backend: {self.exp_execution_backend_label}"
        )
        self.exp_status_label.setText(self._sanitize_exp_status_message(
            f"Done{status_suffix} - {len(self.exp_results)} algorithm(s), {n_total_runs} run(s)"
        ))
        self.exp_status_label.setStyleSheet(
            f"color: {AppStyles.WARNING if cancelled else AppStyles.SUCCESS};"
        )
        self.exp_progress.setValue(100 if not cancelled else self.exp_progress.value())
        save_path = self._normalize_exp_results_storage_path(
            self.exp_results_path_edit.text() if hasattr(self, "exp_results_path_edit") else self.exp_results_storage_path
        )
        self.exp_results_storage_path = save_path
        ok, save_message = self._save_exp_results_payload_to_path(payload, save_path)
        self._exp_append_log(save_message)
        self._refresh_exp_results_storage_info(
            ("Raw results file updated and ready for metric reuse." if ok else save_message)
        )
        if ok:
            self.exp_status_label.setToolTip(str(save_path))

        # Refresh Analysis tab
        # Experiment Module no longer auto-populates nor switches to the Analysis tab.
        self._restore_maximized_window_mode()

    @Slot()
    def _on_exp_thread_finished(self) -> None:
        if self.exp_worker is not None:
            self.exp_worker.deleteLater()
        if self.exp_worker_thread is not None:
            self.exp_worker_thread.deleteLater()

        self.exp_worker = None
        self.exp_worker_thread = None
        self._set_experiment_ui_controls_enabled(True)
        self._refresh_exp_workload_summary()
        self.exp_btn_stop.setEnabled(False)
        self._restore_maximized_window_mode()

    def _populate_analysis_combos(self) -> None:
        """Refresh the Analysis tab combos from current results."""
        current_metric = self.metric_plot_combo.currentText().strip()
        current_problem_id = self.problem_plot_combo.currentData()
        current_algo = self.algo_plot_combo.currentText().strip()
        current_ref = self.stat_reference_combo.currentText().strip()
        all_runs = self._all_analysis_runs()

        # Metric combo
        self.metric_plot_combo.blockSignals(True)
        self.metric_plot_combo.clear()
        for m in self.metric_names:
            self.metric_plot_combo.addItem(m)
        if current_metric and self.metric_plot_combo.findText(current_metric) >= 0:
            self.metric_plot_combo.setCurrentText(current_metric)
        self.metric_plot_combo.blockSignals(False)

        # Problem combo
        problem_entries: dict[str, str] = {}
        for run in all_runs:
            problem_id = str(run.get("problem_id", "")).strip()
            problem_label = str(self._problem_label_for_run(run)).strip()
            if problem_id:
                problem_entries[problem_id] = problem_label or problem_id

        self.problem_plot_combo.blockSignals(True)
        self.problem_plot_combo.clear()
        self.problem_plot_combo.addItem("All problems", "__all__")
        for problem_id, problem_label in sorted(problem_entries.items(), key=lambda x: x[1].lower()):
            self.problem_plot_combo.addItem(problem_label, problem_id)
        target_problem = "__all__"
        if isinstance(current_problem_id, str) and self.problem_plot_combo.findData(current_problem_id) >= 0:
            target_problem = current_problem_id
        elif len(problem_entries) == 1:
            target_problem = next(iter(problem_entries))
        target_idx = self.problem_plot_combo.findData(target_problem)
        if target_idx >= 0:
            self.problem_plot_combo.setCurrentIndex(target_idx)
        self.problem_plot_combo.blockSignals(False)

        # Algorithm combo
        selected_problem_id = self._selected_plot_problem_id()
        algo_options = self._analysis_algorithm_options(selected_problem_id)
        if not algo_options and selected_problem_id is not None:
            algo_options = self._analysis_algorithm_options()

        self.algo_plot_combo.blockSignals(True)
        self.algo_plot_combo.clear()
        for algo_name in algo_options:
            self.algo_plot_combo.addItem(algo_name)
        if current_algo and current_algo in algo_options:
            self.algo_plot_combo.setCurrentText(current_algo)
        elif self.algo_plot_combo.count() > 0:
            self.algo_plot_combo.setCurrentIndex(0)
        self.algo_plot_combo.blockSignals(False)

        # Stat reference combo
        self.stat_reference_combo.blockSignals(True)
        self.stat_reference_combo.clear()
        for algo_name in self._analysis_algorithm_options():
            self.stat_reference_combo.addItem(algo_name)
        idx = self.stat_reference_combo.findText(current_ref)
        if idx >= 0:
            self.stat_reference_combo.setCurrentIndex(idx)
        elif self.stat_reference_combo.count() > 0:
            self.stat_reference_combo.setCurrentIndex(0)
        self.stat_reference_combo.blockSignals(False)
        self._update_stat_controls_state()

    def _save_experiment_config_auto(self, config: dict[str, Any]) -> None:
        """Auto-save experiment config to JSON file beside the script."""
        file_path = self.base_dir / EXP_CONFIG_FILENAME
        try:
            with open(file_path, "w", encoding="utf-8") as handle:
                json.dump(config, handle, indent=2, ensure_ascii=False)
            self._exp_append_log(f"Config auto-saved: {file_path.name}")
        except Exception as exc:  # noqa: BLE001
            self._exp_append_log(f"Warning: could not auto-save config - {exc}")

    def _load_experiment_config_auto(self) -> bool:
        """Load last experiment config from JSON if it exists."""
        file_path = self.base_dir / EXP_CONFIG_FILENAME
        if not file_path.exists():
            return False
        try:
            with open(file_path, "r", encoding="utf-8") as handle:
                config = json.load(handle)
            self._apply_experiment_config(config)
            self._exp_append_log(f"Last experiment config restored from {file_path.name}")
            return True
        except Exception as exc:  # noqa: BLE001
            self._exp_append_log(f"Warning: could not load last config - {exc}")
            return False

    @Slot()
    def _load_last_experiment_config_from_ui(self) -> None:
        file_path = self.base_dir / EXP_CONFIG_FILENAME
        if not file_path.exists():
            QMessageBox.information(
                self,
                "Load last experiment config",
                f"No auto-saved experiment config was found yet:\n{file_path}",
            )
            return
        try:
            ok = self._load_experiment_config_auto()
            if not ok:
                QMessageBox.warning(
                    self,
                    "Load last experiment config",
                    f"Could not load last experiment config from:\n{file_path}",
                )
                return

            # UX: "Load last" should restore the summary table too, using the saved raw
            # experiment payloads referenced by the restored config path.
            raw_path = self._normalize_exp_results_storage_path(
                self.exp_results_path_edit.text()
                if hasattr(self, "exp_results_path_edit")
                else getattr(self, "exp_results_storage_path", None)
            )
            self.exp_results_storage_path = raw_path
            raw_loaded = False
            raw_load_failed = False

            if raw_path.exists() and raw_path.is_file():
                raw_loaded = self._load_exp_results_from_path(raw_path, interactive=False)
                raw_load_failed = not raw_loaded

            if raw_loaded:
                self.exp_status_label.setText(
                    self._sanitize_exp_status_message(
                        f"Last config + raw results loaded: {file_path.name} / {raw_path.name}"
                    )
                )
                self.exp_status_label.setStyleSheet(f"color: {AppStyles.INFO};")
                self.exp_status_label.setToolTip(f"{file_path}\n{raw_path}")
            elif raw_load_failed:
                self.exp_status_label.setText(
                    self._sanitize_exp_status_message(
                        f"Last config loaded, but raw results could not be loaded: {raw_path.name}"
                    )
                )
                self.exp_status_label.setStyleSheet(f"color: {AppStyles.WARNING};")
                self.exp_status_label.setToolTip(f"{file_path}\n{raw_path}")
            else:
                self.exp_status_label.setText(
                    self._sanitize_exp_status_message(
                        f"Last experiment config loaded: {file_path.name} (raw results file not found)"
                    )
                )
                self.exp_status_label.setStyleSheet(f"color: {AppStyles.INFO};")
                self.exp_status_label.setToolTip(f"{file_path}\n{raw_path}")
        except Exception as exc:  # noqa: BLE001
            QMessageBox.warning(
                self,
                "Load last experiment config",
                f"Could not load last experiment config:\n{exc}",
            )

    def _apply_experiment_config(self, config: dict[str, Any]) -> None:
        """Apply a config dict to the Experiment Module widgets."""
        # Restore algorithm selections
        algo_ids = set(config.get("algorithm_ids", []))
        self.exp_algorithm_list.blockSignals(True)
        for i in range(self.exp_algorithm_list.count()):
            item = self.exp_algorithm_list.item(i)
            item_id = item.data(Qt.ItemDataRole.UserRole)
            is_selected = item_id in algo_ids
            item.setSelected(is_selected)
            item.setCheckState(Qt.CheckState.Checked if is_selected else Qt.CheckState.Unchecked)
        self.exp_algorithm_list.blockSignals(False)
        checked_algos = self._count_checked_exp_items(self.exp_algorithm_list)
        self.exp_algo_counter.setText(f"{checked_algos} selected")
        self._refresh_exp_selected_algorithms_summary()

        # Restore problem selections
        prob_ids = {str(v) for v in config.get("problem_ids", [])}
        if not prob_ids and isinstance(config.get("problem_entries"), list):
            for entry in config.get("problem_entries", []):
                if not isinstance(entry, dict):
                    continue
                base_id = str(entry.get("problem_id", "")).strip()
                if base_id in self.problem_specs:
                    prob_ids.add(base_id)
        self.exp_problem_list.blockSignals(True)
        for i in range(self.exp_problem_list.count()):
            item = self.exp_problem_list.item(i)
            item_id = item.data(Qt.ItemDataRole.UserRole)
            is_selected = item_id in prob_ids
            item.setSelected(is_selected)
            item.setCheckState(Qt.CheckState.Checked if is_selected else Qt.CheckState.Unchecked)
        self.exp_problem_list.blockSignals(False)
        checked_probs = self._count_checked_exp_items(self.exp_problem_list)
        self.exp_prob_counter.setText(f"{checked_probs} selected")
        self._refresh_exp_workload_summary()

        # Restore metric selections
        metric_values = config.get("metric_ids", [])
        if not metric_values and "metrics" in config and isinstance(config.get("metrics"), dict):
            metric_values = [key for key, enabled in config.get("metrics", {}).items() if enabled]
        metric_ids = {str(v) for v in metric_values}
        metric_names = {str(v).strip().lower() for v in metric_values}
        self._set_checked_ids(self.exp_metric_list, metric_ids, metric_names, self.metric_specs)
        if not self._exp_checked_ids(self.exp_metric_list) and self.exp_metric_list.count() > 0:
            preferred_idx = -1
            for i in range(self.exp_metric_list.count()):
                item = self.exp_metric_list.item(i)
                metric_id = item.data(Qt.ItemDataRole.UserRole)
                spec = self.metric_specs.get(metric_id) if isinstance(metric_id, str) else None
                if spec is not None and self._is_default_metric_checked(spec):
                    preferred_idx = i
                    break
            if preferred_idx < 0:
                preferred_idx = 0
            self.exp_metric_list.item(preferred_idx).setCheckState(Qt.CheckState.Checked)

        # Restore parameter spinboxes
        self.exp_n_runs_spin.setValue(
            _positive_int(
                config.get("n_runs", DEFAULT_EXPERIMENT_N_RUNS),
                DEFAULT_EXPERIMENT_N_RUNS,
                minimum=1,
            )
        )
        self.exp_pop_size_spin.setValue(
            _positive_int(
                config.get("pop_size", DEFAULT_EXPERIMENT_POP_SIZE),
                DEFAULT_EXPERIMENT_POP_SIZE,
                minimum=10,
            )
        )
        if "n_obj" in config:
            self.exp_n_obj_spin.setValue(int(config["n_obj"]))
        if "n_var" in config:
            self.exp_n_var_spin.setValue(int(config["n_var"]))
        max_fe_raw = config.get("max_fe", config.get("maxFE"))
        if max_fe_raw is not None:
            try:
                self.exp_max_fe_spin.setValue(int(max_fe_raw))
            except (ValueError, TypeError):
                pass

        # Restore backend combos before operator combos to load backend-specific options.
        backend = str(config.get("compute_backend", "cpu")).lower()
        if backend == "auto":
            backend = "cpu"
        backend_idx = self.exp_compute_backend_combo.findData(backend)
        if backend_idx >= 0:
            self.exp_compute_backend_combo.setCurrentIndex(backend_idx)

        gpu_dtype = str(config.get("gpu_dtype", "float64")).lower()
        dtype_idx = self.exp_gpu_dtype_combo.findData(gpu_dtype)
        if dtype_idx >= 0:
            self.exp_gpu_dtype_combo.setCurrentIndex(dtype_idx)

        self._on_exp_backend_mode_changed()

        # Restore operator combos
        for combo, key in [
            (self.exp_crossover_combo, "crossover"),
            (self.exp_mutation_combo, "mutation"),
            (self.exp_selection_combo, "selection"),
            (self.exp_sampling_combo, "sampling"),
        ]:
            val = config.get(key)
            if isinstance(val, str):
                normalized = self._normalize_operator_value(key, val)
                idx = combo.findData(normalized)
                if idx < 0:
                    wanted = self._normalize_operator_class_token(
                        self._operator_class_name(key, normalized)
                    )
                    if wanted:
                        for i in range(combo.count()):
                            data_val = combo.itemData(i)
                            got = self._normalize_operator_class_token(
                                self._operator_class_name(key, data_val)
                            )
                            if got == wanted:
                                idx = i
                                break
                if idx >= 0:
                    combo.setCurrentIndex(idx)

        # Restore operator parameters
        if "crossover_eta" in config:
            self.exp_crossover_eta_spin.setValue(float(config["crossover_eta"]))
        if "crossover_prob" in config:
            self.exp_crossover_prob_spin.setValue(float(config["crossover_prob"]))
        if "mutation_eta" in config:
            self.exp_mutation_eta_spin.setValue(float(config["mutation_eta"]))
        if "mutation_prob" in config:
            val = config["mutation_prob"]
            self.exp_mutation_prob_spin.setValue(float(val) if val is not None else 0.0)
        if "selection_pressure" in config:
            self.exp_selection_pressure_spin.setValue(int(config["selection_pressure"]))

        raw_results_path_cfg = config.get("experiment_results_path", config.get("raw_results_path"))
        if raw_results_path_cfg is not None:
            self._apply_exp_results_storage_path(raw_results_path_cfg, load_if_exists=False)
        else:
            self._refresh_exp_results_storage_info()

        # Restore problem variants (duplicate entries of the same base problem)
        self.exp_problem_variants = {}
        raw_variants = config.get("problem_variants", {})
        if isinstance(raw_variants, dict):
            for variant_key, raw_entry in raw_variants.items():
                key = str(variant_key)
                if not isinstance(raw_entry, dict):
                    continue
                base_id = str(raw_entry.get("problem_id", "")).strip()
                if base_id not in self.problem_specs:
                    continue
                label = str(raw_entry.get("label", "")).strip() or f"{self.problem_specs[base_id].label} (variant)"
                self.exp_problem_variants[key] = {"problem_id": base_id, "label": label}

        # Legacy recovery: infer variants from problem_entries when problem_variants is absent.
        if not self.exp_problem_variants and isinstance(config.get("problem_entries"), list):
            for entry in config.get("problem_entries", []):
                if not isinstance(entry, dict):
                    continue
                instance_id = str(entry.get("instance_id", "")).strip()
                base_id = str(entry.get("problem_id", "")).strip()
                if not instance_id or not base_id or instance_id == base_id or base_id not in self.problem_specs:
                    continue
                label = str(entry.get("label", "")).strip() or f"{self.problem_specs[base_id].label} (variant)"
                self.exp_problem_variants[instance_id] = {"problem_id": base_id, "label": label}

        # Restore per-problem overrides
        self.exp_problem_overrides = {}
        raw_overrides = config.get("problem_overrides", {})
        if isinstance(raw_overrides, dict):
            for problem_id, raw_values in raw_overrides.items():
                pid = str(problem_id)
                if (
                    pid not in self.problem_specs
                    and pid not in self.exp_problem_variants
                ) or not isinstance(raw_values, dict):
                    continue
                normalized: dict[str, int] = {}
                for key in ("pop_size", "n_obj", "n_var"):
                    if key in raw_values:
                        try:
                            normalized[key] = int(raw_values[key])
                        except Exception:  # noqa: BLE001
                            continue
                if normalized:
                    self.exp_problem_overrides[pid] = normalized

        # Keep variant sequence monotonic after loading configs.
        max_variant_idx = 0
        for key in self.exp_problem_variants:
            match = re.search(r"::v(\d+)$", str(key))
            if match:
                try:
                    max_variant_idx = max(max_variant_idx, int(match.group(1)))
                except Exception:  # noqa: BLE001
                    pass
        self._exp_problem_variant_counter = max(self._exp_problem_variant_counter, max_variant_idx)

        # Restore per-algorithm operator overrides
        self.exp_algorithm_operator_overrides = {}
        raw_algo_overrides = config.get("algorithm_operator_overrides", {})
        if isinstance(raw_algo_overrides, dict):
            for algorithm_id, raw_values in raw_algo_overrides.items():
                aid = str(algorithm_id)
                if aid not in self.algorithm_specs or not isinstance(raw_values, dict):
                    continue
                normalized_ops: dict[str, Any] = {}
                for key in ("crossover", "mutation", "selection", "sampling"):
                    if key in raw_values and raw_values[key] is not None:
                        normalized_ops[key] = str(raw_values[key])
                for key in ("crossover_eta", "crossover_prob", "mutation_eta", "mutation_prob"):
                    if key in raw_values and raw_values[key] is not None:
                        try:
                            normalized_ops[key] = float(raw_values[key])
                        except Exception:  # noqa: BLE001
                            continue
                if "selection_pressure" in raw_values and raw_values["selection_pressure"] is not None:
                    try:
                        normalized_ops["selection_pressure"] = int(raw_values["selection_pressure"])
                    except Exception:  # noqa: BLE001
                        pass
                if normalized_ops:
                    self.exp_algorithm_operator_overrides[aid] = normalized_ops

        self.exp_use_pf_check.setChecked(bool(config.get("use_pf", True)))

        exp_seed_mode = str(config.get("seed_mode", SEED_MODE_RANDOM)).strip().lower()
        exp_seed_mode_idx = self.exp_seed_mode_combo.findData(exp_seed_mode)
        if exp_seed_mode_idx >= 0:
            self.exp_seed_mode_combo.setCurrentIndex(exp_seed_mode_idx)
        self.exp_seed_base_spin.setValue(_normalize_seed_value(config.get("seed_base", 1), default=1))
        self.exp_seed_step_spin.setValue(_positive_int(config.get("seed_step", 1), 1, minimum=1))
        self._on_exp_seed_mode_changed()

        stat_method = str(config.get("stat_test_method", "none")).lower()
        stat_method_idx = self.stat_method_combo.findData(stat_method)
        if stat_method_idx >= 0:
            self.stat_method_combo.setCurrentIndex(stat_method_idx)
        stat_reference = str(config.get("stat_test_reference", "")).strip()
        if stat_reference and self.stat_reference_combo.findText(stat_reference) >= 0:
            self.stat_reference_combo.setCurrentText(stat_reference)
        stat_alpha = _float_or_none(config.get("stat_test_alpha"))
        if stat_alpha is not None and math.isfinite(stat_alpha):
            self.stat_alpha_spin.setValue(float(stat_alpha))
        self._update_stat_controls_state()

        self._on_exp_backend_mode_changed()

        # Trigger dynamic visibility
        self._on_exp_crossover_changed()
        self._on_exp_mutation_changed()
        self._on_exp_selection_changed()
        self._refresh_exp_operator_cfg_targets()
        self._refresh_exp_problem_cfg_targets()
        self._refresh_exp_workload_summary()

    def _build_results_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        # Keep selection, validation, and file actions in independent rows.
        # The former single horizontal row overflowed at laptop widths and
        # obscured the primary action behind the left window edge.
        analysis_bar = QGroupBox("Analysis Scope")
        analysis_bar.setObjectName("resultsToolbar")
        analysis_grid = QGridLayout(analysis_bar)
        analysis_grid.setContentsMargins(12, 12, 12, 10)
        analysis_grid.setHorizontalSpacing(8)
        analysis_grid.setVerticalSpacing(8)

        self.metric_plot_combo = QComboBox()
        self.metric_plot_combo.setMinimumWidth(150)
        self.metric_plot_combo.currentTextChanged.connect(self._refresh_convergence_chart)
        self.metric_plot_combo.currentTextChanged.connect(self._refresh_results_table)
        analysis_grid.addWidget(QLabel("Metric"), 0, 0)
        analysis_grid.addWidget(self.metric_plot_combo, 0, 1)

        self.problem_plot_combo = QComboBox()
        self.problem_plot_combo.setMinimumWidth(150)
        self.problem_plot_combo.currentIndexChanged.connect(self._on_plot_problem_changed)
        self.problem_plot_combo.currentIndexChanged.connect(self._refresh_results_table)
        analysis_grid.addWidget(QLabel("Problem"), 0, 2)
        analysis_grid.addWidget(self.problem_plot_combo, 0, 3)

        self.algo_plot_combo = QComboBox()
        self.algo_plot_combo.setMinimumWidth(150)
        self.algo_plot_combo.currentTextChanged.connect(self._on_plot_algo_changed)
        analysis_grid.addWidget(QLabel("Algorithm"), 0, 4)
        analysis_grid.addWidget(self.algo_plot_combo, 0, 5)

        self.run_plot_spin = QSpinBox()
        self.run_plot_spin.setRange(1, 1)
        self.run_plot_spin.valueChanged.connect(self._refresh_pareto_chart)
        self.run_plot_spin.setVisible(False)

        self.stat_method_label = QLabel("Statistical method")
        analysis_grid.addWidget(self.stat_method_label, 1, 0)
        self.stat_method_combo = QComboBox()
        self.stat_method_combo.setMinimumWidth(190)
        self.stat_method_combo.addItem("None (Detailed Runs)", "none")
        self.stat_method_combo.addItem("Wilcoxon vs Reference", "wilcoxon")
        self.stat_method_combo.addItem("Global Friedman Rank", "friedman")
        self.stat_method_combo.setCurrentIndex(0)
        self.stat_method_combo.currentIndexChanged.connect(self._on_stat_controls_changed)
        analysis_grid.addWidget(self.stat_method_combo, 1, 1)

        self.stat_reference_label = QLabel("Reference")
        analysis_grid.addWidget(self.stat_reference_label, 1, 2)
        self.stat_reference_combo = QComboBox()
        self.stat_reference_combo.setMinimumWidth(150)
        self.stat_reference_combo.setEnabled(False)
        self.stat_reference_combo.currentIndexChanged.connect(self._on_stat_controls_changed)
        analysis_grid.addWidget(self.stat_reference_combo, 1, 3)

        self.stat_alpha_label = QLabel("Significance level (α)")
        analysis_grid.addWidget(self.stat_alpha_label, 1, 4)
        self.stat_alpha_spin = QDoubleSpinBox()
        self.stat_alpha_spin.setRange(0.001, 0.5)
        self.stat_alpha_spin.setDecimals(3)
        self.stat_alpha_spin.setSingleStep(0.005)
        self.stat_alpha_spin.setValue(0.05)
        self.stat_alpha_spin.valueChanged.connect(self._on_stat_controls_changed)
        analysis_grid.addWidget(self.stat_alpha_spin, 1, 5)

        self.btn_run_stats = QPushButton("Run Statistical Validation")
        self.btn_run_stats.setIcon(app_icon("fact_check"))
        self.btn_run_stats.setStyleSheet(AppStyles.primary_button_style())
        self.btn_run_stats.setToolTip("Run the selected statistical validation over the current analysis scope.")
        self.btn_run_stats.clicked.connect(self._run_statistical_validation)
        analysis_grid.addWidget(self.btn_run_stats, 1, 6)
        analysis_grid.setColumnStretch(7, 1)
        layout.addWidget(analysis_bar)

        file_actions = QHBoxLayout()
        file_actions.setSpacing(8)
        file_actions.addStretch(1)

        self.btn_load_history = QPushButton("Load History")
        self.btn_load_history.setIcon(app_icon("history"))
        self.btn_load_history.setStyleSheet(AppStyles.secondary_button_style())
        self.btn_load_history.setToolTip("Load saved runs from results.json")
        self.btn_load_history.clicked.connect(self._load_results_history)
        file_actions.addWidget(self.btn_load_history)

        self.btn_export_csv = QPushButton("Export CSV")
        self.btn_export_csv.setIcon(app_icon("description"))
        self.btn_export_csv.setStyleSheet(AppStyles.secondary_button_style())
        self.btn_export_csv.setToolTip("Export the active result table as CSV.")
        self.btn_export_csv.clicked.connect(self._export_csv)
        file_actions.addWidget(self.btn_export_csv)

        self.btn_export_latex = QPushButton("Export LaTeX")
        self.btn_export_latex.setIcon(app_icon("functions"))
        self.btn_export_latex.setStyleSheet(AppStyles.secondary_button_style())
        self.btn_export_latex.setToolTip("Export the active result table as LaTeX.")
        self.btn_export_latex.clicked.connect(self._export_latex)
        file_actions.addWidget(self.btn_export_latex)

        self.btn_clear_history = QPushButton("Clear History")
        self.btn_clear_history.setIcon(app_icon("delete"))
        self.btn_clear_history.setStyleSheet(AppStyles.danger_button_style())
        self.btn_clear_history.setToolTip("Permanently delete saved runs from results.json.")
        self.btn_clear_history.clicked.connect(self._clear_results_history)
        file_actions.addWidget(self.btn_clear_history)

        layout.addLayout(file_actions)

        mcdm_row = QHBoxLayout()
        mcdm_row.setSpacing(8)
        mcdm_row.addWidget(QLabel("MCDM:"))

        self.mcdm_method_combo = QComboBox()
        self.mcdm_method_combo.addItem("TOPSIS", "topsis")
        self.mcdm_method_combo.addItem("Weighted Sum (normalized)", "weighted_sum")
        mcdm_row.addWidget(self.mcdm_method_combo)

        self.mcdm_weights_edit = QLineEdit()
        self.mcdm_weights_edit.setStyleSheet(AppStyles.search_input_style())
        self.mcdm_weights_edit.setPlaceholderText("weights, e.g. 0.5,0.5")
        mcdm_row.addWidget(self.mcdm_weights_edit)

        self.mcdm_apply_btn = QPushButton("Apply MCDM")
        self.mcdm_apply_btn.setStyleSheet(AppStyles.primary_button_style())
        self.mcdm_apply_btn.clicked.connect(self._run_mcdm_selection)
        mcdm_row.addWidget(self.mcdm_apply_btn)
        mcdm_row.addStretch(1)
        layout.addLayout(mcdm_row)

        self.mcdm_result_label = QLabel("MCDM result: not computed")
        self.mcdm_result_label.setStyleSheet(AppStyles.status_pill_style(AppStyles.TEXT_MUTED))

        self.mcdm_info_label = QLabel(
            "MCDM note: objectives are treated as minimization targets; weights are normalized before scoring."
        )
        self.mcdm_info_label.setStyleSheet(AppStyles.panel_hint_style())
        self.mcdm_info_label.setWordWrap(True)

        self.mcdm_details_box = QPlainTextEdit()
        self.mcdm_details_box.setReadOnly(True)
        self.mcdm_details_box.setPlaceholderText(
            "MCDM details will appear here after a solution is selected."
        )
        self.mcdm_details_box.setMaximumHeight(130)
        self.mcdm_details_box.setStyleSheet(AppStyles.log_view_style())

        self.stats_header_label = QLabel("EmoPyLab | Statistical View: Detailed Runs")
        self.stats_header_label.setStyleSheet(f"color: {AppStyles.TEXT_MUTED}; font-weight: 600;")
        self.stats_header_label.setWordWrap(True)

        self.stats_warning_label = QLabel("")
        self.stats_warning_label.setStyleSheet(f"color: {AppStyles.WARNING};")
        self.stats_warning_label.setWordWrap(True)

        self.results_table = QTableWidget()
        self.results_table.setObjectName("resultsTable")
        self.results_table.setAlternatingRowColors(False)  # Keep rows visually neutral for readability.
        self.results_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.results_table.setStyleSheet(AppStyles.table_style("resultsTable"))
        self.results_table.cellClicked.connect(self._on_results_table_cell_clicked)

        self.conv_chart_view = QChartView()
        self.conv_chart_view.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.conv_chart_view.setRubberBand(QChartView.RubberBand.RectangleRubberBand)
        self.conv_chart_view.setAccessibleName("Convergence analysis chart")
        self.conv_chart_view.setToolTip("Drag to zoom. Double-click to reset the chart view.")

        self.pareto_chart_view = QChartView()
        self.pareto_chart_view.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.pareto_chart_view.setRubberBand(QChartView.RubberBand.RectangleRubberBand)
        self.pareto_chart_view.setAccessibleName("Pareto front analysis chart")
        self.pareto_chart_view.setToolTip("Drag to zoom. Double-click to reset the chart view.")

        self.pareto_3d_container = QWidget()
        self.pareto_3d_layout = QVBoxLayout(self.pareto_3d_container)
        self.pareto_3d_layout.setContentsMargins(0, 0, 0, 0)

        self.pareto_chart_stack = QStackedWidget()
        self.pareto_chart_stack.addWidget(self.pareto_chart_view)    # page 0
        self.pareto_chart_stack.addWidget(self.pareto_3d_container)  # page 1

        # Results Display Sub-Tabs: Organizes Table, Convergence, Pareto, and Side-by-Side views
        # preventing compressed/crushed charts on laptops and high-D displays.
        self.results_view_tabs = QTabWidget()
        self.results_view_tabs.setObjectName("resultsViewTabs")
        self.results_view_tabs.setStyleSheet(f"""
            QTabWidget#resultsViewTabs::pane {{
                border: 1px solid {StylesAppStyles.colors.border_light};
                border-radius: 8px;
                background-color: {StylesAppStyles.colors.surface};
                padding: 6px;
            }}
            QTabBar::tab {{
                background: {StylesAppStyles.colors.surface_variant};
                color: {StylesAppStyles.colors.text_secondary};
                border: 1px solid {StylesAppStyles.colors.border_light};
                border-bottom: 0px;
                border-top-left-radius: 6px;
                border-top-right-radius: 6px;
                min-width: 140px;
                padding: 7px 12px;
                margin-right: 4px;
                font-weight: 600;
            }}
            QTabBar::tab:selected {{
                background: {StylesAppStyles.colors.surface};
                color: {StylesAppStyles.colors.primary};
                border-color: {StylesAppStyles.colors.border_light};
                border-bottom: 2px solid {StylesAppStyles.colors.primary};
            }}
            QTabBar::tab:hover:!selected {{
                background: {StylesAppStyles.colors.primary_subtle};
                color: {StylesAppStyles.colors.primary_dark};
            }}
        """)

        # --- Sub-Tab 1: Detailed Table & MCDM View ---
        table_container = QWidget()
        table_layout = QVBoxLayout(table_container)
        table_layout.setContentsMargins(4, 6, 4, 4)
        table_layout.setSpacing(6)

        table_layout.addWidget(self.stats_header_label)
        table_layout.addWidget(self.stats_warning_label)
        table_layout.addWidget(self.results_table, 1)

        table_layout.addWidget(self.mcdm_result_label)
        table_layout.addWidget(self.mcdm_info_label)
        table_layout.addWidget(self.mcdm_details_box)

        self.results_view_tabs.addTab(table_container, app_icon("table_chart"), "Statistical Table & MCDM")

        # --- Sub-Tab 2: Full-Height Convergence Chart ---
        conv_container = QWidget()
        conv_layout = QVBoxLayout(conv_container)
        conv_layout.setContentsMargins(4, 6, 4, 4)
        conv_layout.addWidget(self.conv_chart_view, 1)
        self.results_view_tabs.addTab(conv_container, app_icon("show_chart"), "Convergence Curve")

        # --- Sub-Tab 3: Full-Height Pareto Front (2D/3D) ---
        pareto_container = QWidget()
        pareto_layout = QVBoxLayout(pareto_container)
        pareto_layout.setContentsMargins(4, 6, 4, 4)
        pareto_layout.addWidget(self.pareto_chart_stack, 1)
        self.results_view_tabs.addTab(pareto_container, app_icon("bubble_chart"), "Pareto Front (2D / 3D)")

        layout.addWidget(self.results_view_tabs, 1)

        self._set_empty_convergence_chart()
        self._set_empty_pareto_chart()
        self._update_stat_controls_state()

        return tab

    @Slot()
    def _run_mcdm_selection(self) -> None:
        algo_name = self.algo_plot_combo.currentText().strip()
        if not algo_name:
            QMessageBox.warning(self, "MCDM", "Select an algorithm first.")
            return

        runs = self._runs_for_plot(algo_name)
        if not runs:
            QMessageBox.warning(self, "MCDM", "No runs available for the selected algorithm/problem.")
            return

        run_idx = max(1, int(self.run_plot_spin.value())) - 1
        run_idx = min(run_idx, len(runs) - 1)
        run_payload = runs[run_idx]
        front = np.asarray(run_payload.get("final_front", []), dtype=float)
        if front.ndim != 2 or front.size == 0:
            QMessageBox.warning(self, "MCDM", "Selected run has no Pareto front points.")
            return

        method = str(self.mcdm_method_combo.currentData() or "topsis")
        raw_weights = self.mcdm_weights_edit.text().strip()
        try:
            decision = select_compromise_solution(front=front, method=method, weights_text=raw_weights)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.warning(self, "MCDM", str(exc))
            return

        best_idx = int(decision["index"])
        best_score = float(decision["score"])
        selected = np.asarray(decision["selected"], dtype=float)
        decision_saved_path = self.mcdm_decision_storage_path
        decision_snapshot = {
            "method_id": str(decision.get("method", method)),
            "method_label": str(self.mcdm_method_combo.currentText()).strip(),
            "weights_input": raw_weights,
            "weights_normalized": np.asarray(decision.get("weights", []), dtype=float).tolist(),
            "score": best_score,
            "front_index": best_idx,
            "selected_point": selected.tolist(),
            "n_points_front": int(front.shape[0]),
            "n_obj": int(front.shape[1]),
            "run_ref": {
                "algorithm": str(run_payload.get("algorithm", "")),
                "algorithm_id": str(run_payload.get("algorithm_id", "")),
                "problem": str(run_payload.get("problem", "")),
                "problem_id": str(run_payload.get("problem_id", "")),
                "run_index": run_payload.get("run_index"),
                "seed": run_payload.get("seed"),
                "timestamp_iso": str(run_payload.get("timestamp_iso", "")),
                "timestamp_en_us": str(run_payload.get("timestamp_en_us", "")),
                "timestamp_epoch": _float_or_none(run_payload.get("timestamp_epoch")),
            },
        }
        self.mcdm_last_decision = dict(decision_snapshot)
        ok_mcdm_save, mcdm_save_message = self._save_mcdm_decision_sidecar(decision_snapshot)
        if hasattr(self, "_append_log"):
            self._append_log(mcdm_save_message)

        score_label = f"score={best_score:.6g}"
        values_label = ", ".join(f"f{i+1}={v:.6g}" for i, v in enumerate(selected))
        self.mcdm_result_label.setText(
            f"MCDM result ({self.mcdm_method_combo.currentText()}): point #{best_idx + 1} | {score_label} | {values_label}"
        )
        self.mcdm_result_label.setStyleSheet(AppStyles.status_pill_style(AppStyles.SUCCESS))
        self.mcdm_result_label.setToolTip(
            str(decision_saved_path) if ok_mcdm_save else mcdm_save_message
        )
        self._refresh_mcdm_details_panel(run_payload)
        self._refresh_pareto_chart()

    def _save_mcdm_decision_sidecar(self, decision_snapshot: dict[str, Any]) -> tuple[bool, str]:
        path = Path(getattr(self, "mcdm_decision_storage_path", self.base_dir / "mcdm_results" / "last_mcdm_decision.json"))
        try:
            now_dt = datetime.now().astimezone()
            wrapper = {
                "kind": "emopylab_mcdm_decision",
                "schema_version": 1,
                "saved_at_iso": now_dt.isoformat(),
                "saved_at_en_us": format_timestamp_en_us(now_dt),
                "payload": dict(decision_snapshot),
            }
            self._json_write_with_numpy(path, wrapper)
            return True, f"MCDM decision saved: {path}"
        except Exception as exc:  # noqa: BLE001
            return False, f"Warning: could not save MCDM decision - {exc}"

    def _mcdm_decision_for_run(self, run_payload: dict[str, Any]) -> dict[str, Any] | None:
        decision = getattr(self, "mcdm_last_decision", None)
        if not isinstance(decision, dict):
            return None
        run_ref = decision.get("run_ref")
        if not isinstance(run_ref, dict):
            return None
        try:
            if self._same_run_identity(dict(run_payload), dict(run_ref)):
                return decision
        except Exception:  # noqa: BLE001
            return None
        return None

    def _mcdm_current_plot_run_payload(self) -> dict[str, Any] | None:
        if not hasattr(self, "algo_plot_combo") or not hasattr(self, "run_plot_spin"):
            return None
        algo_name = self.algo_plot_combo.currentText().strip()
        if not algo_name:
            return None
        runs = self._runs_for_plot(algo_name)
        if not runs:
            return None
        run_index = int(self.run_plot_spin.value()) - 1
        if run_index < 0 or run_index >= len(runs):
            return None
        run_payload = runs[run_index]
        return dict(run_payload) if isinstance(run_payload, dict) else None

    def _mcdm_match_selected_point_to_population(
        self,
        run_payload: dict[str, Any],
        selected_point: np.ndarray,
    ) -> dict[str, Any]:
        result: dict[str, Any] = {
            "matched": False,
            "match_kind": "",
            "message": "",
            "population_index": None,
            "population_f": None,
            "population_x": None,
            "candidate_count": 0,
        }
        if not isinstance(run_payload, dict):
            result["message"] = "Invalid run payload."
            return result

        pop = run_payload.get("final_population")
        if not isinstance(pop, dict):
            result["message"] = "final_population snapshot is unavailable."
            return result

        try:
            pop_f = np.asarray(pop.get("F"), dtype=float)
        except Exception:  # noqa: BLE001
            pop_f = np.asarray([], dtype=float)
        if pop_f.ndim == 1 and pop_f.size:
            pop_f = pop_f.reshape(1, -1)
        if pop_f.ndim != 2 or pop_f.size == 0:
            result["message"] = "Population objective matrix F is unavailable."
            return result

        point = np.asarray(selected_point, dtype=float).reshape(-1)
        if point.size == 0:
            result["message"] = "Selected MCDM point is empty."
            return result

        n_cols = min(int(pop_f.shape[1]), int(point.size))
        if n_cols <= 0:
            result["message"] = "Objective dimensions do not match."
            return result

        cmp_f = pop_f[:, :n_cols]
        cmp_point = point[:n_cols]
        exact_mask = np.all(
            np.isclose(cmp_f, cmp_point.reshape(1, -1), rtol=1e-8, atol=1e-10),
            axis=1,
        )
        exact_idx = np.flatnonzero(exact_mask)
        chosen_idx: int | None = None
        match_kind = ""
        candidate_count = int(exact_idx.size)

        if exact_idx.size > 0:
            chosen_idx = int(exact_idx[0])
            match_kind = "exact"
        else:
            try:
                d = np.linalg.norm(cmp_f - cmp_point.reshape(1, -1), axis=1)
                if d.size > 0 and np.any(np.isfinite(d)):
                    chosen_idx = int(np.nanargmin(d))
                    match_kind = "nearest"
            except Exception:  # noqa: BLE001
                chosen_idx = None

        if chosen_idx is None or chosen_idx < 0 or chosen_idx >= int(pop_f.shape[0]):
            result["message"] = "Could not map selected point to final_population rows."
            return result

        try:
            pop_x_arr = np.asarray(pop.get("X"), dtype=float)
        except Exception:  # noqa: BLE001
            pop_x_arr = np.asarray([], dtype=float)
        if pop_x_arr.ndim == 1 and pop_x_arr.size:
            pop_x_arr = pop_x_arr.reshape(1, -1)
        if pop_x_arr.ndim != 2 or pop_x_arr.shape[0] <= chosen_idx:
            pop_x_value = None
        else:
            pop_x_value = np.asarray(pop_x_arr[chosen_idx], dtype=float).tolist()

        result.update(
            {
                "matched": True,
                "match_kind": match_kind,
                "message": (
                    "Matched by exact objective-vector equality."
                    if match_kind == "exact"
                    else "Matched by nearest objective vector in final_population.F."
                ),
                "population_index": int(chosen_idx),
                "population_f": np.asarray(pop_f[chosen_idx], dtype=float).tolist(),
                "population_x": pop_x_value,
                "candidate_count": candidate_count,
            }
        )
        if match_kind == "exact" and candidate_count > 1:
            result["message"] = (
                f"Matched by exact objective-vector equality ({candidate_count} candidates; first row selected)."
            )
        return result

    def _refresh_mcdm_details_panel(self, run_payload: dict[str, Any] | None) -> None:
        box = getattr(self, "mcdm_details_box", None)
        if box is None:
            return

        if not isinstance(run_payload, dict):
            run_payload = self._mcdm_current_plot_run_payload()

        if not isinstance(run_payload, dict):
            box.setPlainText("No active run is selected for MCDM details.")
            return

        decision = self._mcdm_decision_for_run(run_payload)
        if not isinstance(decision, dict):
            box.setPlainText("No MCDM decision is stored for the selected run.")
            return

        selected_point = np.asarray(decision.get("selected_point", []), dtype=float).reshape(-1)
        match_info = self._mcdm_match_selected_point_to_population(run_payload, selected_point)

        lines: list[str] = []
        method_id = str(decision.get("method", decision.get("method_id", "-")))
        lines.append(f"Method: {method_id}")

        weights = np.asarray(decision.get("weights", decision.get("weights_normalized", [])), dtype=float).reshape(-1)
        if weights.size > 0:
            lines.append("Weights (normalized): " + ", ".join(_fmt_number(v, 6) for v in weights.tolist()))

        score = _float_or_none(decision.get("score"))
        if score is not None and math.isfinite(score):
            lines.append(f"Score: {_fmt_number(score, 8)}")

        front_index = decision.get("index", decision.get("front_index"))
        if front_index is not None:
            lines.append(f"Front index: {front_index}")

        if selected_point.size > 0:
            lines.append(
                "Selected objectives (F): "
                + ", ".join(f"f{i+1}={_fmt_number(v, 8)}" for i, v in enumerate(selected_point.tolist()))
            )

        lines.append(
            f"Run: {run_payload.get('algorithm', '-')} | {run_payload.get('problem', '-')} | trial {run_payload.get('run_index', '-')}"
        )

        if bool(match_info.get("matched")):
            lines.append(f"Population row: {match_info.get('population_index')} ({match_info.get('match_kind', '-')})")
            match_msg = str(match_info.get("message", "")).strip()
            if match_msg:
                lines.append(f"Match: {match_msg}")

            pop_x = match_info.get("population_x")
            if isinstance(pop_x, list) and pop_x:
                x_vals = [float(v) for v in pop_x]
                preview_n = 12
                preview = x_vals[:preview_n]
                preview_text = ", ".join(f"x{i+1}={_fmt_number(v, 8)}" for i, v in enumerate(preview))
                if len(x_vals) > preview_n:
                    preview_text += f", ... ({len(x_vals)} vars)"
                lines.append("Decision vars (X): " + preview_text)
            else:
                lines.append("Decision vars (X): unavailable in final_population snapshot.")
        else:
            lines.append("Population match: " + (str(match_info.get("message", "")).strip() or "unavailable"))

        box.setPlainText("\n".join(lines))

    def _append_registry_summary(self) -> None:
        num_builtin_problems = sum(1 for spec in self.problem_specs.values() if spec.source == "pymoo")
        num_local_problems = sum(1 for spec in self.problem_specs.values() if _is_custom_source(spec.source))
        num_builtin_algorithms = sum(1 for spec in self.algorithm_specs.values() if spec.source == "pymoo")
        num_local_algorithms = sum(1 for spec in self.algorithm_specs.values() if _is_custom_source(spec.source))
        num_builtin_metrics = sum(1 for spec in self.metric_specs.values() if spec.source == "pymoo")
        num_local_metrics = sum(1 for spec in self.metric_specs.values() if _is_custom_source(spec.source))
        num_ops = {
            category: len(specs)
            for category, specs in self.operator_specs.items()
        }

        self.problem_count_label.setText(
            f"pymoo: {num_builtin_problems} | local: {num_local_problems}"
        )

        self._append_log(
            "Catalog loaded -> "
            f"algorithms: {num_builtin_algorithms} pymoo + {num_local_algorithms} local, "
            f"problems: {num_builtin_problems} pymoo + {num_local_problems} local, "
            f"metrics: {num_builtin_metrics} pymoo + {num_local_metrics} local, "
            f"operators: cx={num_ops.get('crossover', 0)}, mut={num_ops.get('mutation', 0)}, "
            f"sel={num_ops.get('selection', 0)}, samp={num_ops.get('sampling', 0)}, "
            f"backend-aware={self._backend_aware_loading_enabled()} stage={self._backend_rollout_stage()}"
        )

    def _append_log(self, text: str) -> None:
        timestamp = format_timestamp_en_us()
        self.log_box.appendPlainText(f"[{timestamp}] {text}")

    def _reload_registries(self) -> None:
        if self.test_worker_thread is not None or self.exp_worker_thread is not None:
            QMessageBox.information(self, "Busy", "Stop current execution before reloading registries.")
            return

        (
            self.algorithm_specs,
            self.problem_specs,
            self.metric_specs,
            self.discovery_warnings,
        ) = discover_all_specs(self.base_dir)
        self.operator_specs = discover_operator_specs(self.discovery_warnings, self.base_dir)
        self._rebuild_trait_maps()

        self._populate_problem_combo()
        self._populate_algorithm_list()
        self._populate_metric_list()
        self._populate_exp_lists()
        self._populate_all_operator_combos()
        self._on_crossover_changed()
        self._on_mutation_changed()
        self._on_selection_changed()
        self._on_exp_crossover_changed()
        self._on_exp_mutation_changed()
        self._on_exp_selection_changed()

        self._append_registry_summary()
        for message in self.discovery_warnings:
            self._append_log(f"Registry warning: {message}")

    def _populate_problem_combo(self) -> None:
        current = self.problem_combo.currentData()
        checked_ids = set(self._checked_ids(self.experiment_problem_list))
        self.problem_combo.blockSignals(True)
        self.problem_combo.clear()

        sorted_specs = self._iter_visible_problem_specs(
            prefer_jax=self._test_backend_prefers_jax_for("problems")
        )
        for spec in sorted_specs:
            self.problem_combo.addItem(spec.label, spec.id)

        target_id = current
        if target_id is None or self.problem_combo.findData(target_id) < 0:
            zdt_index = -1
            for i in range(self.problem_combo.count()):
                spec_id = self.problem_combo.itemData(i)
                spec = self.problem_specs.get(spec_id)
                if spec is not None and core_normalize_backend_token(spec.name) == "zdt1":
                    zdt_index = i
                    break
            if zdt_index >= 0:
                self.problem_combo.setCurrentIndex(zdt_index)
            elif self.problem_combo.count() > 0:
                self.problem_combo.setCurrentIndex(0)
        else:
            self.problem_combo.setCurrentIndex(self.problem_combo.findData(target_id))

        self.problem_combo.blockSignals(False)
        self._populate_experiment_problem_list(checked_ids)
        self._populate_problem_catalog_list()
        self._on_problem_changed()
        self._apply_catalog_filters()

    def _populate_experiment_problem_list(self, checked_ids: set[str] | None = None) -> None:
        checked_ids = set(checked_ids or set())
        current_problem_id = self.problem_combo.currentData()
        if not checked_ids and isinstance(current_problem_id, str):
            checked_ids.add(current_problem_id)

        self.experiment_problem_list.clear()
        sorted_specs = self._iter_visible_problem_specs(
            prefer_jax=self._test_backend_prefers_jax_for("problems")
        )
        for spec in sorted_specs:
            item = QListWidgetItem(spec.label)
            item.setData(Qt.ItemDataRole.UserRole, spec.id)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Checked if spec.id in checked_ids else Qt.CheckState.Unchecked)
            self.experiment_problem_list.addItem(item)

        if not self._checked_ids(self.experiment_problem_list) and self.experiment_problem_list.count() > 0:
            self.experiment_problem_list.item(0).setCheckState(Qt.CheckState.Checked)

    def _populate_problem_catalog_list(self) -> None:
        target_id = self.problem_combo.currentData()
        self.problem_catalog_list.blockSignals(True)
        self.problem_catalog_list.clear()

        sorted_specs = self._iter_visible_problem_specs(
            prefer_jax=self._test_backend_prefers_jax_for("problems")
        )
        target_item: QListWidgetItem | None = None
        for spec in sorted_specs:
            item = QListWidgetItem(spec.label)
            item.setData(Qt.ItemDataRole.UserRole, spec.id)
            self.problem_catalog_list.addItem(item)
            if isinstance(target_id, str) and spec.id == target_id:
                target_item = item

        if target_item is None and self.problem_catalog_list.count() > 0:
            target_item = self.problem_catalog_list.item(0)

        if target_item is not None:
            self.problem_catalog_list.setCurrentItem(target_item)
        self.problem_catalog_list.blockSignals(False)

    def _populate_algorithm_list(self) -> None:
        self.algorithm_list.blockSignals(True)
        self.algorithm_list.clear()
        sorted_specs = sorted(
            self.algorithm_specs.values(),
            key=lambda spec: _natural_lexicographic_key(spec.name),
        )
        target_item: QListWidgetItem | None = None
        for spec in sorted_specs:
            item = QListWidgetItem(spec.label)
            item.setData(Qt.ItemDataRole.UserRole, spec.id)
            self.algorithm_list.addItem(item)
            if target_item is None and core_normalize_backend_token(spec.name) == "nsga3":
                target_item = item

        if target_item is None and self.algorithm_list.count() > 0:
            target_item = self.algorithm_list.item(0)
        if target_item is not None:
            self.algorithm_list.setCurrentItem(target_item)
        self.algorithm_list.blockSignals(False)
        self._apply_catalog_filters()
        self._update_selection_cards()

    def _populate_metric_list(self, checked_ids: set[str] | None = None) -> None:
        prefer_jax = self._test_backend_prefers_jax_for("metrics")
        mapped_ids = set(
            self._map_metric_ids_to_backend(checked_ids or set(), prefer_jax=prefer_jax)
        )

        self.metric_list.blockSignals(True)
        self.metric_list.clear()
        visible_specs = self._iter_metric_specs_for_backend(prefer_jax=prefer_jax)
        for spec in visible_specs:
            item = QListWidgetItem(spec.label)
            item.setData(Qt.ItemDataRole.UserRole, spec.id)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            is_checked = spec.id in mapped_ids if mapped_ids else self._is_default_metric_checked(spec)
            item.setCheckState(Qt.CheckState.Checked if is_checked else Qt.CheckState.Unchecked)
            self.metric_list.addItem(item)
        self.metric_list.blockSignals(False)
        self._sync_test_metric_combo()

    def _refresh_test_metric_list_for_backend(self) -> None:
        if not hasattr(self, "metric_list"):
            return
        checked_ids = set(self._checked_ids(self.metric_list))
        self._populate_metric_list(checked_ids=checked_ids if checked_ids else None)

    def _refresh_exp_metric_list_for_backend(self) -> None:
        if not hasattr(self, "exp_metric_list"):
            return
        checked_ids = set(self._exp_checked_ids(self.exp_metric_list))
        self._populate_exp_metric_list(checked_ids=checked_ids if checked_ids else None)
        self._apply_exp_list_filters()
        self._recompute_exp_metric_samples_from_results()

    def _filter_algorithm_items(self, text: str) -> None:
        _ = text
        self._apply_catalog_filters()

    def _apply_exp_list_filters(self, *_args: Any) -> None:
        if hasattr(self, "exp_algorithm_list"):
            query = ""
            if hasattr(self, "exp_algorithm_filter"):
                query = self.exp_algorithm_filter.text().strip().lower()
            for i in range(self.exp_algorithm_list.count()):
                item = self.exp_algorithm_list.item(i)
                item.setHidden(bool(query) and query not in item.text().lower())

        if hasattr(self, "exp_problem_list"):
            query = ""
            if hasattr(self, "exp_problem_filter"):
                query = self.exp_problem_filter.text().strip().lower()
            for i in range(self.exp_problem_list.count()):
                item = self.exp_problem_list.item(i)
                item.setHidden(bool(query) and query not in item.text().lower())

        if hasattr(self, "exp_metric_list"):
            query = ""
            if hasattr(self, "exp_metric_filter"):
                query = self.exp_metric_filter.text().strip().lower()
            for i in range(self.exp_metric_list.count()):
                item = self.exp_metric_list.item(i)
                item.setHidden(bool(query) and query not in item.text().lower())

    def _on_algorithm_selected(
        self,
        _current: QListWidgetItem | None,
        _previous: QListWidgetItem | None,
    ) -> None:
        self._populate_all_operator_combos()
        self._update_selection_cards()
        # Refresh result controls but only auto-focus when the selected pair exists in history.
        self._refresh_test_result_controls(autoload_chart=False)
        self._sync_test_result_to_current_selection_if_available()
        self._apply_catalog_filters()

    def _on_problem_catalog_selected(
        self,
        current: QListWidgetItem | None,
        _previous: QListWidgetItem | None,
    ) -> None:
        if current is None:
            return
        problem_id = current.data(Qt.ItemDataRole.UserRole)
        if not isinstance(problem_id, str):
            return
        combo_index = self.problem_combo.findData(problem_id)
        if combo_index >= 0 and combo_index != self.problem_combo.currentIndex():
            self.problem_combo.setCurrentIndex(combo_index)
        else:
            self._update_selection_cards()
            self._refresh_test_result_controls()
            self._apply_catalog_filters()

    def _active_filter_values(self, button_map: dict[str, QPushButton]) -> set[str]:
        return {key for key, button in button_map.items() if button.isChecked()}

    def _selected_algorithm_id(self) -> str | None:
        item = self.algorithm_list.currentItem()
        if item is None:
            return None
        value = item.data(Qt.ItemDataRole.UserRole)
        if isinstance(value, str):
            return value
        return None

    def _traits_match(
        self,
        trait_info: dict[str, Any] | None,
        selected_objectives: set[str],
        selected_encodings: set[str],
        selected_difficulties: set[str],
    ) -> bool:
        if not trait_info:
            return True

        objective = str(trait_info.get("objective", "")).strip().lower()
        if selected_objectives and objective not in selected_objectives:
            return False

        encodings = trait_info.get("encoding", set())
        if not isinstance(encodings, set):
            encodings = set(encodings) if isinstance(encodings, (list, tuple, set)) else set()
        if selected_encodings and not encodings.intersection(selected_encodings):
            return False

        difficulties = trait_info.get("difficulty", set())
        if not isinstance(difficulties, set):
            difficulties = set(difficulties) if isinstance(difficulties, (list, tuple, set)) else set()
        if selected_difficulties and not difficulties.intersection(selected_difficulties):
            return False

        return True

    def _apply_catalog_filters(self, *_args: Any) -> None:
        selected_objectives = self._active_filter_values(self.objective_filter_buttons)
        selected_encodings = self._active_filter_values(self.encoding_filter_buttons)
        selected_difficulties = self._active_filter_values(self.difficulty_filter_buttons)

        algorithm_query = self.algorithm_filter.text().strip().lower()
        visible_algorithms = 0
        for i in range(self.algorithm_list.count()):
            item = self.algorithm_list.item(i)
            item_id = item.data(Qt.ItemDataRole.UserRole)
            traits = self.algorithm_traits.get(item_id) if isinstance(item_id, str) else None
            matches = self._traits_match(traits, selected_objectives, selected_encodings, selected_difficulties)
            if algorithm_query and algorithm_query not in item.text().lower():
                matches = False
            item.setHidden(not matches)
            if matches:
                visible_algorithms += 1

        selected_algo_item = self.algorithm_list.currentItem()
        if selected_algo_item is None or selected_algo_item.isHidden():
            for i in range(self.algorithm_list.count()):
                item = self.algorithm_list.item(i)
                if not item.isHidden():
                    self.algorithm_list.setCurrentItem(item)
                    break

        problem_query = self.problem_filter.text().strip().lower()
        jax_mode_active = self._test_backend_prefers_jax_for("problems")
        visible_problems = 0
        for i in range(self.problem_catalog_list.count()):
            item = self.problem_catalog_list.item(i)
            item_id = item.data(Qt.ItemDataRole.UserRole)
            traits = self.problem_traits.get(item_id) if isinstance(item_id, str) else None
            matches = self._traits_match(traits, selected_objectives, selected_encodings, selected_difficulties)
            if problem_query and problem_query not in item.text().lower():
                matches = False
            if jax_mode_active and isinstance(item_id, str):
                spec = self.problem_specs.get(item_id)
                if spec is not None:
                    if not self._is_jax_problem_spec(spec):
                        matches = False
            item.setHidden(not matches)
            if matches:
                visible_problems += 1

        selected_problem_item = self.problem_catalog_list.currentItem()
        if selected_problem_item is None or selected_problem_item.isHidden():
            for i in range(self.problem_catalog_list.count()):
                item = self.problem_catalog_list.item(i)
                if not item.isHidden():
                    self.problem_catalog_list.setCurrentItem(item)
                    break

        self.algorithm_filtered_count_label.setText(f"{visible_algorithms} / {self.algorithm_list.count()}")
        self.problem_filtered_count_label.setText(f"{visible_problems} / {self.problem_catalog_list.count()}")

    def _filter_metric_items(self, text: str) -> None:
        query = text.strip().lower()
        for i in range(self.metric_list.count()):
            item = self.metric_list.item(i)
            item.setHidden(bool(query) and query not in item.text().lower())

    def _checked_ids(self, list_widget: QListWidget) -> list[str]:
        ids: list[str] = []
        for i in range(list_widget.count()):
            item = list_widget.item(i)
            if item.checkState() == Qt.CheckState.Checked:
                value = item.data(Qt.ItemDataRole.UserRole)
                if isinstance(value, str):
                    ids.append(value)
        return ids

    def _set_checked_ids(self, list_widget: QListWidget, ids: set[str], name_fallback: set[str], catalog: dict[str, Any]) -> None:
        normalized_name_fallback = {
            str(value).strip().lower() for value in name_fallback if str(value).strip()
        }
        backend_tokens: set[str] = set()
        for raw_value in ids:
            value = str(raw_value).strip()
            if not value:
                continue
            token = core_normalize_backend_token(value)
            if token:
                backend_tokens.add(token)
            spec = catalog.get(value)
            if spec is not None:
                name_token = core_normalize_backend_token(str(getattr(spec, "name", "")))
                if name_token:
                    backend_tokens.add(name_token)
        for raw_value in normalized_name_fallback:
            token = core_normalize_backend_token(raw_value)
            if token:
                backend_tokens.add(token)

        for i in range(list_widget.count()):
            item = list_widget.item(i)
            item_id = item.data(Qt.ItemDataRole.UserRole)
            should_check = False
            if isinstance(item_id, str):
                if item_id in ids:
                    should_check = True
                else:
                    spec = catalog.get(item_id)
                    if spec is not None and str(spec.name).lower() in normalized_name_fallback:
                        should_check = True
                    elif backend_tokens:
                        id_token = core_normalize_backend_token(item_id)
                        name_token = core_normalize_backend_token(
                            str(getattr(spec, "name", "")) if spec is not None else ""
                        )
                        if (id_token and id_token in backend_tokens) or (
                            name_token and name_token in backend_tokens
                        ):
                            should_check = True
            item.setCheckState(Qt.CheckState.Checked if should_check else Qt.CheckState.Unchecked)

    def _on_problem_changed(self) -> None:
        problem_id = self.problem_combo.currentData()
        spec = self.problem_specs.get(problem_id)
        if spec is None:
            return

        if not self._checked_ids(self.experiment_problem_list):
            for i in range(self.experiment_problem_list.count()):
                item = self.experiment_problem_list.item(i)
                if item.data(Qt.ItemDataRole.UserRole) == problem_id:
                    item.setCheckState(Qt.CheckState.Checked)
                    break

        n_var, n_obj = spec.default_n_var, spec.default_n_obj
        if spec.source == "pymoo" and spec.name.lower() in DEFAULT_PROBLEM_DIMS:
            n_var, n_obj = DEFAULT_PROBLEM_DIMS[spec.name.lower()]
        else:
            try:
                probe = spec.factory({"n_var": n_var, "n_obj": n_obj})
                n_var = int(getattr(probe, "n_var", n_var))
                n_obj = int(getattr(probe, "n_obj", n_obj))
            except Exception:  # noqa: BLE001
                pass

        self.n_obj_spin.setValue(max(1, n_obj))
        self.n_var_spin.setValue(max(1, n_var))

        name_lower = spec.name.lower()
        is_fixed_obj = name_lower.startswith("zdt") or name_lower.startswith("imop")

        self._on_test_n_obj_changed()

        self.n_obj_spin.setEnabled(not is_fixed_obj)
        self.n_var_spin.setEnabled(not is_fixed_obj)
        if is_fixed_obj:
            self.n_obj_spin.setToolTip("The number of objectives is fixed and cannot be edited for this problem.")
            self.n_var_spin.setToolTip("The number of variables is fixed and cannot be edited for this problem.")
        else:
            self.n_obj_spin.setToolTip("Number of objectives (M).")
            self.n_var_spin.setToolTip("Number of variables (D).")

        self._sync_ref_point_hint()
        self._update_profile_compare_availability()
        self._sync_problem_catalog_selection()
        self._update_selection_cards()
        self._refresh_test_result_controls()
        self._sync_test_result_to_current_selection_if_available()

    @Slot(int)
    def _on_test_n_obj_changed(self, _value: int = 0) -> None:
        self.n_var_spin.setEnabled(True)
        self._sync_ref_point_hint()

    def _sync_problem_catalog_selection(self) -> None:
        target_id = self.problem_combo.currentData()
        if not isinstance(target_id, str):
            return
        self.problem_catalog_list.blockSignals(True)
        for i in range(self.problem_catalog_list.count()):
            item = self.problem_catalog_list.item(i)
            if item.data(Qt.ItemDataRole.UserRole) == target_id:
                self.problem_catalog_list.setCurrentItem(item)
                break
        self.problem_catalog_list.blockSignals(False)

    def _sync_test_result_to_current_selection_if_available(self) -> None:
        selected_algo_item = self.algorithm_list.currentItem()
        selected_algo_label = selected_algo_item.text().strip() if selected_algo_item is not None else ""
        selected_problem_id = self.problem_combo.currentData()
        if not selected_algo_label or not isinstance(selected_problem_id, str) or not selected_problem_id:
            return

        historical_runs = [
            run
            for run in self.results.get(selected_algo_label, [])
            if isinstance(run, dict) and str(run.get("problem_id", "")).strip() == selected_problem_id
        ]
        if not historical_runs:
            return

        # Sync top-right selectors to the left-side pair.
        problem_idx = self.test_problem_combo.findData(selected_problem_id)
        if problem_idx >= 0 and self.test_problem_combo.currentIndex() != problem_idx:
            self.test_problem_combo.blockSignals(True)
            self.test_problem_combo.setCurrentIndex(problem_idx)
            self.test_problem_combo.blockSignals(False)

        if self.test_algo_combo.findText(selected_algo_label) >= 0 and self.test_algo_combo.currentText() != selected_algo_label:
            self.test_algo_combo.blockSignals(True)
            self.test_algo_combo.setCurrentText(selected_algo_label)
            self.test_algo_combo.blockSignals(False)

        selected_runs = self._test_runs_for_selection(selected_algo_label)
        if not selected_runs:
            self._refresh_test_result_chart()
            return

        latest_idx = self._latest_run_index_by_timestamp(selected_runs)
        self.test_run_spin.setRange(1, max(1, len(selected_runs)))
        self.test_run_spin.setEnabled(True)
        self.test_run_spin.blockSignals(True)
        self.test_run_spin.setValue(max(1, latest_idx + 1))
        self.test_run_spin.blockSignals(False)
        self._refresh_test_result_chart()

    def _update_selection_cards(self) -> None:
        selected_algorithm_item = self.algorithm_list.currentItem()
        if selected_algorithm_item is not None:
            self.selected_algorithm_label.setText(selected_algorithm_item.text())
            algorithm_label = selected_algorithm_item.text()
        else:
            self.selected_algorithm_label.setText("No algorithm selected")
            algorithm_label = "select an algorithm"

        problem_spec = self.problem_specs.get(self.problem_combo.currentData())
        if problem_spec is not None:
            self.selected_problem_label.setText(problem_spec.label)
            problem_label = problem_spec.label
        else:
            self.selected_problem_label.setText("No problem selected")
            problem_label = "select a problem"

        if hasattr(self, "quick_start_summary_label"):
            self.quick_start_summary_label.setText(
                "Test review: 1 algorithm + 1 problem + 1 run. "
                f"Algorithm: {algorithm_label}. Problem: {problem_label}. "
                f"N={self.pop_size_spin.value()}, M={self.n_obj_spin.value()}, "
                f"D={self.n_var_spin.value()}, maxFE={self.max_fe_spin.value()}."
            )

    def _sync_ref_point_hint(self) -> None:
        n_obj = self.n_obj_spin.value()
        self.ref_point_label.setText(
            f"Computed automatically during execution ({n_obj} objective(s))"
        )

    def _update_profile_compare_availability(self) -> None:
        has_gpu = bool(self.gpu_runtime.get("cuda_ok"))
        can_compare = has_gpu
        self.profile_compare_check.setEnabled(can_compare)
        if not can_compare:
            self.profile_compare_check.setChecked(False)
            self.profile_status_label.setText("Profiling unavailable: JAX GPU runtime not detected.")
            self.profile_status_label.setStyleSheet(f"color: {AppStyles.WARNING};")
            return

        self.profile_status_label.setText("CPU vs GPU profiling is available (JAX backend).")
        self.profile_status_label.setStyleSheet(f"color: {AppStyles.SUCCESS};")

    def _on_backend_mode_changed(self) -> None:
        backend = self.compute_backend_combo.currentData()
        gpu_enabled = backend == "gpu"
        self.gpu_dtype_combo.setEnabled(bool(gpu_enabled))
        if backend == "mlx":
            self.gpu_status_label.setText(core_build_backend_status_text(self.mlx_runtime, "mlx"))
            color = AppStyles.TEXT_MUTED
            if not bool(self.mlx_runtime.get("mlx_ok")):
                color = AppStyles.WARNING
        else:
            self.gpu_status_label.setText(core_build_backend_status_text(self.gpu_runtime, backend))
            color = AppStyles.TEXT_MUTED
            if backend == "gpu" and not bool(self.gpu_runtime.get("cuda_ok")):
                color = AppStyles.WARNING
        self.gpu_status_label.setStyleSheet(f"color: {color};")
        self._update_profile_compare_availability()
        if all(
            hasattr(self, attr)
            for attr in ("problem_combo", "experiment_problem_list", "problem_catalog_list")
        ):
            self._populate_problem_combo()
        if hasattr(self, "metric_list"):
            self._refresh_test_metric_list_for_backend()
        self._populate_all_operator_combos()

    def _on_exp_backend_mode_changed(self) -> None:
        backend = self.exp_compute_backend_combo.currentData()
        self.exp_gpu_dtype_combo.setEnabled(bool(backend == "gpu"))
        if hasattr(self, "exp_problem_list"):
            self._refresh_exp_problem_list_for_backend()
        if hasattr(self, "exp_metric_list"):
            self._refresh_exp_metric_list_for_backend()
        self._populate_all_operator_combos()

    @Slot()
    def _on_test_seed_mode_changed(self) -> None:
        mode = str(self.seed_mode_combo.currentData() or SEED_MODE_RANDOM).strip().lower()
        is_fixed = mode == SEED_MODE_FIXED
        self.seed_spin.setEnabled(is_fixed)
        self.seed_spin.setVisible(is_fixed)
        if is_fixed:
            self.seed_mode_hint.setText("Fixed mode uses the same seed for the entire execution.")
        else:
            self.seed_mode_hint.setText("Random mode generates a new seed for each execution.")

    @Slot()
    def _on_exp_seed_mode_changed(self) -> None:
        mode = str(self.exp_seed_mode_combo.currentData() or SEED_MODE_RANDOM).strip().lower()
        show_base = mode in {SEED_MODE_FIXED, SEED_MODE_SEQUENCE}
        show_step = mode == SEED_MODE_SEQUENCE
        self.exp_seed_base_spin.setEnabled(show_base)
        self.exp_seed_base_spin.setVisible(show_base)
        self.exp_seed_step_spin.setEnabled(show_step)
        self.exp_seed_step_spin.setVisible(show_step)

        if mode == SEED_MODE_FIXED:
            self.exp_seed_label.setText(
                f"Fixed mode: every run uses seed={self.exp_seed_base_spin.value()}."
            )
        elif mode == SEED_MODE_SEQUENCE:
            self.exp_seed_label.setText(
                "Sequence mode: seeds follow base + index * step "
                f"({self.exp_seed_base_spin.value()} + i*{self.exp_seed_step_spin.value()})."
            )
        else:
            self.exp_seed_label.setText("Random mode: each run uses an automatically generated seed.")

    @Slot()
    def _on_crossover_changed(self) -> None:
        """Show/hide crossover parameter widgets based on selected operator."""
        class_name = self._operator_class_name("crossover", self.crossover_combo.currentData())
        show_sbx = class_name in {"sbx", "simulatedbinarycrossover"}
        self.crossover_eta_label.setVisible(show_sbx)
        self.crossover_eta_spin.setVisible(show_sbx)
        self.crossover_prob_label.setVisible(show_sbx)
        self.crossover_prob_spin.setVisible(show_sbx)

    @Slot()
    def _on_mutation_changed(self) -> None:
        """Show/hide mutation parameter widgets based on selected operator."""
        class_name = self._operator_class_name("mutation", self.mutation_combo.currentData())
        show_pm = class_name in {"pm", "polynomialmutation"}
        show_prob = class_name in {
            "pm",
            "polynomialmutation",
            "bfm",
            "bitflipmutation",
            "gaussianmutation",
            "gm",
        }
        self.mutation_eta_label.setVisible(show_pm)
        self.mutation_eta_spin.setVisible(show_pm)
        self.mutation_prob_label.setVisible(show_prob)
        self.mutation_prob_spin.setVisible(show_prob)

    @Slot()
    def _on_selection_changed(self) -> None:
        """Show/hide selection parameter widgets based on selected operator."""
        class_name = self._operator_class_name("selection", self.selection_combo.currentData())
        show_pressure = class_name == "tournamentselection"
        self.selection_pressure_label.setVisible(show_pressure)
        self.selection_pressure_spin.setVisible(show_pressure)

    def _sync_test_metric_combo(self) -> None:
        current = self.test_metric_combo.currentText().strip()
        checked_metric_names: list[str] = []
        for i in range(self.metric_list.count()):
            item = self.metric_list.item(i)
            if item.checkState() == Qt.CheckState.Checked:
                checked_metric_names.append(item.text())

        source_metrics = self.metric_names if self.metric_names else checked_metric_names
        self.test_metric_combo.blockSignals(True)
        self.test_metric_combo.clear()
        for metric in source_metrics:
            self.test_metric_combo.addItem(metric)
        if current and self.test_metric_combo.findText(current) >= 0:
            self.test_metric_combo.setCurrentText(current)
        self.test_metric_combo.blockSignals(False)

    def _selected_test_problem_id(self) -> str | None:
        value = self.test_problem_combo.currentData()
        if isinstance(value, str) and value and value != "__all__":
            return value
        return None

    def _test_runs_for_selection(self, algo_name: str) -> list[dict[str, Any]]:
        raw_runs = list(self.results.get(algo_name, []))
        runs: list[dict[str, Any]] = []
        seen: set[str] = set()
        for run_payload in raw_runs:
            if not isinstance(run_payload, dict):
                continue
            normalized = self._ensure_run_dimensions(dict(run_payload))
            run_key = self._analysis_run_key(normalized)
            if run_key in seen:
                continue
            seen.add(run_key)
            runs.append(normalized)
        selected_problem_id = self._selected_test_problem_id()
        if selected_problem_id is None:
            return runs
        return [run for run in runs if str(run.get("problem_id", "")) == selected_problem_id]

    def _latest_run_index_by_timestamp(self, runs: list[dict[str, Any]]) -> int:
        if not runs:
            return -1
        best_idx = 0
        best_epoch = float("-inf")
        for idx, run_payload in enumerate(runs):
            epoch = self._run_timestamp_epoch(run_payload)
            if epoch >= best_epoch:
                best_epoch = epoch
                best_idx = idx
        return best_idx

    @staticmethod
    def _run_timestamp_epoch(run_payload: dict[str, Any]) -> float:
        epoch = _float_or_none(run_payload.get("timestamp_epoch"))
        if epoch is not None and math.isfinite(epoch):
            return float(epoch)

        iso_raw = str(run_payload.get("timestamp_iso", "")).strip()
        if iso_raw:
            iso_norm = iso_raw.replace("Z", "+00:00")
            try:
                dt = datetime.fromisoformat(iso_norm)
                if dt.tzinfo is None:
                    dt = dt.astimezone()
                return float(dt.timestamp())
            except Exception:  # noqa: BLE001
                pass

        return float("-inf")

    def _find_algorithm_id_by_label(self, label: str) -> str | None:
        target = str(label).strip().lower()
        if not target:
            return None
        for spec_id, spec in self.algorithm_specs.items():
            if spec.label.strip().lower() == target or spec.name.strip().lower() == target:
                return spec_id
        return None

    def _find_problem_id_by_label(self, label: str) -> str | None:
        target = str(label).strip().lower()
        if not target:
            return None
        for spec_id, spec in self.problem_specs.items():
            if spec.label.strip().lower() == target or spec.name.strip().lower() == target:
                return spec_id
        return None

    def _algorithm_label_for_run(self, run_payload: dict[str, Any]) -> str:
        algo_id = str(run_payload.get("algorithm_id", "")).strip()
        if algo_id and algo_id in self.algorithm_specs:
            return self.algorithm_specs[algo_id].label
        label = str(run_payload.get("algorithm", "")).strip()
        if label:
            return label
        if algo_id:
            return algo_id
        return "algorithm"

    def _problem_label_for_run(self, run_payload: dict[str, Any]) -> str:
        problem_id = str(run_payload.get("problem_id", "")).strip()
        if problem_id and problem_id in self.problem_specs:
            return self.problem_specs[problem_id].label
        label = str(run_payload.get("problem", "")).strip()
        if label:
            return label
        if problem_id:
            return problem_id
        return "problem"

    def _analysis_group_key(self, run_payload: dict[str, Any]) -> tuple[str, str]:
        algo_id = str(run_payload.get("algorithm_id", "")).strip()
        if algo_id:
            algo_key = f"id:{algo_id.lower()}"
        else:
            algo_key = f"label:{self._algorithm_label_for_run(run_payload).lower()}"

        problem_id = str(run_payload.get("problem_id", "")).strip()
        if problem_id:
            problem_key = f"id:{problem_id.lower()}"
        else:
            problem_key = f"label:{self._problem_label_for_run(run_payload).lower()}"

        return algo_key, problem_key

    @staticmethod
    def _analysis_group_id_from_key(group_key: tuple[str, str]) -> str:
        return f"{group_key[0]}||{group_key[1]}"

    def _analysis_group_id_for_run(self, run_payload: dict[str, Any]) -> str:
        return self._analysis_group_id_from_key(self._analysis_group_key(run_payload))

    def _analysis_run_key(self, run_payload: dict[str, Any]) -> str:
        algo_key, problem_key = self._analysis_group_key(run_payload)
        run_index = str(run_payload.get("run_index", "")).strip()
        seed = str(run_payload.get("seed", "")).strip()
        backend_code = str(run_payload.get("backend_code", "")).strip().lower()
        epoch = self._run_timestamp_epoch(run_payload)
        if math.isfinite(epoch):
            time_key = f"{epoch:.6f}"
        else:
            time_key = str(run_payload.get("timestamp_iso", "")).strip() or str(
                run_payload.get("timestamp_en_us", "")
            ).strip()
        return "|".join((algo_key, problem_key, run_index, seed, backend_code, time_key))

    def _rebuild_analysis_run_index(self) -> None:
        self.analysis_run_keys = set()
        for runs in self.results.values():
            if not isinstance(runs, list):
                continue
            for run_payload in runs:
                if not isinstance(run_payload, dict):
                    continue
                normalized = self._ensure_run_dimensions(dict(run_payload))
                self.analysis_run_keys.add(self._analysis_run_key(normalized))

    def _append_analysis_run(self, run_payload: dict[str, Any], *, algo_hint: str = "") -> bool:
        normalized = self._ensure_run_dimensions(dict(run_payload))

        if not str(normalized.get("algorithm", "")).strip() and algo_hint:
            normalized["algorithm"] = algo_hint

        algo_id = str(normalized.get("algorithm_id", "")).strip()
        if not algo_id:
            resolved_algo_id = self._find_algorithm_id_by_label(str(normalized.get("algorithm", "")))
            if resolved_algo_id:
                normalized["algorithm_id"] = resolved_algo_id
        normalized["algorithm"] = self._algorithm_label_for_run(normalized)

        problem_id = str(normalized.get("problem_id", "")).strip()
        if not problem_id:
            resolved_problem_id = self._find_problem_id_by_label(str(normalized.get("problem", "")))
            if resolved_problem_id:
                normalized["problem_id"] = resolved_problem_id
                problem_id = resolved_problem_id
        normalized["problem"] = self._problem_label_for_run(normalized)
        if problem_id and problem_id in self.problem_specs:
            normalized.setdefault("problem_name", self.problem_specs[problem_id].name)

        run_key = self._analysis_run_key(normalized)
        if run_key in self.analysis_run_keys:
            return False

        self.analysis_run_keys.add(run_key)
        algo_label = self._algorithm_label_for_run(normalized)
        self.results.setdefault(algo_label, []).append(normalized)
        return True

    def _all_analysis_runs(self) -> list[dict[str, Any]]:
        flattened: list[dict[str, Any]] = []
        seen: set[str] = set()
        for algo_name, runs in self.results.items():
            if not isinstance(runs, list):
                continue
            for run_payload in runs:
                if not isinstance(run_payload, dict):
                    continue
                normalized = self._ensure_run_dimensions(dict(run_payload))
                if not str(normalized.get("algorithm", "")).strip():
                    normalized["algorithm"] = str(algo_name)
                normalized["algorithm"] = self._algorithm_label_for_run(normalized)
                normalized["problem"] = self._problem_label_for_run(normalized)
                key = self._analysis_run_key(normalized)
                if key in seen:
                    continue
                seen.add(key)
                flattened.append(normalized)
        return flattened

    def _analysis_algorithm_options(self, problem_id: str | None = None) -> list[str]:
        latest_by_algo: dict[str, tuple[str, float]] = {}
        for run_payload in self._all_analysis_runs():
            run_problem_id = str(run_payload.get("problem_id", "")).strip()
            if problem_id is not None and run_problem_id != problem_id:
                continue

            label = self._algorithm_label_for_run(run_payload).strip()
            if not label:
                continue

            key = label.lower()
            epoch = self._run_timestamp_epoch(run_payload)
            existing = latest_by_algo.get(key)
            if existing is None or epoch >= existing[1]:
                latest_by_algo[key] = (label, epoch)

        ordered = sorted(latest_by_algo.values(), key=lambda item: (-item[1], item[0].lower()))
        return [label for label, _ in ordered]

    @staticmethod
    def _same_run_identity(a: dict[str, Any], b: dict[str, Any]) -> bool:
        ts_a = _float_or_none(a.get("timestamp_epoch"))
        ts_b = _float_or_none(b.get("timestamp_epoch"))
        if ts_a is not None and ts_b is not None and math.isfinite(ts_a) and math.isfinite(ts_b):
            if math.isclose(ts_a, ts_b, rel_tol=0.0, abs_tol=1e-6):
                return True

        return (
            str(a.get("algorithm", "")) == str(b.get("algorithm", ""))
            and str(a.get("problem_id", "")) == str(b.get("problem_id", ""))
            and str(a.get("run_index", "")) == str(b.get("run_index", ""))
            and str(a.get("seed", "")) == str(b.get("seed", ""))
        )

    def _find_run_index(self, runs: list[dict[str, Any]], target_run: dict[str, Any]) -> int:
        for idx, run_payload in enumerate(runs):
            if self._same_run_identity(run_payload, target_run):
                return idx

        target_run_idx = str(target_run.get("run_index", ""))
        if target_run_idx:
            for idx, run_payload in enumerate(runs):
                if str(run_payload.get("run_index", "")) == target_run_idx:
                    return idx
        return -1

    def _latest_run_from_results(self, results_map: dict[str, list[dict[str, Any]]]) -> dict[str, Any] | None:
        latest_payload: dict[str, Any] | None = None
        latest_epoch = float("-inf")

        for runs in results_map.values():
            for run_payload in runs:
                if not isinstance(run_payload, dict):
                    continue
                epoch = self._run_timestamp_epoch(run_payload)
                if latest_payload is None or epoch >= latest_epoch:
                    latest_payload = run_payload
                    latest_epoch = epoch

        return dict(latest_payload) if isinstance(latest_payload, dict) else None

    def _latest_run_from_payload(self, payload: dict[str, Any]) -> dict[str, Any] | None:
        results_map = payload.get("results", {})
        if not isinstance(results_map, dict):
            return None
        return self._latest_run_from_results(results_map)

    def _resolve_problem_pf(
        self,
        run_payload: dict[str, Any],
        *,
        use_test_context: bool = False,
    ) -> np.ndarray | None:
        n_obj_hint = _positive_int(run_payload.get("n_obj"), 0, minimum=0)
        inline_pf = ExperimentBridge._coerce_front_matrix(
            run_payload.get("reference_front"),
            n_obj_hint=(n_obj_hint if n_obj_hint > 0 else None),
        )
        if inline_pf is not None:
            return inline_pf

        run_problem_id = str(run_payload.get("problem_id", ""))
        if use_test_context:
            pf_map = self.test_pareto_fronts if isinstance(self.test_pareto_fronts, dict) else {}
            pf_main = self.test_pareto_front
        else:
            pf_map = self.pareto_fronts if isinstance(self.pareto_fronts, dict) else {}
            pf_main = self.pareto_front

        if run_problem_id:
            if run_problem_id in pf_map:
                mapped_pf = ExperimentBridge._coerce_front_matrix(
                    pf_map[run_problem_id],
                    n_obj_hint=(n_obj_hint if n_obj_hint > 0 else None),
                )
                if mapped_pf is not None:
                    return mapped_pf
                return None
            # Only use the single cached PF fallback when there is no keyed PF map.
            if pf_main is not None and not pf_map:
                fallback_pf = ExperimentBridge._coerce_front_matrix(
                    pf_main,
                    n_obj_hint=(n_obj_hint if n_obj_hint > 0 else None),
                )
                return fallback_pf
            # Avoid plotting a PF from another problem.
            return None

        if len(pf_map) == 1:
            single_pf = ExperimentBridge._coerce_front_matrix(
                next(iter(pf_map.values())),
                n_obj_hint=(n_obj_hint if n_obj_hint > 0 else None),
            )
            return single_pf

        if not pf_map:
            fallback_pf = ExperimentBridge._coerce_front_matrix(
                pf_main,
                n_obj_hint=(n_obj_hint if n_obj_hint > 0 else None),
            )
            return fallback_pf

        return None

    def _focus_test_run(self, run_payload: dict[str, Any] | None) -> None:
        if not isinstance(run_payload, dict):
            return

        problem_id = str(run_payload.get("problem_id", "")).strip()
        if problem_id:
            idx = self.test_problem_combo.findData(problem_id)
            if idx >= 0:
                self.test_problem_combo.setCurrentIndex(idx)
        elif self.test_problem_combo.findData("__all__") >= 0:
            self.test_problem_combo.setCurrentIndex(self.test_problem_combo.findData("__all__"))

        algo_name = str(run_payload.get("algorithm", "")).strip()
        if algo_name and self.test_algo_combo.findText(algo_name) >= 0:
            self.test_algo_combo.setCurrentText(algo_name)

        selected_algo = self.test_algo_combo.currentText().strip()
        runs = self._test_runs_for_selection(selected_algo)
        if not runs:
            self._refresh_test_result_chart()
            return

        idx = self._find_run_index(runs, run_payload)
        if idx < 0:
            idx = len(runs) - 1

        self.test_run_spin.setRange(1, max(1, len(runs)))
        self.test_run_spin.setEnabled(True)
        self.test_run_spin.setValue(int(idx + 1))
        self._refresh_test_result_chart()

    def _focus_analysis_run(self, run_payload: dict[str, Any] | None) -> None:
        if not isinstance(run_payload, dict):
            return

        problem_id = str(run_payload.get("problem_id", "")).strip()
        self.problem_plot_combo.blockSignals(True)
        if problem_id and self.problem_plot_combo.findData(problem_id) >= 0:
            self.problem_plot_combo.setCurrentIndex(self.problem_plot_combo.findData(problem_id))
        elif self.problem_plot_combo.findData("__all__") >= 0:
            self.problem_plot_combo.setCurrentIndex(self.problem_plot_combo.findData("__all__"))
        self.problem_plot_combo.blockSignals(False)

        self._on_plot_problem_changed()

        algo_name = str(run_payload.get("algorithm", "")).strip()
        self.algo_plot_combo.blockSignals(True)
        if algo_name and self.algo_plot_combo.findText(algo_name) >= 0:
            self.algo_plot_combo.setCurrentText(algo_name)
        self.algo_plot_combo.blockSignals(False)

        self._on_plot_algo_changed()

        selected_algo = self.algo_plot_combo.currentText().strip()
        runs = self._runs_for_plot(selected_algo)
        if not runs:
            self._refresh_convergence_chart()
            self._refresh_pareto_chart()
            return

        idx = self._find_run_index(runs, run_payload)
        if idx < 0:
            idx = len(runs) - 1

        self.run_plot_spin.setRange(1, max(1, len(runs)))
        self.run_plot_spin.setEnabled(True)
        self.run_plot_spin.setValue(int(idx + 1))
        self._refresh_convergence_chart()
        self._refresh_pareto_chart()

    def _normalize_test_live_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        normalized = self._ensure_run_dimensions(dict(payload))

        front_raw = normalized.get("final_front", normalized.get("front", []))
        front = np.asarray(front_raw, dtype=float)
        if front.ndim == 1 and front.size:
            front = front.reshape(1, -1)
        if front.ndim != 2:
            front = np.empty((0, 0), dtype=float)
        normalized["final_front"] = front.tolist()

        history_src = normalized.get("history", {})
        history_out: dict[str, list[float]] = {}
        if isinstance(history_src, dict):
            for metric_name, values in history_src.items():
                arr = np.asarray(values, dtype=float).reshape(-1)
                history_out[str(metric_name)] = arr.tolist()
        normalized["history"] = history_out

        x_history_arr = np.asarray(
            normalized.get("x_history", normalized.get("generations", [])),
            dtype=float,
        ).reshape(-1)
        x_history = x_history_arr.tolist()
        normalized["x_history"] = x_history
        normalized["generations"] = list(x_history)

        metrics_src = normalized.get("metrics", {})
        metrics_out: dict[str, float] = {}
        if isinstance(metrics_src, dict):
            for metric_name, value in metrics_src.items():
                number = _float_or_none(value)
                metrics_out[str(metric_name)] = float("nan") if number is None else number
        if not metrics_out:
            for metric_name, values in history_out.items():
                if values:
                    metrics_out[metric_name] = float(values[-1])
        normalized["metrics"] = metrics_out

        if "n_eval" not in normalized and "evaluations" in normalized:
            normalized["n_eval"] = normalized.get("evaluations")
        if "evaluations" not in normalized and "n_eval" in normalized:
            normalized["evaluations"] = normalized.get("n_eval")

        timestamp_epoch = _float_or_none(normalized.get("timestamp_epoch"))
        if timestamp_epoch is None:
            timestamp_epoch = float(datetime.now().astimezone().timestamp())
        normalized["timestamp_epoch"] = float(timestamp_epoch)
        timestamp_iso = str(normalized.get("timestamp_iso", "")).strip()
        if not timestamp_iso:
            normalized["timestamp_iso"] = datetime.now().astimezone().isoformat()
        timestamp_en_us = str(normalized.get("timestamp_en_us", "")).strip()
        if not timestamp_en_us:
            normalized["timestamp_en_us"] = format_timestamp_en_us()

        if not normalized.get("algorithm"):
            current_item = self.algorithm_list.currentItem()
            normalized["algorithm"] = current_item.text() if current_item is not None else "algorithm"

        if not normalized.get("problem"):
            selected_problem = self.problem_combo.currentText().strip()
            normalized["problem"] = selected_problem or "problem"

        problem_id = self.problem_combo.currentData()
        if not normalized.get("problem_id") and isinstance(problem_id, str) and problem_id:
            normalized["problem_id"] = problem_id

        normalized.setdefault("problem_name", str(normalized.get("problem", "problem")))
        normalized.setdefault("backend", "running")
        normalized.setdefault("backend_code", "running")
        normalized["is_partial"] = True

        return normalized

    def _refresh_test_result_controls(self, autoload_chart: bool = True) -> None:
        self._sync_test_metric_combo()

        current_problem_id = self.test_problem_combo.currentData()
        preferred_problem_id = self.problem_combo.currentData()
        problem_entries: dict[str, str] = {}
        for run in self._all_analysis_runs():
            problem_id = str(run.get("problem_id", "")).strip()
            problem_label = self._problem_label_for_run(run)
            if problem_id:
                problem_entries[problem_id] = problem_label

        self.test_problem_combo.blockSignals(True)
        self.test_problem_combo.clear()
        self.test_problem_combo.addItem("All problems", "__all__")
        for problem_id, label in sorted(problem_entries.items(), key=lambda x: x[1].lower()):
            self.test_problem_combo.addItem(label, problem_id)
        target_problem = "__all__"
        if (
            isinstance(current_problem_id, str)
            and current_problem_id != "__all__"
            and self.test_problem_combo.findData(current_problem_id) >= 0
        ):
            target_problem = current_problem_id
        elif isinstance(preferred_problem_id, str) and self.test_problem_combo.findData(preferred_problem_id) >= 0:
            target_problem = preferred_problem_id
        elif len(problem_entries) == 1:
            target_problem = next(iter(problem_entries))
        target_idx = self.test_problem_combo.findData(target_problem)
        if target_idx >= 0:
            self.test_problem_combo.setCurrentIndex(target_idx)
        self.test_problem_combo.blockSignals(False)

        if autoload_chart:
            self._on_test_problem_changed()

    @Slot()
    def _on_test_problem_changed(self) -> None:
        current_algo = self.test_algo_combo.currentText().strip()
        preferred_algo = ""
        selected_item = self.algorithm_list.currentItem()
        if selected_item is not None:
            preferred_algo = selected_item.text().strip()
        selected_problem_id = self._selected_test_problem_id()

        algo_names: list[str] = []
        for algo_name, runs in self.results.items():
            if selected_problem_id is None:
                algo_names.append(algo_name)
                continue
            if any(str(run.get("problem_id", "")) == selected_problem_id for run in runs):
                algo_names.append(algo_name)

        if not algo_names:
            selected_item = self.algorithm_list.currentItem()
            if selected_item is not None:
                algo_names.append(selected_item.text())

        self.test_algo_combo.blockSignals(True)
        self.test_algo_combo.clear()
        self.test_algo_combo.addItems(algo_names)
        if preferred_algo and preferred_algo in algo_names:
            self.test_algo_combo.setCurrentText(preferred_algo)
        elif current_algo and current_algo in algo_names:
            self.test_algo_combo.setCurrentText(current_algo)
        self.test_algo_combo.blockSignals(False)

        self._on_test_algo_changed()

    @Slot()
    def _on_test_algo_changed(self) -> None:
        algo_name = self.test_algo_combo.currentText().strip()
        runs = self._test_runs_for_selection(algo_name)
        run_count = len(runs)
        self.test_run_spin.setRange(1, max(1, run_count))
        self.test_run_spin.setEnabled(run_count > 0)
        if run_count > 0:
            latest_idx = self._latest_run_index_by_timestamp(runs)
            target_value = max(1, int(latest_idx + 1))
            self.test_run_spin.blockSignals(True)
            self.test_run_spin.setValue(target_value)
            self.test_run_spin.blockSignals(False)
        self._refresh_test_result_chart()

    def _test_result_selection_matches_run(
        self,
        run_payload: dict[str, Any],
        *,
        selected_algo_name: str,
        selected_problem_id: str | None,
    ) -> bool:
        if not isinstance(run_payload, dict):
            return False

        if selected_algo_name:
            run_algo_label = self._algorithm_label_for_run(run_payload).strip()
            if run_algo_label != selected_algo_name:
                return False

        if selected_problem_id is not None:
            run_problem_id = str(run_payload.get("problem_id", "")).strip()
            if run_problem_id != selected_problem_id:
                return False

        return True

    def _apply_scientific_chart_theme(
        self,
        chart: QChart,
        axes: tuple[QValueAxis, ...] = (),
    ) -> None:
        """Apply the shared accessible chart treatment across Qt backends.

        In Dark Mode, axes lines and tick labels are rendered in crisp bright white
        (#F8FAFC) to guarantee high visual acuity and match axis markings identically.
        """
        colors = StylesAppStyles.colors
        is_dark = (StylesAppStyles.current_theme == "dark")
        chart_text_color = QColor("#F8FAFC" if is_dark else colors.text_primary)
        axis_line_color = QColor("#F8FAFC" if is_dark else colors.text_secondary)
        grid_line_color = QColor("#334155" if is_dark else colors.border_light)

        chart.setBackgroundBrush(QBrush(QColor(colors.surface)))
        chart.setPlotAreaBackgroundVisible(True)
        chart.setPlotAreaBackgroundBrush(QBrush(QColor(colors.surface)))
        chart.setTitleBrush(QBrush(chart_text_color))

        legend = chart.legend()
        try:
            legend.setLabelColor(chart_text_color)
        except AttributeError:
            pass

        grid_pen = QPen(grid_line_color, 1.0)
        axis_pen = QPen(axis_line_color, 1.2)

        for axis in axes:
            axis.setGridLinePen(grid_pen)
            try:
                axis.setLinePen(axis_pen)
                axis.setLabelsColor(axis_line_color)
                axis.setTitleBrush(QBrush(chart_text_color))
            except AttributeError:
                pass

    @staticmethod
    def _set_chart_series_color(series: Any, color: str, *, width: float = 2.0) -> None:
        """Color Qt line or scatter series without assuming a Qt binding detail."""
        qcolor = QColor(color)
        if isinstance(series, QLineSeries):
            series.setPen(QPen(qcolor, width))
            return
        try:
            series.setColor(qcolor)
            series.setBorderColor(qcolor.darker(120))
        except AttributeError:
            pass

    def _show_3d_scatter(
        self,
        front: np.ndarray,
        problem_pf: np.ndarray | None,
        title: str,
        anchor: bool,
        container: QWidget,
        container_layout: QVBoxLayout,
        stack: QStackedWidget,
        selected_point: np.ndarray | None = None,
        selected_label: str = "MCDM selected",
    ) -> None:
        """Render an interactive 3D scatter plot with matplotlib embedded in Qt."""
        # Clear previous canvas.
        while container_layout.count():
            item = container_layout.takeAt(0)
            w = item.widget()
            if w:
                w.setParent(None)
                w.deleteLater()

        colors = StylesAppStyles.colors
        is_dark = (StylesAppStyles.current_theme == "dark")
        text_color = "#F8FAFC" if is_dark else colors.text_primary
        axis_color = "#F8FAFC" if is_dark else colors.text_secondary
        pane_edge_color = "#475569" if is_dark else colors.border_light

        fig = MplFigure(figsize=(6, 5), dpi=100)
        fig.patch.set_facecolor(colors.surface)
        ax = fig.add_subplot(111, projection="3d")
        ax.set_facecolor(colors.surface)
        ax.tick_params(colors=axis_color, labelsize=8)
        ax.xaxis.label.set_color(text_color)
        ax.yaxis.label.set_color(text_color)
        ax.zaxis.label.set_color(text_color)
        ax.title.set_color(text_color)
        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            try:
                axis.pane.set_facecolor(colors.surface_soft)
                axis.pane.set_edgecolor(pane_edge_color)
                axis.line.set_color(axis_color)
            except AttributeError:
                pass

        # Adjust visualization for the academic-style view (f1/f2 symmetry)
        # azim=45 corrects the apparent x/y visual inversion
        ax.view_init(elev=30, azim=45)

        # Reference PF (crisp white in dark mode, dark in light mode)
        if problem_pf is not None and problem_pf.ndim == 2 and problem_pf.shape[1] >= 3:
            ax.scatter(
                problem_pf[:, 0].tolist(), problem_pf[:, 1].tolist(), problem_pf[:, 2].tolist(),
                c=colors.chart_reference_front, alpha=0.45, s=14, label="Reference PF", depthshade=True,
            )

        # Obtained solutions (blue)
        ax.scatter(
            front[:, 0].tolist(), front[:, 1].tolist(), front[:, 2].tolist(),
            c=colors.chart_series[0], alpha=0.82, s=20, label="Obtained", edgecolors=colors.primary_dark,
            linewidths=0.4, depthshade=True,
        )

        selected_point_3d: np.ndarray | None = None
        try:
            if selected_point is not None:
                point = np.asarray(selected_point, dtype=float).reshape(-1)
                if point.size >= 3 and np.all(np.isfinite(point[:3])):
                    selected_point_3d = point[:3].astype(float, copy=False)
                    ax.scatter(
                        [float(selected_point_3d[0])],
                        [float(selected_point_3d[1])],
                        [float(selected_point_3d[2])],
                        c="#DC2626",
                        marker="*",
                        s=180,
                        edgecolors="#7F1D1D",
                        linewidths=0.8,
                        label=str(selected_label or "MCDM selected"),
                        depthshade=True,
                        zorder=20,
                    )
        except Exception:  # noqa: BLE001
            selected_point_3d = None

        ax.set_xlabel("f1", fontsize=9)
        ax.set_ylabel("f2", fontsize=9)
        ax.set_zlabel("f3", fontsize=9)
        ax.set_title(title, fontsize=10, pad=10)
        ax.legend(fontsize=8, loc="upper right")

        # Ancorar na origem
        if anchor:
            all_data = front
            if problem_pf is not None and problem_pf.ndim == 2 and problem_pf.shape[1] >= 3:
                all_data = np.vstack([all_data, problem_pf[:, :3]])
            if selected_point_3d is not None and selected_point_3d.size >= 3:
                all_data = np.vstack([all_data, selected_point_3d.reshape(1, 3)])
            x_max = float(np.max(all_data[:, 0])) * 1.05
            y_max = float(np.max(all_data[:, 1])) * 1.05
            z_max = float(np.max(all_data[:, 2])) * 1.05
            ax.set_xlim(0, x_max)
            ax.set_ylim(0, y_max)
            ax.set_zlim(0, z_max)

        ax.grid(True, color=colors.border_light, linewidth=0.7, alpha=0.8)
        fig.tight_layout()
        canvas = MplCanvas(fig)
        container_layout.addWidget(canvas)
        stack.setCurrentIndex(1)

    def _set_empty_test_result_chart(self) -> None:
        self.test_chart_stack.setCurrentIndex(0)
        chart = QChart()
        chart.setTitle("Run a test to visualize the population or convergence")
        chart.legend().setVisible(True)

        x_axis = QValueAxis()
        x_axis.setTitleText("x")
        x_axis.setRange(0, 1)

        y_axis = QValueAxis()
        y_axis.setTitleText("y")
        y_axis.setRange(0, 1)

        chart.addAxis(x_axis, Qt.AlignmentFlag.AlignBottom)
        chart.addAxis(y_axis, Qt.AlignmentFlag.AlignLeft)
        self._apply_scientific_chart_theme(chart, (x_axis, y_axis))
        self.test_chart_view.setChart(chart)

    @Slot()
    def _refresh_test_result_chart(self) -> None:
        if not self.results and not self.test_live_payload:
            self._set_empty_test_result_chart()
            self.test_result_summary.clear()
            return

        mode = str(self.test_mode_combo.currentData())
        selected_algo_name = self.test_algo_combo.currentText().strip()
        selected_problem_id = self._selected_test_problem_id()

        if self.test_live_payload and self._test_result_selection_matches_run(
            self.test_live_payload,
            selected_algo_name=selected_algo_name,
            selected_problem_id=selected_problem_id,
        ):
            run_payload = self.test_live_payload
        else:
            runs = self._test_runs_for_selection(selected_algo_name)
            run_index = self.test_run_spin.value() - 1
            if run_index < 0 or run_index >= len(runs):
                run_index = 0
            if not runs:
                self._set_empty_test_result_chart()
                self.test_result_summary.setPlainText("No trial available for the current selection.")
                return
            run_payload = runs[run_index]

        algo_name = str(run_payload.get("algorithm", selected_algo_name or "algorithm"))
        problem_label = str(run_payload.get("problem", "problem"))
        run_timestamp = str(run_payload.get("timestamp_en_us", "")).strip()
        if not run_timestamp:
            run_timestamp = format_timestamp_en_us()
        metric_name = self.test_metric_combo.currentText().strip()
        history_map = run_payload.get("history", {})
        if not isinstance(history_map, dict):
            history_map = {}
        if not metric_name and history_map:
            metric_name = str(next(iter(history_map)))
        run_label = f"trial {run_payload.get('run_index', '-')}"
        is_partial = bool(run_payload.get("is_partial"))
        if is_partial:
            run_label = f"{run_label} (running)"

        self.test_chart_stack.setCurrentIndex(0)
        chart = QChart()
        chart.legend().setVisible(True)

        if mode == "convergence":
            chart.setTitle(f"Convergence - {metric_name} - {algo_name} - {run_label} | {run_timestamp}")
            x_axis = QValueAxis()
            x_axis.setTitleText("Function evaluations (FE)")
            y_axis = QValueAxis()
            y_axis.setTitleText(metric_name or "Metric")
            chart.addAxis(x_axis, Qt.AlignmentFlag.AlignBottom)
            chart.addAxis(y_axis, Qt.AlignmentFlag.AlignLeft)

            history = np.asarray(history_map.get(metric_name, []), dtype=float)
            x_history = np.asarray(run_payload.get("x_history", run_payload.get("generations", [])), dtype=float)
            if history.size == 0:
                self._set_empty_test_result_chart()
                self.test_result_summary.setPlainText("Selected trial has no convergence history for this metric.")
                return
            if x_history.size != history.size:
                x_history = np.arange(1, history.size + 1, dtype=float)

            series = QLineSeries()
            series.setName(f"{algo_name} {run_label}")
            for x_val, y_val in zip(x_history, history, strict=False):
                if math.isfinite(float(y_val)):
                    series.append(float(x_val), float(y_val))

            self._set_chart_series_color(series, AppStyles.chart_palette()[0])
            chart.addSeries(series)
            series.attachAxis(x_axis)
            series.attachAxis(y_axis)
            x_axis.setRange(float(np.min(x_history)), float(np.max(x_history)))
            y_min = float(np.nanmin(history))
            y_max = float(np.nanmax(history))
            if y_min == y_max:
                y_max = y_min + 1.0
            margin = (y_max - y_min) * 0.1
            y_axis.setRange(y_min - margin, y_max + margin)
        else:
            front = np.asarray(run_payload.get("final_front", run_payload.get("front", [])), dtype=float)
            if front.ndim != 2 or front.size == 0 or front.shape[1] < 2:
                self._set_empty_test_result_chart()
                self.test_result_summary.setPlainText("Selected trial has no objective data for population view.")
                return

            n_obj = front.shape[1]
            anchor = self.test_anchor_origin.isChecked()
            problem_pf = self._resolve_problem_pf(run_payload, use_test_context=True)

            if n_obj == 2:
                # -- m=2: Scatter 2D --
                chart.setTitle(f"Population (f1xf2) - {problem_label} - {algo_name} {run_label} | {run_timestamp}")
                x_axis = QValueAxis()
                x_axis.setTitleText("f1")
                y_axis = QValueAxis()
                y_axis.setTitleText("f2")
                chart.addAxis(x_axis, Qt.AlignmentFlag.AlignBottom)
                chart.addAxis(y_axis, Qt.AlignmentFlag.AlignLeft)

                obtained = QScatterSeries()
                obtained.setName("Obtained")
                obtained.setMarkerSize(7.0)
                for x_val, y_val in front:
                    obtained.append(float(x_val), float(y_val))
                self._set_chart_series_color(obtained, AppStyles.chart_palette()[0])
                chart.addSeries(obtained)
                obtained.attachAxis(x_axis)
                obtained.attachAxis(y_axis)

                x_values = [float(np.min(front[:, 0])), float(np.max(front[:, 0]))]
                y_values = [float(np.min(front[:, 1])), float(np.max(front[:, 1]))]

                if problem_pf is not None and problem_pf.ndim == 2 and problem_pf.shape[1] >= 2:
                    pf_2d = problem_pf[:, :2]
                    pf_2d = pf_2d[np.argsort(pf_2d[:, 0])]
                    pf_series = QLineSeries()
                    pf_series.setName("Reference PF")
                    for x_val, y_val in pf_2d:
                        pf_series.append(float(x_val), float(y_val))
                    self._set_chart_series_color(
                        pf_series,
                        StylesAppStyles.colors.chart_reference_front,
                        width=2.0,
                    )
                    chart.addSeries(pf_series)
                    pf_series.attachAxis(x_axis)
                    pf_series.attachAxis(y_axis)
                    x_values.extend([float(np.min(pf_2d[:, 0])), float(np.max(pf_2d[:, 0]))])
                    y_values.extend([float(np.min(pf_2d[:, 1])), float(np.max(pf_2d[:, 1]))])

                if anchor:
                    x_values.append(0.0)
                    y_values.append(0.0)

                x_min, x_max = min(x_values), max(x_values)
                y_min, y_max = min(y_values), max(y_values)
                if x_min == x_max:
                    x_max = x_min + 1.0
                if y_min == y_max:
                    y_max = y_min + 1.0
                x_margin = (x_max - x_min) * 0.05
                y_margin = (y_max - y_min) * 0.05
                x_lower = 0.0 if anchor else (x_min - x_margin)
                y_lower = 0.0 if anchor else (y_min - y_margin)
                x_axis.setRange(x_lower, x_max + x_margin)
                y_axis.setRange(y_lower, y_max + y_margin)

            elif n_obj == 3 and _HAS_MPL_3D:
                # -- m=3: Scatter 3D interativo (matplotlib) --
                title_3d = (
                    f"Population 3D (f1 x f2 x f3) - {problem_label} - "
                    f"{algo_name} {run_label} | {run_timestamp}"
                )
                selected_point_3d = None
                mcdm_decision = self._mcdm_decision_for_run(run_payload)
                if isinstance(mcdm_decision, dict):
                    try:
                        point = np.asarray(mcdm_decision.get("selected_point", []), dtype=float).reshape(-1)
                        if point.size >= 3 and np.all(np.isfinite(point[:3])):
                            selected_point_3d = point[:3]
                    except Exception:  # noqa: BLE001
                        selected_point_3d = None
                self._show_3d_scatter(
                    front[:, :3], problem_pf, title_3d, anchor,
                    self.test_3d_container, self.test_3d_layout, self.test_chart_stack,
                    selected_point=selected_point_3d,
                    selected_label="MCDM selected",
                )
                # Metric summary (built below)
                summary_lines = [
                    f"Problem: {problem_label}",
                    f"Algorithm: {algo_name}",
                    f"Trial: {run_payload.get('run_index', '-')}",
                    f"Timestamp (en-US): {run_timestamp}",
                    f"Backend: {run_payload.get('backend', '-')}",
                    f"M: {run_payload.get('n_obj', '-')}",
                    f"D: {run_payload.get('n_var', '-')}",
                ]
                n_eval = _positive_int(run_payload.get("n_eval", run_payload.get("evaluations", 0)), 0, minimum=0)
                if is_partial:
                    eval_label = f" ({n_eval} evals)" if n_eval > 0 else ""
                    summary_lines.append(f"Status: running{eval_label}")
                elif n_eval > 0:
                    summary_lines.append(f"Evaluations: {n_eval}")
                for metric, value in run_payload.get("metrics", {}).items():
                    summary_lines.append(f"{metric}: {_fmt_number(_float_or_none(value), 6)}")
                speedup = _float_or_none(run_payload.get("profile_speedup_gpu_vs_cpu"))
                if speedup is not None and math.isfinite(speedup):
                    summary_lines.append(f"GPU speedup vs CPU: x{speedup:.3f}")
                self.test_result_summary.setPlainText("\n".join(summary_lines))
                return

            else:
                # -- m>=4: Parallel Coordinates (style used in classic EMO GUIs) --
                chart.setTitle(
                    f"Parallel Coordinates ({n_obj} obj) - {problem_label} - "
                    f"{algo_name} {run_label} | {run_timestamp}"
                )
                x_axis = QValueAxis()
                x_axis.setTitleText("Dimension No.")
                x_axis.setRange(0.5, n_obj + 0.5)
                x_axis.setTickCount(n_obj)
                x_axis.setLabelFormat("%d")
                y_axis = QValueAxis()
                y_axis.setTitleText("Value")
                chart.addAxis(x_axis, Qt.AlignmentFlag.AlignBottom)
                chart.addAxis(y_axis, Qt.AlignmentFlag.AlignLeft)

                y_min_val = float(np.min(front)) if anchor else float(np.min(front))
                y_max_val = float(np.max(front))

                # Draw reference PF first (background)
                if problem_pf is not None and problem_pf.ndim == 2 and problem_pf.shape[1] == n_obj:
                    pf_color = QColor(StylesAppStyles.colors.chart_reference_front)
                    pf_color.setAlpha(72)
                    pf_pen = QPen(pf_color, 1.0)
                    for row in problem_pf:
                        s = QLineSeries()
                        s.setPen(pf_pen)
                        for dim_idx in range(n_obj):
                            s.append(float(dim_idx + 1), float(row[dim_idx]))
                        chart.addSeries(s)
                        s.attachAxis(x_axis)
                        s.attachAxis(y_axis)
                    y_min_val = min(y_min_val, float(np.min(problem_pf)))
                    y_max_val = max(y_max_val, float(np.max(problem_pf)))
                    # Invisible series for legend
                    pf_legend = QLineSeries()
                    pf_legend.setName("Reference PF")
                    pf_legend.setPen(QPen(QColor(StylesAppStyles.colors.chart_reference_front), 2.0))
                    pf_legend.append(0, 0)
                    chart.addSeries(pf_legend)
                    pf_legend.attachAxis(x_axis)
                    pf_legend.attachAxis(y_axis)

                # Draw obtained solutions
                obtained_color = QColor(AppStyles.chart_palette()[0])
                obtained_color.setAlpha(160)
                obtained_pen = QPen(obtained_color, 1.5)
                for row in front:
                    s = QLineSeries()
                    s.setPen(obtained_pen)
                    for dim_idx in range(n_obj):
                        s.append(float(dim_idx + 1), float(row[dim_idx]))
                    chart.addSeries(s)
                    s.attachAxis(x_axis)
                    s.attachAxis(y_axis)
                # Invisible series for legend
                obt_legend = QLineSeries()
                obt_legend.setName("Obtained")
                obt_legend.setPen(QPen(QColor(AppStyles.chart_palette()[0]), 2.5))
                obt_legend.append(0, 0)
                chart.addSeries(obt_legend)
                obt_legend.attachAxis(x_axis)
                obt_legend.attachAxis(y_axis)

                # Hide all legend entries except the explicit legend helper series
                for s in chart.series():
                    if not s.name():
                        chart.legend().markers(s)[0].setVisible(False) if chart.legend().markers(s) else None

                if anchor:
                    y_min_val = min(y_min_val, 0.0)

                if y_min_val == y_max_val:
                    y_max_val = y_min_val + 1.0
                y_margin = (y_max_val - y_min_val) * 0.05
                y_lower = 0.0 if anchor else (y_min_val - y_margin)
                y_axis.setRange(y_lower, y_max_val + y_margin)

        summary_lines = [
            f"Problem: {problem_label}",
            f"Algorithm: {algo_name}",
            f"Trial: {run_payload.get('run_index', '-')}",
            f"Timestamp (en-US): {run_timestamp}",
            f"Backend: {run_payload.get('backend', '-')}",
            f"M: {run_payload.get('n_obj', '-')}",
            f"D: {run_payload.get('n_var', '-')}",
        ]
        n_eval = _positive_int(run_payload.get("n_eval", run_payload.get("evaluations", 0)), 0, minimum=0)
        if is_partial:
            eval_label = f" ({n_eval} evals)" if n_eval > 0 else ""
            summary_lines.append(f"Status: running{eval_label}")
        elif n_eval > 0:
            summary_lines.append(f"Evaluations: {n_eval}")
        for metric, value in run_payload.get("metrics", {}).items():
            summary_lines.append(f"{metric}: {_fmt_number(_float_or_none(value), 6)}")
        speedup = _float_or_none(run_payload.get("profile_speedup_gpu_vs_cpu"))
        if speedup is not None and math.isfinite(speedup):
            summary_lines.append(f"GPU speedup vs CPU: x{speedup:.3f}")
        self.test_result_summary.setPlainText("\n".join(summary_lines))
        self._apply_scientific_chart_theme(chart, (x_axis, y_axis))
        self.test_chart_view.setChart(chart)

    def _collect_test_config(self) -> dict[str, Any] | None:
        """Collect configuration for Test Module (1 algorithm, 1 problem, 1 run)."""
        algorithm_id = self._selected_algorithm_id()
        if not isinstance(algorithm_id, str) or algorithm_id not in self.algorithm_specs:
            QMessageBox.warning(self, "Validation", "Select one valid algorithm.")
            return None
        algorithm_ids = [algorithm_id]

        metric_ids = self._checked_ids(self.metric_list)
        metric_ids = self._map_metric_ids_to_backend(
            metric_ids,
            prefer_jax=self._test_backend_prefers_jax_for("metrics"),
        )
        if not metric_ids:
            QMessageBox.warning(self, "Validation", "Select at least one metric.")
            return None

        problem_id = self.problem_combo.currentData()
        if not isinstance(problem_id, str) or problem_id not in self.problem_specs:
            QMessageBox.warning(self, "Validation", "Select a valid problem.")
            return None

        # Test Module runs one problem at a time.
        problem_ids = [problem_id]

        # Ref point: auto-computed (automatic strategy)
        ref_point = None

        seed_mode = str(self.seed_mode_combo.currentData() or SEED_MODE_RANDOM).strip().lower()
        if seed_mode not in {SEED_MODE_RANDOM, SEED_MODE_FIXED}:
            seed_mode = SEED_MODE_RANDOM

        return {
            "problem_id": problem_id,
            "problem_ids": problem_ids,
            "n_var": self.n_var_spin.value(),
            "n_obj": self.n_obj_spin.value(),
            "pop_size": self.pop_size_spin.value(),
            "max_fe": self.max_fe_spin.value(),
            "maxFE": self.max_fe_spin.value(),
            "n_runs": 1,
            "seed_mode": seed_mode,
            "seed_base": int(self.seed_spin.value()),
            "seed_step": 1,
            "algorithm_ids": algorithm_ids,
            "metric_ids": metric_ids,
            "ref_point": ref_point,
            "use_pf": self.use_pf_check.isChecked(),
            "compute_backend": self.compute_backend_combo.currentData(),
            "gpu_dtype": "float32",
            "joblib_backend": "loky",
            "joblib_n_jobs": -1,
            "profile_compare": False,
            "stat_test_method": self.stat_method_combo.currentData(),
            "stat_test_reference": self.stat_reference_combo.currentText(),
            "stat_test_alpha": self._analysis_stat_alpha(),
            # Operators are intentionally hidden in UI; always use algorithm defaults.
            "crossover": "default",
            "mutation": "default",
            "selection": "default",
            "sampling": "default",
            # Keep canonical defaults in config for compatibility/debugging.
            "crossover_eta": 15.0,
            "crossover_prob": 0.9,
            "mutation_eta": 20.0,
            "mutation_prob": None,
            "selection_pressure": 2,
            "parallel_workers": 1,  # Test Module sempre usa 1 worker
        }

    @Slot()
    def _start_test(self) -> None:
        """Start Test Module execution (1 algorithm, 1 problem, 1 run)."""
        if self.test_worker_thread is not None:
            QMessageBox.information(self, "Execution", "A test is already running.")
            return

        config = self._collect_test_config()
        if config is None:
            return

        # Reset test module state
        self.test_results.clear()
        self.test_metric_names = []
        self.test_pareto_front = None
        self.test_pareto_fronts = {}
        self.test_profile_compare_enabled = False

        # Keep Analysis tab history/results across Test Module runs.
        # New final results are merged in _on_test_finished via _accumulate_payload().
        self.test_problem_combo.clear()
        self.test_algo_combo.clear()
        self.test_metric_combo.clear()
        self.test_run_spin.setRange(1, 1)
        self.test_result_summary.clear()
        self.test_live_payload = None

        self.progress.setValue(0)
        self.log_box.clear()
        self._set_empty_test_result_chart()

        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)

        seed_mode = str(config.get("seed_mode", SEED_MODE_RANDOM)).strip().lower()
        self._append_log(f"Starting test (termination=n_eval, seed_mode={seed_mode})...")

        self.test_worker_thread = QThread(self)
        self.test_worker = ExperimentBridge(
            config=config,
            algorithm_specs=self.algorithm_specs,
            problem_specs=self.problem_specs,
            metric_specs=self.metric_specs,
        )
        self.test_worker.moveToThread(self.test_worker_thread)

        self.test_worker_thread.started.connect(self.test_worker.run)
        self.test_worker.progress.connect(self._on_test_progress)
        self.test_worker.run_ready.connect(self._on_test_run_ready)
        self.test_worker.partial_result.connect(self._on_test_partial_result)
        self.test_worker.warning.connect(self._on_test_warning)
        self.test_worker.error.connect(self._on_test_error)
        self.test_worker.finished.connect(self._on_test_finished)

        self.test_worker.finished.connect(self.test_worker_thread.quit)
        self.test_worker.error.connect(self.test_worker_thread.quit)
        self.test_worker_thread.finished.connect(self._on_test_thread_finished)

        self.test_worker_thread.start()

    @Slot(object)
    def _on_test_partial_result(self, payload: dict[str, Any]) -> None:
        """Handle real-time updates from the test worker."""
        import time
        now = time.perf_counter()
        if not hasattr(self, "_last_test_chart_update"):
            self._last_test_chart_update = 0.0

        normalized = self._normalize_test_live_payload(payload)
        self.test_live_payload = normalized

        metric_keys = list(normalized.get("history", {}).keys())
        if not metric_keys:
            metric_keys = [str(key) for key in normalized.get("metrics", {}).keys()]
        for metric_name in metric_keys:
            if metric_name not in self.test_metric_names:
                self.test_metric_names.append(metric_name)
        self.metric_names = list(self.test_metric_names)
        self._sync_test_metric_combo()

        # Limit UI redraw to max 10 FPS to prevent freezing the main thread
        if now - self._last_test_chart_update < 0.1:
            return
        self._last_test_chart_update = now

        self._refresh_test_result_chart()

    @Slot()
    def _stop_test(self) -> None:
        """Stop running test execution."""
        if self.test_worker is None:
            return
        self.test_worker.cancel()
        self.btn_stop.setEnabled(False)

    @Slot(int, str)
    def _on_test_progress(self, percent: int, message: str) -> None:
        """Handle progress updates from Test Module worker."""
        self.progress.setValue(percent)
        self._append_log(message)

    @Slot(object)
    def _on_test_run_ready(self, payload: dict[str, Any]) -> None:
        """Handle individual run result from Test Module worker."""
        payload = self._ensure_run_dimensions(payload)
        # Keep a final preview in the Test Module chart, but do not persist yet.
        # Persistence happens only in _on_test_finished to avoid duplicate rows in Analysis.
        normalized_preview = self._normalize_test_live_payload(payload)
        normalized_preview["is_partial"] = False
        self.test_live_payload = normalized_preview
        algo = payload.get("algorithm", "algorithm")
        problem = payload.get("problem", "problem")
        trial = payload.get("run_index", "-")
        backend = payload.get("backend", "CPU")
        n_obj = payload.get("n_obj", "-")
        n_var = payload.get("n_var", "-")
        metrics_str = ", ".join(f"{k}={_fmt_number(v, 5)}" for k, v in payload.get("metrics", {}).items())
        speedup = _float_or_none(payload.get("profile_speedup_gpu_vs_cpu"))
        if speedup is not None and math.isfinite(speedup):
            self._append_log(
                f"Result: {problem} | {algo} trial {trial} [{backend}, M={n_obj}, D={n_var}] -> "
                f"{metrics_str} | GPU speedup x{speedup:.3f}"
            )
        else:
            self._append_log(f"Result: {problem} | {algo} trial {trial} [{backend}, M={n_obj}, D={n_var}] -> {metrics_str}")

        block_profile = payload.get("block_profile")
        if isinstance(block_profile, list) and block_profile:
            top = block_profile[0] if isinstance(block_profile[0], dict) else None
            if top is not None:
                block = str(top.get("block", "block"))
                avg_ms = _float_or_none(top.get("avg_ms"))
                calls = _positive_int(top.get("calls"), 0, minimum=0)
                if avg_ms is not None and math.isfinite(avg_ms):
                    self._append_log(f"Profiler top block: {block} avg={avg_ms:.3f} ms ({calls} calls)")

        metric_keys = [str(key) for key in payload.get("metrics", {}).keys()]
        if not self.test_metric_names:
            self.test_metric_names = metric_keys
        else:
            for metric_name in metric_keys:
                if metric_name not in self.test_metric_names:
                    self.test_metric_names.append(metric_name)
        self.metric_names = list(self.test_metric_names)

        # Analysis data is refreshed only when final payload arrives.
        self._refresh_test_result_chart()

    @Slot(str)
    def _on_test_warning(self, message: str) -> None:
        """Handle warning from Test Module worker."""
        text = str(message)
        is_jax_gpu_fallback_notice = (
            (
                "GPU requested, but unavailable:" in text
                or "JAX backend requested, but unavailable:" in text
            )
            and "No JAX GPU device detected" in text
        )
        if is_jax_gpu_fallback_notice:
            self._append_log(
                "Info: JAX GPU not detected. Using JAX on CPU (CPU (JAX)) "
                "for JAX-compatible problems when available."
            )
            return
        self._append_log(f"Warning: {text}")

    @Slot(str)
    def _on_test_error(self, message: str) -> None:
        """Handle error from Test Module worker."""
        self._append_log(f"Error: {message}")
        QMessageBox.critical(self, "Test failed", message)

    @Slot(object)
    def _on_test_finished(self, payload: dict[str, Any]) -> None:
        """Handle final result from Test Module worker."""
        self.test_live_payload = None # Clear live data
        self.test_results = dict(payload.get("results", {}))
        self.test_metric_names = list(payload.get("metrics", []))
        self.test_execution_backend_label = str(
            payload.get("execution_backend_label", self.test_execution_backend_label)
        )
        self.test_profile_compare_enabled = bool(payload.get("profile_compare_enabled", False))

        self.test_pareto_fronts = {}
        pf_map_raw = payload.get("pareto_fronts")
        if isinstance(pf_map_raw, dict):
            for problem_id, pf_raw in pf_map_raw.items():
                key = str(problem_id)
                if pf_raw is None:
                    self.test_pareto_fronts[key] = None
                else:
                    self.test_pareto_fronts[key] = np.asarray(pf_raw, dtype=float)

        pf_raw = payload.get("pareto_front")
        self.test_pareto_front = None if pf_raw is None else np.asarray(pf_raw, dtype=float)
        if self.test_pareto_front is None:
            for value in self.test_pareto_fronts.values():
                if value is not None:
                    self.test_pareto_front = value
                    break

        # Accumulate new results for the Analysis tab
        self._accumulate_payload(payload)

        self._append_log(f"Effective backend: {self.test_execution_backend_label}")

        # Update label with auto-computed ref_point
        hv_ref = payload.get("hv_ref_point")
        hv_src = payload.get("hv_ref_source", "auto")
        if isinstance(hv_ref, list) and hv_ref:
            ref_str = ", ".join(f"{v:.4f}" for v in hv_ref)
            self.ref_point_label.setText(f"[{ref_str}] ({hv_src})")
            self.ref_point_label.setStyleSheet(f"color: {AppStyles.SUCCESS}; font-weight: 600;")
            self._append_log(f"HV ref_point ({hv_src}): [{ref_str}]")

        if payload.get("cancelled"):
            self._append_log("Execution cancelled.")
        else:
            self._append_log("Execution finished.")

        self.progress.setValue(100 if not payload.get("cancelled") else self.progress.value())

        latest_run = self._latest_run_from_payload(payload)

        self._refresh_plot_controls()
        if latest_run is not None:
            self._focus_analysis_run(latest_run)
        else:
            self._refresh_convergence_chart()
            self._refresh_pareto_chart()

        self._refresh_test_result_controls()
        if latest_run is not None:
            self._focus_test_run(latest_run)
        else:
            self._refresh_test_result_chart()
        # Switch to Experiment Module to show results
        self.tabs.setCurrentWidget(self.results_tab)
        self._restore_maximized_window_mode()

    @Slot()
    def _on_test_thread_finished(self) -> None:
        """Clean up Test Module worker thread."""
        if self.test_worker is not None:
            self.test_worker.deleteLater()
        if self.test_worker_thread is not None:
            self.test_worker_thread.deleteLater()

        self.test_worker = None
        self.test_worker_thread = None
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self._restore_maximized_window_mode()

    def _refresh_results_table(self) -> None:
        self._update_stat_controls_state()
        all_runs = self._all_analysis_runs()
        if not all_runs:
            self.results_table.setRowCount(0)
            self.results_table.setColumnCount(0)
            self.analysis_row_meta = {}
            self._last_stat_result = None
            if hasattr(self, "stats_header_label"):
                self.stats_header_label.setText("EmoPyLab | Statistical View: No Results")
            return

        stat_mode = str(self.stat_method_combo.currentData()) if hasattr(self, "stat_method_combo") else "none"
        if stat_mode not in {"none", ""}:
            self._refresh_statistical_results_table(stat_mode)
            return

        self._last_stat_result = None
        if hasattr(self, "stats_header_label"):
            self.stats_header_label.setText("EmoPyLab | Statistical View: Detailed Runs")
        if hasattr(self, "stats_warning_label") and core_scipy_stats_available:
            self.stats_warning_label.setText("")

        updates_enabled = self.results_table.updatesEnabled()
        self.results_table.setUpdatesEnabled(False)

        has_profile_columns = self.profile_compare_enabled or any(
            run.get("profile_cpu_time_s") is not None
            or run.get("profile_gpu_time_s") is not None
            or run.get("profile_speedup_gpu_vs_cpu") is not None
            for run in all_runs
        )

        profile_columns = ["CPU_s", "GPU_s", "GPU_speedup_x"] if has_profile_columns else []
        columns = [
            "Algorithm",
            "Problem",
            "Trial",
            "Timestamp (en-US)",
            "Seed",
            "Backend",
            "Time_s",
            "Evaluations",
            "M",
            "D",
        ] + profile_columns + self.metric_names

        grouped_runs: dict[tuple[str, str], list[dict[str, Any]]] = {}
        for run_payload in all_runs:
            grouped_runs.setdefault(self._analysis_group_key(run_payload), []).append(run_payload)

        ordered_groups = sorted(
            grouped_runs.items(),
            key=lambda item: max(self._run_timestamp_epoch(run) for run in item[1]),
            reverse=True,
        )

        all_group_ids = {self._analysis_group_id_from_key(group_key) for group_key, _ in ordered_groups}
        self.analysis_group_expanded.intersection_update(all_group_ids)
        if not self.analysis_group_user_touched and not self.analysis_group_expanded and ordered_groups:
            self.analysis_group_expanded.add(self._analysis_group_id_from_key(ordered_groups[0][0]))

        row_count = 0
        for group_key, runs in ordered_groups:
            group_id = self._analysis_group_id_from_key(group_key)
            is_expanded = group_id in self.analysis_group_expanded
            row_count += 1 + (len(runs) if is_expanded else 0)

        self.results_table.clear()
        self.results_table.setColumnCount(len(columns))
        self.results_table.setHorizontalHeaderLabels(columns)
        self.results_table.setRowCount(row_count)
        self.analysis_row_meta = {}

        col_idx = {
            "algorithm": 0,
            "problem": 1,
            "trial": 2,
            "timestamp": 3,
            "seed": 4,
            "backend": 5,
            "time_s": 6,
            "evaluations": 7,
            "n_obj": 8,
            "n_var": 9,
        }
        profile_start_col = 10
        metric_start_col = profile_start_col + len(profile_columns)
        latest_run = max(all_runs, key=lambda run: self._run_timestamp_epoch(run), default=None)
        latest_row = -1
        summary_row_by_group: dict[str, int] = {}

        row = 0
        for group_key, runs_in_group in ordered_groups:
            group_id = self._analysis_group_id_from_key(group_key)
            summary_row_by_group[group_id] = row
            problem_runs = sorted(runs_in_group, key=lambda run: self._run_timestamp_epoch(run), reverse=True)
            latest_problem_run = problem_runs[0]
            is_expanded = group_id in self.analysis_group_expanded
            algo_label = self._algorithm_label_for_run(latest_problem_run)
            problem_label = self._problem_label_for_run(latest_problem_run)

            self._set_table_item(row, col_idx["algorithm"], algo_label)
            self._set_table_item(row, col_idx["problem"], problem_label)
            marker = "[-]" if is_expanded else "[+]"
            trial_count = len(problem_runs)
            suffix = "trial" if trial_count == 1 else "trials"
            self._set_table_item(row, col_idx["trial"], f"{marker} summary ({trial_count} {suffix})")
            self._set_table_item(row, col_idx["timestamp"], str(latest_problem_run.get("timestamp_en_us", "-")))
            self._set_table_item(row, col_idx["seed"], "-")
            backend_values = sorted(
                {
                    str(item.get("backend", self.execution_backend_label)).strip()
                    for item in problem_runs
                    if str(item.get("backend", "")).strip()
                }
            )
            backend_summary = backend_values[0] if len(backend_values) == 1 else "mixed"
            self._set_table_item(
                row,
                col_idx["backend"],
                backend_summary if backend_summary else self.execution_backend_label,
            )
            self._set_table_item(row, col_idx["time_s"], "-")
            self._set_table_item(row, col_idx["evaluations"], "-")
            self._set_table_item(row, col_idx["n_obj"], str(latest_problem_run.get("n_obj", "-")))
            self._set_table_item(row, col_idx["n_var"], str(latest_problem_run.get("n_var", "-")))

            if has_profile_columns:
                cpu_values = [_float_or_nan(item.get("profile_cpu_time_s")) for item in problem_runs]
                gpu_values = [_float_or_nan(item.get("profile_gpu_time_s")) for item in problem_runs]
                speed_values = [_float_or_nan(item.get("profile_speedup_gpu_vs_cpu")) for item in problem_runs]

                for col, values in [
                    (profile_start_col + 0, cpu_values),
                    (profile_start_col + 1, gpu_values),
                    (profile_start_col + 2, speed_values),
                ]:
                    mean, std = _mean_std(values)
                    text = "-" if not math.isfinite(mean) else f"{mean:.4g} +- {std:.2g}"
                    self._set_table_item(row, col, text)

            for idx, metric_name in enumerate(self.metric_names, start=metric_start_col):
                values = [_float_or_nan(item.get("metrics", {}).get(metric_name)) for item in problem_runs]
                mean, std = _mean_std(values)
                text = "-" if not math.isfinite(mean) else f"{mean:.4g} +- {std:.2g}"
                self._set_table_item(row, idx, text)

            for col in range(self.results_table.columnCount()):
                item = self.results_table.item(row, col)
                if item is None:
                    continue
                item.setBackground(QColor(AppStyles.PANEL_ALT))
                item.setToolTip("Click summary row to expand or collapse this algorithm/problem group.")
            self.analysis_row_meta[row] = {"type": "summary", "group_id": group_id}
            row += 1

            if not is_expanded:
                continue

            for run_payload in problem_runs:
                self._set_table_item(row, col_idx["algorithm"], "")
                self._set_table_item(row, col_idx["problem"], "")
                self._set_table_item(row, col_idx["trial"], f"  -> trial {run_payload.get('run_index', '-')}")
                self._set_table_item(row, col_idx["timestamp"], str(run_payload.get("timestamp_en_us", "-")))
                self._set_table_item(row, col_idx["seed"], str(run_payload.get("seed", "-")))
                self._set_table_item(
                    row,
                    col_idx["backend"],
                    str(run_payload.get("backend", self.execution_backend_label)),
                )
                self._set_table_item(row, col_idx["time_s"], _fmt_number(run_payload.get("time_s"), 6))
                self._set_table_item(row, col_idx["evaluations"], str(run_payload.get("evaluations", "-")))
                self._set_table_item(row, col_idx["n_obj"], str(run_payload.get("n_obj", "-")))
                self._set_table_item(row, col_idx["n_var"], str(run_payload.get("n_var", "-")))

                if has_profile_columns:
                    self._set_table_item(
                        row, profile_start_col + 0, _fmt_number(run_payload.get("profile_cpu_time_s"), 6)
                    )
                    self._set_table_item(
                        row, profile_start_col + 1, _fmt_number(run_payload.get("profile_gpu_time_s"), 6)
                    )
                    self._set_table_item(
                        row,
                        profile_start_col + 2,
                        _fmt_number(run_payload.get("profile_speedup_gpu_vs_cpu"), 6),
                    )

                for idx, metric_name in enumerate(self.metric_names, start=metric_start_col):
                    value = run_payload.get("metrics", {}).get(metric_name)
                    self._set_table_item(row, idx, _fmt_number(value, 6))

                self.analysis_row_meta[row] = {"type": "trial", "group_id": group_id, "run": dict(run_payload)}
                if latest_run is not None and latest_row < 0 and self._same_run_identity(run_payload, latest_run):
                    latest_row = row
                row += 1

        if latest_row >= 0:
            self._highlight_results_row(latest_row, note="Latest execution (auto-focused).")
        elif latest_run is not None:
            latest_group_id = self._analysis_group_id_for_run(latest_run)
            summary_row = summary_row_by_group.get(latest_group_id, -1)
            if summary_row >= 0:
                self._highlight_results_row(summary_row, note="Latest execution is inside this collapsed group.")

        self.results_table.resizeRowsToContents()
        self.results_table.setUpdatesEnabled(updates_enabled)

    def _set_table_item(self, row: int, col: int, text: str) -> None:
        item = QTableWidgetItem(text)
        item.setFlags(item.flags() ^ Qt.ItemFlag.ItemIsEditable)
        self.results_table.setItem(row, col, item)

    @Slot(int, int)
    def _on_results_table_cell_clicked(self, row: int, _col: int) -> None:
        meta = self.analysis_row_meta.get(int(row), {})
        row_type = str(meta.get("type", "")).strip().lower()
        if row_type == "summary":
            group_id = str(meta.get("group_id", "")).strip()
            if not group_id:
                return
            if group_id in self.analysis_group_expanded:
                self.analysis_group_expanded.remove(group_id)
            else:
                self.analysis_group_expanded.add(group_id)
            self.analysis_group_user_touched = True
            self._refresh_results_table()
            return

        if row_type == "trial":
            run_payload = meta.get("run")
            if isinstance(run_payload, dict):
                self._focus_analysis_run(run_payload)

    def _highlight_results_row(self, row: int, *, note: str = "") -> None:
        if row < 0:
            return
        col_count = self.results_table.columnCount()
        if col_count <= 0:
            return

        bg_color = QColor(AppStyles.PRIMARY_LIGHT).lighter(170)
        border_font = QFont()
        border_font.setBold(True)

        for col in range(col_count):
            item = self.results_table.item(row, col)
            if item is None:
                item = QTableWidgetItem("")
                item.setFlags(item.flags() ^ Qt.ItemFlag.ItemIsEditable)
                self.results_table.setItem(row, col, item)
            item.setBackground(bg_color)
            if note:
                item.setToolTip(note)

        trial_col = 2
        trial_item = self.results_table.item(row, trial_col)
        if trial_item is not None:
            text = trial_item.text().strip()
            if "(latest)" not in text.lower():
                trial_item.setText(f"{text} (latest)")
            trial_item.setFont(border_font)

    @staticmethod
    def _metric_higher_better(metric_name: str) -> bool:
        name = metric_name.split("[", 1)[0].strip().lower()
        if name.startswith("hv") or name.endswith("_hv") or "hypervolume" in name:
            return True
        return name in {"cpf", "dm", "pd", "feasible_rate", "feasible rate"}

    def _analysis_stat_alpha(self) -> float:
        if hasattr(self, "stat_alpha_spin"):
            return float(self.stat_alpha_spin.value())
        return 0.05

    def _analysis_stat_block_key(self, run_payload: dict[str, Any]) -> str:
        problem_id = str(run_payload.get("problem_id", "")).strip().lower()
        if not problem_id:
            problem_id = self._problem_label_for_run(run_payload).strip().lower()

        run_index = str(run_payload.get("run_index", "")).strip()
        seed = str(run_payload.get("seed", "")).strip()
        if run_index:
            run_tag = f"run:{run_index}"
        elif seed:
            run_tag = f"seed:{seed}"
        else:
            epoch = _float_or_none(run_payload.get("timestamp_epoch"))
            if epoch is not None and math.isfinite(epoch):
                run_tag = f"ts:{epoch:.6f}"
            else:
                run_tag = str(run_payload.get("timestamp_iso", "")).strip() or str(
                    run_payload.get("timestamp_en_us", "")
                ).strip()
                if not run_tag:
                    run_tag = "unknown"
                run_tag = f"ts:{run_tag}"
        return f"{problem_id}||{run_tag}"

    def _collect_statistical_blocks(
        self,
        metric_name: str,
        algorithms: list[str],
    ) -> tuple[dict[str, list[float]], dict[str, dict[str, float]], str]:
        values_by_algo: dict[str, list[float]] = {algo: [] for algo in algorithms}
        block_map: dict[str, dict[str, float]] = {}
        backend_values: set[str] = set()

        selected_problem_id = self._selected_plot_problem_id()
        for run_payload in self._all_analysis_runs():
            algo_name = self._algorithm_label_for_run(run_payload).strip()
            if algo_name not in values_by_algo:
                continue

            run_problem_id = str(run_payload.get("problem_id", "")).strip()
            if selected_problem_id is not None and run_problem_id != selected_problem_id:
                continue

            value = _float_or_nan(run_payload.get("metrics", {}).get(metric_name))
            if not math.isfinite(value):
                continue

            values_by_algo[algo_name].append(float(value))
            block_key = self._analysis_stat_block_key(run_payload)
            block_map.setdefault(block_key, {})[algo_name] = float(value)
            backend_label = str(run_payload.get("backend", "")).strip()
            if backend_label:
                backend_values.add(backend_label)

        if len(backend_values) == 1:
            backend_context = next(iter(backend_values))
        elif len(backend_values) > 1:
            backend_context = "mixed"
        else:
            backend_context = self.execution_backend_label
        return values_by_algo, block_map, backend_context

    @Slot()
    def _on_stat_controls_changed(self) -> None:
        self._update_stat_controls_state()
        self._refresh_results_table()

    @Slot()
    def _run_statistical_validation(self) -> None:
        stat_mode = str(self.stat_method_combo.currentData()) if hasattr(self, "stat_method_combo") else "none"
        if stat_mode in {"none", ""}:
            self._refresh_results_table()
            return
        if not core_scipy_stats_available:
            details = f"\n\n{core_scipy_stats_error}" if core_scipy_stats_error else ""
            QMessageBox.warning(
                self,
                "Statistical Validation",
                "scipy is unavailable. Install scipy to execute Wilcoxon/Friedman."
                + details,
            )
            self._update_stat_controls_state()
            return
        self._refresh_results_table()

    def _update_stat_controls_state(self) -> None:
        stat_mode = str(self.stat_method_combo.currentData()) if hasattr(self, "stat_method_combo") else "none"
        is_stat_mode = stat_mode in {"wilcoxon", "friedman"}
        allow_reference = stat_mode == "wilcoxon"

        if hasattr(self, "stat_reference_label"):
            self.stat_reference_label.setEnabled(allow_reference)
        if hasattr(self, "stat_reference_combo"):
            self.stat_reference_combo.setEnabled(allow_reference and self.stat_reference_combo.count() > 0)
        if hasattr(self, "stat_alpha_label"):
            self.stat_alpha_label.setEnabled(is_stat_mode)
        if hasattr(self, "stat_alpha_spin"):
            self.stat_alpha_spin.setEnabled(is_stat_mode)
        if hasattr(self, "btn_run_stats"):
            self.btn_run_stats.setEnabled(is_stat_mode and core_scipy_stats_available)

        warning = ""
        if is_stat_mode and not core_scipy_stats_available:
            warning = "scipy unavailable: statistical tests disabled."
            if core_scipy_stats_error:
                warning += f" Details: {core_scipy_stats_error}"
        if hasattr(self, "stats_warning_label"):
            self.stats_warning_label.setText(warning)

    def _refresh_statistical_results_table(self, stat_mode: str) -> None:
        self._last_stat_result = None
        self.analysis_row_meta = {}
        self._update_stat_controls_state()

        metric_name = self.metric_plot_combo.currentText().strip()
        if not metric_name:
            self.results_table.setRowCount(0)
            self.results_table.setColumnCount(0)
            if hasattr(self, "stats_header_label"):
                self.stats_header_label.setText("EmoPyLab | Statistical view: select a metric")
            return

        selected_problem_id = self._selected_plot_problem_id()
        algorithms = self._analysis_algorithm_options(selected_problem_id)
        if not algorithms and selected_problem_id is not None:
            algorithms = self._analysis_algorithm_options()
        if not algorithms:
            self.results_table.setRowCount(0)
            self.results_table.setColumnCount(0)
            if hasattr(self, "stats_header_label"):
                self.stats_header_label.setText("EmoPyLab | Statistical view: no algorithm data")
            return

        higher_better = self._metric_higher_better(metric_name)
        alpha = self._analysis_stat_alpha()
        values_by_algo, block_map, backend_context = self._collect_statistical_blocks(metric_name, algorithms)

        rows: list[dict[str, Any]] = []
        note_parts: list[str] = []
        global_p: float | None = None
        global_effect: float | None = None
        reference_algorithm = ""

        if stat_mode == "wilcoxon":
            reference_algorithm = self.stat_reference_combo.currentText().strip()
            if reference_algorithm not in algorithms:
                if algorithms:
                    reference_algorithm = algorithms[0]
                    self.stat_reference_combo.blockSignals(True)
                    self.stat_reference_combo.setCurrentText(reference_algorithm)
                    self.stat_reference_combo.blockSignals(False)
                else:
                    note_parts.append("No reference algorithm available.")

            for algo_name in algorithms:
                values = values_by_algo.get(algo_name, [])
                mean, std = _mean_std(values)
                if algo_name == reference_algorithm:
                    rows.append(
                        {
                            "algorithm": algo_name,
                            "n": len(values),
                            "mean": mean,
                            "std": std,
                            "p_value": float("nan"),
                            "effect": float("nan"),
                            "decision": "REF",
                        }
                    )
                    continue

                paired_keys = [
                    key
                    for key, entry in block_map.items()
                    if reference_algorithm in entry and algo_name in entry
                ]
                paired_keys.sort()
                algo_pair = [block_map[key][algo_name] for key in paired_keys]
                ref_pair = [block_map[key][reference_algorithm] for key in paired_keys]
                test_result = core_run_wilcoxon(
                    algo_pair,
                    ref_pair,
                    alpha=alpha,
                    higher_better=higher_better,
                    min_samples=self.stat_min_samples,
                )
                rows.append(
                    {
                        "algorithm": algo_name,
                        "n": int(test_result.get("n", 0)),
                        "mean": mean,
                        "std": std,
                        "p_value": _float_or_nan(test_result.get("p_value")),
                        "effect": _float_or_nan(test_result.get("effect")),
                        "decision": str(test_result.get("decision", "-")),
                    }
                )
                error_text = str(test_result.get("error", "")).strip()
                if error_text:
                    note_parts.append(f"{algo_name}: {error_text}")

        elif stat_mode == "friedman":
            if len(algorithms) < 3:
                note_parts.append("Friedman requires at least 3 algorithms.")

            complete_blocks: list[list[float]] = []
            for key in sorted(block_map):
                entry = block_map[key]
                if all(algo in entry for algo in algorithms):
                    complete_blocks.append([entry[algo] for algo in algorithms])

            friedman_result = core_run_friedman(
                complete_blocks,
                alpha=alpha,
                higher_better=higher_better,
                min_blocks=self.stat_min_samples,
            )
            avg_ranks = list(friedman_result.get("avg_ranks", []))
            decisions = list(friedman_result.get("decisions", []))
            global_p = _float_or_none(friedman_result.get("p_value"))
            global_effect = _float_or_none(friedman_result.get("kendall_w"))

            for idx, algo_name in enumerate(algorithms):
                values = values_by_algo.get(algo_name, [])
                mean, std = _mean_std(values)
                rank_value = avg_ranks[idx] if idx < len(avg_ranks) else float("nan")
                decision = decisions[idx] if idx < len(decisions) else "="
                rows.append(
                    {
                        "algorithm": algo_name,
                        "n": int(friedman_result.get("n_blocks", 0)),
                        "mean": mean,
                        "std": std,
                        "p_value": _float_or_nan(friedman_result.get("p_value")),
                        "effect": _float_or_nan(rank_value),
                        "decision": str(decision),
                    }
                )

            error_text = str(friedman_result.get("error", "")).strip()
            if error_text:
                note_parts.append(error_text)
            if math.isfinite(_float_or_nan(global_effect)):
                note_parts.append(f"Kendall W={float(global_effect):.4g}")
        else:
            note_parts.append("Statistical method not selected.")

        note = " | ".join(dict.fromkeys(part for part in note_parts if part))
        summary = core_summarize_stat_results(
            method=stat_mode,
            metric_name=metric_name,
            alpha=alpha,
            backend_label=backend_context,
            rows=rows,
            higher_better=higher_better,
            reference_algorithm=reference_algorithm,
            note=note,
            global_p_value=global_p,
            global_effect=global_effect,
        )
        self._last_stat_result = dict(summary)

        columns = list(summary.get("columns", []))
        summary_rows = list(summary.get("rows", []))
        self.results_table.clear()
        self.results_table.setColumnCount(len(columns))
        self.results_table.setHorizontalHeaderLabels(columns)
        self.results_table.setRowCount(len(summary_rows))
        for row_idx, row in enumerate(summary_rows):
            self._set_table_item(row_idx, 0, str(row.get("algorithm", "")))
            self._set_table_item(row_idx, 1, str(row.get("n", 0)))
            self._set_table_item(row_idx, 2, str(row.get("mean_std_text", "--")))
            self._set_table_item(row_idx, 3, str(row.get("p_value_text", "--")))
            effect_text = str(row.get("effect_text", "--"))
            if stat_mode == "friedman" and effect_text != "--":
                effect_text = f"rank={effect_text}"
            self._set_table_item(row_idx, 4, effect_text)
            self._set_table_item(row_idx, 5, str(row.get("decision", "-")))

        if hasattr(self, "stats_header_label"):
            self.stats_header_label.setText(str(summary.get("header_text", "EmoPyLab | Statistical view")))
        if hasattr(self, "stats_warning_label"):
            base_warning = self.stats_warning_label.text().strip()
            if note:
                if base_warning and note not in base_warning:
                    self.stats_warning_label.setText(f"{base_warning} | {note}")
                else:
                    self.stats_warning_label.setText(note)
            else:
                self.stats_warning_label.setText(base_warning)

        self.results_table.resizeRowsToContents()

    def _selected_plot_problem_id(self) -> str | None:
        value = self.problem_plot_combo.currentData()
        if isinstance(value, str) and value and value != "__all__":
            return value
        return None

    def _runs_for_plot(self, algo_name: str) -> list[dict[str, Any]]:
        raw_runs = list(self.results.get(algo_name, []))
        runs: list[dict[str, Any]] = []
        seen: set[str] = set()
        for run_payload in raw_runs:
            if not isinstance(run_payload, dict):
                continue
            normalized = self._ensure_run_dimensions(dict(run_payload))
            run_key = self._analysis_run_key(normalized)
            if run_key in seen:
                continue
            seen.add(run_key)
            runs.append(normalized)
        selected_problem_id = self._selected_plot_problem_id()
        if selected_problem_id is not None:
            runs = [run for run in runs if str(run.get("problem_id", "")) == selected_problem_id]
        runs.sort(key=lambda run: self._run_timestamp_epoch(run), reverse=True)
        return runs

    def _refresh_plot_controls(self) -> None:
        current_metric = self.metric_plot_combo.currentText().strip()
        self.metric_plot_combo.blockSignals(True)
        self.metric_plot_combo.clear()
        self.metric_plot_combo.addItems(self.metric_names)
        if current_metric and self.metric_plot_combo.findText(current_metric) >= 0:
            self.metric_plot_combo.setCurrentText(current_metric)
        self.metric_plot_combo.blockSignals(False)

        current_problem_id = self.problem_plot_combo.currentData()
        problem_entries: dict[str, str] = {}
        for run in self._all_analysis_runs():
            pid = str(run.get("problem_id", "")).strip()
            plabel = self._problem_label_for_run(run)
            if pid:
                problem_entries[pid] = plabel

        self.problem_plot_combo.blockSignals(True)
        self.problem_plot_combo.clear()
        self.problem_plot_combo.addItem("All", "__all__")
        for problem_id, problem_label in sorted(problem_entries.items(), key=lambda x: x[1].lower()):
            self.problem_plot_combo.addItem(problem_label, problem_id)
        target_problem = "__all__"
        if isinstance(current_problem_id, str) and self.problem_plot_combo.findData(current_problem_id) >= 0:
            target_problem = current_problem_id
        elif len(problem_entries) == 1:
            target_problem = next(iter(problem_entries))
        target_idx = self.problem_plot_combo.findData(target_problem)
        if target_idx >= 0:
            self.problem_plot_combo.setCurrentIndex(target_idx)
        self.problem_plot_combo.blockSignals(False)

        current_ref = self.stat_reference_combo.currentText().strip()
        self.stat_reference_combo.blockSignals(True)
        self.stat_reference_combo.clear()
        self.stat_reference_combo.addItems(self._analysis_algorithm_options())
        if current_ref and self.stat_reference_combo.findText(current_ref) >= 0:
            self.stat_reference_combo.setCurrentText(current_ref)
        elif self.stat_reference_combo.count() > 0:
            self.stat_reference_combo.setCurrentIndex(0)
        self.stat_reference_combo.blockSignals(False)
        self._update_stat_controls_state()

        self._on_plot_problem_changed()
        self._refresh_results_table()

    @Slot()
    def _on_plot_problem_changed(self) -> None:
        current_algo = self.algo_plot_combo.currentText().strip()
        selected_problem_id = self._selected_plot_problem_id()
        algo_names = self._analysis_algorithm_options(selected_problem_id)
        if not algo_names and selected_problem_id is not None:
            algo_names = self._analysis_algorithm_options()

        self.algo_plot_combo.blockSignals(True)
        self.algo_plot_combo.clear()
        self.algo_plot_combo.addItems(algo_names)
        if current_algo and current_algo in algo_names:
            self.algo_plot_combo.setCurrentText(current_algo)
        elif self.algo_plot_combo.count() > 0:
            self.algo_plot_combo.setCurrentIndex(0)
        self.algo_plot_combo.blockSignals(False)

        self._on_plot_algo_changed()
        self._refresh_convergence_chart()

    @Slot()
    def _on_plot_algo_changed(self) -> None:
        algo_name = self.algo_plot_combo.currentText().strip()
        run_count = len(self._runs_for_plot(algo_name))
        self.run_plot_spin.setRange(1, max(1, run_count))
        self.run_plot_spin.setEnabled(run_count > 0)
        self._refresh_pareto_chart()

    def _set_empty_convergence_chart(self, note: str | None = None) -> None:
        chart = QChart()
        chart.setTitle(str(note).strip() if str(note or "").strip() else "Convergence")
        chart.legend().setVisible(True)

        x_axis = QValueAxis()
        x_axis.setTitleText("Function Evaluations (FE)")
        x_axis.setRange(0, 1)

        y_axis = QValueAxis()
        y_axis.setTitleText("Metric")
        y_axis.setRange(0, 1)

        chart.addAxis(x_axis, Qt.AlignmentFlag.AlignBottom)
        chart.addAxis(y_axis, Qt.AlignmentFlag.AlignLeft)
        self._apply_scientific_chart_theme(chart, (x_axis, y_axis))
        self.conv_chart_view.setChart(chart)

    def _set_empty_pareto_chart(self) -> None:
        self.pareto_chart_stack.setCurrentIndex(0)
        chart = QChart()
        chart.setTitle("Pareto front (f1 x f2)")
        chart.legend().setVisible(True)

        x_axis = QValueAxis()
        x_axis.setTitleText("f1")
        x_axis.setRange(0, 1)

        y_axis = QValueAxis()
        y_axis.setTitleText("f2")
        y_axis.setRange(0, 1)

        chart.addAxis(x_axis, Qt.AlignmentFlag.AlignBottom)
        chart.addAxis(y_axis, Qt.AlignmentFlag.AlignLeft)
        self._apply_scientific_chart_theme(chart, (x_axis, y_axis))
        self.pareto_chart_view.setChart(chart)
        self._refresh_mcdm_details_panel(None)

    @Slot()
    def _refresh_convergence_chart(self) -> None:
        metric_name = self.metric_plot_combo.currentText().strip()
        if not metric_name or not self.results:
            self._set_empty_convergence_chart()
            return

        selected_problem_id = self._selected_plot_problem_id()
        selected_problem_label = self.problem_plot_combo.currentText().strip()

        chart = QChart()
        if selected_problem_id is None:
            chart.setTitle(f"Convergence - {metric_name}")
        else:
            chart.setTitle(f"Convergence - {metric_name} - {selected_problem_label}")
        chart.legend().setVisible(True)

        x_axis = QValueAxis()
        x_axis.setTitleText("Function Evaluations (FE)")
        y_axis = QValueAxis()
        y_axis.setTitleText(metric_name)

        chart.addAxis(x_axis, Qt.AlignmentFlag.AlignBottom)
        chart.addAxis(y_axis, Qt.AlignmentFlag.AlignLeft)

        x_min, x_max = float("inf"), float("-inf")
        y_min, y_max = float("inf"), float("-inf")
        chart_palette = AppStyles.chart_palette()

        any_selected_runs = False
        any_history_found = False
        for algo_index, (algo_name, runs) in enumerate(self.results.items()):
            deduped_runs: list[dict[str, Any]] = []
            seen_keys: set[str] = set()
            for run_payload in runs:
                if not isinstance(run_payload, dict):
                    continue
                normalized = self._ensure_run_dimensions(dict(run_payload))
                run_key = self._analysis_run_key(normalized)
                if run_key in seen_keys:
                    continue
                seen_keys.add(run_key)
                deduped_runs.append(normalized)
            runs = deduped_runs

            if selected_problem_id is not None:
                runs = [run for run in runs if str(run.get("problem_id", "")) == selected_problem_id]
            if runs:
                any_selected_runs = True

            histories: list[np.ndarray] = []
            generations: list[np.ndarray] = []

            for run in runs:
                values = np.asarray(run.get("history", {}).get(metric_name, []), dtype=float)
                x_hist = np.asarray(run.get("x_history", run.get("generations", [])), dtype=float)

                if values.size == 0:
                    continue
                if x_hist.size != values.size:
                    x_hist = np.arange(1, values.size + 1, dtype=float)

                histories.append(values)
                generations.append(x_hist)
                any_history_found = True

            if not histories:
                continue

            min_len = min(series.size for series in histories)
            if min_len <= 0:
                continue

            matrix = np.vstack([series[:min_len] for series in histories])
            if np.all(np.isnan(matrix)):
                continue
            x_values = generations[0][:min_len]
            with np.errstate(invalid="ignore"):
                mean_values = np.nanmean(matrix, axis=0)

            series = QLineSeries()
            series.setName(algo_name)
            for x_val, y_val in zip(x_values, mean_values, strict=False):
                if math.isfinite(float(y_val)):
                    series.append(float(x_val), float(y_val))
                    x_min = min(x_min, float(x_val))
                    x_max = max(x_max, float(x_val))
                    y_min = min(y_min, float(y_val))
                    y_max = max(y_max, float(y_val))

            self._set_chart_series_color(
                series,
                chart_palette[algo_index % len(chart_palette)],
                width=2.0,
            )
            chart.addSeries(series)
            series.attachAxis(x_axis)
            series.attachAxis(y_axis)

        if not math.isfinite(x_min) or not math.isfinite(y_min):
            if any_selected_runs and not any_history_found:
                self._set_empty_convergence_chart(
                    "Convergence - history unavailable for raw-first experiment payloads"
                )
            else:
                self._set_empty_convergence_chart()
            try:
                self.conv_chart_view.setToolTip(
                    "The selected runs do not contain per-generation metric history. "
                    "This is expected for Experiment raw-first results (__exp_raw_first_metrics__)."
                    if any_selected_runs and not any_history_found
                    else ""
                )
            except Exception:  # noqa: BLE001
                pass
            return

        if x_min == x_max:
            x_max = x_min + 1.0
        if y_min == y_max:
            y_max = y_min + 1.0

        x_axis.setRange(x_min, x_max)
        margin = (y_max - y_min) * 0.1
        y_axis.setRange(y_min - margin, y_max + margin)

        self._apply_scientific_chart_theme(chart, (x_axis, y_axis))
        self.conv_chart_view.setChart(chart)

    @Slot()
    def _refresh_pareto_chart(self) -> None:
        algo_name = self.algo_plot_combo.currentText().strip()
        runs = self._runs_for_plot(algo_name)
        run_index = self.run_plot_spin.value() - 1

        if not runs or run_index < 0 or run_index >= len(runs):
            self._set_empty_pareto_chart()
            return

        run_payload = runs[run_index]
        self._refresh_mcdm_details_panel(run_payload)
        front = np.asarray(run_payload.get("final_front", []), dtype=float)

        if front.ndim != 2 or front.size == 0 or front.shape[1] < 2:
            self._set_empty_pareto_chart()
            return
        try:
            n_obj = front.shape[1]
            anchor = self.test_anchor_origin.isChecked()
            run_timestamp = str(run_payload.get("timestamp_en_us", "")).strip()
            if not run_timestamp:
                run_timestamp = format_timestamp_en_us()

            self.pareto_chart_stack.setCurrentIndex(0)
            chart = QChart()
            problem_label = str(run_payload.get("problem", "problem"))
            chart.legend().setVisible(True)

            problem_pf = self._resolve_problem_pf(run_payload, use_test_context=False)

            if n_obj == 2:
                # -- m=2: Scatter 2D --
                chart.setTitle(f"Pareto front (f1xf2) - {problem_label} - {algo_name} trial {run_payload.get('run_index', '-')} | {run_timestamp}")
                x_axis = QValueAxis()
                x_axis.setTitleText("f1")
                y_axis = QValueAxis()
                y_axis.setTitleText("f2")
                chart.addAxis(x_axis, Qt.AlignmentFlag.AlignBottom)
                chart.addAxis(y_axis, Qt.AlignmentFlag.AlignLeft)

                run_series = QScatterSeries()
                run_series.setName("Obtained")
                run_series.setMarkerSize(7.0)
                for x_val, y_val in front:
                    run_series.append(float(x_val), float(y_val))
                self._set_chart_series_color(run_series, AppStyles.chart_palette()[0])
                chart.addSeries(run_series)
                run_series.attachAxis(x_axis)
                run_series.attachAxis(y_axis)

                x_values = [float(np.min(front[:, 0])), float(np.max(front[:, 0]))]
                y_values = [float(np.min(front[:, 1])), float(np.max(front[:, 1]))]

                mcdm_decision = self._mcdm_decision_for_run(run_payload)
                if isinstance(mcdm_decision, dict):
                    selected_point = np.asarray(mcdm_decision.get("selected_point", []), dtype=float).reshape(-1)
                    if selected_point.size >= 2 and np.all(np.isfinite(selected_point[:2])):
                        mcdm_series = QScatterSeries()
                        mcdm_series.setName("MCDM selected")
                        mcdm_series.setMarkerSize(11.0)
                        try:
                            mcdm_series.setColor(QColor(220, 38, 38))
                        except Exception:  # noqa: BLE001
                            pass
                        mcdm_series.append(float(selected_point[0]), float(selected_point[1]))
                        chart.addSeries(mcdm_series)
                        mcdm_series.attachAxis(x_axis)
                        mcdm_series.attachAxis(y_axis)
                        x_values.append(float(selected_point[0]))
                        y_values.append(float(selected_point[1]))

                if problem_pf is not None and problem_pf.ndim == 2 and problem_pf.shape[1] >= 2:
                    pf2d = problem_pf[:, :2]
                    pf2d = pf2d[np.argsort(pf2d[:, 0])]
                    pf_series = QLineSeries()
                    pf_series.setName("Reference PF")
                    for x_val, y_val in pf2d:
                        pf_series.append(float(x_val), float(y_val))
                    self._set_chart_series_color(
                        pf_series,
                        StylesAppStyles.colors.chart_reference_front,
                        width=2.0,
                    )
                    chart.addSeries(pf_series)
                    pf_series.attachAxis(x_axis)
                    pf_series.attachAxis(y_axis)
                    x_values.extend([float(np.min(pf2d[:, 0])), float(np.max(pf2d[:, 0]))])
                    y_values.extend([float(np.min(pf2d[:, 1])), float(np.max(pf2d[:, 1]))])

                if anchor:
                    x_values.append(0.0)
                    y_values.append(0.0)

                x_min, x_max = min(x_values), max(x_values)
                y_min, y_max = min(y_values), max(y_values)
                if x_min == x_max:
                    x_max = x_min + 1.0
                if y_min == y_max:
                    y_max = y_min + 1.0
                x_margin = (x_max - x_min) * 0.05
                y_margin = (y_max - y_min) * 0.05
                x_lower = 0.0 if anchor else (x_min - x_margin)
                y_lower = 0.0 if anchor else (y_min - y_margin)
                x_axis.setRange(x_lower, x_max + x_margin)
                y_axis.setRange(y_lower, y_max + y_margin)

            elif n_obj == 3 and _HAS_MPL_3D:
                # -- m=3: Scatter 3D interativo (matplotlib) --
                title_3d = (
                    f"Pareto 3D (f1 x f2 x f3) - {problem_label} - "
                    f"{algo_name} trial {run_payload.get('run_index', '-')} | {run_timestamp}"
                )
                selected_point_3d = None
                mcdm_decision = self._mcdm_decision_for_run(run_payload)
                if isinstance(mcdm_decision, dict):
                    try:
                        point = np.asarray(mcdm_decision.get("selected_point", []), dtype=float).reshape(-1)
                        if point.size >= 3 and np.all(np.isfinite(point[:3])):
                            selected_point_3d = point[:3]
                    except Exception:  # noqa: BLE001
                        selected_point_3d = None
                self._show_3d_scatter(
                    front[:, :3], problem_pf, title_3d, anchor,
                    self.pareto_3d_container, self.pareto_3d_layout, self.pareto_chart_stack,
                    selected_point=selected_point_3d,
                    selected_label="MCDM selected",
                )
                return

            else:
                # -- m>=4 (and m=3 fallback without mpl 3D): Parallel Coordinates --
                chart.setTitle(
                    f"Parallel Coordinates ({n_obj} obj) - {problem_label} - "
                    f"{algo_name} trial {run_payload.get('run_index', '-')} | {run_timestamp}"
                )
                x_axis = QValueAxis()
                x_axis.setTitleText("Dimension No.")
                x_axis.setRange(0.5, n_obj + 0.5)
                x_axis.setTickCount(n_obj)
                x_axis.setLabelFormat("%d")
                y_axis = QValueAxis()
                y_axis.setTitleText("Value")
                chart.addAxis(x_axis, Qt.AlignmentFlag.AlignBottom)
                chart.addAxis(y_axis, Qt.AlignmentFlag.AlignLeft)

                y_min_val = float(np.min(front))
                y_max_val = float(np.max(front))

                if problem_pf is not None and problem_pf.ndim == 2 and problem_pf.shape[1] == n_obj:
                    pf_color = QColor(StylesAppStyles.colors.chart_reference_front)
                    pf_color.setAlpha(72)
                    pf_pen = QPen(pf_color, 1.0)
                    for row in problem_pf:
                        s = QLineSeries()
                        s.setPen(pf_pen)
                        for dim_idx in range(n_obj):
                            s.append(float(dim_idx + 1), float(row[dim_idx]))
                        chart.addSeries(s)
                        s.attachAxis(x_axis)
                        s.attachAxis(y_axis)
                    y_min_val = min(y_min_val, float(np.min(problem_pf)))
                    y_max_val = max(y_max_val, float(np.max(problem_pf)))
                    pf_legend = QLineSeries()
                    pf_legend.setName("Reference PF")
                    pf_legend.setPen(QPen(QColor(StylesAppStyles.colors.chart_reference_front), 2.0))
                    pf_legend.append(0, 0)
                    chart.addSeries(pf_legend)
                    pf_legend.attachAxis(x_axis)
                    pf_legend.attachAxis(y_axis)

                obtained_color = QColor(AppStyles.chart_palette()[0])
                obtained_color.setAlpha(160)
                obtained_pen = QPen(obtained_color, 1.5)
                for row in front:
                    s = QLineSeries()
                    s.setPen(obtained_pen)
                    for dim_idx in range(n_obj):
                        s.append(float(dim_idx + 1), float(row[dim_idx]))
                    chart.addSeries(s)
                    s.attachAxis(x_axis)
                    s.attachAxis(y_axis)
                obt_legend = QLineSeries()
                obt_legend.setName("Obtained")
                obt_legend.setPen(QPen(QColor(AppStyles.chart_palette()[0]), 2.5))
                obt_legend.append(0, 0)
                chart.addSeries(obt_legend)
                obt_legend.attachAxis(x_axis)
                obt_legend.attachAxis(y_axis)

                for s in chart.series():
                    if not s.name():
                        markers = chart.legend().markers(s)
                        if markers:
                            markers[0].setVisible(False)

                if anchor:
                    y_min_val = min(y_min_val, 0.0)

                if y_min_val == y_max_val:
                    y_max_val = y_min_val + 1.0
                y_margin = (y_max_val - y_min_val) * 0.05
                y_lower = 0.0 if anchor else (y_min_val - y_margin)
                y_axis.setRange(y_lower, y_max_val + y_margin)

            self._apply_scientific_chart_theme(chart, (x_axis, y_axis))
            self.pareto_chart_view.setChart(chart)
        except Exception as exc:  # noqa: BLE001
            self._set_empty_pareto_chart()
            msg = f"Pareto chart refresh failed: {exc}"
            try:
                self._append_log(f"Warning: {msg}")
            except Exception:  # noqa: BLE001
                pass
            try:
                self.pareto_chart_view.setToolTip(msg)
            except Exception:  # noqa: BLE001
                pass

    def _sync_payload_to_analysis_workspace(self, payload: dict[str, Any]) -> int:
        """Merge a payload into the Analysis/MCDM workspace without forcing a tab switch."""
        if not isinstance(payload, dict):
            return 0

        if not hasattr(self, "analysis_run_keys") or self.analysis_run_keys is None:
            self.analysis_run_keys = set()
        before_count = len(self.analysis_run_keys)

        self._accumulate_payload(payload)

        if not hasattr(self, "analysis_run_keys") or self.analysis_run_keys is None:
            self.analysis_run_keys = set()
        added_runs = max(0, len(self.analysis_run_keys) - before_count)

        latest_run = self._latest_run_from_payload(payload)
        self._refresh_plot_controls()
        if latest_run is not None:
            self._focus_analysis_run(latest_run)
        else:
            self._refresh_convergence_chart()
            self._refresh_pareto_chart()
        return added_runs

    def _accumulate_payload(self, payload: dict[str, Any]) -> None:
        """Accumulate results, metrics, and Pareto fronts in the global Analysis state."""
        new_results = payload.get("results", {})
        new_metrics = payload.get("metrics", [])
        backend_label = str(payload.get("execution_backend_label", "CPU"))
        profile_enabled = bool(payload.get("profile_compare_enabled", False))

        if not hasattr(self, "results") or self.results is None:
            self.results = {}
        if not hasattr(self, "analysis_run_keys") or self.analysis_run_keys is None:
            self.analysis_run_keys = set()
        if not self.analysis_run_keys and self.results:
            self._rebuild_analysis_run_index()
        if not hasattr(self, "metric_names") or self.metric_names is None:
            self.metric_names = []

        # 1. Merge results
        for algo, runs in new_results.items():
            algo_name = str(algo)
            for r in runs:
                if isinstance(r, dict):
                    self._append_analysis_run(dict(r), algo_hint=algo_name)

        # 2. Merge metrics
        curr_metrics = set(self.metric_names)
        for m in new_metrics:
            if str(m) not in curr_metrics:
                self.metric_names.append(str(m))

        # 3. Update global flags
        self.execution_backend_label = backend_label
        self.profile_compare_enabled = getattr(self, "profile_compare_enabled", False) or profile_enabled

        # 4. Merge Pareto fronts
        if not hasattr(self, "pareto_fronts") or self.pareto_fronts is None:
            self.pareto_fronts = {}

        pf_map_raw = payload.get("pareto_fronts")
        if isinstance(pf_map_raw, dict):
            for prob_id, pf_raw in pf_map_raw.items():
                key = str(prob_id)
                if pf_raw is None:
                    self.pareto_fronts[key] = None
                else:
                    self.pareto_fronts[key] = np.asarray(pf_raw, dtype=float)

        pf_main = payload.get("pareto_front")
        if pf_main is not None:
            self.pareto_front = np.asarray(pf_main, dtype=float)

    def _ensure_run_dimensions(self, run_payload: dict[str, Any]) -> dict[str, Any]:
        """Ensure run payload has n_obj (M) and n_var (D) when possible."""
        run = dict(run_payload)

        def _as_pos_int(value: Any) -> int | None:
            try:
                parsed = int(value)
            except Exception:  # noqa: BLE001
                return None
            return parsed if parsed > 0 else None

        n_obj = _as_pos_int(run.get("n_obj"))
        n_var = _as_pos_int(run.get("n_var"))

        if n_obj is None:
            front = np.asarray(run.get("final_front", []), dtype=float)
            if front.ndim == 2 and front.shape[1] > 0:
                n_obj = int(front.shape[1])

        problem_id = str(run.get("problem_id", ""))
        spec = self.problem_specs.get(problem_id) if problem_id else None
        if spec is not None:
            if n_obj is None:
                n_obj = max(1, int(spec.default_n_obj))
            if n_var is None:
                n_var = max(1, int(spec.default_n_var))

        if n_obj is not None:
            run["n_obj"] = int(n_obj)
        if n_var is not None:
            run["n_var"] = int(n_var)

        ts_epoch = _float_or_none(run.get("timestamp_epoch"))
        if ts_epoch is None or not math.isfinite(ts_epoch):
            ts_epoch = self._run_timestamp_epoch(run)
        if ts_epoch is None or not math.isfinite(ts_epoch):
            ts_epoch = float(datetime.now().astimezone().timestamp())
        run["timestamp_epoch"] = float(ts_epoch)

        ts_iso = str(run.get("timestamp_iso", "")).strip()
        if not ts_iso:
            run["timestamp_iso"] = datetime.now().astimezone().isoformat()

        ts_en_us = str(run.get("timestamp_en_us", "")).strip()
        if not ts_en_us:
            run["timestamp_en_us"] = format_timestamp_en_us()

        return run

    def _load_results_history_impl(self, *, silent: bool) -> int:
        """Load persisted history into Analysis state. Returns number of loaded runs."""
        base_root = getattr(self, "base_dir", Path(__file__).resolve().parent)
        results_file = base_root / "test_module_results" / "results.json"
        if not results_file.exists():
            if not silent:
                QMessageBox.information(self, "Load History", f"No history file found at {results_file}.")
            return 0
        try:
            content = results_file.read_text(encoding="utf-8")
            if not content.strip():
                if not silent:
                    QMessageBox.information(self, "Load History", "History file is empty.")
                return 0

            history = json.loads(content)
            if not isinstance(history, list):
                if not silent:
                    QMessageBox.warning(self, "Load History", "Invalid history file format (expected list).")
                return 0

            loaded_runs = 0
            if not hasattr(self, "results") or self.results is None:
                self.results = {}
            if not hasattr(self, "analysis_run_keys") or self.analysis_run_keys is None:
                self.analysis_run_keys = set()
            if not self.analysis_run_keys and self.results:
                self._rebuild_analysis_run_index()
            merged_metrics_set = set(self.metric_names if getattr(self, "metric_names", None) is not None else [])

            for entry in history:
                if not isinstance(entry, dict):
                    continue
                payload = entry.get("payload")
                if not isinstance(payload, dict):
                    continue
                entry_ts_iso = str(entry.get("timestamp_iso", entry.get("timestamp", ""))).strip()
                entry_ts_en_us = str(entry.get("timestamp_en_us", "")).strip()
                entry_ts_epoch = _float_or_none(entry.get("timestamp_epoch"))

                metrics = payload.get("metrics", [])
                if isinstance(metrics, (list, tuple, set)):
                    for metric_name in metrics:
                        merged_metrics_set.add(str(metric_name))

                sub_results = payload.get("results", {})
                if isinstance(sub_results, dict):
                    for algo, runs in sub_results.items():
                        if not isinstance(runs, list):
                            continue
                        for run_payload in runs:
                            if isinstance(run_payload, dict):
                                normalized_run = dict(run_payload)
                                if not str(normalized_run.get("timestamp_iso", "")).strip() and entry_ts_iso:
                                    normalized_run["timestamp_iso"] = entry_ts_iso
                                if not str(normalized_run.get("timestamp_en_us", "")).strip() and entry_ts_en_us:
                                    normalized_run["timestamp_en_us"] = entry_ts_en_us
                                run_ts_epoch = _float_or_none(normalized_run.get("timestamp_epoch"))
                                if run_ts_epoch is None and entry_ts_epoch is not None:
                                    normalized_run["timestamp_epoch"] = float(entry_ts_epoch)
                                if self._append_analysis_run(normalized_run, algo_hint=str(algo)):
                                    loaded_runs += 1

            if loaded_runs == 0:
                if not silent:
                    QMessageBox.information(self, "Load History", "No new valid runs found in history.")
                return 0

            self.metric_names = list(merged_metrics_set)

            # Refresh Analysis controls/charts with loaded data.
            self._refresh_plot_controls()
            self._refresh_results_table()
            latest_run = self._latest_run_from_results(self.results)
            if latest_run is not None:
                self._focus_analysis_run(latest_run)
            else:
                self._refresh_convergence_chart()
                self._refresh_pareto_chart()
            return loaded_runs
        except Exception as exc:  # noqa: BLE001
            if not silent:
                QMessageBox.critical(self, "Load History", f"Failed to load history:\n{exc}")
            else:
                self._append_log(f"History auto-load skipped: {exc}")
            return 0

    def _auto_load_results_history_if_available(self) -> None:
        loaded_runs = self._load_results_history_impl(silent=True)
        if loaded_runs > 0:
            self._append_log(f"History auto-loaded: {loaded_runs} run(s).")

    @Slot()
    def _load_results_history(self) -> None:
        """Load test_module_results/results.json history and merge it into the Analysis tab."""
        loaded_runs = self._load_results_history_impl(silent=False)
        if loaded_runs > 0:
            QMessageBox.information(self, "Load History", f"Successfully loaded {loaded_runs} run(s) from history.")

    @Slot()
    def _clear_results_history(self) -> None:
        """Delete test_module_results/results.json after user confirmation."""
        base_root = getattr(self, "base_dir", Path(__file__).resolve().parent)
        results_file = base_root / "test_module_results" / "results.json"
        if not results_file.exists():
            QMessageBox.information(self, "Clear History", "No history file found.")
            return
        ans = QMessageBox.question(
            self,
            "Clear History",
            "Are you sure you want to permanently delete the results history file-",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )

        if ans == QMessageBox.StandardButton.Yes:
            try:
                # 1. Delete the file on disk
                if results_file.exists():
                    results_file.unlink()

                # 2. Reset in-memory state (current Analysis tab state)
                self.results = {}
                self.analysis_run_keys = set()
                self.analysis_group_expanded = set()
                self.analysis_group_user_touched = False
                self.analysis_row_meta = {}
                self.metric_names = []
                self.pareto_front = None
                self.pareto_fronts = {}
                self._last_stat_result = None

                # 3. Refresh the UI to reflect the cleared state
                self._populate_analysis_combos()
                self._refresh_results_table()
                self._refresh_convergence_chart()
                self._refresh_pareto_chart()

                QMessageBox.information(self, "Clear History", "History and active analysis results cleared successfully.")
            except Exception as e:
                QMessageBox.critical(self, "Clear History", f"Failed to clear history:\n{e}")

    def _current_stat_export_payload(self) -> dict[str, Any] | None:
        stat_mode = str(self.stat_method_combo.currentData()) if hasattr(self, "stat_method_combo") else "none"
        if stat_mode in {"none", ""}:
            return None

        metric_name = self.metric_plot_combo.currentText().strip()
        payload = self._last_stat_result if isinstance(self._last_stat_result, dict) else None
        if payload is not None:
            if str(payload.get("method", "")).strip().lower() == stat_mode and str(
                payload.get("metric_name", "")
            ).strip() == metric_name:
                return dict(payload)

        self._refresh_statistical_results_table(stat_mode)
        if isinstance(self._last_stat_result, dict):
            return dict(self._last_stat_result)
        return None

    @staticmethod
    def _latex_escape(text: Any) -> str:
        raw = str(text)
        replacements = {
            "\\": "\\textbackslash{}",
            "&": "\\&",
            "%": "\\%",
            "$": "\\$",
            "#": "\\#",
            "_": "\\_",
            "{": "\\{",
            "}": "\\}",
            "~": "\\textasciitilde{}",
            "^": "\\textasciicircum{}",
        }
        escaped = raw
        for old, new in replacements.items():
            escaped = escaped.replace(old, new)
        return escaped

    def _export_csv(self) -> None:
        all_runs = self._all_analysis_runs()
        if not all_runs:
            QMessageBox.information(self, "Export", "No results available to export.")
            return

        start_dir = getattr(self, "base_dir", Path(__file__).resolve().parent)
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save CSV results",
            str(start_dir / "emopylab_results.csv"),
            "CSV (*.csv)",
        )
        if not file_path:
            return

        has_profile_columns = self.profile_compare_enabled or any(
            run.get("profile_cpu_time_s") is not None
            or run.get("profile_gpu_time_s") is not None
            or run.get("profile_speedup_gpu_vs_cpu") is not None
            for run in all_runs
        )

        profile_headers = ["profile_cpu_time_s", "profile_gpu_time_s", "profile_speedup_gpu_vs_cpu"] if has_profile_columns else []
        headers = [
            "algorithm",
            "problem",
            "problem_id",
            "run",
            "timestamp_en_us",
            "seed",
            "backend",
            "time_s",
            "evaluations",
            "n_obj",
            "n_var",
        ] + profile_headers + self.metric_names
        stat_payload = self._current_stat_export_payload()
        with open(file_path, "w", newline="", encoding="utf-8") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(headers)
            ordered_runs = sorted(all_runs, key=lambda run: self._run_timestamp_epoch(run), reverse=True)
            for run_payload in ordered_runs:
                row = [
                    self._algorithm_label_for_run(run_payload),
                    self._problem_label_for_run(run_payload),
                    run_payload.get("problem_id", ""),
                    run_payload.get("run_index", ""),
                    run_payload.get("timestamp_en_us", ""),
                    run_payload.get("seed", ""),
                    run_payload.get("backend", self.execution_backend_label),
                    run_payload.get("time_s", ""),
                    run_payload.get("evaluations", ""),
                    run_payload.get("n_obj", ""),
                    run_payload.get("n_var", ""),
                ]

                if has_profile_columns:
                    row.extend(
                        [
                            run_payload.get("profile_cpu_time_s", ""),
                            run_payload.get("profile_gpu_time_s", ""),
                            run_payload.get("profile_speedup_gpu_vs_cpu", ""),
                        ]
                    )

                row.extend(run_payload.get("metrics", {}).get(metric_name, float("nan")) for metric_name in self.metric_names)
                writer.writerow(row)

            if stat_payload is not None:
                writer.writerow([])
                writer.writerow(["stat_header", stat_payload.get("header_text", "")])
                writer.writerow(
                    [
                        "method",
                        stat_payload.get("method", ""),
                        "metric",
                        stat_payload.get("metric_name", ""),
                        "reference",
                        stat_payload.get("reference_algorithm", ""),
                        "alpha",
                        stat_payload.get("alpha", ""),
                        "backend",
                        stat_payload.get("backend_label", ""),
                    ]
                )
                note = str(stat_payload.get("note", "")).strip()
                if note:
                    writer.writerow(["note", note])

                columns = list(stat_payload.get("columns", []))
                if columns:
                    writer.writerow(columns)
                for row_data in stat_payload.get("rows", []):
                    effect_text = str(row_data.get("effect_text", "--"))
                    if str(stat_payload.get("method", "")).strip().lower() == "friedman" and effect_text != "--":
                        effect_text = f"rank={effect_text}"
                    writer.writerow(
                        [
                            row_data.get("algorithm", ""),
                            row_data.get("n", 0),
                            row_data.get("mean_std_text", "--"),
                            row_data.get("p_value_text", "--"),
                            effect_text,
                            row_data.get("decision", "-"),
                        ]
                    )

        self._append_log(f"CSV exported: {file_path}")
        QMessageBox.information(self, "Export", f"CSV saved to:\n{file_path}")

    @Slot()
    def _export_latex(self) -> None:
        all_runs = self._all_analysis_runs()
        if not all_runs:
            QMessageBox.information(self, "Export", "No results available to export.")
            return

        start_dir = getattr(self, "base_dir", Path(__file__).resolve().parent)
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save LaTeX table",
            str(start_dir / "emopylab_table.tex"),
            "TeX (*.tex)",
        )
        if not file_path:
            return
        stat_mode = str(self.stat_method_combo.currentData()) if hasattr(self, "stat_method_combo") else "none"
        stat_payload = self._current_stat_export_payload()
        stat_desc = {
            "none": "Detailed runs summary",
            "wilcoxon": "Wilcoxon vs reference",
            "friedman": "Friedman rank",
        }.get(stat_mode, "Detailed runs summary")

        if stat_payload is not None:
            stat_columns = list(stat_payload.get("columns", []))
            align = "l" + ("c" * max(0, len(stat_columns) - 1))
            lines = [
                "% EmoPyLab - Statistical Test Style Header",
                "% Table generated by EmoPyLab",
                f"% Effective backend: {self.execution_backend_label}",
                f"% Statistical view: {stat_desc}",
                f"% Metric: {stat_payload.get('metric_name', '')}",
                "\\begin{tabular}{" + align + "}",
                "\\hline",
                f"\\multicolumn{{{len(stat_columns)}}}{{c}}{{\\textbf{{{self._latex_escape(stat_payload.get('header_text', stat_desc))}}}}} \\\\",
                "\\hline",
                " & ".join(self._latex_escape(col) for col in stat_columns) + " \\\\",
                "\\hline",
            ]
            for row_data in stat_payload.get("rows", []):
                effect_text = str(row_data.get("effect_text", "--"))
                if str(stat_payload.get("method", "")).strip().lower() == "friedman" and effect_text != "--":
                    effect_text = f"rank={effect_text}"
                cells = [
                    self._latex_escape(row_data.get("algorithm", "")),
                    self._latex_escape(row_data.get("n", 0)),
                    self._latex_escape(row_data.get("mean_std_text", "--")),
                    self._latex_escape(row_data.get("p_value_text", "--")),
                    self._latex_escape(effect_text),
                    self._latex_escape(row_data.get("decision", "-")),
                ]
                lines.append(" & ".join(cells) + " \\\\")
            note = str(stat_payload.get("note", "")).strip()
            if note:
                lines.append("\\hline")
                lines.append(
                    f"\\multicolumn{{{len(stat_columns)}}}{{l}}{{\\footnotesize {self._latex_escape(note)}}} \\\\"
                )
            lines += ["\\hline", "\\end{tabular}", ""]
        else:
            align = "ll" + ("c" * len(self.metric_names))
            lines = [
                "% EmoPyLab - Statistical Test Style Header",
                "% Table generated by EmoPyLab",
                f"% Effective backend: {self.execution_backend_label}",
                f"% Statistical view: {stat_desc}",
                "\\begin{tabular}{" + align + "}",
                "\\hline",
                f"\\multicolumn{{{2 + len(self.metric_names)}}}{{c}}{{\\textbf{{EmoPyLab - {self._latex_escape(stat_desc)}}}}} \\\\",
                "\\hline",
                "Algorithm & Problem & " + " & ".join(self._latex_escape(m) for m in self.metric_names) + " \\\\",
                "\\hline",
            ]

            grouped_runs: dict[tuple[str, str], list[dict[str, Any]]] = {}
            for run_payload in all_runs:
                grouped_runs.setdefault(self._analysis_group_key(run_payload), []).append(run_payload)

            ordered_groups = sorted(
                grouped_runs.items(),
                key=lambda item: max(self._run_timestamp_epoch(run) for run in item[1]),
                reverse=True,
            )

            for _, runs_in_group in ordered_groups:
                problem_runs = sorted(runs_in_group, key=lambda run: self._run_timestamp_epoch(run), reverse=True)
                if not problem_runs:
                    continue
                latest_problem_run = problem_runs[0]
                algo_name = self._algorithm_label_for_run(latest_problem_run)
                problem_label = self._problem_label_for_run(latest_problem_run)

                metric_cells: list[str] = []
                for metric_name in self.metric_names:
                    values = [_float_or_nan(item.get("metrics", {}).get(metric_name)) for item in problem_runs]
                    mean, std = _mean_std(values)
                    if math.isfinite(mean):
                        metric_cells.append(f"${mean:.4g}\\pm{std:.2g}$")
                    else:
                        metric_cells.append("--")
                lines.append(
                    self._latex_escape(algo_name)
                    + " & "
                    + self._latex_escape(problem_label)
                    + " & "
                    + " & ".join(metric_cells)
                    + " \\\\"
                )

            lines += ["\\hline", "\\end{tabular}", ""]

        with open(file_path, "w", encoding="utf-8") as tex_file:
            tex_file.write("\n".join(lines))

        self._append_log(f"LaTeX exported: {file_path}")
        QMessageBox.information(self, "Export", f"LaTeX table saved to:\n{file_path}")
    @Slot()
    def _save_config_to_file(self) -> None:
        config = self._collect_test_config()
        if config is None:
            return

        start_dir = getattr(self, "base_dir", Path(__file__).resolve().parent)
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save configuration",
            str(start_dir / "emopylab_config.json"),
            "JSON (*.json)",
        )
        if not file_path:
            return
        with open(file_path, "w", encoding="utf-8") as handle:
            json.dump(config, handle, indent=2, ensure_ascii=False)

        self._append_log(f"Configuration saved: {file_path}")

    @Slot()
    def _load_config_from_file(self) -> None:
        start_dir = getattr(self, "base_dir", Path(__file__).resolve().parent)
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Load configuration",
            str(start_dir),
            "JSON (*.json)",
        )
        if not file_path:
            return
        try:
            with open(file_path, "r", encoding="utf-8") as handle:
                config = json.load(handle)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Error", f"Failed to load configuration:\n{exc}")
            return

        self._apply_loaded_config(config)
        self._append_log(f"Configuration loaded: {file_path}")

    @Slot()
    def _save_experiment_config_to_file(self) -> None:
        """Save experiment config to a user-chosen JSON file."""
        config = self._collect_experiment_config()
        if config is None:
            return

        start_dir = getattr(self, "base_dir", Path(__file__).resolve().parent)
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save experiment configuration",
            str(start_dir / "emopylab_experiment_config.json"),
            "JSON (*.json)",
        )
        if not file_path:
            return
        with open(file_path, "w", encoding="utf-8") as handle:
            json.dump(config, handle, indent=2, ensure_ascii=False)

        self._exp_append_log(f"Experiment config saved: {file_path}")
        self._append_log(f"Experiment config saved: {file_path}")

    @Slot()
    def _load_experiment_config_from_file(self) -> None:
        """Load experiment config from a user-chosen JSON file."""
        start_dir = getattr(self, "base_dir", Path(__file__).resolve().parent)
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Load experiment configuration",
            str(start_dir),
            "JSON (*.json)",
        )
        if not file_path:
            return
        try:
            with open(file_path, "r", encoding="utf-8") as handle:
                config = json.load(handle)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Error", f"Failed to load experiment config:\n{exc}")
            return

        self._apply_experiment_config(config)
        self._exp_append_log(f"Experiment config loaded: {file_path}")
        self._append_log(f"Experiment config loaded: {file_path}")
        # Switch to experiment tab to show the loaded config
        self.tabs.setCurrentWidget(self.experiment_tab)

    def _apply_loaded_config(self, config: dict[str, Any]) -> None:
        problem_value = str(config.get("problem_id", config.get("problem", "")))
        index = self.problem_combo.findData(problem_value)

        if index < 0:
            problem_name = problem_value.lower()
            for i in range(self.problem_combo.count()):
                pid = self.problem_combo.itemData(i)
                spec = self.problem_specs.get(pid)
                if spec is not None and spec.name.lower() == problem_name:
                    index = i
                    break

        if index >= 0:
            self.problem_combo.setCurrentIndex(index)

        problem_values_raw = config.get("problem_ids")
        if isinstance(problem_values_raw, (list, tuple)):
            problem_values = [str(v) for v in problem_values_raw]
        else:
            problem_values = [problem_value]
        problem_ids = {str(v) for v in problem_values}
        problem_names = {str(v).lower() for v in problem_values}
        self._set_checked_ids(self.experiment_problem_list, problem_ids, problem_names, self.problem_specs)
        if not self._checked_ids(self.experiment_problem_list):
            current_id = self.problem_combo.currentData()
            if isinstance(current_id, str):
                self._set_checked_ids(self.experiment_problem_list, {current_id}, set(), self.problem_specs)

        self.n_var_spin.setValue(int(config.get("n_var", self.n_var_spin.value())))
        self.n_obj_spin.setValue(int(config.get("n_obj", self.n_obj_spin.value())))
        self._on_test_n_obj_changed()
        self.pop_size_spin.setValue(int(config.get("pop_size", self.pop_size_spin.value())))

        max_fe_raw = config.get("max_fe", config.get("maxFE"))
        if max_fe_raw is None and "n_gen" in config:
            try:
                max_fe_raw = int(config.get("n_gen", 200)) * int(config.get("pop_size", self.pop_size_spin.value()))
            except Exception:  # noqa: BLE001
                max_fe_raw = None
        try:
            if max_fe_raw is not None:
                self.max_fe_spin.setValue(int(max_fe_raw))
        except Exception:  # noqa: BLE001
            pass

        self.n_runs_spin.setValue(1)
        test_seed_mode = str(config.get("seed_mode", SEED_MODE_RANDOM)).strip().lower()
        test_seed_mode_idx = self.seed_mode_combo.findData(test_seed_mode)
        if test_seed_mode_idx >= 0:
            self.seed_mode_combo.setCurrentIndex(test_seed_mode_idx)
        self.seed_spin.setValue(_normalize_seed_value(config.get("seed_base", 1), default=1))
        self._on_test_seed_mode_changed()

        algorithm_values = config.get("algorithm_ids", config.get("algorithms", []))
        selected_algorithm: str | None = None
        if isinstance(algorithm_values, (list, tuple)) and algorithm_values:
            selected_algorithm = str(algorithm_values[0])
        elif isinstance(algorithm_values, str):
            selected_algorithm = algorithm_values
        if selected_algorithm:
            selected_algorithm_lower = selected_algorithm.lower()
            for i in range(self.algorithm_list.count()):
                item = self.algorithm_list.item(i)
                item_id = item.data(Qt.ItemDataRole.UserRole)
                if isinstance(item_id, str) and item_id == selected_algorithm:
                    self.algorithm_list.setCurrentItem(item)
                    break
                if item.text().lower() == selected_algorithm_lower:
                    self.algorithm_list.setCurrentItem(item)
                    break

        metric_values = config.get("metric_ids", [])
        if not metric_values and "metrics" in config and isinstance(config.get("metrics"), dict):
            metric_values = [key for key, enabled in config.get("metrics", {}).items() if enabled]
        metric_ids = {str(v) for v in metric_values}
        metric_names = {str(v).lower() for v in metric_values}
        self._set_checked_ids(self.metric_list, metric_ids, metric_names, self.metric_specs)

        self.use_pf_check.setChecked(bool(config.get("use_pf", True)))

        backend = str(config.get("compute_backend", "cpu")).lower()
        if backend == "auto":
            backend = "cpu"
        backend_idx = self.compute_backend_combo.findData(backend)
        if backend_idx >= 0:
            self.compute_backend_combo.setCurrentIndex(backend_idx)

        gpu_dtype = str(config.get("gpu_dtype", "float64")).lower()
        dtype_idx = self.gpu_dtype_combo.findData(gpu_dtype)
        if dtype_idx >= 0:
            self.gpu_dtype_combo.setCurrentIndex(dtype_idx)

        self.profile_compare_check.setChecked(bool(config.get("profile_compare", False)))

        stat_method = str(config.get("stat_test_method", "none")).lower()
        stat_method_idx = self.stat_method_combo.findData(stat_method)
        if stat_method_idx >= 0:
            self.stat_method_combo.setCurrentIndex(stat_method_idx)

        stat_reference = str(config.get("stat_test_reference", "")).strip()
        if stat_reference and self.stat_reference_combo.findText(stat_reference) >= 0:
            self.stat_reference_combo.setCurrentText(stat_reference)
        stat_alpha = _float_or_none(config.get("stat_test_alpha"))
        if stat_alpha is not None and math.isfinite(stat_alpha):
            self.stat_alpha_spin.setValue(float(stat_alpha))
        self._update_stat_controls_state()

        # ref_point is now auto-computed; ignore saved value

        # Restore operator combo selections
        for combo, key in [
            (self.crossover_combo, "crossover"),
            (self.mutation_combo, "mutation"),
            (self.selection_combo, "selection"),
            (self.sampling_combo, "sampling"),
        ]:
            val = config.get(key)
            if isinstance(val, str):
                normalized = self._normalize_operator_value(key, val)
                idx = combo.findData(normalized)
                if idx < 0:
                    wanted = self._normalize_operator_class_token(
                        self._operator_class_name(key, normalized)
                    )
                    if wanted:
                        for i in range(combo.count()):
                            data_val = combo.itemData(i)
                            got = self._normalize_operator_class_token(
                                self._operator_class_name(key, data_val)
                            )
                            if got == wanted:
                                idx = i
                                break
                if idx >= 0:
                    combo.setCurrentIndex(idx)

        # Restore operator parameter values
        if "crossover_eta" in config:
            self.crossover_eta_spin.setValue(float(config["crossover_eta"]))
        if "crossover_prob" in config:
            self.crossover_prob_spin.setValue(float(config["crossover_prob"]))
        if "mutation_eta" in config:
            self.mutation_eta_spin.setValue(float(config["mutation_eta"]))
        if "mutation_prob" in config:
            val = config["mutation_prob"]
            self.mutation_prob_spin.setValue(float(val) if val is not None else 0.0)
        if "selection_pressure" in config:
            self.selection_pressure_spin.setValue(int(config["selection_pressure"]))

        # Trigger dynamic visibility for operator params
        self._on_crossover_changed()
        self._on_mutation_changed()
        self._on_selection_changed()

        self._apply_catalog_filters()
        self._update_selection_cards()
        self._refresh_test_result_controls()
        self._sync_ref_point_hint()
        self._on_backend_mode_changed()


def main() -> int:
    """
    Main function for EmoPyLab application.

    Follows scientific workstation guidelines:
    - Default Dark Mode (dark_blue.xml) with full Light Mode support (View menu)
    - Always maximized mode (showMaximized)
    - High-contrast accessible color palette via styles.py
    - Icons using MaterialIcon
    - UTF-8 without BOM
    """
    app = QApplication(sys.argv)

    # Apply default LIGHT theme (light_amber.xml) for clean, high-contrast academic publication aesthetics
    StylesAppStyles.set_theme_mode('light')
    apply_stylesheet(
        app,
        theme='light_amber.xml',
        invert_secondary=True,
        extra=StylesAppStyles.get_qt_material_theme(),
    )
    app.setStyleSheet(StylesAppStyles.get_stylesheet())
    # Show senior visual splash with real-time milestone feedback and local AI initialization
    from core.llm.splash import EmoPyLabStartupSplash
    splash = EmoPyLabStartupSplash(app, preferred_port=SPLASH_DEFAULT_MLX_PORT, base_url=SPLASH_DEFAULT_BASE_URL)
    splash.show()
    splash.start_background_llm_probe()

    splash.update_progress(
        20,
        "Initializing Tensor Runtime & Hardware Acceleration...",
        "Detecting PyTorch CUDA/MPS, JAX, Apple MLX, and CPU SIMD",
        step_text="Phase 1/4",
    )

    splash.update_progress(
        45,
        "Loading Local AI Engine (Qwen2.5-0.5B GGUF / MLX)...",
        "Loading local offline Qwen2.5 (models/) without blocking GUI",
        step_text="Phase 2/4",
    )

    splash.update_progress(
        70,
        "Registering 298+ Metaheuristics & Analytical Suites...",
        "Registering NSGA-II/III, MOEA/D, RVEA, SparseEA, MaACO & benchmarks",
        step_text="Phase 3/4",
    )
    window = EmoPyLabMainWindow()
    window.configure_local_llm_session(
        enabled=True,
        base_url=SPLASH_DEFAULT_BASE_URL,
        port=SPLASH_DEFAULT_MLX_PORT,
    )

    splash.update_progress(
        100,
        "Workspace Ready.",
        "Scientific Workspace Initialized — Opening Workstation…",
        step_text="Phase 4/4",
    )
    splash.complete()

    window.setWindowState(Qt.WindowState.WindowMaximized)
    window.showMaximized()
    QTimer.singleShot(50, window.showMaximized)
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
