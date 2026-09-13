# Author: Professor Thiago Santos at UFOP, Brazil
# -*- coding: utf-8 -*-
"""
Centralized styles and colors system for EmoPyLab application.
All colors, fonts and styles are defined here and aligned with the EmoPyLab logo palette.

UTF-8 without BOM - Central style system for EmoPyLab
"""
from dataclasses import dataclass, field
import sys
from typing import Dict, Tuple


def _system_font_family() -> str:
    """Choose a native UI font that Qt resolves reliably on each platform."""
    if sys.platform == "darwin":
        return "SF Pro Text"
    if sys.platform.startswith("win"):
        return "Segoe UI"
    return "Noto Sans"


def _system_mono_font_family() -> str:
    """Choose a native monospace font with a safe Qt fallback per platform."""
    if sys.platform == "darwin":
        return "SF Mono"
    if sys.platform.startswith("win"):
        return "Cascadia Mono"
    return "DejaVu Sans Mono"


@dataclass
class ColorPalette:
    """
    Application color palette.
    Aligned with EmoPyLab logo palette (cyan-blue + steel gray).
    
    Main colors:
    - primary: Logo blue/cyan
    - background: Cool application canvas
    - surface: White panels
    - text: Slate foreground
    """

    # Brand and interactive accents (Harmonized Deep Amber Gold & Obsidian).
    primary: str = "#B45309"
    primary_dark: str = "#92400E"
    primary_light: str = "#F59E0B"
    primary_subtle: str = "#2A1D0B"

    # Semantic feedback colors.
    success: str = "#10B981"
    warning: str = "#F59E0B"
    warning_gold: str = "#FBBF24"
    danger: str = "#EF4444"
    info: str = "#06B6D4"

    # Light-theme surfaces.
    background: str = "#F8FAFC"
    surface: str = "#FFFFFF"
    surface_variant: str = "#F1F5F9"
    surface_soft: str = "#F8FAFC"
    surface_active: str = "#E2E8F0"

    # Text and accessible contrast roles.
    text_primary: str = "#0F172A"
    text_secondary: str = "#475569"
    text_muted: str = "#64748B"
    text_disabled: str = "#94A3B8"
    text_on_primary: str = "#FFFFFF"

    # Borders and focus affordances.
    border: str = "#CBD5E1"
    border_light: str = "#E2E8F0"
    divider: str = "#E2E8F0"
    border_focus: str = "#B45309"
    focus_ring: str = "#FDE68A"

    # Legacy semantic aliases retained for plugin/UI compatibility.
    accent_blue: str = "#B45309"
    accent_orange: str = "#9A3412"
    selection_blue: str = "#B45309"
    selection_orange: str = "#854D0E"

    # Colorblind-safe chart series for optimization runs.
    chart_series: Tuple[str, ...] = (
        "#F59E0B", "#0A84FF", "#10B981", "#8B5CF6", "#EC4899", "#06B6D4"
    )
    chart_reference_front: str = "#FFFFFF"

    # Dark theme roles (Rich Obsidian Canvas + Deep Amber Gold).
    dark_background: str = "#0B0F19"
    dark_surface: str = "#111827"
    dark_surface_variant: str = "#1F2937"
    dark_surface_soft: str = "#172033"
    dark_surface_active: str = "#2B374D"
    dark_text_primary: str = "#F8FAFC"
    dark_text_secondary: str = "#CBD5E1"
    dark_text_muted: str = "#94A3B8"
    dark_text_disabled: str = "#64748B"
    dark_border: str = "#334155"
    dark_border_light: str = "#253247"

    def chart_palette(self) -> Tuple[str, ...]:
        """Return the stable, colorblind-safe chart series palette."""
        return self.chart_series

    def to_dict(self) -> Dict[str, str]:
        """Return palette as dictionary for use with qt_material."""
        return {
            "primary": self.primary,
            "primary_dark": self.primary_dark,
            "primary_light": self.primary_light,
            "success": self.success,
            "warning": self.warning,
            "danger": self.danger,
            "info": self.info,
            "background": self.background,
            "surface": self.surface,
            "text_primary": self.text_primary,
            "text_secondary": self.text_secondary,
        }

@dataclass
class Typography:
    """Typography configuration."""
    
    # Qt accepts one preferred family; choose it from the host OS and fall back
    # to each platform's standard typefaces when it is not installed.
    font_family: str = field(default_factory=_system_font_family)
    font_family_mono: str = field(default_factory=_system_mono_font_family)

    # Compact but readable scientific workstation scale.
    font_size_xs: int = 10
    font_size_sm: int = 11
    font_size_base: int = 13
    font_size_lg: int = 14
    font_size_xl: int = 16
    font_size_2xl: int = 20
    font_size_3xl: int = 24
    font_size_caption: int = 11
    font_size_code: int = 12
    font_size_heading: int = 16
    font_size_display: int = 20
    line_height_base: float = 1.35

    # Semantic aliases useful to widgets that need a text hierarchy.
    body_size: int = 13
    heading_size: int = 16
    display_size: int = 20
    code_size: int = 12
    caption_size: int = 11

    # Font weights
    font_weight_normal: int = 400
    font_weight_medium: int = 500
    font_weight_semibold: int = 600
    font_weight_bold: int = 700


@dataclass
class Spacing:
    """Spacing system."""
    
    xs: int = 4
    sm: int = 6
    md: int = 8
    lg: int = 12
    xl: int = 16
    xxl: int = 24
    xxxl: int = 32


@dataclass
class AnimationSettings:
    """Animation configuration."""
    
    duration_fast: int = 150
    duration_normal: int = 300
    duration_slow: int = 500
    
    # Easing curves (QEasingCurve.Type)
    easing_default: str = "InOutCubic"
    easing_entrance: str = "OutCubic"
    easing_exit: str = "InCubic"


class AppStyles:
    """
    Main styles class for EmoPyLab application.
    Use as single source of truth for all styles.
    
    This class centralizes all colors, fonts and styles
    to ensure visual consistency across the application.
    
    Usage:
        from styles import AppStyles
        
        # Colors
        color = AppStyles.colors.primary
        
        # Stylesheet
        widget.setStyleSheet(f"background: {AppStyles.colors.surface};")
    """
    
    colors = ColorPalette()
    typography = Typography()
    spacing = Spacing()
    animation = AnimationSettings()
    current_theme = "light"

    @classmethod
    def set_theme_mode(cls, mode: str = "light") -> None:
        """Switch active color tokens between light mode (default for paper prints) and dark mode."""
        cls.current_theme = "dark" if mode.lower() == "dark" else "light"
        if cls.current_theme == "dark":
            cls.colors.background = cls.colors.dark_background
            cls.colors.surface = cls.colors.dark_surface
            cls.colors.surface_variant = cls.colors.dark_surface_variant
            cls.colors.surface_soft = cls.colors.dark_surface_soft
            cls.colors.surface_active = cls.colors.dark_surface_active
            cls.colors.text_primary = cls.colors.dark_text_primary
            cls.colors.text_secondary = cls.colors.dark_text_secondary
            cls.colors.text_muted = cls.colors.dark_text_muted
            cls.colors.text_disabled = cls.colors.dark_text_disabled
            cls.colors.border = cls.colors.dark_border
            cls.colors.border_light = cls.colors.dark_border_light
            cls.colors.divider = cls.colors.dark_border_light
            cls.colors.primary = "#B45309"
            cls.colors.primary_dark = "#92400E"
            cls.colors.primary_light = "#D97706"
            cls.colors.primary_subtle = "#2A1D0B"
            cls.colors.border_focus = "#F59E0B"
            cls.colors.focus_ring = "#78350F"
            cls.colors.text_on_primary = "#FFFFFF"
            cls.colors.warning = "#FBBF24"
            cls.colors.chart_reference_front = "#FFFFFF"
            cls.colors.selection_blue = "#9A3412"
            cls.colors.selection_orange = "#854D0E"
        else:
            cls.colors.background = "#F8FAFC"
            cls.colors.surface = "#FFFFFF"
            cls.colors.surface_variant = "#F1F5F9"
            cls.colors.surface_soft = "#F8FAFC"
            cls.colors.surface_active = "#E2E8F0"
            cls.colors.text_primary = "#0F172A"
            cls.colors.text_secondary = "#475569"
            cls.colors.text_muted = "#64748B"
            cls.colors.text_disabled = "#94A3B8"
            cls.colors.border = "#CBD5E1"
            cls.colors.border_light = "#E2E8F0"
            cls.colors.divider = "#E2E8F0"
            cls.colors.primary = "#D97706"
            cls.colors.primary_dark = "#B45309"
            cls.colors.primary_light = "#F59E0B"
            cls.colors.primary_subtle = "#FEF3C7"
            cls.colors.border_focus = "#D97706"
            cls.colors.focus_ring = "#FDE68A"
            cls.colors.text_on_primary = "#FFFFFF"
            cls.colors.warning = "#F59E0B"
            cls.colors.chart_reference_front = "#0F172A"
            cls.colors.selection_blue = "#B45309"
            cls.colors.selection_orange = "#92400E"
    
    @staticmethod
    def get_stylesheet() -> str:
        """
        Return complete CSS stylesheet for the application.
        Complementary to the application theme.
        """
        colors = AppStyles.colors
        typo = AppStyles.typography
        
        return f"""
        /* EmoPyLab scientific workstation design system */
        QMainWindow, QWidget {{
            background-color: {colors.background};
            color: {colors.text_primary};
            font-family: '{typo.font_family}';
            font-size: {typo.font_size_base}px;
        }}
        QWidget:disabled {{ color: {colors.text_disabled}; }}

        QMenuBar {{
            background-color: {colors.surface};
            color: {colors.text_secondary};
            border-bottom: 1px solid {colors.border_light};
            padding: 2px 6px;
        }}
        QMenuBar::item {{ padding: 6px 10px; border-radius: 6px; }}
        QMenuBar::item:selected {{
            background-color: {colors.primary_subtle};
            color: {colors.primary_dark};
        }}
        QMenu {{
            background-color: {colors.surface};
            color: {colors.text_primary};
            border: 1px solid {colors.border_light};
            padding: 5px;
        }}
        QMenu::item {{ padding: 7px 24px 7px 10px; border-radius: 5px; }}
        QMenu::item:selected {{ background-color: {colors.primary_subtle}; color: {colors.primary_dark}; }}

        QScrollArea, QAbstractScrollArea {{ border: 0; background: transparent; }}
        QScrollBar:vertical {{
            background: transparent; width: 10px; margin: 2px 0;
        }}
        QScrollBar::handle:vertical {{
            background: {colors.border}; min-height: 28px; border-radius: 5px;
        }}
        QScrollBar::handle:vertical:hover {{ background: {colors.text_muted}; }}
        QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical,
        QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {{ background: transparent; height: 0; }}
        QScrollBar:horizontal {{ background: transparent; height: 10px; margin: 0 2px; }}
        QScrollBar::handle:horizontal {{ background: {colors.border}; min-width: 28px; border-radius: 5px; }}
        QScrollBar::handle:horizontal:hover {{ background: {colors.text_muted}; }}
        QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal,
        QScrollBar::add-page:horizontal, QScrollBar::sub-page:horizontal {{ background: transparent; width: 0; }}

        QGroupBox {{
            background-color: {colors.surface};
            color: {colors.text_primary};
            border: 1px solid {colors.border_light};
            border-radius: 8px;
            margin-top: 16px;
            padding: 16px 12px 12px 12px;
            font-weight: {typo.font_weight_semibold};
        }}
        QGroupBox::title {{
            subcontrol-origin: margin; subcontrol-position: top left;
            left: 10px; padding: 0 6px; color: {colors.text_primary};
            background-color: {colors.surface};
        }}
        QLabel {{ color: {colors.text_primary}; background: transparent; }}

        QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox {{
            min-height: 30px; padding: 5px 10px;
            border: 1px solid {colors.border}; border-radius: 7px;
            background-color: {colors.surface}; color: {colors.text_primary};
            selection-background-color: {colors.primary}; selection-color: {colors.text_on_primary};
        }}
        QLineEdit:hover, QComboBox:hover, QSpinBox:hover, QDoubleSpinBox:hover {{ border-color: {colors.primary_light}; }}
        QLineEdit:focus, QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus {{
            border: 2px solid {colors.border_focus};
            background-color: {colors.surface};
        }}
        QLineEdit:disabled, QComboBox:disabled, QSpinBox:disabled, QDoubleSpinBox:disabled {{
            color: {colors.text_disabled}; background-color: {colors.surface_variant}; border-color: {colors.border_light};
        }}
        QLineEdit[readOnly="true"] {{ background-color: {colors.surface_soft}; }}
        QComboBox::drop-down {{ border: 0; width: 26px; }}
        QComboBox QAbstractItemView {{
            background: {colors.surface}; color: {colors.text_primary};
            border: 1px solid {colors.border_light}; selection-background-color: {colors.primary_subtle};
            selection-color: {colors.primary_dark}; padding: 4px;
        }}

        QPlainTextEdit, QTextEdit {{
            border: 1px solid {colors.border}; border-radius: 8px;
            background-color: {colors.surface}; color: {colors.text_primary}; padding: 8px;
            selection-background-color: {colors.primary}; selection-color: {colors.text_on_primary};
        }}
        QPlainTextEdit:focus, QTextEdit:focus {{ border: 2px solid {colors.border_focus}; }}

        QListWidget {{
            border: 1px solid {colors.border_light}; border-radius: 8px;
            background-color: {colors.surface}; alternate-background-color: {colors.surface_soft};
            outline: 0;
        }}
        QListWidget::item {{ min-height: 26px; padding: 5px 8px; border-radius: 5px; margin: 1px 3px; }}
        QListWidget::item:hover {{ background-color: {colors.surface_variant}; }}
        QListWidget::item:selected {{
            background-color: {colors.primary_subtle}; color: {colors.primary_dark}; font-weight: {typo.font_weight_semibold};
        }}
        QListWidget:focus {{ border: 2px solid {colors.border_focus}; }}

        QPushButton {{
            min-height: 30px; padding: 6px 12px; border: 1px solid {colors.border}; border-radius: 7px;
            background-color: {colors.surface}; color: {colors.text_primary}; font-weight: {typo.font_weight_medium};
        }}
        QPushButton:hover {{ background-color: {colors.surface_variant}; border-color: {colors.primary_light}; }}
        QPushButton:pressed {{ background-color: {colors.surface_active}; border-color: {colors.primary_dark}; }}
        QPushButton:focus {{ border: 2px solid {colors.border_focus}; }}
        QPushButton:disabled {{ background-color: {colors.surface_variant}; color: {colors.text_disabled}; border-color: {colors.border_light}; }}

        QCheckBox, QRadioButton {{ spacing: 7px; color: {colors.text_primary}; }}
        QCheckBox:focus, QRadioButton:focus {{ color: {colors.primary_dark}; }}
        QCheckBox::indicator, QRadioButton::indicator {{ width: 16px; height: 16px; }}

        QProgressBar {{
            min-height: 10px; border: 0; border-radius: 5px; text-align: center;
            background-color: {colors.surface_active}; color: {colors.text_secondary};
        }}
        QProgressBar::chunk {{ background-color: {colors.primary}; border-radius: 5px; }}
        QSplitter::handle {{ background-color: transparent; }}
        QSplitter::handle:hover {{ background-color: {colors.border}; }}

        QTableWidget {{
            border: 1px solid {colors.border_light}; border-radius: 8px;
            background-color: {colors.surface}; gridline-color: {colors.border_light};
            alternate-background-color: {colors.surface_soft};
            selection-background-color: {colors.primary_subtle}; selection-color: {colors.text_primary};
        }}
        QTableWidget::item {{ padding: 5px 7px; border: 0; }}
        QTableWidget:focus {{ border: 2px solid {colors.border_focus}; }}
        QHeaderView::section {{
            background-color: {colors.surface_variant}; color: {colors.text_secondary}; border: 0;
            border-bottom: 1px solid {colors.border_light}; padding: 8px 7px;
            font-weight: {typo.font_weight_semibold};
        }}

        QTabWidget#primaryWorkflowTabs {{ background-color: {colors.background}; }}
        QTabWidget#primaryWorkflowTabs::pane {{ border: 0; border-top: 1px solid #334155; background-color: {colors.background}; margin: 0; }}
        QTabWidget#primaryWorkflowTabs::tab-bar {{ alignment: center; background-color: #0F172A; }}
        QTabBar#primaryWorkflowTabBar {{ background-color: #0F172A; min-height: 64px; border: 0; border-bottom: 1px solid #334155; }}
        QTabBar#primaryWorkflowTabBar::tab {{ background: transparent; color: #94A3B8; border: 0; margin: 0; padding: 0; min-height: 64px; }}
        QTabBar#primaryWorkflowTabBar::tab:selected {{ color: #F8FAFC; }}
        QToolTip {{ background-color: {colors.surface_variant}; color: {colors.text_primary}; border: 1px solid {colors.border}; padding: 7px 9px; }}
        """

    @staticmethod
    def get_qt_material_theme() -> Dict[str, str]:
        """
        Return extra configuration for qt_material.
        Use with apply_stylesheet(app, extra=AppStyles.get_qt_material_theme())
        """
        return AppStyles.colors.to_dict()

    @staticmethod
    def get_app_shell_stylesheet() -> str:
        """Return the compact top-shell stylesheet contract."""
        colors = AppStyles.colors
        return (
            f"background-color: {colors.surface}; "
            f"border-bottom: 1px solid {colors.border_light};"
        )
    
    @staticmethod
    def get_catalog_list_style(selection_color: str = None) -> str:
        """Return a consistent catalog list treatment with a semantic selection color."""
        colors = AppStyles.colors
        selected_foreground = selection_color or colors.warning_gold
        return f"""
            QListWidget {{
                border: 1px solid {colors.border_light};
                border-radius: 8px;
                background: {colors.surface};
            }}
            QListWidget::item {{
                color: {colors.text_primary};
                min-height: 26px;
                padding: 5px 8px;
                margin: 1px 3px;
                border-radius: 5px;
            }}
            QListWidget::item:hover {{ background: {colors.surface_variant}; }}
            QListWidget::item:selected {{
                background: {colors.primary_subtle};
                color: {selected_foreground};
                border: 1px solid rgba(245, 158, 11, 0.3);
                font-weight: {AppStyles.typography.font_weight_semibold};
            }}
        """

    @staticmethod
    def get_algorithm_list_style() -> str:
        """Return gold-accented stylesheet for algorithm catalogs."""
        return AppStyles.get_catalog_list_style(AppStyles.colors.warning_gold)

    @staticmethod
    def get_problem_list_style() -> str:
        """Return amber-accented stylesheet for problem catalogs."""
        return AppStyles.get_catalog_list_style(AppStyles.colors.primary_light)

    @staticmethod
    def get_metric_list_style() -> str:
        """Return green-accented stylesheet for metric catalogs."""
        colors = AppStyles.colors
        return AppStyles.get_catalog_list_style(colors.success)

    @staticmethod
    def get_search_input_style() -> str:
        """Return a compact, high-contrast search field treatment."""
        colors = AppStyles.colors
        return (
            f"background: {colors.surface}; color: {colors.text_primary}; "
            f"border: 1px solid {colors.border}; border-radius: 8px; "
            "min-height: 30px; padding: 5px 10px;"
        )

    @staticmethod
    def get_chip_badge_style(variant: str = "neutral") -> str:
        """Return a semantic pill style for counts, filters, and status chips."""
        colors = AppStyles.colors
        variants = {
            "primary": (colors.primary_dark, colors.primary_subtle, "#BFDBFE"),
            "success": ("#047857", "#ECFDF5", "#A7F3D0"),
            "warning": ("#B45309", "#FFFBEB", "#FDE68A"),
            "danger": ("#B91C1C", "#FEF2F2", "#FECACA"),
            "neutral": (colors.text_secondary, colors.surface_soft, colors.border_light),
        }
        foreground, background, border = variants.get(variant, variants["neutral"])
        return (
            f"color: {foreground}; background: {background}; border: 1px solid {border}; "
            f"border-radius: 999px; padding: 2px 7px; font-size: {AppStyles.typography.font_size_caption}px; "
            f"font-weight: {AppStyles.typography.font_weight_semibold};"
        )

    @staticmethod
    def get_kpi_card_style(accent: str = None) -> str:
        """Return a restrained summary-card surface for workload and metric totals."""
        colors = AppStyles.colors
        border = accent or colors.primary
        return (
            f"color: {colors.text_primary}; background: {colors.surface}; "
            f"border: 1px solid {colors.border_light}; border-top: 2px solid {border}; "
            "border-radius: 8px; padding: 10px;"
        )

    @staticmethod
    def get_stat_header_style() -> str:
        """Return the table-header style used by statistical result views."""
        colors = AppStyles.colors
        return (
            f"background: {colors.surface_variant}; color: {colors.text_secondary}; "
            f"border-bottom: 1px solid {colors.border_light}; padding: 8px 7px; "
            f"font-weight: {AppStyles.typography.font_weight_semibold};"
        )

    @staticmethod
    def get_metric_row_highlight_style() -> str:
        """Return an accessible best-result highlight without changing data meaning."""
        return "background: #ECFDF5; color: #065F46; font-weight: 600;"

    @staticmethod
    def get_chart_palette() -> Tuple[str, ...]:
        """Return the shared colorblind-safe palette used by QtCharts and Matplotlib."""
        return AppStyles.colors.chart_palette()

    @staticmethod
    def get_selection_card_style(color: str = None) -> str:
        """
        Return stylesheet for selection cards with pure white text and deep gold background.
        """
        colors = AppStyles.colors
        bg_color = color or colors.primary
        return f"""
            background: {bg_color};
            color: #FFFFFF;
            border: 1px solid rgba(245, 158, 11, 0.4);
            border-radius: 8px;
            padding: 8px;
            font-weight: 700;
        """
    
    @staticmethod
    def get_title_style() -> str:
        """Return stylesheet for titles."""
        colors = AppStyles.colors
        return f"color: {colors.accent_blue}; font-weight: 700;"

    @staticmethod
    def get_muted_style() -> str:
        """Return stylesheet for secondary/muted text."""
        colors = AppStyles.colors
        return f"color: {colors.text_secondary};"

    @staticmethod
    def get_helper_text_style() -> str:
        """Return stylesheet for short helper text below dense controls."""
        colors = AppStyles.colors
        typo = AppStyles.typography
        return f"color: {colors.text_secondary}; font-size: {typo.font_size_sm}px;"

    @staticmethod
    def get_section_subtitle_style() -> str:
        """Return stylesheet for section subtitles and explanatory copy."""
        colors = AppStyles.colors
        typo = AppStyles.typography
        return f"color: {colors.text_secondary}; font-size: {typo.font_size_base}px;"

    @staticmethod
    def get_count_badge_style() -> str:
        """Return the neutral chip used for compact catalog counts."""
        return AppStyles.get_chip_badge_style()

    @staticmethod
    def get_panel_hint_style() -> str:
        """Return stylesheet for wrapped hints inside panels."""
        colors = AppStyles.colors
        return (
            f"color: {colors.text_secondary}; "
            f"background: {colors.surface_soft}; "
            f"border: 1px solid {colors.border_light}; "
            f"border-radius: 8px; "
            f"padding: 8px;"
        )

    @staticmethod
    def get_guided_panel_style() -> str:
        """Return a clear review panel without an ornamental accent rail."""
        colors = AppStyles.colors
        typo = AppStyles.typography
        return (
            f"color: {colors.text_primary}; "
            f"font-size: {typo.font_size_base}px; "
            f"background: {colors.primary_subtle}; "
            f"border: 1px solid #BFDBFE; "
            f"border-radius: 8px; "
            f"padding: 9px 10px;"
        )

    @staticmethod
    def get_primary_button_style() -> str:
        """Return the single high-emphasis action treatment."""
        colors = AppStyles.colors
        typo = AppStyles.typography
        return f"""
            QPushButton {{
                min-height: 30px;
                background-color: {colors.primary};
                color: {colors.text_on_primary};
                border: 1px solid {colors.primary_dark};
                border-radius: 7px;
                padding: 6px 14px;
                font-weight: {typo.font_weight_semibold};
            }}
            QPushButton:hover {{ background-color: {colors.primary_dark}; }}
            QPushButton:pressed {{ background-color: #0057B8; }}
            QPushButton:focus {{ border: 2px solid {colors.focus_ring}; }}
            QPushButton:disabled {{
                background-color: {colors.border}; color: {colors.text_disabled}; border-color: {colors.border_light};
            }}
        """

    @staticmethod
    def get_secondary_button_style() -> str:
        """Return the low-emphasis action treatment."""
        colors = AppStyles.colors
        typo = AppStyles.typography
        return f"""
            QPushButton {{
                min-height: 30px;
                background-color: {colors.surface}; color: {colors.text_primary};
                border: 1px solid {colors.border}; border-radius: 7px;
                padding: 6px 12px; font-weight: {typo.font_weight_medium};
            }}
            QPushButton:hover {{ background-color: {colors.surface_variant}; border-color: {colors.primary_light}; }}
            QPushButton:pressed {{ background-color: {colors.surface_active}; border-color: {colors.primary_dark}; }}
            QPushButton:focus {{ border: 2px solid {colors.border_focus}; }}
            QPushButton:disabled {{
                background-color: {colors.surface_variant}; color: {colors.text_disabled}; border-color: {colors.border_light};
            }}
        """

    @staticmethod
    def get_danger_button_style() -> str:
        """Return a legible destructive-action treatment."""
        colors = AppStyles.colors
        typo = AppStyles.typography
        return f"""
            QPushButton {{
                min-height: 30px;
                background-color: #FEF2F2; color: #B91C1C; border: 1px solid #FECACA;
                border-radius: 7px; padding: 6px 12px; font-weight: {typo.font_weight_semibold};
            }}
            QPushButton:hover {{ background-color: #FEE2E2; border-color: #F87171; }}
            QPushButton:pressed {{ background-color: #FECACA; }}
            QPushButton:focus {{ border: 2px solid {colors.danger}; }}
            QPushButton:disabled {{
                background-color: {colors.surface_variant}; color: {colors.text_disabled}; border-color: {colors.border_light};
            }}
        """

    @staticmethod
    def get_status_pill_style(color: str = None) -> str:
        """Return compact status-pill styling with an accessible foreground."""
        colors = AppStyles.colors
        fg = color or "#047857"
        return (
            f"color: {fg}; background: {colors.surface_soft}; border: 1px solid {colors.border_light}; "
            f"border-radius: 999px; padding: 3px 8px; font-size: {AppStyles.typography.font_size_caption}px; "
            f"font-weight: {AppStyles.typography.font_weight_semibold};"
        )

    @staticmethod
    def get_metric_tile_style(accent: str = None) -> str:
        """Return a summary metric tile without decorative accent rails."""
        return AppStyles.get_kpi_card_style(accent=accent)

    @staticmethod
    def get_log_view_style() -> str:
        """Return cross-platform technical log/editor styling."""
        colors = AppStyles.colors
        typo = AppStyles.typography
        return (
            f"font-family: '{typo.font_family_mono}'; font-size: {typo.font_size_code}px; "
            f"color: {colors.text_secondary}; background: {colors.surface_soft}; "
            f"border: 1px solid {colors.border_light}; border-radius: 8px; padding: 8px;"
        )

    @staticmethod
    def get_table_style(object_name: str = None) -> str:
        """Return table stylesheet, optionally scoped to an object name."""
        colors = AppStyles.colors
        selector = f"QTableWidget#{object_name}" if object_name else "QTableWidget"
        return f"""
            {selector} {{
                border: 1px solid {colors.border_light};
                border-radius: 8px;
                background-color: {colors.surface};
                gridline-color: {colors.border_light};
                alternate-background-color: {colors.surface_soft};
                selection-background-color: {colors.primary_subtle};
                selection-color: {colors.text_primary};
            }}
            {selector}::item {{
                border: none;
                padding: 5px 7px;
            }}
            {selector}::item:selected,
            {selector}::item:selected:active,
            {selector}::item:selected:!active {{
                background-color: {colors.primary_subtle};
                color: {colors.text_primary};
            }}
            {selector}::item:hover {{
                background-color: {colors.surface_variant};
                color: {colors.text_primary};
            }}
            {selector} QHeaderView::section {{
                background-color: {colors.surface_variant};
                color: {colors.text_secondary};
                border: 0;
                border-bottom: 1px solid {colors.border_light};
                padding: 8px 7px;
                font-weight: {AppStyles.typography.font_weight_semibold};
            }}
        """


# Export for easy import
__all__ = ['AppStyles', 'ColorPalette', 'Typography', 'Spacing', 'AnimationSettings']
