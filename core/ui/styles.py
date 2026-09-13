"""Unified AppStyles wrapper and tokens for EmoPyLab UI."""

from __future__ import annotations

from styles import AppStyles as StylesAppStyles


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
    """Dynamic wrapper for styles.py guaranteeing immediate reactive theme adaptation."""

    @classmethod
    def stylesheet(cls) -> str:
        """Return complementary stylesheet from styles.py."""
        return StylesAppStyles.get_stylesheet()

    @classmethod
    def search_input_style(cls) -> str:
        """Return the shared input treatment from styles.py."""
        return StylesAppStyles.search_input_style()

    @classmethod
    def tab_button_style(cls, active: bool = False) -> str:
        """Return the tab button stylesheet."""
        return StylesAppStyles.tab_button_style(active)
