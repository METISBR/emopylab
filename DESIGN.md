---
name: EmoPyLab
description: Scientific visual analytics workstation for multi-objective optimization, benchmark orchestration, and local LLM meta-controllers
colors:
  primary: "#0A84FF"
  primary-dark: "#0066CC"
  primary-light: "#38BDF8"
  primary-subtle: "#EBF5FF"
  success: "#10B981"
  warning: "#F59E0B"
  danger: "#EF4444"
  info: "#06B6D4"
  background: "#F8FAFC"
  surface: "#FFFFFF"
  surface-variant: "#F1F5F9"
  surface-soft: "#F8FAFC"
  surface-active: "#E2E8F0"
  text-primary: "#0F172A"
  text-secondary: "#475569"
  text-muted: "#64748B"
  text-disabled: "#94A3B8"
  text-on-primary: "#FFFFFF"
  border: "#CBD5E1"
  border-light: "#E2E8F0"
  border-focus: "#0A84FF"
  focus-ring: "#93C5FD"
  chart-series-1: "#0A84FF"
  chart-series-2: "#F97316"
  chart-series-3: "#10B981"
  chart-series-4: "#8B5CF6"
  chart-series-5: "#EC4899"
  chart-series-6: "#EAB308"
  dark-background: "#0B1220"
  dark-surface: "#111827"
  dark-surface-variant: "#1F2937"
  dark-text-primary: "#F8FAFC"
  dark-text-secondary: "#CBD5E1"
  dark-border: "#334155"
typography:
  display:
    fontFamily: "SF Pro Text, Segoe UI, Noto Sans, sans-serif"
    fontSize: "20px"
    fontWeight: 700
    lineHeight: 1.2
  heading:
    fontFamily: "SF Pro Text, Segoe UI, Noto Sans, sans-serif"
    fontSize: "16px"
    fontWeight: 600
    lineHeight: 1.3
  body:
    fontFamily: "SF Pro Text, Segoe UI, Noto Sans, sans-serif"
    fontSize: "13px"
    fontWeight: 400
    lineHeight: 1.35
  caption:
    fontFamily: "SF Pro Text, Segoe UI, Noto Sans, sans-serif"
    fontSize: "11px"
    fontWeight: 400
    lineHeight: 1.3
  code:
    fontFamily: "SF Mono, Cascadia Mono, DejaVu Sans Mono, monospace"
    fontSize: "12px"
    fontWeight: 400
    lineHeight: 1.4
rounded:
  xs: "4px"
  sm: "5px"
  md: "6px"
  lg: "7px"
  xl: "8px"
  full: "999px"
spacing:
  xs: "4px"
  sm: "6px"
  md: "8px"
  lg: "12px"
  xl: "16px"
  xxl: "24px"
  xxxl: "32px"
components:
  button-primary:
    backgroundColor: "{colors.primary}"
    textColor: "{colors.text-on-primary}"
    typography: "{typography.body}"
    rounded: "{rounded.lg}"
    padding: "6px 14px"
    height: "30px"
  button-primary-hover:
    backgroundColor: "{colors.primary-dark}"
    textColor: "{colors.text-on-primary}"
  button-secondary:
    backgroundColor: "{colors.surface}"
    textColor: "{colors.text-primary}"
    typography: "{typography.body}"
    rounded: "{rounded.lg}"
    padding: "6px 12px"
    height: "30px"
  button-danger:
    backgroundColor: "#FEF2F2"
    textColor: "#B91C1C"
    typography: "{typography.body}"
    rounded: "{rounded.lg}"
    padding: "6px 12px"
    height: "30px"
  input-text:
    backgroundColor: "{colors.surface}"
    textColor: "{colors.text-primary}"
    typography: "{typography.body}"
    rounded: "{rounded.lg}"
    padding: "5px 10px"
    height: "30px"
  catalog-list-item:
    backgroundColor: "transparent"
    textColor: "{colors.text-primary}"
    typography: "{typography.body}"
    rounded: "{rounded.sm}"
    padding: "5px 8px"
    height: "26px"
  catalog-list-item-selected:
    backgroundColor: "{colors.primary-subtle}"
    textColor: "{colors.primary-dark}"
  kpi-card:
    backgroundColor: "{colors.surface}"
    textColor: "{colors.text-primary}"
    rounded: "{rounded.xl}"
    padding: "12px 14px"
  status-pill:
    backgroundColor: "{colors.surface-soft}"
    textColor: "{colors.text-secondary}"
    typography: "{typography.caption}"
    rounded: "{rounded.full}"
    padding: "3px 8px"
---

## Overview

EmoPyLab is a high-density, scientific visual analytics workstation for evolutionary multi-objective optimization (EMO) and many-objective optimization (MaOPs). It unites benchmark design, real-time live convergence visualization, MCDM compromise decision making, and local LLM meta-controllers within a structured desktop interface.

The design prioritizes **operational clarity, ergonomic visual density, and trustworthy analytical representation**. Because optimization experiments require deep focus across hundreds of algorithms, problems, and seed runs, the interface recedes to let scientific data, Pareto fronts, and statistical distributions take center stage.

**The Workstation First Rule.** Every layout element, form control, and list item is calibrated for scanability and high-density productivity. Decorative padding, oversized touch targets, and distracting motion are strictly avoided in favor of crisp 1px borders, subtle 6-8px corner radii, and colorblind-safe categorical visualization.

**The Single Source of Truth Rule.** All visual tokens (colors, font scales, spacing steps, and component stylesheets) are anchored in `styles.py` and consumed directly across `EmoPyLab.py` and modular core widgets. Hard-coded hex colors in application code are strictly forbidden.

---

## Colors

The EmoPyLab palette is built around a vibrant scientific cyan-blue primary (`#0A84FF`), anchored by a slate neutral hierarchy (`#0F172A` to `#F8FAFC`) and six colorblind-safe categorical series for Pareto front and convergence plotting.
### Brand & Interactive Roles
- **Primary (`#0A84FF`)**: Primary action buttons, active navigation indicators, key plot points, and keyboard focus outlines.
- **Primary Dark (`#0066CC`)**: Active/pressed button states, selected catalog list item text, and high-contrast emphasis.
- **Primary Light (`#38BDF8`)**: Hover border highlights and secondary accent marks.
- **Primary Subtle (`#EBF5FF`)**: Background tint for selected catalog rows, active menu items, and highlighted metric cells.

### Feedback & Status Roles
- **Success (`#10B981`)**: Completed benchmark runs, successful validation checks, and non-dominated Pareto sets.
- **Warning (`#F59E0B`)**: Non-critical warnings, fallback runtime notices, and constrained boundary approaches.
- **Danger (`#EF4444`)**: Execution exceptions, termination aborts, and destructive reset actions.
- **Info (`#06B6D4`)**: Informational status indicators, hardware runtime detection notes, and metadata badges.

### Surface & Canvas Roles
- **Canvas Background (`#F8FAFC`)**: The master cool gray application canvas behind panels, cards, and tab views.
- **Surface (`#FFFFFF`)**: Primary content panels, group boxes, input fields, and table widgets.
- **Surface Variant (`#F1F5F9`)**: Table headers, hovered list rows, disabled inputs, and secondary badges.
- **Surface Soft (`#F8FAFC`)**: Alternating table rows, pill backgrounds, and code log editors.
- **Surface Active (`#E2E8F0`)**: Pressed button states and progress bar track backgrounds.

### Typography & Border Roles
- **Text Primary (`#0F172A`)**: Primary readable content, high-contrast labels, and table values (contrast > 14:1 on white).
- **Text Secondary (`#475569`)**: Secondary descriptive copy, form subheaders, and table column headers.
- **Text Muted (`#64748B`)**: Helper text, placeholder strings, and metadata tags.
- **Text Disabled (`#94A3B8`)**: Inactive options, disabled button labels, and unavailable features.
- **Border (`#CBD5E1`)**: Standard input outlines, button borders, and card perimeters.
- **Border Light (`#E2E8F0`)**: Subtle table gridlines, group box borders, and tab bar divider lines.
- **Focus Ring (`#93C5FD`)**: 2px focus outline with subtle translucent glow for accessible keyboard navigation.

### Scientific Chart Palette
- **Series 1 (`#0A84FF`)**: Primary algorithm run / candidate front.
- **Series 2 (`#F97316`)**: Comparison algorithm A (vivid orange).
- **Series 3 (`#10B981`)**: Comparison algorithm B (emerald green).
- **Series 4 (`#8B5CF6`)**: Comparison algorithm C (purple).
- **Series 5 (`#EC4899`)**: Comparison algorithm D (pink).
- **Series 6 (`#EAB308`)**: Comparison algorithm E (amber).
- **True Reference Front (`#0F172A`)**: Ground truth analytical Pareto front (high-contrast dark slate).

**The Contrast Floor Rule.** All text on interactive and data surfaces must meet or exceed WCAG AA 4.5:1 (and 7:1 AAA for body text). When a status pill uses a tinted background, the foreground text must use a darker shade (e.g. `#047857` on `#F8FAFC`).

---

## Typography

EmoPyLab adopts native operating system typography for maximum font rendering crispness and zero layout latency:
- **macOS**: `SF Pro Text` for UI, `SF Mono` for logs and code.
- **Windows**: `Segoe UI` for UI, `Cascadia Mono` for logs and code.
- **Linux**: `Noto Sans` for UI, `DejaVu Sans Mono` for logs and code.
### Type Scale
- **Display (`20px`, Bold 700, Line Height 1.2)**: Main application title in splash dialog and top-level view headers.
- **Heading (`16px`, Semibold 600, Line Height 1.3)**: Section headers, GroupBox titles, and KPI card headline values.
- **Body (`13px`, Regular 400 & Medium 500, Line Height 1.35)**: Standard workstation body text, form field labels, list item text, and table values.
- **Code (`12px`, Regular 400, Monospace, Line Height 1.4)**: Terminal log console, formula representations, Python code editors, and JSON payloads.
- **Caption / Meta (`11px`, Medium 500 & Semibold 600, Line Height 1.3)**: Status pills, count badges, tooltips, and footnote timestamps.

**The Proportional Density Rule.** Body text is fixed at `13px` with a `1.35` line height to balance comfortable legibility during extended scientific sessions with dense catalog scanability.

---

## Layout

EmoPyLab uses a modular 5-tab primary workflow architecture centered inside a single top-level window (`EmoPyLabMainWindow`):
1. **Configure Tab (`science`)**: Three-column catalog selection (Algorithms, Problems, Metrics) with fine-grained parameter tuning, seed planning, and termination criteria.
2. **Experiment Tab (`biotech`)**: Live orchestration console with execution progress bars, real-time Pareto scatter plots, and live metric convergence curves.
3. **Results Tab (`analytics`)**: Post-experiment analytical workbench featuring Friedman/Wilcoxon statistical tables, boxplots, radar charts, and MCDM compromise ranking.
4. **AI Agent Tab (`smart_toy`)**: Local LLM meta-controller console, problem formulation assistant, and prompt inspection workspace.
5. **Extensibility Tab (`code`)**: Plugin discovery browser, custom algorithm/problem validator, and runtime directory inspector.

### Spacing & Grid System
- **Base Grid Unit**: 4px.
- **`xs` (4px)**: Micro-padding between badge icons and label text.
- **`sm` (6px)**: Internal vertical button padding and list item padding.
- **`md` (8px)**: Standard gap between form fields and card internal padding.
- **`lg` (12px)**: Margin between major card groups and split-pane gutters.
- **`xl` (16px)**: GroupBox internal padding and tab container margins.
- **`xxl` (24px)**: Outer window margins and major layout section separators.
- **`xxxl` (32px)**: Splash screen padding and modal dialog spacing.

**The No-Clip Laptop Rule.** Toolbars, action buttons, and control panels must lay out using fluid horizontal layouts with wrapping or responsive scroll wrappers so that no critical action is clipped or hidden on standard 13-inch laptop displays (1280x800).

---

## Elevation & Depth

As a desktop scientific tool, EmoPyLab avoids blurry web drop shadows that degrade rendering performance across Qt canvases and X11/Wayland/Quartz window managers.
- **Flat Elevated Panels**: Depth is achieved through clear 1px borders (`#E2E8F0` / `#CBD5E1`) and background surface tonal layering (`#F8FAFC` canvas vs `#FFFFFF` cards).
- **Z-Index Layering**:
  1. Canvas background (`#F8FAFC`).
  2. Card / GroupBox panels (`#FFFFFF`).
  3. Interactive inputs, dropdowns, and buttons (`#FFFFFF` with `#CBD5E1` border).
  4. Floating Tooltips and Modal Dialogs (`#0F172A` with `#FFFFFF` text, or modal dialog sheet).
- **Focus Rings**: Focused inputs receive a crisp 2px solid border (`#0A84FF`) to provide immediate keyboard navigation feedback without layout shifting.

---

## Shapes

- **Base Radius (`7px - 8px`)**: Primary buttons, secondary buttons, text inputs, comboboxes, spinboxes, and card containers.
- **List & Menu Radius (`5px - 6px`)**: QListWidget items, QMenu items, and QMenuBar selections.
- **Full Radius (`999px`)**: Count badges, status pills, and categorical filter chips.
- **Sharp / Unrounded (`0px`)**: Table grid cells and top-level tab panes to maintain dense columnar alignment.

---

## Components

### Primary Button
- **Height**: 30px.
- **Background**: `#0A84FF` (Normal), `#0066CC` (Pressed), `#F1F5F9` (Disabled).
- **Text**: `#FFFFFF` (Normal), `#94A3B8` (Disabled), Font-weight 500.
- **Border**: None or 1px solid `#0A84FF`, Radius 7px.
- **Focus**: 2px solid `#0066CC`.

### Secondary / Ghost Button
- **Height**: 30px.
- **Background**: `#FFFFFF` (Normal), `#F1F5F9` (Hover), `#E2E8F0` (Pressed).
- **Text**: `#0F172A` (Normal), `#94A3B8` (Disabled), Font-weight 500.
- **Border**: 1px solid `#CBD5E1`, Radius 7px.

### Danger / Abort Button
- **Height**: 30px.
- **Background**: `#FEF2F2` (Normal), `#FEE2E2` (Hover), `#FECACA` (Pressed).
- **Text**: `#B91C1C` (Semibold 600).
- **Border**: 1px solid `#FECACA`, Radius 7px.

### Form Inputs (QLineEdit, QComboBox, QSpinBox)
- **Height**: 30px.
- **Padding**: 5px 10px.
- **Background**: `#FFFFFF` (Normal), `#F8FAFC` (ReadOnly), `#F1F5F9` (Disabled).
- **Border**: 1px solid `#CBD5E1` (Normal), 1px solid `#38BDF8` (Hover), 2px solid `#0A84FF` (Focus).

### Catalog Selection List (QListWidget)
- **Item Height**: 26px minimum.
- **Padding**: 5px 8px.
- **Selected State**: Background `#EBF5FF`, Text `#0066CC`, Font-weight 600, Radius 5px.
- **Hover State**: Background `#F1F5F9`, Radius 5px.

### Metric KPI Card
- **Background**: `#FFFFFF`.
- **Border**: 1px solid `#E2E8F0`, Radius 8px.
- **Padding**: 12px 14px.
- **Header**: `#475569`, 11px uppercase.
- **Value**: `#0F172A`, 16px-20px semibold.

### Scientific Data Table (QTableWidget)
- **Header Background**: `#F1F5F9`, Text `#475569`, Font-weight 600, 8px 7px padding.
- **Row Alternation**: `#FFFFFF` / `#F8FAFC`.
- **Selection**: Background `#EBF5FF`, Text `#0F172A`.
- **Gridlines**: 1px solid `#E2E8F0`.

---

## Do's and Don'ts

### Do's
- **DO** import all styling from `styles.py` via `AppStyles`.
- **DO** provide clear accessible names (`setAccessibleName`) and tooltips (`setToolTip`) on every interactive icon button and input.
- **DO** use colorblind-safe categorical palettes (`AppStyles.colors.chart_series`) for multi-algorithm comparisons.
- **DO** use monospace fonts (`SF Mono` / `Cascadia Mono` / `DejaVu Sans Mono`) for all numeric tables, code editors, and seed lists.
- **DO** wrap long control strips in responsive scroll areas or two-tier horizontal toolbars to prevent UI clipping at 1280x800.

### Don'ts
- **DON'T** write raw hex strings (e.g. `"#0A84FF"`) directly inside Qt widget setup code.
- **DON'T** apply expensive visual blur or drop-shadow effects that degrade desktop rendering FPS during live optimization loops.
- **DON'T** remove or bypass `setAccessibleName` / `setAccessibleDescription` on custom icon actions.
- **DON'T** mix different icon sets; always use `qtawesome` icons consistently.
- **DON'T** block the main Qt GUI event loop with CPU/GPU optimization loops or LLM queries; always dispatch via `QThread` and worker bridges.
