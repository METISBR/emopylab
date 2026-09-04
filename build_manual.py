#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generator for the EmoPyLab User Manual & Technical Reference (METISBr / UFOP).

Rebuilds EmoPyLab_Manual_METISBr.pdf from data verified directly against the
current standalone tensor-native codebase (local Qwen, 5-tier backend acceleration,
298 metaheuristics, 54 problems, 77 operators, 6 metrics, in-situ MCDM, and SHA-256
reproducibility manifests).

Usage:
    python3 build_manual.py
"""
from __future__ import annotations

import os
import runpy
from datetime import datetime

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    Image,
    PageBreak,
    PageTemplate,
    Paragraph,
    Preformatted,
    Spacer,
    Table,
    TableStyle,
)

_HERE = os.path.dirname(os.path.abspath(__file__))

# Project logo (rendered on the cover); falls back to the text wordmark if absent.
LOGO_PATH = os.path.join(_HERE, "emopylab.png")

# --------------------------------------------------------------------------- #
# METISBr palette                                                             #
# --------------------------------------------------------------------------- #
NAVY = colors.HexColor("#0B2A4A")
BLUE = colors.HexColor("#11598C")
ACCENT = colors.HexColor("#1E88B5")
LIGHT = colors.HexColor("#EAF3F8")
GREY = colors.HexColor("#5B6B79")
CODE_BG = colors.HexColor("#F4F6F8")
CODE_BORDER = colors.HexColor("#D5DCE2")

OUTFILE = os.path.join(_HERE, "EmoPyLab_Manual_METISBr.pdf")
GEN_DATE = datetime.now()

# --------------------------------------------------------------------------- #
# Live codebase facts (computed so the manual never drifts from the source)    #
# --------------------------------------------------------------------------- #


def _count_lines(rel_path: str) -> int:
    """Return the line count of a project file, or 0 if it cannot be read."""
    try:
        with open(os.path.join(_HERE, rel_path), encoding="utf-8") as fh:
            return sum(1 for _ in fh)
    except OSError:
        return 0


def _count_algorithm_dirs() -> int:
    """Return the number of algorithm plugin subfolders under algorithms/."""
    root = os.path.join(_HERE, "algorithms")
    try:
        return sum(
            1
            for name in os.listdir(root)
            if name != "__pycache__"
            and not name.startswith(".")
            and os.path.isdir(os.path.join(root, name))
        )
    except OSError:
        return 0


def _count_py_files(subfolder: str) -> int:
    """Return the number of .py files under a subfolder, excluding __init__ and cache."""
    root = os.path.join(_HERE, subfolder)
    count = 0
    if not os.path.isdir(root):
        return 0
    for dirpath, _, filenames in os.walk(root):
        if "__pycache__" in dirpath:
            continue
        for f in filenames:
            if f.endswith(".py") and f != "__init__.py":
                count += 1
    return count


EMOPYLAB_LINES = _count_lines("emopylab_app.py")
ALGO_COUNT = _count_algorithm_dirs()
PROBLEMS_COUNT = _count_py_files("problems")
OPERATORS_COUNT = _count_py_files("operators")
METRICS_COUNT = _count_py_files("metrics")

_version_globals = runpy.run_path(os.path.join(_HERE, "version.py"))
PACKAGE_VERSION = _version_globals.get("__version__", "1.0.2")

# --------------------------------------------------------------------------- #
# Styles                                                                      #
# --------------------------------------------------------------------------- #
styles = getSampleStyleSheet()


def _style(name, **kw):
    base = kw.pop("parent", styles["Normal"])
    return ParagraphStyle(name, parent=base, **kw)


H1 = _style("H1", fontName="Helvetica-Bold", fontSize=17, textColor=NAVY,
            spaceBefore=18, spaceAfter=10, leading=21)
H2 = _style("H2", fontName="Helvetica-Bold", fontSize=12.5, textColor=BLUE,
            spaceBefore=12, spaceAfter=6, leading=16)
H3 = _style("H3", fontName="Helvetica-Bold", fontSize=10.5, textColor=ACCENT,
            spaceBefore=8, spaceAfter=4, leading=14)
BODY = _style("Body", fontSize=9.6, leading=14.5, textColor=colors.HexColor("#1B2733"),
              alignment=TA_LEFT, spaceAfter=5)
BULLET = _style("Bullet", parent=BODY, leftIndent=14, bulletIndent=4, spaceAfter=2.5)
SMALL = _style("Small", fontSize=8.3, leading=11, textColor=GREY)
CODE = _style("Code", fontName="Courier", fontSize=7.7, leading=10.0,
              textColor=colors.HexColor("#10212E"))
TOC = _style("Toc", fontSize=10.5, leading=19, textColor=NAVY)
COVER_TITLE = _style("CoverTitle", fontName="Helvetica-Bold", fontSize=34,
                     textColor=NAVY, alignment=TA_CENTER, leading=38)
COVER_SUB = _style("CoverSub", fontName="Helvetica", fontSize=13.5,
                   textColor=BLUE, alignment=TA_CENTER, leading=19)
COVER_META = _style("CoverMeta", fontSize=10.5, textColor=GREY,
                    alignment=TA_CENTER, leading=16)
TH = _style("TH", fontName="Helvetica-Bold", fontSize=8.6, textColor=colors.white, leading=11)
TD = _style("TD", fontSize=8.6, leading=11, textColor=colors.HexColor("#1B2733"))
TDB = _style("TDB", fontName="Helvetica-Bold", fontSize=8.6, leading=11,
             textColor=NAVY)

story: list = []


# --------------------------------------------------------------------------- #
# Content helpers                                                             #
# --------------------------------------------------------------------------- #
def h1(text):
    story.append(Paragraph(text, H1))


def h2(text):
    story.append(Paragraph(text, H2))


def h3(text):
    story.append(Paragraph(text, H3))


def p(text):
    story.append(Paragraph(text, BODY))


def bullets(items):
    for it in items:
        story.append(Paragraph(f"&bull;&nbsp;&nbsp;{it}", BULLET))
    story.append(Spacer(1, 4))


def gap(h=6):
    story.append(Spacer(1, h))


def code(text):
    text = text.strip("\n")
    block = Preformatted(text, CODE)
    tbl = Table([[block]], colWidths=[16.4 * cm])
    tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), CODE_BG),
        ("BOX", (0, 0), (-1, -1), 0.6, CODE_BORDER),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    story.append(tbl)
    gap(7)


def table(header, rows, col_widths, first_bold=True):
    data = [[Paragraph(c, TH) for c in header]]
    for r in rows:
        cells = []
        for i, c in enumerate(r):
            st = TDB if (first_bold and i == 0) else TD
            cells.append(Paragraph(str(c), st))
        data.append(cells)
    tbl = Table(data, colWidths=col_widths, repeatRows=1)
    style = [
        ("BACKGROUND", (0, 0), (-1, 0), BLUE),
        ("LINEBELOW", (0, 0), (-1, 0), 0.8, NAVY),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT]),
        ("GRID", (0, 0), (-1, -1), 0.4, CODE_BORDER),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]
    tbl.setStyle(TableStyle(style))
    story.append(tbl)
    gap(8)


# =========================================================================== #
# COVER                                                                       #
# =========================================================================== #
story.append(Spacer(1, 2.8 * cm))
if os.path.exists(LOGO_PATH):
    _logo = Image(LOGO_PATH, width=9.2 * cm, height=4.6 * cm)
    _logo.hAlign = "CENTER"
    story.append(_logo)
else:
    story.append(Paragraph("EmoPyLab", COVER_TITLE))
gap(12)
story.append(Paragraph("User Manual &amp; Technical Reference", COVER_SUB))
gap(2)
story.append(Paragraph("A Tensor-Native, Hardware-Accelerated Laboratory for Multi/Many-Objective "
    "Optimization, High-Throughput Benchmarking, and In-Situ MCDM", COVER_SUB))
story.append(Spacer(1, 2.2 * cm))
rule = Table([[""]], colWidths=[8 * cm], rowHeights=[2])
rule.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), ACCENT)]))
story.append(rule)
story.append(Spacer(1, 0.8 * cm))
story.append(Paragraph(
    "<b>METISBr</b> — Research Group on Multi-Objective and "
    "Many-Objective Evolutionary Optimization", COVER_META))
gap(4)
story.append(Paragraph("UFOP — Federal University of Ouro Preto, Brazil", COVER_META))
story.append(Spacer(1, 1.4 * cm))
story.append(Paragraph(
    f"Package version: {PACKAGE_VERSION} &nbsp;|&nbsp; Manual edition: "
    f"{GEN_DATE.strftime('%Y.%m')} &nbsp;|&nbsp; Generated on "
    f"{GEN_DATE.strftime('%Y-%m-%d')}", COVER_META))
gap(4)
story.append(Paragraph("Contact: santostf+metisbr@ufop.edu.br &nbsp;|&nbsp; https://github.com/METISBR/emopylab", COVER_META))
story.append(PageBreak())

# =========================================================================== #
# TABLE OF CONTENTS                                                            #
# =========================================================================== #
h1("Contents")
toc_items = [
    "1.  Introduction & Core Value Proposition",
    "2.  Requirements and Installation",
    "3.  Tensor-Native Architecture & Design Principles",
    "4.  Repository Directory Structure",
    "5.  Core Engine & Native Tensor Classes (core/)",
    "6.  Extensible Plugin System",
    "7.  Implemented Metaheuristic & Problem Catalog",
    "8.  The Scientific Desktop Workstation (PySide6 GUI)",
    "9.  Cryptographic Reproducibility (SHA-256 Ledger)",
    "10. Non-Parametric Statistical Suite & In-Situ MCDM",
    "11. Authoring Native Optimization Problems",
    "12. Authoring Native Metaheuristics",
    "13. Creating Tensor Quality Metrics",
    "14. Implementing Genetic & Selection Operators",
    "15. High-Throughput Execution & HPC Workflows (Santos Dumont)",
    "16. Troubleshooting & Operational FAQ",
    "Appendix A. 5-Tier Hardware Backend Dispatch (CUDA / MLX / JAX / CuPy / SIMD)",
    "Appendix B. References & Scientific Citations",
]
for t in toc_items:
    story.append(Paragraph(t, TOC))
story.append(PageBreak())

# =========================================================================== #
# 1. INTRODUCTION                                                             #
# =========================================================================== #
h1("1. Introduction & Core Value Proposition")
p("<b>EmoPyLab</b> is an open-source, tensor-native scientific laboratory and visual "
  "decision platform engineered specifically for Evolutionary Multi-Objective (EMO) and "
  "Many-Objective Optimization (MaOP, <i>M &ge; 2</i>). The framework provides a completely "
  "autonomous, high-performance ecosystem designed to overcome the historical memory fragmentation, "
  "interpreter bottlenecks, and object-allocation overheads typical of legacy Python toolchains.")
p("The entire desktop workstation is implemented in "
  f"<font face='Courier'>emopylab_app.py</font> ({EMOPYLAB_LINES:,} lines in PySide6/Qt6), while the "
  "underlying high-performance engine is isolated in <font face='Courier'>core/</font>. "
  "The graphical workstation integrates five dedicated research environments: <b>Quick Test</b>, "
  "<b>Experiment</b>, <b>Results</b>, <b>AI Agent</b> (featuring local in-process Qwen inference), "
  "and <b>Extensibility</b>.")

h2("1.1. Scientific & Technical Highlights")
bullets([
    "<b>Structure-of-Arrays (SoA) Data Model:</b> Eliminates per-individual Python heap objects "
    "in favor of contiguous memory tensors (X &isin; &Ropf;<sup>N&times;D</sup>, "
    "F &isin; &Ropf;<sup>N&times;M</sup>, G &isin; &Ropf;<sup>N&times;K</sup>).",
    "<b>5-Tier Hardware Backend:</b> Transparent, zero-recompilation execution across NVIDIA CUDA, "
    "Apple Silicon MLX (Neural Engine), Google JAX (XLA), CuPy, and optimized multicore CPU C-SIMD.",
    "<b>High-Throughput Many-Objective Sorting:</b> Native deductive Efficient Non-Dominated Sorting "
    "(ENS, &Oscr;(M N &radic;N)) and GPU Boolean Matrix Dominance for massive population regimes (N &ge; 4,000).",
    "<b>Quasi-Monte Carlo Hypervolume (Fast-MC HV):</b> Low-discrepancy Sobol sequence estimator with "
    "bounding-box pruning enabling sub-10 ms evaluation in up to 15-objective spaces.",
    "<b>Local AI Assistant:</b> Zero-cloud code generation and mathematical formulation powered by "
    "the local quantized <font face='Courier'>qwen2.5-0.5b-instruct-q4_k_m.gguf</font> model.",
    "<b>In-Situ MCDM & Back-Mapping:</b> Built-in a posteriori compromise selection (TOPSIS, PROMETHEE II, "
    "Compromise Programming) with deterministic back-mapping to decision space X.",
    "<b>Rigorous Statistical Suite:</b> Non-parametric inferential hypothesis testing (Wilcoxon, Friedman with "
    "Kendall's W, Vargha-Delaney A<sub>12</sub> effect size, and Holm-Bonferroni FWER control).",
    "<b>Cryptographic Provenance:</b> Canonical JSON manifests with SHA-256 digests securing deterministic "
    "reproducibility across supercomputers and local workstations.",
])
story.append(PageBreak())

# =========================================================================== #
# 2. REQUIREMENTS AND INSTALLATION                                            #
# =========================================================================== #
h1("2. Requirements and Installation")
p("EmoPyLab is a pure-Python, standalone tensor-native ecosystem compatible with <b>Python 3.10, 3.11, and 3.12+</b>. "
  "It is packaged under modern PEP 621 conventions (<font face='Courier'>pyproject.toml</font>) and "
  "runs across Linux, macOS (Apple Silicon and Intel), and Windows.")

h2("2.1. Core Dependencies")
table(
    ["Package", "Version Constraint", "Role in EmoPyLab"],
    [
        ["numpy", "&gt;=2.3, &lt;2.5", "Core contiguous numerical tensor arrays"],
        ["numba", "&gt;=0.66, &lt;0.67", "JIT-compiled CPU kernels for fast NDS and dominance"],
        ["scipy", "&gt;=1.16, &lt;2.0", "Statistical test suite (Wilcoxon, Friedman, A12)"],
        ["matplotlib", "&gt;=3.10, &lt;4.0", "Scientific Pareto fronts and convergence plotting"],
        ["psutil", "&gt;=7.2, &lt;8.0", "Hardware telemetry and memory bandwidth monitoring"],
    ],
    [3.2 * cm, 4.4 * cm, 8.8 * cm],
)

h2("2.2. Optional Acceleration & Extension Groups")
table(
    ["Group", "Packages", "Target Acceleration"],
    [
        ["gui", "PySide6, qt-material, qtawesome", "Desktop workstation graphical interface"],
        ["torch", "torch", "NVIDIA CUDA / Apple MPS acceleration (Tier 1)"],
        ["cupy", "cupy-cuda12x", "NVIDIA CUDA drop-in array operations (Tier 2)"],
        ["jax", "jax, jaxlib", "GPU/TPU XLA compilation and 3D vmap tensor batching (Tier 3)"],
        ["mlx", "mlx, mlx-lm", "Apple Silicon M-series GPU & Neural Engine (Tier 4)"],
        ["llm", "llama-cpp-python", "In-process GGUF local model inference (Qwen 2.5)"],
        ["dev", "pytest, scikit-learn", "Automated contract verification and surrogate models"],
    ],
    [2.8 * cm, 5.8 * cm, 7.8 * cm],
)

h2("2.3. Clean Installation Commands")
code(
    "# 1. Clone clean repository\n"
    "git clone https://github.com/METISBR/emopylab.git\n"
    "cd emopylab\n\n"
    "# 2. Install base core dependencies\n"
    "pip install .\n\n"
    "# 3. Install acceleration suite\n"
    "pip install .[gui,llm,jax]        # On Linux / Windows (JAX XLA)\n"
    "pip install .[gui,cupy]           # On Linux / Windows (NVIDIA CUDA 12.x)\n"
    "pip install .[gui,mlx,llm]        # On Apple Silicon macOS"
)
# 3. PROJECT ARCHITECTURE                                                     #
# =========================================================================== #
h1("3. Tensor-Native Architecture & Design Principles")
p("Traditional evolutionary computation toolchains in Python rely on an <i>Array-of-Structures</i> "
  "(AoS) model, where each individual is represented as a separate Python heap object containing "
  "decision vectors, objectives, and metadata. In large-scale campaigns (such as millions of evaluations "
  "across hundreds of runs), this model triggers intense memory fragmentation, cache thrashing, and "
  "heavy serialization overhead between subprocesses.")
p("EmoPyLab solves this architectural limitation through a pure <b>Structure-of-Arrays (SoA)</b> "
  "columnar data model. Populations and problems are expressed as contiguous 2D and 3D tensors:")

code(
    "# Tensor Population Memory Representation\n"
    "X in R^(N x D)   # Continuous / discrete decision variable matrix\n"
    "F in R^(N x M)   # Objective function evaluation matrix\n"
    "G in R^(N x K)   # Inequality constraint violation matrix\n"
    "CV in R^(N)      # Total aggregated constraint violation vector"
)

h2("3.1. Unified 5-Tier Backend Dispatch Engine")
p("The computational kernel (<font face='Courier'>core/tensor/backend.py</font>) auto-detects "
  "available accelerators with transparent fallback at runtime without requiring code modification:")
bullets([
    "<b>Tier 1 (NVIDIA PyTorch CUDA / MPS):</b> Highest throughput for large deep surrogate models and neural architectures.",
    "<b>Tier 2 (NVIDIA CuPy):</b> Direct CUDA array manipulation providing drop-in GPU acceleration for structured array ops.",
    "<b>Tier 3 (Google JAX XLA):</b> Fully vectorizable functional kernels with <font face='Courier'>jax.jit</font> and <font face='Courier'>jax.vmap</font> across parallel seeds.",
    "<b>Tier 4 (Apple MLX):</b> Native Apple Silicon execution operating directly on unified memory, avoiding PCIe bus transfers.",
    "<b>Tier 5 (Vectorized CPU C-SIMD):</b> OpenBLAS/MKL fallback with AVX-512 vectorization and thread oversubscription protection.",
])

h2("3.2. Subprocess Worker & Memory-Safe Ring Buffer")
p("Execution isolation is maintained by <font face='Courier'>emopylab_process_worker.py</font>. "
  "Optimization runs execute in dedicated worker processes communicating through memory-mapped shared buffers, "
  "preventing Python Global Interpreter Lock (GIL) contention and bounding GUI RAM consumption strictly below 150 MB.")
story.append(PageBreak())

# =========================================================================== #
# 4. DIRECTORY STRUCTURE                                                       #
# =========================================================================== #
h1("4. Repository Directory Structure")
code(
    "emopylab/\n"
    f"|-- emopylab_app.py              # Main PySide6 Scientific Desktop GUI ({EMOPYLAB_LINES:,} lines)\n"
    "|-- EmoPyLab.py                  # Entrypoint launcher script\n"
    "|-- emopylab_metadata.py         # Algorithm, problem, and metric catalog descriptors\n"
    "|-- emopylab_process_worker.py   # Isolated subprocess execution worker\n"
    "|-- optimize.py                  # Standalone top-level minimization entrypoint\n"
    "|-- version.py / __init__.py     # Version metadata (v1.0.2)\n"
    "|-- pyproject.toml               # Modern PEP 621 package and build specification\n"
    "|-- requirements.txt             # Pip dependency manifest\n"
    "|-- CITATION.bib                 # Formal BibTeX citation entry\n"
    "|-- models/                      # Local AI model directory\n"
    "|   `-- qwen2.5-0.5b-instruct-q4_k_m.gguf  # 491 MB GGUF model tracked via Git LFS\n"
    "|-- core/                        # Standalone scientific core (Pure Tensors)\n"
    "|   |-- algorithm.py             #   Native population-based Algorithm base class\n"
    "|   |-- problem.py / variable.py #   Optimization Problem contract & variable definitions\n"
    "|   |-- population.py            #   Contiguous Population data structure\n"
    "|   |-- tensor/                  #   5-tier tensor backends (CUDA, MLX, JAX, CuPy, SIMD)\n"
    "|   |-- nds/                     #   Deductive ENS & GPU Boolean matrix sorting\n"
    "|   |-- mcdm/decision.py         #   TOPSIS, PROMETHEE II & Compromise Programming\n"
    "|   |-- analysis/stat_tests_advanced.py # Wilcoxon, Friedman, A12, DeltaP, IGD+\n"
    "|   |-- execution/reproducibility.py    # Canonical SHA-256 manifests & seed plans\n"
    "|   `-- llm/                     #   Local Qwen client and AST-safe formulation\n"
    f"|-- algorithms/                  # {ALGO_COUNT} metaheuristic solvers (NSGA-II, NSGA-III, MOEA/D...)\n"
    f"|-- problems/                    # {PROBLEMS_COUNT} benchmark problems (Single, Multi, Many-Objective)\n"
    f"|-- operators/                   # {OPERATORS_COUNT} genetic operators (crossover, mutation, repair...)\n"
    f"|-- metrics/                     # {METRICS_COUNT} tensor quality indicators (Fast-MC HV, IGD+, GD+)\n"
    "|-- util/                        # Utility shims, dominator logic, reference directions\n"
    "`-- tests/                       # Complete regression and benchmark test suite\n"
)
story.append(PageBreak())

# =========================================================================== #
# 5. CORE ENGINE & NATIVE CLASSES                                             #
# =========================================================================== #
h1("5. Core Engine & Native Tensor Classes (core/)")
p("EmoPyLab operates completely independently of external optimization libraries. "
  "Its base classes are located in <font face='Courier'>core/</font> and provide native "
  "vectorized interfaces designed for high throughput.")

h2("5.1. core/problem.py — Native Problem Base Class")
p("Every optimization problem derives from <font face='Courier'>core.problem.Problem</font>. "
  "Evaluation operates directly on 2D NumPy/JAX arrays:")
code(
    "from core.problem import Problem\n"
    "import numpy as np\n\n"
    "class MyProblem(Problem):\n"
    "    def __init__(self, n_var=30, n_obj=2, **kwargs):\n"
    "        super().__init__(\n"
    "            n_var=n_var, n_obj=n_obj, n_ieq_constr=0,\n"
    "            xl=np.zeros(n_var), xu=np.ones(n_var), **kwargs\n"
    "        )\n\n"
    "    def _evaluate(self, X: np.ndarray, out: dict, *args, **kwargs):\n"
    "        f1 = X[:, 0]\n"
    "        g = 1.0 + 9.0 * np.sum(X[:, 1:], axis=1) / (self.n_var - 1)\n"
    "        f2 = g * (1.0 - np.sqrt(f1 / g))\n"
    "        out['F'] = np.column_stack([f1, f2])"
)

h2("5.2. core/algorithm.py — Native Algorithm Base Class")
p("The base <font face='Courier'>Algorithm</font> class provides automatic hardware accelerator "
  "binding, deterministic seeding, callback hooks, and survival selection:")
code(
    "from core.algorithm import Algorithm\n\n"
    "class MyAlgorithm(Algorithm):\n"
    "    def __init__(self, pop_size: int = 100, use_gpu: bool = False, **kwargs):\n"
    "        super().__init__(use_gpu=use_gpu, **kwargs)\n"
    "        self.pop_size = pop_size\n\n"
    "    def _step(self):\n"
    "        # Vectorized mating, evaluation, and survival loop\n"
    "        offspring = self.mating.do(self.problem, self.pop, self.pop_size)\n"
    "        self.evaluator.eval(self.problem, offspring)\n"
    "        self.pop = self.survival.do(self.problem, self.pop, offspring, self.pop_size)"
)
story.append(PageBreak())

# =========================================================================== #
# 6. EXTENSIBLE PLUGIN SYSTEM                                                 #
# =========================================================================== #
h1("6. Extensible Plugin System")
p("EmoPyLab features a zero-registration plugin discovery architecture. "
  "Any new algorithm, benchmark problem, genetic operator, or metric added to its corresponding "
  "directory is dynamically discovered, inspected, and validated at startup.")

h2("6.1. Discovery Pipeline")
bullets([
    "<b>algorithms/</b>: Auto-scans subfolders containing <font face='Courier'>ALGORITHM_FLAGS</font> "
    "and typed class constructors.",
    "<b>problems/</b>: Auto-categorizes into <font face='Courier'>single/</font>, "
    "<font face='Courier'>multi/</font>, and <font face='Courier'>many/</font> suites.",
    "<b>operators/</b>: Auto-discovers genetic components by category: crossover, mutation, sampling, selection, repair.",
    "<b>metrics/</b>: Discovers indicator functions exposing <font face='Courier'>create_metric(context)</font>.",
])

h2("6.2. Registry Dataclasses")
table(
    ["Dataclass", "Key Fields", "Operational Purpose"],
    [
        ["AlgorithmSpec", "id, name, source, module, factory, flags", "Solver record with scope tags (single, multi, many)"],
        ["ProblemSpec", "id, name, source, module, factory, n_var, n_obj", "Problem record with default problem dimensions"],
        ["MetricSpec", "id, name, source, module, factory", "Quality indicator specification and reference requirements"],
        ["OperatorSpec", "id, name, source, category, module, class_name", "Genetic operator descriptor grouped by category"],
    ],
    [3.0 * cm, 5.2 * cm, 8.2 * cm],
)
story.append(PageBreak())

# =========================================================================== #
# 7. IMPLEMENTED CATALOG                                                      #
# =========================================================================== #
h1("7. Implemented Metaheuristic & Problem Catalog")
p(f"Snapshot verified against the repository on {GEN_DATE.strftime('%Y-%m-%d')}. "
  "All components run on the native EmoPyLab execution engine.")

h2("7.1. Quantitative Overview")
table(
    ["Domain", "Verified Count", "Architecture & Taxonomy"],
    [
        ["Algorithms", f"{ALGO_COUNT} Solvers", "Dominance, Decomposition, Indicator, Reference-Vector, SAEA, LLM-based"],
        ["Problems", f"{PROBLEMS_COUNT} Formulations", "Single (7), Multi (33 suites), Many-objective (9 suites: MaF, DTLZ, WFG)"],
        ["Operators", f"{OPERATORS_COUNT} Modules", "Crossover (24), Mutation (12), Repair (12), Sampling (4), Selection (4)"],
        ["Metrics", f"{METRICS_COUNT} Modules", "Fast-MC HV (Sobol), R2 Indicator, IGD+, GD+, Spacing, Averaged Hausdorff (DeltaP)"],
    ],
    [3.2 * cm, 3.8 * cm, 9.4 * cm],
)

h2("7.2. Representative Algorithm Families")
bullets([
    "<b>Dominance-Based:</b> NSGA-II, SPEA2, BiGE, PESA-II, Micro-GA, Knock-Out EA, E-NSGA-II.",
    "<b>Decomposition-Based:</b> MOEA/D, MOEA/D-DE, MOEA/D-DRA, C-MOEA/D, RPEA, EAG-MOEA/D.",
    "<b>Reference-Vector / Angle:</b> NSGA-III, RVEA, VAEA, MaOEA-CSS, LARC-NSGA3, GCS-MaOEA.",
    "<b>Indicator-Based:</b> IBEA, SMS-EMOA, HypE, MOMBI-II, SRA, AR-MOEA.",
    "<b>Surrogate-Assisted (SAEA):</b> ParEGO, K-RVEA, CSEA, AS-SAEA, SAEA-TL2M, MOEA/D-EGO.",
    "<b>LLM Meta-Controlled:</b> MaACO (Adaptive Ant Colony), LARC-NSGA3 (Reinforced niching).",
])
story.append(PageBreak())

# =========================================================================== #
# 8. THE SCIENTIFIC DESKTOP WORKSTATION                                       #
# =========================================================================== #
h1("8. The Scientific Desktop Workstation (PySide6 GUI)")
p("The graphical interface is built with high-contrast typography, dark/light scientific themes, "
  "and a decoupled event loop guaranteeing responsiveness during massive simulations.")

h2("8.1. The Five Operational Workspaces")
bullets([
    "<b>Quick Test:</b> Rapid prototyping suite for single-run diagnostics, offering real-time 2D/3D "
    "Pareto front projection, camera rotation, convergence curves, and hardware telemetry.",
    "<b>Experiment:</b> High-throughput batch benchmarking orchestration. Allows matrix cross-testing "
    "across dozens of algorithms, test problems, seed replications, and evaluation budgets.",
    "<b>Results:</b> Visual analytics suite offering aggregated performance tables, non-parametric "
    "statistical tests, convergence trajectories, and high-resolution CSV/LaTeX exporters.",
    "<b>AI Agent:</b> Local mathematical formulation assistant powered by in-process Qwen 2.5 GGUF. "
    "Converts plain English into validated Python/JAX problem plugins with strict AST security auditing.",
    "<b>Extensibility:</b> Interactive explorer for manual plugin creation, template scaffolding, "
    "and catalog taxonomy inspection.",
])

h2("8.2. Local AI Assistant & AST Security Architecture")
p("The AI Agent operates 100% locally and offline using <font face='Courier'>qwen2.5-0.5b-instruct-q4_k_m.gguf</font>. "
  "Before any generated Python code is saved or loaded, it must pass a strict <b>Abstract Syntax Tree (AST)</b> "
  "security gate (<font face='Courier'>core/llm/formulation.py</font>):")
bullets([
    "Prohibits dangerous calls (<font face='Courier'>eval, exec, open, __import__, os.system, subprocess</font>).",
    "Restricts imports to authorized scientific libraries (<font face='Courier'>numpy, jax, typing, math, core</font>).",
    "Verifies numerical correctness and CPU/JAX output equivalence on small randomized input batches.",
])
story.append(PageBreak())

# =========================================================================== #
# 9. REPRODUCIBILITY & SHA-256 MANIFESTS                                      #
# =========================================================================== #
h1("9. Cryptographic Reproducibility (SHA-256 Ledger)")
p("To ensure compliance with FAIR scientific principles, EmoPyLab integrates an automated "
  "cryptographic provenance engine (<font face='Courier'>core/execution/reproducibility.py</font>). "
  "Every experimental campaign generates an immutable canonical JSON sidecar manifest containing:")
bullets([
    "<b>Deterministic Seed Schedules:</b> Canonical arithmetic seed progression (S<sub>k</sub> = 1000 + 17k) "
    "guaranteeing reproducible pseudo-random streams.",
    "<b>Hardware & Backend Fingerprint:</b> CPU model, active GPU/NPU accelerator tier, and thread counts.",
    "<b>Software Environment:</b> Python version, library commit hashes, and compiler flags.",
    "<b>Cryptographic SHA-256 Hash:</b> Digest computed over canonicalized parameters and final Pareto front coordinates.",
])
story.append(PageBreak())

# =========================================================================== #
# 10. STATISTICAL ANALYSIS & IN-SITU MCDM                                     #
# =========================================================================== #
h1("10. Non-Parametric Statistical Suite & In-Situ MCDM")
h2("10.1. Advanced Statistical Testing")
p("EmoPyLab integrates automated inferential statistics (<font face='Courier'>core/analysis/stat_tests_advanced.py</font>):")
bullets([
    "<b>Wilcoxon Signed-Rank Test:</b> Non-parametric pairwise comparison with significance threshold &alpha; = 0.05.",
    "<b>Friedman Global Ranking & Kendall's W:</b> Multiple-solver rank ordering with concordance analysis.",
    "<b>Vargha-Delaney A<sub>12</sub> Effect Size:</b> Non-parametric stochastic superiority measure "
    "(negligible, small, medium, large).",
    "<b>Holm-Bonferroni Post-Hoc Correction:</b> Step-down sequential adjustment controlling family-wise error rate (FWER).",
])

h2("10.2. In-Situ Multi-Criteria Decision Making (MCDM)")
p("Pareto optimization yields hundreds of non-dominated solutions. EmoPyLab bridges search and practical "
  "deployment by embedding native decision support (<font face='Courier'>core/mcdm/decision.py</font>):")
table(
    ["Method", "Mathematical Foundation", "Decision-Making Context"],
    [
        ["TOPSIS", "Euclidean distance to ideal best (min) and worst (max)", "Balanced compromise with user-defined attribute weights"],
        ["PROMETHEE II", "Net outranking flows (&Phi; = &Phi;<sup>+</sup> - &Phi;<sup>-</sup>)", "Robust pairwise outranking among non-dominated points"],
        ["Compromise Prog.", "Weighted Chebyshev / L<sub>p</sub> metric norm", "Target-oriented selection under strict preference vectors"],
    ],
    [3.2 * cm, 6.2 * cm, 7.0 * cm],
)
p("All MCDM methods feature <b>Decision-Space Back-Mapping</b>, immediately returning the physical decision "
  "vector X<sup>*</sup> corresponding to the chosen compromise objective trade-off F<sup>*</sup>.")
story.append(PageBreak())

# =========================================================================== #
# 11-14. HOW TO CREATE CUSTOM ARTIFACTS                                       #
# =========================================================================== #
h1("11. Authoring Custom Problems, Algorithms & Metrics")

h2("11.1. Authoring a Problem (problems/multi/ or problems/many/)")
p("Implement a subclass of <font face='Courier'>core.problem.Problem</font>:")
code(
    "from core.problem import Problem\n"
    "import numpy as np\n\n"
    "class CustomZDT(Problem):\n"
    "    def __init__(self, n_var=30, **kwargs):\n"
    "        super().__init__(n_var=n_var, n_obj=2, xl=0.0, xu=1.0, **kwargs)\n\n"
    "    def _evaluate(self, X, out, *args, **kwargs):\n"
    "        f1 = X[:, 0]\n"
    "        g = 1.0 + 9.0 * np.sum(X[:, 1:], axis=1) / (self.n_var - 1)\n"
    "        f2 = g * (1.0 - np.power(f1 / g, 2))\n"
    "        out['F'] = np.column_stack([f1, f2])"
)
h2("11.2. Authoring an Algorithm Plugin (algorithms/)")
p("EmoPyLab discovers algorithms dynamically with zero manual registration. "
  "To create a new metaheuristic solver compatible with the desktop GUI, batch runner, "
  "and 5-tier hardware acceleration, create a module inside <font face='Courier'>algorithms/my_algorithm.py</font>:")
code(
    "# algorithms/my_algorithm.py\n"
    "import numpy as np\n"
    "from core.algorithm import Algorithm\n"
    "from core.population import Population\n"
    "from core.nds.ens import efficient_non_dominated_sort\n\n"
    "# Scope tags for GUI auto-categorization: 'single', 'multi', or 'many'\n"
    "ALGORITHM_FLAGS = {'multi', 'many', 'gpu_compatible'}\n\n"
    "class CustomGeneticAlgorithm(Algorithm):\n"
    "    def __init__(self, pop_size: int = 100, mutation_rate: float = 0.1, **kwargs):\n"
    "        super().__init__(**kwargs)\n"
    "        self.pop_size = pop_size\n"
    "        self.mutation_rate = mutation_rate\n\n"
    "    def _initialize_advance(self, infills=None):\n"
    "        # Initialize population using uniform random continuous variables\n"
    "        X = np.random.uniform(self.problem.xl, self.problem.xu, size=(self.pop_size, self.problem.n_var))\n"
    "        self.pop = Population.new('X', X)\n"
    "        self.evaluator.eval(self.problem, self.pop)\n\n"
    "    def _advance(self, infills=None):\n"
    "        # 1. Selection and Variation\n"
    "        parents_idx = np.random.randint(0, len(self.pop), size=self.pop_size)\n"
    "        X_offspring = self.pop.get('X')[parents_idx] + np.random.normal(0, 0.05, size=(self.pop_size, self.problem.n_var))\n"
    "        X_offspring = np.clip(X_offspring, self.problem.xl, self.problem.xu)\n\n"
    "        # 2. Evaluate offspring population\n"
    "        offspring = Population.new('X', X_offspring)\n"
    "        self.evaluator.eval(self.problem, offspring)\n\n"
    "        # 3. Environmental Survival Selection via ENS Non-Dominated Sorting\n"
    "        combined = Population.merge(self.pop, offspring)\n"
    "        F = combined.get('F')\n"
    "        fronts = efficient_non_dominated_sort(F)\n"
    "        survivors_idx = [idx for front in fronts for idx in front][:self.pop_size]\n"
    "        self.pop = combined[survivors_idx]\n"
)

h2("11.3. Authoring a Quality Metric (metrics/)")
p("To implement a native quality indicator, derive from <font face='Courier'>metrics.indicators.Indicator</font> "
  "or expose a function <font face='Courier'>r2_indicator(F, PF, W)</font>:")
code(
    "# metrics/custom_metric.py\n"
    "import numpy as np\n\n"
    "def create_metric(context: dict):\n"
    "    ref_front = context.get('reference_front')\n"
    "    def _evaluator(front: np.ndarray) -> float:\n"
    "        # Compute distance metric to ref_front\n"
    "        return float(np.mean(np.min(np.linalg.norm(front[:, None] - ref_front, axis=2), axis=1)))\n"
    "    return _evaluator"
)
story.append(PageBreak())

# =========================================================================== #
# 15. HIGH-THROUGHPUT & HPC WORKFLOWS                                         #
# =========================================================================== #
h1("15. High-Throughput Execution & HPC Workflows")
p("EmoPyLab is engineered for seamless transition from local workstations to high-performance supercomputers. "
  "The framework was empirically validated across <b>5,970 independent runs</b> on the Santos Dumont Supercomputer "
  "(Bull Sequana X1000 GPU partition, LNCC/MCTI, Brazil).")

h2("15.1. Headless CLI Optimization")
code(
    "# Run single optimization headlessly\n"
    "emopylab optimize --algorithm NSGA3 --problem DTLZ2 --n-obj 5 --seed 42\n\n"
    "# Launch automated matrix benchmark from JSON specification\n"
    "emopylab benchmark --config benchmark_matrix.json --workers 36 --output results/"
)

h2("15.2. Slurm Supercomputing Deployment")
p("Cluster scripts (<font face='Courier'>sbatch_sdumont_benchmark.sh</font>) prevent thread oversubscription "
  "by strictly setting BLAS thread pools (<font face='Courier'>OPENBLAS_NUM_THREADS=1</font>, "
  "<font face='Courier'>MKL_NUM_THREADS=1</font>) prior to dispatching across massive GPU/CPU worker pools.")
story.append(PageBreak())

# =========================================================================== #
# APPENDIX & CITATION                                                         #
# =========================================================================== #
h1("Appendix A. Hardware Acceleration Backend Architecture")
table(
    ["Backend Tier", "Module Dispatcher", "Platform Support", "Optimization Focus"],
    [
        ["Tier 1: CUDA / MPS", "core.tensor.backend (torch)", "NVIDIA GPUs / Apple Silicon MPS", "Deep surrogates & high VRAM batches"],
        ["Tier 2: CuPy", "core.tensor.backend (cupy)", "NVIDIA CUDA Architecture", "Drop-in structured array acceleration"],
        ["Tier 3: Google JAX", "core.tensor.backend (jax)", "Linux, macOS, Windows (XLA)", "Vectorized multi-seed parallel vmap"],
        ["Tier 4: Apple MLX", "core.tensor.backend (mlx)", "macOS Apple Silicon (ARM64)", "Zero-copy unified memory & Neural Engine"],
        ["Tier 5: CPU C-SIMD", "core.tensor.backend (numpy)", "Universal x86_64 / ARM64", "Optimized OpenBLAS/MKL AVX-512 fallback"],
    ],
    [2.8 * cm, 3.8 * cm, 4.2 * cm, 5.6 * cm],
)

gap(16)
h1("Appendix B. References & Citation")
p("If you utilize EmoPyLab in academic or industrial research, please cite:")
code(
    "@article{santos2026emopylab,\n"
    "  title   = {{EmoPyLab: A Tensor-Native, Hardware-Accelerated Laboratory for High-Throughput Benchmarking and Decision-Making in Many-Objective Optimization}},\n"
    "  author  = {Santos, Thiago and Xavier, Sebasti{\\~a}o},\n"
    "  journal = {Swarm and Evolutionary Computation (under review)},\n"
    "  year    = {2026},\n"
    "  url     = {https://github.com/METISBR/emopylab}\n"
    "}\n\n"
    "@software{Santos_EmoPyLab_2026,\n"
    "  author  = {Santos, Thiago and Xavier, Sebasti{\\~a}o},\n"
    "  title   = {{EmoPyLab: A High-Throughput Benchmarking Ecosystem and Open-Source Decision Platform for Evolutionary Many-Objective Optimization}},\n"
    "  year    = {2026},\n"
    "  version = {1.0.2},\n"
    "  url     = {https://github.com/METISBR/emopylab},\n"
    "  license = {MIT}\n"
    "}"
)
gap(12)
story.append(Paragraph("Copyright &copy; 2026 Professor Thiago Santos, METISBr Research Group, Federal University of Ouro Preto (UFOP), Brazil. Released under the MIT License.", SMALL))


# =========================================================================== #
# Document assembly with header/footer                                        #
# =========================================================================== #
def _decorate(canvas, doc):
    canvas.saveState()
    w, h = A4
    if doc.page > 1:
        canvas.setFillColor(NAVY)
        canvas.setFont("Helvetica-Bold", 8)
        canvas.drawString(2 * cm, h - 1.15 * cm, "EmoPyLab — User Manual & Technical Reference")
        canvas.setStrokeColor(ACCENT)
        canvas.setLineWidth(0.6)
        canvas.line(2 * cm, h - 1.30 * cm, w - 2 * cm, h - 1.30 * cm)
        canvas.setFillColor(GREY)
        canvas.setFont("Helvetica", 8)
        canvas.drawRightString(w - 2 * cm, 1.05 * cm, f"Page {doc.page}")
        canvas.drawString(2 * cm, 1.05 * cm, "UFOP / METISBr — Brazil")
    canvas.restoreState()


def build():
    doc = BaseDocTemplate(
        OUTFILE, pagesize=A4,
        leftMargin=2 * cm, rightMargin=2 * cm,
        topMargin=1.7 * cm, bottomMargin=1.7 * cm,
        title="EmoPyLab — User Manual & Technical Reference",
        author="Professor Thiago Santos (METISBr / UFOP)",
    )
    frame = Frame(doc.leftMargin, doc.bottomMargin,
                  doc.width, doc.height, id="main")
    doc.addPageTemplates([PageTemplate(id="all", frames=[frame],
                                       onPage=_decorate)])
    doc.build(story)
    print(f"OK -> {OUTFILE}")


if __name__ == "__main__":
    build()
