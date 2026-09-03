<p align="center"><img src="emopylab.png" alt="EmoPyLab Logo" width="320"/></p>

# EmoPyLab

**EmoPyLab** is an open-source, tensor-native, hardware-accelerated scientific framework and visual decision platform for Evolutionary Multi-Objective (EMO) and Many-Objective Optimization (MaOP, $M \ge 2$).

[![Research Group](https://img.shields.io/badge/Research_Group-METISBr-blue.svg)](https://metisbr.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://github.com/METISBR/emopylab/blob/main/LICENSE)

> **Author:** Professor Thiago Santos — Federal University of Ouro Preto (UFOP), Brazil  
> **Group:** [METISBr — Multi-Objective and Many-Objective Optimization Research](https://metisbr.com)  
> **Website:** [metisbr.com](https://metisbr.com)  
> **Contact:** `santostf+metisbr@ufop.edu.br`

---

## Table of Contents


1. [Overview & Core Value Proposition](#overview--core-value-proposition)
2. [Key Differentiators & Scientific Highlights](#key-differentiators--scientific-highlights)
3. [Architecture & Design Principles](#architecture--design-principles)
4. [Hardware Acceleration & Benchmarks](#hardware-acceleration--benchmarks)
5. [Installation & Requirements](#installation--requirements)
6. [Quickstart Guide (Headless & Python API)](#quickstart-guide-headless--python-api)
7. [Desktop Workstation GUI](#desktop-workstation-gui)
8. [Multi-Criteria Decision Making (MCDM)](#multi-criteria-decision-making-mcdm)
9. [Statistical Inference & LaTeX Table Synthesis](#statistical-inference--latex-table-synthesis)
10. [Algorithm & Problem Catalog Taxonomy](#algorithm--problem-catalog-taxonomy)
11. [Testing & Quality Assurance](#testing--quality-assurance)
12. [Citation](#citation)
13. [License](#license)

---

## Overview & Core Value Proposition

Empirical research in Multi- and Many-Objective Optimization ($M \ge 2$) has historically been constrained by computational bottlenecks: CPU-bound object allocation overheads, quadratic non-dominated sorting latencies, and exponential (sharp-P-hard, $\mathcal{O}(N^M)$) metric calculations.

**EmoPyLab** addresses these bottlenecks by establishing an end-to-end tensor-native optimization ecosystem in Python:

* **Columnar Structure-of-Arrays (SoA):** Eliminates per-individual Python heap objects in favor of contiguous memory tensors ($X \in \mathbb{R}^{N \times D}$, $F \in \mathbb{R}^{N \times M}$, $G \in \mathbb{R}^{N \times K}$).
* **Hardware Acceleration with Transparent Fallback:** Native execution on NVIDIA CUDA, AMD ROCm, Apple Silicon Metal (JAX/MLX), and vectorized CPU SIMD (AVX-512).
* **High-Throughput Many-Objective Sorting:** Deductive Efficient Non-Dominated Sorting (ENS, $\mathcal{O}(M N \sqrt{N})$) and Bitwise GPU Boolean Matrix Sorting.
* **Scalable Metric Evaluation:** JIT-compiled tensor indicators (IGD+, GD+, Spacing) and dynamic sample-pruned quasi-Monte Carlo Hypervolume (Fast-MC) for $M > 4$.
* **Dual Modality:** Decoupled headless CLI for supercomputing clusters (Slurm/HPC) and a modern, memory-safe Desktop Workstation (PySide6 / Qt6).
* **Integrated Decision Support (MCDM):** In-situ compromise solution extraction (TOPSIS, PROMETHEE II, Compromise Programming) with SHA-256 cryptographic provenance.

---

## Key Differentiators & Scientific Highlights

| Feature / Dimension | Traditional Scripting Toolchains | Monolithic Legacy Platforms | **EmoPyLab (This Framework)** |
|---|---|---|---|
| **Memory Model** | Row-major Python objects (High GC overhead) | Proprietary matrix workspace | **Columnar SoA Tensor Matrices** |
| **Hardware Execution** | Single-threaded CPU / Basic Multiprocessing | CPU-only | **Heterogeneous GPU/NPU & CPU SIMD** |
| **Non-Dominated Sort** | Fast NDS ($\mathcal{O}(M N^2)$) | Standard NDS | **ENS ($\mathcal{O}(M N \sqrt{N})$) & GPU Bitwise** |
| **Hypervolume ($M > 4$)** | Timeout / Combinatorial Explosion | Slow approximation | **Quasi-Monte Carlo Fast-MC ($< 60$ ms)** |
| **Campaign Execution** | Sequential iteration | Manual scripts | **Parallel Multi-Seed Vectorization (`vmap`)** |
| **Decision Support** | External / Manual CSV parsing | Basic manual selection | **Embedded TOPSIS & PROMETHEE II** |
| **Reproducibility** | Manual seed logging | Session export | **Immutable SHA-256 Cryptographic Ledger** |
| **Desktop Stability** | Memory leaks during long runs | High RAM consumption | **Ring-Buffer Decoupled UI ($< 150$ MB RAM)** |

---

## Architecture & Design Principles

```
 ┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                       EMOPYLAB TENSOR-NATIVE CORE ARCHITECTURE                          │
 ├─────────────────────────────────────────────────────────────────────────────────────────┤
 │                                                                                         │
 │   ┌────────────────────────── GPU / NPU / CPU TENSOR ENGINE ─────────────────────────┐ │
 │   │                                                                                   │ │
 │   │   Population Tensor X [N, D] ──[jax.jit / vmap]──> Objectives Tensor F [N, M]     │ │
 │   │               ▲                                              │                    │ │
 │   │               │                                              ▼                    │ │
 │   │   ENS / Mating Selection <──[Bitwise GPU Sort]<── Tensor Recombination / Mutation │ │
 │   │                                                                                   │ │
 │   └───────────────────────────────────────────────────────────────────────────────────┘ │
 │                                           │                                             │
 │                   ┌───────────────────────┴───────────────────────┐                     │
 │                   ▼                                               ▼                     │
 │   ┌───────────────────────────────┐               ┌───────────────────────────────┐     │
 │   │   HEADLESS CLI / HPC RUNNER   │               │   DESKTOP WORKSTATION (GUI)   │     │
 │   │   • Parallel multi-seed `vmap`│               │   • Asynchronous QThreadPool  │     │
 │   │   • Slurm batch integration   │               │   • Ring Buffer (Zero-Leak)   │     │
 │   │   • Automated LaTeX synthesis │               │   • Glassmorphic HUD Tooltips │     │
 │   │   • SHA-256 sidecar archives  │               │   • MCDM Decision Studio      │     │
 │   └───────────────────────────────┘               └───────────────────────────────┘     │
 └─────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## Hardware Acceleration & Benchmarks

Empirical scalability evaluations demonstrate order-of-magnitude throughput gains across objective and population dimensions:

### 1. Distance Metric Scalability (IGD+)

| Dimension ($M$) | Population ($N$) | CPU Baseline | EmoPyLab Vectorized Kernel | **Speedup** |
|:---:|:---:|:---:|:---:|:---:|
| **$M = 3$** | $N = 100$ | 1.55 ms | 0.61 ms | **2.55×** |
| **$M = 3$** | $N = 500$ | 15.19 ms | 1.30 ms | **11.68×** |
| **$M = 3$** | $N = 1000$ | 72.10 ms | 3.77 ms | **19.14×** |
| **$M = 5$** | $N = 1000$ | 93.91 ms | 4.95 ms | **18.97×** |
| **$M = 8$** | $N = 1000$ | 138.70 ms | 6.02 ms | **23.04×** |
| **$M = 10$** | $N = 1000$ | 169.07 ms | 8.97 ms | **18.86×** |
| **$M = 15$** | $N = 1000$ | 257.46 ms | 13.84 ms | **18.60×** |

### 2. High-Dimensional Hypervolume Scaling ($M > 4$)

| Dimension ($M$) | Sample Size ($N$) | Exact Algorithm | EmoPyLab Fast-MC Engine | Status |
|:---:|:---:|:---:|:---:|:---:|
| **$M = 3$** | $N = 50$ | 0.21 ms | 0.21 ms | Exact Sweep-line |
| **$M = 5$** | $N = 50$ | 0.08 ms | 35.39 ms | Exact Decomposition |
| **$M = 6$** | $N = 50$ | *Timeout (>10 min)* | **36.40 ms** | Fast-MC (Sobol Sample Pruning) |
| **$M = 8$** | $N = 50$ | *Timeout (>10 min)* | **48.11 ms** | Fast-MC (Sobol Sample Pruning) |
| **$M = 10$** | $N = 50$ | *Timeout (>10 min)* | **57.04 ms** | Fast-MC (Sobol Sample Pruning) |

### 3. Non-Dominated Sorting Throughput

| Dimension ($M$) | Population ($N$) | Classic FNDS | EmoPyLab Deductive ENS | **Speedup** |
|:---:|:---:|:---:|:---:|:---:|
| **$M = 3$** | $N = 1000$ | 3.17 ms | 0.54 ms | **5.83×** |
| **$M = 5$** | $N = 500$ | 0.85 ms | 0.28 ms | **3.08×** |

---

## Installation & Requirements

### Direct Installation

```bash
# 1) Clone repository
git clone https://github.com/METISBR/emopylab.git emopylab
cd emopylab

# 2) Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 3) Install package in editable development mode
pip install -e .
```

### Dependencies

* **Core Runtime:** `numpy>=2.3`, `scipy>=1.16`, `matplotlib>=3.10`, `psutil>=7.2`
* **Hardware Acceleration (Optional):** `jax>=0.9`, `jaxlib>=0.9`, `mlx>=0.21` (Apple Silicon)
* **Desktop Workstation (Optional):** `PySide6>=6.10`, `qt-material>=2.17`, `qtawesome>=1.3`
* **Test Suite:** `pytest>=9.0`, `scikit-learn>=1.8`

---

## Quickstart Guide (Headless & Python API)

### 1. Single Run Optimization in Python

```python
from algorithms.native.nsga2 import NativeNSGA2
from problems.zdt import ZDT1

# 1. Define problem and instantiate solver
problem = ZDT1(n_var=30)
solver = NativeNSGA2(pop_size=100)

# 2. Execute optimization
result = solver.solve(problem, n_gen=200, seed=42)

# 3. Access Pareto approximation and metrics
print(f"Status: {result.success} in {result.runtime_seconds:.2f}s")
print(f"Non-dominated solutions: {len(result.F)}")
print(f"Quality Indicators: {result.metrics}")
```

### 2. Multi-Seed Campaign via Headless CLI

```bash
# Run a 30-seed benchmark in parallel across 8 CPU/GPU workers:
emopylab run --algo GASDE --problem DTLZ2 --n-var 12 --n-obj 3 --pop-size 100 --n-gen 250 --n-runs 30 --n-workers 8 --out-dir results_dtlz2
```

---

## Desktop Workstation GUI

Launch the desktop visual analytics environment via any of the standard entry points:

```bash
# Launch using the primary launcher:
python EmoPyLab.py

# Or alternative convenience launcher:
python emopylabgui.py

# Or via registered CLI command:
emopylab-gui
```

### Desktop UI/UX Features:
* **Asynchronous Execution:** Solvers run in dedicated background threads (`QThreadPool`), preserving a responsive 60 FPS interface.
* **Ring Buffer Telemetry:** Streaming data is buffered in circular queues with fixed bounds, ensuring memory usage stays strictly below 150 MB RAM even during multi-day campaigns.
* **Glassmorphic HUD Tooltips:** Hovering over Pareto scatter plots reveals real-time objective vectors $f(x^*)$, constraint violations $CV$, and MCDM scores.
* **Hardware Acceleration Badge:** Real-time visual indicator (`⚡ JAX Vectorized`, `🍏 Apple MLX`, or `🖥️ CPU Core Engine`).

---

## Multi-Criteria Decision Making (MCDM)

EmoPyLab integrates *a posteriori* decision support directly into the post-search analytical continuum:

```python
from core.mcdm.decision import select_compromise_solution

# Apply TOPSIS with custom objective preference weights
decision = select_compromise_solution(
    front=result.F,
    method="topsis",
    weights_text="0.5, 0.25, 0.25",
)

print(f"Selected Solution Index: {decision['index']}")
print(f"Closeness Score: {decision['score']:.4f}")
print(f"Objective Coordinates: {decision['selected']}")
```

Supported Methods:
* **TOPSIS:** Distance to Ideal and Anti-Ideal solutions with min-max normalization.
* **PROMETHEE II:** Positive, negative, and net outranking flows ($\Phi$).
* **Normalized Weighted Sum:** Convex compromise programming.

---

## Statistical Inference & LaTeX Table Synthesis

Automated post-hoc statistical analysis evaluates multi-seed campaigns without external scripts:

```bash
# Analyze experimental sidecars and generate publication-ready LaTeX tables:
emopylab stats --results-dir results_dtlz2 --export-latex
```

Output:
```latex
\begin{table*}[t]
\centering
\caption{Statistical comparison of algorithm performance.}
\label{tab:stat_comparison}
\resizebox{\textwidth}{!}{
\begin{tabular}{lccccr}
\toprule
\textbf{Algorithm} & \textbf{Sample Size ($N$)} & \textbf{Mean $\pm$ Std. Dev.} & \textbf{$p$-value} & \textbf{Effect Size} & \textbf{Decision} \\
\midrule
GASDE & 30 & 3.5578e-01 \pm 5.03e-02 & -- & -- & $\approx$ \\
\bottomrule
\end{tabular}
}
\end{table*}
```

Included Tests:
* **Wilcoxon Signed-Rank Test:** Paired non-parametric comparison ($\alpha = 0.05$).
* **Friedman Global Ranking:** Mean rank ordering across multi-problem suites.
* **Holm / Bonferroni Post-Hoc:** Multi-hypothesis family-wise error rate control.

---

## Algorithm & Problem Catalog Taxonomy

Inspect all 298 metaheuristic solvers and benchmark suites:

```bash
# List all registered algorithms:
emopylab list --category algorithms

# Inspect metaheuristic taxonomy tiers:
emopylab list --category taxonomy
```

### Architectural Tiers:
* **Tier 1 (Native Full Implementations):** Authorial architectures with native tensor operators (e.g., LARC-NSGA-III, GCS-MaOEA, GASDE, SAGE-MOEA, SSW-DNV, MAACO).
* **Tier 2 (Canonical Literature Solvers):** Classical EMO/MaOP algorithms (NSGA-II, NSGA-III, MOEA/D, RVEA, SPEA2, BiGE, C-MOEA/D, Two-Arch2).
* **Tier 3 (Domain-Specific & Constrained Specialists):** Benchmark-specific heuristics and surrogate-assisted optimizers.

---

## Testing & Quality Assurance

EmoPyLab includes a comprehensive test suite covering mathematical contracts, hardware acceleration, numerical invariance, and desktop UI stability:

```bash
# Run full unit and integration test suite:
pytest -v
```

All 138+ automated unit and integration tests pass with 100% compliance.

---
## Citation

If you utilize **EmoPyLab** in your scientific research, algorithms, or benchmark evaluations, please cite:

```bibtex
@article{santos2026emopylab,
  title   = {{EmoPyLab: A Tensor-Native, Hardware-Accelerated Laboratory for High-Throughput Benchmarking and Decision-Making in Many-Objective Optimization}},
  author  = {Santos, Thiago and Xavier, Sebasti{\~a}o},
  journal = {Swarm and Evolutionary Computation (under review)},
  year    = {2026},
  url     = {https://github.com/METISBR/emopylab}
}

@software{Santos_EmoPyLab_2026,
  author  = {Santos, Thiago and Xavier, Sebasti{\~a}o},
  title   = {{EmoPyLab: A High-Throughput Benchmarking Ecosystem and Open-Source Decision Platform for Evolutionary Many-Objective Optimization}},
  year    = {2026},
  url     = {https://github.com/METISBR/emopylab},
  license = {MIT}
}
```
---

## License

EmoPyLab is released under the **MIT License**.

Copyright (c) 2026 Professor Thiago Santos, METISBr Research Group, Federal University of Ouro Preto (UFOP), Brazil.

