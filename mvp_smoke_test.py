"""EmoPyLab MVP Smoke Test — pop=30, 1000 evals, M=2,3,5.

Valida que os algoritmos canônicos + algoritmos especializados (amostra),
problemas e métricas estão operacionais end-to-end.

Uso:
    cd /Users/thiagosantos/devSuport/METISBr/emopylab
    python mvp_smoke_test.py
"""

from __future__ import annotations

import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ---------------------------------------------------------------------------
# Helpers de cor
# ---------------------------------------------------------------------------
_GREEN  = "\033[32m"
_RED    = "\033[31m"
_YELLOW = "\033[33m"
_CYAN   = "\033[36m"
_BOLD   = "\033[1m"
_RESET  = "\033[0m"

OK   = f"{_GREEN}✓ OK{_RESET}"
FAIL = f"{_RED}✗ FAIL{_RESET}"
SKIP = f"{_YELLOW}~ SKIP{_RESET}"

results: list[dict] = []


def run_case(label: str, fn) -> bool:
    t0 = time.perf_counter()
    try:
        fn()
        elapsed = time.perf_counter() - t0
        print(f"  {OK}  {label}  [{elapsed:.2f}s]")
        results.append({"label": label, "status": "ok", "elapsed": elapsed})
        return True
    except Exception as exc:
        elapsed = time.perf_counter() - t0
        brief = str(exc)[:120].replace("\n", " ")
        print(f"  {FAIL}  {label}  [{elapsed:.2f}s]")
        print(f"         {_RED}{brief}{_RESET}")
        results.append({"label": label, "status": "fail", "elapsed": elapsed, "error": str(exc)})
        return False


def section(title: str) -> None:
    print(f"\n{_BOLD}{_CYAN}{'─'*60}{_RESET}")
    print(f"{_BOLD}{_CYAN}  {title}{_RESET}")
    print(f"{_BOLD}{_CYAN}{'─'*60}{_RESET}")


# ===========================================================================
# 0. Imports base
# ===========================================================================
section("0 · Imports base do framework")


def _import_core():
    from core.algorithm import Algorithm
    from core.population import Population
    from core.optimize import minimize, Result


def _import_operators():
    from operators.crossover.sbx import SBX
    from operators.mutation.pm import PolynomialMutation
    from operators.sampling.lhs import LatinHypercubeSampling


def _import_problems_many():
    from problems.many.dtlz import DTLZ1, DTLZ2, DTLZ3, DTLZ4, DTLZ5, DTLZ6, DTLZ7
    from problems.many.wfg import WFG1, WFG2, WFG4


def _import_problems_multi():
    from problems.multi.zdt import ZDT1, ZDT2, ZDT3, ZDT4, ZDT6


def _import_metrics():
    from metrics.indicators import IGD, IGDPlus, GD, GDPlus, Spacing
    from metrics.evaluator import MetricEvaluator


def _import_canonical_algos():
    from algorithms.nsga2 import NSGA2
    from algorithms.nsga3 import NSGA3
    from algorithms.rvea import RVEA
    from algorithms.moead import MOEAD
    from algorithms.age2 import AGEMOEA2
    from algorithms.sms import SMSEMOA


run_case("core (Algorithm, Population, minimize)", _import_core)
run_case("operators (SBX, PM, LHS)", _import_operators)
run_case("problems.many (DTLZ1-7, WFG1/2/4)", _import_problems_many)
run_case("problems.multi (ZDT1-4,6)", _import_problems_multi)
run_case("metrics (IGD, IGD+, GD, GD+, Spacing)", _import_metrics)
run_case("algorithms canônicos (NSGA2/3, RVEA, MOEAD, AGE2, SMS)", _import_canonical_algos)


# ===========================================================================
# 1. Factory de problemas e algoritmos para o MVP
# ===========================================================================

POP   = 30
EVALS = 1000


def _make_algo(name: str, n_obj: int) -> Any:
    """Instancia algoritmo canônico com ref_dirs onde necessário."""
    from algorithms.nsga2 import NSGA2
    from algorithms.nsga3 import NSGA3
    from algorithms.rvea import RVEA
    from algorithms.moead import MOEAD
    from algorithms.age2 import AGEMOEA2
    from algorithms.sms import SMSEMOA

    if name == "NSGA2":
        return NSGA2(pop_size=POP)
    if name == "NSGA3":
        try:
            from util.ref_dirs import get_reference_directions
            ref = get_reference_directions("das-dennis", n_obj, n_partitions=3)
        except Exception:
            ref = None
        return NSGA3(ref_dirs=ref, pop_size=POP)
    if name == "RVEA":
        try:
            from util.ref_dirs import get_reference_directions
            ref = get_reference_directions("das-dennis", n_obj, n_partitions=3)
        except Exception:
            ref = None
        return RVEA(ref_dirs=ref, pop_size=POP)
    if name == "MOEAD":
        try:
            from util.ref_dirs import get_reference_directions
            ref = get_reference_directions("das-dennis", n_obj, n_partitions=3)
        except Exception:
            ref = None
        return MOEAD(ref_dirs=ref, pop_size=POP)
    if name == "AGE2":
        return AGEMOEA2(pop_size=POP)
    if name == "SMS":
        return SMSEMOA(pop_size=POP, n_offsprings=1)
    raise ValueError(f"Unknown algo: {name}")


def _make_problem(suite: str, pid: int, n_obj: int) -> Any:
    if suite == "DTLZ":
        mod = __import__(f"problems.many.dtlz", fromlist=[f"DTLZ{pid}"])
        cls = getattr(mod, f"DTLZ{pid}")
        n_var = n_obj + 4 if pid in (1, 3, 5, 6) else n_obj + 9
        return cls(n_var=n_var, n_obj=n_obj)
    if suite == "ZDT":
        mod = __import__(f"problems.multi.zdt", fromlist=[f"ZDT{pid}"])
        cls = getattr(mod, f"ZDT{pid}")
        return cls(n_var=10)
    raise ValueError(f"Unknown suite: {suite}")


# ===========================================================================
# 2. Algoritmos × Problemas × M — bateria principal
# ===========================================================================
section("1 · Algoritmos canônicos × DTLZ × M={2,3,5}")

CANONICAL = ["NSGA2", "NSGA3", "RVEA", "MOEAD", "AGE2", "SMS"]
DTLZ_IDS  = [1, 2, 4]   # representativos: unimodal, cônica, tendenciosa
M_LIST    = [2, 3, 5]


def _smoke(algo_name: str, suite: str, pid: int, n_obj: int) -> None:
    from core.optimize import minimize
    problem = _make_problem(suite, pid, n_obj)
    algo    = _make_algo(algo_name, n_obj)
    res     = minimize(problem, algo, ("n_eval", EVALS), seed=42, verbose=False)
    assert res is not None, "minimize() retornou None"
    F = res.F
    assert F is not None and len(F) > 0, f"res.F vazia: {F}"
    assert F.shape[1] == n_obj, f"shape errado: {F.shape} esperado (*,{n_obj})"
    # pelo menos 1 solução não dominada
    assert len(F) >= 1


for algo_name in CANONICAL:
    for pid in DTLZ_IDS:
        for M in M_LIST:
            lbl = f"{algo_name} / DTLZ{pid} / M={M}"
            run_case(lbl, lambda a=algo_name, p=pid, m=M: _smoke(a, "DTLZ", p, m))


# ===========================================================================
# 3. ZDT (M=2 only — benchmark bi-objetivo clássico)
# ===========================================================================
section("2 · Algoritmos canônicos × ZDT × M=2")

ZDT_IDS = [1, 2, 3, 4, 6]
M2_ALGOS = ["NSGA2", "AGE2", "SMS"]  # ZDT é estritamente 2-objetivo

for algo_name in M2_ALGOS:
    for pid in ZDT_IDS:
        lbl = f"{algo_name} / ZDT{pid} / M=2"
        run_case(lbl, lambda a=algo_name, p=pid: _smoke(a, "ZDT", p, 2))


# ===========================================================================
# 4. Métricas sobre resultado real
# ===========================================================================
section("3 · Métricas (IGD, IGD+, GD, GD+, Spacing) sobre DTLZ2/M=3")


def _metrics_smoke():
    from core.optimize import minimize
    from algorithms.nsga3 import NSGA3
    from problems.many.dtlz import DTLZ2
    from metrics.indicators import IGD, IGDPlus, GD, GDPlus, Spacing

    problem = DTLZ2(n_var=7, n_obj=3)
    try:
        from util.ref_dirs import get_reference_directions
        ref = get_reference_directions("das-dennis", 3, n_partitions=3)
    except Exception:
        ref = None
    algo = NSGA3(ref_dirs=ref, pop_size=POP)
    res  = minimize(problem, algo, ("n_eval", EVALS), seed=0, verbose=False)
    F    = res.F
    assert F is not None and len(F) > 0

    # Approximate PF: use ref_dirs normalized as approximation
    pf = problem.pareto_front(n_points=100) if hasattr(problem, "pareto_front") else F[:10]
    if pf is None or len(pf) == 0:
        pf = F[:10]

    for MetricCls in [IGD, IGDPlus, GD, GDPlus, Spacing]:
        try:
            if MetricCls is Spacing:
                val = MetricCls().do(F)
            else:
                val = MetricCls().do(F, pf)
            assert np.isfinite(val), f"{MetricCls.__name__} retornou não-finito: {val}"
        except TypeError:
            # alguns podem ter assinatura diferente
            val = MetricCls(pf).do(F) if MetricCls is not Spacing else MetricCls().do(F)
            assert np.isfinite(val)


run_case("Métricas IGD/IGD+/GD/GD+/Spacing sobre NSGA3/DTLZ2/M=3", _metrics_smoke)


# ===========================================================================
# 5. Amostra de algoritmos especializados (smoke rápido DTLZ2/M=2)
# ===========================================================================
section("4 · Algoritmos especializados — amostra (DTLZ2, M=2, pop=30, 500 evals)")

SPECIALIZED_SAMPLE = [
    ("algorithms.hype.hype",           "HypE"),
    ("algorithms.spea2_sde.spea2_sde", "SPEA2SDE"),
    ("algorithms.ibea.ibea",           "IBEA"),
    ("algorithms.moead_de.moead_de",   "MOEADDE"),
    ("algorithms.t_dea.t_dea",         "tDEA"),
    ("algorithms.gde3.gde3",           "GDE3"),
    ("algorithms.two_arch2.two_arch2", "Two_Arch2"),
    ("algorithms.rpea.rpea",           "RPEA"),
]


def _specialized_smoke(module_path: str, class_name: str) -> None:
    import importlib
    from core.optimize import minimize
    from problems.many.dtlz import DTLZ2

    mod = importlib.import_module(module_path)
    AlgoCls = getattr(mod, class_name)
    problem = DTLZ2(n_var=6, n_obj=2)

    # Tenta instanciar com pop_size; caso falhe, sem argumentos
    try:
        algo = AlgoCls(pop_size=POP)
    except TypeError:
        try:
            from util.ref_dirs import get_reference_directions
            ref = get_reference_directions("das-dennis", 2, n_partitions=5)
            algo = AlgoCls(ref_dirs=ref, pop_size=POP)
        except TypeError:
            algo = AlgoCls()

    res = minimize(problem, algo, ("n_eval", 500), seed=7, verbose=False)
    assert res is not None and res.F is not None and len(res.F) > 0


for mod_path, cls_name in SPECIALIZED_SAMPLE:
    lbl = f"Especializado: {cls_name} / DTLZ2 / M=2"
    run_case(lbl, lambda m=mod_path, c=cls_name: _specialized_smoke(m, c))


# ===========================================================================
# 6. Termination conditions
# ===========================================================================
section("5 · Termination conditions (n_gen, n_eval, time)")


def _termination_n_gen():
    from core.optimize import minimize
    from algorithms.nsga2 import NSGA2
    from problems.many.dtlz import DTLZ1
    res = minimize(DTLZ1(n_var=6, n_obj=2), NSGA2(pop_size=POP),
                   ("n_gen", 10), seed=1, verbose=False)
    assert res.F is not None and len(res.F) > 0


def _termination_n_eval():
    from core.optimize import minimize
    from algorithms.nsga2 import NSGA2
    from problems.many.dtlz import DTLZ2
    res = minimize(DTLZ2(n_var=7, n_obj=3), NSGA2(pop_size=POP),
                   ("n_eval", EVALS), seed=2, verbose=False)
    assert res.F is not None and len(res.F) > 0


run_case("Termination n_gen=10 / NSGA2 / DTLZ1 / M=2", _termination_n_gen)
run_case("Termination n_eval=1000 / NSGA2 / DTLZ2 / M=3", _termination_n_eval)


# ===========================================================================
# 7. Resultado final
# ===========================================================================
section("RESULTADO FINAL")

total  = len(results)
passed = sum(1 for r in results if r["status"] == "ok")
failed = sum(1 for r in results if r["status"] == "fail")
total_time = sum(r["elapsed"] for r in results)

print(f"\n  {_BOLD}Total:{_RESET} {total}  |  "
      f"{_GREEN}Passed: {passed}{_RESET}  |  "
      f"{_RED}Failed: {failed}{_RESET}  |  "
      f"Tempo total: {total_time:.1f}s\n")

if failed > 0:
    print(f"{_BOLD}{_RED}  FALHAS DETECTADAS:{_RESET}")
    for r in results:
        if r["status"] == "fail":
            print(f"    {_RED}✗{_RESET} {r['label']}")
            if "error" in r:
                # mostra até 2 linhas do traceback
                tb_lines = r["error"].strip().split("\n")
                for ln in tb_lines[:3]:
                    print(f"       {_YELLOW}{ln}{_RESET}")
    print()
    sys.exit(1)
else:
    print(f"  {_GREEN}{_BOLD}Todos os {total} casos passaram. EmoPyLab MVP validado.{_RESET}\n")
    sys.exit(0)
