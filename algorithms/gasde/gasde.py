"""Geometry-Adaptive Shifted-Distance Ensemble (GASDE).

GASDE is an experimental many-objective genetic algorithm.  Its environmental
selection combines nondominated sorting, explicit extreme preservation,
reference-sector champions, and E3A-style incremental shifted distance.  The
reference geometry may be adapted from an online nondominated archive, but a
proposal is accepted only when it improves angular coverage on an internal
held-out archive subset.  This is a heuristic gate, not independent external
validation: all archive points arose from the same evolutionary run.

The implementation is benchmark agnostic: it never inspects a problem name,
known Pareto front, or target objective values.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import numpy as np
from algorithms.moo.nsga2 import NSGA2, binary_tournament
from core.population import Population
from core.survival import Survival
from operators.crossover.sbx import SBX
from operators.mutation.pm import PM
from operators.sampling.lhs import LHS
from operators.selection.tournament import TournamentSelection
from util.nds.non_dominated_sorting import NonDominatedSorting

from operators.utility_functions.UniformPoint import UniformPoint


ALGORITHM_FLAGS = {"GASDE": {"multi", "many"}}


def _unit_rows(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.ndim != 2:
        raise ValueError("Expected a two-dimensional matrix.")
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    out = values / np.maximum(norms, 1e-15)
    zero = norms[:, 0] <= 1e-15
    if np.any(zero):
        out[zero] = 1.0 / np.sqrt(max(values.shape[1], 1))
    return out


def _sanitize_objectives(F: np.ndarray) -> np.ndarray:
    """Replace non-finite values by deterministic column-wise worst values."""
    F = np.asarray(F, dtype=float)
    if F.ndim != 2 or F.shape[1] < 2:
        raise ValueError("GASDE requires a two-dimensional multiobjective matrix.")
    clean = F.copy()
    for j in range(clean.shape[1]):
        finite = np.isfinite(clean[:, j])
        if np.any(finite):
            values = clean[finite, j]
            span = max(float(np.ptp(values)), 1.0)
            replacement = float(np.max(values) + span)
        else:
            replacement = 1.0
        clean[~finite, j] = replacement
    return clean


def robust_normalize(F: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """NSGA-III-style intercept normalization with robust fallbacks.

    A 95th-percentile span prevents one extreme outlier from setting every
    scale.  A valid ASF/intercept estimate is retained when it is not smaller
    than that robust span.  Degenerate objectives receive a relative floor.
    """
    F = _sanitize_objectives(F)
    ideal = np.min(F, axis=0)
    shifted = np.maximum(F - ideal, 0.0)
    m = F.shape[1]
    q95 = np.quantile(shifted, 0.95, axis=0)
    observed = np.max(shifted, axis=0)

    weights = np.full((m, m), 1e-6, dtype=float)
    np.fill_diagonal(weights, 1.0)
    extreme = []
    for j in range(m):
        asf = np.max(shifted / weights[j][None, :], axis=1)
        extreme.append(int(np.argmin(asf)))

    intercept = None
    try:
        matrix = shifted[np.asarray(extreme, dtype=int)]
        if np.linalg.matrix_rank(matrix) == m and np.linalg.cond(matrix) < 1e12:
            plane = np.linalg.solve(matrix, np.ones(m, dtype=float))
            candidate = 1.0 / plane
            if np.all(np.isfinite(candidate)) and np.all(candidate > 0.0):
                intercept = candidate
    except (ValueError, np.linalg.LinAlgError, FloatingPointError):
        intercept = None

    scale = q95 if intercept is None else np.maximum(q95, intercept)
    fallback = np.where(observed > 0.0, observed, 1.0)
    scale = np.where(np.isfinite(scale) & (scale > 0.0), scale, fallback)
    floor = max(float(np.max(scale)) * 1e-12, 1e-12)
    scale = np.maximum(scale, floor)
    return shifted / scale[None, :], ideal, ideal + scale


def shifted_distance_matrix(candidates: np.ndarray, selected: np.ndarray) -> np.ndarray:
    """Return E3A/SDE distances from candidates to already selected points.

    For minimization, a selected point ``y`` is shifted to ``max(x, y)`` with
    respect to candidate ``x``.  The resulting distance is therefore
    ``||max(y - x, 0)||``.
    """
    candidates = np.asarray(candidates, dtype=float)
    selected = np.asarray(selected, dtype=float)
    if candidates.ndim != 2 or selected.ndim != 2:
        raise ValueError("Candidates and selected points must be matrices.")
    if candidates.shape[1] != selected.shape[1]:
        raise ValueError("Candidates and selected points must share dimensions.")
    if selected.shape[0] == 0:
        return np.empty((candidates.shape[0], 0), dtype=float)
    delta = np.maximum(selected[None, :, :] - candidates[:, None, :], 0.0)
    return np.linalg.norm(delta, axis=2)


def _angles(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    cosine = np.clip(_unit_rows(A) @ _unit_rows(B).T, -1.0, 1.0)
    return np.arccos(cosine)


def _angular_coverage_score(samples: np.ndarray, refs: np.ndarray) -> float:
    if len(samples) == 0 or len(refs) == 0:
        return float("inf")
    nearest = np.min(_angles(samples, refs), axis=1)
    return float(np.mean(nearest) + 0.25 * np.quantile(nearest, 0.90))


def _uniform_references(n_points: int, n_obj: int) -> np.ndarray:
    refs, _ = UniformPoint(int(n_points), int(n_obj))
    refs = np.asarray(refs, dtype=float)
    # UniformPoint is deterministic.  Lexicographic sorting makes this
    # invariant explicit even if its internal enumeration changes.
    order = np.lexsort(tuple(refs[:, j] for j in range(refs.shape[1] - 1, -1, -1)))
    return _unit_rows(refs[order])


def _stable_subset(refs: np.ndarray, n_keep: int) -> np.ndarray:
    refs = _unit_rows(refs)
    if len(refs) <= n_keep:
        return refs
    picked = [int(np.argmin(refs[:, 0]))]
    nearest = _angles(refs, refs[picked]).reshape(-1)
    while len(picked) < n_keep:
        nearest[picked] = -np.inf
        nxt = int(np.argmax(nearest))
        picked.append(nxt)
        nearest = np.minimum(nearest, _angles(refs, refs[[nxt]]).reshape(-1))
    return refs[np.asarray(picked, dtype=int)]


def _extreme_indices(Fn: np.ndarray) -> list[int]:
    m = Fn.shape[1]
    weights = np.full((m, m), 1e-6, dtype=float)
    np.fill_diagonal(weights, 1.0)
    chosen: list[int] = []
    for j in range(m):
        score = np.max(Fn / weights[j][None, :], axis=1)
        for idx in np.argsort(score, kind="mergesort"):
            i = int(idx)
            if i not in chosen:
                chosen.append(i)
                break
    return chosen


def _progress(algorithm: Any | None) -> float:
    if algorithm is None:
        return 0.5
    term = getattr(algorithm, "termination", None)
    n_gen = float(getattr(algorithm, "n_gen", 0) or 0)
    n_max_gen = getattr(term, "n_max_gen", None)
    if n_max_gen:
        return float(np.clip(n_gen / float(n_max_gen), 0.0, 1.0))
    n_eval = float(getattr(getattr(algorithm, "evaluator", None), "n_eval", 0) or 0)
    n_max_eval = getattr(term, "n_max_evals", None)
    if n_max_eval:
        return float(np.clip(n_eval / float(n_max_eval), 0.0, 1.0))
    return float(np.clip(n_gen / (n_gen + 20.0), 0.0, 1.0))


class GASDESurvival(Survival):
    """Stateful O(MN^2) environmental selection used by :class:`GASDE`."""

    def __init__(
        self,
        *,
        ref_dirs: np.ndarray | None = None,
        adaptation: bool = True,
        adaptation_interval: int = 10,
        adaptation_warmup: int = 5,
        adaptation_inertia: float = 0.75,
        adaptation_min_gain: float = 0.01,
        holdout_fraction: float = 0.30,
        empty_direction_patience: int = 3,
        uniform_anchor: float = 0.05,
        archive_multiplier: int = 4,
        scalarization: str = "apd",
        pbi_theta: float = 5.0,
        apd_alpha: float = 2.0,
        use_extreme_preservation: bool = True,
        use_sector_champions: bool = True,
        use_shifted_distance: bool = True,
        use_evidence_gate: bool = True,
        use_credit_coupling: bool = True,
        use_step_credit: bool | None = None,
        use_mating_credit: bool | None = None,
    ) -> None:
        super().__init__(filter_infeasible=True)
        if adaptation_interval < 1 or adaptation_warmup < 0:
            raise ValueError("Adaptation interval must be positive and warmup nonnegative.")
        if not 0.0 <= adaptation_inertia < 1.0:
            raise ValueError("adaptation_inertia must be in [0, 1).")
        if adaptation_min_gain < 0.0:
            raise ValueError("adaptation_min_gain must be nonnegative.")
        if not 0.0 < holdout_fraction < 0.5:
            raise ValueError("holdout_fraction must be in (0, 0.5).")
        if scalarization not in {"apd", "pbi"}:
            raise ValueError("scalarization must be 'apd' or 'pbi'.")

        self._provided_refs = None if ref_dirs is None else np.asarray(ref_dirs, dtype=float)
        self.adaptation = bool(adaptation)
        self.adaptation_interval = int(adaptation_interval)
        self.adaptation_warmup = int(adaptation_warmup)
        self.adaptation_inertia = float(adaptation_inertia)
        self.adaptation_min_gain = float(adaptation_min_gain)
        self.holdout_fraction = float(holdout_fraction)
        self.empty_direction_patience = int(max(0, empty_direction_patience))
        self.uniform_anchor = float(np.clip(uniform_anchor, 0.0, 1.0))
        self.archive_multiplier = int(max(1, archive_multiplier))
        self.scalarization = scalarization
        self.pbi_theta = float(max(pbi_theta, 0.0))
        self.apd_alpha = float(max(apd_alpha, 0.0))
        self.use_extreme_preservation = bool(use_extreme_preservation)
        self.use_sector_champions = bool(use_sector_champions)
        self.use_shifted_distance = bool(use_shifted_distance)
        self.use_evidence_gate = bool(use_evidence_gate)
        # ``use_credit_coupling`` is the backward-compatible master alias.
        # Explicit axis switches override it independently.
        self.use_credit_coupling = bool(use_credit_coupling)
        self.use_step_credit = (
            self.use_credit_coupling
            if use_step_credit is None
            else bool(use_step_credit)
        )
        self.use_mating_credit = (
            self.use_credit_coupling
            if use_mating_credit is None
            else bool(use_mating_credit)
        )
        self.reset()

    def reset(self) -> None:
        self.n_obj = 0
        self.pop_size = 0
        self.base_ref_dirs = np.empty((0, 0), dtype=float)
        self.ref_dirs = np.empty((0, 0), dtype=float)
        self.empty_streak = np.empty(0, dtype=int)
        self.nd_archive_F = np.empty((0, 0), dtype=float)
        self.generation = 0
        # No mating credit exists before an internal-holdout improvement has
        # been observed.  This is heuristic validation within the same run;
        # the adaptation step retains a 0.5 exploration floor so learning is
        # not deadlocked at startup.
        self.adaptation_credit = 0.0
        self.adaptation_history: list[dict[str, Any]] = []
        self.diagnostics: dict[str, Any] = {}

    def configure(self, n_obj: int, pop_size: int) -> None:
        self.reset()
        self.n_obj, self.pop_size = int(n_obj), int(pop_size)
        if self.n_obj < 2:
            raise ValueError("GASDE requires at least two objectives.")
        if self._provided_refs is None:
            refs = _uniform_references(self.pop_size, self.n_obj)
        else:
            refs = np.asarray(self._provided_refs, dtype=float)
            if refs.ndim != 2 or refs.shape[1] != self.n_obj or len(refs) == 0:
                raise ValueError("ref_dirs must be a nonempty (K, n_obj) matrix.")
            if np.any(~np.isfinite(refs)) or np.any(refs < 0.0):
                raise ValueError("ref_dirs must contain finite nonnegative values.")
            refs = _stable_subset(refs, self.pop_size)
        self.base_ref_dirs = _unit_rows(refs)
        self.ref_dirs = self.base_ref_dirs.copy()
        self.empty_streak = np.zeros(len(refs), dtype=int)
        self.nd_archive_F = np.empty((0, self.n_obj), dtype=float)

    def _update_archive(self, F: np.ndarray) -> None:
        finite = np.all(np.isfinite(F), axis=1)
        incoming = np.asarray(F[finite], dtype=float)
        if len(incoming) == 0:
            return
        merged = incoming if len(self.nd_archive_F) == 0 else np.vstack([self.nd_archive_F, incoming])
        merged = np.unique(merged, axis=0)
        front = NonDominatedSorting().do(merged, only_non_dominated_front=True)
        archive = merged[np.asarray(front, dtype=int)]
        cap = self.archive_multiplier * self.pop_size
        if len(archive) > cap:
            Fn, _, _ = robust_normalize(archive)
            selected = _extreme_indices(Fn)
            selected = selected[:cap]
            remaining = [i for i in range(len(archive)) if i not in selected]
            nearest = np.min(
                shifted_distance_matrix(Fn[remaining], Fn[selected]), axis=1
            )
            while len(selected) < cap and remaining:
                pos = int(np.argmax(nearest))
                chosen = remaining.pop(pos)
                selected.append(chosen)
                nearest = np.delete(nearest, pos)
                if remaining:
                    update = shifted_distance_matrix(
                        Fn[remaining], Fn[[chosen]]
                    )[:, 0]
                    nearest = np.minimum(nearest, update)
            archive = archive[np.asarray(selected, dtype=int)]
        self.nd_archive_F = archive

    def _adapt_references(self, rng: np.random.Generator) -> None:
        if not self.adaptation:
            return
        due = self.generation >= self.adaptation_warmup and (
            (self.generation - self.adaptation_warmup) % self.adaptation_interval == 0
        )
        if not due:
            return

        record: dict[str, Any] = {"generation": self.generation, "accepted": False}
        if len(self.nd_archive_F) < max(6, self.n_obj + 2):
            record["reason"] = "insufficient_archive_evidence"
            self.adaptation_history.append(record)
            return

        Fn, _, _ = robust_normalize(self.nd_archive_F)
        directions = _unit_rows(Fn)
        order = rng.permutation(len(directions))
        n_holdout = max(2, int(np.ceil(self.holdout_fraction * len(order))))
        n_holdout = min(n_holdout, len(order) - 2)
        holdout = directions[order[:n_holdout]]
        train = directions[order[n_holdout:]]

        angle = _angles(train, self.ref_dirs)
        association = np.argmin(angle, axis=1)
        learned = self.ref_dirs.copy()
        occupied = np.zeros(len(self.ref_dirs), dtype=bool)
        for j in range(len(self.ref_dirs)):
            members = train[association == j]
            if len(members):
                learned[j] = _unit_rows(np.mean(members, axis=0, keepdims=True))[0]
                occupied[j] = True
                self.empty_streak[j] = 0
            else:
                self.empty_streak[j] += 1

        # An empty direction survives several update attempts.  Only then may
        # it be replaced by the training direction farthest from all occupied
        # learned directions.
        active = learned[occupied]
        if len(active):
            nearest_training_angle = np.min(_angles(train, active), axis=1)
        else:
            nearest_training_angle = np.full(len(train), np.inf, dtype=float)
        for j in np.where(~occupied)[0]:
            if self.empty_streak[j] <= self.empty_direction_patience or len(train) == 0:
                continue
            # Maintain nearest angular distance incrementally.  Recomputing
            # against the growing active matrix in every iteration would turn
            # empty-sector repair into O(MN^3).
            if np.all(np.isinf(nearest_training_angle)):
                replacement = 0
            else:
                replacement = int(np.argmax(nearest_training_angle))
            learned[j] = train[replacement]
            update = _angles(train, train[[replacement]])[:, 0]
            nearest_training_angle = np.minimum(nearest_training_angle, update)

        coupling = 0.5 + 0.5 * self.adaptation_credit if self.use_step_credit else 1.0
        step = (1.0 - self.adaptation_inertia) * coupling
        proposal = _unit_rows((1.0 - step) * self.ref_dirs + step * learned)
        proposal = _unit_rows((1.0 - self.uniform_anchor) * proposal + self.uniform_anchor * self.base_ref_dirs)
        old_score = _angular_coverage_score(holdout, self.ref_dirs)
        new_score = _angular_coverage_score(holdout, proposal)
        gain = (old_score - new_score) / max(old_score, 1e-15)
        accepted = (not self.use_evidence_gate) or gain >= self.adaptation_min_gain
        record.update(
            old_score=old_score,
            proposed_score=new_score,
            relative_gain=float(gain),
            occupied_training_sectors=int(np.sum(occupied)),
            accepted=bool(accepted),
            reason="coverage_improved" if accepted else "holdout_gate_rejected",
        )
        if accepted:
            motion = float(np.mean(np.min(_angles(proposal, self.ref_dirs), axis=1)))
            self.ref_dirs = proposal
            record["mean_reference_motion"] = motion
        if self.use_step_credit or self.use_mating_credit:
            evidence = float(np.clip(gain / max(self.adaptation_min_gain, 0.01), 0.0, 1.0))
            if not accepted:
                evidence = 0.0
            self.adaptation_credit = 0.8 * self.adaptation_credit + 0.2 * evidence
        self.adaptation_history.append(record)

    def _sector_scores(self, Fn: np.ndarray, association: np.ndarray, algorithm: Any) -> np.ndarray:
        W = self.ref_dirs[association]
        d1 = np.sum(Fn * W, axis=1)
        d2 = np.linalg.norm(Fn - d1[:, None] * W, axis=1)
        if self.scalarization == "pbi":
            return d1 + self.pbi_theta * d2
        ref_angle = _angles(Fn, W)[np.arange(len(Fn)), np.arange(len(Fn))]
        if len(self.ref_dirs) == 1:
            gamma = np.array([np.pi / 2.0])
        else:
            pair = _angles(self.ref_dirs, self.ref_dirs)
            np.fill_diagonal(pair, np.inf)
            gamma = np.min(pair, axis=1)
        penalty = self.n_obj * (_progress(algorithm) ** self.apd_alpha) * ref_angle / np.maximum(gamma[association], 1e-12)
        return np.linalg.norm(Fn, axis=1) * (1.0 + penalty)

    def _critical_select(
        self,
        Fn: np.ndarray,
        critical: np.ndarray,
        prefix: list[int],
        slots: int,
        algorithm: Any,
    ) -> tuple[list[int], dict[int, str], dict[str, int]]:
        selected: list[int] = []
        roles: dict[int, str] = {}
        counts = {"extremes": 0, "sector_champions": 0, "shifted_distance": 0, "fallback": 0}

        if self.use_extreme_preservation:
            for local in _extreme_indices(Fn[critical]):
                idx = int(critical[local])
                if idx not in selected and len(selected) < slots:
                    selected.append(idx)
                    roles[idx] = "extreme"
                    counts["extremes"] += 1

        remaining = np.asarray([i for i in critical if int(i) not in selected], dtype=int)
        if self.use_sector_champions and len(remaining) and len(selected) < slots:
            association = np.argmin(_angles(Fn[remaining], self.ref_dirs), axis=1)
            score = self._sector_scores(Fn[remaining], association, algorithm)
            champions: list[tuple[float, int]] = []
            for sector in np.unique(association):
                positions = np.where(association == sector)[0]
                best = int(positions[np.argmin(score[positions])])
                champions.append((float(score[best]), int(remaining[best])))
            for _, idx in sorted(champions, key=lambda item: (item[0], item[1])):
                if idx not in selected and len(selected) < slots:
                    selected.append(idx)
                    roles[idx] = "sector_champion"
                    counts["sector_champions"] += 1

        remaining = [int(i) for i in critical if int(i) not in selected]
        anchors = prefix + selected
        if not anchors and remaining:
            seed = min(remaining, key=lambda i: (float(np.linalg.norm(Fn[i])), i))
            selected.append(seed)
            roles[seed] = "fallback_seed"
            counts["fallback"] += 1
            remaining.remove(seed)
            anchors = [seed]

        if remaining and len(selected) < slots:
            D = shifted_distance_matrix(Fn[remaining], Fn[anchors])
            nearest = np.min(D, axis=1)
            while remaining and len(selected) < slots:
                if self.use_shifted_distance:
                    best_pos = max(range(len(remaining)), key=lambda p: (float(nearest[p]), -float(np.linalg.norm(Fn[remaining[p]])), -remaining[p]))
                    role = "shifted_distance"
                    counts["shifted_distance"] += 1
                else:
                    euclid = np.linalg.norm(Fn[remaining, None, :] - Fn[np.asarray(anchors), :][None, :, :], axis=2)
                    ordinary = np.min(euclid, axis=1)
                    best_pos = max(range(len(remaining)), key=lambda p: (float(ordinary[p]), -remaining[p]))
                    role = "euclidean_ablation"
                    counts["fallback"] += 1
                idx = remaining.pop(best_pos)
                selected.append(idx)
                roles[idx] = role
                anchors.append(idx)
                if remaining and self.use_shifted_distance:
                    update = shifted_distance_matrix(Fn[remaining], Fn[[idx]])[:, 0]
                    nearest = np.delete(nearest, best_pos)
                    nearest = np.minimum(nearest, update)

        return selected[:slots], roles, counts

    def _do(self, problem, pop: Population, *args, n_survive=None, random_state=None, algorithm=None, **kwargs):
        n_survive = int(n_survive or len(pop))
        if self.n_obj == 0:
            self.configure(int(problem.n_obj), n_survive)
        rng = random_state if isinstance(random_state, np.random.Generator) else np.random.default_rng(random_state)
        F_raw = np.asarray(pop.get("F"), dtype=float)
        F = _sanitize_objectives(F_raw)
        fronts, rank = NonDominatedSorting().do(F, return_rank=True)
        rank = np.asarray(rank, dtype=int)
        pop.set("rank", rank)
        self._update_archive(F_raw)
        self._adapt_references(rng)
        Fn, _, _ = robust_normalize(F)

        selected: list[int] = []
        roles: dict[int, str] = {}
        counts = {"extremes": 0, "sector_champions": 0, "shifted_distance": 0, "fallback": 0}
        for front in fronts:
            front = np.asarray(front, dtype=int)
            if len(selected) + len(front) <= n_survive:
                selected.extend(int(i) for i in front)
                for i in front:
                    roles[int(i)] = "complete_front"
                if len(selected) == n_survive:
                    break
            else:
                slots = n_survive - len(selected)
                chosen, critical_roles, critical_counts = self._critical_select(Fn, front, selected, slots, algorithm)
                selected.extend(chosen)
                roles.update(critical_roles)
                for key in counts:
                    counts[key] += critical_counts[key]
                break

        if len(selected) < n_survive:
            missing = [i for i in range(len(pop)) if i not in selected]
            selected.extend(missing[: n_survive - len(selected)])

        association = np.argmin(_angles(Fn, self.ref_dirs), axis=1)
        pop.set("gasde_sector", association)
        pop.set("gasde_score", np.linalg.norm(Fn, axis=1))
        for i, individual in enumerate(pop):
            individual.set("gasde_role", roles.get(i, "not_selected"))

        # NSGA-II's binary tournament expects a larger-is-better crowding value.
        survivors_F = Fn[np.asarray(selected, dtype=int)]
        if len(selected) <= 1:
            crowding = np.full(len(selected), np.inf)
        else:
            pair = _angles(survivors_F, survivors_F)
            np.fill_diagonal(pair, np.inf)
            crowding = np.min(pair, axis=1)
            if self.use_mating_credit:
                survivor_sectors = association[np.asarray(selected, dtype=int)]
                sector_counts = np.bincount(
                    survivor_sectors, minlength=len(self.ref_dirs)
                )
                local_counts = sector_counts[survivor_sectors]
                occupied_counts = sector_counts[sector_counts > 0]
                mean_count = float(np.mean(occupied_counts))
                rarity = np.clip(
                    mean_count / np.maximum(local_counts, 1), 0.5, 2.0
                )
                # Credit earned by accepted internal-holdout geometry updates
                # is coupled to mating: it gives a bounded tournament
                # advantage to survivors in under-occupied sectors.
                strength = 0.25 * self.adaptation_credit
                crowding *= 1.0 + strength * (rarity - 1.0)
            for p, idx in enumerate(selected):
                if roles.get(idx) == "extreme":
                    crowding[p] = np.inf
        for p, idx in enumerate(selected):
            pop[idx].set("crowding", float(crowding[p]))

        occupied = int(len(np.unique(association[np.asarray(selected, dtype=int)])))
        accepted = sum(bool(item.get("accepted")) for item in self.adaptation_history)
        proposed = sum("proposed_score" in item for item in self.adaptation_history)
        self.diagnostics = {
            "generation": self.generation,
            "archive_size": int(len(self.nd_archive_F)),
            "reference_count": int(len(self.ref_dirs)),
            "occupied_sectors": occupied,
            "empty_sectors": int(len(self.ref_dirs) - occupied),
            "extremes_selected": counts["extremes"],
            "sector_champions_selected": counts["sector_champions"],
            "shifted_distance_selected": counts["shifted_distance"],
            "fallback_selected": counts["fallback"],
            "adaptations_proposed": proposed,
            "adaptations_accepted": accepted,
            "adaptation_credit": float(self.adaptation_credit),
            "mating_credit_strength": float(
                0.25 * self.adaptation_credit if self.use_mating_credit else 0.0
            ),
            "last_adaptation": deepcopy(self.adaptation_history[-1]) if self.adaptation_history else None,
            "ablations": {
                "adaptation": self.adaptation,
                "extreme_preservation": self.use_extreme_preservation,
                "sector_champions": self.use_sector_champions,
                "shifted_distance": self.use_shifted_distance,
                "evidence_gate": self.use_evidence_gate,
                "credit_coupling": self.use_credit_coupling,
                "step_credit": self.use_step_credit,
                "mating_credit": self.use_mating_credit,
            },
        }
        if algorithm is not None:
            algorithm.ref_dirs = self.ref_dirs.copy()
            algorithm.nd_archive_F = self.nd_archive_F.copy()
            algorithm.diagnostics = deepcopy(self.diagnostics)
            algorithm.adaptation_history = deepcopy(self.adaptation_history)
        self.generation += 1
        return pop[np.asarray(selected[:n_survive], dtype=int)]


class GASDE(NSGA2):
    """Geometry-Adaptive Shifted-Distance Ensemble.

    All geometry controls are explicit constructor arguments so experiments
    can disable one component at a time without modifying implementation code.
    """

    ALGO_FLAGS = {"multi", "many"}
    OBJECTIVE_SCOPE = "many"

    def __init__(
        self,
        pop_size: int = 100,
        ref_dirs: np.ndarray | None = None,
        sampling=None,
        selection=None,
        crossover=None,
        mutation=None,
        adaptation: bool = True,
        adaptation_interval: int = 10,
        adaptation_warmup: int = 5,
        adaptation_inertia: float = 0.75,
        adaptation_min_gain: float = 0.01,
        holdout_fraction: float = 0.30,
        empty_direction_patience: int = 3,
        uniform_anchor: float = 0.05,
        archive_multiplier: int = 4,
        scalarization: str = "apd",
        pbi_theta: float = 5.0,
        apd_alpha: float = 2.0,
        use_extreme_preservation: bool = True,
        use_sector_champions: bool = True,
        use_shifted_distance: bool = True,
        use_evidence_gate: bool = True,
        use_credit_coupling: bool = True,
        use_step_credit: bool | None = None,
        use_mating_credit: bool | None = None,
        **kwargs: Any,
    ) -> None:
        if int(pop_size) < 4:
            raise ValueError("pop_size must be at least 4.")
        survival = GASDESurvival(
            ref_dirs=ref_dirs,
            adaptation=adaptation,
            adaptation_interval=adaptation_interval,
            adaptation_warmup=adaptation_warmup,
            adaptation_inertia=adaptation_inertia,
            adaptation_min_gain=adaptation_min_gain,
            holdout_fraction=holdout_fraction,
            empty_direction_patience=empty_direction_patience,
            uniform_anchor=uniform_anchor,
            archive_multiplier=archive_multiplier,
            scalarization=scalarization,
            pbi_theta=pbi_theta,
            apd_alpha=apd_alpha,
            use_extreme_preservation=use_extreme_preservation,
            use_sector_champions=use_sector_champions,
            use_shifted_distance=use_shifted_distance,
            use_evidence_gate=use_evidence_gate,
            use_credit_coupling=use_credit_coupling,
            use_step_credit=use_step_credit,
            use_mating_credit=use_mating_credit,
        )
        super().__init__(
            pop_size=int(pop_size),
            sampling=LHS() if sampling is None else sampling,
            selection=TournamentSelection(func_comp=binary_tournament) if selection is None else selection,
            crossover=SBX(prob=0.9, eta=15) if crossover is None else crossover,
            mutation=PM(eta=20) if mutation is None else mutation,
            survival=survival,
            **kwargs,
        )
        self.ref_dirs = np.empty((0, 0), dtype=float)
        self.nd_archive_F = np.empty((0, 0), dtype=float)
        self.diagnostics: dict[str, Any] = {}
        self.adaptation_history: list[dict[str, Any]] = []

    def _setup(self, problem, **kwargs):
        super()._setup(problem, **kwargs)
        self.survival.configure(int(problem.n_obj), int(self.pop_size))
        self.ref_dirs = self.survival.ref_dirs.copy()
        self.nd_archive_F = self.survival.nd_archive_F.copy()
        self.diagnostics = {}
        self.adaptation_history = []


ALGORITHMS = {"GASDE": GASDE}


__all__ = [
    "ALGORITHM_FLAGS",
    "ALGORITHMS",
    "GASDE",
    "GASDESurvival",
    "robust_normalize",
    "shifted_distance_matrix",
]
