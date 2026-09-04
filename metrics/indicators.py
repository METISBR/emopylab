"""EmoPyLab Native Hypervolume and Quality Indicators."""

from __future__ import annotations

from typing import Any, Optional
import numpy as np


class Indicator:
    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.do(*args, **kwargs)

    def do(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError


def _exact_2d_hv(F: np.ndarray, ref_point: np.ndarray) -> float:
    """Exact 2D hypervolume using sweep-line algorithm O(N log N)."""
    # Filter points dominated by ref_point
    valid = np.all(F <= ref_point, axis=1)
    F = F[valid]
    if F.shape[0] == 0:
        return 0.0

    # Sort by first objective ascending, second descending
    sorted_idx = np.lexsort((-F[:, 1], F[:, 0]))
    F = F[sorted_idx]

    # Filter out dominated points in 2D
    non_dom = []
    current_min_y = np.inf
    for pt in F:
        if pt[1] < current_min_y:
            non_dom.append(pt)
            current_min_y = pt[1]

    if not non_dom:
        return 0.0

    pts = np.array(non_dom)
    hv = 0.0
    # Add rectangles
    for i in range(len(pts)):
        width = (pts[i + 1, 0] if i + 1 < len(pts) else ref_point[0]) - pts[i, 0]
        height = ref_point[1] - pts[i, 1]
        if width > 0 and height > 0:
            hv += width * height
    return float(hv)


def _exact_3d_hv_inclusion_exclusion(F: np.ndarray, ref_point: np.ndarray) -> float:
    """Exact 3D hypervolume via 2D cross-section slicing."""
    valid = np.all(F <= ref_point, axis=1)
    F = F[valid]
    if F.shape[0] == 0:
        return 0.0

    # Sort uniquely by z-coordinate
    z_coords = np.unique(np.append(F[:, 2], ref_point[2]))
    z_coords = np.sort(z_coords)

    total_hv = 0.0
    for i in range(len(z_coords) - 1):
        z_low = z_coords[i]
        z_high = z_coords[i + 1]
        dz = z_high - z_low
        if dz <= 0:
            continue
        # Points that dominate this slice in z
        active = F[F[:, 2] <= z_low]
        if active.shape[0] > 0:
            slice_2d_hv = _exact_2d_hv(active[:, :2], ref_point[:2])
            total_hv += slice_2d_hv * dz

    return float(total_hv)


class HV(Indicator):
    """Hypervolume quality indicator with pure NumPy implementation."""

    def __init__(self, ref_point: np.ndarray | Sequence[float], method: str = "auto", **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.ref_point = np.asarray(ref_point, dtype=float)
        self.method = str(method).lower().strip()

    def do(self, F: np.ndarray) -> float:
        if F is None or len(F) == 0:
            return 0.0
        F = np.atleast_2d(np.asarray(F, dtype=float))
        M = F.shape[1]

        if self.method == "iqhv":
            from metrics.iqhv import iqhv
            return iqhv(F, self.ref_point)
        elif self.method == "hbda":
            from metrics.hbda import hbda
            return hbda(F, self.ref_point)

        if M == 2:
            return _exact_2d_hv(F, self.ref_point)
        elif M == 3:
            return _exact_3d_hv_inclusion_exclusion(F, self.ref_point)
        else:
            # For M >= 4, use Monte Carlo approximation against reference point
            valid = np.all(F <= self.ref_point, axis=1)
            F_valid = F[valid]
            if F_valid.shape[0] == 0:
                return 0.0
            min_val = np.min(F_valid, axis=0)
            rng = np.random.default_rng(1)
            sample_num = int(self.kwargs.get("sample_num", 10_000))
            samples = rng.uniform(low=min_val, high=self.ref_point, size=(sample_num, M))
            dom = np.zeros(sample_num, dtype=bool)
            for pt in F_valid:
                dom |= np.all(samples >= pt, axis=1)
            box_vol = np.prod(self.ref_point - min_val)
            return float(np.mean(dom) * box_vol)

class IQHV(Indicator):
    """Improved Quick Hypervolume (IQHV) indicator (Jaszkiewicz, 2018)."""

    def __init__(self, ref_point: np.ndarray | Sequence[float], **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.ref_point = np.asarray(ref_point, dtype=float)

    def do(self, F: np.ndarray) -> float:
        from metrics.iqhv import iqhv
        return iqhv(F, self.ref_point)


class HBDA(Indicator):
    """Hypervolume Box Decomposition Algorithm (HBDA) indicator (Lacour et al., 2017)."""

    def __init__(self, ref_point: np.ndarray | Sequence[float], **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.ref_point = np.asarray(ref_point, dtype=float)

    def do(self, F: np.ndarray) -> float:
        from metrics.hbda import hbda
        return hbda(F, self.ref_point)


class QEHVC(Indicator):
    """Quick Extreme Hypervolume Contribution (QEHVC) indicator (Jaszkiewicz & Zielniewicz, 2021)."""

    def __init__(self, ref_point: np.ndarray | Sequence[float], **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.ref_point = np.asarray(ref_point, dtype=float)

    def do(self, F: np.ndarray) -> np.ndarray:
        from metrics.qehvc import qehvc
        return qehvc(F, self.ref_point)

class IGD(Indicator):
    """Inverted Generational Distance (IGD) indicator."""

    def __init__(self, pf: np.ndarray | Sequence[float], **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.pf = np.atleast_2d(np.asarray(pf, dtype=float))

    def do(self, F: np.ndarray) -> float:
        if F is None or len(F) == 0:
            return float("nan")
        F = np.atleast_2d(np.asarray(F, dtype=float))
        # Euclidean distance from each point in PF to nearest in F
        dists = np.min(np.sqrt(np.sum((self.pf[:, None, :] - F[None, :, :]) ** 2, axis=2)), axis=1)
        return float(np.mean(dists))


class IGDPlus(Indicator):
    """IGD+ (Modified Inverted Generational Distance) indicator (Ishibuchi et al., 2015)."""

    def __init__(self, pf: np.ndarray | Sequence[float], **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.pf = np.atleast_2d(np.asarray(pf, dtype=float))

    def do(self, F: np.ndarray) -> float:
        if F is None or len(F) == 0:
            return float("nan")
        F = np.atleast_2d(np.asarray(F, dtype=float))
        # Modified distance: max(F - PF, 0)
        diff = np.maximum(F[None, :, :] - self.pf[:, None, :], 0.0)
        dists = np.min(np.sqrt(np.sum(diff ** 2, axis=2)), axis=1)
        return float(np.mean(dists))


class GD(Indicator):
    """Generational Distance (GD) indicator."""

    def __init__(self, pf: np.ndarray | Sequence[float], **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.pf = np.atleast_2d(np.asarray(pf, dtype=float))

    def do(self, F: np.ndarray) -> float:
        if F is None or len(F) == 0:
            return float("nan")
        F = np.atleast_2d(np.asarray(F, dtype=float))
        dists = np.min(np.sqrt(np.sum((F[:, None, :] - self.pf[None, :, :]) ** 2, axis=2)), axis=1)
        return float(np.mean(dists))


class GDPlus(Indicator):
    """GD+ (Modified Generational Distance) indicator."""

    def __init__(self, pf: np.ndarray | Sequence[float], **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.pf = np.atleast_2d(np.asarray(pf, dtype=float))

    def do(self, F: np.ndarray) -> float:
        if F is None or len(F) == 0:
            return float("nan")
        F = np.atleast_2d(np.asarray(F, dtype=float))
        diff = np.maximum(self.pf[None, :, :] - F[:, None, :], 0.0)
        dists = np.min(np.sqrt(np.sum(diff ** 2, axis=2)), axis=1)
        return float(np.mean(dists))


def _adaptive_r2_partitions(n_obj: int) -> int:
    """Calculates canonical Das-Dennis partitions to avoid combinatorial explosion."""
    m = int(n_obj)
    if m <= 2:
        return 99  # 100 weight vectors
    elif m == 3:
        return 13  # 105 weight vectors
    elif m == 4:
        return 7   # 120 weight vectors
    elif m == 5:
        return 5   # 126 weight vectors
    else:
        return 3   # e.g., M=8 -> 120, M=10 -> 220, M=15 -> 680


class R2(Indicator):
    """R2 Performance Indicator (Hansen & Jaszkiewicz, 1998; Brockhoff et al., 2012).

    Evaluates convergence and spread of an approximation front F against a set of
    weight vectors W using the classical multiply-form Tchebycheff scalarizing utility:
        u_w(a) = max_m { w_m * |a_m - z*_m| }
        R2(A, W, z*) = (1 / |W|) * sum_{w in W} min_{a in A} u_w(a)
    """

    def __init__(
        self,
        pf: Optional[np.ndarray | Sequence[float]] = None,
        ref_dirs: Optional[np.ndarray] = None,
        n_partitions: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.pf = np.atleast_2d(np.asarray(pf, dtype=float)) if pf is not None else None
        self.ref_dirs = np.atleast_2d(np.asarray(ref_dirs, dtype=float)) if ref_dirs is not None else None
        self.n_partitions = int(n_partitions) if n_partitions is not None else None

    def do(self, F: np.ndarray) -> float:
        if F is None:
            return float("nan")
        F_arr = np.asarray(F, dtype=float)
        if F_arr.size == 0:
            return float("nan")
        if not np.all(np.isfinite(F_arr)):
            raise ValueError("R2 indicator input front contains non-finite values (NaN or Inf)")
        if F_arr.ndim == 1:
            F_arr = F_arr.reshape(1, -1)
        if self.pf is not None:
            if self.pf.size == 0:
                return float("nan")
            if not np.all(np.isfinite(self.pf)):
                raise ValueError("R2 indicator reference front contains non-finite values (NaN or Inf)")
            if self.pf.shape[1] != F_arr.shape[1]:
                raise ValueError(
                    f"Reference front dimension {self.pf.shape[1]} does not match front dimension {F_arr.shape[1]}"
                )

        N, M = F_arr.shape
        if M < 2:
            raise ValueError(f"R2 indicator requires at least 2 objectives, got {M}")

        # Weight generation via Das-Dennis simplex lattice
        if self.ref_dirs is not None:
            W = np.atleast_2d(np.asarray(self.ref_dirs, dtype=float))
        else:
            from core.tensor.ref_dirs import get_reference_directions
            p = self.n_partitions if self.n_partitions is not None else _adaptive_r2_partitions(M)
            W = get_reference_directions("das-dennis", n_obj=M, n_partitions=p)

        if W.ndim != 2 or W.shape[1] != M:
            raise ValueError(f"Weight vectors dimension {W.shape} does not match front objectives {M}")

        row_sums = np.sum(W, axis=1)
        if np.any(row_sums <= 1e-12):
            raise ValueError("Weight vectors must not contain all-zero rows")

        # Subsample deterministically if weight lattice exceeds 2000 directions
        if W.shape[0] > 2000:
            sub_idx = np.random.default_rng(0).choice(W.shape[0], size=2000, replace=False)
            W = W[np.sort(sub_idx)]
        # Coordinate normalization
        if self.pf is not None and self.pf.size > 0:
            z_min = np.min(self.pf, axis=0)
            z_max = np.max(self.pf, axis=0)
        else:
            z_min = np.min(F_arr, axis=0)
            z_max = np.max(F_arr, axis=0)

        den = z_max - z_min
        den = np.where(np.abs(den) <= 1e-12, 1.0, den)

        F_norm = (F_arr - z_min) / den
        z_ideal = np.zeros(M, dtype=float)

        # Vectorized 3D Tensor Broadcasting:
        # diff: [N, 1, M], W: [1, K, M] -> utility: [N, K, M]
        diff = np.abs(F_norm[:, None, :] - z_ideal[None, None, :])
        utility = W[None, :, :] * diff
        chebyshev = np.max(utility, axis=2)  # [N, K]
        min_over_pop = np.min(chebyshev, axis=0)  # [K]

        return float(np.mean(min_over_pop))
