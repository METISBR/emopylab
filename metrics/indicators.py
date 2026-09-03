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

    def __init__(self, ref_point: np.ndarray | Sequence[float], **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.ref_point = np.asarray(ref_point, dtype=float)

    def do(self, F: np.ndarray) -> float:
        if F is None or len(F) == 0:
            return 0.0
        F = np.atleast_2d(np.asarray(F, dtype=float))
        M = F.shape[1]
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
            sample_num = 100_000
            samples = rng.uniform(low=min_val, high=self.ref_point, size=(sample_num, M))
            dom = np.zeros(sample_num, dtype=bool)
            for pt in F_valid:
                dom |= np.all(samples >= pt, axis=1)
            box_vol = np.prod(self.ref_point - min_val)
            return float(np.mean(dom) * box_vol)


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
