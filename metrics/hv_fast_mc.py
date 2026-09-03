import numpy as np


def HV_fast_MC(pop_obj: np.ndarray, optimum: np.ndarray, sample_num: int = 100_000) -> float:
    """Fast Hypervolume calculation with Adaptive Hybrid Dispatcher.

    Scientific Principle:
    - For M <= 3 (e.g. ZDT1, DTLZ2 2D/3D): Exact 2D/3D HV is O(N log N) (< 0.1 ms).
      Monte Carlo sampling in low dimensions is needlessly slow and introduces sampling noise.
    - For M >= 4 (Many-Objective, e.g. MaF, DTLZ 5-10D): Exact HV is #P-Hard (O(N^(M/2))).
      Fast Monte Carlo with Dynamic Sample Pruning (PlatEMO standard) is utilized.
    """
    if pop_obj.size == 0 or optimum.size == 0:
        return float("nan")

    pop_obj = np.atleast_2d(pop_obj)
    optimum = np.atleast_2d(optimum)

    if pop_obj.shape[1] != optimum.shape[1]:
        return float("nan")

    N, M = pop_obj.shape

    # 1. Normalization (matching PlatEMO's fmin / fmax scaling)
    fmin = np.minimum(np.min(pop_obj, axis=0), np.zeros(M))
    fmax = np.max(optimum, axis=0)

    den = (fmax - fmin) * 1.1
    den = np.where(np.abs(den) <= 1e-12, 1.0, den)

    pop_obj = (pop_obj - fmin) / den
    pop_obj = pop_obj[~np.any(pop_obj > 1.0, axis=1)]

    if pop_obj.size == 0:
        return 0.0

    ref_point = np.ones(M)

    # For M <= 3, execute exact O(N log N) calculation (instantaneous < 0.1ms and 100% exact)
    if M <= 3:
        try:
            from metrics.indicators import HV
            hv_calc = HV(ref_point=ref_point)
            return float(hv_calc(pop_obj))
        except Exception:
            pass

    max_value = ref_point
    min_value = np.min(pop_obj, axis=0)

    if np.any(max_value < min_value):
        return 0.0

    # 2. Monte Carlo Estimation with Dynamic Sample Pruning (for M >= 4)
    rng = np.random.default_rng(1)
    samples = rng.uniform(low=min_value, high=max_value, size=(sample_num, M))

    for i in range(pop_obj.shape[0]):
        if samples.shape[0] == 0:
            break

        # Match PlatEMO's short-circuit domination check
        domi = np.ones(samples.shape[0], dtype=bool)
        for m in range(M):
            if not np.any(domi):
                break
            domi &= (pop_obj[i, m] <= samples[:, m])

        # Dynamically shrink the sample pool
        samples = samples[~domi]

    score = np.prod(max_value - min_value) * (1.0 - (samples.shape[0] / sample_num))
    return float(score)


def _emopylab_wrapper(front: np.ndarray, context: dict) -> float:
    pf = context.get("pareto_front")
    if pf is None:
        return float("nan")
    return HV_fast_MC(front, pf)


METRICS = {
    "HV_fast_MC": _emopylab_wrapper
}
