# emopylab 2026
"""NRV-MOEA (adaptive normal reference vector-based MOEA).

Reference:
Y. Hua, Q. Liu, and K. Hao. Adaptive normal vector guided evolutionary multi-
and many-objective optimization. Complex & Intelligent Systems, 2024,
10: 3709-3726.
"""

from __future__ import annotations

import numpy as np
from util.array_backend import backend_cdist

from algorithms.community_utils.base import LoopAlgorithm, decs, objs
from core.population import Population
from operators.utility_functions.CrowdingDistance import CrowdingDistance
from operators.utility_functions.NDSort import NDSort
from operators.utility_functions.OperatorGA import OperatorGA
from operators.sampling.lhs import LatinHypercubeSampling


ALGORITHM_FLAGS = {"NRVMOEA": {"binary", "integer", "label", "many", "multi", "permutation", "real"}}


def _ward_clustering(data, n_clusters):
    """
    Perform Ward hierarchical clustering.
    If scipy is available, it uses it for speed.
    Otherwise, uses a naive O(n^3) pure NumPy implementation.
    """
    import numpy as standard_np
    
    n = data.shape[0]
    if n <= n_clusters:
        return np.arange(n)
        
    try:
        from scipy.cluster.hierarchy import fcluster, linkage
        from scipy.spatial.distance import pdist
        data_cpu = standard_np.asarray(data) if not hasattr(data, 'get') else standard_np.asarray(data.get())
        z = linkage(pdist(data_cpu, metric="euclidean"), method="ward")
        return fcluster(z, t=n_clusters, criterion="maxclust") - 1
    except ImportError:
        pass
        
    data_cpu = standard_np.asarray(data) if not hasattr(data, 'get') else standard_np.asarray(data.get())
    
    clusters = [[i] for i in range(n)]
    means = standard_np.copy(data_cpu)
    sizes = standard_np.ones(n)
    
    diff = means[:, None, :] - means[None, :, :]
    dist_sq = standard_np.sum(diff**2, axis=-1)
    
    sz_prod = sizes[:, None] * sizes[None, :]
    sz_sum = sizes[:, None] + sizes[None, :]
    standard_np.fill_diagonal(sz_sum, 1)
    ward_d = sz_prod / sz_sum * dist_sq
    standard_np.fill_diagonal(ward_d, standard_np.inf)
    
    active = standard_np.ones(n, dtype=bool)
    n_active = n
    
    while n_active > n_clusters:
        valid_d = standard_np.where(active[:, None] & active[None, :], ward_d, standard_np.inf)
        min_idx = standard_np.argmin(valid_d)
        i = min_idx // n
        j = min_idx % n
        
        clusters[i].extend(clusters[j])
        clusters[j] = []
        active[j] = False
        
        new_size = sizes[i] + sizes[j]
        new_mean = (sizes[i]*means[i] + sizes[j]*means[j]) / new_size
        sizes[i] = new_size
        means[i] = new_mean
        
        diff_i = means[active] - means[i]
        dist_sq_i = standard_np.sum(diff_i**2, axis=-1)
        
        ward_i = (sizes[active] * sizes[i]) / (sizes[active] + sizes[i]) * dist_sq_i
        
        active_idx = standard_np.where(active)[0]
        ward_d[i, active_idx] = ward_i
        ward_d[active_idx, i] = ward_i
        ward_d[i, i] = standard_np.inf
        
        n_active -= 1
        
    labels = standard_np.zeros(n, dtype=int)
    c_idx = 0
    for c in clusters:
        if c:
            labels[standard_np.array(c)] = c_idx
            c_idx += 1
            
    return np.asarray(labels)


def _rng(algo):
    import numpy as np_std
    rng = getattr(algo, "random_state", None)
    if isinstance(rng, np_std.random.Generator):
        return rng
    if rng is None:
        rng = np_std.random.default_rng()
        algo.random_state = rng
        return rng
    return np_std.random.default_rng(int(rng))

def _sample_initial(problem, n, sampling, rng):
    if sampling is None:
        sampling = LatinHypercubeSampling()
    return sampling.do(problem, int(n), random_state=rng)


def _update_archive(population: Population, archive: Population | None, max_size: int) -> Population:
    """Maintains a non-dominated archive with a diversity-based truncation."""
    if archive is None or len(archive) == 0:
        archive = population
    else:
        archive = Population.merge(archive, population)

    objs = archive.get("F")
    if objs is None or len(objs) == 0:
        return archive

    _, unique_idx = np.unique(np.round(objs, 10), axis=0, return_index=True)
    archive = archive[unique_idx]

    front_no, _ = NDSort(archive.get("F"), len(archive))
    archive = archive[front_no == 1]

    n = len(archive)
    if n <= max_size:
        return archive

    f = np.asarray(archive.get("F"), dtype=float)
    f_min = f.min(axis=0)
    f_max = f.max(axis=0)
    denom = np.maximum(f_max - f_min, 1e-12)
    f_norm = (f - f_min) / denom

    i_mat = np.max(f_norm[:, None, :] - f_norm[None, :, :], axis=2)      # I(i, j) = max(f_i - f_j)
    c = np.maximum(np.max(np.abs(i_mat), axis=0), 1e-12)                  # column maxima C(j)
    f_fit = np.sum(-np.exp(-i_mat / c[None, :] / 0.05), axis=0) + 1.0      # F(j) sums over the rows i

    choose = np.arange(n)
    while len(choose) > max_size:
        x = int(np.argmin(f_fit[choose]))
        cx = choose[x]
        f_fit = f_fit + np.exp(-i_mat[cx, :] / c[cx] / 0.05)
        choose = np.delete(choose, x)

    archive = archive[choose]

    # Remove extreme outliers.
    o = np.asarray(archive.get("F"), dtype=float)
    o = o - o.min(axis=0)
    d = np.sqrt(np.sum(o ** 2, axis=1))
    mean_d = d.mean()
    if mean_d > 1e-12:
        archive = archive[d <= 10 * mean_d]

    return archive


def _asf_function(sol: np.ndarray, index: int, z_ideal: np.ndarray, z_nadir: np.ndarray) -> float:
    epsilon = 1.0e-6
    val = np.abs((sol - z_ideal) / np.maximum(z_nadir - z_ideal, 1e-12))
    mask = np.ones(len(sol), dtype=bool)
    if 0 <= index < len(sol):
        mask[index] = False
        val[mask] = val[mask] / epsilon
    return float(np.max(val))


def _asf_matrix(rows: np.ndarray, z_ideal: np.ndarray, z_nadir: np.ndarray,
                epsilon: float = 1.0e-6) -> np.ndarray:
    """Vectorized ASF: entry [k, j] is the ASF of ``rows[k]`` for axis ``j``.

    Equivalent to calling ``_asf_function(rows[k], j, ...)`` for every (k, j),
    but computed in one pass.  ``ASF_j(x) = max(val_j, max_{m!=j} val_m/eps)``
    with ``val = |(x - z_ideal)/(z_nadir - z_ideal)|``.
    """
    rows = np.asarray(rows, dtype=float)
    denom = np.maximum(z_nadir - z_ideal, 1e-12)
    val = np.abs((rows - z_ideal) / denom)         # (K, M)
    scaled = val / epsilon
    k, m = rows.shape
    amax = scaled.argmax(axis=1)
    mx = scaled.max(axis=1)
    tmp = scaled.copy()
    tmp[np.arange(k), amax] = -np.inf
    mx2 = tmp.max(axis=1)
    # max over m' != j of scaled[:, m']: drop the column-wise argmax when it is j.
    leave = np.where(
        np.arange(m)[None, :] == amax[:, None], mx2[:, None], mx[:, None]
    )
    return np.maximum(val, leave)


def _update_nadir_point(archive_obj: np.ndarray, z_ideal: np.ndarray, z_nadir: np.ndarray,
                        extrem_point: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    m = archive_obj.shape[1]
    # Vectorized extreme-point update (exact equivalent of the former double
    # loop over individuals x objectives).  For each axis j, replace the extreme
    # point with the archive solution of minimum ASF_j when it strictly improves.
    if archive_obj.shape[0] > 0:
        asf_arc = _asf_matrix(archive_obj, z_ideal, z_nadir)        # (N, M)
        best_i = asf_arc.argmin(axis=0)                             # (M,)
        best_val = asf_arc[best_i, np.arange(m)]
        asf_ext = _asf_matrix(extrem_point, z_ideal, z_nadir)       # (M, M)
        ext_diag = asf_ext[np.arange(m), np.arange(m)]
        replace = best_val < ext_diag
        if np.any(replace):
            extrem_point[replace] = archive_obj[best_i[replace]]

    temp = extrem_point - z_ideal
    rank = np.linalg.matrix_rank(temp)
    if rank == temp.shape[0]:
        try:
            al = np.linalg.solve(temp, np.ones((m, m)))
            for j in range(m):
                aj = 1.0 / al[j, 0] + z_ideal[j]
                if aj > z_ideal[j] and np.isfinite(aj):
                    z_nadir[j] = aj
                else:
                    break
        except Exception:
            z_nadir = archive_obj.max(axis=0)
    else:
        z_nadir = archive_obj.max(axis=0)
    return z_nadir, extrem_point


def _vertmap(arc_obj: np.ndarray, pop_obj: np.ndarray,
             hyperplane_bp: np.ndarray | None) -> tuple[np.ndarray, np.ndarray]:
    n, m = pop_obj.shape
    map_pop = np.zeros((n, m), dtype=float)

    rank = np.argsort(-arc_obj, axis=0, kind="stable")
    extreme = np.zeros(m, dtype=int)
    extreme[0] = rank[0, 0]
    for j in range(1, m):
        k = 0
        extreme[j] = rank[k, j]
        while (extreme[j] in extreme[:j]) and k < rank.shape[0] - 1:
            k += 1
            extreme[j] = rank[k, j]

    try:
        if arc_obj.shape[0] >= m:
            hyperplane = np.linalg.solve(arc_obj[extreme, :], np.ones(m))
        else:
            hyperplane = np.linalg.solve(pop_obj[extreme, :], np.ones(m))
    except Exception:
        hyperplane = np.ones(m, dtype=float)

    if not np.all(np.isfinite(hyperplane)):
        hyperplane = np.ones(m, dtype=float)

    for i in range(n):
        p = pop_obj[i]
        t1 = np.sum(p * hyperplane) - 1.0
        t2 = np.sum(hyperplane ** 2)
        for mm in range(m):
            map_pop[i, mm] = (
                -hyperplane[mm] * (t1 - hyperplane[mm] * p[mm]) +
                p[mm] * (t2 - hyperplane[mm] ** 2)
            ) / t2

    return map_pop, hyperplane


class NRVMOEA(LoopAlgorithm):
    """Adaptive normal-reference-vector MOEA: the last front of [population, archive, offspring] is projected onto
    the hyperplane through the archive extremes, Ward-clustered, and one solution per cluster (closest to the cluster
    centre and farthest beyond the hyperplane) joins the better fronts; an oversized result is truncated by removing
    the most crowded solutions while keeping one extreme per objective."""

    def __init__(self, pop_size: int = 100, sampling=None, **kwargs):
        super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)

    def start(self):
        f = objs(self.pop)
        self.z_nadir = f.max(axis=0)
        self.z_min = f.min(axis=0)
        self.scale = f.max(axis=0) - self.z_min
        self.nrv_archive = _update_archive(self.pop, None, self.N)
        self.extrem_point = np.full((f.shape[1], f.shape[1]), 10e30)
        with np.errstate(all="ignore"):
            _, self.hyperplane = _vertmap((objs(self.nrv_archive) - self.z_min) / self.scale,
                                          (f - self.z_min) / self.scale, None)

    def step(self):
        rng, N, M = self.rng, self.N, self.M
        self.nrv_archive = _update_archive(self.pop, self.nrv_archive, N)
        pop = Population.merge(self.pop, self.nrv_archive)
        off = self.evaluate(np.asarray(OperatorGA(self.problem, decs(pop[rng.integers(0, len(pop), size=N)]),
                                                  rng=rng), dtype=float))
        uni = Population.merge(pop, off)
        U = objs(uni)
        front_no, max_f = NDSort(U, N)
        pareto_p = np.where(front_no == max_f)[0]
        chosen = np.where(front_no < max_f)[0]
        ids = chosen[rng.permutation(len(chosen))]
        if len(pareto_p) < M:
            pareto_p = np.concatenate([pareto_p, ids[: 10 - len(pareto_p)]])     # literal: fills up to 10
        P = U[pareto_p]
        self.z_min = np.minimum(U.min(axis=0), P.min(axis=0))
        z_max = P.max(axis=0)
        self.z_nadir, self.extrem_point = _update_nadir_point(objs(self.nrv_archive), self.z_min, self.z_nadir,
                                                              self.extrem_point)
        if self.FE % int(np.ceil(0.1 * self.max_FE)) == 0:
            self.scale = z_max - self.z_min
        self.scale = np.where(self.scale == 0, 1e-5, self.scale)
        with np.errstate(all="ignore"):
            Pn = (P - self.z_min) / self.scale
            An = (objs(self.nrv_archive) - self.z_min) / self.scale
            map_pop, self.hyperplane = _vertmap(An, Pn, None)
        K = N - len(chosen)
        try:
            t = _ward_clustering(map_pop, K)
        except Exception:
            t = _ward_clustering(Pn, K)
        ep = []                                                   # indices into ``uni`` (unique by identity)
        hp = self.hyperplane
        for c in range(K):
            current = np.where(t == c)[0]
            if current.size == 0:                                 # an empty cluster contributes nothing
                ep = sorted(set(ep) | set(chosen.tolist()))
                continue
            ref = map_pop[current].mean(axis=0)
            if len(current) > 1:
                with np.errstate(all="ignore"):
                    d1 = np.linalg.norm(ref - map_pop[current], axis=1)
                    d2 = -(Pn[current] @ hp - 1.0) / np.sqrt(np.sum(hp ** 2))
                pick = current[int(np.argmin(d1 - d2))]
            else:
                pick = current[0]
            ep = sorted(set(ep) | {int(pareto_p[pick])} | set(chosen.tolist()))
        EP = uni[np.asarray(ep, dtype=int)]
        if len(EP) > N or self.FE >= 0.9 * self.max_FE:
            fno, mfno = NDSort(objs(EP), N)
            EP = EP[fno <= mfno]
            E = objs(EP)
            rank = np.argsort(E, axis=0, kind="stable")
            extreme = np.zeros(M, dtype=int)
            extreme[0] = rank[0, 0]
            for j in range(1, M):
                k = 0
                extreme[j] = rank[k, j]
                while extreme[j] in extreme[:j]:
                    k += 1
                    extreme[j] = rank[k, j]
            keep = np.ones(len(EP), dtype=bool)
            keep[extreme] = False
            temp, rest = EP[extreme], EP[keep]
            for _ in range(len(rest) - N):                        # literal: Extreme is emptied before the count
                O = (objs(rest) - self.z_min) / self.scale
                dis = backend_cdist(O, O)
                np.fill_diagonal(dis, 1e10)
                d = int(np.argmin(dis.min(axis=1)))
                rest = rest[np.delete(np.arange(len(rest)), d)]
            EP = Population.merge(rest, temp)
        self.pop = EP
