# emopylab 2026
"""NRV-MOEA (adaptive normal reference vector-based MOEA).

Reference:
Y. Hua, Q. Liu, and K. Hao. Adaptive normal vector guided evolutionary multi-
and many-objective optimization. Complex & Intelligent Systems, 2024,
10: 3709-3726.
"""

from __future__ import annotations

from util.array_backend import xp as np
from util.array_backend import backend_cdist

from core.algorithm import Algorithm
from core.population import Population
from operators.utility_functions.CrowdingDistance import CrowdingDistance
from operators.utility_functions.NDSort import NDSort
from operators.utility_functions.OperatorGA import OperatorGA
from operators.utility_functions.TournamentSelection import TournamentSelection
from algorithms.community_utils.moead_family import max_fe, set_optimum_from_pop
from operators.sampling.rnd import FloatRandomSampling


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
    rng = getattr(algo, "random_state", None)
    if isinstance(rng, np.random.Generator):
        return rng
    if rng is None:
        rng = np.random.default_rng()
        algo.random_state = rng
        return rng
    return np.random.default_rng(int(rng))


def _sample_initial(problem, n, sampling, rng):
    if sampling is None:
        sampling = FloatRandomSampling()
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

    i_mat = np.zeros((n, n), dtype=float)
    for i in range(n):
        i_mat[i, :] = np.max(f_norm[i] - f_norm, axis=1)

    c = np.max(np.abs(i_mat), axis=1)
    c = np.maximum(c, 1e-12)
    f_fit = np.sum(-np.exp(-i_mat / c[:, None] / 0.05), axis=1) + 1.0

    choose = np.arange(n)
    while len(choose) > max_size:
        local_fit = f_fit[choose]
        x = int(np.argmin(local_fit))
        chosen_x = choose[x]
        f_fit = f_fit + np.exp(-i_mat[chosen_x, :] / c[chosen_x] / 0.05)
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

    rank = np.argsort(-arc_obj, axis=0)
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
            hyperplane = np.linalg.lstsq(arc_obj[extreme, :], np.ones(m), rcond=None)[0]
        else:
            hyperplane = np.linalg.lstsq(pop_obj[extreme, :], np.ones(m), rcond=None)[0]
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


class NRVMOEA(Algorithm):
    def __init__(self, pop_size: int = 100, sampling=None, n_max_evals: int | None = None, **kwargs):
        super().__init__(**kwargs)
        self.pop_size = int(pop_size)
        self.sampling = sampling
        self.n_max_evals = n_max_evals

    def _initialize_infill(self):
        rng = _rng(self)
        return _sample_initial(self.problem, self.pop_size, self.sampling, rng)

    def _initialize_advance(self, infills=None, **kwargs):
        self.pop = infills
        f = np.asarray(self.pop.get("F"), dtype=float)

        self.z_nadir = f.max(axis=0)
        self.z_min = f.min(axis=0)
        self.z_max = f.max(axis=0)
        self.scale = self.z_max - self.z_min
        self.scale = np.where(np.abs(self.scale) <= 1e-12, 1.0, self.scale)

        self.nrv_archive = _update_archive(self.pop, None, self.pop_size)
        self.extrem_point = np.full((f.shape[1], f.shape[1]), 1e31, dtype=float)

        pop_obj_n = (f - self.z_min) / self.scale
        arc_obj_n = (np.asarray(self.nrv_archive.get("F"), dtype=float) - self.z_min) / self.scale
        self.hyperplane_bp = np.array([], dtype=float)
        _, self.hyperplane = _vertmap(arc_obj_n, pop_obj_n, self.hyperplane_bp)

    def _infill(self):
        rng = _rng(self)
        self._prev_pop = self.pop.copy()
        self.nrv_archive = _update_archive(self.pop, self.nrv_archive, self.pop_size)

        mating_pool = rng.integers(0, len(self.pop), size=self.pop_size)
        # Pass decision vectors (not a Population) so the operator does NOT
        # evaluate internally via problem.evaluate (which bypasses the
        # algorithm's evaluator and would leave these offspring uncounted).
        # The returned, still-unevaluated offspring are evaluated and counted by
        # the framework before _advance, keeping n_eval honest for a fair budget.
        off_dec = OperatorGA(self.problem, self.pop[mating_pool].get("X"), rng=rng)
        if hasattr(off_dec, "get"):
            off_dec = off_dec.get("X")
        off_dec = np.atleast_2d(np.asarray(off_dec, dtype=float))
        offspring = Population.new("X", off_dec)
        self._offspring = offspring

        # Prepare data structures used in advance.
        self._union_pop = Population.merge(self.pop, self.nrv_archive)
        self._union_pop = Population.merge(self._union_pop, offspring)
        return offspring

    def _advance(self, infills=None, **kwargs):
        pop = self._prev_pop
        offspring = self._offspring
        union_pop = self._union_pop
        union_obj = np.asarray(union_pop.get("F"), dtype=float)

        front_no, max_f_no = NDSort(union_obj, self.pop_size)
        pareto_p = np.where(front_no == max_f_no)[0]
        chosen_mask = front_no < max_f_no
        chosen_idx = np.where(chosen_mask)[0]
        chosen_pop = union_pop[chosen_idx]

        # Randomize other indices for backup.
        other_idx = chosen_idx.copy()
        rng = _rng(self)
        rng.shuffle(other_idx)

        if len(pareto_p) < self.problem.n_obj:
            need = self.problem.n_obj - len(pareto_p)
            extra = other_idx[:need]
            pareto_p = np.concatenate([pareto_p, extra])

        pop_obj = union_obj[pareto_p]

        self.z_min = union_obj.min(axis=0)
        self.z_max = pop_obj.max(axis=0)
        self.z_min = np.minimum(self.z_min, pop_obj.min(axis=0))

        archive_obj = np.asarray(self.nrv_archive.get("F"), dtype=float)
        self.z_nadir, self.extrem_point = _update_nadir_point(
            archive_obj, self.z_min, self.z_nadir, self.extrem_point
        )

        max_evals = self.n_max_evals if self.n_max_evals is not None else max_fe(self)

        fe = self.evaluator.n_eval
        if max_evals > 0 and fe > 0 and fe % max(1, int(np.ceil(0.1 * max_evals))) == 0:
            self.scale = self.z_max - self.z_min
            self.scale = np.where(np.abs(self.scale) <= 1e-12, 1.0, self.scale)
            self.hyperplane_bp = self.hyperplane.copy()

        self.scale = np.where(np.abs(self.scale) <= 1e-12, 1e-12, self.scale)
        pop_obj_n = (pop_obj - self.z_min) / self.scale
        arc_obj_n = (archive_obj - self.z_min) / self.scale
        map_pop, self.hyperplane = _vertmap(arc_obj_n, pop_obj_n, self.hyperplane_bp)

        n_clusters = min(self.pop_size - len(chosen_pop), len(pareto_p))
        if n_clusters <= 0:
            self.pop = chosen_pop
            return

        try:
            t = _ward_clustering(map_pop, n_clusters)
        except Exception:
            t = _ward_clustering(pop_obj_n, n_clusters)

        ep = Population()
        seen_labels = set()
        for c in range(n_clusters):
            current = np.where(t == c)[0]
            if current.size == 0:
                continue
            seen_labels.add(c)
            pn = len(current)
            ref = np.sum(map_pop[current, :], axis=0) / pn
            if pn > 1:
                d12 = np.zeros(pn, dtype=float)
                for pc in range(pn):
                    d1 = np.linalg.norm(ref - map_pop[current[pc], :])
                    d2 = -(pop_obj_n[current[pc], :] @ self.hyperplane - 1.0) / np.sqrt(np.sum(self.hyperplane ** 2))
                    d12[pc] = d1 - d2
                ct = int(np.argmin(d12))
                choose = current[ct]
            else:
                choose = current[0]

            ep = Population.merge(ep, union_pop[[pareto_p[choose]]])
            ep = Population.merge(ep, chosen_pop)
            _, uidx = np.unique(np.round(ep.get("F"), 10), axis=0, return_index=True)
            ep = ep[uidx]

        # Ensure requested number of clusters are represented; fill missing ones.
        missing = n_clusters - len(seen_labels)
        if missing > 0 and len(pareto_p) > len(seen_labels):
            leftover = [i for i in range(len(pareto_p)) if i not in seen_labels]
            rng.shuffle(leftover)
            for i in leftover[:missing]:
                ep = Population.merge(ep, union_pop[[pareto_p[i]]])
                ep = Population.merge(ep, chosen_pop)
                _, uidx = np.unique(np.round(ep.get("F"), 10), axis=0, return_index=True)
                ep = ep[uidx]

        if len(ep) > self.pop_size or fe >= 0.9 * max_evals:
            ep_front_no, ep_max_f_no = NDSort(ep.get("F"), self.pop_size)
            ep = ep[ep_front_no <= ep_max_f_no]
            ep_obj = np.asarray(ep.get("F"), dtype=float)
            rank = np.argsort(ep_obj, axis=0)
            extreme = np.zeros(self.problem.n_obj, dtype=int)
            extreme[0] = rank[0, 0]
            for j in range(1, len(extreme)):
                k = 0
                extreme[j] = rank[k, j]
                while extreme[j] in extreme[:j] and k < rank.shape[0] - 1:
                    k += 1
                    extreme[j] = rank[k, j]

            ep_temp = ep[extreme]
            ep_mask = np.ones(len(ep), dtype=bool)
            ep_mask[extreme] = False
            ep_rest = ep[ep_mask]
            n_remove = len(ep_rest) - self.pop_size + len(extreme)
            ep_indices = list(range(len(ep_rest)))
            while n_remove > 0 and len(ep_indices) > 0:
                obj = np.asarray(ep_rest[ep_indices].get("F"), dtype=float)
                norm_obj = (obj - self.z_min) / self.scale
                dis = backend_cdist(norm_obj, norm_obj, metric="euclidean")
                np.fill_diagonal(dis, 1e10)
                mindis = dis.min(axis=1)
                del_idx = int(np.argmin(mindis))
                ep_indices.pop(del_idx)
                n_remove -= 1

            ep = Population.merge(ep_rest[ep_indices], ep_temp)

        self.pop = ep

    def _set_optimum(self):
        set_optimum_from_pop(self)
