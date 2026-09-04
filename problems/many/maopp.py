from __future__ import annotations

"""
MaOPP benchmark suite.

Reference
---------
J. Weise and S. Mostaghim.
A scalable many-objective pathfinding benchmark suite.
IEEE Transactions on Evolutionary Computation, 2022, 26(1): 188-194.
"""

import math
from dataclasses import dataclass

import numpy as np
from core.problem import Problem


def _map_linear(x: float, in_min: float, in_max: float, out_min: float, out_max: float) -> float:
    return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min


def _heaviside(x: float) -> float:
    if x > 0:
        return 1.0
    if x < 0:
        return 0.0
    return 0.5


def _lookup(matrix: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    xi = np.clip(np.rint(x).astype(int), 1, matrix.shape[0]) - 1
    yi = np.clip(np.rint(y).astype(int), 1, matrix.shape[1]) - 1
    return matrix[xi, yi]


@dataclass
class _MaOPPConfig:
    x_max: int = 10
    y_max: int = 10
    obstacle_value: int = 0
    nh: int = 1
    neighborhood: int = 2
    backtracking: int = 0
    allow_obstacles_on_path: int = 1
    overhead_variables_factor: float = 1.5


class _BaseMaOPP(Problem):
    def __init__(self, *, binary: bool, **kwargs):
        cfg = _MaOPPConfig(
            x_max=int(kwargs.pop("x_max", 10)),
            y_max=int(kwargs.pop("y_max", 10)),
            obstacle_value=int(kwargs.pop("obstacleValue", kwargs.pop("obstacle_value", 0))),
            nh=int(kwargs.pop("nh", 1)),
            neighborhood=int(kwargs.pop("neighbourhood", kwargs.pop("neighborhood", 2))),
            backtracking=int(kwargs.pop("backtracking", 0)),
            allow_obstacles_on_path=int(
                kwargs.pop("allowObstaclesOnPath", kwargs.pop("allow_obstacles_on_path", 1))
            ),
            overhead_variables_factor=float(
                kwargs.pop("overheadVariablesFactor", kwargs.pop("overhead_variables_factor", 1.5))
            ),
        )

        self.x_max = max(2, cfg.x_max)
        self.y_max = max(2, cfg.y_max)
        self.obstacle_value = int(np.clip(cfg.obstacle_value, 0, 2))
        self.nh = int(np.clip(cfg.nh, 1, 4))
        self.neighborhood = 3 if int(cfg.neighborhood) >= 3 else 2
        self.backtracking = 1 if int(cfg.backtracking) else 0
        self.allow_obstacles_on_path = 1 if int(cfg.allow_obstacles_on_path) else 0
        self.overhead_variables_factor = float(max(0.1, cfg.overhead_variables_factor))

        self.vmax_high = float(kwargs.pop("vmax_high", 130.0))
        self.vmax_medium = float(kwargs.pop("vmax_medium", 100.0))
        self.vmax_low = float(kwargs.pop("vmax_low", 50.0))

        self.manhattan_steps = int(self.x_max + self.y_max - 2)
        self.path_length = max(1, int(round(self.manhattan_steps * self.overhead_variables_factor)))

        self.binary = bool(binary)
        self.number_of_bits = 0
        if self.binary:
            self._set_binary_encoding()
            xl = np.zeros(self.n_var, dtype=float)
            xu = np.ones(self.n_var, dtype=float)
            vtype = int
        else:
            self.n_var = self.path_length
            xl = np.zeros(self.n_var, dtype=float)
            xu = np.ones(self.n_var, dtype=float)
            vtype = float

        super().__init__(n_var=self.n_var, n_obj=5, n_ieq_constr=1, xl=xl, xu=xu, vtype=vtype, **kwargs)

        self.v_max = np.zeros((self.x_max, self.y_max), dtype=float)
        self.elevation = np.zeros((self.x_max, self.y_max), dtype=float)
        self.upper_bounds_for_objectives = np.zeros(5, dtype=float)
        self._build_maps()

    def _set_binary_encoding(self) -> None:
        if self.neighborhood == 2 and self.backtracking == 0:
            self.number_of_bits = 1
        elif self.neighborhood == 2 and self.backtracking == 1:
            self.number_of_bits = 2
        elif self.neighborhood == 3 and self.backtracking == 0:
            self.number_of_bits = 2
        else:
            self.number_of_bits = 3
        self.n_var = self.path_length * self.number_of_bits

    def _build_maps(self) -> None:
        obstacle = self.obstacle_value
        for x in range(1, self.x_max + 1):
            for y in range(1, self.y_max + 1):
                w_xy = max(math.sin(x - 1), math.cos(y - 1))
                if w_xy > 0.9:
                    vel = self.vmax_high
                elif w_xy < -0.4:
                    vel = self.vmax_low
                else:
                    vel = self.vmax_medium

                if obstacle == 1:
                    cond = (
                        math.copysign(1.0, math.sin(math.pi / 2 + math.pi * x))
                        + math.copysign(1.0, math.sin(math.pi / 2 + math.pi * y))
                        - 2.0
                        * (_heaviside(x - self.x_max + 0.5) - _heaviside(x - self.x_max - 0.5))
                        * (_heaviside(y - self.y_max + 0.5) - _heaviside(y - self.y_max - 0.5))
                    )
                    if cond == 2:
                        vel = 0.0
                elif obstacle == 2:
                    if (x - 1 - self.x_max / 2.0) ** 2 + (y - 1 - self.y_max / 2.0) ** 2 - (0.25 * self.x_max) ** 2 < 0:
                        vel = 0.0

                self.v_max[x - 1, y - 1] = vel

                xs = _map_linear(x, 1, self.x_max + 1, -3, 3)
                ys = _map_linear(y, 1, self.y_max + 1, -3, 3)
                if self.nh == 1:
                    h = 5 * math.exp(-((xs - 1.5) ** 2) - ((ys + 1.5) ** 2))
                elif self.nh == 2:
                    h = 5 * math.exp(-((xs + 1.5) ** 2) - ((ys + 1.5) ** 2)) + 5 * math.exp(-((xs - 1.5) ** 2) - ((ys - 1.5) ** 2))
                elif self.nh == 3:
                    h = (
                        5 * math.exp(-((xs + 1.5) ** 2) - ((ys + 1.5) ** 2))
                        + 5 * math.exp(-((xs - 1.5) ** 2) - ((ys - 1.5) ** 2))
                        + 5 * math.exp(-((xs - 1.5) ** 2) - ((ys + 1.5) ** 2))
                    )
                else:
                    h = (
                        3 * (1 - xs) ** 2 * math.exp(-(xs**2) - (ys + 1) ** 2)
                        - 10 * math.exp(-(xs**2) - ys**2) * (-(xs**3) + xs / 5 - ys**5)
                        - (1 / 3) * math.exp(-(xs + 1) ** 2 - ys**2)
                    )
                self.elevation[x - 1, y - 1] = h

        self.upper_bounds_for_objectives = np.array(
            [
                self.manhattan_steps * 1.5,
                self.manhattan_steps * 1.5,
                1.5 * 5 * self.nh,
                1.5 * self.manhattan_steps / 50,
                0.5 * self.manhattan_steps * math.pi / 2,
            ],
            dtype=float,
        )

    def _get_possible_neighbors(self, x_curr: int, y_curr: int) -> np.ndarray:
        e = np.array([1, 0], dtype=int)
        se = np.array([1, 1], dtype=int)
        s = np.array([0, 1], dtype=int)
        sw = np.array([-1, 1], dtype=int)
        w = np.array([-1, 0], dtype=int)
        nw = np.array([-1, -1], dtype=int)
        n = np.array([0, -1], dtype=int)
        ne = np.array([1, -1], dtype=int)

        if self.backtracking == 0:
            neighbors = np.vstack([e, s]) if self.neighborhood == 2 else np.vstack([e, se, s])
        else:
            neighbors = np.vstack([e, s, w, n]) if self.neighborhood == 2 else np.vstack([e, se, s, sw, w, nw, n, ne])

        current = np.array([x_curr, y_curr], dtype=int)
        next_coords = neighbors + current
        inside = (
            (next_coords[:, 0] >= 1)
            & (next_coords[:, 0] <= self.x_max)
            & (next_coords[:, 1] >= 1)
            & (next_coords[:, 1] <= self.y_max)
        )
        neighbors = neighbors[inside]

        if self.allow_obstacles_on_path == 0 and neighbors.size > 0:
            next_coords = neighbors + current
            # Keep the coordinate swap used in the MATLAB getPossNeighbours.
            obstacles = _lookup(self.v_max, next_coords[:, 1], next_coords[:, 0]) == 0.0
            neighbors = neighbors[~obstacles]

        return neighbors

    def _decode_path_real_values(self, pop_dec: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
        x = np.asarray(pop_dec, dtype=float)
        if x.ndim == 1:
            x = x[None, :]
        n, d = x.shape

        x_coords = np.zeros((n, d + 1), dtype=int)
        y_coords = np.zeros((n, d + 1), dtype=int)
        x_coords[:, 0] = 1
        y_coords[:, 0] = 1

        for i in range(n):
            x_current = 1
            y_current = 1
            for j in range(d):
                possible = self._get_possible_neighbors(x_current, y_current)
                no_possible = int(possible.shape[0])
                if no_possible == 0:
                    break

                value = float(x[i, j])
                chosen = no_possible - 1
                for a in range(1, no_possible + 1):
                    if value >= (a - 1) / no_possible and value < a / no_possible:
                        chosen = a - 1
                        break

                x_add, y_add = possible[chosen]
                x_current += int(x_add)
                y_current += int(y_add)
                x_coords[i, j + 1] = x_current
                y_coords[i, j + 1] = y_current
                if x_current == self.x_max and y_current == self.y_max:
                    break
        return x_coords, y_coords, d

    def _decode_path(self, pop_dec: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
        x = np.asarray(pop_dec)
        if x.ndim == 1:
            x = x[None, :]
        if not self.binary:
            return self._decode_path_real_values(np.clip(x.astype(float), 0.0, 1.0))

        bits = (x > 0.5).astype(int)
        n_bits = int(max(1, self.number_of_bits))
        usable = (bits.shape[1] // n_bits) * n_bits
        bits = bits[:, :usable]
        steps = bits.shape[1] // n_bits

        real_values = np.zeros((bits.shape[0], steps), dtype=float)
        powers = 2.0 ** np.arange(n_bits, dtype=float)
        for i in range(steps):
            chunk = bits[:, i * n_bits : (i + 1) * n_bits]
            dec = np.sum(chunk * powers[None, :], axis=1)
            real_values[:, i] = dec / (2.0**n_bits)
        return self._decode_path_real_values(real_values)

    def _objective_values_path(self, pop_dec: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        x_coords, y_coords, d = self._decode_path(pop_dec)
        n = x_coords.shape[0]

        f1 = np.zeros(n, dtype=float)
        f2 = np.zeros(n, dtype=float)
        f3 = np.zeros(n, dtype=float)
        f4 = np.zeros(n, dtype=float)
        f5 = np.zeros(n, dtype=float)
        f_obstacle = np.zeros(n, dtype=float)
        distances = np.zeros((n, d), dtype=float)

        for j in range(d):
            x1 = x_coords[:, j].astype(float)
            x2 = x_coords[:, j + 1].astype(float)
            y1 = y_coords[:, j].astype(float)
            y2 = y_coords[:, j + 1].astype(float)

            has_ended = (x2 != 0.0) & (x1 != 0.0)
            x1[x1 == 0.0] = 1.0
            y1[y1 == 0.0] = 1.0
            x2[x2 == 0.0] = 1.0
            y2[y2 == 0.0] = 1.0

            v1 = _lookup(self.v_max, x1, y1)
            v2 = _lookup(self.v_max, x2, y2)

            if self.allow_obstacles_on_path == 1:
                f_obstacle += ((v1 == 0.0) | (v2 == 0.0)).astype(float)

            distances[:, j] = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            res = distances[:, j] * has_ended
            res[res > math.sqrt(2.0)] = 0.0
            f1 += res

            delay = (v1 != v2).astype(float) * 2.0
            same = v1 == v2
            same_vel = np.zeros(n, dtype=float)
            same_vel[same] = v1[same]
            delay += (same_vel == 50.0).astype(float) * 3.0
            delay += (same_vel == 100.0).astype(float) * 1.0
            delay += (delay == 0.0).astype(float) * 0.2
            f2 += delay * has_ended

            h1 = _lookup(self.elevation, x1, y1)
            h2 = _lookup(self.elevation, x2, y2)
            f3 += ((h1 < h2).astype(float) * (h2 - h1)) * has_ended

            with np.errstate(divide="ignore", invalid="ignore"):
                travel = 2.0 * distances[:, j] / (v1 + v2)
            travel[~np.isfinite(travel)] = 0.0
            f4 += travel * has_ended

            if j > 0:
                x0 = x_coords[:, j - 1].astype(float)
                y0 = y_coords[:, j - 1].astype(float)
                x0[x0 < 1.0] = 1.0
                y0[y0 < 1.0] = 1.0

                dotp = (x0 - x1) * (x1 - x2) + (y0 - y1) * (y1 - y2)
                den = distances[:, j - 1] * distances[:, j]
                with np.errstate(divide="ignore", invalid="ignore"):
                    cos_theta = np.divide(dotp, den)
                cos_theta = np.clip(cos_theta, -1.0, 1.0)
                angle = np.arccos(cos_theta) * has_ended
                angle[~np.isfinite(angle)] = 0.0
                f5 += angle

        objectives = np.column_stack([f1, f2, f3, f4, f5])

        mask = x_coords != 0
        last_idx = x_coords.shape[1] - 1 - np.argmax(mask[:, ::-1], axis=1)
        last_x = x_coords[np.arange(n), last_idx]
        last_y = y_coords[np.arange(n), last_idx]
        dist_to_target = np.abs(self.x_max - last_x) + np.abs(self.y_max - last_y)
        return objectives, dist_to_target.astype(float), f_obstacle

    def _evaluate(self, x, out, *args, **kwargs):
        arr = np.asarray(x, dtype=float)
        if arr.ndim == 1:
            arr = arr[None, :]
        arr = np.clip(arr, self.xl, self.xu)

        objectives, dist_to_target, f_obstacle = self._objective_values_path(arr)
        penalized = objectives.copy()
        penalized[:, 0] = penalized[:, 0] + 2.0 * dist_to_target + f_obstacle
        con = (dist_to_target + f_obstacle)[:, None]

        out["F"] = penalized
        out["G"] = con

    def _calc_pareto_front(self, n_pareto_points=100):
        return np.repeat(self.upper_bounds_for_objectives[None, :], max(1, int(n_pareto_points)), axis=0)


class MaOPP_real(_BaseMaOPP):
    def __init__(self, **kwargs):
        super().__init__(binary=False, **kwargs)


class MaOPP_binary(_BaseMaOPP):
    def __init__(self, **kwargs):
        super().__init__(binary=True, **kwargs)


class MaOPP_real_JAX(MaOPP_real):
    pass


class MaOPP_binary_JAX(MaOPP_binary):
    pass


__all__ = ["MaOPP_real", "MaOPP_binary", "MaOPP_real_JAX", "MaOPP_binary_JAX"]
