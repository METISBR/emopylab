from __future__ import annotations

import numpy as np


def as_2d(x, dtype=float) -> np.ndarray:
    arr = np.asarray(x, dtype=dtype)
    if arr.ndim == 1:
        arr = arr[None, :]
    return arr


def distance_function(x: np.ndarray, problem: int) -> np.ndarray:
    x = as_2d(x, float)
    if x.shape[1] == 0:
        return np.zeros(x.shape[0], dtype=float)
    x = 10.0 * x  # [0,1] -> [0,10]

    p = int(problem)
    if p == 1:
        f = np.sum(x**2, axis=1)
    elif p == 2:
        f = np.max(np.abs(x), axis=1)
    elif p == 3:
        f = np.sum(x**2 - 10.0 * np.cos(2.0 * np.pi * x) + 10.0, axis=1)
    elif p == 4:
        idx = np.sqrt(np.arange(1, x.shape[1] + 1, dtype=float))[None, :]
        f = np.sum(x**2, axis=1) / 4000.0 - np.prod(np.cos(x / idx), axis=1) + 1.0
    elif p == 5:
        n = float(max(1, x.shape[1]))
        f = 20.0 - 20.0 * np.exp(-0.2 * np.sqrt(np.sum(x**2, axis=1) / n)) - np.exp(np.sum(np.cos(2.0 * np.pi * x), axis=1) / n) + np.e
    else:
        raise ValueError(f"Unsupported distance function: {problem}")
    f[np.abs(f) < 1e-8] = 0.0
    return f


def cec2006_information(problem: int):
    p = int(problem)
    aaa = None
    if p == 1:
        lu = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [1, 1, 1, 1, 1, 1, 1, 1, 1, 100, 100, 100, 1]], dtype=float)
        n, optimal = 13, -15.0
    elif p == 2:
        lu = np.vstack([np.zeros(20), 10.0 * np.ones(20)])
        n, optimal = 20, -0.803619
    elif p == 3:
        lu = np.vstack([np.zeros(10), np.ones(10)])
        n, optimal = 10, -1.0
    elif p == 5:
        lu = np.array([[0, 0, -0.55, -0.55], [1200, 1200, 0.55, 0.55]], dtype=float)
        n, optimal = 4, 5126.4981
    elif p == 6:
        lu = np.array([[13, 0], [100, 100]], dtype=float)
        n, optimal = 2, -6961.81388
    elif p == 9:
        lu = np.array([[-10, -10, -10, -10, -10, -10, -10], [10, 10, 10, 10, 10, 10, 10]], dtype=float)
        n, optimal = 7, 680.6300573
    elif p == 10:
        lu = np.array([[100, 1000, 1000, 10, 10, 10, 10, 10],
                       [10000, 10000, 10000, 1000, 1000, 1000, 1000, 1000]], dtype=float)
        n, optimal = 8, 7049.2480
    elif p == 11:
        lu = np.array([[-1, -1], [1, 1]], dtype=float)
        n, optimal = 2, 0.75
    elif p == 12:
        lu = np.array([[0, 0, 0], [10, 10, 10]], dtype=float)
        n = 3
        grid = np.arange(1, 10, dtype=float)
        aaa = np.array(np.meshgrid(grid, grid, grid, indexing="ij")).reshape(3, -1).T
        optimal = -1.0
    elif p == 14:
        lu = np.vstack([np.zeros(10), 10.0 * np.ones(10)])
        n, optimal = 10, -47.7648884595
    elif p == 15:
        lu = np.vstack([np.zeros(3), 10.0 * np.ones(3)])
        n, optimal = 3, 961.7150222899
    elif p == 18:
        lu = np.array([[-10, -10, -10, -10, -10, -10, -10, -10, 0],
                       [10, 10, 10, 10, 10, 10, 10, 10, 20]], dtype=float)
        n, optimal = 9, -0.8660254038
    elif p == 19:
        lu = np.vstack([np.zeros(15), 10.0 * np.ones(15)])
        n, optimal = 15, 32.6555929502
    elif p == 24:
        lu = np.array([[0, 0], [3, 4]], dtype=float)
        n, optimal = 2, -5.5080132716
    else:
        raise ValueError(f"Unsupported CEC2006 problem in local helper: {problem}")

    if aaa is None:
        aaa = np.empty((0, 0), dtype=float)
    return lu, int(n), aaa, float(optimal)


def cec2006_fitness(P: np.ndarray, problem: int, aaa: np.ndarray | None, optimal: float):
    P = as_2d(P, float)
    n = P.shape[0]
    # Pad one column to preserve MATLAB 1-based indexing in formulas.
    X = np.column_stack([np.zeros(n, dtype=float), P])

    p = int(problem)
    g_cols: list[np.ndarray] = []

    if p == 1:
        g_cols.extend(
            [
                2 * X[:, 1] + 2 * X[:, 2] + X[:, 10] + X[:, 11] - 10,
                2 * X[:, 1] + 2 * X[:, 3] + X[:, 10] + X[:, 12] - 10,
                2 * X[:, 2] + 2 * X[:, 3] + X[:, 11] + X[:, 12] - 10,
                -8 * X[:, 1] + X[:, 10],
                -8 * X[:, 2] + X[:, 11],
                -8 * X[:, 3] + X[:, 12],
                -2 * X[:, 4] - X[:, 5] + X[:, 10],
                -2 * X[:, 6] - X[:, 7] + X[:, 11],
                -2 * X[:, 8] - X[:, 9] + X[:, 12],
            ]
        )
        f = 5 * np.sum(X[:, 1:5], axis=1) - 5 * np.sum(X[:, 1:5] ** 2, axis=1) - np.sum(X[:, 5:14], axis=1)

    elif p == 2:
        g_cols.extend([0.75 - np.prod(P, axis=1), np.sum(P, axis=1) - 7.5 * P.shape[1]])
        idx = np.arange(1, P.shape[1] + 1, dtype=float)[None, :]
        f = -np.abs(np.sum(np.cos(P) ** 4, axis=1) - 2 * np.prod(np.cos(P) ** 2, axis=1)) / np.sqrt(
            1e-30 + np.sum(idx * (P**2), axis=1)
        )

    elif p == 3:
        g_cols.append(np.abs(np.sum(P**2, axis=1) - 1.0) - 0.0001)
        f = -(10.0**0.5) ** 10 * np.prod(P, axis=1)

    elif p == 5:
        g_cols.extend(
            [
                -X[:, 4] + X[:, 3] - 0.55,
                -X[:, 3] + X[:, 4] - 0.55,
                np.abs(1000 * np.sin(-X[:, 3] - 0.25) + 1000 * np.sin(-X[:, 4] - 0.25) + 894.8 - X[:, 1]) - 0.0001,
                np.abs(1000 * np.sin(X[:, 3] - 0.25) + 1000 * np.sin(X[:, 3] - X[:, 4] - 0.25) + 894.8 - X[:, 2]) - 0.0001,
                np.abs(1000 * np.sin(X[:, 4] - 0.25) + 1000 * np.sin(X[:, 4] - X[:, 3] - 0.25) + 1294.8) - 0.0001,
            ]
        )
        f = 3 * X[:, 1] + 1e-6 * X[:, 1] ** 3 + 2 * X[:, 2] + (2e-6 / 3.0) * X[:, 2] ** 3

    elif p == 6:
        g_cols.extend(
            [
                -(X[:, 1] - 5) ** 2 - (X[:, 2] - 5) ** 2 + 100,
                (X[:, 1] - 6) ** 2 + (X[:, 2] - 5) ** 2 - 82.81,
            ]
        )
        f = (X[:, 1] - 10) ** 3 + (X[:, 2] - 20) ** 3

    elif p == 9:
        g_cols.extend(
            [
                -127 + 2 * X[:, 1] ** 2 + 3 * X[:, 2] ** 4 + X[:, 3] + 4 * X[:, 4] ** 2 + 5 * X[:, 5],
                -282 + 7 * X[:, 1] + 3 * X[:, 2] + 10 * X[:, 3] ** 2 + X[:, 4] - X[:, 5],
                -196 + 23 * X[:, 1] + X[:, 2] ** 2 + 6 * X[:, 6] ** 2 - 8 * X[:, 7],
                4 * X[:, 1] ** 2 + X[:, 2] ** 2 - 3 * X[:, 1] * X[:, 2] + 2 * X[:, 3] ** 2 + 5 * X[:, 6] - 11 * X[:, 7],
            ]
        )
        f = (
            (X[:, 1] - 10) ** 2
            + 5 * (X[:, 2] - 12) ** 2
            + X[:, 3] ** 4
            + 3 * (X[:, 4] - 11) ** 2
            + 10 * X[:, 5] ** 6
            + 7 * X[:, 6] ** 2
            + X[:, 7] ** 4
            - 4 * X[:, 6] * X[:, 7]
            - 10 * X[:, 6]
            - 8 * X[:, 7]
        )

    elif p == 10:
        g_cols.extend(
            [
                -1 + 0.0025 * (X[:, 4] + X[:, 6]),
                -1 + 0.0025 * (X[:, 5] + X[:, 7] - X[:, 4]),
                -1 + 0.01 * (X[:, 8] - X[:, 5]),
                -X[:, 1] * X[:, 6] + 833.33252 * X[:, 4] + 100 * X[:, 1] - 83333.333,
                -X[:, 2] * X[:, 7] + 1250 * X[:, 5] + X[:, 2] * X[:, 4] - 1250 * X[:, 4],
                -X[:, 3] * X[:, 8] + 1250000 + X[:, 3] * X[:, 5] - 2500 * X[:, 5],
            ]
        )
        f = X[:, 1] + X[:, 2] + X[:, 3]

    elif p == 11:
        g_cols.append(np.abs(X[:, 2] - X[:, 1] ** 2) - 0.0001)
        f = X[:, 1] ** 2 + (X[:, 2] - 1) ** 2

    elif p == 12:
        f = -(100 - (X[:, 1] - 5) ** 2 - (X[:, 2] - 5) ** 2 - (X[:, 3] - 5) ** 2) / 100
        if aaa is None or np.size(aaa) == 0:
            grid = np.arange(1, 10, dtype=float)
            aaa = np.array(np.meshgrid(grid, grid, grid, indexing="ij")).reshape(3, -1).T
        g0 = np.zeros(n, dtype=float)
        for j in range(n):
            g0[j] = np.min(np.sum((aaa - P[j, :][None, :]) ** 2, axis=1)) - 0.0625
        g_cols.append(g0)

    elif p == 14:
        c = np.array([-6.089, -17.164, -34.054, -5.914, -24.721, -14.986, -24.1, -10.708, -26.662, -22.179], dtype=float)
        g_cols.extend(
            [
                np.abs(X[:, 1] + 2 * X[:, 2] + 2 * X[:, 3] + X[:, 6] + X[:, 10] - 2) - 0.0001,
                np.abs(X[:, 4] + 2 * X[:, 5] + X[:, 6] + X[:, 7] - 1) - 0.0001,
                np.abs(X[:, 3] + X[:, 7] + X[:, 8] + 2 * X[:, 9] + X[:, 10] - 1) - 0.0001,
            ]
        )
        denom = 1e-30 + np.sum(P, axis=1, keepdims=True)
        f = np.sum(P * (c[None, :] + np.log(1e-30 + P / denom)), axis=1)

    elif p == 15:
        g_cols.extend(
            [
                np.abs(X[:, 1] ** 2 + X[:, 2] ** 2 + X[:, 3] ** 2 - 25) - 0.0001,
                np.abs(8 * X[:, 1] + 14 * X[:, 2] + 7 * X[:, 3] - 56) - 0.0001,
            ]
        )
        f = 1000 - X[:, 1] ** 2 - 2 * X[:, 2] ** 2 - X[:, 3] ** 2 - X[:, 1] * X[:, 2] - X[:, 1] * X[:, 3]

    elif p == 18:
        g_cols.extend(
            [
                X[:, 3] ** 2 + X[:, 4] ** 2 - 1,
                X[:, 9] ** 2 - 1,
                X[:, 5] ** 2 + X[:, 6] ** 2 - 1,
                X[:, 1] ** 2 + (X[:, 2] - X[:, 9]) ** 2 - 1,
                (X[:, 1] - X[:, 5]) ** 2 + (X[:, 2] - X[:, 6]) ** 2 - 1,
                (X[:, 1] - X[:, 7]) ** 2 + (X[:, 2] - X[:, 8]) ** 2 - 1,
                (X[:, 3] - X[:, 5]) ** 2 + (X[:, 4] - X[:, 6]) ** 2 - 1,
                (X[:, 3] - X[:, 7]) ** 2 + (X[:, 4] - X[:, 8]) ** 2 - 1,
                X[:, 7] ** 2 + (X[:, 8] - X[:, 9]) ** 2 - 1,
                X[:, 2] * X[:, 3] - X[:, 1] * X[:, 4],
                -X[:, 3] * X[:, 9],
                X[:, 5] * X[:, 9],
                X[:, 6] * X[:, 7] - X[:, 5] * X[:, 8],
            ]
        )
        f = -0.5 * (
            X[:, 1] * X[:, 4]
            - X[:, 2] * X[:, 3]
            + X[:, 3] * X[:, 9]
            - X[:, 5] * X[:, 9]
            + X[:, 5] * X[:, 8]
            - X[:, 6] * X[:, 7]
        )

    elif p == 19:
        a = np.array(
            [
                [-16, 2, 0, 1, 0],
                [0, -2, 0, 0.4, 2],
                [-3.5, 0, 2, 0, 0],
                [0, -2, 0, -4, -1],
                [0, -9, -2, 1, -2.8],
                [2, 0, -4, 0, 0],
                [-1, -1, -1, -1, -1],
                [-1, -2, -3, -2, -1],
                [1, 2, 3, 4, 5],
                [1, 1, 1, 1, 1],
            ],
            dtype=float,
        )
        b = np.array([-40, -2, -0.25, -4, -4, -1, -40, -60, 5, 1], dtype=float)
        c = np.array(
            [
                [30, -20, -10, 32, -10],
                [-20, 39, -6, -31, 32],
                [-10, -6, 10, -6, -10],
                [32, -31, -6, 39, -20],
                [-10, 32, -10, -20, 30],
            ],
            dtype=float,
        )
        d = np.array([4, 8, 10, 6, 2], dtype=float)
        e = np.array([-15, -27, -36, -18, -12], dtype=float)
        x1 = X[:, 1:11]
        x2 = X[:, 11:16]
        for j in range(5):
            g_cols.append(-2 * np.sum(c[:, j][None, :] * x2, axis=1) - 3 * d[j] * X[:, 11 + j] ** 2 - e[j] + np.sum(a[:, j][None, :] * x1, axis=1))
        f = (
            np.sum(c[:, 0][None, :] * x2, axis=1) * X[:, 11]
            + np.sum(c[:, 1][None, :] * x2, axis=1) * X[:, 12]
            + np.sum(c[:, 2][None, :] * x2, axis=1) * X[:, 13]
            + np.sum(c[:, 3][None, :] * x2, axis=1) * X[:, 14]
            + np.sum(c[:, 4][None, :] * x2, axis=1) * X[:, 15]
            + 2 * np.sum(d[None, :] * (x2**3), axis=1)
            - np.sum(b[None, :] * x1, axis=1)
        )

    elif p == 24:
        g_cols.extend(
            [
                -2 * X[:, 1] ** 4 + 8 * X[:, 1] ** 3 - 8 * X[:, 1] ** 2 + X[:, 2] - 2,
                -4 * X[:, 1] ** 4 + 32 * X[:, 1] ** 3 - 88 * X[:, 1] ** 2 + 96 * X[:, 1] + X[:, 2] - 36,
            ]
        )
        f = -X[:, 1] - X[:, 2]

    else:
        raise ValueError(f"Unsupported CEC2006 problem in local helper: {problem}")

    g = np.column_stack(g_cols) if g_cols else np.zeros((n, 0), dtype=float)
    objF = np.asarray(f, dtype=float) - float(optimal)
    conV = np.sum(np.maximum(0.0, g), axis=1)
    conV[conV < 1e-6] = 0.0
    objF[np.abs(objF) <= 1e-3] = 0.0
    return objF, conV


def nd_mask(F: np.ndarray) -> np.ndarray:
    F = as_2d(F, float)
    n = F.shape[0]
    keep = np.ones(n, dtype=bool)
    for i in range(n):
        if not keep[i]:
            continue
        dominates_i = np.all(F <= F[i], axis=1) & np.any(F < F[i], axis=1)
        dominates_by_i = np.all(F[i] <= F, axis=1) & np.any(F[i] < F, axis=1)
        if np.any(dominates_i):
            keep[i] = False
        else:
            keep[dominates_by_i] = False
            keep[i] = True
    return keep


def simplex_points_2d(n: int) -> np.ndarray:
    n = max(2, int(n))
    x = np.linspace(0.0, 1.0, n)
    return np.column_stack([x, 1.0 - x])


def quarter_sphere_points_2d(n: int) -> np.ndarray:
    r = simplex_points_2d(n)
    d = np.linalg.norm(r, axis=1, keepdims=True)
    d[d == 0] = 1.0
    return r / d


def quarter_sphere_points_3d(n: int) -> np.ndarray:
    g = max(5, int(np.ceil(np.sqrt(max(4, n)))))
    t = np.linspace(0.0, np.pi / 2.0, g)
    a, b = np.meshgrid(t, t)
    x = np.sin(a) * np.cos(b)
    y = np.sin(a) * np.sin(b)
    z = np.cos(a)
    return np.column_stack([x.ravel(), y.ravel(), z.ravel()])

