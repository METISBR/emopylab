# emopylab 2026
"""Bounded scalar minimisation (Brent's method: golden section with parabolic interpolation)."""

from __future__ import annotations

import math

__all__ = ["fminbnd"]


def fminbnd(f, a: float, b: float, tol: float = 1e-4, max_iter: int = 500) -> float:
    """Minimiser of ``f`` inside ``[a, b]`` (Forsythe-Malcolm-Moler ``fmin``)."""
    c = 0.5 * (3.0 - math.sqrt(5.0))
    v = w = x = a + c * (b - a)
    fx = fv = fw = f(x)
    d = e = 0.0
    eps = math.sqrt(2.220446049250313e-16)
    for _ in range(max_iter):
        xm = 0.5 * (a + b)
        tol1 = eps * abs(x) + tol / 3.0
        tol2 = 2.0 * tol1
        if abs(x - xm) <= tol2 - 0.5 * (b - a):
            break
        golden = True
        if abs(e) > tol1:
            r = (x - w) * (fx - fv)
            q = (x - v) * (fx - fw)
            p = (x - v) * q - (x - w) * r
            q = 2.0 * (q - r)
            if q > 0:
                p = -p
            q = abs(q)
            r, e = e, d
            if abs(p) < abs(0.5 * q * r) and p > q * (a - x) and p < q * (b - x):
                d = p / q
                u = x + d
                if (u - a) < tol2 or (b - u) < tol2:
                    d = tol1 if xm >= x else -tol1
                golden = False
        if golden:
            e = (b - x) if x < xm else (a - x)
            d = c * e
        u = x + (d if abs(d) >= tol1 else (tol1 if d > 0 else -tol1))
        fu = f(u)
        if fu <= fx:
            if u < x:
                b = x
            else:
                a = x
            v, w, x = w, x, u
            fv, fw, fx = fw, fx, fu
        else:
            if u < x:
                a = u
            else:
                b = u
            if fu <= fw or w == x:
                v, w = w, u
                fv, fw = fw, fu
            elif fu <= fv or v == x or v == w:
                v, fv = u, fu
    return x
