"""Level-set geometry for unfitted cut-cell computations.

Convention throughout the package: ``phi > 0`` inside the physical domain
Omega, ``phi < 0`` outside, ``phi = 0`` on the boundary Gamma = dOmega.

Each level set provides analytic gradients, Hessians, and curvature so that no
geometric quantity entering the stabilization formula or the Nitsche terms is
ever obtained by finite differencing.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def implicit_curvature(grad: np.ndarray, hess: np.ndarray) -> np.ndarray:
    """Signed curvature of the implicit curve phi = 0.

    For a planar level set, kappa = div(grad(phi)/|grad(phi)|) evaluated as

        (phi_xx phi_y^2 - 2 phi_xy phi_x phi_y + phi_yy phi_x^2)
            / (phi_x^2 + phi_y^2)^{3/2}.

    With the convention phi > 0 inside Omega this is the curvature of the
    boundary curve with respect to the *outward* normal; only its magnitude is
    consumed downstream. ``grad`` has shape (..., 2), ``hess`` (..., 2, 2).
    """
    gx, gy = grad[..., 0], grad[..., 1]
    hxx, hxy, hyy = hess[..., 0, 0], hess[..., 0, 1], hess[..., 1, 1]
    denom = (gx * gx + gy * gy) ** 1.5
    return (hxx * gy * gy - 2.0 * hxy * gx * gy + hyy * gx * gx) / denom


@dataclass(frozen=True)
class Circle:
    """Disk boundary: phi = r^2 - |x - c|^2 (positive inside)."""

    center: tuple[float, float]
    radius: float

    def phi(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        d2 = (x[..., 0] - self.center[0]) ** 2 + (x[..., 1] - self.center[1]) ** 2
        return self.radius**2 - d2

    def grad(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        g = np.empty_like(x)
        g[..., 0] = -2.0 * (x[..., 0] - self.center[0])
        g[..., 1] = -2.0 * (x[..., 1] - self.center[1])
        return g

    def hess(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        h = np.zeros(x.shape[:-1] + (2, 2))
        h[..., 0, 0] = -2.0
        h[..., 1, 1] = -2.0
        return h

    def curvature(self, x: np.ndarray) -> np.ndarray:
        # Constant along the whole circle: signed value -1/r for this
        # orientation; magnitude 1/r.
        x = np.asarray(x, dtype=float)
        return np.full(x.shape[:-1], -1.0 / self.radius)

    def outward_normal(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        return (x - np.asarray(self.center)) / self.radius


@dataclass(frozen=True)
class Ellipse:
    """Elliptical boundary: phi = 1 - ((x-cx)/a)^2 - ((y-cy)/b)^2."""

    center: tuple[float, float]
    semi_axes: tuple[float, float]

    def phi(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        u = (x[..., 0] - self.center[0]) / self.semi_axes[0]
        v = (x[..., 1] - self.center[1]) / self.semi_axes[1]
        return 1.0 - u * u - v * v

    def grad(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        g = np.empty_like(x)
        g[..., 0] = -2.0 * (x[..., 0] - self.center[0]) / self.semi_axes[0] ** 2
        g[..., 1] = -2.0 * (x[..., 1] - self.center[1]) / self.semi_axes[1] ** 2
        return g

    def hess(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        h = np.zeros(x.shape[:-1] + (2, 2))
        h[..., 0, 0] = -2.0 / self.semi_axes[0] ** 2
        h[..., 1, 1] = -2.0 / self.semi_axes[1] ** 2
        return h

    def curvature(self, x: np.ndarray) -> np.ndarray:
        return implicit_curvature(self.grad(x), self.hess(x))


@dataclass(frozen=True)
class Superellipse:
    """Superellipse boundary of even exponent n:

    phi = 1 - |(x-cx)/a|^n - |(y-cy)/b|^n.

    The set {phi >= 0} is the L_n ball scaled by (a, b); convex for n >= 1,
    with corners that sharpen as n grows.
    """

    center: tuple[float, float]
    semi_axes: tuple[float, float]
    exponent: int = 6

    def phi(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        u = np.abs((x[..., 0] - self.center[0]) / self.semi_axes[0])
        v = np.abs((x[..., 1] - self.center[1]) / self.semi_axes[1])
        n = float(self.exponent)
        return 1.0 - u**n - v**n

    def grad(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        n = float(self.exponent)
        dx = x[..., 0] - self.center[0]
        dy = x[..., 1] - self.center[1]
        ax = abs(dx)
        ay = abs(dy)
        sgn_x = np.sign(dx)
        sgn_y = np.sign(dy)
        g = np.empty_like(x)
        g[..., 0] = -n * sgn_x * (ax / self.semi_axes[0]) ** (n - 1.0) / self.semi_axes[0]
        g[..., 1] = -n * sgn_y * (ay / self.semi_axes[1]) ** (n - 1.0) / self.semi_axes[1]
        return g

    def hess(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        n = float(self.exponent)
        dx = x[..., 0] - self.center[0]
        dy = x[..., 1] - self.center[1]
        ax = np.maximum(abs(dx), 1e-300)
        ay = np.maximum(abs(dy), 1e-300)
        h = np.zeros(x.shape[:-1] + (2, 2))
        coef_x = n * (n - 1.0) * (ax / self.semi_axes[0]) ** (n - 2.0) / self.semi_axes[0] ** 2
        coef_y = n * (n - 1.0) * (ay / self.semi_axes[1]) ** (n - 2.0) / self.semi_axes[1] ** 2
        h[..., 0, 0] = -coef_x
        h[..., 1, 1] = -coef_y
        return h

    def curvature(self, x: np.ndarray) -> np.ndarray:
        return implicit_curvature(self.grad(x), self.hess(x))
