"""Gauss quadrature rules on intervals and simplices.

Rules are generated from tensor products of Gauss-Legendre rules via collapsed
(Duffy-type) maps rather than embedded tables: they are exact for polynomials
of total degree 2n-1 on the simplex at order n and converge exponentially for
analytic integrands, which covers every integrand in this study (polynomial
basis functions, manufactured solutions built from trigonometric functions).
"""

from __future__ import annotations

import numpy as np
from numpy.polynomial.legendre import leggauss


def gauss_interval(n: int) -> tuple[np.ndarray, np.ndarray]:
    """Order-n Gauss-Legendre nodes and weights mapped to [0, 1]."""
    nodes, weights = leggauss(n)
    return 0.5 * (nodes + 1.0), 0.5 * weights


def gauss_triangle(n: int) -> tuple[np.ndarray, np.ndarray]:
    """Order-n rule on the reference triangle {(s, t): s >= 0, t >= 0, s+t <= 1}.

    Exact through polynomial TOTAL degree 2n-2. Collapsed coordinates map the
    unit square by (s, t) -> (s, (1-s) t); the Jacobian (1-s) raises the
    outer integrand's degree by one, which costs a single order relative to a
    plain product rule.
    """
    s, ws = gauss_interval(n)
    t, wt = gauss_interval(n)
    S, T = np.meshgrid(s, t, indexing="ij")
    WS, WT = np.meshgrid(ws, wt, indexing="ij")
    pts = np.stack([S.ravel(), ((1.0 - S) * T).ravel()], axis=1)
    wts = (WS * WT).ravel() * (1.0 - S).ravel()
    return pts, wts


def gauss_tetrahedron(n: int) -> tuple[np.ndarray, np.ndarray]:
    """Order-n rule on the reference tetrahedron via two collapsed maps.

    Exact through total degree 2n-1. The cube->tetrahedron map
        p = s, q = t(1-s), r = u(1-t)(1-s)
    has Jacobian (1-s)^2 (1-t).
    """
    s, ws = gauss_interval(n)
    t, wt = gauss_interval(n)
    u, wu = gauss_interval(n)
    S, T, U = np.meshgrid(s, t, u, indexing="ij")
    WS, WT, WU = np.meshgrid(ws, wt, wu, indexing="ij")
    a = (1.0 - S).ravel()
    b = (1.0 - T).ravel()
    pts = np.stack(
        [S.ravel(), (T * (1.0 - S)).ravel(), (U * (1.0 - T) * (1.0 - S)).ravel()],
        axis=1,
    )
    wts = (WS * WT * WU).ravel() * a * a * b
    return pts, wts


def affine_map_to_physical(ref_pts: np.ndarray, verts: np.ndarray):
    """Map reference-triangle points to physical coordinates.

    ``verts`` is (3, dim) counterclockwise vertex ordering; reference points
    are barycentric pairs (lambda1, lambda2) so the physical point is
    v0 + lambda1 (v1 - v0) + lambda2 (v2 - v0).
    """
    v0, v1, v2 = verts[0], verts[1], verts[2]
    e1 = v1 - v0
    e2 = v2 - v0
    lam1 = ref_pts[:, 0][:, None]
    lam2 = ref_pts[:, 1][:, None]
    phys = v0[None, :] + lam1 * e1[None, :] + lam2 * e2[None, :]
    jac = np.stack([e1, e2], axis=1)  # (dim, 2)
    det = jac[0, 0] * jac[1, 1] - jac[0, 1] * jac[1, 0]
    return phys, jac, det


def composite_gauss_line(endpoints_pair: tuple[np.ndarray, np.ndarray], panels: int, order: int):
    """Composite Gauss-Legendre rule on the segment between two endpoints.

    Returns points (m, dim), ds-weights (m,) summing exactly to the segment
    length up to floating-point round-off.
    """
    p0 = np.asarray(endpoints_pair[0], dtype=float)
    p1 = np.asarray(endpoints_pair[1], dtype=float)
    length = float(np.linalg.norm(p1 - p0))
    if length == 0.0:
        z_dim = p0.size
        return np.zeros((0, z_dim)), np.zeros((0,))
    nodes, weights = leggauss(order)
    edges = np.linspace(0.0, length, panels + 1)
    pts_list = []
    wts_list = []
    unit_dir = (p1 - p0) / length
    for a, b in zip(edges[:-1], edges[1:]):
        half, mid = 0.5 * (b - a), 0.5 * (b + a)
        local_s = mid + half * nodes
        global_w = half * weights
        pts_list.append(p0[None, :] + local_s[:, None] * unit_dir[None, :])
        wts_list.append(global_w)
    pts = np.concatenate(pts_list, axis=0)
    wts = np.concatenate(wts_list, axis=0)
    return pts, wts
