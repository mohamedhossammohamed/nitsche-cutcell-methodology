"""Quadrature polynomial-exactness verification."""

import numpy as np

from geometry.quadrature import (
    affine_map_to_physical,
    composite_gauss_line,
    gauss_interval,
    gauss_triangle,
    gauss_tetrahedron,
)


def _integrate_triangle_poly(order, a, b):
    pts, wts = gauss_triangle(order)
    return float(np.sum(wts * pts[:, 0] ** a * pts[:, 1] ** b))


def test_gauss_triangle_exactness_clean():
    """Reference values from the closed form ∫ x^a y^b = a! b! / (a+b+2)!.

    The collapsed-coordinate rule is exact through total degree 2n-2 (the
    (1-s) Jacobian consumes one order).
    """
    from math import factorial

    for order in (3, 6, 10):
        max_deg = 2 * order - 2
        for total in range(0, max_deg + 1):
            for a in range(total + 1):
                b = total - a
                exact = factorial(a) * factorial(b) / factorial(total + 2)
                got = _integrate_triangle_poly(order, a, b)
                assert abs(got - exact) <= 1e-14 * max(exact, 1.0)


def test_gauss_interval_sums_to_one():
    _, wts = gauss_interval(12)
    assert abs(wts.sum() - 1.0) < 1e-15


def test_gauss_tetrahedron_volume():
    pts, wts = gauss_tetrahedron(8)
    assert abs(wts.sum() - 1.0 / 6.0) < 1e-14
    # first moments of the reference tetrahedron are all 1/24
    for comp in range(3):
        m = float(np.sum(wts * pts[:, comp]))
        assert abs(m - 1.0 / 24.0) < 1e-14


def test_composite_gauss_line_length():
    p0, p1 = np.array([0.3, -0.2]), np.array([0.9, 0.7])
    pts, wts = composite_gauss_line((p0, p1), panels=5, order=6)
    length = float(np.linalg.norm(p1 - p0))
    assert abs(wts.sum() - length) <= 1e-14 * length
    assert abs(np.linalg.norm(pts[-1] - p1)) < 0.25 * length / 5


def test_affine_map_roundtrip():
    verts = np.array([[0.1, 0.2], [0.7, 0.25], [0.3, 0.8]])
    ref = np.array([[0.2, 0.3], [0.5, 0.1]])
    phys, jac, det = affine_map_to_physical(ref, verts)
    edges = np.stack([verts[1] - verts[0], verts[2] - verts[0]], axis=1)
    # phys - v0 = M @ lam (columns of M are edge vectors); rows recover as
    # lam_row = (phys - v0)_row @ M^{-T}
    back = (phys - verts[0]) @ np.linalg.inv(edges).T
    assert np.allclose(back, ref, atol=1e-14)
    expected_det = (verts[1][0] - verts[0][0]) * (verts[2][1] - verts[0][1]) - (
        verts[2][0] - verts[0][0]) * (verts[1][1] - verts[0][1])
    assert abs(det - expected_det) < 1e-14
