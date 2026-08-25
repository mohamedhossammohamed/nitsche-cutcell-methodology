"""Level-set geometry analytic checks."""

import numpy as np

from geometry.levelsets import Circle, Ellipse, Superellipse, implicit_curvature


def test_circle_curvature_and_normals():
    circ = Circle(center=(0.4, 0.6), radius=0.25)
    th = np.linspace(0.0, 2.0 * np.pi, 17)[:-1]
    pts = np.asarray(circ.center) + 0.25 * np.stack([np.cos(th), np.sin(th)], axis=1)
    assert np.allclose(circ.phi(pts), 0.0, atol=1e-13)
    assert np.allclose(circ.curvature(pts), -1.0 / 0.25)
    n_out = circ.outward_normal(pts)
    assert np.allclose(np.linalg.norm(n_out, axis=1), 1.0)
    # phi must decrease when stepping outward along the normal
    stepped = pts + 1e-6 * n_out
    assert np.all(circ.phi(stepped) < 0.0)


def test_ellipse_curvature_at_axis_endpoints():
    ax, ay = 0.5, 0.2
    ell = Ellipse(center=(0.0, 0.0), semi_axes=(ax, ay))
    # curvature at (a, 0): kappa = a / b^2; at (0, b): kappa = b / a^2
    k_right = float(ell.curvature(np.array([[ax, 0.0]]))[0])
    k_top = float(ell.curvature(np.array([[0.0, ay]]))[0])
    assert abs(abs(k_right) - ax / ay**2) < 1e-12
    assert abs(abs(k_top) - ay / ax**2) < 1e-12


def test_superellipse_reduces_to_ellipse_for_n2():
    se = Superellipse(center=(0.3, 0.4), semi_axes=(0.5, 0.35), exponent=2)
    el = Ellipse(center=(0.3, 0.4), semi_axes=(0.5, 0.35))
    rng = np.random.default_rng(7)
    pts = rng.uniform(0.0, 0.8, size=(50, 2))
    assert np.allclose(se.phi(pts), el.phi(pts), atol=1e-13)


def test_implicit_curvature_matches_analytic():
    ell = Ellipse(center=(0.2, 0.3), semi_axes=(0.45, 0.3))
    th = np.linspace(0.0, 2.0 * np.pi, 13)[:-1]
    pts = np.asarray(ell.center)[None, :] + np.stack(
        [0.45 * np.cos(th), 0.3 * np.sin(th)], axis=1)
    k_impl = implicit_curvature(ell.grad(pts), ell.hess(pts))
    a, b = 0.45, 0.3
    k_param = a * b / (a**2 * np.sin(th) ** 2 + b**2 * np.cos(th) ** 2) ** 1.5
    # sign conventions differ between the two derivations; magnitudes agree
    assert np.allclose(np.abs(k_impl), k_param, rtol=1e-12)
