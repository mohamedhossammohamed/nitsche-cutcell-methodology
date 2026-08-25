"""Verification of cut-region machinery against independent references."""

import numpy as np
import pytest

from geometry.cutting import (
    clip_circle,
    clip_ellipse,
    clip_subdivision,
    cut_cell,
    polygon_disk_reference,
    ray_fire_reference,
)
from geometry.levelsets import Circle, Ellipse, Superellipse


CIRCLE = Circle(center=(0.42, 0.51), radius=0.37)


def _region_moments(region):
    w = region.wts[:, None]
    x, y = region.pts[:, 0], region.pts[:, 1]
    return {
        "area": float(region.wts.sum()),
        "mx": float(np.sum(region.wts * x)),
        "my": float(np.sum(region.wts * y)),
        "x2": float(np.sum(region.wts * x**2)),
        "xy": float(np.sum(region.wts * x * y)),
        "y2": float(np.sum(region.wts * y**2)),
    }


def test_full_and_empty_classification():
    tri_in = np.array([[0.40, 0.48], [0.50, 0.47], [0.44, 0.56]])  # deep inside
    reg = clip_circle(tri_in, CIRCLE)
    assert reg.status == "full"
    assert abs(reg.area - 0.5 * abs(
        (tri_in[1][0] - tri_in[0][0]) * (tri_in[2][1] - tri_in[0][1])
        - (tri_in[2][0] - tri_in[0][0]) * (tri_in[1][1] - tri_in[0][1]))) < 1e-15
    assert reg.gamma_length == 0.0

    tri_out = np.array([[0.9, 0.9], [0.99, 0.9], [0.95, 0.99]])
    reg = clip_circle(tri_out, CIRCLE)
    assert reg.status == "empty"
    assert reg.area == 0.0


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_clip_circle_against_polygon_reference(seed):
    """Random benign and multi-crossing cuts vs the polygon-clipping oracle."""
    rng = np.random.default_rng(seed)
    circ = Circle(center=tuple(rng.uniform(0.3, 0.7, size=2)),
                  radius=float(rng.uniform(0.2, 0.45)))
    tri = np.sort(rng.uniform(0.0, 1.0, size=(3, 2)), axis=0) + 1e-3 * rng.normal(
        size=(3, 2))
    # ensure CCW
    if ((tri[1][0] - tri[0][0]) * (tri[2][1] - tri[0][1])
            - (tri[2][0] - tri[0][0]) * (tri[1][1] - tri[0][1])) < 0:
        tri = tri[::-1]
    reg = clip_circle(tri, circ)
    if reg.status != "cut":
        pytest.skip("configuration not a proper cut")
    ref = polygon_disk_reference(tri, circ.center, circ.radius)
    got = _region_moments(reg)
    scale = max(ref["area"], 1e-300)
    assert abs(got["area"] - ref["area"]) <= 1e-11 * scale
    for key in ("mx", "my"):
        assert abs(got[key] - ref[key]) <= 1e-9 * max(scale, 1e-12)
    for key in ("x2", "xy", "y2"):
        assert abs(got[key] - ref[key]) <= 1e-8 * max(scale, 1e-12)
    # weights must reproduce the closed-form area bit-for-bit after scaling
    assert abs(float(reg.wts.sum()) - reg.area) <= 1e-15 * reg.area


@pytest.mark.parametrize("seed", [10, 11, 12])
def test_ray_fire_loose_area_agreement(seed):
    """Ray-fire oracle: algebraically convergent near edge kinks, so only a
    loose area-level smoke check is asserted against it."""
    rng = np.random.default_rng(seed)
    circ = Circle(center=tuple(rng.uniform(0.3, 0.7, size=2)),
                  radius=float(rng.uniform(0.25, 0.4)))
    tri = rng.uniform(0.0, 1.0, size=(3, 2))
    if ((tri[1][0] - tri[0][0]) * (tri[2][1] - tri[0][1])
            - (tri[2][0] - tri[0][0]) * (tri[1][1] - tri[0][1])) < 0:
        tri = tri[::-1]
    reg = clip_circle(tri, circ)
    if reg.status != "cut":
        pytest.skip("configuration not a proper cut")
    ref = ray_fire_reference(tri, circ.center, circ.radius)
    assert abs(reg.area - ref["area"]) <= 1e-7 * max(ref["area"], 1e-300)


@pytest.mark.parametrize("depth", [0.9, 0.99, 0.999, 0.9999, 0.99999, 0.999999])
def test_sliver_against_closed_form_segment(depth):
    """Circular-segment sliver: exact area and boundary length are known.

    Triangle spanning [-2, 2] horizontally at height y_c = -depth, apex well
    below; the disk portion is then exactly the circular segment below the
    chord y = y_c, whose area is (theta - sin theta)/2 and whose Gamma length
    is R*theta with theta = 2*acos(|y_c|/R).
    """
    yc = -depth
    tri = np.array([[-2.0, yc], [2.0, yc], [0.0, yc - 1.0]])
    circ = Circle(center=(0.0, 0.0), radius=1.0)
    reg = clip_circle(tri, circ)
    theta = 2.0 * np.arccos(depth)
    area_exact = 0.5 * (theta - np.sin(theta))
    gamma_exact = theta
    # absolute floors reflect float64 noise for O(1) coordinate arithmetic,
    # which dominates once sliver areas shrink below ~1e-13
    assert abs(reg.area - area_exact) <= 1e-9 * area_exact + 1e-13
    assert abs(reg.gamma_length - gamma_exact) <= 1e-9 * gamma_exact + 1e-13
    # ratio inherits the relative rounding of two independently verified
    # quantities; the bound is relative because rho itself grows like 1/eps
    assert abs(reg.cut_ratio - gamma_exact / area_exact) <= 1e-7 * (gamma_exact / area_exact)


def test_gamma_length_weights_sum():
    tri = np.array([[0.0, 0.2], [0.8, 0.25], [0.4, 0.75]])
    circ = Circle(center=(0.4, 0.35), radius=0.28)
    reg = clip_circle(tri, circ)
    if reg.status != "cut" or len(reg.bnd_wts) == 0:
        pytest.skip("not an arc-bearing cut")
    assert abs(float(reg.bnd_wts.sum()) - reg.gamma_length) <= 1e-14 * reg.gamma_length
    normals_unit = np.linalg.norm(reg.bnd_nrm, axis=1)
    assert np.allclose(normals_unit, 1.0)


def test_ellipse_affine_matches_mapped_reference():
    ell = Ellipse(center=(0.5, 0.5), semi_axes=(0.42, 0.21))
    tri = np.array([[0.55, 0.40], [0.95, 0.55], [0.60, 0.80]])
    reg = clip_ellipse(tri, ell)
    if reg.status != "cut":
        pytest.skip("configuration not a proper cut")
    # Reference: polygon clipping of the mapped triangle against the unit disk,
    # pulled back through the affine map.
    mapped = (tri - np.asarray(ell.center)) / np.asarray(ell.semi_axes)
    ref_disk = polygon_disk_reference(mapped, (0.0, 0.0), 1.0)
    det = ell.semi_axes[0] * ell.semi_axes[1]
    assert abs(reg.area - det * ref_disk["area"]) <= 1e-11 * reg.area
    # First moment transforms as \int x dA = det*(c*A_y + D*m_y).
    cx, cy = ell.center
    ax, ay = ell.semi_axes
    mx_expected = det * (cx * ref_disk["area"] + ax * ref_disk["mx"])
    my_expected = det * (cy * ref_disk["area"] + ay * ref_disk["my"])
    got = _region_moments(reg)
    assert abs(got["mx"] - mx_expected) <= 1e-9 * max(reg.area, 1e-12)
    assert abs(got["my"] - my_expected) <= 1e-9 * max(reg.area, 1e-12)


def test_subdivision_path_agrees_with_exact_on_quadratic_shape():
    """Superellipse exponent 2 IS an ellipse: cross-validate both paths."""
    ell = Ellipse(center=(0.5, 0.5), semi_axes=(0.38, 0.30))
    se = Superellipse(center=(0.5, 0.5), semi_axes=(0.38, 0.30), exponent=2)
    tri = np.array([[0.45, 0.40], [0.90, 0.52], [0.52, 0.86]])
    exact = clip_ellipse(tri, ell)
    approx = clip_subdivision(tri, se)
    assert exact.status == "cut"
    assert approx.status == "cut"
    rel = abs(approx.area - exact.area) / exact.area
    assert rel < 1e-7, f"subdivision area off by {rel:.2e}"
    assert approx.meta.get("self_check_rel", 1.0) < 1e-7
    gamma_rel = abs(approx.gamma_length - exact.gamma_length) / exact.gamma_length
    assert gamma_rel < 1e-5, f"gamma length off by {gamma_rel:.2e}"


def test_dispatcher_dispatches():
    tri = np.array([[0.3, 0.3], [0.7, 0.3], [0.5, 0.6]])
    assert cut_cell(Circle(center=(0.5, 0.45), radius=0.2), tri).method == "circle-exact"
    assert cut_cell(Ellipse(center=(0.5, 0.45), semi_axes=(0.2, 0.15)),
                    tri).method == "ellipse-affine"
    reg = cut_cell(Superellipse(center=(0.5, 0.45), semi_axes=(0.2, 0.15), exponent=6),
                   tri)
    assert reg.method.startswith("subdivision")
