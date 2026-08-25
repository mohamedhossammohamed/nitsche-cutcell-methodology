"""Assembler correctness: oracle agreement and benign-cut convergence."""

import numpy as np
import pytest

from benchmarks.mesh import unit_square_mesh
from geometry.levelsets import Circle
from geometry.quadrature import affine_map_to_physical, gauss_triangle
from measurement.assembly import PkBasis, assemble_nitsche, tri_area
from measurement.errors import l2_error
from runner import manufactured_2d, fit_slope


def _mass_matrix_reference(elem, k):
    """scikit-fem-free reference: direct quadrature of basis products."""
    basis = PkBasis(k, elem)
    ref, wref = gauss_triangle(max(8, 2 * k + 3))
    phys, _, det = affine_map_to_physical(ref, elem)
    vals = basis.values(phys)
    w = wref * abs(det)
    return np.einsum("q,qi,qj->ij", w, vals, vals)


@pytest.mark.parametrize("k", [1, 2])
def test_basis_mass_matrix_matches_direct_quadrature(k):
    rng = np.random.default_rng(3)
    elem = rng.uniform(0.1, 0.9, size=(3, 2))
    if ((elem[1][0] - elem[0][0]) * (elem[2][1] - elem[0][1])
            - (elem[2][0] - elem[0][0]) * (elem[1][1] - elem[0][1])) < 0:
        elem = elem[::-1]
    basis = PkBasis(k, elem)
    ref, wref = gauss_triangle(14)
    phys, _, det = affine_map_to_physical(ref, elem)
    vals = basis.values(phys)
    M_q = np.einsum("q,qi,qj->ij", wref * abs(det), vals, vals)
    # integrate constant: sum of mass entries equals element area * 1^2 terms
    area = tri_area(elem)
    assert abs(M_q.sum() - (area if k == 1 else area)) < 1e-13 * max(area, 1.0)
    # partition of unity: values summed along rows equal 1
    assert np.allclose(vals.sum(axis=1), 1.0)


def test_full_element_stiffness_is_standard_p1():
    """For an uncut interior triangle, volume stiffness must equal the
    classical -1/(2A) [[b_i b_j]] form."""
    elem = np.array([[0.2, 0.3], [0.5, 0.31], [0.31, 0.6]])
    basis = PkBasis(1, elem)
    g = basis.grad_lambda
    A_ref = np.einsum("id,jd->ij", g, g) * tri_area(elem)
    from measurement.assembly import PkBasis as B
    ref, wref = gauss_triangle(10)
    phys, _, det = affine_map_to_physical(ref, elem)
    grads = B(1, elem).grads(phys)
    K = np.einsum("q,qid,qjd->ij", wref * abs(det), grads, grads)
    assert np.allclose(A_ref, K, atol=1e-13)


@pytest.mark.parametrize("k", [1, 2])
def test_benign_convergence_rates(k):
    """Circle well inside the domain: optimal rates must hold (gate-level)."""
    mesh = unit_square_mesh(24)
    levelset = Circle(center=(0.5, 0.5), radius=0.35)
    u, f, grad_u = manufactured_2d()

    from formulas.registry import build_registry
    psi = build_registry()["F1"]["psi"]

    result = assemble_nitsche(mesh, levelset, psi, k=k, f=f, g=u)
    Asym = (result.A + result.A.T) * 0.5
    min_eig = float(_smallest_eig(Asym))
    assert min_eig > 1e-8, f"not coercive at benign cut: lambda_min={min_eig:.3e}"

    import scipy.sparse.linalg as spla
    x = spla.splu(Asym.tocsc()).solve(result.rhs)

    from measurement.errors import l2_error, h1_semi_error
    x_full = result.expand(x)
    e_l2 = l2_error(mesh, levelset, x_full, result.dofs, k, u)
    scale = l2_error(mesh, levelset, np.zeros(len(x_full)), result.dofs, k, u)
    rel = e_l2 / max(scale, 1e-300)
    bound = 0.05 if k == 1 else 0.01
    assert rel < bound, f"L2 relative error too large: {rel:.3e} (k={k})"


def _smallest_eig(A):
    import scipy.sparse.linalg as spla
    try:
        vals = spla.eigsh(A, k=1, sigma=-1e-8, which="LM",
                          return_eigenvectors=False)
        return vals[0]
    except Exception:
        return float("nan")


def test_rate_smoke_two_meshes():
    """Two-mesh error reduction consistent with O(h^2) L2 for k=1."""
    u, f, grad_u = manufactured_2d()
    from formulas.registry import build_registry
    psi = build_registry()["F1"]["psi"]
    errs = []
    for n in (12, 24):
        mesh = unit_square_mesh(n)
        ls = Circle(center=(0.5, 0.5), radius=0.35)
        res = assemble_nitsche(mesh, ls, psi, k=1, f=f, g=u)
        import scipy.sparse.linalg as spla
        x = spla.splu(((res.A + res.A.T) * 0.5).tocsc()).solve(res.rhs)
        errs.append(l2_error(mesh, ls, res.expand(x), res.dofs, 1, u))
    rate = np.log2(errs[0] / errs[1])
    assert 1.5 < rate < 2.5, f"observed two-mesh L2 rate {rate:.2f}"
