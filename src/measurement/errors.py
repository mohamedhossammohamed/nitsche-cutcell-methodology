"""Error norms against manufactured solutions on exact cut geometry."""

from __future__ import annotations

import numpy as np

from measurement.assembly import PkBasis
from geometry.cutting import cut_cell


def l2_error(mesh, levelset, u_h: np.ndarray, dofs: np.ndarray, k: int,
             u_exact, vol_order: int | None = None) -> float:
    """||u - u_h||_{L2(Omega)} via exact-cut quadrature."""
    if vol_order is None:
        vol_order = max(10, 2 * k + 4)
    total = 0.0
    for e in range(mesh.n_elements):
        region = cut_cell(levelset, mesh.nodes[mesh.elements[e]],
                          vol_order=vol_order)
        if region.status == "empty":
            continue
        basis = PkBasis(k, mesh.nodes[mesh.elements[e]])
        loc = dofs[e]
        vals = basis.values(region.pts)
        diff = u_exact(region.pts) - vals @ u_h[loc]
        total += float(np.sum(region.wts * diff * diff))
    return float(np.sqrt(total))


def h1_semi_error(mesh, levelset, u_h: np.ndarray, dofs: np.ndarray, k: int,
                  grad_u_exact, vol_order: int | None = None) -> float:
    """|u - u_h|_{H^1(Omega)} (seminorm) via exact-cut quadrature."""
    if vol_order is None:
        vol_order = max(10, 2 * k + 4)
    total = 0.0
    for e in range(mesh.n_elements):
        region = cut_cell(levelset, mesh.nodes[mesh.elements[e]],
                          vol_order=vol_order)
        if region.status == "empty":
            continue
        basis = PkBasis(k, mesh.nodes[mesh.elements[e]])
        loc = dofs[e]
        grads = basis.grads(region.pts)
        guh = np.einsum("i,qid->qd", u_h[loc], grads)
        diff = grad_u_exact(region.pts) - guh
        total += float(np.sum(region.wts * np.sum(diff * diff, axis=1)))
    return float(np.sqrt(total))


def energy_error(mesh, levelset, u_h: np.ndarray, dofs: np.ndarray, k: int,
                 grad_u_exact, dn_u_exact, lam_by_element: dict[int, float],
                 vol_order: int | None = None, bnd_order: int = 14) -> float:
    """sqrt(a_h(e, e)) using the SAME per-element lambdas as assembly.

    With e = u - u_h and Gamma_T = T ∩ dOmega,

        a_h(e, e) = sum_T int_{T∩Omega} |grad e|^2
                    - 2 int_Gamma_T e (dn e) + lambda_T int_Gamma_T e^2.

    ``grad_u_exact`` maps points -> (q, 2); ``dn_u_exact`` maps
    (points, outward normals) -> (q,). a_h(v,v) can be negative when some
    lambda_T is too small for coercivity; this function returns the sqrt of
    the value clamped at zero, and coercivity statements must consult
    gamma_h rather than infer from this norm.
    """
    if vol_order is None:
        vol_order = max(10, 2 * k + 4)
    total = 0.0
    for e in range(mesh.n_elements):
        region = cut_cell(levelset, mesh.nodes[mesh.elements[e]],
                          vol_order=vol_order, bnd_order=bnd_order)
        if region.status == "empty":
            continue
        basis = PkBasis(k, mesh.nodes[mesh.elements[e]])
        loc = dofs[e]
        vals = basis.values(region.pts)
        grads = basis.grads(region.pts)
        ge = grad_u_exact(region.pts) - np.einsum("i,qid->qd", u_h[loc], grads)
        total += float(np.sum(region.wts * np.sum(ge * ge, axis=1)))

        if region.status != "cut" or e not in lam_by_element:
            continue
        lam = lam_by_element[e]
        bp, bw, bn = region.bnd_pts, region.bnd_wts, region.bnd_nrm
        vb = basis.values(bp)
        gb = basis.grads(bp)
        dn_uh = np.einsum("i,qid->q", u_h[loc],
                          np.einsum("qid,qd->qi", gb, bn))
        e_b = u_exact(bp) - vb @ u_h[loc]
        dn_e = dn_u_exact(bp, bn) - dn_uh
        total += float(np.sum(
            bw * (-2.0 * e_b * dn_e + lam * e_b * e_b)))
    return float(np.sqrt(max(total, 0.0)))
