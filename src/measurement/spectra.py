"""Spectral measures: gamma_h (smallest eigenvalue), lambda_max, condition
number — with explicit non-convergence flags and an svds cross-check."""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.sparse.linalg import ArpackNoConvergence


def symmetrized(A: sp.csr_matrix) -> tuple[sp.csr_matrix, float]:
    As = ((A + A.T) * 0.5).tocsr()
    asym = float(sp.linalg.norm((A - A.T).tocsr()) /
                 max(float(sp.linalg.norm(As)), 1e-300))
    return As, asym


def spectral_measures(
    A: sp.csr_matrix,
    tol: float = 1e-10,
    maxiter: int = 500,
    sigma_shift: float = -1e-8,
) -> dict:
    """Smallest/largest eigenvalues via shift-invert Lanczos.

    Non-convergence is reported explicitly (NaN + flag), never silently.
    ``sigma_shift`` sits just below the spectrum so SPD systems invert stably.
    """
    As, asym = symmetrized(A)
    out = {
        "asym_rel": asym,
        "lambda_min": np.nan,
        "lambda_max": np.nan,
        "kappa": np.nan,
        "min_converged": False,
        "max_converged": False,
    }

    try:
        vals_min = spla.eigsh(As, k=1, sigma=sigma_shift, which="LM",
                              tol=tol, maxiter=maxiter, return_eigenvectors=False)
        out["lambda_min"] = float(vals_min[0])
        out["min_converged"] = True
    except ArpackNoConvergence as exc:
        if len(exc.eigenvalues) > 0:
            out["lambda_min"] = float(exc.eigenvalues[0])
            out["min_converged"] = False

    try:
        vals_max = spla.eigsh(As, k=1, which="LA",
                              tol=tol, maxiter=maxiter, return_eigenvectors=False)
        out["lambda_max"] = float(vals_max[0])
        out["max_converged"] = True
    except ArpackNoConvergence as exc:
        if len(exc.eigenvalues) > 0:
            out["lambda_max"] = float(exc.eigenvalues[-1])
            out["max_converged"] = False

    if out["min_converged"] and out["max_converged"]:
        denom = abs(out["lambda_min"])
        out["kappa"] = abs(out["lambda_max"]) / denom if denom > 0 else np.inf
    return out


def mass_normalized_lambda_min(A: sp.csr_matrix, M: sp.csr_matrix,
                               tol: float = 1e-10) -> float:
    """Smallest eigenvalue of the mass-normalized pencil (A v = lambda M v).

    Interpretability companion to the raw gamma_h gate; not part of any gate.
    """
    lu = spla.splu(M.tocsc())
    Minv = spla.LinearOperator(A.shape, matvec=lu.solve)
    try:
        vals = spla.eigsh(A, k=1, M=M, Minv=Minv, which="SM",
                          tol=tol, return_eigenvectors=False)
        return float(vals[0])
    except ArpackNoConvergence:
        return np.nan


def svds_cross_check(A: sp.csr_matrix, tol: float = 1e-10) -> tuple[float, float]:
    """Largest/smallest singular values via svds for solver independence."""
    s_min = float(spla.svds(A, k=1, which="SM", tol=tol,
                            return_singular_vectors=False)[0])
    s_max = float(spla.svds(A, k=1, which="LM", tol=tol,
                            return_singular_vectors=False)[0])
    return s_min, s_max
