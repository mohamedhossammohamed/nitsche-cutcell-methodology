"""Solver trio (preregistered): sparse LU, CG+Jacobi, CG+AMG."""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


@dataclass
class SolveResult:
    x: np.ndarray | None
    iterations: int  # 0 for direct solves
    converged: bool
    wall_time: float


def solve_direct(A: sp.csr_matrix, b: np.ndarray, rtol: float = 1e-10) -> SolveResult:
    t0 = time.perf_counter()
    lu = spla.splu(A.tocsc())
    x = lu.solve(b)
    resid = float(np.linalg.norm(A @ x - b) / max(np.linalg.norm(b), 1e-300))
    return SolveResult(x=x, iterations=0, converged=resid <= rtol * 100,
                       wall_time=time.perf_counter() - t0)


def solve_cg_jacobi(A: sp.csr_matrix, b: np.ndarray, rtol: float = 1e-10,
                    maxiter: int = 5000) -> SolveResult:
    t0 = time.perf_counter()
    diag = np.asarray(A.diagonal(), dtype=float)
    inv = np.where(np.abs(diag) > 0, 1.0 / np.where(diag == 0, 1.0, diag), 1.0)
    M = spla.LinearOperator(A.shape, matvec=lambda v: inv * v)
    counter = {"it": 0}

    def cb(_x):
        counter["it"] += 1

    x, info = spla.cg(A, b, M=M, rtol=rtol, atol=0.0, maxiter=maxiter, callback=cb)
    return SolveResult(x=x, iterations=counter["it"], converged=(info == 0),
                       wall_time=time.perf_counter() - t0)


def solve_cg_amg(A: sp.csr_matrix, b: np.ndarray, rtol: float = 1e-10,
                 maxiter: int = 5000) -> SolveResult:
    import pyamg

    t0 = time.perf_counter()
    ml = pyamg.smoothed_aggregation_solver(A.tocsr())
    M = ml.aspreconditioner()
    counter = {"it": 0}

    def cb(_x):
        counter["it"] += 1

    x, info = spla.cg(A, b, M=M, rtol=rtol, atol=0.0, maxiter=maxiter, callback=cb)
    return SolveResult(x=x, iterations=counter["it"], converged=(info == 0),
                       wall_time=time.perf_counter() - t0)


SOLVERS = {
    "S1_lu": solve_direct,
    "S2_cg_jacobi": solve_cg_jacobi,
    "S3_cg_amg": solve_cg_amg,
}
