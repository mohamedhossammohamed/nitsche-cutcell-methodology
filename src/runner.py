"""Fixed experiment entrypoint (branch-encoded behaviour).

This module is the single command executed by every orx run; what it measures
is determined by the branch's code, never by runtime arguments. The baseline
branch runs the preregistered validation suite: manufactured-solution
convergence on well-cut configurations for F1, k in {1, 2}.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from benchmarks.mesh import unit_square_mesh
from formulas.registry import build_registry, C_k
from geometry.levelsets import Circle
from measurement.assembly import assemble_nitsche
from measurement.errors import h1_semi_error, l2_error
from measurement.solvers import SOLVERS
from measurement.spectra import spectral_measures


def manufactured_2d():
    def u(x):
        return np.sin(np.pi * x[:, 0]) * np.cos(np.pi * x[:, 1])

    def f(x):  # -Delta u = 2 pi^2 sin(pi x) cos(pi y)
        return 2 * np.pi**2 * np.sin(np.pi * x[:, 0]) * np.cos(np.pi * x[:, 1])

    def grad_u(x):
        return np.stack([
            np.pi * np.cos(np.pi * x[:, 0]) * np.cos(np.pi * x[:, 1]),
            -np.pi * np.sin(np.pi * x[:, 0]) * np.sin(np.pi * x[:, 1]),
        ], axis=1)

    return u, f, grad_u


def fit_slope(hs: np.ndarray, errs: np.ndarray) -> tuple[float, float]:
    mask = np.asarray(errs) > 0
    p, log_b = np.polyfit(np.log(np.asarray(hs)[mask]),
                          np.log(np.asarray(errs)[mask]), 1)
    resid = np.log(np.asarray(errs)[mask]) - (p * np.log(np.asarray(hs)[mask]) + log_b)
    ss_res = float(np.sum(resid**2))
    ss_tot = float(np.sum((np.log(np.asarray(errs)[mask])
                           - np.mean(np.log(np.asarray(errs)[mask])))**2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return float(p), r2


def validate_benign() -> pd.DataFrame:
    """Well-cut convergence: circle fully inside, no slivers, F1 only."""
    reg = build_registry()
    psi = reg["F1"]["psi"]
    u, f, grad_u = manufactured_2d()
    rows = []
    for k in (1, 2):
        hs, e_l2, e_h1 = [], [], []
        for n in (8, 16, 32, 64):
            mesh = unit_square_mesh(n)
            levelset = Circle(center=(0.5, 0.5), radius=0.35)
            t0 = time.perf_counter()
            result = assemble_nitsche(mesh, levelset, psi, k=k, f=f, g=u)
            t_asm = time.perf_counter() - t0

            sp_result = spectral_measures(result.A)
            sol = SOLVERS["S1_lu"](result.A, result.rhs)

            u_full = result.expand(sol.x)
            hs.append(1.0 / n)
            e_l2.append(l2_error(mesh, levelset, u_full, result.dofs, k, u))
            e_h1.append(h1_semi_error(mesh, levelset, u_full, result.dofs, k,
                                      grad_u))

            rows.append({
                "tier": "validation-benign", "formula": "F1", "k": k,
                "n": n, "eps": np.nan, "solver": "S1_lu",
                "lambda_min": sp_result["lambda_min"],
                "lambda_max": sp_result["lambda_max"],
                "kappa": sp_result["kappa"],
                "min_converged": sp_result["min_converged"],
                "asym_rel": sp_result["asym_rel"],
                "l2_err": e_l2[-1], "h1_err": e_h1[-1],
                "cg_iters": sol.iterations,
                "assemble_time": t_asm,
                "seed": 0,
            })
            print(f"METRIC n {n} k {k} l2 {e_l2[-1]:.6e} h1 {e_h1[-1]:.6e} "
                  f"kappa {sp_result['kappa']:.6e}", flush=True)
        for name, errs in (("l2", e_l2), ("h1", e_h1)):
            p, r2 = fit_slope(np.array(hs), np.array(errs))
            print(f"METRIC slope_{name}_k{k} {p:.4f} R2 {r2:.6f}", flush=True)
            rows.append({
                "tier": "validation-slope", "formula": "F1", "k": k,
                "n": np.nan, "eps": np.nan, "solver": "-",
                "quantity": name, "slope_p": p, "slope_R2": r2, "seed": 0,
            })
    return pd.DataFrame(rows)


def main() -> int:
    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)
    df = validate_benign()
    path = out_dir / "validation.parquet"
    df.to_parquet(path, index=False)
    versions = {
        "numpy": np.__version__,
        "pandas": pd.__version__,
    }
    print("METRIC package_versions "
          + ";".join(f"{a}={b}" for a, b in sorted(versions.items())), flush=True)
    print(f"METRIC results_file {path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
