"""Fixed experiment entrypoint (branch-encoded behaviour).

T1 controlled-sliver sweep: all closed-form variants plus aggregation hybrid
over PREREG epsilon/n/k grid. Also retains benign validation as sanity check.

Logs per Section 5 quantities: lambda_min, kappa, L2/H1/energy errors,
solver iters, wall-clock, with explicit non-convergence flags.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

from benchmarks.mesh import unit_square_mesh
from formulas.registry import build_registry, C_k
from geometry.levelsets import Circle
from geometry.cutting import cut_cell
from measurement.assembly import assemble_nitsche, PkBasis, default_aggregator
from measurement.errors import h1_semi_error, l2_error, energy_error
from measurement.solvers import SOLVERS
from measurement.spectra import spectral_measures, svds_cross_check


# ---------------------------------------------------------------------------
# manufactured data (2D)
# ---------------------------------------------------------------------------

def manufactured_2d():
    def u(x):
        return np.sin(np.pi * x[:, 0]) * np.cos(np.pi * x[:, 1])

    def f(x):
        return 2 * np.pi**2 * np.sin(np.pi * x[:, 0]) * np.cos(np.pi * x[:, 1])

    def grad_u(x):
        return np.stack([
            np.pi * np.cos(np.pi * x[:, 0]) * np.cos(np.pi * x[:, 1]),
            -np.pi * np.sin(np.pi * x[:, 0]) * np.sin(np.pi * x[:, 1]),
        ], axis=1)

    def dn_u(x, n):
        g = grad_u(x)
        return np.sum(g * n, axis=1)

    return u, f, grad_u, dn_u


def fit_slope(hs: np.ndarray, errs: np.ndarray) -> tuple[float, float]:
    errs = np.asarray(errs, dtype=float)
    hs = np.asarray(hs, dtype=float)
    mask = np.isfinite(errs) & (errs > 0) & np.isfinite(hs) & (hs > 0)
    if int(mask.sum()) < 2:
        return float("nan"), float("nan")
    p, log_b = np.polyfit(np.log(hs[mask]), np.log(errs[mask]), 1)
    resid = np.log(errs[mask]) - (p * np.log(hs[mask]) + log_b)
    ss_res = float(np.sum(resid**2))
    mean_log = float(np.mean(np.log(errs[mask])))
    ss_tot = float(np.sum((np.log(errs[mask]) - mean_log)**2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return float(p), float(r2)


# ---------------------------------------------------------------------------
# T1 geometry helper — cap height bisection (PREREG exact)
# ---------------------------------------------------------------------------

def cap_area(s: float, r: float) -> float:
    # A_cap = r^2 arccos((r-s)/r) - (r-s) sqrt(2 r s - s^2), s in [0,2r]
    if s <= 0:
        return 0.0
    if s >= 2*r:
        return np.pi * r * r
    d = r - s
    return r*r * np.arccos(d / r) - d * np.sqrt(max(2*r*s - s*s, 0.0))


def solve_cap_height(eps: float, h: float, r: float = 0.25) -> float:
    target = eps * h * h
    # Handle eps=0 edge (not in our grid)
    if target <= 0:
        return 0.0
    lo, hi = 0.0, 2.0 * r
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if cap_area(mid, r) < target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def t1_circle(n: int, eps: float, r: float = 0.25) -> Circle:
    h = 1.0 / n
    s = solve_cap_height(eps, h, r)
    cx = 3.5 * h
    cy = 4.0 * h + s - r
    return Circle(center=(cx, cy), radius=r)


# ---------------------------------------------------------------------------
# mass matrix for mass-normalized eigenvalue (optional)
# ---------------------------------------------------------------------------

def assemble_mass(mesh, levelset, k: int):
    from geometry.cutting import cut_cell
    from measurement.assembly import PkBasis
    n_dof = mesh.dof_count(k)
    dofs = mesh.element_dofs(k)
    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []
    for e in range(mesh.n_elements):
        region = cut_cell(levelset, mesh.nodes[mesh.elements[e]])
        if region.status == "empty":
            continue
        basis = PkBasis(k, mesh.nodes[mesh.elements[e]])
        loc = dofs[e]
        nd = len(loc)
        vals_q = basis.values(region.pts)
        w = region.wts
        Mloc = np.einsum("q,qi,qj->ij", w, vals_q, vals_q)
        for a in range(nd):
            for b in range(nd):
                rows.append(int(loc[a]))
                cols.append(int(loc[b]))
                vals.append(float(Mloc[a, b]))
    M = sp.coo_matrix((vals, (rows, cols)), shape=(n_dof, n_dof)).tocsr()
    # restrict to active space same as assembly does (drop empty rows) – we
    # need the same permutation; caller should restrict using active_ids
    return M


# ---------------------------------------------------------------------------
# validation (benign) — retained as smoke check
# ---------------------------------------------------------------------------

def validate_benign() -> pd.DataFrame:
    reg = build_registry()
    psi = reg["F1"]["psi"]
    u, f, grad_u, dn_u = manufactured_2d()
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
            l2 = l2_error(mesh, levelset, u_full, result.dofs, k, u)
            h1 = h1_semi_error(mesh, levelset, u_full, result.dofs, k, grad_u)
            e_l2.append(l2)
            e_h1.append(h1)
            rows.append({
                "tier": "validation-benign", "formula": "F1", "k": k,
                "n": n, "eps": np.nan, "solver": "S1_lu",
                "lambda_min": sp_result["lambda_min"],
                "lambda_max": sp_result["lambda_max"],
                "kappa": sp_result["kappa"],
                "min_converged": sp_result["min_converged"],
                "max_converged": sp_result["max_converged"],
                "asym_rel": sp_result["asym_rel"],
                "l2_err": l2, "h1_err": h1, "energy_err": np.nan,
                "cg_iters": sol.iterations,
                "assemble_time": t_asm,
                "seed": 0,
            })
            print(f"METRIC n {n} k {k} l2 {l2:.6e} h1 {h1:.6e} kappa {sp_result['kappa']:.6e}", flush=True)
        for name, errs in (("l2", e_l2), ("h1", e_h1)):
            p, r2 = fit_slope(np.array(hs), np.array(errs))
            print(f"METRIC slope_{name}_k{k} {p:.4f} R2 {r2:.6f}", flush=True)
            rows.append({
                "tier": "validation-slope", "formula": "F1", "k": k,
                "n": np.nan, "eps": np.nan, "solver": "-",
                "quantity": name, "slope_p": p, "slope_R2": r2, "seed": 0,
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# T1 sweep
# ---------------------------------------------------------------------------

PREREG_EPS = [0.5, 0.1, 0.01, 1e-3, 1e-4, 1e-5, 1e-6]
PREREG_N = [8, 16, 32, 64, 128]
PREREG_K = [1, 2]

# For runtime we include all 13 closed-form plus F7 hybrid; F8 deferred
T1_FORMULAS = ["F1", "F2a", "F2b", "F2c", "F2d", "F3", "F4a", "F4b", "F4c", "F4d", "F5", "F6a", "F6b", "F7"]


def run_t1() -> pd.DataFrame:
    reg = build_registry()
    u, f, grad_u, dn_u = manufactured_2d()
    rows: list[dict] = []
    rng_svds = np.random.default_rng(12345)
    # track which configs get svds check (10% random)
    # we will decide per assembly after creation
    total_configs = len(T1_FORMULAS) * len(PREREG_EPS) * len(PREREG_N) * len(PREREG_K)
    print(f"METRIC t1_total_configs {total_configs}", flush=True)
    t_start = time.perf_counter()
    count = 0
    for fid in T1_FORMULAS:
        psi = reg[fid]["psi"]
        is_agg = (fid == "F7")
        for k in PREREG_K:
            for eps in PREREG_EPS:
                hs_for_slope: dict = {}
                # we compute per eps/k/formula across n for slope later
                errs_l2 = []
                errs_h1 = []
                errs_en = []
                hs = []
                for n in PREREG_N:
                    count += 1
                    mesh = unit_square_mesh(n)
                    h = 1.0 / n
                    circ = t1_circle(n, eps)
                    # assemble
                    t0 = time.perf_counter()
                    try:
                        if is_agg:
                            result = assemble_nitsche(mesh, circ, psi, k=k, f=f, g=u,
                                                      aggregator=default_aggregator, eps_c=1e-3)
                        else:
                            result = assemble_nitsche(mesh, circ, psi, k=k, f=f, g=u)
                    except Exception as exc:
                        print(f"METRIC assemble_failed fid {fid} n {n} eps {eps:.0e} k {k} err {exc}", flush=True)
                        rows.append({
                            "tier": "T1", "formula": fid, "k": k, "n": n, "h": h, "eps": eps,
                            "solver": "all", "lambda_min": np.nan, "lambda_max": np.nan,
                            "kappa": np.nan, "min_converged": False, "max_converged": False,
                            "asym_rel": np.nan, "l2_err": np.nan, "h1_err": np.nan, "energy_err": np.nan,
                            "cg_iters": -1, "solve_converged": False, "assemble_time": np.nan,
                            "solve_time": np.nan, "seed": 12345, "svds_checked": False,
                        })
                        continue
                    t_asm = time.perf_counter() - t0

                    # spectra (once per assembly)
                    sp_res = spectral_measures(result.A)
                    # mass-normalized companion (best effort)
                    mass_lmin = np.nan
                    try:
                        M_full = assemble_mass(mesh, circ, k)
                        alive = np.diff(result.A.indptr) > 0  # not correct shape; use active set
                        # instead restrict M to same active dofs as A
                        active = result.active_ids
                        if len(active) > 0 and M_full.shape[0] >= len(active):
                            # M_full is n_dof x n_dof, need to slice to active
                            M_red = M_full[active][:, active].tocsr()
                            # small regularization for solver
                            if M_red.shape[0] > 2:
                                # Use dense for tiny systems maybe; but try sparse eig
                                from measurement.spectra import mass_normalized_lambda_min
                                mass_lmin = mass_normalized_lambda_min(result.A.tocsr(), M_red)
                    except Exception as e:
                        mass_lmin = np.nan

                    # svds cross-check on 10% subsample
                    do_svds = rng_svds.random() < 0.10
                    svds_smin = np.nan
                    svds_smax = np.nan
                    svds_kappa = np.nan
                    if do_svds:
                        try:
                            smin, smax = svds_cross_check(result.A)
                            svds_smin, svds_smax = smin, smax
                            svds_kappa = smax / max(smin, 1e-300)
                        except Exception:
                            pass

                    # per-element lambdas for energy error
                    lam_by_elem = {int(row["element"]): float(row["lambda"]) for row in result.cell_table if row["status"] == "cut"}

                    # solve with trio
                    for solver_id in ("S1_lu", "S2_cg_jacobi", "S3_cg_amg"):
                        t1 = time.perf_counter()
                        sol = SOLVERS[solver_id](result.A, result.rhs)
                        t_solve = time.perf_counter() - t1
                        u_full = result.expand(sol.x) if sol.x is not None else np.zeros(result.n_dof_total)
                        # errors (use direct solution for H1/L2/energy if solver failed, fallback to S1)
                        # compute errors for this solver's solution
                        try:
                            l2 = l2_error(mesh, circ, u_full, result.dofs, k, u)
                        except Exception:
                            l2 = np.nan
                        try:
                            h1 = h1_semi_error(mesh, circ, u_full, result.dofs, k, grad_u)
                        except Exception:
                            h1 = np.nan
                        try:
                            en = energy_error(mesh, circ, u_full, result.dofs, k, grad_u, dn_u, u, lam_by_elem)
                        except Exception as e:
                            en = np.nan

                        # For slope, use S1_lu only (most accurate)
                        if solver_id == "S1_lu":
                            hs.append(h)
                            errs_l2.append(l2)
                            errs_h1.append(h1)
                            errs_en.append(en)

                        rows.append({
                            "tier": "T1",
                            "formula": fid,
                            "k": k,
                            "n": n,
                            "h": h,
                            "eps": eps,
                            "solver": solver_id,
                            "lambda_min": sp_res["lambda_min"],
                            "lambda_max": sp_res["lambda_max"],
                            "kappa": sp_res["kappa"],
                            "min_converged": bool(sp_res["min_converged"]),
                            "max_converged": bool(sp_res["max_converged"]),
                            "asym_rel": float(sp_res["asym_rel"]),
                            "mass_lambda_min": float(mass_lmin),
                            "svds_smin": float(svds_smin),
                            "svds_smax": float(svds_smax),
                            "svds_kappa": float(svds_kappa),
                            "svds_checked": bool(do_svds),
                            "l2_err": float(l2) if np.isfinite(l2) else np.nan,
                            "h1_err": float(h1) if np.isfinite(h1) else np.nan,
                            "energy_err": float(en) if np.isfinite(en) else np.nan,
                            "cg_iters": int(sol.iterations),
                            "solve_converged": bool(sol.converged),
                            "assemble_time": float(t_asm),
                            "solve_time": float(t_solve),
                            "seed": 12345,
                            "n_dof_active": int(result.A.shape[0]),
                            "n_dof_total": int(result.n_dof_total),
                        })
                    # periodic progress
                    if count % 50 == 0:
                        elapsed = time.perf_counter() - t_start
                        # find current kappa for F1 as diagnostic
                        print(f"METRIC progress {count}/{total_configs} fid {fid} n {n} eps {eps:.0e} k {k} kappa {sp_res['kappa']:.2e} lmin {sp_res['lambda_min']:.2e} elapsed {elapsed:.1f}s", flush=True)

                # after mesh loop, compute slopes per (fid,k,eps) using S1_lu errors
                if len(hs) >= 2:
                    for name, errs in (("l2", errs_l2), ("h1", errs_h1), ("energy", errs_en)):
                        p, r2 = fit_slope(np.array(hs), np.array(errs))
                        rows.append({
                            "tier": "T1-slope",
                            "formula": fid,
                            "k": k,
                            "n": np.nan,
                            "h": np.nan,
                            "eps": eps,
                            "solver": "S1_lu",
                            "quantity": name,
                            "slope_p": float(p),
                            "slope_R2": float(r2),
                            "seed": 12345,
                        })
                        print(f"METRIC t1_slope fid {fid} k {k} eps {eps:.0e} {name} p {p:.3f} R2 {r2:.4f}", flush=True)

    df = pd.DataFrame(rows)
    return df


def main() -> int:
    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)

    # First, quick benign sanity ( Logs separately but also included )
    print("METRIC stage benign_validation start", flush=True)
    df_benign = validate_benign()
    print("METRIC stage benign_validation done", flush=True)

    print("METRIC stage T1 start", flush=True)
    df_t1 = run_t1()
    print("METRIC stage T1 done", flush=True)

    # Combine
    df_all = pd.concat([df_benign, df_t1], ignore_index=True, sort=False)
    path = out_dir / "t1_results.parquet"
    df_all.to_parquet(path, index=False)
    # Also write csv for quick inspection
    csv_path = out_dir / "t1_results.csv"
    # limit csv to T1 rows without NaN explosion
    try:
        df_t1.to_csv(csv_path, index=False)
    except Exception:
        pass

    versions = {"numpy": np.__version__, "pandas": pd.__version__}
    try:
        import scipy
        versions["scipy"] = scipy.__version__
    except Exception:
        pass
    try:
        import pyamg
        versions["pyamg"] = pyamg.__version__
    except Exception:
        pass

    print("METRIC package_versions " + ";".join(f"{a}={b}" for a, b in sorted(versions.items())), flush=True)
    print(f"METRIC results_file {path}", flush=True)
    # Print summary for H1 gate quick check
    # Count coercivity violations per formula
    for fid in T1_FORMULAS:
        sub = df_t1[(df_t1["formula"] == fid) & (df_t1["tier"] == "T1") & (df_t1["solver"] == "S1_lu")]
        if len(sub) == 0:
            continue
        # check gamma <1e-8 at n=128
        worst = sub[sub["n"] == 128]
        if len(worst):
            min_gamma = worst["lambda_min"].min()
            viol = (worst["lambda_min"] < 1e-8).sum()
            print(f"METRIC h1_gate fid {fid} n128_min_gamma {min_gamma:.2e} violations_lt1e-8 {viol}/{len(worst)}", flush=True)
        # check slope gates
        slope_df = df_t1[(df_t1["formula"] == fid) & (df_t1["tier"] == "T1-slope")]
        # energy slopes
        if len(slope_df):
            bad = slope_df[(slope_df["quantity"] == "energy") & ((slope_df["slope_R2"] < 0.98) | (np.abs(slope_df["slope_p"] - slope_df["k"]) > 0.1 * slope_df["k"]))]
            print(f"METRIC slope_gate fid {fid} energy_bad {len(bad)}/{len(slope_df[slope_df['quantity']=='energy'])}", flush=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
